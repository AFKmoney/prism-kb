"""Streaming multi-dataset dataloader for scaled PRISM training.

Loads multiple HuggingFace datasets, samples from them according to configured
weights, tokenizes on the fly (pretrain) or formats into prompt/completion
(instruct), and yields fixed-length token batches ready for the model.

Design choices:
  * **Streaming** — the datasets are huge (TBs). We use ``streaming=True`` to
    avoid materializing them. Shuffling is done via a small buffer.
  * **Weighted sampling** — a per-step dataset index is drawn from a
    multinomial over the weights, so the empirical mix matches the spec.
  * **Document packing** — for pretrain, we pack tokenized docs back-to-back
    into seq_len windows with a separator, maximizing token density (no padding).
  * **Loss masking for instruct** — the prompt tokens are masked out of the
    loss; only the completion is supervised.

This module is import-safe even without datasets/transformers installed — the
heavy imports happen inside ``build_dataloader`` so the rest of the package and
the test suite keep working.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterator

import torch

from prism.train_scale import DatasetSpec


@dataclass
class Batch:
    """A training batch.

    Attributes:
        input_ids: (B, T) long.
        labels: (B, T) long, with -100 where loss is ignored.
        loss_mask: (B, T) float, 1.0 where loss counts.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor


def _load_tokenizer(name: str = "gpt2"):
    """Load a tokenizer. Default: GPT-2 BPE (vocab 50257, no auth required).

    The tokenizer is the one piece of "prior art" PRISM reuses — designing a
    new tokenizer from scratch is orthogonal to the architecture. GPT-2's BPE
    is universally available and well-tested. Override with any HF tokenizer
    via the CLI's ``--tokenizer`` flag.

    The model's vocab_size should be set to at least ``len(tokenizer)``, rounded
    up to a multiple of 64 for kernel efficiency (the trainer does this).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _iter_pretrain(spec: DatasetSpec, tokenizer):
    """Stream a pretrain dataset, yielding token-id lists per document."""
    from datasets import load_dataset

    ds = load_dataset(spec.path, spec.config, split=spec.split, streaming=True)
    col = spec.text_column
    count = 0
    for ex in ds:
        text = ex.get(col)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        yield ids
        count += 1
        if spec.max_samples and count >= spec.max_samples:
            break


def _format_instruct(example, tokenizer, spec: DatasetSpec) -> list[int]:
    """Format an instruction example into a single token-id list with mask.

    Returns (token_ids, mask) where mask is 1 on completion tokens, 0 on prompt.
    Handles the common column schemas (OpenOrca / Open-Platypus / openchat /
    GPT4-LLM-Cleaned).
    """
    # Detect schema.
    if "messages" in example:
        # chat-style (openchat). Use tokenizer chat template if available.
        prompt = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    else:
        # prompt/response schema (OpenOrca uses 'question'/'response';
        # Open-Platypus / GPT4-LLM use 'instruction'/'output' or 'prompt'/'output').
        q = example.get("question") or example.get("instruction") or example.get("prompt") or ""
        a = example.get("response") or example.get("output") or example.get("completion") or ""
        # A simple, robust template.
        prompt = f"### Instruction:\n{q}\n\n### Response:\n"
        full = f"{prompt}{a}{tokenizer.eos_token}"

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    # completion = full minus prompt prefix
    comp_ids = full_ids[len(prompt_ids):]
    token_ids = prompt_ids + comp_ids
    mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
    return token_ids, mask


def _iter_instruct(spec: DatasetSpec, tokenizer):
    """Stream an instruct dataset, yielding (token_ids, mask) tuples."""
    from datasets import load_dataset

    ds = load_dataset(spec.path, spec.config, split=spec.split, streaming=True)
    count = 0
    for ex in ds:
        try:
            token_ids, mask = _format_instruct(ex, tokenizer, spec)
        except Exception:
            continue
        if len(token_ids) < 2:
            continue
        yield token_ids, mask
        count += 1
        if spec.max_samples and count >= spec.max_samples:
            break


class PackedStream:
    """Pack variable-length documents into fixed-length seq_len windows.

    For pretrain: documents separated by a separator token; loss on all tokens.
    For instruct: the per-token mask is carried through packing.
    """

    def __init__(self, source: Iterator, seq_len: int, phase: str, separator_id: int = 0):
        self.source = source
        self.seq_len = seq_len
        self.phase = phase
        self.sep = separator_id

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        buf_ids: list[int] = []
        buf_mask: list[int] = []
        for item in self.source:
            if self.phase == "pretrain":
                ids = item
                buf_ids.extend(ids)
                buf_mask.extend([1] * len(ids))
            else:
                ids, mask = item
                buf_ids.extend(ids)
                buf_mask.extend(mask)
            # Flush full windows.
            while len(buf_ids) >= self.seq_len:
                w_ids = buf_ids[: self.seq_len]
                w_mask = buf_mask[: self.seq_len]
                yield w_ids, w_mask
                buf_ids = buf_ids[self.seq_len:]
                buf_mask = buf_mask[self.seq_len:]
        # Flush remainder (padded).
        if buf_ids:
            pad = self.seq_len - len(buf_ids)
            w_ids = buf_ids + [self.sep] * pad
            w_mask = buf_mask + [0] * pad
            yield w_ids, w_mask


def build_dataloader(
    mix: list[DatasetSpec],
    tokenizer,
    seq_len: int,
    batch_size: int,
    seed: int = 42,
    shuffle_buffer: int = 10_000,
) -> Iterator[Batch]:
    """Build an infinite weighted-mix streaming dataloader.

    Args:
        mix: list of DatasetSpecs (all must share the same phase).
        tokenizer: a HF tokenizer.
        seq_len: packed sequence length.
        batch_size: number of seq_len windows per yielded Batch.
        seed: RNG seed for the multinomial sampling.
        shuffle_buffer: number of packed windows to buffer for shuffling.

    Yields:
        Batch objects forever.
    """
    phase = mix[0].phase
    for spec in mix:
        if spec.phase != phase:
            raise ValueError(f"mix phase mismatch: {spec.path} is {spec.phase}, expected {phase}")

    # Build one packed stream per dataset.
    streams = []
    weights = []
    for spec in mix:
        if phase == "pretrain":
            src = _iter_pretrain(spec, tokenizer)
        else:
            src = _iter_instruct(spec, tokenizer)
        streams.append(iter(PackedStream(src, seq_len, phase)))
        weights.append(spec.weight)

    total = sum(weights)
    probs = [w / total for w in weights]
    rng = random.Random(seed)

    # Interleave streams by weighted sampling. We keep a small reservoir per
    # stream so a slow/blocked stream doesn't stall the others.
    reservoirs = [list(itertools.islice(s, 8)) for s in streams]
    # Ensure each reservoir is primed (refill as we consume).
    def refill(i):
        while len(reservoirs[i]) < 8:
            try:
                reservoirs[i].append(next(streams[i]))
            except StopIteration:
                # Dataset exhausted (rare in streaming mode); restart it.
                if phase == "pretrain":
                    streams[i] = iter(PackedStream(_iter_pretrain(mix[i], tokenizer), seq_len, phase))
                else:
                    streams[i] = iter(PackedStream(_iter_instruct(mix[i], tokenizer), seq_len, phase))
                reservoirs[i].append(next(streams[i]))

    # Shuffle buffer for cross-document mixing.
    buffer: list[tuple[list[int], list[int]]] = []

    while True:
        # Draw a dataset according to the mix weights and pull a window.
        i = rng.choices(range(len(streams)), weights=probs, k=1)[0]
        refill(i)
        window_ids, window_mask = reservoirs[i].pop(0)

        buffer.append((window_ids, window_mask))
        if len(buffer) >= shuffle_buffer:
            j = rng.randint(0, len(buffer) - 1)
            window_ids, window_mask = buffer.pop(j)

            # Accumulate into a batch lazily.
            # (We use a simple stateful closure to batch.)
            yield _make_batch_singleton(window_ids, window_mask)

    # Note: the function never returns; it is an infinite generator.


# Helper kept simple: real batching is done by the caller via a wrapper.
_state = {"batch": []}


def _make_batch_singleton(ids, mask):
    """Emit one window at a time; batching happens in batched()."""
    return (ids, mask)


def batched(source: Iterator[tuple[list[int], list[int]]], batch_size: int) -> Iterator[Batch]:
    """Group singletons from build_dataloader into Batch objects."""
    acc_ids, acc_mask = [], []
    for ids, mask in source:
        acc_ids.append(ids)
        acc_mask.append(mask)
        if len(acc_ids) == batch_size:
            yield _to_batch(acc_ids, acc_mask)
            acc_ids, acc_mask = [], []
    if acc_ids:
        yield _to_batch(acc_ids, acc_mask)


def _to_batch(ids_list, mask_list) -> Batch:
    import numpy as np

    input_ids = torch.tensor(np.array(ids_list), dtype=torch.long)
    loss_mask = torch.tensor(np.array(mask_list), dtype=torch.float32)
    # labels: -100 where masked, else the input id (shifted by the trainer).
    labels = input_ids.clone()
    labels[loss_mask == 0] = -100
    return Batch(input_ids=input_ids, labels=labels, loss_mask=loss_mask)
