"""Task: Reasoning (synthetic multi-step arithmetic).

A mini-GSM8K-style task that probes multi-step reasoning — exactly what the
symbolic expert (compare/count/threshold/gate) should excel at and a pure MLP
struggles with.

Each example presents a small word problem of the form::

    "Start with <a>. Add <b>. Subtract <c>. The answer is ___"

encoded as a token sequence. The model must track intermediate state across
steps (multi-step) and emit the final numeric answer. The challenge is the
*chained* arithmetic: getting step 2 right depends on step 1's result, which
is not stored in a single linear attention span at small scale.

Token layout (length = 3·n_steps + 1)::

    tokens:  NUM a  NUM b  NUM c  ...  SEP  ANSWER
    where NUM = operator (ADD/SUB) + a value digit, ANSWER = the final value.

For the toy CPU validation, we use single-digit values and small vocab so the
task is learnable in ~200 steps but still requires multi-step bookkeeping.

Label convention (shared): ``targets[t]`` = what the model emits at ``t`` given
``0..t-1``; ``loss_mask[t]`` = 1 only on the ANSWER token(s).
"""

from __future__ import annotations

import random

import torch

PAD = 0
SEP = 1
ADD = 2      # operator token: "add the following value"
SUB = 3      # operator token: "subtract the following value"
NUM_OFFSET = 4   # value tokens live in [4, 4+value_range)


def _encode_value(v: int, value_range: int) -> list[int]:
    """Encode an integer as a single value token (toy: 1 token per value)."""
    # Clamp into [0, value_range-1]; offset by NUM_OFFSET.
    return [NUM_OFFSET + max(0, min(value_range - 1, v))]


def generate_example(
    n_steps: int,
    value_range: int,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Generate one reasoning example. Returns (token_sequence, answer)..

    The problem: start at some value, apply n_steps ADD/SUB operations, emit
    the final value. The model sees the whole sequence and must predict the
    final ANSWER token at the end.
    """
    value = rng.randint(0, value_range - 1)
    seq: list[int] = [NUM_OFFSET + value]   # initial value
    for _ in range(n_steps):
        op = ADD if rng.random() < 0.5 else SUB
        delta = rng.randint(1, max(1, value_range // 4))
        if op == ADD:
            value = min(value_range - 1, value + delta)
        else:
            value = max(0, value - delta)
        seq.append(op)
        seq.extend(_encode_value(delta, value_range))
    seq.append(SEP)
    answer_token = NUM_OFFSET + value
    return seq, answer_token


def generate_batch(
    batch_size: int,
    n_steps: int,
    value_range: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of reasoning examples, padded to equal length.

    Returns:
        input_ids: (B, T)
        targets:   (B, T) with PAD where ignored
        loss_mask: (B, T) float, 1.0 only at the answer position
    """
    # Seed a python RNG from the torch generator for reproducibility.
    seed = int(torch.randint(0, 2**31, (1,), generator=generator).item()) if generator else None
    rng = random.Random(seed)

    examples = [generate_example(n_steps, value_range, rng) for _ in range(batch_size)]
    max_len = max(len(seq) + 1 for seq, _ in examples)   # +1 for answer slot

    input_ids = torch.full((batch_size, max_len), PAD, dtype=torch.long)
    targets = torch.full((batch_size, max_len), PAD, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for b, (seq, answer) in enumerate(examples):
        L = len(seq)
        input_ids[b, :L] = torch.tensor(seq, dtype=torch.long)
        # The answer is emitted at the position right after SEP (index L).
        input_ids[b, L] = answer
        targets[b, L] = answer          # predict the answer given the sequence
        loss_mask[b, L] = 1.0

    input_ids = input_ids.to(device)
    targets = targets.to(device)
    loss_mask = loss_mask.to(device)
    return input_ids, targets, loss_mask


def description(n_steps: int, value_range: int) -> str:
    return (
        f"reasoning(steps={n_steps}, values={value_range}): "
        f"track {n_steps} chained ADD/SUB operations and emit the final value."
    )
