"""Ingest datasets into a KnowledgeBase (PRISM-KB Module 5).

CLI::

    python -m prism.ingest --checkpoint CKPT --dataset PATH \\
        --text-column text --max-docs 10000 --out /path/to/kb

Encodes documents with a PrismEncoder (embedding loaded from the checkpoint,
or fresh init if no checkpoint) and saves a JSON KnowledgeBase.

Works with HuggingFace datasets (``--dataset HuggingFaceFW/fineweb-edu``) or a
local text file (``--dataset ./my.txt``, one doc per line, or ``--jsonl``).
"""

from __future__ import annotations

import argparse
import sys

import torch

from prism.config import PrismConfig
from prism.encoder import EncoderConfig, PrismEncoder
from prism.kb import KnowledgeBase
from prism.memory import MemoryConfig


def _load_encoder(args, config: PrismConfig, device) -> PrismEncoder:
    enc_cfg = EncoderConfig(
        d_model=config.d_model,
        d_mem=config.memory.d_mem,
        num_slots_per_doc=args.slots_per_doc,
        num_heads=min(4, config.d_model // 8 or 1),
    )
    embed = None
    if args.checkpoint:
        import os

        ckpt = os.path.join(args.checkpoint, "pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu")
        # Pull the embedding weights out of the checkpoint.
        emb_key = "embed.weight"
        if emb_key in state:
            import torch.nn as nn

            vocab, d = state[emb_key].shape
            embed = nn.Embedding(vocab, d)
            embed.weight.data.copy_(state[emb_key])
            embed.requires_grad_(False)
    return PrismEncoder(enc_cfg, embed=embed, vocab_size=config.vocab_size).to(device).eval()


def _iter_docs(args):
    if args.dataset.endswith(".jsonl") or args.jsonl:
        for line in open(args.dataset, encoding="utf-8"):
            import json

            yield json.loads(line).get(args.text_column, "")
    elif args.local:
        for line in open(args.dataset, encoding="utf-8"):
            line = line.strip()
            if line:
                yield line
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)
        count = 0
        for ex in ds:
            yield ex.get(args.text_column, "")
            count += 1
            if args.max_docs and count >= args.max_docs:
                break


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest a dataset into a PRISM-KB KnowledgeBase")
    p.add_argument("--checkpoint", default=None, help="Prism checkpoint dir (for frozen embedding)")
    p.add_argument("--config-preset", choices=["tiny", "300m", "350m", "1b"], default="tiny")
    p.add_argument("--dataset", required=True, help="HF dataset id or local file path")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--local", action="store_true", help="treat --dataset as a local text file")
    p.add_argument("--jsonl", action="store_true", help="local file is JSONL with text-column")
    p.add_argument("--max-docs", type=int, default=1000)
    p.add_argument("--slots-per-doc", type=int, default=2)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", required=True, help="output KB path (.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args(argv)

    from prism.train_scale import PRESETS

    config: PrismConfig = PRESETS[cli.config_preset]()
    device = torch.device(cli.device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = _load_encoder(cli, config, device)
    kb = KnowledgeBase(d_mem=config.memory.d_mem)

    count = 0
    batch: list[str] = []
    BATCH = 32
    for doc in _iter_docs(cli):
        if not doc:
            continue
        batch.append(doc[:1024])   # cap doc length
        if len(batch) >= BATCH:
            kb.add_texts(batch, encoder, tokenizer, device)
            count += len(batch)
            print(f"  ingested {count} docs ({kb.matrix.shape[0]} slots)", flush=True)
            batch = []
    if batch:
        kb.add_texts(batch, encoder, tokenizer, device)
        count += len(batch)

    kb.save(cli.out)
    print(f"\nSaved {kb.summary()} -> {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
