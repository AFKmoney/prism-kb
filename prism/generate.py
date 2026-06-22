"""Generation CLI for PRISM-KB (Module 5).

Three modes:
  * ``scratch``   — plain Prism generation (tape starts at zeros).
  * ``knowledge`` — retrieve top-k slots from a KB and seed the tape.
  * ``oneshot``   — encode (input, output) pairs from a JSON file and seed.

Usage::

    python -m prism.generate --checkpoint CKPT --mode scratch --prompt "..."
    python -m prism.generate --checkpoint CKPT --kb-path kb.json --mode knowledge \\
        --prompt "..." --top-k 8
    python -m prism.generate --checkpoint CKPT --mode oneshot \\
        --examples examples.json --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from torch import nn

from prism.config import PrismConfig
from prism.encoder import EncoderConfig, PrismEncoder
from prism.incontext import OneShotLearner
from prism.kb import KnowledgeBase
from prism.memory import MemoryState


def _load_model_and_encoder(args, device):
    from prism.train_scale import PRESETS

    config: PrismConfig = PRESETS[args.config_preset]()
    from prism.model import Prism

    model = Prism(config).to(device).eval()
    if args.checkpoint:
        import os

        ckpt = os.path.join(args.checkpoint, "pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)

    enc_cfg = EncoderConfig(
        d_model=config.d_model,
        d_mem=config.memory.d_mem,
        num_slots_per_doc=args.slots_per_doc,
        num_heads=min(4, config.d_model // 8 or 1),
    )
    embed = model.embed
    embed.requires_grad_(False)
    encoder = PrismEncoder(enc_cfg, embed=embed).to(device).eval()
    return model, encoder, config


@torch.no_grad()
def generate_scratch(model, tokenizer, prompt, device, max_new_tokens, temperature):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mem = None  # model will create a zero tape
    generated = []
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        out = model(input_ids, mem=mem)
        logits = out.logits[:, -1, :]
        mem = out.final_mem
        if temperature == 0.0:
            nid = logits.argmax(dim=-1, keepdim=True)
        else:
            nid = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
        input_ids = torch.cat([input_ids, nid], 1)
        generated.append(nid.item())
        if nid.item() == eos:
            break
    return tokenizer.decode(torch.tensor(generated).unsqueeze(0)[0], skip_special_tokens=True)


@torch.no_grad()
def generate_knowledge(model, encoder, kb, tokenizer, prompt, device, config,
                       top_k, blend_ratio, max_new_tokens, temperature):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    # Retrieve top-k slots for the prompt.
    q_slots = encoder.encode_text(prompt, tokenizer, device).squeeze(0)   # (k, d_mem)
    kb_slots = kb.retrieve(q_slots, top_k=top_k).to(device)
    mem = MemoryState.from_knowledge(
        kb_slots=kb_slots, batch_size=1, config=config.memory,
        device=device, dtype=next(model.parameters()).dtype, blend_ratio=blend_ratio,
    )
    generated = []
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        out = model(input_ids, mem=mem)
        logits = out.logits[:, -1, :]
        mem = out.final_mem
        if temperature == 0.0:
            nid = logits.argmax(dim=-1, keepdim=True)
        else:
            nid = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
        input_ids = torch.cat([input_ids, nid], 1)
        generated.append(nid.item())
        if nid.item() == eos:
            break
    return tokenizer.decode(torch.tensor(generated).unsqueeze(0)[0], skip_special_tokens=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PRISM-KB generation CLI")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config-preset", choices=["tiny", "300m", "350m", "1b"], default="tiny")
    p.add_argument("--mode", choices=["scratch", "knowledge", "oneshot"], default="scratch")
    p.add_argument("--kb-path", default=None, help="path to a KB .json (mode=knowledge)")
    p.add_argument("--examples", default=None, help="JSON file of [[input, output], ...] (mode=oneshot)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--slots-per-doc", type=int, default=2)
    p.add_argument("--blend-ratio", type=float, default=1.0)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args(argv)

    from transformers import AutoTokenizer

    device = torch.device(cli.device)
    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, encoder, config = _load_model_and_encoder(cli, device)

    if cli.mode == "scratch":
        out = generate_scratch(model, tokenizer, cli.prompt, device,
                               cli.max_new_tokens, cli.temperature)
    elif cli.mode == "knowledge":
        if not cli.kb_path:
            sys.exit("--mode knowledge requires --kb-path")
        kb = KnowledgeBase.load(cli.kb_path)
        print(f"[kb] {kb.summary()}", file=sys.stderr)
        out = generate_knowledge(model, encoder, kb, tokenizer, cli.prompt, device,
                                 config, cli.top_k, cli.blend_ratio,
                                 cli.max_new_tokens, cli.temperature)
    elif cli.mode == "oneshot":
        if not cli.examples:
            sys.exit("--mode oneshot requires --examples")
        pairs = json.loads(open(cli.examples, encoding="utf-8").read())
        learner = OneShotLearner(model, encoder, config)
        for inp, otp in pairs:
            learner.add_example(inp, otp)
        out = learner.generate(cli.prompt, tokenizer, cli.max_new_tokens,
                               cli.temperature, cli.blend_ratio)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
