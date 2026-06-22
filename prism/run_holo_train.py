"""PRISM-Holo training script.

Trains the holographic encoder + backbone together, with a combined loss:
  * LM loss on the full sequence (trains backbone + neural expert)
  * Retrieval-consistency loss (trains the split key/value encoders)
  * Contrastive InfoNCE loss (trains the encoder to discriminate)

The consistency + contrastive losses are what close the +0.0525 random-init
specificity gap toward the +0.355 VSA ceiling.

Usage::

    # Smoke test on CPU (validates the full pipeline, no GPU):
    python -m prism.run_holo_train --smoke

    # Real training on 8xA100 (the cluster path):
    torchrun --nproc_per_node=8 -m prism.run_holo_train \
        --preset 1b --steps 50000 --seq-len 2048 \
        --consistency-weight 0.3 --contrastive-weight 0.2 \
        --out-dir runs/prism-holo-1b --wandb

The script reuses run_scale.py's infrastructure (DDP, bf16, checkpointing,
gradient accumulation) and adds the Holo-specific losses on top.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from prism.config import MemoryConfig, PrismConfig
from prism.holo_data import get_holo_mix, holo_mix_summary, format_retrieval_example
from prism.holo_loss import (
    holo_consistency_loss,
    holo_contrastive_loss,
    pooled_embedding,
)
from prism.model import Prism
from prism.pcs import DEFAULT_SCHEDULE, grow_model, resolve_schedule, schedule_summary
from prism.train_scale import PRESETS


def build_holo_config(preset: str, vocab_size: int) -> PrismConfig:
    """Build a config with holo_mode=True and VSA-appropriate memory dims."""
    base = PRESETS[preset](vocab_size=vocab_size)
    # Override memory dims for a good VSA dimensionality (D = num_slots * d_mem >= 4096).
    # For 1b: 256 * 32 = 8192. For tiny: 64 * 32 = 2048.
    if preset == "tiny":
        base.memory = MemoryConfig(d_mem=32, num_slots=64)
    else:
        base.memory = MemoryConfig(d_mem=32, num_slots=256)
    base.holo_mode = True
    return base


def get_holo_encoders(model: Prism) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Extract the (key_encoder, value_encoder) from the first HoloHead in the model."""
    for block in model.blocks:
        for expert in block.router.experts:
            head = getattr(expert, "head", None)
            if head is not None and head.__class__.__name__ == "HoloHead":
                return head.key_encoder, head.value_encoder
    raise RuntimeError("No HoloHead found in model — is holo_mode=True set?")


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    return 0, 1, False


def cosine_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device, generator=None):
    """Synthetic (question, answer) batch for smoke testing without HF downloads.

    Each example: [BOS q_tokens SEP a_tokens PAD...].
    Returns input_ids, question_mask (1 on q tokens), answer_mask (1 on a tokens).
    """
    q_len = seq_len // 3
    a_len = seq_len // 3
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    q_mask = torch.zeros(batch_size, seq_len, device=device)
    a_mask = torch.zeros(batch_size, seq_len, device=device)
    for b in range(batch_size):
        ids[b, 0] = 1  # BOS
        ids[b, 1:1+q_len] = torch.randint(2, vocab_size, (q_len,), device=device, generator=generator)
        ids[b, 1+q_len] = 2  # SEP
        ids[b, 2+q_len:2+q_len+a_len] = torch.randint(2, vocab_size, (a_len,), device=device, generator=generator)
        ids[b, 2+q_len+a_len] = 0  # EOS/pad
        q_mask[b, 1:1+q_len] = 1.0
        a_mask[b, 2+q_len:2+q_len+a_len] = 1.0
    return ids, q_mask, a_mask


def train(args):
    local_rank, world_size, is_ddp = setup_distributed()
    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dtype = torch.float32 if args.force_cpu else (
        {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    )

    # --- Tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = math.ceil(len(tokenizer) / 64) * 64

    # --- Model (holo_mode=True) ---
    cfg = build_holo_config(args.preset, vocab_size)
    model = Prism(cfg).to(device=device, dtype=dtype)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    key_enc, val_enc = get_holo_encoders(raw)
    n_params = sum(p.numel() for p in raw.parameters())
    if is_main:
        print("=" * 70)
        print(f"PRISM-Holo training — preset={args.preset} holo_mode=True")
        print(f"  {n_params/1e6:.1f}M params, D={cfg.memory.num_slots * cfg.memory.d_mem}")
        print(f"  consistency_weight={args.consistency_weight}, contrastive_weight={args.contrastive_weight}")
        print(f"  steps={args.steps}, lr={args.lr}, dtype={args.dtype}")
        print("-" * 70)
        if not args.smoke:
            print("Dataset mix:")
            print(holo_mix_summary())
        else:
            print("[smoke] using synthetic batches (no HF downloads)")
        print("=" * 70)

    # --- Optimizer (separate LR groups: encoders get higher LR) ---
    enc_params = list(key_enc.parameters()) + list(val_enc.parameters())
    enc_param_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_param_ids and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": enc_params, "lr": args.lr * args.encoder_lr_mult},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr(s, args.warmup_steps, args.steps, 1.0, 0.1)  # relative
    )

    # --- Data ---
    if args.smoke:
        g = torch.Generator(device=device).manual_seed(42)
        def batch_iter():
            while True:
                yield make_synthetic_batch(args.micro_batch_size, args.seq_len, vocab_size, device, g)
    else:
        from prism.data_scale import _load_tokenizer, build_dataloader, batched
        mix = get_holo_mix()
        stream = build_dataloader(mix, tokenizer, args.seq_len, args.micro_batch_size, seed=args.seed)
        batches = batched(stream, args.micro_batch_size)
        def batch_iter():
            while True:
                b = next(batches)
                # Build question/answer masks from the retrieval formatting.
                ids = b.input_ids
                # Heuristic: split at SEP token (id=2) if present, else first half.
                sep_pos = (ids == 2).nonzero(as_tuple=True)
                q_mask = torch.zeros_like(ids, dtype=torch.float)
                a_mask = torch.zeros_like(ids, dtype=torch.float)
                for i in range(ids.shape[0]):
                    row_sep = (ids[i] == 2).nonzero(as_tuple=True)[0]
                    if len(row_sep) > 0:
                        sp = row_sep[0].item()
                        q_mask[i, :sp] = 1.0
                        a_mask[i, sp+1:] = 1.0
                    else:
                        half = ids.shape[1] // 2
                        q_mask[i, :half] = 1.0
                        a_mask[i, half:] = 1.0
                yield ids, q_mask, a_mask

    # --- Train loop ---
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    t0 = time.time()
    model.train()
    accum = 0
    opt.zero_grad(set_to_none=True)
    running = {"lm": 0.0, "cons": 0.0, "contr": 0.0}

    for step in range(args.steps):
        input_ids, q_mask, a_mask = next(batch_iter())
        input_ids = input_ids.to(device)
        q_mask = q_mask.to(device)
        a_mask = a_mask.to(device)

        with autocast_ctx:
            out = model(input_ids)
            logits = out.logits

            # LM loss (shifted causal LM; mask padding by ignoring id 0).
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            )

            # Holo retrieval losses: pool question and answer embeddings.
            q_emb = pooled_embedding(raw, input_ids * q_mask.long(), q_mask)   # (B, d_model)
            a_emb = pooled_embedding(raw, input_ids * a_mask.long(), a_mask)   # (B, d_model)
            cons_loss = holo_consistency_loss(key_enc, val_enc, q_emb, a_emb)
            contr_loss = holo_contrastive_loss(key_enc, val_enc, q_emb, a_emb)

            loss = (
                lm_loss
                + args.consistency_weight * cons_loss
                + args.contrastive_weight * contr_loss
            )
            loss = loss / args.grad_accum_steps

        loss.backward()
        running["lm"] += lm_loss.item()
        running["cons"] += cons_loss.item()
        running["contr"] += contr_loss.item()
        accum += 1

        if accum >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            accum = 0

            if (step + 1) % args.log_every == 0 and is_main:
                dt = time.time() - t0
                n = args.log_every
                print(
                    f"step {step+1:>6}/{args.steps} | "
                    f"lm {running['lm']/n:.4f} | "
                    f"cons {running['cons']/n:.4f} | "
                    f"contr {running['contr']/n:.4f} | "
                    f"lr {sched.get_last_lr()[0]:.2e} | "
                    f"{dt/n:.2f}s/step"
                )
                running = {k: 0.0 for k in running}
                t0 = time.time()

            if (step + 1) % args.save_every == 0 and is_main:
                ckpt_dir = os.path.join(args.out_dir, f"ckpt-{step+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(raw.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
                print(f"  saved -> {ckpt_dir}")

    if is_main:
        ckpt_dir = os.path.join(args.out_dir, f"ckpt-{args.steps}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(raw.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        print(f"\nTraining complete. Final checkpoint -> {ckpt_dir}")


def train_progressive(args):
    """Progressive Capacity Stacking: train 350M -> 700M -> 1B, growing weights.

    Each stage runs the standard train() loop at its preset, then grow_model()
    transfers weights to the next (larger) stage. The bulk of tokens train at
    the small stages (cheap per-step FLOPs), cutting total wall-clock ~40-50%.
    """
    local_rank, world_size, is_ddp = setup_distributed()
    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = math.ceil(len(tokenizer) / 64) * 64

    schedule = resolve_schedule(args.steps, DEFAULT_SCHEDULE)
    if is_main:
        print("=" * 70)
        print("PRISM-Holo PROGRESSIVE training (PCS)")
        print(f"  Total steps: {args.steps} across {len(schedule)} stages")
        print(schedule_summary(schedule))
        print(f"  holo_mode=True, consistency_weight={args.consistency_weight}")
        print("=" * 70)

    model = None
    for stage_idx, stage in enumerate(schedule):
        stage_cfg = build_holo_config(stage.preset, vocab_size)
        if model is None:
            # First stage: fresh model.
            dtype = torch.float32 if args.force_cpu else (
                {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
            )
            model = Prism(stage_cfg).to(device=device, dtype=dtype)
            if is_main:
                n = sum(p.numel() for p in model.parameters())
                print(f"\n[Stage {stage_idx+1}/{len(schedule)}] {stage.preset} "
                      f"({n/1e6:.1f}M) — fresh init, {stage.steps} steps")
        else:
            # Subsequent stages: grow the model from the previous stage.
            if is_main:
                print(f"\n[Stage {stage_idx+1}/{len(schedule)}] {stage.preset} "
                      f"— growing from previous stage, {stage.steps} steps")
            model = grow_model(model, stage_cfg).to(device=device, dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)
            if is_ddp:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank] if device.type == "cuda" else None
                )

        # Re-wrap in DDP if needed (fresh model case).
        if is_ddp and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == "cuda" else None
            )

        # Run this stage's training.
        stage_args = _stage_args(args, stage, stage_idx, len(schedule))
        _run_one_stage(model, stage_args, tokenizer, vocab_size, device, is_main)

        # Unwrap for growth.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        if is_main:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt-stage{stage_idx+1}-{stage.preset}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
            print(f"  saved stage {stage_idx+1} -> {ckpt_dir}")

    if is_main:
        final = os.path.join(args.out_dir, "ckpt-final")
        os.makedirs(final, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final, "pytorch_model.bin"))
        print(f"\nProgressive training complete. Final -> {final}")


def _stage_args(args, stage, stage_idx, n_stages):
    """Make a copy of args scoped to one stage."""
    import copy
    sa = copy.copy(args)
    sa.steps = stage.steps
    sa.preset = stage.preset
    sa.out_dir = args.out_dir
    # Warmup is shorter for later stages (the model is already warmed up).
    sa.warmup_steps = max(1, args.warmup_steps // (2 ** stage_idx))
    return sa


def _run_one_stage(model, args, tokenizer, vocab_size, device, is_main):
    """Run one stage of training (reuses the train() inner loop logic)."""
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    key_enc, val_enc = get_holo_encoders(raw)
    enc_params = list(key_enc.parameters()) + list(val_enc.parameters())
    enc_param_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_param_ids and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": enc_params, "lr": args.lr * args.encoder_lr_mult},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95),
       fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr(s, args.warmup_steps, args.steps, 1.0, 0.1)
    )

    dtype = next(model.parameters()).dtype
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )

    if args.smoke:
        g = torch.Generator(device=device).manual_seed(42 + hash(args.preset) % 1000)
        def batch_iter():
            while True:
                yield make_synthetic_batch(args.micro_batch_size, args.seq_len, vocab_size, device, g)
    else:
        from prism.data_scale import build_dataloader, batched
        mix = get_holo_mix()
        stream = build_dataloader(mix, tokenizer, args.seq_len, args.micro_batch_size, seed=args.seed)
        batches = batched(stream, args.micro_batch_size)
        def batch_iter():
            while True:
                b = next(batches)
                ids = b.input_ids
                q_mask = torch.zeros_like(ids, dtype=torch.float)
                a_mask = torch.zeros_like(ids, dtype=torch.float)
                for i in range(ids.shape[0]):
                    row_sep = (ids[i] == 2).nonzero(as_tuple=True)[0]
                    if len(row_sep) > 0:
                        sp = row_sep[0].item()
                        q_mask[i, :sp] = 1.0
                        a_mask[i, sp+1:] = 1.0
                    else:
                        half = ids.shape[1] // 2
                        q_mask[i, :half] = 1.0
                        a_mask[i, half:] = 1.0
                yield ids, q_mask, a_mask

    t0 = time.time()
    model.train()
    accum = 0
    opt.zero_grad(set_to_none=True)
    running = {"lm": 0.0, "cons": 0.0, "contr": 0.0}

    for step in range(args.steps):
        input_ids, q_mask, a_mask = next(batch_iter())
        input_ids = input_ids.to(device)
        q_mask = q_mask.to(device)
        a_mask = a_mask.to(device)

        with autocast_ctx:
            out = model(input_ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=0,
            )
            q_emb = pooled_embedding(raw, input_ids * q_mask.long(), q_mask)
            a_emb = pooled_embedding(raw, input_ids * a_mask.long(), a_mask)
            cons_loss = holo_consistency_loss(key_enc, val_enc, q_emb, a_emb)
            contr_loss = holo_contrastive_loss(key_enc, val_enc, q_emb, a_emb)
            loss = lm_loss + args.consistency_weight * cons_loss + args.contrastive_weight * contr_loss
            loss = loss / args.grad_accum_steps

        loss.backward()
        running["lm"] += lm_loss.item()
        running["cons"] += cons_loss.item()
        running["contr"] += contr_loss.item()
        accum += 1

        if accum >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            if (step + 1) % args.log_every == 0 and is_main:
                dt = time.time() - t0
                n = args.log_every
                print(f"  [{args.preset}] step {step+1:>5}/{args.steps} | "
                      f"lm {running['lm']/n:.4f} | cons {running['cons']/n:.4f} | "
                      f"contr {running['contr']/n:.4f} | {dt/n:.2f}s/step")
                running = {k: 0.0 for k in running}
                t0 = time.time()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PRISM-Holo training (holo_mode + retrieval losses)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="1b")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--consistency-weight", type=float, default=0.3,
                   help="weight of the bind/unbind consistency loss")
    p.add_argument("--contrastive-weight", type=float, default=0.2,
                   help="weight of the InfoNCE contrastive loss in VSA space")
    p.add_argument("--encoder-lr-mult", type=float, default=3.0,
                   help="LR multiplier for the Holo encoders (they need to learn fast)")
    p.add_argument("--no-ddp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out-dir", default="runs/prism-holo")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="tiny preset, 10 steps, CPU, synthetic batches — pipeline validation")
    p.add_argument("--progressive", action="store_true",
                   help="Progressive Capacity Stacking: 350M -> 700M -> 1B with weight inheritance. "
                        "~40-50% wall-clock reduction vs from-scratch 1B.")
    p.add_argument("--wandb", action="store_true")
    cli = p.parse_args(argv)

    if cli.smoke:
        cli.preset = "tiny"
        cli.steps = 10
        cli.seq_len = 64
        cli.global_batch_size = 4
        cli.micro_batch_size = 2
        cli.warmup_steps = 1
        cli.dtype = "fp32"
        cli.no_ddp = True
        cli.log_every = 2
        cli.save_every = 100
        # If --progressive is also set, use a tiny-only schedule (2 stages,
        # tiny->tiny) to validate the grow + multi-stage plumbing on CPU.
        if cli.progressive:
            import prism.pcs as _pcs
            _pcs.DEFAULT_SCHEDULE[:] = [
                _pcs.StageSpec(preset="tiny", token_fraction=0.6),
                _pcs.StageSpec(preset="tiny", token_fraction=0.4),
            ]
            cli.steps = 8   # 5 + 3

    grad_accum = max(1, cli.global_batch_size // (cli.micro_batch_size * max(1, int(os.environ.get("WORLD_SIZE", "1")))))

    class Args:
        pass
    args = Args()
    for k, v in vars(cli).items():
        setattr(args, k, v)
    args.force_cpu = cli.cpu or cli.smoke
    args.grad_accum_steps = grad_accum

    if cli.wandb and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        try:
            import wandb
            wandb.init(project="prism-holo", config=vars(cli))
        except Exception as e:
            print(f"[warn] wandb init failed: {e}")

    if cli.progressive:
        train_progressive(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
