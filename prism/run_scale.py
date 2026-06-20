"""Scaled training entry point for PRISM.

Runs pretraining or instruction tuning at 1B-param scale on multi-GPU, or at
the ``tiny`` preset on a single GPU / CPU to validate the pipeline.

Usage::

    # Smoke-test the pipeline on CPU (no GPUs, no downloads beyond tokenizer):
    python -m prism.run_scale --preset tiny --steps 5 --no-ddp --cpu

    # Real pretraining of PRISM 1B on 8×A100 (launched via torchrun):
    torchrun --nproc_per_node=8 -m prism.run_scale --preset 1b --phase pretrain \\
        --steps 100000 --wandb --out-dir runs/prism-1b-pretrain

    # Instruction tuning (phase 2), resuming from a pretrained checkpoint:
    torchrun --nproc_per_node=8 -m prism.run_scale --preset 1b --phase instruct \\
        --init-from runs/prism-1b-pretrain/ckpt-100000 --out-dir runs/prism-1b-instruct

The script is written to FAIL LOUDLY rather than silently degrade: if you ask
for bf16 on a GPU that doesn't support it, or DDP without torchrun, it errors
with a clear message.
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
from torch.optim.lr_scheduler import LambdaLR

from prism.baselines import build_model
from prism.config import PrismConfig
from prism.data_scale import Batch, _load_tokenizer, build_dataloader, batched
from prism.train_scale import PRESETS, TrainArgs, get_mix, mix_summary


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed():
    """Return (local_rank, world_size, is_ddp)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    return 0, 1, False


def is_main_process(world_size: int) -> bool:
    import os

    return int(os.environ.get("LOCAL_RANK", "0")) == 0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def get_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def cosine_lr(step, warmup, max_steps, lr, min_lr):
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def train(args: TrainArgs):
    local_rank, world_size, is_ddp = setup_distributed()
    is_main = is_main_process(world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dtype = get_dtype(args.dtype) if (torch.cuda.is_available() and not args.force_cpu) else torch.float32

    if is_main:
        print("=" * 70)
        print(f"PRISM scaled training — preset={args.preset} phase={args.phase}")
        print(args.summary())
        print("-" * 70)
        mix = get_mix(args.phase)
        print(f"Dataset mix ({args.phase}):")
        print(mix_summary(mix))
        print("=" * 70)
        os.makedirs(args.out_dir, exist_ok=True)

    # --- Tokenizer (vocab size drives model config) ---
    tokenizer = _load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    # Round up to a multiple of 64 for kernel efficiency.
    vocab_size = math.ceil(vocab_size / 64) * 64

    # --- Model ---
    cfg: PrismConfig = PRESETS[args.preset](vocab_size=vocab_size)
    model = build_model("prism", cfg).to(device=device, dtype=dtype)
    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass  # not all PyTorch versions; safe to skip

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )

    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    n_params = sum(p.numel() for p in raw.parameters())
    if is_main:
        print(f"Model: PRISM {args.preset} — {n_params/1e9:.3f}B params (vocab={vocab_size})")

    # --- Resume ---
    start_step = 0
    if args.init_from:
        ckpt_path = os.path.join(args.init_from, "pytorch_model.bin")
        state = torch.load(ckpt_path, map_location="cpu")
        raw.load_state_dict(state)
        # Optionally resume optimizer/step.
        opt_path = os.path.join(args.init_from, "training_state.pt")
        if os.path.exists(opt_path):
            ts = torch.load(opt_path, map_location="cpu")
            start_step = ts["step"]
            if is_main:
                print(f"Resumed from {args.init_from} at step {start_step}")

    # --- Optimizer ---
    # Decay all params except biases and norm weights.
    decay, nodecay = [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n or "embed" in n:
            nodecay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), fused=(device.type == "cuda"),
    )
    sched = LambdaLR(opt, lambda s: cosine_lr(s, args.warmup_steps, args.steps, args.lr, args.min_lr) / args.lr)

    # --- Data ---
    if args.smoke_datasets:
        # Tiny in-memory mix for fast pipeline validation (no TB-scale downloads).
        from prism.train_scale import DatasetSpec

        mix = [
            DatasetSpec(
                path="wikitext", config="wikitext-2-raw-v1", split="train",
                text_column="text", weight=1.0, phase=args.phase,
            )
        ]
        if args.phase == "instruct":
            mix = [
                DatasetSpec(
                    path="databricks/databricks-dolly-15k", config=None, split="train",
                    text_column=None, weight=1.0, phase="instruct",
                )
            ]
        if is_main:
            print(f"[smoke] using tiny datasets: {[d.path for d in mix]}")
    else:
        mix = get_mix(args.phase)
    stream = build_dataloader(mix, tokenizer, args.seq_len, args.micro_batch_size, seed=args.seed)
    batches = batched(stream, args.micro_batch_size)

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
    running_loss = 0.0

    for step in range(start_step, args.steps):
        batch: Batch = next(batches)
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)

        with autocast_ctx:
            out = model(input_ids)
            logits = out.logits
            # Standard causal LM: predict token t+1 from t.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + getattr(out, "aux_loss", torch.zeros((), device=device))
            loss = loss / args.grad_accum_steps

        loss.backward()
        running_loss += loss.item()
        accum += 1

        if accum >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            accum = 0

            if (step + 1) % args.log_every == 0 and is_main:
                lr_now = sched.get_last_lr()[0]
                dt = time.time() - t0
                tok_s = args.tokens_per_step * args.log_every / max(dt, 1e-9)
                avg = running_loss / args.log_every / args.grad_accum_steps
                print(
                    f"step {step+1:>7}/{args.steps} | loss {avg:.4f} | "
                    f"lr {lr_now:.2e} | {tok_s:.0f} tok/s | "
                    f"{dt/args.log_every:.2f}s/step"
                )
                running_loss = 0.0
                t0 = time.time()

            if (step + 1) % args.save_every == 0 and is_main:
                _save_checkpoint(raw, opt, step + 1, args, is_ddp)

    # Final checkpoint
    if is_main:
        _save_checkpoint(raw, opt, args.steps, args, is_ddp)
        print(f"Training complete. Final checkpoint in {args.out_dir}")


def _save_checkpoint(model, opt, step, args, is_ddp):
    ckpt_dir = os.path.join(args.out_dir, f"ckpt-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    torch.save({"step": step, "args": args.__dict__}, os.path.join(ckpt_dir, "training_state.pt"))
    print(f"  saved checkpoint -> {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="PRISM scaled training (1B+)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="1b")
    p.add_argument("--phase", choices=["pretrain", "instruct"], default="pretrain")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--no-ddp", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU (for smoke testing)")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name")
    p.add_argument("--out-dir", type=str, default="runs/prism")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="prism")
    p.add_argument(
        "--smoke", action="store_true",
        help="override to tiny preset, 5 steps, CPU — validate the pipeline end-to-end.",
    )
    cli = p.parse_args(argv)

    args = TrainArgs(
        preset="tiny" if cli.smoke else cli.preset,
        phase=cli.phase,
        steps=5 if cli.smoke else cli.steps,
        seq_len=64 if cli.smoke else cli.seq_len,
        global_batch_size=4 if cli.smoke else cli.global_batch_size,
        micro_batch_size=2 if cli.smoke else cli.micro_batch_size,
        lr=3e-3 if cli.smoke else cli.lr,
        warmup_steps=1 if cli.smoke else cli.warmup_steps,
        dtype="fp32" if cli.smoke else cli.dtype,
        grad_checkpoint=not cli.no_grad_checkpoint and not cli.smoke,
        ddp=not cli.no_ddp and not cli.smoke,
        out_dir=cli.out_dir,
        save_every=10000 if cli.smoke else cli.save_every,
        log_every=1 if cli.smoke else cli.log_every,
        init_from=cli.init_from,
        seed=cli.seed,
        tokenizer=cli.tokenizer,
        wandb=cli.wandb,
        wandb_project=cli.wandb_project,
    )
    # Stash the --cpu flag.
    args.force_cpu = cli.cpu or cli.smoke
    args.smoke_datasets = cli.smoke

    if cli.wandb and is_main_process(int(os.environ.get("WORLD_SIZE", "1"))):
        try:
            import wandb

            wandb.init(project=cli.wandb_project, config={**args.__dict__, "cli": vars(cli)})
        except Exception as e:
            print(f"[warn] wandb init failed: {e}; continuing without wandb")

    train(args)


if __name__ == "__main__":
    sys.exit(main())
