# Scaling PRISM to 1B — Training Guide

This is the path to make PRISM competitive with standard 1B LLMs. The toy model
in the repo proves the architecture learns; the scaled trainer turns it into a
real model.

> **Reality check:** training PRISM 1B to competence needs ~8×A100 (or
> equivalent) for ~1–2 weeks and ~500B+ tokens. The script is correct and
> ready; it just needs compute you point it at. You can validate the *pipeline*
> on a laptop in <1 minute with `--smoke` (no GPUs, no TB-scale downloads).

---

## Quick start

### 1. Validate the pipeline (laptop, <1 min, no GPU)

```bash
# Validates: tokenizer load → dataset streaming → packing → forward/backward
# → loss → checkpoint save. Uses tiny wikitext/dolly instead of TB-scale data.
python -m prism.run_scale --smoke --phase pretrain
python -m prism.run_scale --smoke --phase instruct
```

### 2. Real pretraining of PRISM 1B (8×A100)

```bash
torchrun --nproc_per_node=8 -m prism.run_scale \
    --preset 1b \
    --phase pretrain \
    --steps 100000 \
    --seq-len 4096 \
    --global-batch-size 256 \
    --micro-batch-size 8 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --dtype bf16 \
    --out-dir runs/prism-1b-pretrain \
    --wandb
```

### 3. Instruction tuning (phase 2, resume from pretrain)

```bash
torchrun --nproc_per_node=8 -m prism.run_scale \
    --preset 1b \
    --phase instruct \
    --init-from runs/prism-1b-pretrain/ckpt-100000 \
    --steps 10000 \
    --seq-len 2048 \
    --global-batch-size 128 \
    --lr 1e-4 \
    --warmup-steps 100 \
    --out-dir runs/prism-1b-instruct \
    --wandb
```

---

## Configuration

### Model presets (`--preset`)

| Preset | Params | d_model | Layers | Rates | Target hardware |
|---|---|---|---|---|---|
| `tiny` | ~7M | 128 | 4 | 4 | CPU (pipeline test only) |
| `350m` | ~350M | 1024 | 18 | 6 | 1×A100 80GB |
| `1b` | ~1.0B | 2048 | 24 | 8 | 8×A100 80GB |

Presets live in [`prism/train_scale.py`](prism/train_scale.py). Edit
`prism_1b()` to tune depth/width/rates for your hardware.

### Dataset mixes

**Pretrain** (2025 consensus recipe — sources in README):

| Dataset | Weight | Purpose |
|---|---:|---|
| FineWeb-Edu (sample-10BT) | 70% | high-quality educational text backbone |
| OpenWebMath | 10% | mathematical reasoning |
| The Stack v2 | 15% | code |
| RedPajama-1T-Sample | 5% | books / arxiv / wikipedia |

**Instruct**:

| Dataset | Weight | Purpose |
|---|---:|---|
| OpenOrca | 50% | broad instruction following |
| Open-Platypus | 20% | reasoning (GPT-4 quality) |
| openchat_3.5 | 15% | conversational |
| GPT4-LLM-Cleaned | 15% | instruction following |

Mixes are in `prism/train_scale.py` (`MIX_PRETRAIN`, `MIX_INSTRUCT`). To swap a
dataset, edit the `DatasetSpec`. To change the mix ratio, edit the `weight`.

### Key hyperparameters (1B pretrain defaults)

| Param | Value | Rationale |
|---|---|---|
| `--lr` | 3e-4 | standard peak LR for 1B-scale AdamW |
| `--min-lr` | 3e-5 | 10× decay (cosine) |
| `--warmup-steps` | 2000 | stabilize early routing |
| `--weight-decay` | 0.1 | standard |
| `--grad-clip` | 1.0 | standard |
| AdamW betas | (0.9, 0.95) | 0.95 β2 is the LLM pretraining standard |
| `--dtype` | bf16 | A100+ native; stable for this scale |

---

## Distributed training

The script uses `torchrun` + `DistributedDataParallel` (DDP). Launch with:

```bash
torchrun --nproc_per_node=N -m prism.run_scale ...
```

For multi-node, add `--rdzv_backend=c10d --rdzv_endpoint=HOST:PORT`. The script
auto-detects `RANK`/`WORLD_SIZE` from `torchrun`'s env vars.

**Gradient accumulation** is computed automatically:
```
grad_accum = global_batch_size / (micro_batch_size × world_size)
```
so the effective batch is constant regardless of GPU count.

**Gradient checkpointing** is on by default for 1B (`--grad_checkpoint`). The
Multi-Rate Bus's linear activation memory means PRISM uses less activation
memory than an equivalent Transformer, allowing longer sequences at the same VRAM.

---

## Tokenizer

Default: **GPT-2 BPE** (vocab 50257, no auth required). Override with any HF
tokenizer:

```bash
--tokenizer tiiuae/falcon-7b          # larger vocab, multilingual
--tokenizer HuggingFaceTB/SmolLM2-360M # modern, efficient
```

The model's `vocab_size` is set to `ceil(len(tokenizer)/64)*64` automatically.
The tokenizer is the one piece of prior art PRISM reuses — a custom tokenizer
is orthogonal to the architecture.

---

## Resuming

Checkpoints save to `--out-dir/ckpt-{step}/` every `--save-every` steps
(default 2000). To resume:

```bash
--init-from runs/prism-1b-pretrain/ckpt-40000
```

The script loads model weights, optimizer state, and the step counter.

---

## Monitoring

```bash
--wandb                    # log loss/lr/tok_s to Weights & Biases
--wandb-project my-prism   # project name (default: prism)
```

Console output (every `--log-every` steps):
```
step    40000/100000 | loss 2.8431 | lr 1.92e-04 | 12450 tok/s | 1.34s/step
```

---

## What "competitive with 1B LLMs" means

For PRISM 1B to genuinely compete, you need:
- **~500B–1T tokens** of pretraining (Chinchilla-optimal for 1B is ~20B tokens,
  but modern models overtrain 50–100× for quality).
- **~1–2 weeks** on 8×A100, or proportionally more on fewer GPUs.
- **Instruction tuning** (phase 2) for chat/instruction capability.
- **Evaluation** on standard benchmarks (MMLU, HellaSwag, HumanEval, GSM8K).

This script handles the first two. PRISM's architectural advantages (multi-rate
memory, symbolic lookup) should show up most on reasoning/code tasks
(HumanEval, GSM8K) — exactly where the toy induction results already showed it
beating Transformer and SSM baselines.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OOM` on GPU | reduce `--micro-batch-size`, increase grad accum, confirm `--grad-checkpoint` is on |
| `dataset not found` / 401 | some HF datasets need auth: `hf auth login` or set `HF_TOKEN` |
| CUDA bf16 not supported | use older GPU → `--dtype fp16` (needs GradScaler) or `fp32` |
| slow first step | normal — streaming datasets prefetch on first read; subsequent steps are fast |
| router collapse (one expert dominates) | increase `router_load_balance_weight` in the preset |
