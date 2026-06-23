# Maintenance.md — PRISM-KB project guide

> Guide for AI agents (and humans) working on this codebase. Read this first.

## What this project is

PRISM-KB is a fork of [PRISM](https://github.com/AFKmoney/prism) that adds a
**holographic (VSA) memory** — an algebraic knowledge store that requires zero
gradient to add facts. The central result: holographic tape achieves +0.355
retrieval specificity vs +0.006 for the neural attention tape (60× better,
zero training). See `PRISM-HOLO.md` and `paper/PRISM-Holo_Paper.md`.

## The two axes (CRITICAL — do not conflate)

| Axis | What | Training? | How |
|---|---|---|---|
| **1. Scaling** (350M→1B) | model capacity | YES (PCS makes it cheaper) | `--progressive` |
| **2. Adding knowledge** | what the model knows | **NO** (zero gradient) | `tape.bind(key, value)` |

Axis 2 is the breakthrough. Axis 1 is just cheaper training. See README §"Two axes".

## Repository layout

```
prism/           # the model + training + Holo memory
  config.py      # PrismConfig (holo_mode flag lives here)
  model.py       # Prism (the full model)
  memory.py      # MemoryState (from_knowledge = the seeding patch)
  holo.py        # HoloTape (VSA), HoloEncoder, HoloHead (production integration)
  holo_loss.py   # consistency + contrastive losses (train the encoder)
  holo_data.py   # retrieval dataset mix (NQ, TriviaQA, MS-MARCO, FineWeb-Edu)
  holo_inference.py  # bind facts → generate (Axis 2 user API)
  run_holo_train.py  # the trainer (--progressive for PCS)
  pcs.py         # Progressive Capacity Stacking (grow_model)
  train_encoder.py    # CPU-only encoder training (the gap-closing step)
  eval_holo.py   # evaluation harness (3 probes, reproducible)
  two_axes_demo.py    # end-to-end demo of both axes
tasks/           # synthetic task datasets (copy, induction, reasoning, retrieval)
tests/           # 103 tests — run with: pytest tests/
paper/           # the academic paper (English, Philippe-Antoine Robert)
```

## Commands

```bash
# Run all tests (must stay green — 103 tests):
pytest tests/

# Validate the Holo pipeline end-to-end on CPU:
python -m prism.run_holo_train --smoke
python -m prism.two_axes_demo

# Run the eval harness:
python -m prism.eval_holo --probe all --D 2048

# Train just the encoder on CPU (the gap-closing step):
python -m prism.train_encoder --steps 300

# Production training (cluster, 8xA100):
torchrun --nproc_per_node=8 -m prism.run_holo_train \
    --progressive --steps 50000 --out-dir runs/prism-holo-1b --wandb
```

## Honest status (do not overclaim)

- **Pure VSA retrieval (+0.355)**: PROVEN, reproducible (test_holo.py).
- **Integrated HoloHead (split key/value)**: WORKS, beats neural at random init
  (+0.0525 vs +0.0096), but below the +0.355 ceiling because the encoder is
  untrained.
- **Encoder training (train_encoder.py)**: WORKS (encoder learns), but the
  toy model has random embeddings so the integrated specificity doesn't
  improve yet. Needs a real trained PRISM to see the effect.
- **PCS (scaling)**: WORKS (grow_model preserves weights, validated).
- **Real 1B training**: NOT DONE — requires GPU. The script is ready.

When reporting results, always cite the measured number and the regime. Do
NOT claim the +0.355 transfers to real text without measuring it on a real
model. Do NOT claim "no retraining" for scaling — only for adding knowledge.

## Conventions

- **Honesty over hype.** If a result is negative, document it (see
  COGLOOP.md Phase 3, train_encoder.py result). Never tune a test to pass.
- **English only** in code, docs, paper, commits.
- **Every new module gets tests.** The suite must stay green.
- **VSA init matters**: HoloTape.H starts at ZEROS (not ones). Binarization
  happens only at retrieval. Getting this wrong silently destroys the signal.
- **dim-matched injection**: when injecting encoder weights into HoloHead,
  D must equal num_slots * d_mem exactly. Truncation breaks the structure.

## The VSA bug that determined everything

The first HoloTape impl initialized H to +1^D and re-binarized after each
bind. This scored ρ ≈ 0 (same as neural baseline). Fixing to zeros-init +
late binarization raised ρ to +0.355. If you ever touch holo.py, re-read
this. The test `test_bind_unbind_roundtrip_single_fact` guards it.
