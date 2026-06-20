# PRISM

**Polymorphic Recurrent Intelligence with Shared Memory.**

A novel language-model architecture that unifies four paradigms — sub-quadratic
recurrent mixing, heterogeneous Mixture-of-Experts, differentiable external
memory, and differentiable symbolic reasoning — under a single abstraction.

> 📺 **8-minute explainer video:** [`renders/prism_explainer.mp4`](renders/prism_explainer.mp4)
> covers the architecture and how PRISM differs from standard LLMs.

---

## The idea in one sentence

A router picks, **for each token**, which *kind* of computation it needs:
transform it (neural), store or retrieve it (memory), or reason over it
(symbolic) — all sharing one differentiable memory tape.

## Why this is different from anything on the market

| Family | Core mechanism | PRISM's distinction |
|---|---|---|
| Transformer | O(n²) self-attention | replaced by **Multi-Rate Bus** (O(n)) |
| Mamba / RWKV | single-rate recurrence | generalized to **structured multi-rate** with a per-token scale gate |
| MoE Transformers | routed **homogeneous** MLPs | generalized to **heterogeneous** experts (neural / memory / symbolic) |
| Memory nets (RETRO, NTM) | external retrieval, orthogonal to MoE | unified: memory is the **communication channel** between expert kinds |
| Neuro-symbolic | neural + external solver (not differentiable) | symbolic primitives **live in the weights**, trained end-to-end |

No model combines all four. PRISM does.

## Architecture

```
                ┌─────────────────────────────────────────┐
   token ──►    │  Multi-Rate Bus (MRB)                   │   K recurrent filters,
                │  + learned per-token scale gate         │   log-spaced decay rates
                └────────────────┬────────────────────────┘
                                 ▼
                        ┌────────────────┐
                        │  Polymorphic   │   picks ONE per token:
                        │    Router      │     neural / memory / symbolic
                        └───┬────┬───┬───┘
                            ▼    ▼   ▼
                       ┌────┐┌────┐┌─────────┐
                       │ N  ││ M  ││    S    │
                       │(MLP)││(rw)││(primitives)
                       └────┘└─┬──┘└─────────┘
                              ▼
                    ╔═══════════════════╗
                    ║  Shared Memory    ║   one tape flows through
                    ║      Bus          ║   ALL layers and time steps
                    ╚═══════════════════╝
```

- **Multi-Rate Bus** — sub-quadratic backbone, O(n), interpretable scale gates.
- **Polymorphic Router** — top-1 routing, epsilon-soft straight-through, load-balancing loss.
- **Shared Memory Bus** — one `S × d_mem` tape, NTM-style read/write, shared across layers.
- **Symbolic Expert** — 6 differentiable primitives (compare, gate, select, threshold, count, shift).

Full spec: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Results (honest benchmark)

Same training loop, optimizer, hyperparameters, seeds. CPU-only, toy scale.

| Task | Metric | PRISM | Transformer | SSM | Winner |
|---|---|---:|---:|---:|---|
| **Induction** (lookup) | cross-entropy | **1.78** | 1.95 | 2.04 | **PRISM** 🏆 |
| Copy (k=8) | cross-entropy | 2.11 | 1.64 | **1.45** | SSM |
| Mini-LM (char) | cross-entropy | 0.89 | 1.33 | **0.31** | SSM |

**PRISM wins on induction** — the associative-lookup task its heterogeneous MoE
was designed for. Routing inspection confirms genuine specialization: on a copy
task the first layer routes ~53% to memory (storing the sequence); on lookup the
last layer routes ~73% to the symbolic expert (reasoning). The architecture does
what it claims.

On dense language modelling a state-space model wins — and that's documented
honestly. PRISM's edge is compositional reasoning, not universal replacement.

Full analysis: [`RESULTS.md`](RESULTS.md).

## Quick start

```bash
git clone https://github.com/AFKmoney/prism.git
cd prism
pip install -e ".[dev]"          # torch, numpy, pytest

# Train PRISM on induction (its strongest task)
python -m prism.train --model prism --task induction --steps 400

# Compare all three models on a task
python -m prism.train --compare --task induction --steps 400

# Run the test suite (33 tests)
pytest tests/
```

### Scaling to 1B params (real pretraining, needs GPUs)

A complete scaled training pipeline — multi-GPU DDP, bf16, gradient
checkpointing, HuggingFace dataset streaming — is in
[`TRAINING.md`](TRAINING.md). Validate it on a laptop in <1 minute:

```bash
python -m prism.run_scale --smoke --phase pretrain   # tiny, CPU, no big downloads
```

Then train PRISM 1B for real (8×A100, ~1–2 weeks):

```bash
torchrun --nproc_per_node=8 -m prism.run_scale \
    --preset 1b --phase pretrain --steps 100000 \
    --seq-len 4096 --global-batch-size 256 --lr 3e-4 --dtype bf16 \
    --out-dir runs/prism-1b-pretrain --wandb
```

## Project layout

```
prism/
├── prism/
│   ├── config.py        # all hyperparameters as dataclasses
│   ├── mrb.py           # Multi-Rate Bus (sub-quadratic temporal mixing)
│   ├── memory.py        # Shared Memory Bus + read/write head
│   ├── symbolic.py      # differentiable primitive library
│   ├── experts.py       # Neural / Memory / Symbolic experts
│   ├── router.py        # Polymorphic Router (epsilon-soft top-1)
│   ├── block.py         # PRISM block (MRB + routing + residual)
│   ├── model.py         # full PRISM model
│   ├── baselines.py     # Transformer + SSM baselines (fair comparison)
│   ├── train.py         # toy training harness (CLI)
│   ├── train_scale.py   # 1B/350m/tiny presets + dataset mixes
│   ├── data_scale.py    # streaming multi-dataset dataloader
│   └── run_scale.py     # scaled trainer (DDP, bf16, checkpointing)
├── tasks/
│   ├── copy.py          # working-memory probe
│   ├── induction.py     # associative lookup (PRISM's target)
│   └── mini_lm.py       # char-level LM on bundled corpus
├── tests/               # 33 unit tests
├── renders/
│   └── prism_explainer.mp4   # 8-min architecture explainer video
├── docs/ARCHITECTURE.md
├── TRAINING.md          # 1B-scale training guide (datasets, DDP, hyperparams)
├── RESULTS.md
└── pyproject.toml
```

## Design principles

1. **One abstraction, many implementations.** The Expert interface is uniform;
   implementations differ in *kind* — this is what makes the routing meaningful.
2. **Every parameter must learn.** No dead weights. The last block drops its
   memory expert (writes would never be read) so nothing is wasted.
3. **Interpretability by construction.** The MRB gate and router selection are
   directly readable — you can see *which scale* and *which strategy* each token uses.
4. **Honest baselines.** Same loop, optimizer, seeds. Results reported whether
   or not PRISM wins.

## Scaling to 1B+ parameters

The toy config (~186k params) runs on CPU. The architecture is designed to
scale: `d_model`, `num_layers`, `num_rates`, and `num_slots` are all in
`PrismConfig`, and the code runs on GPU unchanged (`.to('cuda')`). The
sub-quadratic backbone and constant-memory-size tape make long-context scaling
particularly favourable.

## License

MIT.
