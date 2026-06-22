# PRISM-KB

> **Fork of [PRISM](https://github.com/AFKmoney/prism)** exploring the dormant
> knowledge mechanism — culminating in a **measured holographic breakthrough**.

## 🏆 The breakthrough — PRISM-Holo

> 👉 **[`PRISM-HOLO.md`](PRISM-HOLO.md)** — full writeup + reproduction.

Replacing the soft-attention memory tape with an **algebraic holographic (VSA)
tape** yields:

| Metric | Neural attention tape | **Holographic tape** |
|---|---:|---:|
| Specificity correlation | +0.006 (random) | **+0.355** ✅ |
| 200 facts retrieved | n/a | **100%** |
| Training required | full Prism | **NONE** |

The holographic tape stores and retrieves facts **algebraically, with zero
training, at 60× the specificity of the neural tape**. This is the path out of
the "6·N·D" Chinchilla trap (which only applies to monolithic Transformers, not
to PRISM's externalized-memory architecture).

## The two axes — be precise about what "no retraining" means

PRISM-Holo provides **two independent capabilities**, and it is important not
to conflate them:

| Axis | What it changes | Requires training? | How |
|---|---|---|---|
| **1. Scaling the model** (350M → 1B) | Model capacity | ✅ **Yes** (but ~40-50% cheaper via PCS) | `--progressive` flag: train at small scale, grow weights, fine-tune |
| **2. Adding knowledge** (facts, datasets) | What the model knows | ❌ **NO** — zero gradient | `tape.bind(key, value)`: pure algebra, instant |

- **Axis 1 (PCS)** makes training a 1B model affordable: stages 350M → 700M → 1B with `grow_model()` transferring weights at each step. The bulk of tokens train on the small model. Real gradient descent, but ~40-50% less wall-clock than from-scratch 1B.
- **Axis 2 (Holo)** is the true "no retraining" path: once a model exists at any size, you bind new facts into the tape algebraically. Zero backward, zero GPU. This is the +0.355 specificity result.

Run `python -m prism.two_axes_demo` to see both axes composed end-to-end (CPU, ~seconds).

**Together:** scale to 1B cheaply (PCS), then customize per-client or per-task by binding facts (Holo). Adding new knowledge to an already-trained model requires no retraining.

## Layers of work in this repo

1. **[`PRISM-KB.md`](PRISM-KB.md)** — seed the tape for one-shot learning.
   Honest baseline: +0.006 specificity (random) without training.
2. **[`COGLOOP.md`](COGLOOP.md)** — PERCEIVE → REFLECT → RESPOND → CONSOLIDATE
   cognitive loop. Persistent memory, multi-pass reflection, double-layer
   consolidation. Phase 3 probe confirmed the neural read head doesn't
   generalize from 40 seeds.
3. **[`PRISM-HOLO.md`](PRISM-HOLO.md)** — the answer. Algebraic VSA tape.
   +0.355 specificity, zero training, 200 facts 100% retrieved. **This is the
   real breakthrough.**

4. **PCS (Progressive Capacity Stacking)** — scale 350M → 700M → 1B with
   weight inheritance. ~40-50% wall-clock reduction vs from-scratch 1B.
   `--progressive` flag in `run_holo_train.py`.

**Modules:**

| Layer | Files |
|---|---|
| KB seeding | `encoder/kb/incontext/generate/ingest` |
| COGLOOP | `capture/reflect/cogmemory/cogloop` |
| Holo memory | `holo.py` (HoloTape, HoloEncoder, HoloHead) |
| Holo training | `holo_data.py`, `holo_loss.py`, `run_holo_train.py` |
| PCS (scaling) | `pcs.py` (grow_model, stage schedule) |
| Two-axes demo | `two_axes_demo.py` (scaling + knowledge composed) |
| Phase 3 probe | `tasks/retrieval.py`, `train_retrieval.py` |

100 tests (52 PRISM + 48 KB/COGLOOP/Holo/PCS), all passing.

The base PRISM architecture below is unchanged — see the upstream repo for
full details.

---

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
- **Polymorphic Router** — real **top-k** routing (k=2 for 300m+), epsilon-soft straight-through, load-balancing loss.
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

## Training innovation levers (reduce compute, not physics)

Three independent training techniques — each a CLI flag, each measurable in
isolation. They reduce the *useful* token count; `FLOPs = 6·N·D` still holds.
Honest toy-scale results in [`RESULTS_LEVERS.md`](RESULTS_LEVERS.md).

| Lever | Flag | Toy-scale | Production |
|---|---|---|---|
| **Modular pretraining** | `--modular-phase neural\|memory\|symbolic\|assemble` | ✅ Validated | Low risk, run it |
| **Curriculum** | `--curriculum` | ✅ Validated | Low risk, run it |
| **Token recycling** | `--token-recycling` | ⚠️ Negative (wrong regime) | Enable ≥1B tokens |

**Modular pretraining** trains each expert kind separately on its optimal data
(neural→text, memory→retrieval, symbolic→code+math), then assembles them with a
short router+MRB fine-tune. 2–3× speedup via specialization + parallelism.

```bash
# Train experts in parallel (different GPU pools), then assemble
torchrun --nproc_per_node=8 -m prism.run_scale --preset 1b --modular-phase neural --out-dir runs/neural
torchrun --nproc_per_node=8 -m prism.run_scale --preset 1b --modular-phase symbolic --out-dir runs/symbolic
torchrun --nproc_per_node=8 -m prism.run_scale --preset 1b --modular-phase assemble \
    --modular-neural-ckpt runs/neural/ckpt-N --modular-symbolic-ckpt runs/symbolic/ckpt-M \
    --out-dir runs/prism-assembled
```

**Curriculum** re-weights the dataset mix across training: neural-heavy early
(fluency) → memory (retrieval) → symbolic (reasoning). The dataloader rebuilds
at each phase transition.

## Quick start

```bash
git clone https://github.com/AFKmoney/prism.git
cd prism
pip install -e ".[dev]"          # torch, numpy, pytest

# Train PRISM on induction (its strongest task)
python -m prism.train --model prism --task induction --steps 400

# Compare all three models on a task
python -m prism.train --compare --task induction --steps 400

# Run the test suite (52 tests)
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
│   ├── router.py        # Polymorphic Router (real top-k, epsilon-soft, load-balance)
│   ├── block.py         # PRISM block (MRB + routing + residual)
│   ├── model.py         # full PRISM model
│   ├── baselines.py     # Transformer + SSM baselines (fair comparison)
│   ├── train.py         # toy training harness (CLI)
│   ├── train_scale.py   # 1b/350m/300m/tiny presets + dataset mixes
│   ├── data_scale.py    # streaming multi-dataset dataloader
│   ├── modular.py       # lever 1: modular pretraining + assemble_experts
│   ├── curriculum.py    # lever 2: curriculum schedule + token recycler
│   └── run_scale.py     # scaled trainer (DDP, bf16, checkpointing, levers)
├── tasks/
│   ├── copy.py          # working-memory probe
│   ├── induction.py     # associative lookup (PRISM's target)
│   ├── reasoning.py     # chained arithmetic (lever-3 validation target)
│   └── mini_lm.py       # char-level LM on bundled corpus
├── tests/               # 52 unit tests (33 core + 19 levers)
├── renders/
│   └── prism_explainer.mp4   # 8-min architecture explainer video
├── docs/ARCHITECTURE.md
├── TRAINING.md          # 1B-scale training guide (datasets, DDP, hyperparams)
├── RESULTS.md           # toy benchmark: PRISM vs Transformer vs SSM
├── RESULTS_LEVERS.md    # honest toy-scale results for the 3 training levers
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

The toy config (~186k params) runs on CPU. Four production presets are available
via `--preset`:

| Preset | Params | Target hardware |
|---|---|---|
| `tiny` | ~7M | CPU (pipeline validation only) |
| `300m` | ~319M | 1×A100 80GB — the "small brain, big reasoning" thesis |
| `350m` | ~371M | 1×A100 80GB (validation runs) |
| `1b` | ~1.8B | 8×A100 80GB |

The architecture scales: `d_model`, `num_layers`, `num_rates`, and `num_slots`
are all in `PrismConfig`, and the code runs on GPU unchanged (`.to('cuda')`).
The sub-quadratic backbone and constant-memory-size tape make long-context
scaling particularly favourable. The 300m preset specifically compensates for
fewer parameters with more rate groups (8), wider memory (64 slots), and top-2
routing — see [`RESULTS_LEVERS.md`](RESULTS_LEVERS.md) for the honest status of
the "300M beats 1B" thesis.

## License

MIT.
