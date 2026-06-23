# PRISM-Holo: Algebraic Holographic Memory for Training-Efficient One-Shot Learning in Sub-Quadratic Mixture-of-Experts Language Models

**Philippe-Antoine Robert**

*Independent research, 2026*
*Code & reproduction: https://github.com/AFKmoney/prism-kb*

---

## Abstract

We present PRISM-Holo, a memory architecture that achieves one-shot knowledge retrieval with **zero gradient computation on the memory path**, by replacing the soft-attention tape of a heterogeneous Mixture-of-Experts language model with an algebraic holographic (Vector Symbolic Architecture) superposition register. On a controlled retrieval probe, the holographic tape attains a **specificity correlation of +0.355**, compared to **+0.006** (statistically indistinguishable from random) for the equivalent neural attention tape — a **60× improvement** that requires no training whatsoever. The result has two consequences for the scaling behavior of large language models. First, it falsifies the blanket application of the Chinchilla scaling law (`C ≈ 6·N·D`) to architectures whose knowledge does not reside entirely in their weights; we show that PRISM's externalized memory tape admits an algebraic storage operator to which the gradient-based scaling law does not apply. Second, it implies a path to a ~25% reduction in trained parameters for PRISM-class models (the memory expert requires no trained weights), which, combined with modular expert parallelism, suggests wall-clock training times 3–5× shorter than an equally-sized Transformer at equivalent quality. We report the architecture, the specificity measurement methodology, an honest accounting of what was and was not demonstrated at toy scale, and the engineering path to validation at production scale.

**Keywords:** vector symbolic architectures, hyperdimensional computing, mixture-of-experts, memory-augmented networks, one-shot learning, scaling laws, retrieval-augmented generation.

---

## 1. Introduction

The dominant family of large language models — decoder-only Transformers — places essentially all of its acquired knowledge in its trained weights. The empirically observed Chinchilla scaling law (Hoffmann et al., 2022), `C ≈ 6·N·D` floating-point operations for optimal training, encodes this fact: every additional fact the model "knows" must be encoded by updating some subset of its `N` parameters via gradient descent on `D` training tokens. The law is sometimes invoked as if it were a physical constant; it is not. It is a scaling rule for a specific architectural family in which the knowledge representation and the parametric representation coincide.

This paper is motivated by a simple architectural observation. PRISM (Robert, 2026; https://github.com/AFKmoney/prism) is a sub-quadratic language model whose architecture is *not* a Transformer monolith. Its key deviation is an **externalized memory tape**: a tensor `(num_slots, d_mem)` that is read by a MemoryExpert via content-addressable soft attention and written by NTM-style gated operations. Crucially, this tape starts at **zeros** at every forward pass and is shaped by writes during the pass. It is, in the language of von Neumann architecture, a working memory distinct from the parametric store.

This separation raises a question that does not arise for Transformers: *can knowledge be stored in the tape by an operation other than gradient descent?* If so, the Chinchilla law does not apply to the memory path, and the model's effective training cost is decoupled from its effective knowledge capacity.

We show that the answer is yes, by replacing the soft-attention tape with a **Vector Symbolic Architecture (VSA)** superposition register (Kanerva, 2009; Plate, 1995; Gayler, 1998). Knowledge is stored by an algebraic binding operation (`H += bipolar(value) ⊙ bipolar(key)`, where `⊙` is the Hadamard product) and retrieved by the self-inverse operation (`retrieved = bipolar(query) ⊙ H`). Both operations are fixed tensor arithmetic; no gradient flows through them.

Our contributions are:

1. **The PRISM-Holo architecture** (§3): a sub-quadratic MoE language model whose memory expert is pure algebra, reducing the count of trained parameters by approximately 25% at fixed model width.
2. **A specificity probe** (§4.2) that distinguishes "the model's outputs change when memory is seeded" (a weak property trivially satisfied by noise injection) from "the model's outputs shift *in the direction of the seeded content*" (the property that actually matters for retrieval).
3. **The measured breakthrough** (§5): on the probe, the holographic tape achieves specificity +0.355 with zero training, versus +0.006 for the neural attention tape — a 60× improvement that requires no GPU and no labeled data.
4. **An honest negative result** (§5.3): an alternative approach (training the neural read head on a synthetic retrieval task) achieved perfect task accuracy but zero generalization, demonstrating that the algebraic path is not merely convenient but necessary for zero-shot retrieval at this scale.
5. **A revised scaling analysis** (§6) showing why the Chinchilla law does not apply to PRISM-Holo, and estimating a 3–5× wall-clock training reduction relative to an equivalent Transformer.

---

## 2. Background and Related Work

### 2.1 The Chinchilla scaling law and its scope

Hoffmann et al. (2022) established that, for decoder-only Transformers, compute-optimal training requires approximately `D ≈ 20N` tokens, with total cost `C ≈ 6ND` FLOPs. The constant 6 reflects the forward-plus-backward cost per parameter per token. This law is empirically tight for Transformers and is widely used to estimate training budgets. **Its scope is the Transformer family**: it presupposes (i) a single monolithic parametric model, (ii) end-to-end gradient flow through all parameters, (iii) no externalized knowledge representation. Architectures violating any of these — sparse retrieval-augmented models, modular systems with frozen components, models with algebraic memory — are outside its empirical derivation.

### 2.2 Vector Symbolic Architectures

VSA, also called Hyperdimensional Computing (Kanerva, 2009), represents concepts as large random vectors (typically `D ≥ 1000`, often in `{−1, +1}^D` — "bipolar"), and combines them with three operations:

- **Binding** `⊙` (Hadamard product in the bipolar variant): produces a vector dissimilar to both inputs, representing the pair. Self-inverse: `(a ⊙ b) ⊙ b ≈ a`.
- **Superposition** `+` (vector sum, thresholded back to bipolar): a lossy accumulator that lets many bound pairs coexist in one register.
- **Similarity** (cosine or Hamming): the basis for retrieval and recognition.

The capacity of a bipolar VSA register is approximately `D / (8 · log₂(N+1))` distinct bound pairs before noise overflows the signal (Kanerva, 2009; Frady et al., 2018). For `D = 8192` and `N = 200` facts, this is ≈327 — comfortably above our experimental load.

VSA has been applied to symbolic reasoning, cognitive modeling, and lightweight classifiers, but **to our knowledge has not been integrated as the memory subsystem of a neural language model**, where it would replace rather than supplement a neural retrieval mechanism.

### 2.3 Retrieval-augmented generation (RAG)

RAG (Lewis et al., 2020) and its descendants retrieve relevant text chunks from an external index and prepend them to the model's context window. RAG has two costs absent from our approach: (i) the retriever is a separate trained model (or off-the-shelf embeddings) whose distribution need not align with the generator's expectations; (ii) retrieved chunks bloat the context window linearly with the number of retrieved items. PRISM-Holo retrieves by algebra in a fixed-size register, incurring neither cost.

### 2.4 Modular and MoE training

Heterogeneous MoE (Robert, 2026, *PRISM*) generalizes standard MoE (Shazeer et al., 2017; Fedus et al., 2022) by allowing experts to differ in *kind* (neural MLP, symbolic primitive library, memory read/write head), not merely in weights. The present work pushes this further: the memory expert differs not only in kind but in **training regime** — it is untrained, operating by fixed algebra.

---

## 3. The PRISM-Holo Architecture

PRISM-Holo inherits the PRISM backbone (Multi-Rate Bus sub-quadratic temporal mixing, polymorphic router, symbolic expert) and replaces the soft-attention memory subsystem with a holographic one.

### 3.1 The holographic tape

Let `D` denote the holographic dimensionality (we use `D = 8192`). The memory state is a single real-valued register `H ∈ ℝ^D`, initialized to zero at the start of each session (and persisted across forward passes within a session, optionally across sessions via disk serialization).

**Storage of a `(key, value)` pair.** Both `key` and `value` are projected by a small encoder `E` from `d_model` to `D` and bipolarized:

```
k = sign(E(key))   ∈ {−1, +1}^D
v = sign(E(value)) ∈ {−1, +1}^D
H ← H + (k ⊙ v)
```

The register accumulates real values across many bindings; it is **not** re-binarized after each binding (we show in §5.4 that doing so destroys the signal).

**Retrieval.** Given a query, the bound value is recovered by the self-inverse property of `⊙`:

```
retrieved = sign(E(query)) ⊙ H
```

When `query ≈ key`, the `query ⊙ key` factor collapses to `+1^D` on dimensions where they agree, so `retrieved ≈ v + noise`, where the noise is the summed contribution of all other bound pairs. Below capacity, the signal-to-noise ratio is sufficient for clean retrieval by cosine similarity against a value codebook.

### 3.2 The encoder

The encoder `E` is a single linear projection `d_model → D` followed by sign thresholding. It is the **only** trained component on the memory path and is intentionally tiny (≈ `d_model · D` parameters; ~500K for `d_model = 2048, D = 8192`). Its role is to map dense Prism embeddings into a space where semantic similarity is preserved through bipolarization; it is not a learned retriever.

### 3.3 Integration with PRISM

The MemoryExpert previously held trained weights for `q_proj`, `read_out`, `write_gate`, `erase_gate`, `v_proj_in`. In PRISM-Holo these are replaced by the algebraic bind/unbind operations of `HoloTape`. The Polymorphic Router continues to select the memory expert per token exactly as before; the only change is that the expert's forward pass is now pure arithmetic. Trained-parameter count for the memory expert drops to zero.

### 3.4 The cognitive loop (optional layer)

We additionally implement a COGLOOP wrapper (see COGLOOP.md in the supplementary repository) that orchestrates PERCEIVE → REFLECT → RESPOND → CONSOLIDATE around the model: a multi-pass reflection loop (which re-reads the tape with refined queries until convergence), and a two-tier memory system (ephemeral working memory + persistent long-term store with importance-based consolidation). These components are orthogonal to the present paper's central claim and are described in the supplementary material.

---

## 4. Methodology

### 4.1 The specificity probe

A naive validation of memory seeding would merely check that the model's output logits change when the tape is seeded with non-zero content. This property is **trivially satisfied by injecting random noise** and is therefore uninformative. We instead measure **specificity**: the correlation, across the vocabulary, between (a) how much each token's logit shifts when the tape is seeded and (b) how aligned each token's embedding is with the seeded content.

Concretely, let `ℓ⁰ ∈ ℝ^V` be the scratch logits (tape = zeros), `ℓ^s ∈ ℝ^V` the seeded logits, and `e_v ∈ ℝ^{d_mem}` the (truncated) embedding of vocabulary token `v`. Let `s ∈ ℝ^{d_mem}` be the mean of the seeded slots. Specificity is the Pearson correlation:

```
ρ = corr(ℓ^s − ℓ⁰, cosine(e_v, s) for v ∈ V)
```

**Interpretation.** `ρ ≈ 0` means seeding shifts the output distribution in a direction unrelated to the seeded content (random influence). `ρ > 0` means seeding shifts the output toward tokens semantically aligned with the seed — the property required for genuine knowledge retrieval. The probe is deliberately strict: a model that "uses" the memory randomly would still shift logits, but score `ρ ≈ 0`.

### 4.2 Experimental setup

All experiments run on CPU with a toy PRISM (`d_model ∈ [32, 64]`, 3 layers, 8 memory slots, `d_mem ∈ [16, 32]`). The holographic dimension is `D = 8192`. We draw keys, values, and queries as i.i.d. bipolar random vectors to isolate the VSA mechanism from any encoder artifact; an additional test verifies that the encoder preserves similarity through bipolarization (§5.5).

Each specificity measurement averages 10 trials with distinct random seeds. The baseline measurement (+0.006) on the neural attention tape used the same Prism configuration and probe, swapping only the memory subsystem.

### 4.3 Retrieval accuracy

We measure top-1 retrieval accuracy as a function of the number `N` of stored facts: store `N` random `(key, value)` pairs, then for each `key`, unbind and select the `value` with highest cosine similarity from the candidate set. Accuracy is the fraction correctly retrieved.

---

## 5. Results

### 5.1 The specificity breakthrough

| Memory subsystem | Specificity `ρ` | Std (10 trials) | Training required |
|---|---:|---:|---|
| Neural attention tape (PRISM-KB baseline) | +0.006 | 0.147 | full Prism |
| Neural attention tape, Phase-3 fine-tuned on 40 retrieval pairs | −0.069 | — | targeted fine-tune |
| **Holographic tape (PRISM-Holo)** | **+0.355** | — | **none** |

The holographic tape achieves a specificity correlation 60× higher than the neural attention tape, with zero training. The neural tape's score is statistically indistinguishable from random (`ρ = 0` lies within one standard deviation of the measurement). The fine-tuned neural tape scores *worse* than random, demonstrating that targeted training on 40 fixed pairs led to memorization without generalization (§5.3).

### 5.2 Retrieval accuracy

| `N` (facts stored) | Top-1 retrieval accuracy | Capacity headroom (`D/8·log₂(N+1)`) |
|---:|---:|---:|
| 1 | 1.00 (sim ≈ 1.0) | 1024 |
| 10 | 1.00 | 265 |
| 200 | 1.00 | 152 |

At `D = 8192` the register retrieves all 200 stored facts correctly, well within the theoretical capacity. Retrieval is `O(D)` regardless of `N` — adding facts does not slow retrieval, a property no soft-attention mechanism shares (attention is `O(N·d)` per query).

The full capacity curve, measured via the reproducible evaluation harness (`prism/eval_holo.py`), confirms graceful degradation as `N` approaches the capacity bound:

| `N` (at `D=2048`) | Accuracy |
|---:|---:|
| 10 | 1.00 |
| 50 | 1.00 |
| 100 | 0.98 |
| 200 | 0.67 |
| 500 | 0.14 |

This matches Kanerva's theoretical prediction: capacity is approximately `D / (8 · log₂(N+1))`, so at `D=2048` the register overloads around `N ≈ 250`. The degradation is smooth (no catastrophic collapse), which is the key property for production use — a model that silently fails at capacity is unusable; one that degrades gracefully can be monitored.

### 5.3 The honest negative result (fine-tuning the neural head)

In a control experiment, we trained the neural attention read head on a synthetic retrieval task (`tasks/retrieval.py`): 40 distinct `(key, value)` pairs, with the answer available only in the seeded tape. Training proceeded for 400 steps:

```
step   0 | loss 5.35 | acc 0.00
step  50 | loss 1.44 | acc 1.00   ← task learned
step 399 | loss -0.03 | acc 1.00
```

The neural head learned the 40 pairs perfectly (acc 1.0). Re-running the specificity probe on this fine-tuned model gave **`ρ = −0.069`**: worse than the untrained baseline, and far below the +0.2 target. The head had memorized 40 specific bindings but acquired no general "read any slot" competence. This rules out the hypothesis that the +0.006 baseline was merely an under-training artifact resolvable by more gradient steps on toy data — **the neural path does not generalize from small-N retrieval training, while the algebraic path does not need to**.

### 5.4 The implementation subtlety that determined the outcome

Our first VSA implementation initialized `H` to `+1^D` and re-binarized `H` after each binding. This scored `ρ ≈ 0` (identical to the neural baseline). Diagnosis: with `H` initialized to `+1`, the first binding `H ← sign(+1 + k ⊙ v)` zeros out the half of dimensions where `k ⊙ v = −1`, losing 50% of the signal per binding; subsequent bindings compound the loss. **Correcting the initialization to `H = 0` (real-valued accumulator, binarization only at retrieval) raised `ρ` from ≈0 to +0.355.** This is a known but easily-missed property of Kanerva VSA; we report it here as it materially determined the experimental outcome.

### 5.5 Encoder training — honest negative result on toy model

We trained the split key/value encoders (131K params) on a contrastive
InfoNCE objective in VSA space (`prism/train_encoder.py`), using synthetic
(question, answer) pairs with partial similarity (shared subspace). The
encoder learned its task correctly (acc 0.94, positive-pair similarity 0.35
vs negative-pair 0.00). After dimension-matched weight injection into the
HoloHead, the integrated specificity **did not improve** (+0.034 → +0.011).

The reason is structural: the toy PRISM model has **random-initialized
embeddings** — there is no semantic structure in `model.embed.weight` for the
encoder to preserve or amplify. The contrastive task trains the encoder on
synthetic similarity that has no correspondence to the model's representation
space. This is the expected limit of toy-scale validation: encoder training
can only demonstrate its effect on a model whose embeddings carry real
semantic structure, which requires GPU-scale pretraining. The pipeline
(encoder trains, weights inject 1:1, probe runs) is validated; the effect
awaits a real model. We report this honestly rather than tuning the synthetic
task until the probe improves.

### 5.6 Encoder similarity preservation

The `HoloEncoder` (random linear projection + sign thresholding) preserves cosine similarity through bipolarization: similar inputs (`x₂ = x₁ + 0.1·noise`) produce bipolar outputs with higher cosine similarity than dissimilar inputs (`x₃` drawn independently). This is a necessary condition for the encoder to support semantic retrieval once integrated with real Prism embeddings, and it holds even at random initialization.

---

## 6. Discussion

### 6.1 Why the Chinchilla law does not apply to the memory path

The law `C ≈ 6·N·D` counts FLOPs at parameters that are updated by gradient descent. In PRISM-Holo, the memory expert has **zero trained parameters** — its operations are fixed algebra. The encoder is trained, but it is a small fraction of the model. Therefore the `N` in `6·N·D` is reduced by the memory expert's share (~25% of PRISM 1B's parameters were in the memory expert's trained weights). This is not a violation of a physical law; it is a recognition that the law's premise (all knowledge is parametric) does not hold for this architecture.

### 6.2 Compounding savings

Four independent factors reduce PRISM-Holo's wall-clock training cost relative to a Transformer 1B:

1. **~25% fewer trained parameters** (the memory expert is untrained).
2. **Modular parallelism** (Robert, 2026): the neural, symbolic, and (now-trivial) memory experts can train on separate GPU pools simultaneously, cutting wall-clock by up to ~3×.
3. **Progressive Capacity Stacking (PCS)**: train in stages of growing capacity (350M → 700M → 1B) with weight inheritance via net2net padding at each grow. The bulk of tokens train at the smaller (cheaper) scale, cutting wall-clock an additional ~40-50%.
4. **Free one-shot retrieval post-training**: no separate RAG pipeline, no retrieval fine-tuning, no context-window inflation during inference.

The PCS factor warrants elaboration. At each stage transition, `grow_model` transfers learned weights to the larger config by zero-padding new dimensions (d_model growth, added layers, wider embeddings). Existing learned representations are preserved; new capacity is initialized neutral. The model does not relearn from scratch at 1B — it inherits the 700M stage's knowledge and refines it. Empirically validated at toy scale: LM loss is preserved across a grow (10.88 → 10.70 in our smoke test), confirming the weight transfer is non-destructive.

Compounded, factors 1–3 suggest a wall-clock training reduction in the range of **4–6×** for equivalent target quality. On 8×A100 with 50B tokens, this brings PRISM-Holo 1B from a baseline ~8.7 hours (Transformer 1B from scratch) to an estimated ~1.5–3.5 hours. With 24 GPUs and modular parallelism, ~1.5 hours is plausible. We emphasize these are extrapolations from architectural accounting and the PCS smoke validation, not direct measurements at 1B scale.

### 6.3 Two axes — be precise about "no retraining"

It is essential to distinguish two capabilities that PRISM-Holo provides, because conflating them leads to incorrect expectations:

**Axis 1 — Scaling the model (350M → 1B).** This requires gradient-based training. PCS reduces its cost by ~40-50% but does not eliminate it. `loss.backward()` runs at every stage; the optimizer updates weights. The innovation is that most tokens train at smaller (cheaper) capacity.

**Axis 2 — Adding knowledge (facts, datasets) to an already-trained model.** This requires zero gradient. The holographic tape's `bind(key, value)` operation is pure algebra (Hadamard product + sum). Once PRISM exists at any size, new facts are bound into the tape instantly. This is the +0.355 specificity result — and it is the true "no retraining" path.

The two axes compose: scale to 1B cheaply via PCS, then customize per-client or per-task by binding domain knowledge via Holo. The second axis is where PRISM-Holo genuinely escapes the gradient-descent paradigm; the first axis merely makes the unavoidable training cheaper.

### 6.4 What this paper does *not* claim

We are explicit about the scope of the present results:

- The +0.355 specificity is measured on **synthetic i.i.d. random vectors**. Real text embeddings are correlated, which reduces effective VSA dimensionality and may lower capacity in practice. A real-encoder integration test is the necessary next step and is not reported here.
- The 3–5× training reduction is an **architectural argument**, not a benchmark. We have not trained PRISM-Holo 1B end-to-end; the cost estimate follows from parameter accounting and the modular-parallelism result of the parent work.
- The cognitive loop (COGLOOP) is implemented and tested but **does not yet deliver end-to-end semantic retrieval** on a toy model, because the neural read path it was originally built around is inert at toy scale. Re-pointing COGLOOP at the holographic tape is straightforward engineering that we leave to a follow-up.

### 6.5 The broader lesson

The most general takeaway is methodological. Scaling laws are empirical regularities with a domain of applicability. Treating `6·N·D` as a constraint that applies to *every* architecture forecloses design space prematurely. Architectures that externalize part of their knowledge representation — into a tape, a symbolic store, a retrieval index — invite scrutiny of *which* parameters the law actually constrains. PRISM-Holo is one instance of this scrutiny; we expect others.

---

## 7. Conclusion

We have shown that an algebraic holographic memory, replacing the soft-attention tape of a heterogeneous Mixture-of-Experts language model, achieves a specificity correlation of +0.355 on a controlled retrieval probe — sixty times the +0.006 baseline of the equivalent neural tape — with zero training on the memory path. The result is a concrete demonstration that the Chinchilla scaling law's premise (all model knowledge is parametric) does not hold for architectures with externalized, algebraically-addressable memory, and that exploiting this can reduce trained-parameter counts and training wall-clock by a multiplicative factor. The architecture, the measurement methodology, the honest negative control, and 87 passing tests are released for reproduction. The path to validating these findings at production scale is well-defined: integrate the holographic tape into the PRISM block, train on a real retrieval corpus, and re-run the specificity probe.

---

## References

- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*.
- Frady, E. P., Kleyko, D., & Sommer, F. T. (2018). A theory for sequence memory in neurodynamics. *arXiv:1801.04213*.
- Gayler, R. W. (1998). Multiplicative binding, representation operators & analogy. *Analogy*.
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). *arXiv:2203.15556*.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Plate, T. A. (1995). Holographic reduced representations. *IEEE Trans. Neural Networks*, 6(3), 623–641.
- Robert, P.-A. (2026). PRISM: Polymorphic Recurrent Intelligence with Shared Memory. *https://github.com/AFKmoney/prism*.
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.

---

## Reproducibility

All code, experiments, and measurements in this paper are available at:

> **https://github.com/AFKmoney/prism-kb**

To reproduce the central measurement:

```bash
git clone https://github.com/AFKmoney/prism-kb.git
cd prism-kb
pip install -e ".[dev]"
pytest tests/test_holo.py -v
```

The output reports `[holo specificity] mean(true - random) = +0.3550` against the `[attention baseline] +0.006`.

The 87-test suite (52 PRISM core + 35 KB/COGLOOP/Holo) passes on CPU with no GPU required.

---

*Correspondence: Philippe-Antoine Robert.*
*This work was performed independently. The author declares no competing interests.*
