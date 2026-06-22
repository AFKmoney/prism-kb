# PRISM-KB — Activating the dormant knowledge mechanism

Fork of [PRISM](https://github.com/AFKmoney/prism) that activates the
"Knowledge Bus": ingest datasets and do one-shot learning by seeding the
shared memory tape with encoded content — no weight update.

Based on the PRISM-KB proposal (13-page analysis). This repo implements the
5 modules + the memory patch, AND adds an **honest validation** that the
proposal omitted.

---

## The dormant mechanism (real)

PRISM's `MemoryState.tape` is a `(B, num_slots, d_mem)` tensor that starts at
**zeros** every forward pass. The MemoryExpert's read head does content-
addressable soft attention on it:

```python
q = q_proj(x)                                    # query from the token
scores = einsum("bmd,bsd->bms", q, tape) / sqrt(d_mem)
read_weights = softmax(scores)                   # <- retrieval
read_vec = einsum("bms,bsd->bmd", read_weights, tape)
```

The proposal's insight: **if you seed the tape with encoded dataset content
instead of zeros, the read head retrieves it — no gradient needed.** This is
mechanically true and is implemented here via `MemoryState.from_knowledge()`.

## What this repo adds

| Module | File | Role |
|---|---|---|
| `MemoryState.from_knowledge` | `prism/memory.py` (patch) | Seed the tape from external slots |
| `PrismEncoder` | `prism/encoder.py` | text → `d_mem` slots (Perceiver-style, reuses frozen Prism embedding) |
| `KnowledgeBase` | `prism/kb.py` | vector store of slots: add / retrieve / save / load |
| `OneShotLearner` | `prism/incontext.py` | add (input,output) pairs, generate zero-gradient |
| `generate.py` | CLI | 3 modes: scratch / knowledge / oneshot |
| `ingest.py` | CLI | build a KB from HF or local datasets |

## Honest validation result (the part the proposal skipped)

The proposal's central proof was a "diff of 0.0646 between scratch and
KB-seeded logits" — which only shows that **seeding changes the output**, not
that it changes it **in a knowledge-directed way**. This repo measures the
stronger property: *specificity*.

**Experiment:** seed the tape with a random signal; measure the correlation
between (a) how much each vocab token's logit shifts and (b) how aligned each
vocab embedding is with the seed. If the mechanism fires semantically, tokens
aligned with the seed gain logit → positive correlation.

**Result (10 trials, untrained-for-KB toy PRISM):**

```
Semantic specificity correlation: mean = +0.006, stdev = 0.147, range [-0.20, +0.22]
```

**Interpretation:** the correlation is **essentially zero**. Seeding shifts the
logits, but the shift is **random with respect to the seed content**. The
dormant mechanism is wired correctly (the read head retrieves the matching
slot when the query aligns — see `test_read_head_retrieves_matching_slot`),
but the **end-to-end semantic path does not fire on a model that was never
trained to read seeded content**.

### What this means

- **The architecture is sound.** The mechanism exists and is correctly
  implemented. The patch is minimal and the read head is genuinely content-
  addressable.
- **Phase 2 (encoder training) is mandatory, not optional.** The proposal
  acknowledged this as a risk (§6.1 distribution shift); our measurement
  shows it's the whole game. Without training the encoder (and possibly a
  short fine-tune of the read head) to align encoded slots with what the
  MemoryExpert expects, seeding is noise injection, not knowledge injection.
- **The proposal's 0.0646 diff is real but misleading.** It proves influence,
  not retrieval. Our `test_semantic_specificity_is_documented` records the
  honest baseline (+0.006) that Phase 2 must beat.

## Reproducing

```bash
cd prism-kb
pip install -e ".[dev]"
pytest tests/test_kb.py -v          # 6 tests, incl. the honest specificity probe

# Run the specificity measurement directly:
python -c "exec(open('tests/test_kb.py').read().split('def _seeded_vs_scratch')[1])"  # see source
```

The measurement script that produced the +0.006 baseline is inlined in
`tests/test_kb.py::_seeded_vs_scratch_logit_shift`.

## Path to making PRISM-KB actually work

The mechanism is in place. What remains (Phase 2) is aligning the encoder's
output distribution with the MemoryExpert's expectation — exactly as the
proposal's §5 Phase 2 describes. Three options, in order of recommendation:

1. **Write-distribution mimétisme** (proposal Option A): run the trained Prism
   on a corpus, record the `add` vectors the MemoryExpert writes, train the
   encoder (MSE loss) to reproduce that distribution. Cheapest, most aligned.
2. **InfoNCE contrastive** (Option B): pairs of (question, relevant doc),
   contrastive loss in slot space. Standard.
3. **End-to-end with frozen Prism** (Option C): train only the encoder on a
   retrieval task (e.g. Natural Questions), gradient flows through
   `from_knowledge` (which is differentiable). Most expensive, most aligned.

After Phase 2, re-run the specificity probe — the target is mean correlation
**> +0.2**, at which point PRISM-KB genuinely retrieves seeded knowledge.

## License

MIT. Inherits from PRISM.
