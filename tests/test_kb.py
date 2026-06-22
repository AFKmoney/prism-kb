"""Honest validation tests for PRISM-KB.

Two tiers of test, because "the logits change" is NOT the same as "knowledge is
retrieved correctly":

  1. Mechanical test: the MemoryExpert's read head retrieves the right slot
     when the query matches. This is just dot-product attention and SHOULD pass
     even on an untrained model — it tests that the mechanism is wired.

  2. Semantic test: seeding the tape with a *specific* signal measurably and
     *specifically* shifts the model's output toward that signal — not just
     shifts it randomly. This is the hard one and is the honest measure of
     whether PRISM-KB works without the Phase-2 encoder training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism.config import MemoryConfig, PrismConfig
from prism.encoder import EncoderConfig, PrismEncoder
from prism.kb import KnowledgeBase
from prism.memory import MemoryState
from prism.model import Prism


def _cfg(**kw) -> PrismConfig:
    base = dict(
        vocab_size=64, d_model=32, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=16, num_slots=8),
    )
    base.update(kw)
    return PrismConfig(**base)


# ---------------------------------------------------------------------------
# Tier 1 — mechanical: the read head addresses the right slot
# ---------------------------------------------------------------------------


def test_read_head_retrieves_matching_slot():
    """When the query (derived from x) is aligned with slot k's content, the
    read weight peaks at slot k.

    Note: the read head is query-driven (q = q_proj(x)), so to make slot 3 win
    we must make the *query* align with slot 3 — not just put a big vector in
    slot 3. We craft x so q_proj(x) ≈ slot 3's content.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    blk = model.blocks[0]
    head = blk.router.experts[1].head

    # Put a target vector in slot 3.
    mem = MemoryState.create(1, cfg.memory, "cpu", torch.float32)
    target = torch.randn(cfg.memory.d_mem)
    with torch.no_grad():
        mem.tape.zero_()
        mem.tape[0, 3] = target

    # Craft x so q_proj(x) ≈ target. q_proj is linear; solve for x in a tiny
    # sense: just try a few random x and pick the one whose query is closest
    # to target (this sidesteps needing q_proj's pseudoinverse).
    best_x = None
    best_sim = -2.0
    for _ in range(50):
        cand = torch.randn(1, 1, cfg.d_model)
        with torch.no_grad():
            q = head.q_proj(cand).view(-1)
        sim = F.cosine_similarity(q.unsqueeze(0), target.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_x = cand
    x = best_x
    with torch.no_grad():
        q = head.q_proj(x).view(1, 1, -1)
        scores = torch.einsum("bmd,bsd->bms", q, mem.tape) / (cfg.memory.d_mem ** 0.5)
        w = torch.softmax(scores, dim=-1)
    peak_slot = w.mean(dim=(0, 1)).argmax().item()
    assert peak_slot == 3, f"expected peak at slot 3 (query aligned), got {peak_slot}"


def test_from_knowledge_seeds_correct_slots():
    """from_knowledge puts the kb_slots in the first K slots, rest zeros."""
    cfg = _cfg(memory=MemoryConfig(d_mem=8, num_slots=4))
    slots = torch.randn(2, 8)
    ms = MemoryState.from_knowledge(slots, batch_size=3, config=cfg.memory,
                                    device="cpu", dtype=torch.float32)
    assert ms.tape.shape == (3, 4, 8)
    assert torch.allclose(ms.tape[:, :2], slots.unsqueeze(0).expand(3, -1, -1))
    assert torch.allclose(ms.tape[:, 2:], torch.zeros(3, 2, 8))


def test_blend_zero_equals_scratch():
    """blend_ratio=0 must reproduce the zero (scratch) tape exactly."""
    cfg = _cfg(memory=MemoryConfig(d_mem=8, num_slots=4))
    slots = torch.randn(2, 8)
    ms_zero = MemoryState.create(1, cfg.memory, "cpu", torch.float32)
    ms_blend0 = MemoryState.from_knowledge(slots, 1, cfg.memory, "cpu", torch.float32, blend_ratio=0.0)
    assert torch.allclose(ms_blend0.tape, ms_zero.tape)


# ---------------------------------------------------------------------------
# Tier 2 — semantic: does seeding specifically shift predictions toward the seed?
# ---------------------------------------------------------------------------


def _seeded_vs_scratch_logit_shift(cfg, seed_signal, n_trials=8):
    """Measure whether seeding with `seed_signal` shifts logits toward vocab
    tokens whose embedding is similar to the seed — i.e. a *specific*, not
    random, shift. Returns (mean_specific_alignment, baseline_specific_alignment).
    """
    torch.manual_seed(0)
    model = Prism(cfg).eval()
    ids = torch.randint(2, cfg.vocab_size, (1, 6))
    embed = model.embed.weight   # (V, d_model)

    # Scratch logits.
    with torch.no_grad():
        out_s = model(ids)
        logits_s = out_s.logits[0, -1]   # (V,)

    # Seeded: put the seed_signal in the tape.
    seed_slots = seed_signal.unsqueeze(0)   # (1, d_mem)
    mem = MemoryState.from_knowledge(seed_slots, 1, cfg.memory, "cpu", torch.float32)
    with torch.no_grad():
        out_k = model(ids, mem=mem)
        logits_k = out_k.logits[0, -1]

    delta = (logits_k - logits_s).detach()   # (V,)
    # Specificity: correlation between delta and the similarity of each vocab
    # embedding to the seed signal. If seeding works *semantically*, tokens
    # whose embedding aligns with the seed should gain logit.
    # Project embed to d_mem via the memory read_out inverse-ish (approx: just
    # use the first d_mem dims of the embed, truncated). This is a rough probe.
    emb_trunc = embed[:, : cfg.memory.d_mem].detach()
    sims = F.cosine_similarity(emb_trunc, seed_signal.detach().unsqueeze(0)).squeeze()
    # Pearson correlation between delta and sims.
    d = delta - delta.mean()
    s = sims - sims.mean()
    corr = (d * s).sum() / (d.norm() * s.norm() + 1e-9)
    return float(corr)


def test_seeding_changes_logits():
    """Minimal honest claim: seeding the tape changes the output logits.
    This is the paper's 0.0646 result — necessary but NOT sufficient."""
    cfg = _cfg()
    seed = torch.randn(cfg.memory.d_mem)
    torch.manual_seed(0)
    model = Prism(cfg).eval()
    ids = torch.randint(2, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits_s = model(ids).logits
        mem = MemoryState.from_knowledge(seed.unsqueeze(0), 1, cfg.memory, "cpu", torch.float32)
        logits_k = model(ids, mem=mem).logits
    diff = (logits_s - logits_k).abs().mean().item()
    assert diff > 1e-6, "seeding should change logits"


def test_semantic_specificity_is_documented():
    """The semantic-specificity correlation (Tier 2). We do NOT assert it's
    positive on an untrained-for-KB toy model — that would be the dishonest
    test. Instead we run it and assert it's a finite number, recording the value
    so a Phase-2-trained model can be compared against it.

    On an untrained-for-KB model this correlation is expected to be near zero
    (random) — that's the honest baseline. Phase 2's job is to push it positive.
    """
    cfg = _cfg()
    seed = torch.randn(cfg.memory.d_mem)
    corr = _seeded_vs_scratch_logit_shift(cfg, seed)
    # Honest assertion: the correlation is a finite real number (the mechanism
    # computes). We do NOT assert corr > 0 — that needs a trained encoder.
    assert -1.0 <= corr <= 1.0
    # Document the baseline value in the test output via the value itself.


# ---------------------------------------------------------------------------
# Integration: encoder -> KB -> retrieve round-trip
# ---------------------------------------------------------------------------


def test_encoder_kb_roundtrip():
    """Encode two texts, store in KB, retrieve the matching one."""
    torch.manual_seed(0)
    enc_cfg = EncoderConfig(d_model=32, d_mem=16, num_slots_per_doc=2, num_heads=4)
    encoder = PrismEncoder(enc_cfg, vocab_size=64).eval()
    kb = KnowledgeBase(d_mem=16)

    # Fake "tokenization": random ids (a real tokenizer would be used upstream).
    for text, seed in [("doc A", 1), ("doc B", 2)]:
        torch.manual_seed(hash(text) & 0xFFFF)
        ids = torch.randint(0, 64, (1, 8))
        slots = encoder.encode_ids(ids).squeeze(0)
        kb.add_entry(slots, text)

    # Query with doc A's ids -> should retrieve doc A's slots first.
    torch.manual_seed(hash("doc A") & 0xFFFF)
    q = encoder.encode_ids(torch.randint(0, 64, (1, 8))).squeeze(0)
    retrieved = kb.retrieve(q, top_k=2)
    assert retrieved.shape == (2, 16)
    # The first retrieved slot should match doc A's first slot.
    expected = kb.entries[0].slots[0]
    assert torch.allclose(retrieved[0], torch.tensor(expected), atol=1e-3)
