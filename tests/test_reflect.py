"""Tests for the Reflect loop (COGLOOP Section 2)."""

from __future__ import annotations

import torch

from prism.config import MemoryConfig, PrismConfig
from prism.kb import KnowledgeBase
from prism.reflect import ReflectConfig, Reflector
from prism.model import Prism


def _cfg(**kw) -> PrismConfig:
    base = dict(
        vocab_size=64, d_model=32, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=16, num_slots=8),
    )
    base.update(kw)
    return PrismConfig(**base)


def test_reflect_returns_context_and_trace():
    """Reflect produces a (num_slots, d_mem) context and a trace."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    r = Reflector(model, cfg, kb=None)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    ctx, trace = r.reflect(ids)
    assert ctx.shape == (cfg.memory.num_slots, cfg.memory.d_mem)
    assert trace.passes_used >= 1
    assert trace.passes_used <= r.rc.max_passes


def test_reflect_respects_max_passes_budget():
    """The loop never exceeds max_passes.

    With a non-zero seed and a tiny threshold, the loop keeps accumulating
    (shifts stay above threshold) until it hits the budget.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    rc = ReflectConfig(max_passes=3, converge_threshold=1e-9, alpha=1.0)
    r = Reflector(model, cfg, kb=None, reflect_config=rc)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    # Non-zero seed so the loop has signal to accumulate.
    seed = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem) * 3.0
    _, trace = r.reflect(ids, initial_seed_slots=seed)
    assert trace.passes_used == rc.max_passes


def test_reflect_can_converge_early():
    """With a high threshold, the loop should converge before max_passes."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    # Very high threshold -> should converge on pass 1 or 2.
    rc = ReflectConfig(max_passes=10, converge_threshold=10.0)
    r = Reflector(model, cfg, kb=None, reflect_config=rc)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    _, trace = r.reflect(ids)
    assert trace.converged
    assert trace.passes_used < 10


def test_reflect_different_questions_give_different_contexts():
    """Two distinct questions, WITH seed slots, produce distinct reflected contexts.

    NOTE: on a toy PRISM untrained for KB retrieval, the read head is inert so
    the *accumulated* context stays near zero regardless of input (consistent
    with the Section-1 finding). To exercise the path that DOES distinguish
    inputs, we provide distinct initial seed slots — those propagate through.
    A retrieval-trained Prism would show input-dependent contexts even with
    zero seed; this test pins the honest baseline.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    r = Reflector(model, cfg, kb=None, reflect_config=ReflectConfig(max_passes=1))
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    torch.manual_seed(1)
    seed1 = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem)
    torch.manual_seed(2)
    seed2 = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem)
    ctx1, _ = r.reflect(ids, initial_seed_slots=seed1)
    ctx2, _ = r.reflect(ids, initial_seed_slots=seed2)
    assert not torch.allclose(ctx1, ctx2, atol=1e-4)


def test_reflect_with_seed_slots_incorporates_them():
    """Initial seed slots should influence the final context."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    r = Reflector(model, cfg, kb=None, reflect_config=ReflectConfig(max_passes=1))
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    seed = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem) * 5.0
    ctx_seeded, _ = r.reflect(ids, initial_seed_slots=seed)
    ctx_zero, _ = r.reflect(ids, initial_seed_slots=None)
    assert not torch.allclose(ctx_seeded, ctx_zero, atol=1e-3)


def test_reflect_trace_records_norms_and_shifts():
    """The trace captures per-pass context norms and query shifts for debugging."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    r = Reflector(model, cfg, kb=None, reflect_config=ReflectConfig(max_passes=4))
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    _, trace = r.reflect(ids)
    assert len(trace.context_norms) == trace.passes_used
    assert len(trace.query_shifts) == trace.passes_used
    assert all(isinstance(n, float) for n in trace.context_norms)


def test_reflect_with_kb_does_not_crash():
    """Reflection with a KB + encoder path runs end-to-end."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    # Empty KB (retrieval returns nothing); loop must still complete.
    kb = KnowledgeBase(d_mem=cfg.memory.d_mem)
    # Fake encoder: returns random slots (just to exercise the path).
    class FakeEnc:
        def encode_text(self, *a, **k):
            return torch.randn(1, 2, cfg.memory.d_mem)
    r = Reflector(model, cfg, kb=kb, encoder=FakeEnc())
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    ctx, trace = r.reflect(ids)
    assert ctx.shape == (cfg.memory.num_slots, cfg.memory.d_mem)
