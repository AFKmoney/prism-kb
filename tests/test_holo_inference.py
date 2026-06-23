"""Tests for HoloInference (end-to-end knowledge-driven generation)."""

from __future__ import annotations

import torch

from prism.config import MemoryConfig, PrismConfig
from prism.holo import HoloEncoder
from prism.holo_inference import HoloAnswer, HoloInference
from prism.model import Prism


class _StubTokenizer:
    """Byte-level stub: maps chars to ids 1..255 (0 = pad/eos)."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        ids = [min(255, max(1, ord(c))) for c in text[:max_length]]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids if i > 0)


def _cfg(**kw):
    base = dict(
        vocab_size=256, d_model=64, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=32, num_slots=64),  # D = 2048
        holo_mode=True,
    )
    base.update(kw)
    return PrismConfig(**base)


def test_generate_with_facts_runs_end_to_end():
    """Bind facts, generate — no crash, returns HoloAnswer."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg)
    encoder = HoloEncoder(d_model=cfg.d_model, D=cfg.memory.num_slots * cfg.memory.d_mem)
    inf = HoloInference(model, encoder, cfg, _StubTokenizer())

    ans = inf.generate_with_facts(
        prompt="hello",
        facts=[("capital of France", "Paris"), ("speed of light", "3e8")],
        max_new_tokens=4,
    )
    assert isinstance(ans, HoloAnswer)
    assert isinstance(ans.text, str)
    assert "facts=" in ans.tape_summary or "D=" in ans.tape_summary


def test_generate_with_facts_empty_facts():
    """Empty facts list = generate from scratch (tape starts empty)."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg)
    encoder = HoloEncoder(d_model=cfg.d_model, D=cfg.memory.num_slots * cfg.memory.d_mem)
    inf = HoloInference(model, encoder, cfg, _StubTokenizer())

    ans = inf.generate_with_facts(prompt="hello", facts=[], max_new_tokens=2)
    assert isinstance(ans.text, str)


def test_retrieval_sim_is_finite():
    """The retrieved_sim diagnostic is a finite float."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg)
    encoder = HoloEncoder(d_model=cfg.d_model, D=cfg.memory.num_slots * cfg.memory.d_mem)
    inf = HoloInference(model, encoder, cfg, _StubTokenizer())

    ans = inf.generate_with_facts(
        prompt="q", facts=[("k1", "v1"), ("k2", "v2")], max_new_tokens=2,
    )
    assert -1.0 <= ans.retrieved_sim <= 1.0
