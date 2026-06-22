"""End-to-end smoke test for the full CogLoop.

PERCEIVE -> REFLECT -> RESPOND -> CONSOLIDATE, then verify memory persists
across a "session restart" (new CogLoop instance pointing at the same disk KB).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from prism.cogloop import CogLoop
from prism.config import MemoryConfig, PrismConfig
from prism.model import Prism


def _cfg(**kw) -> PrismConfig:
    base = dict(
        vocab_size=256, d_model=64, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=32, num_slots=8),
    )
    base.update(kw)
    return PrismConfig(**base)


class _StubTokenizer:
    """A minimal byte-level tokenizer stub so the test has no HF dependency.

    Maps characters to ids 1..255 (0 reserved for pad/eos).
    """

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
        ids = [min(255, max(1, ord(c))) for c in text[:max_length]]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids if i > 0)


def test_cogloop_answer_runs_end_to_end():
    """The full loop: answer a question, no crash, returns a CogAnswer."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    with tempfile.TemporaryDirectory() as td:
        loop = CogLoop(model, cfg, _StubTokenizer(), long_term_path=str(Path(td) / "kb.json"))
        ans = loop.answer("hello", max_new_tokens=4, importance=0.0)
        assert isinstance(ans.text, str)
        assert ans.passes_used >= 1
        assert "working=" in ans.memory_summary


def test_cogloop_remember_persists_across_sessions():
    """remember() writes to disk; a new CogLoop sees it."""
    torch.manual_seed(0)
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "kb.json")
        m1 = Prism(cfg).eval()
        loop1 = CogLoop(m1, cfg, _StubTokenizer(), long_term_path=path)
        assert len(loop1.memory.long_term) == 0
        loop1.remember("the sky is blue", importance=1.0)
        assert len(loop1.memory.long_term) == 1

        # "New session": fresh model + loop, same disk KB.
        m2 = Prism(cfg).eval()
        loop2 = CogLoop(m2, cfg, _StubTokenizer(), long_term_path=path)
        assert len(loop2.memory.long_term) == 1
        assert loop2.memory.long_term.kb.entries[0].text == "the sky is blue"


def test_cogloop_two_questions_accumulate_working_memory():
    """Successive answers accumulate episodes in working memory."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    with tempfile.TemporaryDirectory() as td:
        loop = CogLoop(model, cfg, _StubTokenizer(), long_term_path=str(Path(td) / "kb.json"))
        loop.answer("q1", max_new_tokens=2)
        assert len(loop.memory.working) == 1
        loop.answer("q2", max_new_tokens=2)
        assert len(loop.memory.working) == 2


def test_cogloop_reflection_trace_is_reported():
    """The answer reports reflection diagnostics (passes, convergence)."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = Prism(cfg).eval()
    with tempfile.TemporaryDirectory() as td:
        loop = CogLoop(model, cfg, _StubTokenizer(), long_term_path=str(Path(td) / "kb.json"))
        ans = loop.answer("think about this", max_new_tokens=2)
        assert ans.passes_used >= 1
        assert isinstance(ans.converged, bool)
