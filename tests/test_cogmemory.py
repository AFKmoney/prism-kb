"""Tests for the double-layer memory system (COGLOOP Section 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from prism.cogmemory import (
    CogMemory,
    ConsolidationPolicy,
    Episode,
    LongTermStore,
    WorkingMemory,
)


def _episode(text: str, importance: float, d_mem: int = 8, n_slots: int = 2) -> Episode:
    return Episode(
        text=text,
        slots=torch.randn(n_slots, d_mem).tolist(),
        importance=importance,
    )


def test_working_memory_respects_capacity():
    wm = WorkingMemory(capacity=4)
    for i in range(10):
        wm.add(_episode(f"e{i}", importance=0.1 * i))
    assert len(wm) == 4
    # The dropped ones should be the lowest-importance (0.0, 0.1, ...).
    importances = sorted(e.importance for e in wm.episodes)
    assert len(importances) == 4
    assert abs(importances[0] - 0.6) < 1e-6
    assert abs(importances[-1] - 0.9) < 1e-6


def test_working_memory_top_k():
    wm = WorkingMemory(capacity=10)
    for i in range(5):
        wm.add(_episode(f"e{i}", importance=i / 10))
    top = wm.top_k_for_consolidation(2)
    assert [e.importance for e in top] == [0.4, 0.3]


def test_long_term_store_persists_across_sessions():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "kb.json"
        lt = LongTermStore(d_mem=8, path=path, max_entries=100)
        lt.consolidate([_episode("hello", 0.9)])
        assert len(lt) == 1
        # Reload in a "new session".
        lt2 = LongTermStore(d_mem=8, path=path)
        assert len(lt2) == 1
        assert lt2.kb.entries[0].text == "hello"


def test_long_term_store_evicts_when_full():
    with tempfile.TemporaryDirectory() as td:
        lt = LongTermStore(d_mem=8, path=Path(td) / "kb.json", max_entries=3)
        lt.consolidate([_episode(f"e{i}", importance=i / 10) for i in range(3)])
        assert len(lt) == 3
        # Adding one more should evict the lowest-importance (e0, importance 0.0).
        lt.consolidate([_episode("new", importance=0.5)])
        assert len(lt) == 3
        texts = [e.text for e in lt.kb.entries]
        assert "e0" not in texts
        assert "new" in texts


def test_cogmemory_consolidation_threshold():
    with tempfile.TemporaryDirectory() as td:
        policy = ConsolidationPolicy(importance_threshold=0.5, consolidate_every_n_episodes=3)
        cm = CogMemory(d_mem=8, long_term_path=Path(td) / "kb.json", policy=policy)
        # Add low-importance episodes — should NOT consolidate.
        cm.observe(_episode("low1", 0.1))
        cm.observe(_episode("low2", 0.2))
        cm.observe(_episode("low3", 0.3))   # triggers consolidation check
        assert len(cm.long_term) == 0   # none above threshold
        # Add a high-importance episode.
        cm.observe(_episode("low4", 0.1))
        cm.observe(_episode("low5", 0.1))
        cm.observe(_episode("high", 0.9))   # triggers; high passes threshold
        assert len(cm.long_term) == 1


def test_remember_explicitly_bypasses_threshold():
    with tempfile.TemporaryDirectory() as td:
        cm = CogMemory(d_mem=8, long_term_path=Path(td) / "kb.json")
        cm.remember_explicitly("important fact", torch.randn(2, 8), importance=1.0)
        assert len(cm.long_term) == 1
        assert cm.long_term.kb.entries[0].metadata["source"] == "explicit"


def test_cogmemory_summary_runs():
    with tempfile.TemporaryDirectory() as td:
        cm = CogMemory(d_mem=8, long_term_path=Path(td) / "kb.json")
        s = cm.summary()
        assert "working=" in s and "LongTermStore" in s
