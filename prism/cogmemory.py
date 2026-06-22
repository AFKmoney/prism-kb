"""CogMemory — the double-layer memory system (COGLOOP Section 3).

Two layers, mirroring human cognition:

  1. WORKING MEMORY (ephemeral) — the in-RAM tape of the current session.
     Cheap, fast, lost when the process exits. This is what Reflect operates
     on during a single question.

  2. LONG-TERM MEMORY (persistent) — a KnowledgeBase on disk. Survives across
     sessions. Grows over time as the user interacts with PRISM.

Between them: CONSOLIDATION. Most working-memory episodes are forgotten
(human: you don't remember every word you read today). A few — flagged by
     importance or explicit "remember this" — are consolidated into long-term.
Rarely (the "deep sleep" analogue), consolidation also includes a micro
     weight-tune (Phase-2-style) to bake recurring patterns into the model.

    Working Memory  ──[consolidate: important only]──▶  Long-Term KB (disk)
                         │
                         └─[rare, on signal]──▶  Weight Tune (micro CPU)

The LongTermStore wraps a KnowledgeBase with:
  * importance scoring (how salient was an episode?)
  * capacity management (evict least-important when full)
  * cross-session persistence
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from prism.kb import KnowledgeBase


@dataclass
class Episode:
    """One working-memory episode awaiting possible consolidation.

    Attributes:
        text: the human-readable content (question, answer, fact).
        slots: (num_slots, d_mem) encoded slots.
        importance: 0..1 salience score (higher = more likely to consolidate).
        timestamp: wall-clock for recency.
        metadata: free-form dict (source, tags, ...).
    """

    text: str
    slots: list[list[float]]
    importance: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class WorkingMemory:
    """Ephemeral in-RAM buffer of recent episodes.

    Acts as a ring buffer: when full, the lowest-importance episode is the
    first candidate for either consolidation or forgetting.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self.episodes: list[Episode] = []

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            # Drop the single least-important (pure forgetting by default).
            worst = min(range(len(self.episodes)), key=lambda i: self.episodes[i].importance)
            self.episodes.pop(worst)

    def __len__(self) -> int:
        return len(self.episodes)

    def top_k_for_consolidation(self, k: int) -> list[Episode]:
        """Return the k most-important episodes (candidates for long-term)."""
        return sorted(self.episodes, key=lambda e: e.importance, reverse=True)[:k]

    def clear(self) -> None:
        self.episodes.clear()


class LongTermStore:
    """Persistent KB-backed long-term memory.

    Wraps a KnowledgeBase with importance-aware addition and capacity
    management. Persists to a JSON file on disk; reloadable across sessions.
    """

    def __init__(self, d_mem: int, path: str | Path, max_entries: int = 10_000) -> None:
        self.path = Path(path)
        self.max_entries = max_entries
        if self.path.exists():
            self.kb = KnowledgeBase.load(self.path)
            # Align d_mem to the loaded KB.
            self.d_mem = self.kb.d_mem
        else:
            self.d_mem = d_mem
            self.kb = KnowledgeBase(d_mem=d_mem)

    def consolidate(self, episodes: list[Episode]) -> int:
        """Add episodes to long-term memory. Returns how many were stored.

        Respects max_entries: if full, evicts the lowest-importance existing
        entry to make room (LRU-by-importance).
        """
        added = 0
        for ep in episodes:
            if len(self.kb) >= self.max_entries:
                self._evict_one()
            slots_t = torch.tensor(ep.slots, dtype=torch.float32)
            self.kb.add_entry(slots_t, ep.text, {"importance": ep.importance, **ep.metadata})
            added += 1
        self.save()
        return added

    def _evict_one(self) -> None:
        """Evict the lowest-importance entry from the KB."""
        if not self.kb.entries:
            return
        worst = min(
            range(len(self.kb.entries)),
            key=lambda i: self.kb.entries[i].metadata.get("importance", 0.0),
        )
        self.kb.entries.pop(worst)
        self.kb._matrix = None

    def save(self) -> None:
        self.kb.save(self.path)

    def __len__(self) -> int:
        return len(self.kb)

    def summary(self) -> str:
        return f"LongTermStore({self.kb.summary()}, cap={self.max_entries}, path={self.path})"


@dataclass
class ConsolidationPolicy:
    """Decides what gets consolidated and when.

    Defaults mimic human cognition: most episodes forgotten, only the
    salient ones consolidated. Weight-tuning is rare and opt-in.
    """

    importance_threshold: float = 0.5
    """Only episodes with importance >= this get consolidated to long-term."""

    consolidate_every_n_episodes: int = 16
    """Trigger a consolidation pass after this many working-memory additions."""

    weight_tune_signal: str = "explicit"
    """When to do a (rare) micro weight-tune. 'explicit' = only on user signal."""


class CogMemory:
    """The double-layer memory coordinator.

    Ties WorkingMemory + LongTermStore + a consolidation policy together.
    The CogLoop (in cogloop.py) calls into this after each interaction.
    """

    def __init__(
        self,
        d_mem: int,
        long_term_path: str | Path,
        working_capacity: int = 64,
        long_term_capacity: int = 10_000,
        policy: ConsolidationPolicy | None = None,
    ) -> None:
        self.working = WorkingMemory(capacity=working_capacity)
        self.long_term = LongTermStore(d_mem=d_mem, path=long_term_path, max_entries=long_term_capacity)
        self.policy = policy or ConsolidationPolicy()
        self._since_last_consolidation = 0

    def observe(self, episode: Episode) -> None:
        """Record an episode in working memory; maybe trigger consolidation."""
        self.working.add(episode)
        self._since_last_consolidation += 1
        if self._since_last_consolidation >= self.policy.consolidate_every_n_episodes:
            self.run_consolidation()

    def run_consolidation(self) -> int:
        """Consolidate salient working-memory episodes into long-term.

        Returns the number of episodes consolidated.
        """
        candidates = [
            e for e in self.working.top_k_for_consolidation(self.working.capacity)
            if e.importance >= self.policy.importance_threshold
        ]
        if not candidates:
            self._since_last_consolidation = 0
            return 0
        added = self.long_term.consolidate(candidates)
        # Remove consolidated episodes from working memory (they're now persistent).
        consolidated_texts = {id(e) for e in candidates}
        self.working.episodes = [e for e in self.working.episodes if id(e) not in consolidated_texts]
        self._since_last_consolidation = 0
        return added

    def remember_explicitly(self, text: str, slots: torch.Tensor, importance: float = 1.0) -> None:
        """User-facing 'remember this for real' — immediate consolidation."""
        ep = Episode(
            text=text,
            slots=slots.detach().cpu().tolist(),
            importance=importance,
            metadata={"source": "explicit"},
        )
        self.long_term.consolidate([ep])

    def summary(self) -> str:
        return (
            f"CogMemory(working={len(self.working)}/{self.working.capacity}, "
            f"{self.long_term.summary()})"
        )


def description() -> str:
    return (
        "CogMemory: double-layer memory (ephemeral working + persistent long-term) "
        "with importance-based consolidation. Mirrors human cognition: most "
        "episodes forgotten, salient ones consolidated, rare weight-tuning."
    )
