"""KnowledgeBase — vector store of d_mem slots (PRISM-KB Module 2).

A simple, dependency-free vector store. Each document is encoded by the
PrismEncoder into ``num_slots_per_doc`` slots of dimension ``d_mem``; we store
the flattened slots plus the original text/metadata. Retrieval is cosine
similarity in slot space, returning the top-k slots for seeding the tape.

For ≤100k documents a flat PyTorch tensor in CPU is enough (the math is just a
matmul). The API is designed so a FAISS/Milvus backend can slot in later for
>1M documents without changing callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class KBEntry:
    """One stored document: its slots + provenance."""

    text: str
    slots: list[list[float]]          # (num_slots_per_doc, d_mem) as nested lists
    metadata: dict = field(default_factory=dict)


class KnowledgeBase:
    """A vector store of encoded document slots.

    Attributes:
        d_mem: slot dimension (must match the model's MemoryConfig.d_mem).
        entries: list of KBEntry (text + slots + metadata).
        slot_matrix: (N * num_slots_per_doc, d_mem) stacked slots for retrieval.
    """

    def __init__(self, d_mem: int) -> None:
        self.d_mem = d_mem
        self.entries: list[KBEntry] = []
        self._matrix: torch.Tensor | None = None  # lazily built

    # --- mutation --------------------------------------------------------

    def add_entry(self, slots: torch.Tensor, text: str, metadata: dict | None = None) -> None:
        """Add one document's slots. ``slots`` is (num_slots_per_doc, d_mem)."""
        if slots.dim() != 2 or slots.shape[1] != self.d_mem:
            raise ValueError(f"slots must be (k, {self.d_mem}), got {tuple(slots.shape)}")
        self.entries.append(KBEntry(
            text=text,
            slots=slots.detach().cpu().tolist(),
            metadata=metadata or {},
        ))
        self._matrix = None  # invalidate cache

    def add_text(self, text: str, encoder, tokenizer, device, metadata: dict | None = None) -> None:
        """Encode and add a single text using ``encoder``."""
        slots = encoder.encode_text(text, tokenizer, device).squeeze(0)   # (k, d_mem)
        self.add_entry(slots, text, metadata)

    def add_texts(self, texts: list[str], encoder, tokenizer, device, metadata: list[dict] | None = None) -> None:
        """Batch-encode and add many texts."""
        if metadata is None:
            metadata = [{} for _ in texts]
        slots_batch = encoder.encode_texts(texts, tokenizer, device)      # (B, k, d_mem)
        for i, text in enumerate(texts):
            self.add_entry(slots_batch[i], text, metadata[i])

    # --- retrieval -------------------------------------------------------

    def _build_matrix(self) -> torch.Tensor:
        if not self.entries:
            self._matrix = torch.zeros(0, self.d_mem)
            return self._matrix
        flat = []
        for e in self.entries:
            flat.extend(e.slots)
        self._matrix = torch.tensor(flat, dtype=torch.float32)
        return self._matrix

    @property
    def matrix(self) -> torch.Tensor:
        if self._matrix is None:
            self._build_matrix()
        return self._matrix

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, query_slots: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        """Return the top-k slots (as a (top_k, d_mem) tensor) most similar to the query.

        The query is a single slot or set of slots (we pool by mean first).
        Cosine similarity in slot space.
        """
        mat = self.matrix
        if mat.shape[0] == 0:
            return torch.zeros(0, self.d_mem)

        # Pool the query to a single (d_mem,) vector.
        if query_slots.dim() == 1:
            q = query_slots
        else:
            q = query_slots.mean(dim=0)
        q = q.to(mat.dtype).to(mat.device)

        # Cosine similarity.
        sims = F.cosine_similarity(q.unsqueeze(0), mat, dim=-1)   # (N*k,)
        k = min(top_k, sims.shape[0])
        top_idx = sims.topk(k).indices
        return mat[top_idx]

    # --- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a JSON file (slots as nested lists). Portable, no pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"d_mem": self.d_mem, "entries": [asdict(e) for e in self.entries]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        kb = cls(d_mem=payload["d_mem"])
        for e in payload["entries"]:
            kb.entries.append(KBEntry(text=e["text"], slots=e["slots"], metadata=e.get("metadata", {})))
        return kb

    def summary(self) -> str:
        return f"KnowledgeBase(d_mem={self.d_mem}, {len(self.entries)} docs, {self.matrix.shape[0]} slots)"
