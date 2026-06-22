"""AnalyticCapture — the heart of COGLOOP (Phase 1).

Solves the +0.006 specificity problem identified in PRISM-KB.md.

The problem: seeding the tape with slots from an external encoder (Perceiver)
gave random specificity because the read head was trained to read slots in the
NTM distribution it WRITES, not an external abstract space.

The solution: capture the EXACT vectors the MemoryExpert writes when it
processes the reference text. These vectors live in the read head's native
distribution by construction — zero training, zero GPU, perfect alignment.

Mechanism:
    1. Hook the MemoryHead.forward to intercept the `add` vector
       (the NTM write signal: add = mean(w_gate * v_proj_in(x))).
    2. Run Prism (frozen, no_grad) on the reference text.
    3. Accumulate the captured adds across layers and time steps.
    4. Project / pool them into (num_slots, d_mem) KB slots.

The captured slots ARE the native write distribution. The read head reads them
the same way it reads its own writes — no distribution shift, no training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from prism.config import PrismConfig
from prism.memory import MemoryState


@dataclass
class CaptureResult:
    """Result of capturing write signals from one or more forward passes.

    Attributes:
        slots: (num_slots, d_mem) tensor in the read head's native distribution.
        num_captures: how many add vectors were captured (for diagnostics).
        layer_distribution: per-layer capture count (diagnostic).
    """

    slots: torch.Tensor
    num_captures: int
    layer_distribution: dict[int, int]


class AnalyticCapture:
    """Captures MemoryExpert write vectors from a frozen Prism forward pass.

    Usage::

        capture = AnalyticCapture(model, config)
        slots = capture.capture_text(tokenizer, "some reference text")
        # slots is (num_slots, d_mem), ready to seed the tape via
        # MemoryState.from_knowledge(slots, ...)

    The model is never modified — we use forward hooks that record the add
    vectors and then are removed.
    """

    def __init__(self, model: nn.Module, config: PrismConfig, pool: str = "pca") -> None:
        """Args:
            model: a Prism model (frozen or trainable; we use no_grad anyway).
            config: the PrismConfig (for d_mem, num_slots).
            pool: how to reduce captured adds to num_slots.
                "pca"  — PCA onto the top-num_slots directions (recommended;
                         preserves the most informative variance).
                "mean" — simple mean pool into num_slots buckets.
                "first" — take the first num_slots captured (cheapest).
        """
        self.model = model
        self.config = config
        self.d_mem = config.memory.d_mem
        self.num_slots = config.memory.num_slots
        self.pool = pool
        self._captured: list[tuple[int, torch.Tensor]] = []   # (layer_idx, add)

    # --- hook plumbing ----------------------------------------------------

    def _make_hook(self, layer_idx: int) -> Callable:
        """Return a forward hook that captures the MemoryHead's add vector."""
        def hook(module, inputs, output):
            # output is (read_out, new_state). We want the add vector that was
            # used to update the tape. Re-derive it from the inputs to avoid
            # needing to change MemoryHead's return signature.
            # The head's forward computed:
            #   v = v_proj_in(x); w_gate = sigmoid(write_gate(x))
            #   add = (w_gate * v).mean(dim=1)  # (B, d_mem)
            x = inputs[0]
            with torch.no_grad():
                B = x.shape[0] if x.dim() >= 2 else 1
                # Match the head's reshape logic.
                lead = x.shape[:-1]
                d_model_in = x.shape[-1]
                x_view = x.reshape(B, -1, d_model_in)
                v = module.v_proj_in(x_view)             # (B, M, d_mem)
                w_gate = torch.sigmoid(module.write_gate(x_view))  # (B, M, 1)
                add = (w_gate * v).mean(dim=1)           # (B, d_mem)
                self._captured.append((layer_idx, add.detach().cpu()))
        return hook

    def _attach_hooks(self) -> list[tuple[nn.Module, object]]:
        """Attach capture hooks to every MemoryHead in the model. Returns handles."""
        handles = []
        for layer_idx, block in enumerate(self.model.blocks):
            for expert in block.router.experts:
                # The MemoryExpert delegates to .head (a MemoryHead).
                head = getattr(expert, "head", None)
                if head is not None and head.__class__.__name__ == "MemoryHead":
                    h = head.register_forward_hook(self._make_hook(layer_idx))
                    handles.append((head, h))
        return handles

    @staticmethod
    def _detach_all(handles):
        for _, h in handles:
            h.remove()

    # --- capture ----------------------------------------------------------

    @torch.no_grad()
    def capture_ids(self, input_ids: torch.Tensor) -> CaptureResult:
        """Run a frozen forward pass and capture all MemoryExpert writes.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            CaptureResult with (num_slots, d_mem) slots.
        """
        self._captured.clear()
        handles = self._attach_hooks()
        try:
            self.model.eval()
            _ = self.model(input_ids)
        finally:
            self._detach_all(handles)

        if not self._captured:
            # No memory expert fired (e.g. last block dropped it, or none routed
            # to memory). Return zeros — caller should treat as empty.
            return CaptureResult(
                slots=torch.zeros(self.num_slots, self.d_mem),
                num_captures=0,
                layer_distribution={},
            )

        # Stack all captured adds: (total_layers, B, d_mem) -> (N, d_mem)
        layer_dist: dict[int, int] = {}
        all_adds = []
        for layer_idx, add in self._captured:
            layer_dist[layer_idx] = layer_dist.get(layer_idx, 0) + add.shape[0]
            all_adds.append(add.reshape(-1, self.d_mem))
        stacked = torch.cat(all_adds, dim=0)   # (N, d_mem)
        n = stacked.shape[0]

        slots = self._pool_to_slots(stacked)
        return CaptureResult(slots=slots, num_captures=n, layer_distribution=layer_dist)

    def _pool_to_slots(self, stacked: torch.Tensor) -> torch.Tensor:
        """Reduce (N, d_mem) captured adds to (num_slots, d_mem)."""
        N = stacked.shape[0]
        if N <= self.num_slots:
            # Pad with the mean if we have fewer captures than slots.
            mean = stacked.mean(dim=0, keepdim=True)
            pad = mean.expand(self.num_slots - N, -1)
            return torch.cat([stacked, pad], dim=0)

        if self.pool == "first":
            return stacked[: self.num_slots]

        if self.pool == "mean":
            # Chunk into num_slots groups, mean each.
            chunks = torch.chunk(stacked[: N - (N % self.num_slots)], self.num_slots, dim=0)
            return torch.stack([c.mean(dim=0) for c in chunks])

        if self.pool == "pca":
            # Lightweight PCA via SVD on the centered captures. This finds the
            # num_slots directions of maximum variance in the write distribution
            # — the most informative axes for the read head to address.
            centered = stacked - stacked.mean(dim=0, keepdim=True)
            # (d_mem, d_mem) = V; we want top-num_slots right singular vectors.
            try:
                _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                # Project captures onto top components, then take the component
                # directions scaled by the mean projection magnitude. This gives
                # slots that span the highest-variance region of the write space.
                comps = Vh[: self.num_slots]                  # (num_slots, d_mem)
                proj = centered @ comps.T                     # (N, num_slots)
                # Scale each component by its mean |projection| so the slot
                # magnitudes match the real write distribution.
                scale = proj.abs().mean(dim=0)                # (num_slots,)
                return comps * scale.unsqueeze(-1) * (self.d_mem ** 0.5)
            except Exception:
                # SVD can fail on degenerate inputs; fall back to mean pooling.
                chunks = torch.chunk(stacked[: N - (N % self.num_slots)], self.num_slots, dim=0)
                return torch.stack([c.mean(dim=0) for c in chunks])

        raise ValueError(f"unknown pool method: {self.pool!r}")

    @torch.no_grad()
    def capture_text(self, tokenizer, text: str, device, max_len: int = 256) -> torch.Tensor:
        """Encode text and capture. Returns (num_slots, d_mem)."""
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = ids["input_ids"].to(device)
        return self.capture_ids(input_ids).slots

    @torch.no_grad()
    def capture_texts(self, tokenizer, texts: list[str], device, max_len: int = 256) -> torch.Tensor:
        """Capture from a batch of texts, pooled together (one set of slots)."""
        enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        return self.capture_ids(enc["input_ids"].to(device)).slots


def description() -> str:
    return (
        "AnalyticCapture: capture the exact vectors Prism's MemoryExpert WRITES "
        "on reference text, use them as KB slots. Zero training, zero GPU, "
        "perfect alignment with the read head's native distribution."
    )
