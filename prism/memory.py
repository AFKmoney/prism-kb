"""Shared Memory Bus.

A single memory tape of shape ``(num_slots, d_mem)`` flows through every block
and every time step. Memory experts read from and write to this tape via soft
attention (read) and NTM-style gated erase+add (write).

This is the *Global Workspace*: it is how heterogeneous experts (neural,
memory, symbolic) communicate across layers and across time.

Design notes
------------
* The tape is **initialized fresh each forward pass** (not learned content).
  Its content is shaped entirely by writes during the forward pass. This keeps
  it purely a working memory, not a parameter bank.
* Reads use content-based addressing (softmax over Q·Kᵀ).
* Writes use erase+add (Graves et al. NTM, 2014) which is differentiable and
  keeps the tape bounded.
* The tape is mutated *in place across blocks* within a single time step's
  column of blocks, and *across time steps* via the running carry state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prism.config import MemoryConfig


@dataclass
class MemoryState:
    """The shared memory tape plus bookkeeping for regularizers.

    ``tape`` has shape ``(B, num_slots, d_mem)``. It is the *single* tape shared
    across all blocks and time steps of a forward pass.
    """

    tape: torch.Tensor
    """Shape (B, num_slots, d_mem)."""

    read_entropy: torch.Tensor
    """Scalar accumulator for the read-distribution entropy regularizer."""

    @classmethod
    def create(cls, batch_size: int, config: MemoryConfig, device, dtype) -> "MemoryState":
        # The tape is a *working memory*: it starts empty (zeros) and is shaped
        # entirely by writes during the forward pass. Starting from zeros (not
        # random noise) makes inference deterministic and is semantically
        # correct — the model has seen nothing yet.
        tape = torch.zeros(
            batch_size, config.num_slots, config.d_mem, device=device, dtype=dtype
        )
        read_entropy = torch.zeros((), device=device, dtype=dtype)
        return cls(tape=tape, read_entropy=read_entropy)

    @classmethod
    def empty(cls) -> "MemoryState":
        """A no-op state, used by experts that don't touch memory."""
        return cls(tape=torch.empty(0), read_entropy=torch.zeros(()))

    @classmethod
    def from_knowledge(
        cls,
        kb_slots: torch.Tensor,
        batch_size: int,
        config: MemoryConfig,
        device,
        dtype,
        blend_ratio: float = 1.0,
    ) -> "MemoryState":
        """Initialize the tape from external knowledge-base slots (the KB mechanism).

        Instead of starting at zeros, the first ``K`` slots of the tape are
        seeded with encoded content from a dataset. The rest stay at zero
        (preserving working-memory capacity). This is what activates PRISM-KB:
        the MemoryExpert's read head then retrieves the seeded content via its
        content-addressable soft attention, with no weight update.

        Args:
            kb_slots: (K, d_mem) or (B, K, d_mem). External encoded knowledge.
            batch_size: B.
            config: the MemoryConfig (gives num_slots, d_mem).
            device, dtype: target placement.
            blend_ratio: 0.0 = pure zeros (scratch), 1.0 = full seed, in-between
                = soft injection. Keeps the operation differentiable so the
                encoder can be trained end-to-end later (Phase 2).

        Returns:
            A MemoryState whose tape is (B, num_slots, d_mem), seeded.
        """
        S = config.num_slots
        d = config.d_mem

        if kb_slots.dim() == 2:
            # (K, d_mem) -> broadcast to (B, K, d_mem)
            kb_slots = kb_slots.unsqueeze(0).expand(batch_size, -1, -1)
        elif kb_slots.dim() == 3:
            assert kb_slots.shape[0] == batch_size, (
                f"kb_slots batch {kb_slots.shape[0]} != batch_size {batch_size}"
            )
        else:
            raise ValueError(f"kb_slots must be 2D or 3D, got {kb_slots.dim()}D")

        K = kb_slots.shape[1]
        if K > S:
            raise ValueError(
                f"kb_slots has {K} slots but config.num_slots is {S}; "
                f"retrieve top-{S} before seeding."
            )

        # Build the seeded tape: [kb_slots | zeros for remaining working memory].
        full = torch.zeros(batch_size, S, d, device=device, dtype=dtype)
        if K > 0:
            full[:, :K, :] = kb_slots.to(device=device, dtype=dtype)
        # Blend toward zeros (so blend_ratio=0 == scratch mode).
        full = blend_ratio * full

        read_entropy = torch.zeros((), device=device, dtype=dtype)
        return cls(tape=full, read_entropy=read_entropy)


class MemoryHead(nn.Module):
    """A read+write head over the shared memory bus.

    Used by the Memory expert. Given a query vector, it reads from the tape
    (soft attention over slots) and writes an update (gated erase+add).
    """

    def __init__(self, d_model: int, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.d_mem = config.d_mem
        self.num_slots = config.num_slots

        # Query / Key / Value projections from the input feature.
        self.q_proj = nn.Linear(d_model, config.d_mem, bias=False)
        # Keys are derived from the tape directly (no projection) so that
        # addressing stays content-based in the tape's native space.
        self.v_proj_in = nn.Linear(d_model, config.d_mem, bias=False)

        # Write gate (how much to write) + erase gate (how much to forget).
        self.write_gate = nn.Linear(d_model, 1, bias=True)
        self.erase_gate = nn.Linear(d_model, config.d_mem, bias=True)

        # Project the read-out back into d_model for the residual stream.
        self.read_out = nn.Linear(config.d_mem, d_model, bias=False)

    def forward(self, x: torch.Tensor, state: MemoryState) -> tuple[torch.Tensor, MemoryState]:
        """Read from the tape and return (read_out, updated_state).

        Args:
            x: shape (..., d_model) — features driving the head. Leading dims
                are batch and (optionally) time. The *first* leading dim is the
                batch dim and must match ``state.tape.shape[0]``.
            state: current memory state. ``tape`` is (B, S, d_mem).

        Returns:
            (read_out, new_state) where read_out has shape (..., d_model) and
            ``new_state.tape`` is (B, S, d_mem).
        """
        lead = x.shape[:-1]
        d_model_in = x.shape[-1]
        B = state.tape.shape[0]
        # The caller may pass (B, T, d) or (B, d); we operate per-position and
        # keep the batch axis aligned with the tape.
        tape = state.tape                              # (B, S, d_mem)
        S, d_mem = self.num_slots, self.d_mem

        # --- READ: content-based soft attention over slots, per position. ---
        q = self.q_proj(x)                              # (..., d_mem)
        # Reshape to (B, *, d_mem) and broadcast the tape (B, S, d_mem) over the
        # extra time dims.
        q_view = q.view(B, -1, d_mem)                   # (B, M, d_mem), M = prod(lead[1:])
        M = q_view.shape[1]
        # scores[b, m, s] = q_view[b, m] · tape[b, s]
        scores = torch.einsum("bmd,bsd->bms", q_view, tape) / math_sqrt(d_mem)
        read_weights = torch.softmax(scores, dim=-1)    # (B, M, S)

        # Entropy of the read distribution (collapse-prevention regularizer).
        eps = 1e-8
        entropy = -(read_weights * (read_weights + eps).log()).sum(-1)  # (B, M)
        read_entropy = state.read_entropy + entropy.mean()

        # read_vec[b, m] = Σ_s read_weights[b, m, s] · tape[b, s]
        read_vec = torch.einsum("bms,bsd->bmd", read_weights, tape)  # (B, M, d_mem)
        read_vec = read_vec.view(*lead, d_mem)
        read_out = self.read_out(read_vec)              # (..., d_model)

        # --- WRITE: NTM-style gated erase + add, aggregated over positions. ---
        # Each position produces an update; we aggregate them across the time
        # axis by mean so the tape stays bounded and the batch dim is preserved.
        v = self.v_proj_in(x).view(B, M, d_mem)         # (B, M, d_mem)
        w_gate = torch.sigmoid(self.write_gate(x)).view(B, M, 1)   # (B, M, 1)
        e_gate = torch.sigmoid(self.erase_gate(x)).view(B, M, d_mem)  # (B, M, d_mem)

        # Aggregate per-position writes into a single per-batch update.
        erase = (w_gate * e_gate).mean(dim=1)           # (B, d_mem)
        add = (w_gate * v).mean(dim=1)                  # (B, d_mem)
        new_tape = tape * (1.0 - erase.unsqueeze(1)) + add.unsqueeze(1)

        new_state = MemoryState(tape=new_tape, read_entropy=read_entropy)
        return read_out, new_state


def math_sqrt(x: float) -> float:
    import math

    return math.sqrt(x)
