"""Polymorphic Experts.

All experts share one interface::

    forward(x: Tensor, mem: MemoryState) -> (output: Tensor, mem': MemoryState,
                                              stats: ExpertStats)

This uniform interface is what lets the router treat them as interchangeable.
The three implementations differ in *kind* of computation:

* ``NeuralExpert``  — SwiGLU MLP. No memory interaction (returns mem unchanged).
* ``MemoryExpert``  — read/write head over the shared bus.
* ``SymbolicExpert`` — differentiable primitive library.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from prism.config import PrismConfig
from prism.memory import MemoryHead, MemoryState
from prism.symbolic import SymbolicLibrary


@dataclass
class ExpertStats:
    """Per-expert auxiliary signals (regularizers, diagnostics).

    All values are scalars (shape ()) already averaged/reduced as appropriate.
    """

    load_balance_contrib: torch.Tensor = None  # type: ignore[assignment]
    """Contribution to the load-balancing loss (fraction of tokens routed)."""

    memory_entropy: torch.Tensor = None  # type: ignore[assignment]
    """Memory read-distribution entropy (Memory expert only)."""

    symbolic_entropy: torch.Tensor = None  # type: ignore[assignment]
    """Symbolic primitive selection entropy (Symbolic expert only)."""

    @classmethod
    def empty(cls, device, dtype) -> "ExpertStats":
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(load_balance_contrib=z, memory_entropy=z, symbolic_entropy=z)

    def merge(self, other: "ExpertStats") -> "ExpertStats":
        return ExpertStats(
            load_balance_contrib=(self.load_balance_contrib or 0) + (other.load_balance_contrib or 0),
            memory_entropy=(self.memory_entropy or 0) + (other.memory_entropy or 0),
            symbolic_entropy=(self.symbolic_entropy or 0) + (other.symbolic_entropy or 0),
        )


class Expert(nn.Module):
    """Base class. Subclasses implement forward()."""

    expert_type: str = "base"

    def forward(self, x: torch.Tensor, mem: MemoryState) -> tuple[torch.Tensor, MemoryState, ExpertStats]:
        raise NotImplementedError


class NeuralExpert(Expert):
    """SwiGLU MLP. The 'classical' expert — no memory interaction."""

    expert_type = "neural"

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        hidden = config.neural_hidden
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.d_model, bias=False)
        nn.init.normal_(self.w_gate.weight, std=config.init_std)
        nn.init.normal_(self.w_up.weight, std=config.init_std)
        nn.init.normal_(self.w_down.weight, std=config.init_std)

    def forward(self, x: torch.Tensor, mem: MemoryState) -> tuple[torch.Tensor, MemoryState, ExpertStats]:
        g = F.silu(self.w_gate(x))
        u = self.w_up(x)
        h = g * u
        out = self.w_down(h)
        return out, mem, ExpertStats.empty(x.device, x.dtype)


class MemoryExpert(Expert):
    """Read/write head over the shared memory bus."""

    expert_type = "memory"

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.head = MemoryHead(config.d_model, config.memory)

    def forward(self, x: torch.Tensor, mem: MemoryState) -> tuple[torch.Tensor, MemoryState, ExpertStats]:
        # The head handles arbitrary leading dims as long as the first dim
        # (batch) matches mem.tape.shape[0].
        out, new_mem = self.head(x, mem)
        stats = ExpertStats(
            load_balance_contrib=torch.zeros((), device=x.device, dtype=x.dtype),
            memory_entropy=new_mem.read_entropy,
            symbolic_entropy=torch.zeros((), device=x.device, dtype=x.dtype),
        )
        return out, new_mem, stats


class HoloMemoryExpert(Expert):
    """Algebraic holographic (VSA) memory expert — PRISM-Holo.

    Drop-in replacement for MemoryExpert. Uses HoloHead (algebraic bind/unbind,
    no soft attention) so the memory path requires no trained read/write weights
    beyond a tiny shared encoder + read-out. The MemoryState.tape is interpreted
    as a flat holographic register of dimension D = num_slots * d_mem.
    """

    expert_type = "memory"   # same kind string so the router treats it identically

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        from prism.holo import HoloHead

        self.head = HoloHead(
            d_model=config.d_model,
            num_slots=config.memory.num_slots,
            d_mem=config.memory.d_mem,
        )

    def forward(self, x: torch.Tensor, mem: MemoryState) -> tuple[torch.Tensor, MemoryState, ExpertStats]:
        out, new_mem = self.head(x, mem)
        stats = ExpertStats(
            load_balance_contrib=torch.zeros((), device=x.device, dtype=x.dtype),
            memory_entropy=new_mem.read_entropy,
            symbolic_entropy=torch.zeros((), device=x.device, dtype=x.dtype),
        )
        return out, new_mem, stats


class SymbolicExpert(Expert):
    """Differentiable primitive library."""

    expert_type = "symbolic"

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.lib = SymbolicLibrary(config)

    def forward(self, x: torch.Tensor, mem: MemoryState) -> tuple[torch.Tensor, MemoryState, ExpertStats]:
        out, p_soft = self.lib(x)
        # Entropy of the primitive selection — encourage exploration early.
        eps = 1e-8
        ent = -(p_soft * (p_soft + eps).log()).sum(-1).mean()
        stats = ExpertStats(
            load_balance_contrib=torch.zeros((), device=x.device, dtype=x.dtype),
            memory_entropy=torch.zeros((), device=x.device, dtype=x.dtype),
            symbolic_entropy=ent,
        )
        return out, mem, stats


def build_expert(kind: str, config: PrismConfig) -> Expert:
    """Factory keyed by expert kind string."""
    if kind == "neural":
        return NeuralExpert(config)
    if kind == "memory":
        # PRISM-Holo: when holo_mode is set, swap the soft-attention head for
        # the algebraic VSA head. Same expert_type ("memory") so the router
        # and load-balancing treat it identically.
        if getattr(config, "holo_mode", False):
            return HoloMemoryExpert(config)
        return MemoryExpert(config)
    if kind == "symbolic":
        return SymbolicExpert(config)
    raise ValueError(f"unknown expert kind: {kind!r}")
