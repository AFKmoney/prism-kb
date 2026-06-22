"""Configuration dataclasses for PRISM.

All hyperparameters live here as plain dataclasses so the entire model can be
rebuilt deterministically from a ``PrismConfig`` instance. No magic, no global
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Configuration for the Shared Memory Bus.

    The bus is a single tape of shape ``(num_slots, d_mem)`` that flows through
    every block and every time step.
    """

    d_mem: int = 64
    """Width of each memory slot."""

    num_slots: int = 32
    """Number of slots in the tape. Fixed across the whole model."""

    num_read_heads: int = 1
    """Read heads used by each Memory expert (typically 1 for toy scale)."""

    read_entropy_weight: float = 0.01
    """Weight of the read-distribution entropy regularizer. Prevents collapse."""

    init_std: float = 0.05
    """Std of the initial memory content (drawn once per forward pass)."""


@dataclass
class PrismConfig:
    """Top-level configuration for a PRISM model.

    Toy-scale defaults are CPU-friendly (~1–3M params).
    """

    # --- vocab / io ---
    vocab_size: int = 256
    """Size of the token vocabulary (byte-level by default)."""

    pad_token_id: int = 0
    """Pad token id, used to mask padding in the loss."""

    # --- backbone ---
    d_model: int = 128
    """Hidden width of the model."""

    num_layers: int = 4
    """Number of PRISM blocks."""

    # --- Multi-Rate Bus ---
    num_rates: int = 4
    """Number of temporal-rate groups K in the MRB."""

    mrb_delta0: float = 0.6931471805599453
    """Base delta (ln 2). Half-life of group 0 is 1 step."""

    mrb_max_delta: float = 5.545177444479562
    """Max delta (8 * ln 2). Half-life of group K-1 is 8 steps by default.
    The full schedule is geometric from mrb_delta0 to mrb_max_delta."""

    # --- polymorphic experts ---
    expert_types: tuple = ("neural", "memory", "symbolic")
    """Which expert kinds live in each block's router. Defaults to all three."""

    router_topk: int = 1
    """Top-k experts activated per token. 1 keeps CPU cost minimal."""

    holo_mode: bool = False
    """If True, the memory expert uses HoloHead (algebraic VSA) instead of
    the soft-attention MemoryHead. Activates PRISM-Holo: zero trained weights
    on the memory read/write path. Set num_slots*d_mem >= 1024 for good VSA
    dimensionality (e.g. num_slots=256, d_mem=32 -> D=8192)."""

    router_load_balance_weight: float = 0.01
    """Weight of the auxiliary load-balancing loss (Switch Transformer style)."""

    # --- Neural expert (SwiGLU MLP) ---
    neural_hidden_mult: int = 2
    """Multiplier on d_model for the SwiGLU hidden dim."""

    # --- Symbolic expert ---
    symbolic_num_primitives: int = 6
    """Number of primitives in the differentiable library (see symbolic.py)."""

    # --- Memory ---
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # --- embeddings / head ---
    tie_embeddings: bool = True
    """If True, the output head shares weights with the input embedding."""

    # --- norm ---
    norm_eps: float = 1e-6
    """Epsilon for RMSNorm layers."""

    # --- init ---
    init_std: float = 0.02
    """Std of the weight init for linear layers."""

    def __post_init__(self) -> None:
        if self.d_model % self.num_rates != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_rates "
                f"({self.num_rates}) so each rate group has equal width."
            )
        if self.router_topk > len(self.expert_types):
            raise ValueError(
                f"router_topk ({self.router_topk}) cannot exceed the number "
                f"of expert types ({len(self.expert_types)})."
            )

    # Convenience -----------------------------------------------------------

    @property
    def d_rate(self) -> int:
        """Width of a single rate group."""
        return self.d_model // self.num_rates

    @property
    def neural_hidden(self) -> int:
        return self.neural_hidden_mult * self.d_model

    def rate_deltas(self) -> list[float]:
        """Return the geometric decay schedule Δ_k for k = 0..K-1.

        Log-spaced from mrb_delta0 to mrb_max_delta. Using a log schedule means
        the half-lives grow geometrically, covering local and global context
        with K groups.
        """
        import math

        k = self.num_rates
        if k == 1:
            return [self.mrb_delta0]
        log_lo = math.log(self.mrb_delta0)
        log_hi = math.log(self.mrb_max_delta)
        return [math.exp(log_lo + (log_hi - log_lo) * i / (k - 1)) for i in range(k)]
