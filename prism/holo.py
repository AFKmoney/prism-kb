"""PRISM-Holo — holographic memory tape (Vector Symbolic Architecture).

Replaces the (num_slots, d_mem) soft-attention tape with a SINGLE high-
dimensional bipolar vector H. Knowledge is superposed algebraically:

    binding:    H += bind(enc(fact), enc(key))     # element-wise multiply
    unbinding:  retrieved = enc(query) * H          # same op (XOR/bipolar is self-inverse)
    cleanup:    retrieved ≈ enc(fact)               # by the Johnson-Lindenstrauss property

Why this is the breakthrough path for PRISM:
  * Adding knowledge is an ALGEBRAIC operation, not a gradient step. True one-shot.
  * Storage is superposed: every dimension of H holds a fragment of every stored
    fact. Capacity scales ~D/(8·log N) facts before noise overflows (Kanerva).
  * Retrieval is O(D), independent of how many facts are stored. No attention.
  * Zero training needed for the memory path — the operations are fixed math.

This is the "holography" route out of the 6·N·D trap: the memory expert is no
longer a trained neural module, it's an algebraic operator. Only the encoder
and the main model are trained; memory is free.

We use BIPOLAR VSA (Kanerva 2009): vectors in {-1, +1}^D. Binding is element-
wise multiplication (Hadamard), which is its own inverse — unbinding uses the
same op. The bundle (sum) of bound pairs is thresholded back to bipolar.
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class HoloTape:
    """A bipolar Vector Symbolic Architecture memory.

    Attributes:
        D: holographic dimensionality (large: 4096-16384).
        H: (D,) bipolar superposition register, in {-1, +1}.
        count: how many facts are bound into H.
    """

    def __init__(self, D: int = 8192, device=None, dtype=torch.float32) -> None:
        self.D = D
        # IMPORTANT: H starts at ZEROS, not ones. In Kanerva VSA the superposition
        # register is a real-valued accumulator; the bound pairs are SUMMED, and
        # binarization happens ONLY at retrieval time (to clean up the unbound
        # vector against a codebook). Binarizing H after every bind destroys the
        # signal because the "empty" state (+1 everywhere) doubles the +1
        # positions and zeros the -1 positions of the bound pair — half the
        # information is lost on the first bind.
        self.H = torch.zeros(D, device=device, dtype=dtype)
        self.count = 0
        self.device = device
        self.dtype = dtype

    # --- core VSA operations (all algebraic, zero gradient) ---------------

    @staticmethod
    def _to_bipolar(v: torch.Tensor) -> torch.Tensor:
        """Threshold a real vector to {-1, +1}."""
        return torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))

    def bind(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Bind a (key, value) pair into the superposition register.

        Both key and value are bipolarized, multiplied element-wise
        (Hadamard = binding op, self-inverse), and ADDED to H. H stays
        REAL-valued across binds — only retrieval (unbind) bipolarizes the
        query and the result. This preserves the Kanerva capacity rule.

        Args:
            key, value: (D,) real vectors. Bipolarized internally.
        """
        assert key.shape[-1] == self.D and value.shape[-1] == self.D
        k = self._to_bipolar(key)
        v = self._to_bipolar(value)
        bound = k * v                  # (D,) binding op; self-inverse
        self.H = self.H + bound        # superposition (real-valued sum)
        self.count += 1

    def unbind(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve the value bound to `query`.

        Since binding is self-inverse, unbinding is the SAME op applied to the
        (real-valued) accumulator:
            retrieved = bipolar(query) * H
        The retrieved real vector is a noisy version of the original value;
        the component aligned with the true value dominates as long as we're
        under capacity (~D / 8 log N facts). We return the REAL vector —
        callers can bipolarize it or compare cosines directly.

        Returns:
            (D,) real vector. Compare its similarity to candidate values.
        """
        assert query.shape[-1] == self.D
        q = self._to_bipolar(query)
        return q * self.H              # (D,) real; dominant signal = bound value

    def reset(self) -> None:
        """Clear the tape."""
        self.H = torch.zeros_like(self.H)
        self.count = 0

    @property
    def capacity_estimate(self) -> float:
        """Kanerva's capacity rule of thumb: ~D / (8 · log2(N+1)) facts."""
        if self.count == 0:
            return float(self.D / 8.0)
        return float(self.D / (8.0 * math.log2(self.count + 1)))

    def summary(self) -> str:
        return f"HoloTape(D={self.D}, facts={self.count}, capacity~{self.capacity_estimate:.0f})"


# ---------------------------------------------------------------------------
# Encoder: project a dense embedding into the D-dim VSA space.
# ---------------------------------------------------------------------------


class HoloEncoder(nn.Module):
    """Project a d_model embedding into D-dimensional VSA space.

    A single linear projection + bipolar thresholding. This is the ONLY
    trained piece on the memory path (and it's tiny: d_model x D params).
    Everything downstream (bind/unbind) is fixed algebra.

    The projection is trained so that semantically similar embeddings map to
    similar bipolar vectors (high cosine similarity preserved through binarization).
    """

    def __init__(self, d_model: int, D: int = 8192) -> None:
        super().__init__()
        self.D = D
        self.proj = nn.Linear(d_model, D, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor, bipolar: bool = True) -> torch.Tensor:
        """Encode (..., d_model) -> (..., D). Bipolarized by default."""
        h = self.proj(x)
        if bipolar:
            h = torch.where(h >= 0, torch.ones_like(h), -torch.ones_like(h))
        return h


# ---------------------------------------------------------------------------
# Retrieval utilities.
# ---------------------------------------------------------------------------


def cosine_retrieve(query_vec: torch.Tensor, candidates: torch.Tensor) -> int:
    """Return the index of the candidate most similar to the query.

    Args:
        query_vec: (D,).
        candidates: (N, D).

    Returns:
        argmax index.
    """
    sims = F.cosine_similarity(query_vec.unsqueeze(0), candidates, dim=-1)
    return int(sims.argmax().item())


def description() -> str:
    return (
        "PRISM-Holo: holographic (VSA) memory tape. Knowledge stored by "
        "algebraic binding (Hadamard), retrieved by self-inverse unbinding. "
        "Zero gradient on the memory path — true one-shot, true parallelism "
        "target (only encoder + main model trained)."
    )
