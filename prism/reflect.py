"""Reflect — multi-pass internal reflection loop (COGLOOP Section 2).

Before answering, PRISM "thinks": it makes N passes over its memory, each pass
enriching the query with retrieved context. The loop stops when the context
converges (||ctx_n - ctx_{n-1}|| < threshold) or a max-passes budget is hit.

Why this is an architectural advantage: a Transformer+RAG does multi-hop by
stuffing more chunks into the prompt — context bloat is linear per hop. PRISM
accumulates context in the FIXED-SIZE tape (num_slots x d_mem), so the cost per
reflection pass is constant. Thinking longer costs no extra memory.

Pipeline::

    q_0 = encode(question)
    ctx_0 = 0
    for pass in 1..max_passes:
        slots_n = retrieve(KB, q_{n-1})            # multi-hop: query evolves
        tape_n  = seed_tape(slots_n + ctx_{n-1})
        read_n  = read_head(q_{n-1}, tape_n)       # native retrieval
        ctx_n   = ctx_{n-1} + alpha * read_n        # accumulate
        q_n     = q_{n-1} + beta * read_n           # refined query
        if ||ctx_n - ctx_{n-1}|| < converge_threshold: break
    answer = generate(prompt, seed=ctx_N)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from prism.config import PrismConfig
from prism.kb import KnowledgeBase
from prism.memory import MemoryState


@dataclass
class ReflectConfig:
    """Hyperparameters for the reflection loop."""

    max_passes: int = 5
    """Hard cap on reflection passes (budget)."""

    converge_threshold: float = 0.05
    """Stop when ||ctx_n - ctx_{n-1}|| drops below this (relative to ctx norm)."""

    alpha: float = 0.5
    """Context accumulation rate (how much each pass's read adds to ctx)."""

    beta: float = 0.3
    """Query refinement rate (how much each pass's read shifts the query)."""

    top_k_per_pass: int = 4
    """Number of KB slots retrieved per pass."""


@dataclass
class ReflectTrace:
    """Diagnostics for one reflection episode (for inspection/debugging)."""

    passes_used: int = 0
    converged: bool = False
    context_norms: list = field(default_factory=list)
    query_shifts: list = field(default_factory=list)


class Reflector:
    """The reflection loop. Stateless w.r.t. history — call per question.

    Args:
        model: a Prism model.
        config: the PrismConfig.
        kb: a KnowledgeBase to retrieve from (None = reflect on tape only).
        reflect_config: reflection hyperparameters.
        encoder: optional PrismEncoder or AnalyticCapture to encode the query
            into slots for retrieval. If None, retrieval is skipped and the
            loop only re-reads the seeded tape.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PrismConfig,
        kb: KnowledgeBase | None,
        reflect_config: ReflectConfig | None = None,
        encoder=None,
    ) -> None:
        self.model = model
        self.config = config
        self.kb = kb
        self.rc = reflect_config or ReflectConfig()
        self.encoder = encoder
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.d_mem = config.memory.d_mem

    @torch.no_grad()
    def reflect(
        self,
        query_input_ids: torch.Tensor,
        initial_seed_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ReflectTrace]:
        """Run the reflection loop. Returns (final_context_slots, trace).

        Args:
            query_input_ids: (1, T_q) the question's token ids.
            initial_seed_slots: optional (K0, d_mem) slots to start the tape
                with (e.g. from a first-pass retrieval or capture).

        Returns:
            final_context: (num_slots, d_mem) accumulated context to seed the
                final generation tape.
            trace: ReflectTrace for diagnostics.
        """
        rc = self.rc
        trace = ReflectTrace()

        # Initial context: the seed slots (or zeros).
        S = self.config.memory.num_slots
        if initial_seed_slots is not None:
            ctx = initial_seed_slots.to(self.device, self.dtype).clone()
            if ctx.shape[0] < S:
                pad = torch.zeros(S - ctx.shape[0], self.d_mem, device=self.device, dtype=self.dtype)
                ctx = torch.cat([ctx, pad], dim=0)
            elif ctx.shape[0] > S:
                ctx = ctx[:S]
        else:
            ctx = torch.zeros(S, self.d_mem, device=self.device, dtype=self.dtype)

        # Query vector: pool the model's embedding of the question.
        q_vec = self.model.embed(query_input_ids).mean(dim=1).squeeze(0)  # (d_model,)

        for n in range(rc.max_passes):
            ctx_before = ctx.clone()
            ctx_norm_before = ctx.norm().item()
            trace.context_norms.append(ctx_norm_before)

            # --- Retrieve more slots if we have a KB and an encoder. ---
            if self.kb is not None and self.encoder is not None:
                # Encode the current query into d_mem slots for retrieval.
                # Use the encoder on the (refined) query embedding.
                q_slots = self._query_to_slots(q_vec)   # (top_k, d_mem)
                retrieved = self.kb.retrieve(q_slots, top_k=rc.top_k_per_pass)
                retrieved = retrieved.to(self.device, self.dtype)
                # Inject retrieved slots into the first top_k positions.
                k = min(retrieved.shape[0], S)
                if k > 0:
                    ctx[:k] = retrieved[:k]

            # --- Read the tape through the model's read head to get a readout. ---
            # We run a single forward pass with the current ctx as the tape seed,
            # and pull out the memory expert's read vector at the query position.
            read_vec = self._read_at(query_input_ids, ctx)   # (d_mem,)

            # --- Accumulate context + refine query. ---
            ctx_new = ctx + rc.alpha * read_vec.unsqueeze(0).expand_as(ctx)
            q_vec = q_vec + rc.beta * self._project_read_to_query(read_vec)

            # --- Convergence check (relative). ---
            shift = (ctx_new - ctx_before).norm().item()
            rel_shift = shift / max(ctx_norm_before, 1e-6)
            trace.query_shifts.append(rel_shift)
            ctx = ctx_new
            trace.passes_used = n + 1

            if rel_shift < rc.converge_threshold:
                trace.converged = True
                break

        return ctx, trace

    def _query_to_slots(self, q_vec: torch.Tensor) -> torch.Tensor:
        """Project the query embedding into d_mem slot space for retrieval."""
        # If the encoder is an AnalyticCapture, it works on input_ids not
        # embeddings; we approximate by using the query embedding directly
        # truncated/projected to d_mem. For a PrismEncoder we'd call encode.
        # Simple robust path: linear projection of the (d_model,) query to d_mem.
        if q_vec.shape[0] == self.d_mem:
            return q_vec.unsqueeze(0)
        # Truncate or pad to d_mem.
        if q_vec.shape[0] > self.d_mem:
            return q_vec[: self.d_mem].unsqueeze(0)
        pad = torch.zeros(self.d_mem - q_vec.shape[0], device=self.device, dtype=self.dtype)
        return torch.cat([q_vec, pad]).unsqueeze(0)

    def _read_at(self, query_input_ids: torch.Tensor, ctx_slots: torch.Tensor) -> torch.Tensor:
        """Run one forward pass seeded with ctx_slots; return the memory readout
        at the last query position, in d_mem space.

        We do this by seeding the tape and reading the MemoryExpert's output
        (which is already a d_model readout), then projecting to d_mem via the
        inverse-ish of the read head's read_out (approx: truncate).
        """
        mem = MemoryState.from_knowledge(
            ctx_slots, batch_size=1, config=self.config.memory,
            device=self.device, dtype=self.dtype,
        )
        out = self.model(query_input_ids, mem=mem)
        # The logits' last position reflects the read-influenced hidden state.
        # We use the final hidden (pre-head) as a proxy for the readout.
        # Easiest robust signal: the difference between this logits row and the
        # scratch logits row tells us what the seeded tape contributed.
        # For the loop we just need a stable read signal; use the embedding-space
        # difference projected to d_mem.
        scratch_logits = self.model(query_input_ids).logits[:, -1, :]
        seeded_logits = out.logits[:, -1, :]
        delta = (seeded_logits - scratch_logits).squeeze(0)   # (vocab,)
        # Project to d_mem via the (transposed) embedding matrix — this gives a
        # d_model signal pointing in the direction of the changed tokens, then
        # truncate to d_mem. Stable and differentiable-in-spirit.
        d_model_signal = (self.model.embed.weight.T @ delta)   # (d_model,)
        return d_model_signal[: self.d_mem]

    def _project_read_to_query(self, read_vec: torch.Tensor) -> torch.Tensor:
        """Project a d_mem read vector back to d_model for query refinement."""
        d = self.config.d_model
        if read_vec.shape[0] == d:
            return read_vec
        if read_vec.shape[0] > d:
            return read_vec[:d]
        pad = torch.zeros(d - read_vec.shape[0], device=self.device, dtype=self.dtype)
        return torch.cat([read_vec, pad])


def description() -> str:
    return (
        "Reflect: multi-pass internal reflection loop. PRISM 'thinks' before "
        "answering — each pass retrieves + reads + refines the query. Stops at "
        "convergence. Constant memory cost per pass (no context bloat)."
    )
