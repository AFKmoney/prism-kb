"""Polymorphic Router.

Per token, selects which expert *kind* (neural / memory / symbolic) computes
the output. Top-1 hard routing in the forward pass with a straight-through
gradient, plus a Switch-Transformer-style auxiliary load-balancing loss to keep
all expert kinds used.

The router sees the input feature *and* a summary of the current memory state
(mean over slots), so it can decide "I need to retrieve from memory" vs "I need
to transform".
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from prism.config import PrismConfig
from prism.experts import Expert, ExpertStats, build_expert
from prism.memory import MemoryState


class PolymorphicRouter(nn.Module):
    """Routes each token to one expert (top-1), differentiably.

    Holds the pool of heterogeneous experts. The routing decision is made from
    a learned projection over [x || mem_summary].

    Routing is *epsilon-soft*: the output is a convex mix of the hard one-hot
    routed output and a small (epsilon) contribution from all experts via the
    soft distribution. This guarantees every expert receives gradient signal
    on every step, preventing 'dead experts' that classic hard top-1 routing
    suffers from. ``epsilon`` can be annealed to 0 for pure symbolic behaviour
    at inference.
    """

    # Fraction of the output contributed by the soft (all-experts) path.
    # Kept as a buffer so it can be annealed by the trainer if desired.
    epsilon: torch.Tensor

    def __init__(self, config: PrismConfig, epsilon: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.config = config
        self.kinds = list(config.expert_types)
        self.num_experts = len(self.kinds)

        # Router input: x (d_model) concatenated with a memory summary (d_mem).
        # If memory is empty/unused we still pass zeros of width d_mem.
        router_in = config.d_model + config.memory.d_mem
        self.gate = nn.Linear(router_in, self.num_experts, bias=True)

        # Build the heterogeneous expert pool.
        self.experts = nn.ModuleList([build_expert(k, config) for k in self.kinds])

        nn.init.normal_(self.gate.weight, std=config.init_std)
        nn.init.zeros_(self.gate.bias)

    def _mem_summary(self, mem: MemoryState, batch_size: int, T: int, device, dtype) -> torch.Tensor:
        """Reduce the tape to a single (B, d_mem) summary, tiled to (B, T, d_mem)."""
        d_mem = self.config.memory.d_mem
        if mem.tape.numel() == 0:
            return torch.zeros(batch_size, T, d_mem, device=device, dtype=dtype)
        # tape: (B, S, d_mem) -> mean over slots -> (B, d_mem)
        summary = mem.tape.mean(dim=1)                 # (B, d_mem)
        return summary.unsqueeze(1).expand(batch_size, T, d_mem)

    def forward(
        self, x: torch.Tensor, mem: MemoryState
    ) -> tuple[torch.Tensor, MemoryState, ExpertStats, torch.Tensor]:
        """Route and apply experts.

        Args:
            x: (B, T, d_model) post-norm features.
            mem: current shared memory state.

        Returns:
            out: (B, T, d_model)
            new_mem: updated memory state
            stats: merged auxiliary stats
            aux_loss: scalar load-balancing loss
        """
        B, T, d = x.shape
        device, dtype = x.device, x.dtype

        mem_sum = self._mem_summary(mem, B, T, device, dtype)   # (B, T, d_mem)
        router_x = torch.cat([x, mem_sum], dim=-1)              # (B, T, d_model+d_mem)
        logits = self.gate(router_x)                            # (B, T, E)
        p_soft = torch.softmax(logits, dim=-1)                  # (B, T, E)

        # Top-k hard selection with straight-through gradient (k = router_topk).
        # The hard mask is a {0,1} tensor marking the k highest-logit experts per
        # token; we renormalize the soft probs *within* the selected set so the
        # selected experts split weight 1.0 among themselves (proper top-k MoE).
        k = min(self.config.router_topk, self.num_experts)
        topk_idx = logits.topk(k, dim=-1).indices               # (B, T, k)
        hard_mask = torch.zeros_like(p_soft).scatter_(
            -1, topk_idx, 1.0
        )                                                       # (B, T, E) {0,1}
        # Renormalized soft probs over the selected experts only.
        masked_soft = p_soft * hard_mask
        denom = masked_soft.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        p_selected = masked_soft / denom                        # sums to 1 over the k
        # Straight-through: forward uses p_selected, backward uses p_soft's grad.
        p_st = p_selected + p_soft - p_soft.detach()
        # To keep the "selected experts split weight 1.0" semantics in the ST
        # forward while still letting non-selected experts get a little gradient
        # (epsilon-soft), we blend: most weight on the hard-selected renormalized
        # path, a small fraction on the full soft distribution.
        eps = float(self.epsilon.item())
        p = (1.0 - eps) * p_st + eps * p_soft

        # --- Run all experts, then mask-combine. ---
        # At toy scale (few experts) running all and masking is simpler
        # and avoids gather/scatter. Cost is E x expert FLOPs, fine here.
        out_acc = torch.zeros_like(x)
        merged_stats = ExpertStats.empty(device, dtype)
        per_expert_counts = []
        mem_tape_after = None            # tape produced by the memory expert, if any
        mem_read_entropy_after = None
        for i, expert in enumerate(self.experts):
            expert_out, new_mem_i, stats_i = expert(x, mem)
            mask_i = p[..., i].unsqueeze(-1)                    # (B, T, 1)
            out_acc = out_acc + expert_out * mask_i
            merged_stats = merged_stats.merge(stats_i)
            # Count how many tokens *selected* this expert (via the hard mask),
            # so the load-balancing loss reflects true top-k assignment.
            per_expert_counts.append(hard_mask[..., i].sum())
            # Capture the memory expert's resulting tape (only one such expert).
            if expert.expert_type == "memory":
                mem_tape_after = new_mem_i.tape
                mem_read_entropy_after = new_mem_i.read_entropy

        # --- Memory update: blend the memory expert's tape by selection weight. ---
        # The memory expert mutates a per-batch tape, but selection is per-token.
        # We blend its resulting tape with the original tape by the fraction of
        # tokens that selected the memory expert — a convex, differentiable mix.
        new_mem = mem
        if mem_tape_after is not None:
            mem_idx = self.kinds.index("memory")
            frac = hard_mask[..., mem_idx].float().mean().clamp(0.0, 1.0)
            blended_tape = mem.tape * (1.0 - frac) + mem_tape_after * frac
            new_mem = MemoryState(
                tape=blended_tape, read_entropy=mem_read_entropy_after
            )

        # --- Auxiliary load-balancing loss (Switch Transformer). ---
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability for expert i
        # loss = E * Σ f_i · P_i   (encourages uniform routing)
        E = self.num_experts
        total_tokens = max(B * T, 1)
        f = torch.stack(per_expert_counts) / total_tokens        # (E,)
        P = p_soft.mean(dim=(0, 1))                              # (E,)
        aux_loss = E * (f * P).sum()

        return out_acc, new_mem, merged_stats, aux_loss
