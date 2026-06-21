"""PRISM Model.

Stacks L PrismBlocks over a token embedding, with a final RMSNorm and a
(vocab) output head. Produces logits plus an auxiliary loss aggregating
load-balancing, memory-read entropy, and symbolic-selection entropy.

The forward pass carries a single MemoryState through all blocks and (when
``carry_memory`` is True) reuses the final tape of one step as the initial
tape of the next — giving true persistent memory across generation steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prism.block import PrismBlock
from prism.config import PrismConfig
from prism.memory import MemoryState
from prism.norm import RMSNorm


@dataclass
class PrismOutput:
    """Output of a PRISM forward pass.

    Attributes:
        logits: shape (B, T, vocab_size).
        aux_loss: scalar combining load-balancing (all blocks).
        aux_breakdown: dict of named auxiliary scalars for logging.
        final_mem: the memory state after this forward pass (for carry-over).
        router_probs: per-block mean routing probability tensor (B, T, E)
            averaged over blocks — for diagnostics. None if not retained.
    """

    logits: torch.Tensor
    aux_loss: torch.Tensor
    aux_breakdown: dict
    final_mem: MemoryState
    router_probs: torch.Tensor | None


class Prism(nn.Module):
    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        # The last block omits the memory expert: its writes would go into a tape
        # that no downstream block reads, so those write weights would receive
        # no gradient (dead parameters). Dropping it keeps every parameter live.
        # Exception: if memory is the *only* expert kind (modular-memory mode),
        # we must keep it — otherwise the last block has zero experts.
        default_types = list(config.expert_types)
        blocks = []
        for i in range(config.num_layers):
            is_last = i == config.num_layers - 1
            if (
                is_last
                and "memory" in default_types
                and config.num_layers > 1
                and len(default_types) > 1
            ):
                last_types = [t for t in default_types if t != "memory"]
                blocks.append(PrismBlock(config, expert_types=last_types))
            else:
                blocks.append(PrismBlock(config))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.tie_embeddings:
            # Tied head: project through the (transposed) embedding matrix.
            self.lm_head = lambda h: nn.functional.linear(h, self.embed.weight)
        else:
            head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.head = head
            self.lm_head = lambda h: head(h)

        nn.init.normal_(self.embed.weight, std=config.init_std)
        if not config.tie_embeddings:
            nn.init.normal_(self.head.weight, std=config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mem: MemoryState | None = None,
    ) -> PrismOutput:
        """Forward pass.

        Args:
            input_ids: (B, T) long tensor of token ids.
            mem: optional initial memory state. If None, a fresh tape is
                created for this forward pass.

        Returns:
            PrismOutput.
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)                          # (B, T, d_model)

        if mem is None:
            mem = MemoryState.create(B, self.config.memory, device=device, dtype=x.dtype)

        total_aux = torch.zeros((), device=device, dtype=x.dtype)
        total_mem_ent = torch.zeros((), device=device, dtype=x.dtype)
        total_sym_ent = torch.zeros((), device=device, dtype=x.dtype)

        for block in self.blocks:
            x, mem, stats, aux_loss = block(x, mem)
            total_aux = total_aux + aux_loss
            if stats.memory_entropy is not None:
                total_mem_ent = total_mem_ent + stats.memory_entropy
            if stats.symbolic_entropy is not None:
                total_sym_ent = total_sym_ent + stats.symbolic_entropy

        h = self.final_norm(x)
        logits = self.lm_head(h)                           # (B, T, vocab)

        # Scale aux losses by their weights and sum.
        cfg = self.config
        aux = (
            cfg.router_load_balance_weight * total_aux
            - cfg.memory.read_entropy_weight * total_mem_ent   # encourage entropy
            - 0.0 * total_sym_ent                              # symbolic entropy is logged only
        )

        breakdown = {
            "load_balance": total_aux.detach(),
            "memory_entropy": total_mem_ent.detach(),
            "symbolic_entropy": total_sym_ent.detach(),
        }
        return PrismOutput(
            logits=logits,
            aux_loss=aux,
            aux_breakdown=breakdown,
            final_mem=mem,
            router_probs=None,
        )

    # --- convenience -------------------------------------------------------

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
