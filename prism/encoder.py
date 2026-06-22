"""PrismEncoder — text → d_mem slots (the PRISM-KB mechanism, Module 1).

The encoder turns a piece of text into a fixed number of ``d_mem``-dimensional
slots that live in the same space the MemoryExpert's read head operates in. The
key design choice (from the PRISM-KB proposal) is to **reuse the Prism model's
embedding matrix** so the slots are in a distribution the read head has seen,
rather than an arbitrary external embedding space (e.g. a sentence-transformer)
that would shift the distribution and confuse the head.

Architecture (Perceiver / SET-transformer style):

    text → token-ids → Prism embedding (frozen) → (T, d_model)
                    ↓
        learned queries (num_slots_per_doc, d_model)
                    ↓ cross-attention (queries attend to token embeddings)
                (num_slots_per_doc, d_model)
                    ↓ linear d_model → d_mem
                (num_slots_per_doc, d_mem)   <- these are the KB slots

The cross-attention + query projection are the only trained parameters
(~num_slots_per_doc * d_model + small attention weights, well under 1% of
Prism). Everything else is frozen. After a one-time contrastive training
(Phase 2 in the proposal), the encoder is frozen forever.

This module is import-safe without a trained Prism checkpoint — the encoder
builds with the config's vocab/d_model, and you can ``load_state_dict`` the
frozen embedding from a checkpoint later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    """Configuration for the PrismEncoder."""

    d_model: int = 128
    """Width of the Prism model (must match the model you seed)."""

    d_mem: int = 64
    """Slot dimension (must match the MemoryConfig.d_mem of the model)."""

    num_slots_per_doc: int = 2
    """Number of slots produced per document."""

    num_heads: int = 4
    """Cross-attention heads."""


class PrismEncoder(nn.Module):
    """Encode text into a small set of d_mem slots.

    Args:
        encoder_config: dimensions.
        embed: an optional frozen ``nn.Embedding`` to reuse (the Prism input
            embedding). If None, a fresh embedding is created (for the
            no-checkpoint smoke path; real use passes the trained embedding).
        vocab_size: required if ``embed`` is None.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        embed: nn.Embedding | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = encoder_config
        d = encoder_config.d_model

        if embed is not None:
            self.embed = embed
            self.embed.requires_grad_(False)   # frozen — knowledge lives in slots, not here
        else:
            if vocab_size is None:
                raise ValueError("vocab_size required when embed is None")
            self.embed = nn.Embedding(vocab_size, d)
            self.embed.requires_grad_(False)

        # Learned perceiver-style queries.
        self.queries = nn.Parameter(torch.randn(encoder_config.num_slots_per_doc, d) * 0.02)

        # Cross-attention: queries (Q) attend to token embeddings (K, V).
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.num_heads = encoder_config.num_heads
        assert d % self.num_heads == 0

        self.norm = nn.LayerNorm(d)
        # Project the attended query-vectors down to d_mem slots.
        self.to_slot = nn.Linear(d, encoder_config.d_mem, bias=False)

        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.to_slot.weight, std=0.02)

    def encode_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token ids to slots.

        Args:
            input_ids: (B, T) long.

        Returns:
            slots: (B, num_slots_per_doc, d_mem).
        """
        B, T = input_ids.shape
        d = self.cfg.d_model
        h = self.num_heads
        dh = d // h

        tok = self.embed(input_ids)                       # (B, T, d)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, S, d)
        q = self.q_proj(q).view(B, -1, h, dh).transpose(1, 2)
        k = self.k_proj(tok).view(B, T, h, dh).transpose(1, 2)
        v = self.v_proj(tok).view(B, T, h, dh).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)    # (B, h, S, dh)
        attn = attn.transpose(1, 2).reshape(B, -1, d)     # (B, S, d)
        attn = self.norm(attn)
        slots = self.to_slot(attn)                        # (B, S, d_mem)
        return slots

    def encode_text(self, text: str, tokenizer, device, max_len: int = 256) -> torch.Tensor:
        """Encode a single text string to slots (1, S, d_mem)."""
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = ids["input_ids"].to(device)
        return self.encode_ids(input_ids)

    def encode_texts(self, texts: list[str], tokenizer, device, max_len: int = 256) -> torch.Tensor:
        """Encode a batch of texts to slots (B, S, d_mem), padded."""
        enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        return self.encode_ids(enc["input_ids"].to(device))
