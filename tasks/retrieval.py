"""Retrieval task for training the read head (COGLOOP Phase 3).

THE TASK THAT WAKES THE READ HEAD.

On this task, the model MUST learn to read seeded KB slots, because the answer
is only available in the seeded tape — it's not in the input sequence.

Layout per example (vocab is small + synthetic so it's learnable on CPU)::

    input sequence : [QUERY, key_token, PAD, PAD, ...]
    seeded tape    : key_token encoded as slot 0 (value encoded as slot 1)
    target         : value_token at the position right after the key

The gradient flows: target -> logits -> hidden -> read_head -> seeded tape.
So the read head's q_proj / read_out are forced to learn to retrieve the value
from the slot that matches the key. This is exactly the induction-head
mechanism but routed through the SEED path — which is what Phase 3 needs.

This is deliberately tiny and synthetic so it converges on CPU in a few hundred
steps. The point is to prove the read head CAN learn seeded retrieval; the
cluster run scales the same principle to real text.
"""

from __future__ import annotations

import torch

# Vocab layout for the retrieval task (kept tiny for fast CPU learning).
PAD = 0
QUERY = 1                 # the "ask" token
KEY_OFFSET = 2            # key tokens live in [2, 2+n_pairs)
VAL_OFFSET = 100          # value tokens live in a separate range


def generate_batch(
    batch_size: int,
    n_pairs: int,
    seq_len: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a retrieval batch.

    Returns:
        input_ids : (B, seq_len) — [QUERY, key, PAD...]
        targets   : (B, seq_len) — PAD except position 2 = value
        loss_mask : (B, seq_len) — 1 only at position 2
        seed_keys : (B,) the key token id (so the trainer can build a matching seed)
    """
    B = batch_size
    # Pick a random key per example.
    keys = torch.randint(KEY_OFFSET, KEY_OFFSET + n_pairs, (B,), device=device, generator=generator)
    # The value is a deterministic function of the key (so the model can learn
    # the mapping; the seed carries the value).
    values = keys + (VAL_OFFSET - KEY_OFFSET)   # distinct value range

    input_ids = torch.full((B, seq_len), PAD, dtype=torch.long, device=device)
    input_ids[:, 0] = QUERY
    input_ids[:, 1] = keys

    targets = torch.full((B, seq_len), PAD, dtype=torch.long, device=device)
    targets[:, 2] = values

    loss_mask = torch.zeros((B, seq_len), dtype=torch.float32, device=device)
    loss_mask[:, 2] = 1.0
    return input_ids, targets, loss_mask, keys


def build_seed_slots(keys: torch.Tensor, d_mem: int, device, dtype) -> torch.Tensor:
    """Build a seeded tape where slot 0 encodes the key and slot 1 encodes
    a value-marker.

    The exact slot content is learned via the model's own write path in real
    use (AnalyticCapture). For this TRAINING task we use a simple deterministic
    encoding: one-hot-ish vectors scaled to a useful magnitude, so the model
    has a learnable signal. The read head must learn to map (query embedding
    of key) -> (correct value) by reading the slot.
    """
    B = keys.shape[0]
    S = 2   # two slots: key slot, value slot
    slots = torch.zeros(B, S, d_mem, device=device, dtype=dtype)
    # Slot 0: encode the key (a scaled one-hot in the first n_pairs dims).
    keys_long = keys.long()
    for b in range(B):
        slots[b, 0, keys_long[b] % d_mem] = 1.0
    # Slot 1: encode the value (a distinct pattern).
    values = keys_long + (VAL_OFFSET - KEY_OFFSET)
    for b in range(B):
        slots[b, 1, values[b] % d_mem] = 1.0
    return slots


def description(n_pairs: int) -> str:
    return (
        f"retrieval(n_pairs={n_pairs}): the read head MUST learn to read seeded "
        f"slots — the answer is only in the tape, not in the input. Wakes the "
        f"inert head."
    )
