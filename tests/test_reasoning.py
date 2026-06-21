"""Tests for the reasoning task (lever 3 validation target)."""

from __future__ import annotations

import random

import torch

from tasks import reasoning


def test_reasoning_generates_valid_shapes():
    g = torch.Generator().manual_seed(0)
    ids, tgt, mask = reasoning.generate_batch(8, n_steps=3, value_range=16, device="cpu", generator=g)
    assert ids.shape == tgt.shape == mask.shape
    assert ids.shape[0] == 8
    # Exactly one answer position per example (loss_mask sums to 1).
    assert torch.allclose(mask.sum(-1), torch.ones(8))


def test_reasoning_answer_is_derivable():
    """The answer token must equal the result of applying the chained ops."""
    rng = random.Random(42)
    for _ in range(20):
        seq, answer = reasoning.generate_example(n_steps=4, value_range=16, rng=rng)
        # Re-derive.
        val = seq[0] - reasoning.NUM_OFFSET
        i = 1
        while seq[i] != reasoning.SEP:
            op = seq[i]
            delta = seq[i + 1] - reasoning.NUM_OFFSET
            val = min(15, val + delta) if op == reasoning.ADD else max(0, val - delta)
            i += 2
        assert answer == reasoning.NUM_OFFSET + val


def test_reasoning_vocab_is_small():
    """Toy vocab must be learnable quickly (values+offset fits in ~20 tokens)."""
    g = torch.Generator().manual_seed(0)
    ids, _, _ = reasoning.generate_batch(4, n_steps=3, value_range=16, device="cpu", generator=g)
    max_tok = int(ids.max().item())
    assert max_tok < 25   # NUM_OFFSET(4) + 16 values = 20, +operators


def test_reasoning_description():
    desc = reasoning.description(3, 16)
    assert "reasoning" in desc and "3" in desc
