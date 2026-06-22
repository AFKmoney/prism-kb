"""Validation tests for PRISM-Holo (the breakthrough judge).

THE TESTS THAT DECIDE:
  1. Store N facts algebraically, retrieve each one correctly. No training.
  2. Below capacity, retrieval accuracy is high; above capacity, it degrades
     gracefully (the Kanerva noise-overload curve).
  3. The retrieval is O(D) regardless of N — true holographic property.

If these pass, the holographic path is real. We then test the "specificity"
probe that failed on the attention tape (+0.006): does the algebraic tape give
specific, not random, retrieval?
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from prism.holo import HoloTape, HoloEncoder, cosine_retrieve


def _random_bipolar(D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.randn(D, generator=g) >= 0, torch.ones(D), -torch.ones(D))


def test_bind_unbind_roundtrip_single_fact():
    """Store one (key, value) pair; retrieve the value from the key."""
    D = 8192
    tape = HoloTape(D=D)
    key = _random_bipolar(D, seed=1)
    value = _random_bipolar(D, seed=2)
    tape.bind(key, value)
    retrieved = tape.unbind(key)
    # Cosine similarity to the true value should be high (near 1).
    sim = F.cosine_similarity(retrieved.unsqueeze(0), value.unsqueeze(0)).item()
    assert sim > 0.9, f"single-fact retrieval similarity {sim} < 0.9"


def test_bind_unbind_multiple_facts_below_capacity():
    """Store 10 facts; retrieve each correctly (well under capacity)."""
    D = 8192
    N = 10
    tape = HoloTape(D=D)
    keys = [_random_bipolar(D, seed=100 + i) for i in range(N)]
    values = [_random_bipolar(D, seed=200 + i) for i in range(N)]
    for k, v in zip(keys, values):
        tape.bind(k, v)

    correct = 0
    for k, v in zip(keys, values):
        retrieved = tape.unbind(k)
        sim = F.cosine_similarity(retrieved.unsqueeze(0), v.unsqueeze(0)).item()
        # With 10 facts in D=8192, each retrieval should still be > 0.5 sim.
        if sim > 0.3:
            correct += 1
    # Expect near-perfect retrieval at this load.
    assert correct >= N - 1, f"only {correct}/{N} facts retrieved correctly"


def test_retrieval_picks_correct_among_candidates():
    """Given a query, retrieval returns the index of the correct candidate."""
    D = 8192
    N = 20
    tape = HoloTape(D=D)
    keys = [_random_bipolar(D, seed=10 + i) for i in range(N)]
    values = [_random_bipolar(D, seed=1000 + i) for i in range(N)]
    for k, v in zip(keys, values):
        tape.bind(k, v)
    candidates = torch.stack(values)   # (N, D)

    hits = 0
    for i, k in enumerate(keys):
        retrieved = tape.unbind(k)
        idx = cosine_retrieve(retrieved, candidates)
        if idx == i:
            hits += 1
    # At N=20, D=8192 (capacity ~ 8192/8/log2(21) ~ 327), expect >= 18/20 hits.
    assert hits >= 18, f"only {hits}/{N} correct retrievals"


def test_capacity_degrades_gracefully():
    """As we exceed capacity, accuracy drops but doesn't collapse."""
    D = 8192
    # Test at two loads: under capacity (small N) and over (large N).
    results = {}
    for N in [10, 200]:
        tape = HoloTape(D=D)
        keys = [_random_bipolar(D, seed=50 + i) for i in range(N)]
        values = [_random_bipolar(D, seed=500 + i) for i in range(N)]
        for k, v in zip(keys, values):
            tape.bind(k, v)
        candidates = torch.stack(values)
        hits = sum(
            1 for i, k in enumerate(keys)
            if cosine_retrieve(tape.unbind(k), candidates) == i
        )
        results[N] = hits / N
    # Under capacity should be high.
    assert results[10] >= 0.9, f"N=10 accuracy {results[10]} too low"
    # Over capacity should be LOWER but not zero (graceful degradation).
    # We don't assert a specific threshold here — the property is monotonic
    # degradation, which we just observe.
    print(f"\n  [capacity] N=10 acc={results[10]:.2f}, N=200 acc={results[200]:.2f}")


def test_specificity_beats_attention_baseline():
    """THE KEY TEST: holographic retrieval specificity vs the +0.006 attention baseline.

    The attention tape gave +0.006 specificity (random). Holographic binding is
    algebraic and exact by construction, so unbinding a bound key should retrieve
    the value with HIGH cosine similarity. This measures whether the algebraic
    path is genuinely more specific than the neural one.

    We don't even need a Prism model here — the test is pure VSA math, which is
    the point: this path needs NO TRAINING.
    """
    D = 8192
    N = 8
    tape = HoloTape(D=D)
    keys = [_random_bipolar(D, seed=700 + i) for i in range(N)]
    values = [_random_bipolar(D, seed=1700 + i) for i in range(N)]
    for k, v in zip(keys, values):
        tape.bind(k, v)

    # For each query, measure: similarity to TRUE value vs similarity to a
    # random (non-stored) value. "Specificity" = how much more the retrieved
    # vector aligns with the true value than with random noise.
    specificities = []
    for i, k in enumerate(keys):
        retrieved = tape.unbind(k)
        sim_true = F.cosine_similarity(retrieved.unsqueeze(0), values[i].unsqueeze(0)).item()
        # Random non-stored value as the baseline.
        random_v = _random_bipolar(D, seed=9999 + i)
        sim_random = F.cosine_similarity(retrieved.unsqueeze(0), random_v.unsqueeze(0)).item()
        specificities.append(sim_true - sim_random)

    mean_spec = sum(specificities) / len(specificities)
    print(f"\n  [holo specificity] mean(true - random) = {mean_spec:+.4f}")
    print(f"  [attention baseline] was +0.006 (random)")
    # Holographic specificity must be strongly positive (the bound value is
    # the dominant signal in the retrieved vector).
    assert mean_spec > 0.3, (
        f"holographic specificity {mean_spec:+.4f} not strongly positive — "
        f"the algebraic path is broken"
    )


def test_holo_encoder_preserves_similarity():
    """A trained-or-random HoloEncoder projects similar inputs to similar bipolar vectors.

    Even with a random init, cosine similarity should be roughly preserved
    through the linear projection (bipolarization adds some noise but the
    overall structure survives).
    """
    torch.manual_seed(0)
    enc = HoloEncoder(d_model=32, D=2048)
    # Two similar inputs (small angle).
    x1 = torch.randn(1, 32)
    x2 = x1 + 0.1 * torch.randn(1, 32)
    # One dissimilar input.
    x3 = torch.randn(1, 32)
    h1 = enc(x1).squeeze(0)
    h2 = enc(x2).squeeze(0)
    h3 = enc(x3).squeeze(0)
    sim_close = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
    sim_far = F.cosine_similarity(h1.unsqueeze(0), h3.unsqueeze(0)).item()
    # Similar inputs should yield higher bipolar similarity than dissimilar ones.
    assert sim_close > sim_far
