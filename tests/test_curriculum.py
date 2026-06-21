"""Tests for curriculum scheduling and token recycling (lever 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism.curriculum import CurriculumSchedule, TokenRecycler


# --- Curriculum ----------------------------------------------------------


def test_curriculum_phases_progress():
    sched = CurriculumSchedule(total_steps=1000)
    assert sched.phase_for_step(0) == "neural"
    assert sched.phase_for_step(100) == "neural"
    assert sched.phase_for_step(400) == "memory"
    assert sched.phase_for_step(900) == "symbolic"


def test_curriculum_weights_change_over_time():
    """Weights at step 100 != weights at step 900 (mix shifts)."""
    sched = CurriculumSchedule(total_steps=1000)
    base = [0.7, 0.1, 0.15, 0.05]
    focuses = ["neural", "symbolic", "symbolic", "memory"]
    w_early = sched.weights_for_step(100, base, focuses)
    w_late = sched.weights_for_step(900, base, focuses)
    assert w_early != w_late


def test_curriculum_focus_dataset_upweighted_in_its_phase():
    sched = CurriculumSchedule(total_steps=1000)
    base = [0.7, 0.1]
    focuses = ["neural", "symbolic"]
    w = sched.weights_for_step(100, base, focuses)   # neural phase
    assert w[0] > base[0]   # neural up-weighted in neural phase
    assert w[1] == base[1]  # symbolic unchanged


def test_curriculum_weights_sum_changes():
    """Total weight is not preserved (the focus dataset is amplified, not swapped)."""
    sched = CurriculumSchedule(total_steps=1000)
    base = [0.7, 0.3]
    focuses = ["neural", "symbolic"]
    w = sched.weights_for_step(100, base, focuses)
    assert sum(w) > sum(base)   # amplification increases total


# --- Token recycling -----------------------------------------------------


def test_recycler_upweights_hard_tokens():
    recycler = TokenRecycler(num_buckets=8, strength=1.0, device="cpu")
    # Mostly easy tokens + one hard.
    per_tok = torch.tensor([[0.1, 0.1, 0.1, 0.1, 5.0, 0.1]])
    recycler.update(per_tok)
    w = recycler.token_weights(per_tok)
    assert w.flatten()[4].item() > w.flatten()[0].item()


def test_recycler_zero_weights_masked_tokens():
    recycler = TokenRecycler(num_buckets=8, strength=1.0, device="cpu")
    per_tok = torch.tensor([[0.0, 0.0, 2.0, 0.0]])   # zeros = masked
    recycler.update(per_tok)
    w = recycler.token_weights(per_tok)
    assert w.flatten()[0].item() == 0.0
    assert w.flatten()[2].item() > 0.0


def test_recycler_strength_zero_is_uniform():
    """With strength=0, weights should all be ~1 (no recycling)."""
    recycler = TokenRecycler(num_buckets=8, strength=0.0, device="cpu")
    per_tok = torch.tensor([[0.1, 0.2, 3.0, 0.5]])
    recycler.update(per_tok)
    w = recycler.token_weights(per_tok)
    # All non-masked weights should be ~1.
    non_zero = w[w > 0]
    assert torch.allclose(non_zero, torch.ones_like(non_zero), atol=1e-3)


def test_recycler_update_does_not_crash_on_all_masked():
    recycler = TokenRecycler(num_buckets=8, strength=1.0, device="cpu")
    per_tok = torch.zeros(2, 5)   # all masked
    recycler.update(per_tok)      # should be a no-op, not crash
    w = recycler.token_weights(per_tok)
    assert torch.all(w == 0)
