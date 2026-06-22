"""Tests for Progressive Capacity Stacking (PCS)."""

from __future__ import annotations

import torch

from prism.config import MemoryConfig, PrismConfig
from prism.model import Prism
from prism.pcs import (
    DEFAULT_SCHEDULE,
    StageSpec,
    grow_model,
    resolve_schedule,
    schedule_summary,
)


def _tiny_cfg(d_model=32, layers=2, **kw):
    base = dict(
        vocab_size=64, d_model=d_model, num_layers=layers, num_rates=4,
        memory=MemoryConfig(d_mem=16, num_slots=8),
    )
    base.update(kw)
    return PrismConfig(**base)


def test_resolve_schedule_sums_to_total():
    sched = resolve_schedule(total_steps=1000)
    assert sum(s.steps for s in sched) == 1000
    # First stage gets the most steps (cheapest per step).
    assert sched[0].steps >= sched[-1].steps


def test_resolve_schedule_custom():
    custom = [StageSpec(preset="tiny", token_fraction=0.5), StageSpec(preset="350m", token_fraction=0.5)]
    sched = resolve_schedule(100, custom)
    assert sched[0].steps + sched[1].steps == 100
    assert sched[0].steps == 50


def test_grow_model_preserves_matching_weights():
    """Weights that exist in both old and new configs are transferred."""
    torch.manual_seed(0)
    old_cfg = _tiny_cfg(d_model=32, layers=2)
    new_cfg = _tiny_cfg(d_model=48, layers=3)
    old_model = Prism(old_cfg)
    # Corrupt a specific weight to verify it's transferred.
    with torch.no_grad():
        old_model.blocks[0].mrb.in_proj.weight.fill_(0.777)

    new_model = grow_model(old_model, new_cfg)

    # The first 32 output dims of the grown in_proj should be 0.777 (transferred),
    # the remaining 16 should be 0 (zero-init padding).
    grown_w = new_model.blocks[0].mrb.in_proj.weight  # (48, 48)
    assert torch.allclose(grown_w[:32, :32], torch.full((32, 32), 0.777), atol=1e-5)
    assert torch.allclose(grown_w[32:, :], torch.zeros(grown_w.shape[0] - 32, grown_w.shape[1]), atol=1e-6)


def test_grow_model_extra_layer_added():
    """Growing from 2 to 3 layers produces a 3-layer model."""
    torch.manual_seed(0)
    old = Prism(_tiny_cfg(layers=2))
    new = grow_model(old, _tiny_cfg(layers=3))
    assert len(new.blocks) == 3


def test_grow_model_forward_works():
    """The grown model produces valid output."""
    torch.manual_seed(0)
    old = Prism(_tiny_cfg(d_model=32, layers=2))
    new = grow_model(old, _tiny_cfg(d_model=48, layers=3))
    ids = torch.randint(0, 64, (2, 8))
    out = new(ids)
    assert out.logits.shape == (2, 8, 64)
    assert not torch.isnan(out.logits).any()


def test_grow_model_more_params():
    """The grown model has more parameters than the original."""
    torch.manual_seed(0)
    old = Prism(_tiny_cfg(d_model=32, layers=2))
    new = grow_model(old, _tiny_cfg(d_model=48, layers=3))
    n_old = sum(p.numel() for p in old.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    assert n_new > n_old


def test_default_schedule_has_three_stages():
    assert len(DEFAULT_SCHEDULE) == 3
    assert DEFAULT_SCHEDULE[0].preset == "350m"
    assert DEFAULT_SCHEDULE[1].preset == "700m"
    assert DEFAULT_SCHEDULE[2].preset == "1b"
    assert abs(sum(s.token_fraction for s in DEFAULT_SCHEDULE) - 1.0) < 1e-6


def test_schedule_summary_runs():
    sched = resolve_schedule(50000)
    s = schedule_summary(sched)
    assert "350m" in s and "1b" in s and "Stage" in s
