"""Tests for modular pretraining (lever 1)."""

from __future__ import annotations

import os
import tempfile

import torch

from prism.config import MemoryConfig, PrismConfig
from prism.memory import MemoryState
from prism.model import Prism
from prism.modular import assemble_experts, modular_config, MIX_MODULAR


def _cfg(**kw) -> PrismConfig:
    base = dict(vocab_size=32, d_model=16, num_layers=2, num_rates=4,
                memory=MemoryConfig(d_mem=8, num_slots=4))
    base.update(kw)
    return PrismConfig(**base)


def test_modular_config_single_expert():
    """A modular config has exactly one expert kind and a degenerate router."""
    base = _cfg()
    for kind in ("neural", "symbolic", "memory"):
        cfg = modular_config(base, kind)
        assert cfg.expert_types == (kind,)
        assert cfg.router_topk == 1
        m = Prism(cfg)
        # Each block has exactly 1 expert of the right kind.
        for blk in m.blocks:
            experts = blk.router.experts
            assert len(experts) == 1
            assert experts[0].expert_type == kind


def test_modular_config_memory_needs_two_layers():
    """Memory modular config must force >=2 layers so writes get read."""
    base = _cfg(num_layers=1)
    cfg = modular_config(base, "memory")
    assert cfg.num_layers >= 2


def test_assemble_preserves_neural_weights():
    """Assembling a neural checkpoint copies its expert weights verbatim."""
    base = _cfg()
    full = Prism(base)

    # Create a single-expert source with recognizable weights.
    src = Prism(modular_config(base, "neural"))
    with torch.no_grad():
        src.blocks[0].router.experts[0].w_down.weight.fill_(0.777)

    tmp = tempfile.mkdtemp()
    neu_dir = os.path.join(tmp, "neu", "ckpt-1")
    os.makedirs(neu_dir)
    torch.save(src.state_dict(), os.path.join(neu_dir, "pytorch_model.bin"))

    # Also need a symbolic checkpoint for assemble_experts (it takes both).
    sym_src = Prism(modular_config(base, "symbolic"))
    sym_dir = os.path.join(tmp, "sym", "ckpt-1")
    os.makedirs(sym_dir)
    torch.save(sym_src.state_dict(), os.path.join(sym_dir, "pytorch_model.bin"))

    assemble_experts(full, neu_dir, sym_dir, memory_ckpt=None)

    # The neural expert in the full model is at index 0.
    neu_w = full.blocks[0].router.experts[0].w_down.weight
    assert abs(neu_w.mean().item() - 0.777) < 1e-4, "neural weights not copied"


def test_assemble_preserves_symbolic_weights():
    """Assembling a symbolic checkpoint copies its expert weights verbatim."""
    base = _cfg()
    full = Prism(base)
    sym_src = Prism(modular_config(base, "symbolic"))
    with torch.no_grad():
        sym_src.blocks[0].router.experts[0].lib.out_proj.weight.fill_(0.333)
    neu_src = Prism(modular_config(base, "neural"))

    tmp = tempfile.mkdtemp()
    for name, src in [("neu", neu_src), ("sym", sym_src)]:
        d = os.path.join(tmp, name, "ckpt-1")
        os.makedirs(d)
        torch.save(src.state_dict(), os.path.join(d, "pytorch_model.bin"))

    assemble_experts(full, os.path.join(tmp, "neu", "ckpt-1"), os.path.join(tmp, "sym", "ckpt-1"))
    # Symbolic is at index 2 in the full config.
    sym_w = full.blocks[0].router.experts[2].lib.out_proj.weight
    assert abs(sym_w.mean().item() - 0.333) < 1e-4, "symbolic weights not copied"


def test_modular_mixes_present():
    assert set(MIX_MODULAR.keys()) == {"neural", "memory", "symbolic"}
    for kind, mix in MIX_MODULAR.items():
        assert len(mix) >= 1
        for spec in mix:
            assert spec.weight > 0
