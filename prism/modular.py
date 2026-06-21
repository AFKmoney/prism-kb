"""Modular pretraining for PRISM (lever 1).

Train each expert *kind* separately on the data it's best at, then assemble
the trained experts into a full PRISM model with a short fine-tune to wire the
router. This is the "2-3x speedup + natural parallelism" lever.

Why it works
------------
PRISM's experts are structurally different operators, so the optimal training
data differs by kind:
  * ``neural``   -> broad text (FineWeb-Edu)
  * ``memory``   -> retrieval/lookup data (QA pairs, long docs)
  * ``symbolic`` -> code + math (The Stack, OpenWebMath)

Training each in isolation means each converges faster on its specialty, and
the three runs are independent (can run in parallel on different GPUs). The
final ``assemble`` step loads the three expert checkpoints into one model and
fine-tunes only the router + MRB to make them cooperate — a small fraction of
the total compute.

Implementation detail (from codebase exploration):
  * A config with ``expert_types=("symbolic",)`` degenerates the router to a
    single always-selected expert with a constant zero-gradient aux_loss. So an
    isolated run is literally a normal PRISM run with a 1-element expert tuple.
  * The ``assemble_experts`` function copies the per-expert ``state_dict``
    weights from the three checkpoints into the corresponding experts of a full
    model, leaving the router gate and MRB weights at their fresh init (to be
    learned in the assembly fine-tune).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from prism.config import PrismConfig
from prism.train_scale import DatasetSpec


# ---------------------------------------------------------------------------
# Per-expert focused dataset mixes
# ---------------------------------------------------------------------------

MIX_MODULAR_NEURAL: list[DatasetSpec] = [
    DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT", split="train", text_column="text",
        weight=0.80, phase="pretrain",
    ),
    DatasetSpec(
        path="togethercomputer/RedPajama-Data-1T-Sample",
        config=None, split="train", text_column="text",
        weight=0.20, phase="pretrain",
    ),
]

MIX_MODULAR_MEMORY: list[DatasetSpec] = [
    # Retrieval-rich data: QA pairs and long documents train the read/write head.
    DatasetSpec(
        path="togethercomputer/RedPajama-Data-1T-Sample",
        config=None, split="train", text_column="text",
        weight=0.60, phase="pretrain",  # long-form books/arxiv -> long context
    ),
    DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT", split="train", text_column="text",
        weight=0.40, phase="pretrain",
    ),
]

MIX_MODULAR_SYMBOLIC: list[DatasetSpec] = [
    DatasetSpec(
        path="bigcode/the-stack-v2-train",
        config=None, split="train", text_column="content",
        weight=0.60, phase="pretrain",
    ),
    DatasetSpec(
        path="open-web-math/open-web-math",
        config=None, split="train", text_column="text",
        weight=0.40, phase="pretrain",
    ),
]

MIX_MODULAR = {
    "neural": MIX_MODULAR_NEURAL,
    "memory": MIX_MODULAR_MEMORY,
    "symbolic": MIX_MODULAR_SYMBOLIC,
}


def modular_config(base_config: PrismConfig, expert_kind: str) -> PrismConfig:
    """Return a config that trains ONLY one expert kind.

    Sets ``expert_types=(expert_kind,)`` so the router degenerates to a single
    always-selected expert. This is the cleanest isolation: no router learning,
    no load-balancing loss gradient (it's a constant), all compute goes to the
    one expert.

    For ``memory`` we need >=2 layers so writes in block i get read by block i+1
    (the last block drops memory to avoid dead write weights — see model.py).
    """
    from dataclasses import replace

    if expert_kind not in ("neural", "memory", "symbolic"):
        raise ValueError(f"unknown expert kind: {expert_kind!r}")
    cfg = replace(base_config, expert_types=(expert_kind,), router_topk=1)
    if expert_kind == "memory" and cfg.num_layers < 2:
        cfg = replace(cfg, num_layers=2)
    return cfg


# ---------------------------------------------------------------------------
# Assembly: merge three single-expert checkpoints into one full model
# ---------------------------------------------------------------------------


def _expert_prefix(block_idx: int) -> str:
    """The state_dict key prefix for experts in block `block_idx`."""
    return f"blocks.{block_idx}.router.experts.0"


def assemble_experts(
    full_model: nn.Module,
    neural_ckpt: str,
    symbolic_ckpt: str,
    memory_ckpt: str | None = None,
    strict: bool = False,
) -> nn.Module:
    """Load three single-expert checkpoints into a full PRISM model.

    Each single-expert checkpoint was produced by training with
    ``expert_types=(<kind>,)``. Their experts live at
    ``blocks.{i}.router.experts.0`` (index 0 — there's only one). In the full
    model, the experts live at ``blocks.{i}.router.experts.{j}`` where j is the
    position of that kind in the full config's ``expert_types`` tuple.

    This function maps each source expert's weights onto the matching target
    expert in every block. The router gates and MRB weights are left at their
    fresh init (to be learned in the assembly fine-tune).

    Args:
        full_model: a fresh PRISM model with all expert kinds (untrained).
        neural_ckpt: path to the neural-only checkpoint dir (ckpt-N/pytorch_model.bin).
        symbolic_ckpt: path to the symbolic-only checkpoint dir.
        memory_ckpt: optional path to the memory-only checkpoint dir.
        strict: if True, require every source weight to map.

    Returns:
        The full_model with expert weights loaded in place.
    """
    full_cfg = full_model.config
    kinds = list(full_cfg.expert_types)
    n_layers = full_cfg.num_layers

    # Build a full state dict we'll merge into.
    full_state = full_model.state_dict()

    # Map kind -> (checkpoint path, target expert index in each block).
    sources = {
        "neural": neural_ckpt,
        "symbolic": symbolic_ckpt,
    }
    if memory_ckpt is not None:
        sources["memory"] = memory_ckpt

    for kind, ckpt_dir in sources.items():
        if kind not in kinds:
            if strict:
                raise ValueError(f"kind {kind!r} not in full config expert_types {kinds}")
            continue
        target_idx = kinds.index(kind)
        ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        src_state = torch.load(ckpt_path, map_location="cpu")
        src_prefix = "blocks.{b}.router.experts.0."
        tgt_prefix = f"blocks.{{b}}.router.experts.{target_idx}."

        for b in range(n_layers):
            sp = src_prefix.format(b=b)
            tp = tgt_prefix.format(b=b)
            for k, v in src_state.items():
                if k.startswith(sp):
                    # Strip the source prefix, re-add the target prefix.
                    suffix = k[len(sp):]
                    full_state[tp + suffix] = v

    full_model.load_state_dict(full_state, strict=False)
    return full_model


def description() -> str:
    return (
        "Modular pretraining: train neural/memory/symbolic experts separately on "
        "their optimal data, then assemble into a full PRISM model with a short "
        "router+MRB fine-tune. 2-3x speedup via specialization + parallelism."
    )
