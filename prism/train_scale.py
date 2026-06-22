"""Scaled training configuration and dataset mixing for PRISM 1B.

This module provides:
  * ``PRESETS`` — ready-to-use PrismConfig instances for 1B (and smaller presets
    for smoke-testing the pipeline without 8×A100).
  * ``DatasetSpec`` / ``MIX_PRETRAIN`` / ``MIX_INSTRUCT`` — HF dataset specs and
    the mixing ratios used by the trainer. The mix is the consensus 2025
    pretraining recipe (see README sources): FineWeb-Edu backbone + OpenWebMath
    + The Stack v2 code.

Run ``python -m prism.train_scale --help`` to see all options.

The trainer is written to be *correct first*: it must run end-to-end on a single
GPU (or even CPU) at the ``tiny`` preset before being scaled up. All scale-up
features (DDP, bf16, grad accumulation, grad checkpointing, WandB) are opt-in
and degrade gracefully when unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from prism.config import MemoryConfig, PrismConfig


# ---------------------------------------------------------------------------
# Model size presets
# ---------------------------------------------------------------------------


def prism_1b(vocab_size: int = 50304) -> PrismConfig:
    """PRISM 1B params. Target ~1.0e9 parameters.

    Sized for 8×A100 80GB with bf16 + gradient checkpointing. The Multi-Rate
    Bus keeps activation memory linear in sequence length, so we can afford
    d_model=2048 with 24 layers at seq_len=4096.

    VRAM estimate (bf16, seq 4096, micro-batch 8, grad checkpoint):
      params       ~ 1.0 GB  (bf16) + 1.0 GB (fp32 master) = 2.0 GB
      grads        ~ 2.0 GB
      optimizer    ~ 8.0 GB  (AdamW fp32 states)
      activations  ~ 6 GB    (grad checkpoint + linear MRB)
      total/GPU    ~ 18 GB   (fits A100 40GB with headroom)
    """
    return PrismConfig(
        vocab_size=vocab_size,
        d_model=2048,
        num_layers=24,
        num_rates=8,                 # more rates for longer-range multi-scale
        mrb_delta0=0.3466,           # ln 2 / 2  -> half-life 2 steps for group 0
        mrb_max_delta=11.09,         # 16 * ln 2 -> half-life 16 steps for group K-1
        expert_types=("neural", "memory", "symbolic"),
        router_topk=2,               # top-2 routing for 1B (better capacity util)
        router_load_balance_weight=0.01,
        neural_hidden_mult=3,        # SwiGLU ~8/3 rule for hidden dim
        symbolic_num_primitives=6,
        memory=MemoryConfig(d_mem=1024, num_slots=64, num_read_heads=2),
        tie_embeddings=True,
        init_std=0.02,
    )


def prism_350m(vocab_size: int = 50304) -> PrismConfig:
    """PRISM 350M params. Fits a single A100 80GB for validation runs."""
    return PrismConfig(
        vocab_size=vocab_size,
        d_model=1024,
        num_layers=18,
        num_rates=8,               # must divide d_model (1024/8=128)
        expert_types=("neural", "memory", "symbolic"),
        router_topk=2,
        neural_hidden_mult=3,
        memory=MemoryConfig(d_mem=512, num_slots=32, num_read_heads=2),
        tie_embeddings=True,
    )


def prism_700m(vocab_size: int = 50304) -> PrismConfig:
    """PRISM 700M params — the intermediate stage for PCS (350M -> 700M -> 1B).

    Sits between prism_350m and prism_1b. Used by Progressive Capacity
    Stacking so the bulk of tokens train at a cheaper per-step FLOPs cost.
    """
    return PrismConfig(
        vocab_size=vocab_size,
        d_model=1536,
        num_layers=20,
        num_rates=8,
        expert_types=("neural", "memory", "symbolic"),
        router_topk=2,
        neural_hidden_mult=3,
        memory=MemoryConfig(d_mem=512, num_slots=64, num_read_heads=2),
        tie_embeddings=True,
    )


def prism_300m(vocab_size: int = 50304) -> PrismConfig:
    """PRISM 300M params — the "small brain, big reasoning" config.

    This is the lever-3 thesis: a 300M model that *beats a 1B* on reasoning
    tasks by leaning on the symbolic + memory experts instead of raw dense
    capacity. Knobs that compensate for fewer params:

      * More rate groups (num_rates=8) -> longer multi-scale temporal memory
        for free (the MRB is linear in sequence length, so extra rates cost
        almost nothing in activation memory).
      * Wider memory bus (num_slots=64, d_mem=512) -> more working memory,
        which a small model needs for multi-step reasoning.
      * router_topk=2 -> two expert kinds active per token, more capacity per
        token than top-1 at the same param count.
      * Smaller neural_hidden_mult=2 -> trim the dense expert; budget to
        symbolic + memory.

    Fits a single A100 80GB or 2x A100 40GB with bf16.
    """
    return PrismConfig(
        vocab_size=vocab_size,
        d_model=896,
        num_layers=24,
        num_rates=8,
        mrb_delta0=0.3466,           # ln2/2
        mrb_max_delta=16.64,         # 24*ln2 -> half-life 24 steps (long context)
        expert_types=("neural", "memory", "symbolic"),
        router_topk=2,
        router_load_balance_weight=0.01,
        neural_hidden_mult=2,        # trim dense expert; budget to symbolic+memory
        symbolic_num_primitives=6,
        memory=MemoryConfig(d_mem=512, num_slots=64, num_read_heads=2),
        tie_embeddings=True,
        init_std=0.02,
    )


def prism_tiny(vocab_size: int = 50304) -> PrismConfig:
    """Tiny preset (~5M params) for smoke-testing the *pipeline* on CPU/GPU.
    Use this to verify the full training script runs before committing GPU-days.
    """
    return PrismConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=4,
        num_rates=4,
        expert_types=("neural", "memory", "symbolic"),
        router_topk=1,
        neural_hidden_mult=2,
        memory=MemoryConfig(d_mem=64, num_slots=16),
        tie_embeddings=True,
    )


PRESETS = {"1b": prism_1b, "700m": prism_700m, "350m": prism_350m, "300m": prism_300m, "tiny": prism_tiny}


# ---------------------------------------------------------------------------
# Dataset mixing
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """A HuggingFace dataset spec for the trainer.

    Attributes:
        path: HF repo id (e.g. "HuggingFaceFW/fineweb-edu").
        config: config/subset name (None for default).
        split: split name (e.g. "train[:5%]").
        text_column: name of the column holding raw text (pretrain) or None
            for instruction datasets (which use prompt/completion or messages).
        weight: sampling weight in the mix (relative). The trainer normalizes.
        phase: which phase(s) this dataset belongs to.
        max_samples: optional cap on number of samples streamed (for debugging).
    """

    path: str
    config: str | None
    split: str
    text_column: str | None
    weight: float
    phase: str            # "pretrain" or "instruct"
    max_samples: int | None = None


# Pretraining mix — the 2025 consensus recipe for a general-capability model.
# Weights sum to 1.0. FineWeb-Edu is the backbone; OpenWebMath adds reasoning;
# The Stack v2 adds code. A small slice of long-form books (Gutenberg/OpenBooks)
# improves coherence.
MIX_PRETRAIN: list[DatasetSpec] = [
    DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",          # 10B-token sample; use "sample-100BT" for bigger
        split="train",
        text_column="text",
        weight=0.70,
        phase="pretrain",
    ),
    DatasetSpec(
        path="open-web-math/open-web-math",
        config=None,
        split="train",
        text_column="text",
        weight=0.10,
        phase="pretrain",
    ),
    DatasetSpec(
        path="bigcode/the-stack-v2-train",
        config=None,
        split="train",
        text_column="content",
        weight=0.15,
        phase="pretrain",
    ),
    DatasetSpec(
        path="togethercomputer/RedPajama-Data-1T-Sample",
        config=None,
        split="train",
        text_column="text",            # includes books / arxiv / wikipedia slices
        weight=0.05,
        phase="pretrain",
    ),
]

# Instruction-tuning mix — general instruction following + reasoning + code.
MIX_INSTRUCT: list[DatasetSpec] = [
    DatasetSpec(
        path="Open-Orca/OpenOrca",
        config=None,
        split="train",
        text_column=None,              # uses question/response columns
        weight=0.50,
        phase="instruct",
    ),
    DatasetSpec(
        path="garage-bAInd/Open-Platypus",
        config=None,
        split="train",
        text_column=None,
        weight=0.20,
        phase="instruct",
    ),
    DatasetSpec(
        path="openchat/openchat_3.5",
        config=None,
        split="train_sft",
        text_column=None,
        weight=0.15,
        phase="instruct",
    ),
    DatasetSpec(
        path="teknium/GPT4-LLM-Cleaned",
        config=None,
        split="train",
        text_column=None,
        weight=0.15,
        phase="instruct",
    ),
]


def get_mix(phase: str) -> list[DatasetSpec]:
    """Return the dataset mix for a given phase."""
    if phase == "pretrain":
        return MIX_PRETRAIN
    if phase == "instruct":
        return MIX_INSTRUCT
    raise ValueError(f"unknown phase: {phase!r} (expected 'pretrain' or 'instruct')")


def mix_summary(mix: list[DatasetSpec]) -> str:
    """Human-readable mix description for logging."""
    total = sum(d.weight for d in mix)
    lines = []
    for d in mix:
        pct = 100 * d.weight / total
        cfg = f"[{d.config}]" if d.config else ""
        lines.append(f"  {pct:5.1f}%  {d.path}{cfg}  (split={d.split})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training hyperparameters by phase
# ---------------------------------------------------------------------------


@dataclass
class TrainArgs:
    """Training hyperparameters, decoupled from model config."""

    # --- scale ---
    preset: str = "1b"                  # 1b | 350m | 300m | tiny
    phase: str = "pretrain"             # pretrain | instruct

    # --- levers (opt-in, each measurable in isolation) ---
    modular_phase: str | None = None    # None | neural | memory | symbolic | assemble
    curriculum: bool = False            # 3-phase dataset re-weighting (neural→memory→symbolic)
    token_recycling: bool = False       # up-weight hard tokens (enable >=1B tokens)
    recycling_strength: float = 1.0     # 0=off, 1=full inverse-freq weighting

    # --- data ---
    seq_len: int = 4096
    global_batch_size: int = 256        # effective batch (across all GPUs + accum)
    micro_batch_size: int = 8           # per-GPU batch before grad accumulation

    # --- optimization ---
    steps: int = 100_000
    warmup_steps: int = 2000
    lr: float = 3e-4                    # peak LR for 1B
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95                 # 0.95 is the standard for LLM pretraining
    scheduler: str = "cosine"           # cosine | wsd (warmup-stable-decay)

    # --- precision / memory ---
    dtype: str = "bf16"                 # bf16 | fp16 | fp32
    grad_checkpoint: bool = True

    # --- distributed ---
    ddp: bool = True                    # auto-disabled if world_size == 1

    # --- checkpointing / logging ---
    out_dir: str = "runs/prism"
    save_every: int = 2000
    log_every: int = 20
    eval_every: int = 2000
    eval_steps: int = 50
    init_from: str | None = None        # resume from a checkpoint dir
    wandb: bool = False
    wandb_project: str = "prism"

    # --- tokenizer ---
    tokenizer: str = "gpt2"            # HF tokenizer name (vocab drives model config)

    # --- runtime flags (set by CLI, not config) ---
    force_cpu: bool = False
    smoke_datasets: bool = False

    # --- reproducibility ---
    seed: int = 42

    @property
    def grad_accum_steps(self) -> int:
        """Number of micro-batches to accumulate per optimizer step."""
        # world_size is read at runtime; assume 1 if not in DDP context.
        import os

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        per_step = self.micro_batch_size * world_size
        return max(1, self.global_batch_size // per_step)

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.seq_len

    def estimate_train_tokens(self) -> int:
        return self.tokens_per_step * self.steps

    def summary(self) -> str:
        return (
            f"preset={self.preset} phase={self.phase}\n"
            f"  seq_len={self.seq_len} global_batch={self.global_batch_size} "
            f"micro_batch={self.micro_batch_size} grad_accum={self.grad_accum_steps}\n"
            f"  lr={self.lr} (warmup {self.warmup_steps}, decay to {self.min_lr})\n"
            f"  steps={self.steps} -> {self.estimate_train_tokens()/1e9:.1f}G tokens\n"
            f"  dtype={self.dtype} grad_checkpoint={self.grad_checkpoint} ddp={self.ddp}"
        )
