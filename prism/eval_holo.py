"""PRISM-Holo evaluation harness.

Reproducible retrieval-accuracy benchmarks for PRISM-Holo models. Runs the
three core probes and reports metrics in a standard format.

Probes:
  1. VSA retrieval accuracy: bind N facts, retrieve each by key.
     (pure algebra, no model — the +0.355 result.)
  2. Integrated specificity: seed the model's tape, measure correlation
     between logit shift and seed alignment.
  3. Capacity curve: accuracy as N grows from 10 to 1000.

Usage::
    python -m prism.eval_holo
    python -m prism.eval_holo --checkpoint runs/prism-holo/ckpt-50000 --probe all
    python -m prism.eval_holo --probe capacity --D 4096

Outputs JSON to stdout and optionally to --out.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from prism.config import MemoryConfig, PrismConfig
from prism.holo import HoloEncoder, HoloTape, cosine_retrieve
from prism.memory import MemoryState
from prism.model import Prism


def _random_bipolar(D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.randn(D, generator=g) >= 0, torch.ones(D), -torch.ones(D))


def probe_vsa_retrieval(D: int = 8192, N_values: list[int] | None = None) -> dict:
    """Probe 1: VSA retrieval accuracy across fact counts."""
    N_values = N_values or [1, 10, 50, 100, 200, 500]
    results = {}
    for N in N_values:
        if N > D // 4:
            continue
        tape = HoloTape(D=D)
        keys = [_random_bipolar(D, seed=100 + i) for i in range(N)]
        values = [_random_bipolar(D, seed=1000 + i) for i in range(N)]
        for k, v in zip(keys, values):
            tape.bind(k, v)
        candidates = torch.stack(values)
        hits = sum(
            1 for i, k in enumerate(keys)
            if cosine_retrieve(tape.unbind(k), candidates) == i
        )
        results[N] = hits / N
    return {"probe": "vsa_retrieval", "D": D, "accuracy_by_N": results}


def probe_integrated_specificity(cfg: PrismConfig, n_trials: int = 10) -> dict:
    """Probe 2: integrated specificity (logit shift vs seed alignment)."""
    torch.manual_seed(0)
    model = Prism(cfg).eval()
    ids = torch.randint(2, cfg.vocab_size, (1, 8))

    rhos = []
    for t in range(n_trials):
        torch.manual_seed(t + 500)
        seed = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem)
        with torch.no_grad():
            l0 = model(ids).logits[0, -1]
            mem = MemoryState.from_knowledge(seed, 1, cfg.memory, "cpu", torch.float32)
            l1 = model(ids, mem=mem).logits[0, -1]
        delta = (l1 - l0).detach()
        emb = model.embed.weight.detach()
        emb_red = emb[:, : cfg.memory.d_mem]
        s_red = seed.mean(dim=0)
        sims = F.cosine_similarity(emb_red, s_red.unsqueeze(0)).squeeze()
        d_ = delta - delta.mean()
        s_ = sims - sims.mean()
        rhos.append(float((d_ * s_).sum() / (d_.norm() * s_.norm() + 1e-9)))
    return {
        "probe": "integrated_specificity",
        "mean": statistics.mean(rhos),
        "stdev": statistics.stdev(rhos) if len(rhos) > 1 else 0.0,
        "n_trials": n_trials,
        "holo_mode": cfg.holo_mode,
    }


def probe_capacity_curve(D: int = 8192, max_N: int = 1000, step: int = 50) -> dict:
    """Probe 3: capacity curve — accuracy vs number of facts stored."""
    N_values = list(range(10, max_N + 1, step))
    results = {}
    for N in N_values:
        tape = HoloTape(D=D)
        keys = [_random_bipolar(D, seed=200 + i) for i in range(N)]
        values = [_random_bipolar(D, seed=2000 + i) for i in range(N)]
        for k, v in zip(keys, values):
            tape.bind(k, v)
        candidates = torch.stack(values)
        # Sample 50 queries (full N is slow for retrieval check).
        sample_idx = torch.linspace(0, N - 1, min(50, N)).int().tolist()
        hits = sum(
            1 for i in sample_idx
            if cosine_retrieve(tape.unbind(keys[i]), candidates) == i
        )
        results[N] = hits / len(sample_idx)
    return {"probe": "capacity_curve", "D": D, "accuracy_by_N": results}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PRISM-Holo evaluation harness")
    p.add_argument("--probe", choices=["vsa", "specificity", "capacity", "all"], default="all")
    p.add_argument("--D", type=int, default=8192, help="holographic dimension for VSA probes")
    p.add_argument("--preset", choices=["tiny", "300m", "350m", "700m", "1b"], default="tiny")
    p.add_argument("--holo", action="store_true", help="use holo_mode for the specificity probe")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--out", type=str, default=None, help="write JSON results to this path")
    cli = p.parse_args(argv)

    results = {}

    if cli.probe in ("vsa", "all"):
        print("Running VSA retrieval probe...", file=sys.stderr)
        results["vsa_retrieval"] = probe_vsa_retrieval(D=cli.D)

    if cli.probe in ("specificity", "all"):
        print("Running integrated specificity probe...", file=sys.stderr)
        from prism.train_scale import PRESETS
        cfg = PRESETS[cli.preset](vocab_size=128)
        # Override memory for reasonable VSA D on small models.
        if cli.preset == "tiny":
            cfg = PrismConfig(
                vocab_size=128, d_model=64, num_layers=3, num_rates=4,
                memory=MemoryConfig(d_mem=32, num_slots=64), holo_mode=cli.holo,
            )
        results["integrated_specificity"] = probe_integrated_specificity(cfg, cli.n_trials)

    if cli.probe in ("capacity", "all"):
        print("Running capacity curve probe...", file=sys.stderr)
        results["capacity_curve"] = probe_capacity_curve(D=cli.D, max_N=min(1000, cli.D // 4))

    out = json.dumps(results, indent=2)
    print(out)
    if cli.out:
        Path(cli.out).write_text(out, encoding="utf-8")
        print(f"\nResults written to {cli.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
