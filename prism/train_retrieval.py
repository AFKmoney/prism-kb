"""Phase 3: train the read head to read seeded KB content (CPU toy-scale).

Run::

    python -m prism.train_retrieval --steps 400

This trains a tiny PRISM on the retrieval task (answer is ONLY in the seeded
tape), then re-runs the specificity probe from tests/test_capture.py.

SUCCESS CRITERION: specificity correlation rises from the inert baseline
(+0.006) to > +0.2. If it does, COGLOOP's capture+reflect path is validated
end-to-end — the read head now retrieves seeded content semantically.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from prism.config import MemoryConfig, PrismConfig
from prism.memory import MemoryState
from prism.model import Prism
from tasks import retrieval


def train_read_head(steps: int = 400, lr: float = 3e-3, seed: int = 0, log_every: int = 50) -> Prism:
    """Train a tiny PRISM on the seeded-retrieval task."""
    torch.manual_seed(seed)
    cfg = PrismConfig(
        vocab_size=200,
        d_model=48,
        num_layers=3,
        num_rates=4,
        memory=MemoryConfig(d_mem=24, num_slots=8),
    )
    model = Prism(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n_pairs = 40   # 40 distinct (key,value) mappings to learn

    print(f"Phase 3: training read head on retrieval ({n_pairs} pairs, {steps} steps)")
    for step in range(steps):
        input_ids, targets, mask, keys = retrieval.generate_batch(
            batch_size=32, n_pairs=n_pairs, seq_len=8, device="cpu", generator=g,
        )
        seed_slots = retrieval.build_seed_slots(keys, cfg.memory.d_mem, "cpu", torch.float32)
        mem = MemoryState.from_knowledge(
            seed_slots, batch_size=32, config=cfg.memory, device="cpu", dtype=torch.float32,
        )
        out = model(input_ids, mem=mem)
        # CE only at the answer position (position 2).
        logits = out.logits[:, 2, :]   # (B, vocab)
        loss = F.cross_entropy(logits, targets[:, 2]) + out.aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == targets[:, 2]).float().mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.3f}")
    return model, cfg


def measure_specificity(model: Prism, cfg: PrismConfig, trials: int = 10) -> float:
    """Re-run the specificity probe. Returns mean correlation.

    Higher = read head retrieves seeded content semantically. Baseline (inert
    head) was +0.006. Target: > +0.2.
    """
    import statistics

    model.eval()
    embed = model.embed.weight
    corrs = []
    for trial in range(trials):
        torch.manual_seed(trial + 100)
        ids = torch.randint(2, cfg.vocab_size, (1, 16))
        # Use a random seed vector (probes generalization, not the trained keys).
        seed = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem)
        with torch.no_grad():
            logits_s = model(ids).logits[0, -1]
            mem = MemoryState.from_knowledge(seed, 1, cfg.memory, "cpu", torch.float32)
            logits_k = model(ids, mem=mem).logits[0, -1]
        delta = (logits_k - logits_s).detach()
        emb_trunc = embed[:, : cfg.memory.d_mem].detach()
        sims = F.cosine_similarity(emb_trunc, seed.mean(dim=0, keepdim=True)).squeeze()
        d_ = delta - delta.mean()
        s_ = sims - sims.mean()
        corrs.append(float((d_ * s_).sum() / (d_.norm() * s_.norm() + 1e-9)))
    return statistics.mean(corrs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 3: train + probe the read head")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args(argv)

    model, cfg = train_read_head(steps=cli.steps, lr=cli.lr, seed=cli.seed)
    print("\nRe-running specificity probe (baseline was +0.006, target > +0.2)...")
    spec = measure_specificity(model, cfg)
    print(f"\n  SPECIFICITY CORRELATION = {spec:+.4f}")
    if spec > 0.2:
        print("  ✅ BREAKTHROUGH: read head now retrieves seeded content semantically.")
        print("     COGLOOP's capture+reflect path is validated end-to-end.")
    elif spec > 0.05:
        print("  ⚠️ PARTIAL: improvement over baseline, but below the +0.2 target.")
        print("     The head is learning; more steps or capacity needed.")
    else:
        print("  ❌ NO IMPROVEMENT: the head did not learn to generalize to random seeds.")
        print("     The retrieval task trained it on structured seeds; generalization")
        print("     to arbitrary seeds needs more diverse training (cluster scale).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
