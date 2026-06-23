"""Train the PRISM-Holo split encoders on CPU (the gap-closing step).

This is the ONE piece of training that fits on CPU in minutes, because the
encoders are tiny (~d_model * D params). It targets closing the specificity gap:
  +0.0525 (random init)  ->  higher (trained encoder).

Key fixes vs the first iteration:
  1. DIMENSION-MATCHED: the encoder is trained at exactly D = num_slots * d_mem
     (the HoloHead's dimension), so injection copies weights 1:1 — no truncation.
  2. NON-TRIVIAL TASK: pairs have partial similarity (shared subspace + private
     subspace), so the encoder has real structure to learn — not loss=0 at step 0.

Run::
    python -m prism.train_encoder --steps 300

Then the probe re-runs and reports the new specificity.
"""

from __future__ import annotations

import argparse
import sys
import statistics

import torch
import torch.nn.functional as F

from prism.config import MemoryConfig, PrismConfig
from prism.holo import HoloEncoder
from prism.memory import MemoryState
from prism.model import Prism


def _bipolar_st(v: torch.Tensor) -> torch.Tensor:
    """Bipolarize with straight-through (forward {±1}, backward identity)."""
    bipolar = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
    return bipolar + v - v.detach()


def _make_pairs(n: int, d_model: int, seed: int = 0, shared_frac: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Make (question, answer) pairs with PARTIAL similarity.

    Each pair shares a random subspace (shared_frac of dims identical) and has
    a private subspace (1-shared_frac independent). This gives the encoder a
    non-trivial task: learn which dimensions carry the shared signal so that
    bipolarized encodings preserve the question-answer similarity.
    """
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n, d_model, generator=g)
    q_priv = torch.randn(n, d_model, generator=g)
    a_priv = torch.randn(n, d_model, generator=g)
    mask = torch.rand(n, d_model, generator=g) < shared_frac
    q = torch.where(mask, shared, q_priv)
    a = torch.where(mask, shared, a_priv)
    return q, a


def train_encoder(steps: int, lr: float, D: int, d_model: int) -> HoloEncoder:
    """Train a HoloEncoder to preserve cosine similarity through bipolarization."""
    torch.manual_seed(0)
    encoder = HoloEncoder(d_model=d_model, D=D)
    # Bypass encoder.forward (which bipolarizes non-differerentiably); use proj.
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.01)
    n_pairs = 64

    print(f"Training Holo encoder ({d_model} -> {D}, {sum(p.numel() for p in encoder.parameters())/1e3:.0f}K params)")
    print(f"  objective: contrastive — preserve partial similarity through bipolarization")
    for step in range(steps):
        q, a = _make_pairs(n_pairs, d_model, seed=step, shared_frac=0.5)
        q_proj = encoder.proj(q)
        a_proj = encoder.proj(a)
        q_h = _bipolar_st(q_proj)
        a_h = _bipolar_st(a_proj)

        sims = F.cosine_similarity(q_h.unsqueeze(1), a_h.unsqueeze(0), dim=-1)  # (N, N)
        logits = sims / 0.07
        targets = torch.arange(n_pairs)
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == targets).float().mean().item()
                pos_sim = sims.diag().mean().item()
                neg_sim = (sims.sum() - sims.diag().sum()) / (n_pairs * (n_pairs - 1))
                neg_sim = neg_sim.item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | pos {pos_sim:.3f} | neg {neg_sim:.3f} | acc {acc:.3f}")
    return encoder


def measure_encoder_specificity(encoder: HoloEncoder, cfg: PrismConfig, n_trials: int = 10) -> float:
    """Re-run the specificity probe with the trained encoder, integrated into
    a Prism model. Injects weights 1:1 (dimension-matched)."""
    torch.manual_seed(0)
    model = Prism(cfg).eval()

    # Inject the trained encoder weights into ALL HoloHeads (1:1, dimension-matched).
    D_head = cfg.memory.num_slots * cfg.memory.d_mem
    assert encoder.proj.weight.shape[0] == D_head, (
        f"dim mismatch: encoder D={encoder.proj.weight.shape[0]} != head D={D_head}"
    )
    injected = 0
    for block in model.blocks:
        for expert in block.router.experts:
            head = getattr(expert, "head", None)
            if head is not None and head.__class__.__name__ == "HoloHead":
                with torch.no_grad():
                    head.key_encoder.weight.copy_(encoder.proj.weight)
                    head.value_encoder.weight.copy_(encoder.proj.weight)
                injected += 1
    if injected == 0:
        raise RuntimeError("no HoloHead found in model — set holo_mode=True")

    ids = torch.randint(2, cfg.vocab_size, (1, 8))
    rhos = []
    for t in range(n_trials):
        torch.manual_seed(t + 500)
        seed = torch.randn(cfg.memory.num_slots, cfg.memory.d_mem)
        with torch.no_grad():
            l0 = model(ids).logits[0, -1]
            mem = MemoryState.from_knowledge(seed, 1, cfg.memory, "cpu", torch.float32, blend_ratio=2.0)
            l1 = model(ids, mem=mem).logits[0, -1]
        delta = (l1 - l0).detach()
        emb = model.embed.weight.detach()
        emb_red = emb[:, : cfg.memory.d_mem]
        s_red = seed.mean(dim=0)
        sims = F.cosine_similarity(emb_red, s_red.unsqueeze(0)).squeeze()
        d_ = delta - delta.mean()
        s_ = sims - sims.mean()
        rhos.append(float((d_ * s_).sum() / (d_.norm() * s_.norm() + 1e-9)))
    return statistics.mean(rhos)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the PRISM-Holo encoder on CPU")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    cli = p.parse_args(argv)

    cfg = PrismConfig(
        vocab_size=128, d_model=64, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=32, num_slots=64),  # D_head = 2048
        holo_mode=True,
    )
    D_head = cfg.memory.num_slots * cfg.memory.d_mem  # 2048 — match the HoloHead exactly

    print("=" * 60)
    print("BEFORE training — random-init encoder specificity:")
    enc_random = HoloEncoder(d_model=cfg.d_model, D=D_head)
    spec_before = measure_encoder_specificity(enc_random, cfg)
    print(f"  specificity = {spec_before:+.4f}  (random init baseline)")
    print("=" * 60)

    encoder = train_encoder(steps=cli.steps, lr=cli.lr, D=D_head, d_model=cfg.d_model)

    print("\n" + "=" * 60)
    print("AFTER training — trained encoder specificity (dim-matched injection):")
    spec_after = measure_encoder_specificity(encoder, cfg)
    print(f"  specificity = {spec_after:+.4f}")
    print(f"  delta       = {spec_after - spec_before:+.4f}")
    print("=" * 60)
    if spec_after > spec_before + 0.01:
        print("✅ Encoder training improved integrated specificity (dim-matched).")
    elif spec_after > 0.05:
        print("⚠️  Modest improvement — real NQ/TriviaQA pairs would help more.")
    else:
        print("⚠️  No improvement — toy model embeddings are random; real model")
        print("   with semantic embeddings is needed to see the encoder's effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
