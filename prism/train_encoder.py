"""Train the PRISM-Holo split encoders on CPU (the gap-closing step).

This is the ONE piece of training that fits on CPU in minutes, because the
encoders are tiny (~d_model * D params). It closes the specificity gap:
  +0.0525 (random init)  ->  target > +0.2 (trained encoder)

The loss is contrastive in VSA space: for a batch of (question, answer)
embedding pairs, the encoder must map each question near its answer and far
from the other answers' questions. This is exactly what makes bipolarized
embeddings preserve semantic similarity — the property the +0.355 ceiling
depends on.

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
from prism.holo import HoloEncoder, HoloHead


from prism.memory import MemoryState
from prism.model import Prism


def _bipolar_st(v: torch.Tensor) -> torch.Tensor:
    """Bipolarize with straight-through (forward {±1}, backward identity)."""
    bipolar = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
    return bipolar + v - v.detach()


def _make_pairs(n: int, d_model: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Make synthetic (question, answer) pairs where each answer is close to
    its question (so the encoder has a learnable signal)."""
    g = torch.Generator().manual_seed(seed)
    # Each question is random; the answer is the question + small noise.
    # This means question_i and answer_i are SIMILAR (the signal to preserve),
    # while question_i and answer_j (i != j) are dissimilar (to push apart).
    q = torch.randn(n, d_model, generator=g)
    a = q + 0.3 * torch.randn(n, d_model, generator=g)
    return q, a


def _bipolar_st(v: torch.Tensor) -> torch.Tensor:
    bipolar = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
    return bipolar + v - v.detach()


def train_encoder(steps: int = 300, lr: float = 1e-2, D: int = 8192, d_model: int = 64) -> HoloEncoder:
    """Train a HoloEncoder to preserve cosine similarity through bipolarization."""
    torch.manual_seed(0)
    encoder = HoloEncoder(d_model=d_model, D=D)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.01)
    n_pairs = 64

    print(f"Training Holo encoder ({d_model} -> {D}, {sum(p.numel() for p in encoder.parameters())/1e3:.0f}K params)")
    print(f"  objective: contrastive — map question_i near answer_i, far from answer_j")
    for step in range(steps):
        # Fresh pairs each step so the encoder generalizes (doesn't memorize).
        q, a = _make_pairs(n_pairs, d_model, seed=step)
        # Encode (use proj directly so gradient flows, then bipolarize with ST).
        # HoloEncoder.forward bipolarizes non-differerentiably; for training we
        # bypass it and apply ST bipolarization on the raw projection.
        q_proj = encoder.proj(q)              # (N, D) — gradient flows here
        a_proj = encoder.proj(a)              # (N, D)
        q_h = _bipolar_st(q_proj)             # (N, D) — ST: forward ±1, backward identity
        a_h = _bipolar_st(a_proj)             # (N, D)

        # Contrastive InfoNCE: q_i should be more similar to a_i than to a_j.
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
            print(f"  step {step:4d} | loss {loss.item():.4f} | pos_sim {pos_sim:.3f} | acc {acc:.3f}")
    return encoder


def measure_encoder_specificity(encoder: HoloEncoder, cfg: PrismConfig, n_trials: int = 10) -> float:
    """Re-run the specificity probe with the trained encoder, integrated into
    a Prism model.

    The trained encoder's projections are injected into the HoloHead, then we
    measure whether seeding the tape with encoder-encoded content produces a
    specificity > the +0.0525 random-init baseline.
    """
    torch.manual_seed(0)
    model = Prism(cfg).eval()

    # Inject the trained encoder weights into the HoloHead's key/value encoders.
    for block in model.blocks:
        for expert in block.router.experts:
            head = getattr(expert, "head", None)
            if head is not None and head.__class__.__name__ == "HoloHead":
                # The HoloHead's encoders are d_model -> D_head where D_head =
                # num_slots * d_mem. Our trained encoder is d_model -> D (8192).
                # If dims match, copy directly; otherwise copy the first rows.
                with torch.no_grad():
                    ke_w = head.key_encoder.weight  # (D_head, d_model)
                    ve_w = head.value_encoder.weight
                    src_w = encoder.proj.weight     # (D, d_model)
                    D_head = ke_w.shape[0]
                    copy_rows = min(D_head, src_w.shape[0])
                    ke_w[:copy_rows] = src_w[:copy_rows]
                    ve_w[:copy_rows] = src_w[:copy_rows]
                break
        break  # first block only (demo)

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
    p.add_argument("--D", type=int, default=8192)
    cli = p.parse_args(argv)

    cfg = PrismConfig(
        vocab_size=128, d_model=64, num_layers=3, num_rates=4,
        memory=MemoryConfig(d_mem=32, num_slots=64),  # D_head = 2048
        holo_mode=True,
    )

    print("=" * 60)
    print("BEFORE training — random-init encoder specificity:")
    enc_random = HoloEncoder(d_model=cfg.d_model, D=cli.D)
    spec_before = measure_encoder_specificity(enc_random, cfg)
    print(f"  specificity = {spec_before:+.4f}  (baseline ~+0.05)")
    print("=" * 60)

    encoder = train_encoder(steps=cli.steps, lr=cli.lr, D=cli.D, d_model=cfg.d_model)

    print("\n" + "=" * 60)
    print("AFTER training — trained encoder specificity:")
    spec_after = measure_encoder_specificity(encoder, cfg)
    print(f"  specificity = {spec_after:+.4f}")
    print(f"  delta       = {spec_after - spec_before:+.4f}")
    print("=" * 60)
    if spec_after > 0.1:
        print("✅ Encoder training improved integrated specificity.")
        print("   This is the gap-closing step: +0.05 -> higher, via contrastive training.")
    else:
        print("⚠️  Improvement below threshold — the toy synthetic pairs may not")
        print("   transfer to real model embeddings. Real NQ/TriviaQA pairs needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
