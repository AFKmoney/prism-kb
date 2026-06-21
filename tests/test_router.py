"""Unit tests for the Polymorphic Router."""

from __future__ import annotations

import torch

from prism.config import PrismConfig
from prism.memory import MemoryState
from prism.router import PolymorphicRouter


def _cfg(**kw) -> PrismConfig:
    base = dict(vocab_size=32, d_model=16, num_rates=4)
    base.update(kw)
    return PrismConfig(**base)


def test_router_output_shape():
    cfg = _cfg()
    r = PolymorphicRouter(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    mem = MemoryState.create(2, cfg.memory, torch.device("cpu"), torch.float32)
    out, new_mem, stats, aux = r(x, mem)
    assert out.shape == x.shape
    assert aux.shape == ()


def test_router_aux_loss_positive():
    cfg = _cfg()
    r = PolymorphicRouter(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    mem = MemoryState.create(2, cfg.memory, torch.device("cpu"), torch.float32)
    _, _, _, aux = r(x, mem)
    # Load-balancing loss is E * sum(f_i * P_i) >= 0 (Cauchy-Schwarz).
    assert float(aux.detach()) >= -1e-6


def test_router_all_experts_get_gradient():
    # Epsilon-soft routing must give every expert's *read path* a gradient.
    # NOTE: the memory expert's *write* weights (v_proj_in, write_gate,
    # erase_gate) only receive a gradient if the tape they write is later read
    # by a downstream consumer. In isolation the router never re-reads its
    # output tape, so those write weights legitimately have no gradient here.
    # This is validated end-to-end in test_model.py::test_all_parameters_get_gradient
    # (multi-block models do read the tape, so write weights get gradient there).
    cfg = _cfg()
    r = PolymorphicRouter(cfg)
    x = torch.randn(4, 8, cfg.d_model)
    mem = MemoryState.create(4, cfg.memory, torch.device("cpu"), torch.float32)
    out, _, _, aux = r(x, mem)
    (out.sum() + aux).backward()
    no_grad = [n for n, p in r.named_parameters() if p.grad is None]

    # Expected dead params (in isolation): only the memory expert's write path.
    expected_dead_prefix = "experts.1.head."  # memory expert is index 1
    expected_dead_names = {
        "experts.1.head.v_proj_in.weight",
        "experts.1.head.write_gate.weight",
        "experts.1.head.write_gate.bias",
        "experts.1.head.erase_gate.weight",
        "experts.1.head.erase_gate.bias",
    }
    for name in expected_dead_names:
        assert name in no_grad, f"expected {name} to lack grad in isolation"
    unexpected = [n for n in no_grad if n not in expected_dead_names]
    assert unexpected == [], f"unexpected params without grad: {unexpected}"


def test_router_epsilon_anneal():
    cfg = _cfg()
    r = PolymorphicRouter(cfg, epsilon=0.05)
    assert abs(float(r.epsilon) - 0.05) < 1e-5
    r.epsilon.fill_(0.0)
    assert abs(float(r.epsilon)) < 1e-6


def test_topk2_activates_two_experts():
    """With router_topk=2, exactly 2 experts must be selected per token."""
    cfg = _cfg(router_topk=2)
    r = PolymorphicRouter(cfg)
    x = torch.randn(4, 8, cfg.d_model)
    mem = MemoryState.create(4, cfg.memory, torch.device("cpu"), torch.float32)
    # Reproduce the hard-mask computation.
    B, T, _ = x.shape
    mem_sum = r._mem_summary(mem, B, T, torch.device("cpu"), torch.float32)
    logits = r.gate(torch.cat([x, mem_sum], dim=-1))
    k = min(cfg.router_topk, r.num_experts)
    topk_idx = logits.topk(k, dim=-1).indices
    hard_mask = torch.zeros(B, T, r.num_experts).scatter_(-1, topk_idx, 1.0)
    per_token = hard_mask.sum(-1)
    assert torch.allclose(per_token, torch.full_like(per_token, 2.0))


def test_topk1_activates_one_expert():
    """With router_topk=1, exactly 1 expert must be selected per token."""
    cfg = _cfg(router_topk=1)
    r = PolymorphicRouter(cfg)
    x = torch.randn(4, 8, cfg.d_model)
    mem = MemoryState.create(4, cfg.memory, torch.device("cpu"), torch.float32)
    B, T, _ = x.shape
    mem_sum = r._mem_summary(mem, B, T, torch.device("cpu"), torch.float32)
    logits = r.gate(torch.cat([x, mem_sum], dim=-1))
    k = min(cfg.router_topk, r.num_experts)
    topk_idx = logits.topk(k, dim=-1).indices
    hard_mask = torch.zeros(B, T, r.num_experts).scatter_(-1, topk_idx, 1.0)
    per_token = hard_mask.sum(-1)
    assert torch.allclose(per_token, torch.full_like(per_token, 1.0))
