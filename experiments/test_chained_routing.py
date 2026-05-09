import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _classes():
    from train_gpt import CausalSelfAttention, MLP, RMSNorm, SoftDenseRouter

    return SoftDenseRouter, CausalSelfAttention, MLP, RMSNorm


def _make_stack(preset: str, *, num_experts: int = 8, num_shared: int = 1):
    from experiments.components.chained_routing import ChainedExpertStack, parse_stages

    soft_router, attn_cls, mlp_cls, norm_cls = _classes()
    stages = parse_stages(preset, num_experts=num_experts, num_shared_experts=num_shared)
    assert stages is not None
    stack = ChainedExpertStack(
        dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        rope_base=1000.0,
        qk_gain_init=1.2,
        kv_latent_dim=16,
        attn_expert_rank=8,
        mlp_expert_rank=8,
        stages=stages,
        soft_dense_router_cls=soft_router,
        causal_self_attention_cls=attn_cls,
        mlp_cls=mlp_cls,
        rms_norm_cls=norm_cls,
    )
    stack.train(True)
    return stack


def test_stage_spec_validation():
    from experiments.components.chained_routing import StageSpec

    assert StageSpec(attn_experts=3, shared_attn=1).attn_total() == 4
    assert StageSpec(mlp_experts=2, shared_mlp=1).mlp_total() == 3
    with pytest.raises(ValueError):
        StageSpec()
    with pytest.raises(ValueError):
        StageSpec(attn_experts=-1)


def test_parse_stages_preserves_expert_budget():
    from experiments.components.chained_routing import parse_stages

    for preset in ("split_2stage", "attn_first_2stage", "mlp_first_2stage"):
        stages = parse_stages(preset, num_experts=16, num_shared_experts=1)
        assert stages is not None
        assert sum(s.attn_total() for s in stages) == 16
        assert sum(s.mlp_total() for s in stages) == 16
    split = parse_stages("split_2stage", num_experts=16, num_shared_experts=1)
    assert split is not None
    assert [s.attn_total() for s in split] == [8, 8]
    assert [s.mlp_total() for s in split] == [8, 8]
    assert [s.attn_experts for s in split] == [7, 7]
    assert [s.mlp_experts for s in split] == [7, 7]
    assert [s.shared_attn for s in split] == [1, 1]
    assert [s.shared_mlp for s in split] == [1, 1]


def test_parse_json_and_none():
    from experiments.components.chained_routing import parse_stages

    assert parse_stages(None) is None
    assert parse_stages("") is None
    assert parse_stages("none") is None
    stages = parse_stages('[{"attn_experts": 2}, {"mlp_experts": 2}]')
    assert stages is not None
    assert len(stages) == 2
    with pytest.raises(ValueError):
        parse_stages("not_a_preset")


def test_module_toggle_roundtrip():
    from experiments.components.chained_routing import (
        chained_routing_enabled,
        set_chained_routing_enabled,
    )

    set_chained_routing_enabled(False)
    assert not chained_routing_enabled()
    set_chained_routing_enabled(True)
    assert chained_routing_enabled()
    set_chained_routing_enabled(False)
    assert not chained_routing_enabled()


@pytest.mark.parametrize("preset", ["split_2stage", "attn_first_2stage", "mlp_first_2stage"])
def test_chained_stack_forward_backward(preset):
    stack = _make_stack(preset)
    z = torch.randn(2, 4, 32, requires_grad=True)
    x0 = torch.randn(2, 4, 32)
    out = stack(z, x0, lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),), eps=1e-6))
    assert out.shape == z.shape
    out.square().mean().backward()

    for name, module in list(stack.iter_attn_modules()) + list(stack.iter_mlp_modules()):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), name
    for idx, router in enumerate(stack.iter_routers()):
        grads = [p.grad for p in router.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), idx


def test_no_parameter_sharing_across_stages():
    stack = _make_stack("split_2stage")
    owners: dict[int, str] = {}
    for name, param in stack.named_parameters():
        pid = id(param)
        assert pid not in owners, f"{name} shares parameter with {owners[pid]}"
        owners[pid] = name


def test_typed_stage_flow():
    from experiments.components.chained_routing import parse_stages

    attn_first = parse_stages("attn_first_2stage", num_experts=8, num_shared_experts=1)
    mlp_first = parse_stages("mlp_first_2stage", num_experts=8, num_shared_experts=1)
    assert attn_first is not None and mlp_first is not None
    assert attn_first[0].attn_total() == 8 and attn_first[0].mlp_total() == 0
    assert attn_first[1].attn_total() == 0 and attn_first[1].mlp_total() == 8
    assert mlp_first[0].mlp_total() == 8 and mlp_first[0].attn_total() == 0
    assert mlp_first[1].mlp_total() == 0 and mlp_first[1].attn_total() == 8


def test_router_usage_ema_min_tracks_across_diagnostic_batches():
    soft_router, _, _, _ = _classes()
    router = soft_router(4, 4)
    router.train(True)

    p1 = torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.7, 0.2, 0.1, 0.0]])
    router._mean_share_last = p1.mean(dim=0)
    router.usage_ema_update(distributed=False)
    router._record_diagnostics(p1, (0,))
    router._materialize_diag_lists()
    assert router._expert_usage_ema == pytest.approx([0.7, 0.2, 0.1, 0.0])
    assert router._expert_usage_ema_min_share == pytest.approx(0.0)

    p2 = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    router._mean_share_last = p2.mean(dim=0)
    router._record_diagnostics(p2, (0,))
    router._materialize_diag_lists()
    assert router._expert_usage_ema_min_share == pytest.approx(0.0)
    router.usage_ema_update(distributed=False)
    router._materialize_diag_lists()
    expected = [0.99 * 0.7, 0.99 * 0.2, 0.99 * 0.1, 0.01]
    assert router._expert_usage_ema == pytest.approx(expected)
    assert router._expert_usage_ema_min_share == pytest.approx(min(expected))


def test_dirichlet_ucb_router_default_off_path_is_differentiable():
    soft_router, _, _, _ = _classes()
    router = soft_router(
        4,
        4,
        scoring="dirichlet_ucb",
        dirichlet_ucb_beta=0.5,
        use_router_sigmoid_gate=False,
    )
    router.train(True)
    with torch.no_grad():
        router._expert_usage_ema_gpu.copy_(torch.tensor([0.0, 0.5, 0.25, 0.25]))
        router._expert_usage_ema_initialized.fill_(True)

    x = torch.randn(2, 3, 4, requires_grad=True)
    p = router(x)
    assert p.shape == (2, 3, 4)
    assert torch.allclose(p.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
    with torch.no_grad():
        x_score = x * router.score_norm_weight.to(dtype=x.dtype)
        evidence_logits = router.router(x_score) + router.expert_bias.to(dtype=x.dtype)
        evidence = torch.nn.functional.softplus(evidence_logits.float())
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mu = alpha / strength
        sigma = torch.sqrt(
            (alpha * (strength - alpha)
             / (strength.square() * (strength + 1.0)).clamp_min(1e-8)).clamp_min(0.0)
        )
        acq = (mu + 0.5 * sigma).clamp_min(1e-8)
        expected = acq / acq.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        assert torch.allclose(p.float(), expected, atol=1e-5)
    assert router._ema_alive_raw_loss is not None
    assert router._ema_balance_raw_loss is not None
    assert router._ema_balance_raw_loss.requires_grad
    assert router._ema_specialization_raw_loss is not None
    loss = (
        p[..., 0].mean()
        + router._ema_alive_raw_loss
        + router._ema_balance_raw_loss
        + router._ema_specialization_raw_loss
    )
    loss.backward()
    assert router.router.weight.grad is not None
    assert torch.isfinite(router.router.weight.grad).all()
    assert router.router.bias is not None
    assert router.router.bias.grad is not None
    assert torch.isfinite(router.router.bias.grad).all()


def test_gpt_chained_preset_smoke():
    from train_gpt import GPT

    model = GPT(
        vocab_size=64,
        num_layers=2,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        tie_embeddings=True,
        tied_embed_init_std=0.01,
        rope_base=1000.0,
        qk_gain_init=1.2,
        bigram_vocab_size=0,
        bigram_dim=0,
        kv_latent_dim=16,
        num_refinements=0,
        num_experts=8,
        num_shared_experts=1,
        attn_expert_rank=8,
        mlp_expert_rank=8,
        use_ctp=False,
        chained_stages_preset="split_2stage",
    )
    assert model.shared_block.chained_stack is not None
    x = torch.randint(0, 64, (1, 8))
    y = torch.randint(0, 64, (1, 8))
    loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
