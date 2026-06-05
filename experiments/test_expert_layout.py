import os
import sys

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _classes():
    from legacy.train_gpt_rich import CausalSelfAttention, MLP, RMSNorm, SoftDenseRouter

    return SoftDenseRouter, CausalSelfAttention, MLP, RMSNorm


def test_uniform_expert_layout_preserves_slot_counts():
    from experiments.components.expert_layout import make_uniform_expert_layout

    slots = make_uniform_expert_layout(
        experts_per_slot=4,
        expert_slots=4,
        num_shared_experts=0,
    )
    assert len(slots) == 4
    assert [s.attn_total() for s in slots] == [4, 4, 4, 4]
    assert [s.mlp_total() for s in slots] == [4, 4, 4, 4]
    assert sum(s.attn_total() for s in slots) == 16
    assert sum(s.mlp_total() for s in slots) == 16


def test_uniform_expert_layout_applies_shared_experts_per_slot():
    from experiments.components.expert_layout import make_uniform_expert_layout

    slots = make_uniform_expert_layout(
        experts_per_slot=4,
        expert_slots=2,
        num_shared_experts=1,
    )
    assert [s.attn_experts for s in slots] == [3, 3]
    assert [s.mlp_experts for s in slots] == [3, 3]
    assert [s.shared_attn for s in slots] == [1, 1]
    assert [s.shared_mlp for s in slots] == [1, 1]
    with pytest.raises(ValueError):
        make_uniform_expert_layout(
            experts_per_slot=1,
            expert_slots=2,
            num_shared_experts=1,
        )


class _FakeRouter(nn.Module):
    def __init__(self, dim, num_experts, **kwargs):
        super().__init__()
        self.num_experts = int(num_experts)
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, h, pre_normed=False):
        shape = (*h.shape[:-1], self.num_experts)
        return h.new_full(shape, 1.0 / float(self.num_experts))


class _FakeAttention(nn.Module):
    def __init__(self, *args, num_experts, **kwargs):
        super().__init__()
        self.num_experts = int(num_experts)
        self.inputs = []
        self.weight = nn.Parameter(torch.ones(()))

    def forward_experts(self, h):
        self.inputs.append(h.detach().clone())
        out = h.new_ones(*h.shape[:-1], self.num_experts, h.shape[-1])
        return out * self.weight.to(dtype=h.dtype)


class _FakeMlp(nn.Module):
    def __init__(self, *args, num_experts, **kwargs):
        super().__init__()
        self.num_experts = int(num_experts)
        self.inputs = []
        self.weight = nn.Parameter(torch.ones(()))

    def mix_experts(self, h, w, *, num_shared=0, shared_gate=None):
        self.inputs.append(h.detach().clone())
        return h.new_full(h.shape, 2.0) * self.weight.to(dtype=h.dtype)


class _IdentityNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x


def _make_fake_stack(order: str):
    from experiments.components.expert_layout import ExpertSlotStack, make_uniform_expert_layout

    return ExpertSlotStack(
        dim=8,
        num_heads=2,
        num_kv_heads=1,
        mlp_mult=2.0,
        rope_base=1000.0,
        qk_gain_init=1.0,
        kv_latent_dim=4,
        attn_expert_rank=2,
        mlp_expert_rank=2,
        slots=make_uniform_expert_layout(
            experts_per_slot=2,
            expert_slots=1,
            num_shared_experts=0,
        ),
        slot_order=order,
        soft_dense_router_cls=_FakeRouter,
        causal_self_attention_cls=_FakeAttention,
        mlp_cls=_FakeMlp,
        rms_norm_cls=_IdentityNorm,
    )


def test_parallel_slot_order_uses_one_state_for_attention_and_mlp():
    stack = _make_fake_stack("parallel")
    z = torch.zeros(1, 2, 8)
    x0 = torch.randn(1, 2, 8)
    stack(z, x0, lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),), eps=1e-6))
    assert torch.allclose(stack.attns[0].inputs[0], stack.mlps[0].inputs[0])


def test_attn_mlp_slot_order_feeds_mlp_the_attention_updated_state():
    stack = _make_fake_stack("attn_mlp")
    z = torch.zeros(1, 2, 8)
    x0 = torch.randn(1, 2, 8)
    stack(z, x0, lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),), eps=1e-6))
    assert not torch.allclose(stack.attns[0].inputs[0], stack.mlps[0].inputs[0])


def test_expert_slot_stack_forward_backward_for_4x4_layout():
    from experiments.components.expert_layout import ExpertSlotStack, make_uniform_expert_layout

    soft_router, attn_cls, mlp_cls, norm_cls = _classes()
    stack = ExpertSlotStack(
        dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        rope_base=1000.0,
        qk_gain_init=1.2,
        kv_latent_dim=16,
        attn_expert_rank=8,
        mlp_expert_rank=8,
        slots=make_uniform_expert_layout(
            experts_per_slot=4,
            expert_slots=4,
            num_shared_experts=0,
        ),
        slot_order="parallel",
        soft_dense_router_cls=soft_router,
        causal_self_attention_cls=attn_cls,
        mlp_cls=mlp_cls,
        rms_norm_cls=norm_cls,
    )
    z = torch.randn(2, 4, 32, requires_grad=True)
    x0 = torch.randn(2, 4, 32)
    out = stack(z, x0, lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),), eps=1e-6))
    assert out.shape == z.shape
    out.square().mean().backward()

    assert len(list(stack.iter_attn_modules())) == 4
    assert len(list(stack.iter_mlp_modules())) == 4
    for _, module in list(stack.iter_attn_modules()) + list(stack.iter_mlp_modules()):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_gpt_exposes_explicit_expert_layout_without_stage0_alias_for_multislot():
    from legacy.train_gpt_rich import GPT

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
        experts_per_slot=2,
        expert_slots=2,
        expert_slot_order="parallel",
        num_shared_experts=0,
        attn_expert_rank=8,
        mlp_expert_rank=8,
        use_ctp=False,
    )
    assert model.experts_per_slot == 2
    assert model.expert_slots == 2
    assert model.expert_layout == "2x2"
    assert len(model.shared_block.active_attn_modules()) == 2
    assert len(model.shared_block.active_mlp_modules()) == 2
    assert not hasattr(model.shared_block, "attn")
    assert not hasattr(model.shared_block, "mlp")
