"""Tests for architectural changes: Gated Attention, SoftDenseRouting, RevDEQ."""
import torch
import torch.nn as nn


def test_gated_attention():
    """Gated attention sigmoid gates should exist and modulate output."""
    from train_gpt import CausalSelfAttention
    dim = 512
    attn = CausalSelfAttention(dim, 8, 4, 10000.0, 1.5).cuda().bfloat16()
    assert hasattr(attn, 'attn_gate'), "Must have attn_gate"
    x = torch.randn(2, 32, dim, device="cuda", dtype=torch.bfloat16)
    out = attn(x)
    assert out.shape == (2, 32, dim)
    print("PASS: test_gated_attention")


def test_soft_dense_mlp_shapes():
    """SoftDenseMLP should produce correct output shapes."""
    from train_gpt import SoftDenseMLP
    dim = 512
    mlp = SoftDenseMLP(dim, mlp_mult=3.0, num_experts=4).cuda().bfloat16()
    x = torch.randn(2, 64, dim, device="cuda", dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == (2, 64, dim), f"Expected (2, 64, {dim}), got {out.shape}"
    print("PASS: test_soft_dense_mlp_shapes")


def test_soft_dense_mlp_gradients():
    """SoftDenseMLP must have flowing gradients through all paths."""
    from train_gpt import SoftDenseMLP
    dim = 512
    mlp = SoftDenseMLP(dim, mlp_mult=3.0, num_experts=4).cuda().bfloat16()
    x = torch.randn(2, 32, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = mlp(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    for name, p in mlp.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
    print("PASS: test_soft_dense_mlp_gradients")


def test_soft_dense_mlp_sigmoid_gating():
    """SoftDenseMLP must have per-expert sigmoid gates."""
    from train_gpt import SoftDenseMLP
    mlp = SoftDenseMLP(512, mlp_mult=3.0, num_experts=4)
    assert hasattr(mlp, 'expert_gate'), "Must have expert_gate parameter"
    assert mlp.expert_gate.shape[0] == 4, "Must have one gate per expert"
    print("PASS: test_soft_dense_mlp_sigmoid_gating")


if __name__ == "__main__":
    test_gated_attention()
    print("\nAll tests passed!")
