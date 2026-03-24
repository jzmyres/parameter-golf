"""Tests for architectural changes."""
import torch
import torch.nn as nn


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


def test_deq_shapes():
    """DEQ mode should produce correct output shapes."""
    from train_gpt import GPT
    model = GPT(
        vocab_size=1024, num_layers=10, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3.0, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, num_experts=4, deq_iterations=10,
    ).cuda().bfloat16()
    x = torch.randint(0, 1024, (2, 32), device="cuda")
    y = torch.randint(0, 1024, (2, 32), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    assert loss.ndim == 0, "Loss should be scalar"
    loss.backward()
    print(f"DEQ model params: {sum(p.numel() for p in model.parameters())}")
    print("PASS: test_deq_shapes")


def test_standard_with_experts():
    """Standard mode with experts should work."""
    from train_gpt import GPT
    model = GPT(
        vocab_size=1024, num_layers=10, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3.0, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, num_experts=4, deq_iterations=0,
    ).cuda().bfloat16()
    x = torch.randint(0, 1024, (2, 32), device="cuda")
    y = torch.randint(0, 1024, (2, 32), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    assert loss.ndim == 0
    loss.backward()
    print(f"Standard+experts params: {sum(p.numel() for p in model.parameters())}")
    print("PASS: test_standard_with_experts")


if __name__ == "__main__":
    test_soft_dense_mlp_shapes()
    test_soft_dense_mlp_gradients()
    test_soft_dense_mlp_sigmoid_gating()
    test_deq_shapes()
    test_standard_with_experts()
    print("\nAll tests passed!")
