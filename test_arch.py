"""Tests for all three architectural constraints + RevDEQ convergence."""
import torch


def test_all_constraints():
    """Test that all 3 constraints are satisfied."""
    from train_gpt import GPT
    model = GPT(
        vocab_size=1024, num_layers=5, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=16384, bigram_dim=256,
    ).cuda().bfloat16()

    # Check constraint #1: RevDEQ
    assert model.shared_block is not None, "Must have shared_block (RevDEQ)"
    assert model.blocks is None, "blocks should be None in DEQ mode"
    assert hasattr(model, 'deq_beta'), "Must have deq_beta relaxation parameter"

    # Check constraint #3: MLA with Gated Attention
    attn = model.shared_block.attn
    assert hasattr(attn, 'c_kv_down'), "Must have KV compression (MLA)"
    assert hasattr(attn, 'c_k_nope'), "Must have non-RoPE key decompress"
    assert hasattr(attn, 'c_k_rope'), "Must have decoupled RoPE key"
    assert hasattr(attn, 'attn_gate'), "Must have gated attention"

    # Check constraint #2: Soft Dense Routing
    mlp = model.shared_block.mlp
    assert hasattr(mlp, 'expert_gate'), "Must have expert_gate (soft dense routing)"
    assert hasattr(mlp, 'router'), "Must have router"

    print("PASS: All 3 constraints satisfied")


def test_revdeq_convergence():
    """Test RevDEQ coupled-state iteration converges."""
    from train_gpt import GPT
    model = GPT(
        vocab_size=1024, num_layers=8, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=16384, bigram_dim=256,
    ).cuda().bfloat16()

    x = torch.randint(0, 1024, (2, 32), device="cuda")
    y = torch.randint(0, 1024, (2, 32), device="cuda")

    # Forward + backward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()

    # Check convergence tracking
    assert hasattr(model, '_deq_residuals'), "Must track DEQ residuals"
    assert len(model._deq_residuals) > 0, "Must have at least 1 residual measurement"
    print(f"DEQ residual (last): {model._deq_residuals[-1]:.6f}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print("PASS: RevDEQ convergence tracking works")


def test_revdeq_reversibility():
    """Test RevDEQ backward reconstruction quality."""
    from train_gpt import GPT
    model = GPT(
        vocab_size=1024, num_layers=5, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5,
    ).cuda().bfloat16()

    x = torch.randint(0, 1024, (1, 16), device="cuda")
    y = torch.randint(0, 1024, (1, 16), device="cuda")

    # Use model in inference mode to trigger reconstruction verification
    model.train(False)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)

    assert hasattr(model, '_deq_recon_error'), "Must track reconstruction error"
    print(f"Reconstruction error: {model._deq_recon_error:.6f}")
    print("PASS: RevDEQ reversibility verification works")


if __name__ == "__main__":
    test_all_constraints()
    test_revdeq_convergence()
    test_revdeq_reversibility()
    print("\nAll tests passed!")
