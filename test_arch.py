"""Tests for all three architectural constraints."""
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

    # Check constraints
    # 1. RevDEQ: shared_block exists, blocks is None
    assert model.shared_block is not None, "Must have shared_block (RevDEQ)"
    assert model.blocks is None, "blocks should be None in DEQ mode"

    # 2. MLA: check for kv compression layers
    attn = model.shared_block.attn
    assert hasattr(attn, 'c_kv_down'), "Must have KV compression (MLA)"
    assert hasattr(attn, 'c_k_nope'), "Must have non-RoPE key decompress"
    assert hasattr(attn, 'c_k_rope'), "Must have decoupled RoPE key"
    assert hasattr(attn, 'attn_gate'), "Must have gated attention"

    # 3. Soft Dense Routing: check for expert_gate
    mlp = model.shared_block.mlp
    assert hasattr(mlp, 'expert_gate'), "Must have expert_gate (soft dense routing)"
    assert hasattr(mlp, 'router'), "Must have router"

    # Test forward/backward
    x = torch.randint(0, 1024, (2, 32), device="cuda")
    y = torch.randint(0, 1024, (2, 32), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print("PASS: all constraints satisfied and forward/backward works")


if __name__ == "__main__":
    test_all_constraints()
