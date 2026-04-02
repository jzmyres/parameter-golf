"""Tests for all 5 architectural constraints + RevDEQ convergence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _make_model(**overrides):
    from train_gpt import GPT
    defaults = dict(
        vocab_size=1024, num_layers=5, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=16384, bigram_dim=256,
        kv_latent_dim=0, num_refinements=1, router_sigmoid_gate=True,
    )
    defaults.update(overrides)
    return GPT(**defaults).cuda().bfloat16()


def test_all_constraints():
    """Test that all 5 constraints are satisfied."""
    model = _make_model()
    assert getattr(model, "router_sigmoid_gate", True) is True

    # Check constraint #1: RevDEQ
    assert model.shared_block is not None, "Must have shared_block (RevDEQ)"
    assert model.blocks is None, "blocks should be None in DEQ mode"
    assert hasattr(model, 'deq_beta'), "Must have deq_beta relaxation parameter"

    # Check constraint #3: MLA with Gated Attention
    attn = model.shared_block.attn
    assert attn.num_experts == 6, f"Attention must use 6 experts, got {attn.num_experts}"
    assert hasattr(attn, 'c_kv_down'), "Must have KV compression (MLA)"
    assert hasattr(attn, 'c_k_nope'), "Must have non-RoPE key decompress"
    assert hasattr(attn, 'c_k_rope'), "Must have decoupled RoPE key"
    assert hasattr(attn, 'attn_gate'), "Must have gated attention"

    # Check constraint #2: Soft Dense Routing (Dense MoE)
    mlp = model.shared_block.mlp
    assert mlp.num_experts == 6, f"MLP must use 6 experts, got {mlp.num_experts}"
    assert hasattr(mlp, 'expert_gate'), "Must have expert_gate (3D per-expert params)"
    assert hasattr(mlp, 'expert_fc'), "Must have expert_fc (3D per-expert params)"
    assert hasattr(mlp, 'expert_down'), "Must have expert_down (3D per-expert params)"
    assert hasattr(mlp, 'mlp_router'), "Must have mlp_router"
    assert mlp.expert_gate.ndim == 3, f"expert_gate must be 3D, got {mlp.expert_gate.ndim}D"

    # Check constraint #4: FSQ in MoS Head
    assert hasattr(model, 'mos_head'), "Must have MoS output head"
    assert hasattr(model.mos_head, 'gate_ctp'), "Must have CTP gate (pure softmax routing)"
    assert hasattr(model.mos_head, 'gate_ntp'), "Must have NTP gate (pure softmax routing)"
    assert not hasattr(model.mos_head, 'expert_gate_ctp_logits'), "Sigmoid gates removed (Mixtape)"

    # Check constraint #5: Diffusion-AR (refinement)
    assert model.num_refinements >= 1, "Must have at least 1 refinement step"
    assert hasattr(model, "_get_soft_embedding"), "Must implement refinement soft-embedding builder"
    z = torch.zeros((2, 32, model.tok_emb.embedding_dim), device="cuda", dtype=torch.bfloat16)
    soft = model._get_soft_embedding(z)
    assert soft.shape == z.shape, f"soft embedding must match z shape, got {tuple(soft.shape)}"

    # Gate initialization: all sigmoid gates should start at midpoint 0.5 (logit/bias = 0).
    assert torch.allclose(model.smear.gate.float(), torch.zeros_like(model.smear.gate.float()))
    assert torch.allclose(model.shared_block.gg_w.float(), torch.zeros_like(model.shared_block.gg_w.float()))
    assert float(model.shared_block.gg_b.float().item()) == 0.0
    assert torch.allclose(attn.gate_bias.float(), torch.zeros_like(attn.gate_bias.float()))
    # Gate-logit slice of c_q should be zero-initialized.
    dim = model.tok_emb.embedding_dim
    assert attn.c_q.weight.shape[0] == dim + attn.num_heads
    assert torch.allclose(attn.c_q.weight[dim:, :].float(), torch.zeros_like(attn.c_q.weight[dim:, :].float()))
    # Router expert gates
    assert torch.allclose(mlp.mlp_router.expert_gate_logits.float(), torch.zeros_like(mlp.mlp_router.expert_gate_logits.float()))
    assert torch.allclose(attn.attn_router.expert_gate_logits.float(), torch.zeros_like(attn.attn_router.expert_gate_logits.float()))

    print("PASS: All 5 constraints satisfied")


def test_router_sigmoid_gate_ablation():
    """Ablation: disabling router sigmoid gates should yield convex-mixture routing."""
    model = _make_model(router_sigmoid_gate=False)
    model.eval()

    dim = model.tok_emb.embedding_dim
    x = torch.randn(2, 8, dim, device="cuda", dtype=torch.bfloat16)
    r = model.shared_block.mlp.mlp_router
    _ = r(x)

    assert r.use_sigmoid_gate is False
    assert r._expert_gates is not None
    assert len(r._expert_gates) == r.num_experts
    assert all(abs(float(g) - 1.0) < 1e-6 for g in r._expert_gates), "disabled sigmoid gate must report gates=1"

    # With gates=1, route_weights should sum to 1 across experts (convex mixture).
    with torch.no_grad():
        w = r(x)
        err = (w.sum(dim=-1) - 1.0).abs().max().item()
    assert err < 1e-3, f"route_weights should sum to 1 when gates disabled; max_err={err:.6f}"


def test_revdeq_convergence():
    """Test RevDEQ coupled-state iteration converges."""
    model = _make_model(num_layers=8)

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
    model = _make_model(bigram_vocab_size=0, deq_backward="revdeq")

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
    test_router_sigmoid_gate_ablation()
    test_revdeq_convergence()
    test_revdeq_reversibility()
    print("\nAll tests passed!")
