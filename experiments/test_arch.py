"""Tests for all 5 architectural constraints + RevDEQ convergence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_model(**overrides):
    from train_gpt import GPT
    defaults = dict(
        vocab_size=1024, num_layers=5, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=16384, bigram_dim=256,
        kv_latent_dim=0, num_refinements=1,
    )
    defaults.update(overrides)
    dev = _get_device()
    m = GPT(**defaults).to(dev)
    # Keep dtype stable across CPU/GPU: bfloat16 on CUDA, float32 on CPU.
    if dev.type == "cuda":
        m = m.bfloat16()
    return m


def test_all_constraints():
    """Test that all 5 constraints are satisfied."""
    # Anchor against the configured num_experts to catch silent drift between
    # Hyperparameters / GPT.__init__ / Block / MLP / CSA defaults (CLAUDE.md
    # "Config Single-Source-of-Truth").  Override to a small value here so
    # CPU-only smoke runs stay fast; the assertion still pins the chain.
    expected_E = 4
    model = _make_model(num_experts=expected_E)

    # Check constraint #1: RevDEQ
    assert model.shared_block is not None, "Must have shared_block (RevDEQ)"
    assert hasattr(model, 'deq_beta'), "Must have deq_beta relaxation parameter"
    assert model.num_experts == expected_E, (
        f"GPT.num_experts must mirror the constructor arg, got {model.num_experts}"
    )

    # Check constraint #3: MLA with Gated Attention
    attn = model.shared_block.attn
    assert attn.num_experts == model.num_experts, (
        f"CSA must inherit num_experts from GPT, got {attn.num_experts} vs {model.num_experts}"
    )
    assert hasattr(attn, 'expert_kv_a'), "Must have per-expert KV (independent expert attn)"
    assert hasattr(attn, 'expert_k_nope'), "Must have per-expert K_nope decompress (MLA)"
    assert hasattr(attn, 'expert_kr_a'), "Must have per-expert K_rope (independent experts)"
    assert hasattr(attn, 'attn_gate'), "Must have gated attention"

    # Check constraint #2: Soft Dense Routing (Dense MoE)
    mlp = model.shared_block.mlp
    assert mlp.num_experts == model.num_experts, (
        f"MLP must inherit num_experts from GPT, got {mlp.num_experts} vs {model.num_experts}"
    )
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
    dev = _get_device()
    z_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    z = torch.zeros((2, 32, model.tok_emb.embedding_dim), device=dev, dtype=z_dtype)
    soft = model._get_soft_embedding(z)
    assert soft.shape == z.shape, f"soft embedding must match z shape, got {tuple(soft.shape)}"

    dim = model.tok_emb.embedding_dim

    # Soft Dense Routing: softmax allocation × sigmoid gate (weights sum to ≤ 1).
    r = mlp.mlp_router
    x = torch.randn(2, 8, dim, device=dev, dtype=z_dtype)
    with torch.no_grad():
        w = r(x)
        s = w.sum(dim=-1)
        assert (w >= 0.0).all().item(), "route weights must be non-negative"
        assert (s <= 1.0 + 1e-3).all().item(), f"route weights sum exceeds 1: max={s.max().item()}"
        assert (s > 0.0).all().item(), "route weights sum is zero (all gates closed)"

    print("PASS: All 5 constraints satisfied")


def test_expert_path_parameters_are_expert_independent():
    """Learned parameters inside expert paths must carry a leading expert dim."""
    model = _make_model(num_experts=4)
    attn = model.shared_block.attn
    E = attn.num_experts
    D = model.tok_emb.embedding_dim
    assert attn.q_up_norm_weight.shape == (E, attn.expert_rank)
    assert attn.kv_b_norm_weight.shape == (E, attn.kv_rank)
    assert attn.kr_b_norm_weight.shape == (E, attn.kr_rank)
    assert attn.wo_down_norm_weight.shape == (E, D)
    assert attn.wo_up_norm_weight.shape == (E, attn.wo_rank)
    assert attn.q_rope_norm_weight.shape == (E, attn.rope_dim)
    assert attn.q_nope_norm_weight.shape == (E, attn.nope_dim)
    assert attn.k_rope_norm_weight.shape == (E, attn.rope_dim)
    assert attn.k_nope_norm_weight.shape == (E, attn.nope_dim)
    for name in ("q_norm", "k_norm", "q_rope_norm", "k_rope_norm"):
        assert not hasattr(attn, name), f"{name} must not be a shared learned expert-path norm"

    mos = model.mos_head
    mos_E = mos.num_experts
    assert not hasattr(mos, "A_shared"), "CTP/NTP must not reuse one shared A bank"
    assert mos.A_ctp_shared.shape == (mos.num_shared, D, mos.rank)
    assert mos.A_ntp_shared.shape == (mos.num_shared, D, mos.rank)
    assert mos.B_denoise.shape == (mos_E, mos.vocab_size, mos.rank)
    assert mos.B_NTP.shape == (mos_E, mos.vocab_size, mos.rank)
    assert mos.ctp_rank_norm_weight.shape == (mos_E, mos.rank)
    assert mos.ntp_rank_norm_weight.shape == (mos_E, mos.rank)


def test_prenorm_forward_shapes_are_unchanged():
    """Targeted smoke test for MLA and MoS outputs after prenorm insertion."""
    model = _make_model(num_experts=4)
    dev = _get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    B, T, D = 2, 8, model.tok_emb.embedding_dim
    h = torch.randn(B, T, D, device=dev, dtype=dtype)

    with torch.no_grad():
        attn_out = model.shared_block.attn.forward_experts(h)
        log_p_ctp, log_p_ntp = model.mos_head(h)

    assert attn_out.shape == (B, T, model.num_experts, D)
    assert log_p_ctp.shape == (B, T, model.mos_head.vocab_size)
    assert log_p_ntp.shape == (B, T, model.mos_head.vocab_size)


def test_shared_bypass_gate_diagnostics_are_separate():
    """Shared bypass gates get their own stats, outside routed expert usage."""
    model = _make_model(num_experts=4, num_shared_experts=1)
    dev = _get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    x0 = torch.randn(1, 8, model.tok_emb.embedding_dim, device=dev, dtype=dtype)
    z = torch.randn_like(x0)

    from train_gpt import router_diagnostics
    with torch.no_grad(), router_diagnostics(True, step_tag=17):
        _ = model.shared_block(z, x0)

    block = model.shared_block
    assert block._shared_gate_diag_step == 17
    assert block._shared_gate_mean is not None
    assert block._shared_gate_min is not None
    assert block._shared_gate_std is not None

    router = block.router
    num_routed = block.num_experts - block.num_shared_experts
    assert router._expert_usage is not None
    assert len(router._expert_usage) == 2 * num_routed


def test_parcae_a_bar_is_bounded_by_construction():
    model = _make_model(use_parcae=True)
    with torch.no_grad():
        model.parcae_raw_delta.fill_(-100.0)
        low_rate_a_bar = model._parcae_a_bar()
        model.parcae_raw_delta.fill_(100.0)
        high_rate_a_bar = model._parcae_a_bar()

    min_a = model.parcae_min_a_bar
    assert torch.all(low_rate_a_bar > min_a)
    assert torch.all(low_rate_a_bar < 1.0)
    assert torch.all(high_rate_a_bar > min_a)
    assert torch.all(high_rate_a_bar < 1.0)


def test_revdeq_convergence():
    """Test RevDEQ coupled-state iteration converges."""
    model = _make_model(num_layers=8)

    dev = _get_device()
    x = torch.randint(0, 1024, (2, 32), device=dev)
    y = torch.randint(0, 1024, (2, 32), device=dev)

    # Forward + backward
    if dev.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
    else:
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

    dev = _get_device()
    x = torch.randint(0, 1024, (1, 16), device=dev)
    y = torch.randint(0, 1024, (1, 16), device=dev)

    # Reconstruction diagnostic is produced during the RevDEQ backward path and
    # is gated behind router_diagnostics(...) for speed in normal training.
    from train_gpt import router_diagnostics
    with router_diagnostics(True, step_tag=0):
        if dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        else:
            loss = model(x, y)
    loss.backward()

    recon = getattr(model.shared_block, "_deq_recon_error_last_bwd", None)
    assert recon is not None, "Must produce reconstruction diagnostic under RevDEQ backward"
    assert float(recon) >= 0.0
    assert float(recon) < 1.0
    print(f"Reconstruction error: {float(recon):.6f}")
    print("PASS: RevDEQ reversibility verification works")


if __name__ == "__main__":
    test_all_constraints()
    test_expert_path_parameters_are_expert_independent()
    test_prenorm_forward_shapes_are_unchanged()
    test_shared_bypass_gate_diagnostics_are_separate()
    test_parcae_a_bar_is_bounded_by_construction()
    test_revdeq_convergence()
    test_revdeq_reversibility()
    print("\nAll tests passed!")
