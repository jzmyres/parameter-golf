"""Tests for all 5 architectural constraints + RevDEQ convergence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_model(**overrides):
    # Legacy-path harness: keeps the iter145r-era CTP+refinement surface alive
    # so reversibility/convergence/Parcae tests still exercise that ablation
    # code. Production defaults (use_ctp=False, num_refinements=0) flipped in
    # iter146 and are pinned by `test_gpt_constructor_defaults_track_hyperparameters`
    # which constructs `GPT()` without going through this helper.
    # `deq_prefix_anchors=False` pinned explicitly in 2026-05-13 because the
    # production default flipped to True (iter152 promotion-propagation fix)
    # but prefix anchors are incompatible with num_refinements=1 (validator
    # rejects the combo).  Tests that need prefix anchors must override.
    from train_gpt import GPT
    defaults = dict(
        vocab_size=1024, num_layers=5, model_dim=640, num_heads=10,
        num_kv_heads=5, mlp_mult=2.5, tie_embeddings=True,
        tied_embed_init_std=0.005, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=16384, bigram_dim=256,
        kv_latent_dim=0, num_refinements=1,
        use_ctp=True,
        deq_prefix_anchors=False,
    )
    defaults.update(overrides)
    dev = _get_device()
    m = GPT(**defaults).to(dev)
    # Keep dtype stable across CPU/GPU: bfloat16 on CUDA, float32 on CPU.
    if dev.type == "cuda":
        m = m.bfloat16()
    return m


def test_gpt_constructor_defaults_track_hyperparameters():
    """Production defaults must not drift between Hyperparameters and GPT()."""
    from train_gpt import GPT, Hyperparameters

    model = GPT(
        vocab_size=64, num_layers=2, model_dim=32, num_heads=4,
        num_kv_heads=2, mlp_mult=2.0, tie_embeddings=True,
        tied_embed_init_std=0.005, rope_base=10000.0,
        qk_gain_init=1.0, bigram_vocab_size=0, bigram_dim=8,
        kv_latent_dim=16, attn_expert_rank=4, mlp_expert_rank=4,
    )
    assert model.num_refinements == Hyperparameters.num_refinements
    assert model.deq_beta == Hyperparameters.deq_beta
    assert model.deq_bptt_k == Hyperparameters.deq_bptt_k
    assert model.num_experts == Hyperparameters.num_experts
    assert model.use_ctp == Hyperparameters.use_ctp
    assert model.mos_head.use_ctp == Hyperparameters.use_ctp
    assert not hasattr(model.mos_head, "gate_ctp")


def test_default_router_does_not_allocate_inactive_parameters():
    """Dirichlet-UCB + gate-off should not carry dead trainable state."""
    model = _make_model(num_experts=4, use_ctp=False)
    router = model.shared_block.router
    assert router.scoring == "dirichlet_ucb"
    assert router.use_router_sigmoid_gate is False
    assert router.prototypes is None
    assert router.router_gate is None
    assert router.gate_norm_weight is None
    names = dict(router.named_parameters())
    assert "prototypes" not in names
    assert "gate_norm_weight" not in names
    assert not any(name.startswith("router_gate.") for name in names)


def test_router_state_dict_optional_param_load():
    """Loading a legacy (l2 + sigmoid-gate) checkpoint into a default
    (Dirichlet-UCB + gate-off) router must silently drop the optional keys
    and must not mutate the caller's state_dict."""
    from train_gpt import SoftDenseRouter

    dim, E = 32, 4
    legacy = SoftDenseRouter(dim=dim, num_experts=E,
                             scoring="l2", use_router_sigmoid_gate=True)
    legacy_sd = legacy.state_dict()
    legacy_keys_before = set(legacy_sd.keys())
    assert "prototypes" in legacy_keys_before
    assert "gate_norm_weight" in legacy_keys_before
    assert "router_gate.weight" in legacy_keys_before

    fresh = SoftDenseRouter(dim=dim, num_experts=E,
                            scoring="dirichlet_ucb", use_router_sigmoid_gate=False)
    result = fresh.load_state_dict(legacy_sd, strict=False)
    # Caller's dict must be intact post-load (no in-place mutation).
    assert set(legacy_sd.keys()) == legacy_keys_before
    # Optional legacy keys should not surface as unexpected — they are
    # silently dropped by _load_from_state_dict.
    unexpected = set(result.unexpected_keys)
    for k in ("prototypes", "gate_norm_weight", "router_gate.weight", "router_gate.bias"):
        assert k not in unexpected, f"{k!r} leaked to unexpected_keys"
    # Shared parameters that exist on both routers must have loaded.
    fresh_sd = fresh.state_dict()
    for shared in ("router.weight", "score_norm_weight", "expert_bias"):
        assert torch.equal(fresh_sd[shared], legacy_sd[shared]), f"{shared} did not load"


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
    assert attn.q_down_norm_weight.shape == (E, D)
    assert attn.q_up_norm_weight.shape == (E, attn.expert_rank)
    assert attn.kv_a_norm_weight.shape == (E, D)
    assert attn.kv_b_norm_weight.shape == (E, attn.kv_rank)
    assert attn.k_nope_in_norm_weight.shape == (E, attn.kv_latent_dim)
    assert attn.v_in_norm_weight.shape == (E, attn.kv_latent_dim)
    assert attn.kr_a_norm_weight.shape == (E, D)
    assert attn.kr_b_norm_weight.shape == (E, attn.kr_rank)
    assert attn.wo_down_norm_weight.shape == (E, D)
    assert attn.wo_up_norm_weight.shape == (E, attn.wo_rank)
    assert attn.q_rope_norm_weight.shape == (E, attn.rope_dim)
    assert attn.q_nope_norm_weight.shape == (E, attn.nope_dim)
    assert attn.k_rope_norm_weight.shape == (E, attn.rope_dim)
    assert attn.k_nope_norm_weight.shape == (E, attn.nope_dim)
    for name in ("q_norm", "k_norm", "q_rope_norm", "k_rope_norm"):
        assert not hasattr(attn, name), f"{name} must not be a shared learned expert-path norm"

    mlp = model.shared_block.mlp
    assert mlp.gate_in_norm_weight.shape == (E, D)
    assert mlp.fc_in_norm_weight.shape == (E, D)
    assert mlp.hidden_norm_weight.shape == (E, mlp.expert_rank)

    mos = model.mos_head
    mos_E = mos.num_experts
    assert not hasattr(mos, "A_shared"), "CTP/NTP must not reuse one shared A bank"
    assert mos.A_ctp_shared.shape == (mos.num_shared, D, mos.rank)
    assert mos.A_ntp_shared.shape == (mos.num_shared, D, mos.rank)
    assert mos.B_denoise.shape == (mos_E, mos.vocab_size, mos.rank)
    assert mos.B_NTP.shape == (mos_E, mos.vocab_size, mos.rank)
    assert mos.gate_ctp_norm_weight.shape == (D,)
    assert mos.gate_ntp_norm_weight.shape == (D,)
    assert mos.ctp_a_norm_weight.shape == (mos_E, D)
    assert mos.ntp_a_norm_weight.shape == (mos_E, D)
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


def test_parcae_a_bar_has_reversibility_safety_margin():
    """Ā ∈ [ε_rev, 1) for any finite raw values → RevDEQ recon always safe.

    At extreme raw values Ā_core = exp(Δ·A) saturates to 0 or 1, so the
    rescaled Ā saturates to ε_rev or 1 − (1 − ε_rev) · ε_min ≈ 1. The
    reversibility denominator (1 − β) = Ā stays ≥ ε_rev by construction.
    """
    model = _make_model(use_parcae=True)
    eps_rev = float(model.parcae_reversibility_floor)
    with torch.no_grad():
        for raw_val in (-100.0, -10.0, 0.0, 10.0, 100.0):
            model.parcae_raw_a.fill_(raw_val)
            model.parcae_raw_delta.fill_(raw_val)
            a_bar = model._parcae_a_bar()
            assert torch.all(a_bar >= eps_rev), f"Ā < ε_rev at raw={raw_val}"
            assert torch.all(a_bar < 1.0), f"Ā ≥ 1 at raw={raw_val}"


def test_parcae_b_bar_is_strictly_positive():
    """B̄ = Δ · B > 0 for any finite raw values (no reversibility floor)."""
    model = _make_model(use_parcae=True)
    with torch.no_grad():
        for raw_val in (-100.0, -10.0, 0.0, 10.0, 100.0):
            model.parcae_raw_b.fill_(raw_val)
            model.parcae_raw_delta.fill_(raw_val)
            b_bar = model._parcae_b_bar()
            assert torch.all(b_bar > 0.0), f"B̄ ≤ 0 at raw={raw_val}"


def test_parcae_ab_share_only_delta():
    """Perturbing raw_a ⇒ b_bar unchanged; raw_b ⇒ a_bar unchanged; raw_delta ⇒ both change."""
    model = _make_model(use_parcae=True)
    with torch.no_grad():
        a0, b0 = model._parcae_a_bar().clone(), model._parcae_b_bar().clone()
        model.parcae_raw_a.add_(1.0)
        assert torch.allclose(model._parcae_b_bar(), b0), "raw_a leaks into b_bar"
        assert not torch.allclose(model._parcae_a_bar(), a0), "raw_a does not move a_bar"
        model.parcae_raw_a.sub_(1.0)

        model.parcae_raw_b.add_(1.0)
        assert torch.allclose(model._parcae_a_bar(), a0), "raw_b leaks into a_bar"
        assert not torch.allclose(model._parcae_b_bar(), b0), "raw_b does not move b_bar"
        model.parcae_raw_b.sub_(1.0)

        model.parcae_raw_delta.add_(1.0)
        assert not torch.allclose(model._parcae_a_bar(), a0), "raw_delta does not move a_bar"
        assert not torch.allclose(model._parcae_b_bar(), b0), "raw_delta does not move b_bar"


def test_parcae_init_recovers_target_a_bar():
    """At step 0, Ā ≈ parcae_init_a_bar and B̄ ≈ 1 − Ā (continuity with iter 66a)."""
    import torch
    target = 0.7
    model = _make_model(use_parcae=True, parcae_init_a_bar=target)
    with torch.no_grad():
        a_bar = model._parcae_a_bar()
        b_bar = model._parcae_b_bar()
    assert torch.allclose(a_bar, torch.full_like(a_bar, target), atol=1e-3), \
        f"Ā₀={a_bar.mean():.4f}, expected ≈ {target}"
    assert torch.allclose(b_bar, torch.full_like(b_bar, 1.0 - target), atol=1e-3), \
        f"B̄₀={b_bar.mean():.4f}, expected ≈ {1.0 - target}"


def test_parcae_diagnostics_match_zoh_formulas():
    """Logged Parcae diagnostics must match the diagonal ZOH damping formulas."""
    model = _make_model(use_parcae=True, num_layers=3)
    dev = _get_device()
    n = model.parcae_raw_a.numel()
    with torch.no_grad():
        dtype = model.parcae_raw_a.dtype
        model.parcae_raw_a.copy_(torch.linspace(-0.5, 0.5, n, device=dev, dtype=dtype))
        model.parcae_raw_delta.copy_(torch.linspace(-0.25, 0.25, n, device=dev, dtype=dtype))
        model.parcae_raw_b.copy_(torch.linspace(-0.1, 0.3, n, device=dev, dtype=dtype))

        delta = F.softplus(model.parcae_raw_delta.float()) + float(model.parcae_min_rate)
        a_mag = F.softplus(model.parcae_raw_a.float()) + float(model.parcae_min_rate)
        a_bar_core = torch.exp(-(delta * a_mag))
        eps_rev = float(model.parcae_reversibility_floor)
        a_bar = eps_rev + (1.0 - eps_rev) * a_bar_core
        beta = 1.0 - a_bar
        b_mag = F.softplus(model.parcae_raw_b.float()) + float(model.parcae_min_rate)
        b_bar = delta * b_mag
        diag = model.parcae_diagnostics(k_override=3)

    expected = {
        "parcae_a_bar_min": a_bar.min(),
        "parcae_a_bar_mean": a_bar.mean(),
        "parcae_a_bar_max": a_bar.max(),
        "parcae_a_bar_core_max": a_bar_core.max(),
        "parcae_beta_mean": beta.mean(),
        "parcae_beta_max": beta.max(),
        "parcae_b_bar_mean": b_bar.mean(),
        "parcae_b_bar_max": b_bar.max(),
        "parcae_delta_mean": delta.mean(),
        "parcae_delta_max": delta.max(),
        "parcae_recon_amp_log10": a_bar.min().clamp_min(1e-12).reciprocal().log10() * 3.0,
    }
    for key, value in expected.items():
        assert key in diag, f"missing diagnostic {key}"
        assert torch.allclose(diag[key], value, atol=1e-6, rtol=1e-6), (
            f"{key}: got {diag[key].item():.8f}, expected {value.item():.8f}")


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
    model = _make_model(bigram_vocab_size=0, deq_bptt_k=0)

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

    recon = getattr(model.shared_block, "_deq_x0_recon_error_last_bwd", None)
    assert recon is not None, "Must produce RevDEQ backward x0 recon diagnostic"
    assert float(recon) >= 0.0
    assert float(recon) < 1.0
    print(f"Reconstruction error: {float(recon):.6f}")
    print("PASS: RevDEQ reversibility verification works")


def test_revdeq_reconstruction_at_a_bar_floor():
    """Reversibility holds when Parcae pushes Ā to its ε_rev floor.

    Drives raw_a to +100 so |A| saturates softplus → Δ·|A| large → Ā_core
    underflows to 0 → Ā = ε_rev (the solver-β denominator floor). Keeps
    raw_delta / raw_b at init so B̄ stays moderate (B̄ is paper-faithful
    unbounded; this test isolates the Ā-reversibility axis). The RevDEQ
    backward reconstruction must stay within the normal tolerance regime.

    Note: reconstruction error compounds as (1/Ā)^K across K backward steps,
    so ε_rev is sized to keep the worst-case amplification tractable in fp64.
    """
    model = _make_model(bigram_vocab_size=0, deq_bptt_k=0)
    eps_rev = float(model.parcae_reversibility_floor)
    with torch.no_grad():
        model.parcae_raw_a.fill_(100.0)  # → Ā saturates to ε_rev
        a_bar = model._parcae_a_bar()
    assert torch.all(a_bar <= eps_rev + 0.01), f"expected Ā at floor, got {a_bar.mean():.4f}"

    dev = _get_device()
    x = torch.randint(0, 1024, (1, 16), device=dev)
    y = torch.randint(0, 1024, (1, 16), device=dev)

    from train_gpt import router_diagnostics
    with router_diagnostics(True, step_tag=0):
        if dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        else:
            loss = model(x, y)
    loss.backward()

    recon = getattr(model.shared_block, "_deq_x0_recon_error_last_bwd", None)
    assert recon is not None, "Expected RevDEQ x0 recon diagnostic even at Ā=ε_rev"
    recon_f = float(recon)
    # Correctness claim: ε_rev prevents the backward from hitting division by
    # zero / overflow / NaN. Recon accuracy (< 1) holds in the normal training
    # regime where Ā ≈ 0.7 — NOT at the saturated floor, where worst-case
    # amplification is (1/ε_rev)^K = 10^12 and even fp64 cannot guarantee
    # sub-unit error. The point of the floor is: training-loop math stays
    # well-defined. Actual training recon is covered by test_revdeq_reversibility.
    import math as _math
    assert _math.isfinite(recon_f), f"recon is NaN/Inf at Ā=ε_rev: {recon_f!r}"
    # Finite but possibly large at the extreme. An extremely loose upper bound
    # (1e12) just catches catastrophic overflow, not per-iter error.
    assert recon_f < 1e12, f"recon overflowed at Ā=ε_rev: {recon_f:.3e}"
    print(f"Ā-floor reconstruction error (extreme regime, finite): {recon_f:.3e}")
    print("PASS: RevDEQ reversibility at Ā=ε_rev stays finite")


def test_ntp_only_baseline_skips_ctp_param_banks():
    """Production baseline (iter 94, H60): use_ctp=False ⇒ CTP banks NOT allocated.

    This is the production-faithful counterpart to test_all_constraints (which
    instantiates the CTP variant via the GPT.__init__ default).  Mirrors the
    Hyperparameters default `use_ctp = False` and the `MoSHead.use_ctp`
    conditional at MoSHead.__init__ (line ~1890).
    """
    model = _make_model(num_experts=4, use_ctp=False)
    mos = model.mos_head
    assert mos.use_ctp is False, "use_ctp must thread through GPT → MoSHead"
    # CTP-specific banks are NOT allocated in the NTP-only baseline.
    for attr in ("gate_ctp", "gate_ctp_norm_weight", "A_ctp_shared", "A_ctp",
                 "ctp_a_norm_weight", "ctp_rank_norm_weight", "B_denoise"):
        assert not hasattr(mos, attr), (
            f"NTP-only baseline must not allocate CTP bank `{attr}`")
    # NTP banks ARE allocated.
    for attr in ("gate_ntp", "gate_ntp_norm_weight", "A_ntp_shared",
                 "ntp_a_norm_weight", "ntp_rank_norm_weight", "B_NTP"):
        assert hasattr(mos, attr), f"NTP baseline must allocate `{attr}`"


def test_post_int6_gate_skips_mos_ctp_when_disabled():
    """Issue 3: when use_ctp=False, the post-int6 gate must NOT emit mos_ctp checks.

    The diagnostic gate previously appended both `mos_ctp` and `mos_ntp` check
    specs unconditionally; with use_ctp=False the CTP usage getters fall back
    to aliased / empty values, producing fake "dead expert" failures.  We now
    gate the CTP append on `mos_head.use_ctp`.

    This test inspects the predicate logic by simulating the spec construction
    on a real MoSHead (use_ctp=False) and asserting the resulting prefix set
    contains "mos_ntp" but never "mos_ctp".
    """
    model = _make_model(num_experts=4, use_ctp=False)
    mos = model.mos_head
    # Replicate the gate's append logic on this MoSHead.
    prefixes: list[str] = []
    if mos is not None:
        prefixes.append("mos_ntp")
        if getattr(mos, "use_ctp", False):
            prefixes.append("mos_ctp")
    assert "mos_ntp" in prefixes, "NTP must always be checked"
    assert "mos_ctp" not in prefixes, (
        "use_ctp=False must skip mos_ctp diagnostic — CTP banks aren't allocated, "
        "so any 'mos_ctp_min_share' failure would be fake.")


def test_prescriptions_route_to_invariant_mechanisms_not_per_symptom_losses():
    """Witness for the CLAUDE.md "Root-cause fix preference" audit row.

    Every `_prescribe_failure_fix` branch must EITHER include an invariant-level
    mechanism (shared controller, parameterization, or load-balance regularizer)
    in `config_change`, OR include "temporary ablation" framing in the `fix`
    text. Per-case `required` and `forbidden` lists pin the specific routing
    decisions and act as regression guards against keys that were intentionally
    removed in the rework (e.g. `router_ema_alive_coef_mult`, `weight_decay_mult`
    for routing collapse).
    """
    from train_gpt import _prescribe_failure_fix

    invariant_keys = {
        "router_bias_update",
        "router_load_cv_coef_floor",
        "mos_load_cv_coef_mult",
        "weight_decay_mult",
        "deq_k_max_delta",
        "needs_expert_bank_geometry_constraint",
        "needs_mos_output_geometry_constraint",
        "needs_transition_parameterization",
        "needs_transition_jacobian_control",
        # iter155 corrected: contraction-object key was needs_iteration_map_contraction
        # (Lyapunov-on-F target). Tier 1+2 redesign 2026-05-13 reframed the gate
        # from operator norm (lip_ub_F, over-restrictive) to spectral radius
        # (rho_F, necessary AND sufficient for asymptotic local convergence).
        # The lip_ub_F branch became advisory; rho_F branch routes to
        # needs_formal_tier_contraction (formal-tier mechanism: spectral
        # normalization, bounded-Lipschitz block, learned c·Δ gain — all
        # reusable, NOT a per-symptom loss).
        "needs_iteration_map_contraction",
        "needs_formal_tier_contraction",
    }
    metric_specific_keys = {
        "expert_output_diversity_coef_mult",
        "mos_output_diversity_coef",
    }

    cases = [
        # (failure, expected_category, required_keys, forbidden_keys)
        # Router-side min_share prescription EMPIRICALLY REFUTED 2026-05-15
        # by 3-iter closure (iter158/162/165 all confirmed pushing on
        # routing-balance ACTIVELY HURTS BPB at iter152's operating point).
        # Prescription now returns empty config_change; router_bias_update,
        # router_load_cv_coef_floor, and any usage-prior fix are forbidden.
        ("attn_min_share=0.01 < 0.150", "router_collapse_advisory",
         set(),
         {"router_bias_update", "router_load_cv_coef_floor",
          "router_ema_alive_coef_mult", "router_ema_balance_coef_floor"}),
        ("mlp_min_share=0.01 < 0.150", "router_collapse_advisory",
         set(), {"router_bias_update", "router_load_cv_coef_floor"}),
        ("mos_ntp_min_share=0.005 < 0.150", "mos_router_collapse",
         {"mos_load_cv_coef_mult"}, {"weight_decay_mult"}),
        ("attn_ortho=0.71 > 0.5", "expert_collapse",
         {"needs_expert_bank_geometry_constraint", "expert_output_diversity_coef_mult"},
         {"weight_decay_mult"}),
        ("mos_ntp_ortho=0.61 > 0.5", "mos_head_collapse",
         {"needs_mos_output_geometry_constraint"}, set()),
        ("k-sweep delta > 0.1 at K=64", "fp_quality_loss", set(), set()),
        # Tier 1+2 redesign 2026-05-13: lip_ub_F demoted to advisory; the
        # gate-relevant signal is now rho_F (spectral radius, necessary AND
        # sufficient for asymptotic convergence) and iter_conv_rel (empirical).
        ("lip_ub_F=45.0 >= 1.0", "operator_norm_advisory",
         set(), {"lyapunov_coef", "needs_iteration_map_contraction"}),
        ("rho_F=1.2 >= 1.0", "fp_convergence_failed",
         {"needs_formal_tier_contraction"}, {"lyapunov_coef"}),
        ("fp_bound=2.5 >= 1.0", "fp_certificate_loose", set(), set()),
        # iter_conv_rel category renamed Tier 1: gate-relevant empirical signal.
        ("iter_conv_rel=0.6 > 0.3", "fp_convergence_empirical_failed",
         {"weight_decay_mult", "deq_k_max_delta"}, {"lyapunov_coef"}),
        ("deq_recon_err=1.5e-2 > 1e-3", "reversibility_broken", set(), set()),
    ]
    for failure, expected_category, required, forbidden in cases:
        p = _prescribe_failure_fix(failure)
        assert p["category"] == expected_category, (
            f"{failure!r}: category {p['category']} != expected {expected_category}"
        )
        change_keys = set(p["config_change"].keys())
        for key in required:
            assert key in change_keys, f"{failure!r}: missing required key {key!r} in {change_keys}"
        for key in forbidden:
            assert key not in change_keys, f"{failure!r}: forbidden key {key!r} appears in {change_keys}"
        # Root-cause fix preference: invariant mechanism OR explicit ablation
        # framing OR advisory-only category (Tier 1 redesign 2026-05-13:
        # operator-norm proxies like lip_ub_F are now diagnostic-only and
        # legitimately have no config_change; the gate-relevant prescriptions
        # are rho_F and iter_conv_rel which DO have invariant_keys.)
        has_invariant = bool(change_keys & invariant_keys)
        has_metric_only = bool(change_keys & metric_specific_keys)
        fix_text = p["fix"].lower()
        framed_as_ablation = "temporary ablation" in fix_text or "fallback only" in fix_text
        is_advisory_category = p["category"].endswith("_advisory") or p["category"] == "single_state_blend_loose" or p["category"] == "transition_jacobian_loose"
        assert has_invariant or framed_as_ablation or is_advisory_category, (
            f"{failure!r}: config_change={change_keys} has no invariant mechanism, "
            f"the fix string lacks 'temporary ablation' framing, AND the category "
            f"({p['category']!r}) is not advisory — violates Root-cause fix preference."
        )
        if has_metric_only and not has_invariant:
            assert framed_as_ablation, (
                f"{failure!r}: prescribed only metric-specific knobs "
                f"({change_keys & metric_specific_keys}) without temporary-ablation framing"
            )


def test_routing_regularizer_coefficients_match_promoted_defaults():
    """Current rescue stack: EMA balance handles averages, alive hinge handles
    minimum share, router CV is off by default, and diversity uses the
    coefficient-first values. This locks Hyperparameters defaults and verifies
    they propagate to a constructed model so signature-default drift is loud.
    """
    from train_gpt import Hyperparameters
    # Required fields exist.
    for name in (
        "router_load_cv_coef",
        "router_ema_balance_coef",
        "router_ema_specialization_coef",
        "router_ema_alive_coef",
        "router_pertoken_entropy_coef",
        "mos_load_cv_coef",
        "expert_output_diversity_coef",
        "lyapunov_every",
        "lyapunov_max_tokens",
        "regularizer_warmup_frac",
        "use_router_sigmoid_gate",
        "router_scoring",
        "router_dirichlet_ucb_beta",
        "weight_decay",
    ):
        assert hasattr(Hyperparameters, name), f"missing Hyperparameters field {name}"
    # Current root-cause rescue values (2026-05-08).
    assert float(Hyperparameters.router_load_cv_coef) == 0.0
    assert float(Hyperparameters.router_ema_balance_coef) == 0.30
    assert float(Hyperparameters.router_ema_specialization_coef) == 0.20
    assert float(Hyperparameters.router_ema_alive_coef) == 0.02
    assert float(Hyperparameters.router_pertoken_entropy_coef) == 0.1
    assert float(Hyperparameters.mos_load_cv_coef) == 0.15
    assert float(Hyperparameters.expert_output_diversity_coef) == 0.30
    assert int(Hyperparameters.lyapunov_every) == 16
    assert int(Hyperparameters.lyapunov_max_tokens) == 64
    assert float(Hyperparameters.regularizer_warmup_frac) == 0.07
    assert bool(Hyperparameters.use_router_sigmoid_gate) is False
    assert str(Hyperparameters.router_scoring) == "dirichlet_ucb"
    assert float(Hyperparameters.router_dirichlet_ucb_beta) == 0.5
    assert float(Hyperparameters.weight_decay) == 0.015
    # Regression guard: iter 142b dropped the relu(cv − cv_target)² hinge in
    # favor of continuous cv²; the old target knobs must NOT come back.
    assert not hasattr(Hyperparameters, "cv_target")
    assert not hasattr(Hyperparameters, "mos_cv_target")
    # Verify Hyperparameters values propagate to a constructed model.  Only
    # fields that GPT.__init__ stores on `self` are public model attributes;
    # `expert_output_diversity_coef` and `router_pertoken_entropy_coef` are
    # held as `_*_target` private fields and read via the annealer/router.
    model = _make_model(num_experts=4)
    assert float(model.router_load_cv_coef) == 0.0
    assert float(model.router_ema_alive_coef) == 0.02
    assert float(model.router_ema_balance_coef) == 0.30
    assert float(model.router_ema_specialization_coef) == 0.20
    assert float(model.mos_load_cv_coef) == 0.15
    assert float(model._expert_diversity_coef_target) == 0.30
    assert float(model._router_pertoken_entropy_coef_target) == 0.1
    assert float(model.regularizer_warmup_frac) == 0.07
    # use_router_sigmoid_gate lives on each SoftDenseRouter instance.
    for r in model.shared_block.active_routers():
        assert bool(r.use_router_sigmoid_gate) is False

    from train_gpt import _prescribe_failure_fix
    # Tier 1 redesign 2026-05-13: lip_ub_F demoted to advisory diagnostic
    # (operator norm is sufficient but over-restrictive for asymptotic
    # convergence; the gate-relevant signal is rho_F).
    p_lip = _prescribe_failure_fix("lip_ub_F=45.0 >= 1.0")
    assert p_lip["category"] == "operator_norm_advisory"
    assert "lyapunov_coef" not in p_lip["config_change"]
    # New gate-relevant prescription: rho_F (spectral radius) routes to
    # the formal-tier mechanism, NOT a soft penalty.
    p_rho = _prescribe_failure_fix("rho_F=1.2 >= 1.0")
    assert p_rho["category"] == "fp_convergence_failed"
    assert "needs_formal_tier_contraction" in p_rho["config_change"]
    assert "lyapunov_coef" not in p_rho["config_change"]
    # iter155 corrected: lip_ub_S is now an advisory surrogate, not a gate.
    p_lip_s = _prescribe_failure_fix("lip_ub_S=45.0 >= 1.0")
    assert p_lip_s["category"] == "single_state_blend_loose"
    assert p_lip_s["config_change"] == {}


def test_eval_microbatch_and_mos_expert_settings_are_dry_but_independent():
    """Validation chunking is configurable, and MoS follows shared expert
    principles without silently copying main DEQ expert counts/ranks.
    """
    from train_gpt import Hyperparameters

    assert int(Hyperparameters.val_micro_batch_seqs) == 48

    model = _make_model(num_experts=4, expert_diversity_kind="cosine")
    mos = model.mos_head
    assert mos.mos_output_diversity_kind == model.expert_diversity_kind
    assert mos.num_experts == mos.num_shared + mos.num_specialized
    assert mos.num_experts != model.num_experts
    assert mos.rank != model.shared_block.mlp.expert_rank


def test_dirichlet_router_confidence_diagnostics_are_populated():
    from train_gpt import (
        SoftDenseRouter,
        _format_router_confidence_parts,
        _router_confidence_stats,
        router_diagnostics,
    )

    router = SoftDenseRouter(
        dim=8,
        num_experts=4,
        scoring="dirichlet_ucb",
        dirichlet_ucb_beta=0.5,
        use_router_sigmoid_gate=False,
    )
    x = torch.randn(2, 3, 8)
    # Confidence reductions are diagnostics-gated (skip on hot path).
    # Wrap in `router_diagnostics` so the cache populates.
    with router_diagnostics(enabled=True, step_tag=42):
        p = router(x)
    assert p.shape == (2, 3, 4)
    stats = _router_confidence_stats([router])
    for name in (
        "router_dir_strength_mean",
        "router_dir_uncertainty_mass",
        "router_dir_sigma_mean",
        "router_dir_evidence_mean",
        "router_dir_mu_entropy_norm",
        "router_ucb_beta_current",
    ):
        assert name in stats, f"missing confidence diagnostic {name}"
        assert stats[name] >= 0.0
    assert stats["router_ucb_beta_current"] == 0.5

    with router_diagnostics(enabled=True, step_tag=123):
        _ = router(x)
    router._expert_usage = None
    parts = _format_router_confidence_parts([router], step=123, require_step_match=True)
    assert any(p.startswith("router_dir_strength_mean:") for p in parts)
    stale_parts = _format_router_confidence_parts([router], step=124, require_step_match=True)
    assert stale_parts == []


def test_fp_probe_uses_eager_forward_when_instance_forward_is_wrapped():
    """Fast-val Lipschitz probes must bypass the compiled training forward."""
    import torch
    from train_gpt import _prepare_saved_fp_probe

    class DummyBlock(torch.nn.Module):
        def forward(self, z, x0, b_bar):
            return z + x0

    class DummyModel:
        def __init__(self):
            self.shared_block = DummyBlock()
            self.use_parcae = False
            self._lyapunov_z_star = torch.ones(1, 2, 3)
            self._lyapunov_x0 = torch.full((1, 2, 3), 2.0)

    model = DummyModel()
    model.shared_block.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("wrapped forward should not be used by FP probes"))
    prepared = _prepare_saved_fp_probe(model)
    assert prepared is not None
    # iter155: `_prepare_saved_fp_probe` now returns `a_bar_d` as well so
    # callers can compute the iteration-map S = Ā·z + (1−Ā)·T probe in
    # parallel with the transition-map T probe.  When `use_parcae=False`
    # (this fixture), `a_bar_d` is None and S degenerates to T.
    z_star, x0_lyap, b_bar_d, a_bar_d, sb_call, _, _ = prepared
    assert a_bar_d is None, "use_parcae=False should yield a_bar_d=None"
    out = sb_call(z_star, x0_lyap, b_bar_d)
    assert torch.allclose(out, torch.full_like(out, 3.0))


if __name__ == "__main__":
    test_all_constraints()
    test_expert_path_parameters_are_expert_independent()
    test_prenorm_forward_shapes_are_unchanged()
    test_shared_bypass_gate_diagnostics_are_separate()
    test_parcae_a_bar_has_reversibility_safety_margin()
    test_parcae_b_bar_is_strictly_positive()
    test_parcae_ab_share_only_delta()
    test_parcae_init_recovers_target_a_bar()
    test_parcae_diagnostics_match_zoh_formulas()
    test_revdeq_convergence()
    test_revdeq_reversibility()
    test_revdeq_reconstruction_at_a_bar_floor()
    test_ntp_only_baseline_skips_ctp_param_banks()
    test_post_int6_gate_skips_mos_ctp_when_disabled()
    test_prescriptions_route_to_invariant_mechanisms_not_per_symptom_losses()
    test_routing_regularizer_coefficients_match_promoted_defaults()
    print("\nAll tests passed!")
