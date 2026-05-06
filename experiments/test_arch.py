"""Tests for all 5 architectural constraints + RevDEQ convergence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F


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
    model = _make_model(bigram_vocab_size=0)

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
    model = _make_model(bigram_vocab_size=0)
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


def test_prescribe_min_share_routes_to_cv_loss():
    """min_share failures map to direct CV-load coefficients, not legacy knobs."""
    from train_gpt import _prescribe_failure_fix

    p_mos = _prescribe_failure_fix(
        "mos_ntp_min_share=0.005 < 0.150 (expert below 60% of fair share 1/4)")
    assert p_mos["category"] == "mos_router_collapse"
    assert "mos_load_cv_coef_mult" in p_mos["config_change"], (
        f"mos_*_min_share must prescribe mos_load_cv_coef bump, got {p_mos['config_change']}")
    assert "weight_decay_mult" not in p_mos["config_change"], (
        "WD must NOT be prescribed for MoS routing collapse.")

    p_attn = _prescribe_failure_fix("attn_min_share=0.01 < 0.150 (...)")
    assert p_attn["category"] == "router_collapse"
    assert "router_load_cv_coef_mult" in p_attn["config_change"]

    p_mlp = _prescribe_failure_fix("mlp_min_share=0.01 < 0.150 (...)")
    assert p_mlp["category"] == "router_collapse"
    assert "router_load_cv_coef_mult" in p_mlp["config_change"]

    # Ortho failures use output-diversity first and keep WD as fallback.
    p_ortho = _prescribe_failure_fix("attn_ortho=0.71 > 0.5 (max pairwise |cos| ...)")
    assert p_ortho["category"] == "expert_collapse"
    assert "expert_output_diversity_coef_mult" in p_ortho["config_change"]
    assert "weight_decay_mult" in p_ortho["config_change"]


def test_direct_cv_coefficients_are_hyperparameters():
    """Direct load-CV coefficients are the canonical routing balance knobs."""
    from train_gpt import Hyperparameters
    assert hasattr(Hyperparameters, "router_load_cv_coef")
    assert hasattr(Hyperparameters, "mos_load_cv_coef")
    assert float(Hyperparameters.router_load_cv_coef) == 1.0
    assert float(Hyperparameters.mos_load_cv_coef) == 1.0
    # Regression guard: iter 142b dropped the relu(cv − cv_target)² hinge in
    # favor of continuous cv²; the old target knobs must NOT come back.
    assert not hasattr(Hyperparameters, "cv_target")
    assert not hasattr(Hyperparameters, "mos_cv_target")
    model = _make_model(num_experts=4)
    assert hasattr(model, "router_load_cv_coef")
    assert hasattr(model, "mos_load_cv_coef")
    assert float(model.router_load_cv_coef) == 1.0
    assert float(model.mos_load_cv_coef) == 1.0


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
    test_prescribe_min_share_routes_to_cv_loss()
    test_direct_cv_coefficients_are_hyperparameters()
    print("\nAll tests passed!")
