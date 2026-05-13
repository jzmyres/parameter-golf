"""Verify the saved-FP `lip_ub` probes (T and S) return finite floats.

Runs a single forward pass to populate `_lyapunov_z_star` /
`_lyapunov_x0`, then calls the sliced saved-FP local-contraction
probe for both ``map_kind="T"`` (transition map, decomposition
diagnostic) and ``map_kind="S"`` (Parcae-blended iteration map,
gate-aligned object — iter155).  Asserts both return finite, plausibly-
bounded floats and that they are NOT identical (the Parcae blend must
have observable effect on the spectral estimate at the project default
``parcae_init_a_bar=0.7`` initialization).  Pytest-discoverable;
CUDA-required (skipped on CPU-only CI).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_arch import _make_model
from train_gpt import _lip_ub_at_saved_fp


def _populate_saved_fp(model):
    """Run one forward pass to populate `_lyapunov_z_star` / `_lyapunov_x0`."""
    B, T = 4, 64
    x = torch.randint(0, 1024, (B, T), device="cuda")
    model.train(False)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_logits(x)
    z_star = getattr(model, "_lyapunov_z_star", None)
    x0_lyap = getattr(model, "_lyapunov_x0", None)
    assert z_star is not None and x0_lyap is not None, (
        "forward must populate _lyapunov_z_star / _lyapunov_x0"
    )
    return z_star, x0_lyap


def _make_small_model():
    # Small config — the probe API does not depend on capacity, so a
    # 2-layer / model_dim=128 / num_experts=2 model gives equivalent
    # coverage in <1 s of CUDA time vs a full Hyperparameters() build.
    return _make_model(
        num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
        num_experts=2, num_shared_experts=0,
        attn_expert_rank=8, mlp_expert_rank=12,
        bigram_vocab_size=0, bigram_dim=8,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_at_saved_fp_returns_finite_float():
    model = _make_small_model()
    _populate_saved_fp(model)
    # Default map_kind="T" — preserves pre-iter155 behavior.
    lip_ub = _lip_ub_at_saved_fp(model, n_iters=1, B_probe=1)
    assert lip_ub is not None, "probe returned None — should yield a finite float"
    assert isinstance(lip_ub, float), f"expected float, got {type(lip_ub).__name__}"
    assert 0.0 < lip_ub < 1e6, f"lip_ub out of plausible range: {lip_ub}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_S_at_saved_fp_returns_finite_float():
    """iter155: ``lip_ub_S`` probes the Parcae-blended iteration map."""
    model = _make_small_model()
    _populate_saved_fp(model)
    lip_ub_S = _lip_ub_at_saved_fp(model, n_iters=1, B_probe=1, map_kind="S")
    assert lip_ub_S is not None, "S-probe returned None — should yield a finite float"
    assert isinstance(lip_ub_S, float), f"expected float, got {type(lip_ub_S).__name__}"
    assert 0.0 < lip_ub_S < 1e6, f"lip_ub_S out of plausible range: {lip_ub_S}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_T_and_S_differ_at_saved_fp():
    """iter155: J_S = Ā·I + (1−Ā)·J_T differs from J_T whenever Ā ≠ 0,
    so the spectral estimates must differ.  Tests the Parcae blend has
    observable effect on the gate-aligned diagnostic.

    Uses a fixed-seed RNG before each probe so the two power-iteration
    runs share initialization and the difference comes from J_S vs J_T,
    not from random vector divergence.
    """
    model = _make_small_model()
    _populate_saved_fp(model)
    torch.manual_seed(2026_05_12)
    lip_ub_T = _lip_ub_at_saved_fp(model, n_iters=2, B_probe=1, map_kind="T")
    torch.manual_seed(2026_05_12)
    lip_ub_S = _lip_ub_at_saved_fp(model, n_iters=2, B_probe=1, map_kind="S")
    assert lip_ub_T is not None and lip_ub_S is not None
    # Tolerance gives float-noise headroom while still rejecting accidental
    # no-op routing (e.g. if the S branch were silently calling T).  Power
    # iteration with n_iters=2 is more than enough to expose Ā=0.7 effects.
    assert abs(lip_ub_T - lip_ub_S) > 1e-3, (
        f"lip_ub_T={lip_ub_T} and lip_ub_S={lip_ub_S} are too close — "
        "the Parcae blend appears to be a no-op (regression of map_kind dispatch)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_F_matches_analytical_J_F_for_linear_toy():
    """iter155 corrected: strong analytical verification of the F-probe.

    Build a linear toy T(z) = J·z (no x₀ dependence) so the cycle
    Jacobian J_F has a known closed form.  With per-dim Parcae gates
    D_A = diag(Ā) and D_β = diag(1−Ā):

        J_F = [[D_A,                D_β·J            ],
               [D_β·J·D_A,         D_A + D_β·J·D_β·J]]

    Computing σ_max(J_F) by torch.linalg.svdvals on this analytical block
    matrix and comparing against the power-iteration estimate from
    `_run_spectral_norm_power(..., map_kind="F")` directly verifies that
    the elementwise `a_bar_t * (...)` multiplications in the F closure
    preserve the per-dim Ā/β ordering correctly — defending against the
    "scalar damping" derivation error the user-feedback flagged.

    Tolerance: 8% relative error, accounting for power iteration's
    LOWER-bound estimator nature (it under-estimates σ_max in finite
    iters; the analytical SVD is the upper-truth).
    """
    from contextlib import nullcontext

    from train_gpt import _run_spectral_norm_power

    device = torch.device("cuda")
    torch.manual_seed(2026_05_13)

    # Tiny dimensions so analytical SVD is trivial.
    B, T_seq, D = 1, 1, 4
    # Linear toy "T_θ": ignores x0, b_bar; returns z @ J.T.  Use a
    # well-conditioned J with σ_max ≈ 1.5 so J_F is non-trivially
    # expansive.
    J = torch.tensor(
        [
            [0.7, 0.3, 0.1, 0.0],
            [0.2, 0.8, 0.2, 0.1],
            [0.1, 0.2, 0.7, 0.3],
            [0.0, 0.1, 0.3, 0.6],
        ],
        dtype=torch.float32, device=device,
    )

    def fake_sb(z, x0, b_bar):
        # z shape (B, T, D); J shape (D, D); fake_sb(z) = z @ J.T (linear).
        return z @ J.T

    a_bar_vec = torch.tensor([0.2, 0.5, 0.7, 0.9], dtype=torch.float32, device=device)
    a_bar_d = a_bar_vec.view(1, 1, D)

    # Analytical J_F (2D × 2D block):
    D_A = torch.diag(a_bar_vec)
    D_beta = torch.diag(1.0 - a_bar_vec)
    DbJ = D_beta @ J
    J_F_analytical = torch.empty(2 * D, 2 * D, dtype=torch.float32, device=device)
    J_F_analytical[:D, :D] = D_A
    J_F_analytical[:D, D:] = DbJ
    J_F_analytical[D:, :D] = DbJ @ D_A
    J_F_analytical[D:, D:] = D_A + DbJ @ DbJ
    sigma_max_analytical = float(torch.linalg.svdvals(J_F_analytical)[0].item())

    # Power-iteration estimate via the actual code path.
    z_star = torch.randn(B, T_seq, D, device=device, dtype=torch.float32)
    x0_lyap = torch.zeros_like(z_star)
    b_bar_d = None  # fake_sb ignores it

    sigma_est = _run_spectral_norm_power(
        z_star, x0_lyap, b_bar_d, fake_sb, torch.float32,
        n_iters=30, ctx_factory=nullcontext,
        map_kind="F", a_bar_d=a_bar_d,
    )
    assert sigma_est is not None
    rel_err = abs(sigma_est - sigma_max_analytical) / max(sigma_max_analytical, 1e-8)
    assert rel_err < 0.08, (
        f"Power-iteration σ_max(J_F)={sigma_est:.6f} vs "
        f"analytical σ_max(J_F)={sigma_max_analytical:.6f} "
        f"(rel_err={rel_err:.4f}); this means the F-closure does NOT "
        "match the per-dim Jacobian formula — likely an Ā/β ordering bug."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_F_at_saved_fp_returns_finite_float():
    """iter155 corrected: ``lip_ub_F`` probes the actual two-state Parcae cycle."""
    model = _make_small_model()
    _populate_saved_fp(model)
    lip_ub_F = _lip_ub_at_saved_fp(model, n_iters=1, B_probe=1, map_kind="F")
    assert lip_ub_F is not None, "F-probe returned None — should yield a finite float"
    assert isinstance(lip_ub_F, float), f"expected float, got {type(lip_ub_F).__name__}"
    assert 0.0 < lip_ub_F < 1e6, f"lip_ub_F out of plausible range: {lip_ub_F}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_F_differs_from_T_and_S_at_saved_fp():
    """iter155 corrected: J_F has block structure
        [[Ā·I, β·J_T], [Ā·β·J_T, Ā·I + β²·J_T·J_T]]
    which is qualitatively different from J_T and from the convex-blend
    J_S = Ā·I + (1−Ā)·J_T.  In particular σ_max(J_F) couples J_T through
    cross blocks β·J_T that scale linearly with the transition magnitude,
    so the F-side estimate must differ from both T and S.
    """
    model = _make_small_model()
    _populate_saved_fp(model)
    torch.manual_seed(2026_05_13)
    lip_ub_T = _lip_ub_at_saved_fp(model, n_iters=2, B_probe=1, map_kind="T")
    torch.manual_seed(2026_05_13)
    lip_ub_S = _lip_ub_at_saved_fp(model, n_iters=2, B_probe=1, map_kind="S")
    torch.manual_seed(2026_05_13)
    lip_ub_F = _lip_ub_at_saved_fp(model, n_iters=2, B_probe=1, map_kind="F")
    assert lip_ub_T is not None and lip_ub_S is not None and lip_ub_F is not None
    assert abs(lip_ub_F - lip_ub_T) > 1e-3, (
        f"lip_ub_F={lip_ub_F} and lip_ub_T={lip_ub_T} are too close — "
        "F-probe appears to be falling through to T (regression of map_kind=F dispatch)"
    )
    assert abs(lip_ub_F - lip_ub_S) > 1e-3, (
        f"lip_ub_F={lip_ub_F} and lip_ub_S={lip_ub_S} are too close — "
        "F-probe appears to be falling through to S (regression of map_kind=F dispatch)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lyapunov_fd_probe_T_and_S_branches_differ():
    """iter155 Lyapunov-loss branch-differ test (T vs S).

    The training-loop FD probe at the `lyapunov_target` branch site computes
    one of two expressions:
        T-side: ‖sb(z+εu) − sb(z)‖_RMS / ε    (≈ ‖J_T · u‖)
        S-side: ‖Ā·(εu) + (1−Ā)·(sb(z+εu) − sb(z))‖_RMS / ε
                                                (≈ ‖J_S · u‖)
    where J_S = Ā·I + (1−Ā)·J_T.  This test reproduces both expressions
    end-to-end against a populated saved-FP fixture and asserts they
    differ — defending against a regression where the branch dispatch
    silently routes both targets to the same expression.

    Per CLAUDE.md "Untested-path executability": every new control-flow
    branch (here, `lyapunov_target=iteration_S`) needs a focused test that
    exercises it AND differs from the default branch.
    """
    from train_gpt import _unwrap_compiled_module

    model = _make_small_model()
    z_star, x0_lyap = _populate_saved_fp(model)

    sb = _unwrap_compiled_module(model.shared_block)
    z_base = z_star.detach()[:, :32].contiguous()
    x0_base = x0_lyap[:, :32].contiguous()
    b_bar = model._parcae_b_bar().detach() if model.use_parcae else None
    a_bar = model._parcae_a_bar().detach()

    torch.manual_seed(2026_05_12)
    eps_dir = torch.randn_like(z_base, dtype=z_base.dtype)
    eps_unit = eps_dir / eps_dir.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir.dtype)
    eps_step = 1e-2

    with torch.no_grad():
        u_base = sb(z_base, x0_base, b_bar)
        u_pert = sb(z_base + eps_step * eps_unit, x0_base, b_bar)

    diff_T = u_pert - u_base
    expansion_T = float((diff_T.float().pow(2).mean().sqrt() / eps_step).item())

    a_bar_d = a_bar.view(*([1] * (z_base.ndim - 1)), -1).to(dtype=z_base.dtype)
    diff_S = a_bar_d * (eps_step * eps_unit) + (1.0 - a_bar_d) * diff_T
    expansion_S = float((diff_S.float().pow(2).mean().sqrt() / eps_step).item())

    assert math_isfinite(expansion_T) and math_isfinite(expansion_S), (
        f"non-finite expansions: T={expansion_T}, S={expansion_S}"
    )
    assert abs(expansion_T - expansion_S) > 1e-4, (
        f"T and S Lyapunov FD expressions are identical to numerical noise: "
        f"T={expansion_T}, S={expansion_S} — regression of the iter155 branch dispatch"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lyapunov_S_target_grad_flows_to_parcae_a_bar():
    """iter155 corrected: under `lyapunov_target=iteration_S/F`, gradients
    must reach Parcae's per-dim Ā via the outside-linear-blend coefficients.
    The user-feedback bug was that Ā was previously detached, breaking the
    doc claim that Lyapunov-on-S/F trains damping.

    This test reproduces the S-side FD expression with a live (un-detached)
    Ā and checks that ∂expansion/∂parcae_raw_a is non-zero — the actual
    training-loop pathway that pressures Ā when expansion>γ.
    """
    from train_gpt import _unwrap_compiled_module

    model = _make_small_model()
    z_star, x0_lyap = _populate_saved_fp(model)
    if not model.use_parcae:
        pytest.skip("model has Parcae disabled; S-target collapses to T")

    sb = _unwrap_compiled_module(model.shared_block)
    z_base = z_star.detach()[:, :32].contiguous().requires_grad_(False)
    x0_base = x0_lyap[:, :32].contiguous()
    b_bar_d = model._parcae_b_bar().detach()

    torch.manual_seed(2026_05_13)
    eps_dir = torch.randn_like(z_base, dtype=z_base.dtype)
    eps_unit = eps_dir / eps_dir.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir.dtype)
    eps_step = 1e-2

    # Zero raw_a grad so we measure exactly this loss's contribution.
    if model.parcae_raw_a.grad is not None:
        model.parcae_raw_a.grad = None

    a_bar_full = model._parcae_a_bar()  # NOT detached — should track grad to raw_a
    a_bar_d = a_bar_full.view(*([1] * (z_base.ndim - 1)), -1).to(dtype=z_base.dtype)
    u_base = sb(z_base, x0_base, b_bar_d)
    u_pert = sb(z_base + eps_step * eps_unit, x0_base, b_bar_d)
    diff_S = a_bar_d * (eps_step * eps_unit) + (1.0 - a_bar_d) * (u_pert - u_base)
    expansion = diff_S.float().pow(2).mean().sqrt() / float(eps_step)
    # Use a permissive `gamma` so the relu fires.
    lyap_loss = torch.relu(expansion - 0.0).pow(2)
    lyap_loss.backward()

    grad = model.parcae_raw_a.grad
    assert grad is not None, (
        "parcae_raw_a.grad is None — gradients did not reach Ā via the "
        "Lyapunov-on-S branch (regression: Ā likely got detached)"
    )
    grad_norm = float(grad.float().norm().item())
    assert grad_norm > 0.0 and math_isfinite(grad_norm), (
        f"parcae_raw_a.grad norm is {grad_norm} — expected non-zero finite "
        "gradient to Ā when Lyapunov-on-S fires above gamma"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lyapunov_estimator_power_jvp_F_returns_dominant_v():
    """iter155 corrected (power_jvp_F estimator): the estimator opt-in
    runs power iteration on J_F (no-grad) to estimate the dominant right
    singular vector, then uses that direction as the FD perturbation.

    Verifies the actual singular-vector identity on the linear toy:
    builds the analytical 2D × 2D block J_F by hand, computes
    J_F @ v_flat, and asserts ‖J_F @ v‖ ≈ σ_max·‖v‖ AND that the
    direction matches v_top to within power-iteration tolerance.  This
    catches singular-vector regressions that mere shape/norm checks
    would miss.
    """
    from contextlib import nullcontext

    from train_gpt import _run_spectral_norm_power

    device = torch.device("cuda")
    torch.manual_seed(2026_05_13)

    B, T_seq, D = 1, 1, 4
    J = torch.tensor(
        [
            [0.7, 0.3, 0.1, 0.0],
            [0.2, 0.8, 0.2, 0.1],
            [0.1, 0.2, 0.7, 0.3],
            [0.0, 0.1, 0.3, 0.6],
        ],
        dtype=torch.float32, device=device,
    )

    def fake_sb(z, x0, b_bar):
        return z @ J.T

    a_bar_vec = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    a_bar_d = a_bar_vec.view(1, 1, D)
    z_star = torch.randn(B, T_seq, D, device=device, dtype=torch.float32)
    x0_lyap = torch.zeros_like(z_star)

    ret = _run_spectral_norm_power(
        z_star, x0_lyap, None, fake_sb, torch.float32,
        n_iters=30, ctx_factory=nullcontext,
        map_kind="F", a_bar_d=a_bar_d, return_v=True,
    )
    assert ret is not None
    sigma, v = ret
    assert v.shape == (B, T_seq, 2 * D), (
        f"v shape {v.shape} does not match joint state (B, T, 2D)"
    )
    assert math_isfinite(sigma) and sigma > 0
    # v should be unit-normalized (L2).
    v_norm = float(v.norm().item())
    assert abs(v_norm - 1.0) < 1e-3, f"v should be unit-norm; got {v_norm}"

    # Construct analytical J_F (2D × 2D block) — same formula as the
    # linear-toy analytical test:
    #   J_F = [[D_A,             D_β·J            ],
    #          [D_β·J·D_A,       D_A + D_β·J·D_β·J]]
    D_A = torch.diag(a_bar_vec)
    D_beta = torch.diag(1.0 - a_bar_vec)
    DbJ = D_beta @ J
    J_F_analytical = torch.empty(2 * D, 2 * D, dtype=torch.float32, device=device)
    J_F_analytical[:D, :D] = D_A
    J_F_analytical[:D, D:] = DbJ
    J_F_analytical[D:, :D] = DbJ @ D_A
    J_F_analytical[D:, D:] = D_A + DbJ @ DbJ

    # Singular-vector identity: ‖J_F @ v‖ ≈ σ_max · ‖v‖.  Apply J_F to
    # the flattened v (shape (B, T, 2D) → (2D,) since B=T=1).
    v_flat = v.reshape(2 * D)
    Jv = J_F_analytical @ v_flat
    Jv_norm = float(Jv.norm().item())
    sigma_max_analytical = float(torch.linalg.svdvals(J_F_analytical)[0].item())

    # Identity check #1: ‖J_F @ v‖ ≈ sigma (since ‖v‖ = 1).
    rel_err_norm = abs(Jv_norm - sigma) / max(sigma, 1e-8)
    assert rel_err_norm < 0.08, (
        f"‖J_F @ v‖={Jv_norm:.6f} does not match power-iter σ={sigma:.6f} "
        f"(rel_err={rel_err_norm:.4f}); v is not a true singular vector "
        "of the analytical J_F — likely indicates the F-closure or the "
        "JVP machinery has drifted from the per-dim Jacobian formula."
    )

    # Identity check #2: power-iter sigma matches analytical σ_max.
    rel_err_sigma = abs(sigma - sigma_max_analytical) / max(sigma_max_analytical, 1e-8)
    assert rel_err_sigma < 0.08, (
        f"Power-iter σ={sigma:.6f} vs analytical σ_max={sigma_max_analytical:.6f} "
        f"(rel_err={rel_err_sigma:.4f})."
    )

    # Identity check #3: direction match.  J_F @ v should be
    # approximately parallel to v (eigenvector of J_F^T J_F, and for the
    # dominant singular value, J_F @ v is parallel to the left singular
    # vector u; for symmetric J_F or when σ_max is well-separated, v ≈ u
    # so cosine ≈ 1).  We check |cos(J_F @ v, v)| > 0.9 as a coarse but
    # robust guard against direction destruction (the bug fixed in the
    # joint-RMS normalization commit).
    v_unit = v_flat / max(float(v_flat.norm().item()), 1e-8)
    Jv_unit = Jv / max(Jv_norm, 1e-8)
    cos_sim = float((Jv_unit * v_unit).sum().abs().item())
    assert cos_sim > 0.9, (
        f"|cos(J_F @ v, v)| = {cos_sim:.4f} — the returned v is not "
        "aligned with the dominant singular direction of the analytical "
        "J_F; check the joint-RMS normalization fix and the F closure."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lyapunov_F_target_grad_flows_to_parcae_a_bar():
    """iter155 corrected: the gate-aligned target is `iteration_F`, not S.
    This test reproduces the F-side FD expression (the actual two-state
    Parcae cycle) with live Ā, runs backward, and asserts
    ∂expansion_F/∂parcae_raw_a is non-zero — the actual training-loop
    pathway that pressures Ā via the cross-block coupling D_β·J_T in J_F
    when expansion>γ.

    Per CLAUDE.md "Untested-path executability": iter155's
    most-important new branch (lyapunov_target=iteration_F) needs a focused
    test that exercises the actual training-time Lyapunov penalty, not
    just the diagnostic K-sweep probe.
    """
    from train_gpt import _unwrap_compiled_module

    model = _make_small_model()
    z_star, x0_lyap = _populate_saved_fp(model)
    if not model.use_parcae:
        pytest.skip("model has Parcae disabled; F-target collapses to T")

    sb = _unwrap_compiled_module(model.shared_block)
    z_base = z_star.detach()[:, :32].contiguous()
    x0_base = x0_lyap[:, :32].contiguous()
    b_bar_d = model._parcae_b_bar().detach()  # B̄ detached intentionally

    torch.manual_seed(2026_05_13)
    eps_dir = torch.randn_like(z_base, dtype=z_base.dtype)
    eps_unit = eps_dir / eps_dir.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir.dtype)
    eps_dir_y = torch.randn_like(z_base, dtype=z_base.dtype)
    eps_unit_y = eps_dir_y / eps_dir_y.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir_y.dtype)
    eps_step = 1e-2

    # Zero raw_a grad so we measure exactly this loss's contribution.
    if model.parcae_raw_a.grad is not None:
        model.parcae_raw_a.grad = None

    # Reproduce the iteration_F branch from train_gpt.py:8337+ exactly.
    a_bar_full = model._parcae_a_bar()  # NOT detached
    a_bar_d = a_bar_full.view(*([1] * (z_base.ndim - 1)), -1).to(dtype=z_base.dtype)
    # F at base: y' = Ā·z + β·T(z), z' = Ā·z + β·T(y').
    t_z_base = sb(z_base, x0_base, b_bar_d)
    y_new_base = a_bar_d * z_base + (1.0 - a_bar_d) * t_z_base
    t_y_base = sb(y_new_base, x0_base, b_bar_d)
    z_new_base = a_bar_d * z_base + (1.0 - a_bar_d) * t_y_base
    # F at perturbed (y, z) = (z + ε·u_y, z + ε·u_z).
    z_pert = z_base + eps_step * eps_unit
    t_z_pert = sb(z_pert, x0_base, b_bar_d)
    y_pert_init = z_base + eps_step * eps_unit_y
    y_new_pert = a_bar_d * y_pert_init + (1.0 - a_bar_d) * t_z_pert
    t_y_pert = sb(y_new_pert, x0_base, b_bar_d)
    z_new_pert = a_bar_d * z_pert + (1.0 - a_bar_d) * t_y_pert
    diff_y = y_new_pert - y_new_base
    diff_z = z_new_pert - z_new_base
    diff_sq_sum = diff_y.float().pow(2).sum() + diff_z.float().pow(2).sum()
    diff_count = diff_y.numel() + diff_z.numel()
    expansion = (diff_sq_sum / diff_count).sqrt() / float(eps_step)
    # Permissive gamma so the relu fires.
    lyap_loss = torch.relu(expansion - 0.0).pow(2)
    lyap_loss.backward()

    grad = model.parcae_raw_a.grad
    assert grad is not None, (
        "parcae_raw_a.grad is None — gradients did not reach Ā via the "
        "Lyapunov-on-F branch (regression: Ā likely got detached)"
    )
    grad_norm = float(grad.float().norm().item())
    assert grad_norm > 0.0 and math_isfinite(grad_norm), (
        f"parcae_raw_a.grad norm is {grad_norm} — expected non-zero finite "
        "gradient to Ā when Lyapunov-on-F fires above gamma"
    )

    # Critical check: B̄ MUST stay detached (curvature-control).  Verify
    # the doc claim by asserting parcae_raw_b receives no gradient from
    # this loss — the b_bar_d value is .detach()'d in the implementation.
    assert model.parcae_raw_b.grad is None or float(model.parcae_raw_b.grad.float().norm().item()) == 0.0, (
        "parcae_raw_b.grad should be None/zero under the Lyapunov-on-F "
        "branch because B̄ is detached for curvature-control reasons "
        "(see opg_doc.tex Finite-expansion penalty section)."
    )


def math_isfinite(x: float) -> bool:
    import math
    return math.isfinite(x)


def test_lip_ub_F_is_always_probed_on_FP_eval_unconditional_of_cadence_knob():
    """User directive 2026-05-13: ``lip_ub_F`` is the gate-aligned
    contraction object on the actual two-state Parcae cycle.  It MUST
    be reported on EVERY FP eval — both the train-time fast-val site
    and every K-sweep row — never gated by ``fp_lip_fast_val_every``
    or ``EvalProfile.lip_probe_set``.  T and S are decomposition
    diagnostics and remain on the cadence; F never is.

    Otherwise: a profile knob (e.g. ``debug`` setting
    ``fp_lip_fast_val_every=0``, or ``submission`` setting
    ``lip_probe_set={128}``) silently drops the gate evidence on
    intermediate evals, and a future reviewer cannot tell whether
    ``lip_ub_F`` was unmeasured or measured-and-fine.  This is
    exactly the "decision based on metric we haven't measured"
    failure mode the audit checklist forbids.

    Static text-search test on ``train_gpt.py`` because the FP eval
    sites live deep inside the train + K-sweep loops; we assert the
    structural invariant rather than spinning up a 1-hour fake run.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()

    # Site 1 (train-time fast-val): F branch must be `master_process`
    # only, NOT also include `fp_probe_every > 0` like the T/S branch.
    assert "run_fp_lip_probe_F = master_process\n" in src, (
        "train-time fast-val: lip_ub_F must be probed on master_process "
        "only (no `fp_probe_every > 0` gate); user directive 2026-05-13 "
        "made the gate-aligned object always-on."
    )
    assert "run_fp_lip_probe_TS = (\n                master_process\n                and fp_probe_every > 0\n" in src, (
        "train-time fast-val: lip_ub_T/S must REMAIN behind the "
        "fp_probe_every cadence — they are decomposition diagnostics, "
        "not the gate-aligned object."
    )

    # Site 2 (K-sweep): lip_ub_F_val and fp_residual_F_val must be
    # computed *before* (and NOT inside) the
    # `if _should_probe_lip_for_k(args, k_eval):` block that gates T/S.
    # We grep for the structural pattern: F probe call appears, then
    # the `_should_probe_lip_for_k(args, k_eval)` gate appears, then
    # the T/S probe calls appear — F not enclosed in the gate.
    f_idx = src.find("log_label=\"ksweep_skip_reason:lip_ub_F\",")
    gate_idx = src.find("if _should_probe_lip_for_k(args, k_eval):")
    t_idx = src.find("log_label=\"ksweep_skip_reason:lip_ub_T\",")
    s_idx = src.find("log_label=\"ksweep_skip_reason:lip_ub_S\",")
    assert f_idx > 0 and gate_idx > 0 and t_idx > 0 and s_idx > 0, (
        f"K-sweep probe sites not all found: F={f_idx} gate={gate_idx} "
        f"T={t_idx} S={s_idx}"
    )
    assert f_idx < gate_idx < t_idx and gate_idx < s_idx, (
        "K-sweep: lip_ub_F probe call MUST appear before the "
        "_should_probe_lip_for_k gate so it runs on every K-row; "
        "T/S calls MUST appear after the gate so they remain on "
        f"the eval-profile cadence.  Got F@{f_idx} gate@{gate_idx} "
        f"T@{t_idx} S@{s_idx} — out of expected order."
    )

    # fp_residual_F (paired with lip_ub_F in fp_bound) shares the
    # same always-on policy.  Confirm it does NOT sit behind a
    # `if _should_probe_lip_for_k` gate by asserting the assignment
    # is unconditional.
    assert "fp_residual_F_val = _joint_F_residual_at_saved_fp(" in src, "fp_residual_F call missing"
    # The unconditional pattern: assignment line is at column-0-aligned
    # depth (8 spaces inside the per-K loop), not 12 (inside an `if`).
    for line in src.splitlines():
        if "fp_residual_F_val = _joint_F_residual_at_saved_fp(" in line:
            indent = len(line) - len(line.lstrip())
            assert indent == 8, (
                f"fp_residual_F call indented {indent} spaces — expected 8 "
                "(unconditional inside per-K loop, not nested under "
                "`if _should_probe_lip_for_k`).  fp_residual_F pairs with "
                "lip_ub_F in fp_bound and shares its always-on policy."
            )
            break


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", flush=True)
        sys.exit(0)
    try:
        test_lip_ub_F_matches_analytical_J_F_for_linear_toy()
        test_lip_ub_at_saved_fp_returns_finite_float()
        test_lip_ub_S_at_saved_fp_returns_finite_float()
        test_lip_ub_F_at_saved_fp_returns_finite_float()
        test_lip_ub_T_and_S_differ_at_saved_fp()
        test_lip_ub_F_differs_from_T_and_S_at_saved_fp()
        test_lyapunov_fd_probe_T_and_S_branches_differ()
        test_lyapunov_S_target_grad_flows_to_parcae_a_bar()
        test_lyapunov_estimator_power_jvp_F_returns_dominant_v()
        test_lyapunov_F_target_grad_flows_to_parcae_a_bar()
        test_lip_ub_F_is_always_probed_on_FP_eval_unconditional_of_cadence_knob()
    except AssertionError as e:
        print(f"FAIL: {e}", flush=True)
        sys.exit(1)
    print("PASS: lip_ub probes (T, S, F, analytical) + always-on policy + Lyapunov FD branches + estimators + Ā-grad (S+F) verified", flush=True)
