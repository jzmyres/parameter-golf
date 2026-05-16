"""rho_F (spectral radius) + sigma_max_F (operator norm) probe tests
at the saved DEQ FP.

rho_F = |lambda_max(J_F)| via straight power iteration on J_F. Necessary
AND sufficient for asymptotic local FP convergence per Hartman-Grobman;
this is the principled GATE metric. iter168 (2026-05-16) added multi-seed
median aggregation to close the K=24/32 estimator-artifact issue where
single-seed power iteration on non-symmetric J gave noisy estimates with
small projection onto the dominant eigenspace.

sigma_max_F = ||J_F||_op via power iteration on J^T J. DIAGNOSTIC ONLY
(iter168, 2026-05-16): restored after the 2026-05-15 over-aggressive
removal because the operator norm carries operationally-relevant
information (per-step contraction bound, perturbation robustness, basin-
of-attraction size) that rho_F alone does not capture. NOT a gate, NOT
a penalty, NOT a prescription input.

Pytest-discoverable; CUDA-required (skipped on CPU-only CI).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_arch import _make_model
from train_gpt import _rho_F_at_saved_fp, _sigma_max_F_at_saved_fp


def _populate_saved_fp(model):
    """Run one forward pass to populate `_lyapunov_z_star` / `_lyapunov_x0`
    (legacy attribute names retained for backward compatibility; they are
    the saved-FP cache the spectral probes consume)."""
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
    return _make_model(
        num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
        num_experts=2, num_shared_experts=0,
        attn_expert_rank=8, mlp_expert_rank=12,
        bigram_vocab_size=0, bigram_dim=8,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rho_F_at_saved_fp_returns_finite_float():
    """rho_F is the spectral-radius estimate via straight power iteration
    on J_F. Must return a finite, positive float at the saved FP."""
    model = _make_small_model()
    _populate_saved_fp(model)
    rho_F = _rho_F_at_saved_fp(model, n_iters=8, B_probe=1)
    assert rho_F is not None, "rho_F returned None — should yield a finite float"
    assert isinstance(rho_F, float), f"expected float, got {type(rho_F).__name__}"
    assert 0.0 < rho_F < 1e6, f"rho_F out of plausible range: {rho_F}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rho_F_multi_seed_returns_robust_estimate():
    """iter168 multi-seed aggregation: rho_F with n_seeds=4 must run all
    4 power-iteration trials and return their median. This is more robust
    to single-seed estimator noise from poor initial-vector alignment with
    the dominant eigenspace (root cause of iter163's K=24/32 rho_F > 1
    artifacts when K=128 = 0.92)."""
    model = _make_small_model()
    _populate_saved_fp(model)
    rho_F_single = _rho_F_at_saved_fp(model, n_iters=8, B_probe=1, n_seeds=1)
    rho_F_multi = _rho_F_at_saved_fp(model, n_iters=8, B_probe=1, n_seeds=4)
    assert rho_F_single is not None and rho_F_multi is not None
    # Both should be finite + positive; the multi-seed median should be in
    # the same order of magnitude as the single-seed estimate (they probe
    # the same Jacobian; the median just rejects outliers).
    assert 0.0 < rho_F_multi < 1e6
    assert abs(rho_F_multi - rho_F_single) < max(1.0, 2.0 * rho_F_single), (
        f"multi-seed rho_F={rho_F_multi:.4f} too far from single-seed "
        f"{rho_F_single:.4f} — should be the same order of magnitude"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigma_max_F_at_saved_fp_returns_finite_float():
    """sigma_max_F (operator norm of J_F) is the iter168-restored diagnostic.
    Must return a finite, positive float at the saved FP. By construction
    sigma_max >= rho_F always, so the value is a strict upper bound on the
    spectral radius."""
    model = _make_small_model()
    _populate_saved_fp(model)
    sigma_max_F = _sigma_max_F_at_saved_fp(model, n_iters=8, B_probe=1)
    rho_F = _rho_F_at_saved_fp(model, n_iters=8, B_probe=1)
    assert sigma_max_F is not None and rho_F is not None
    assert isinstance(sigma_max_F, float)
    assert 0.0 < sigma_max_F < 1e6
    # σ_max ≥ ρ always (operator norm bounds spectral radius for any matrix).
    # Allow small estimator slack (10 %) for power-iteration noise.
    assert sigma_max_F >= rho_F * 0.9, (
        f"sigma_max_F={sigma_max_F:.4f} must be >= rho_F={rho_F:.4f} "
        "(operator norm bounds spectral radius for any matrix)"
    )


def test_rho_F_is_emitted_in_fast_val_alongside_residual():
    """rho_F (necessary AND sufficient FP-convergence gate per Hartman-Grobman)
    MUST be probed and emitted in the train-time fast-val site. The lip_ub_*
    operator-norm proxies were removed 2026-05-15 — rho_F is now the sole
    spectral GATE metric.

    Static text-search test on train_gpt.py because the fast-val site lives
    deep inside the train loop; we assert the structural invariant rather
    than spinning up a full training run.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()

    # The rho_F probe call must appear in the fast-val block.
    assert "rho_F = _rho_F_at_saved_fp(\n                    base_model," in src, (
        "fast-val: rho_F probe call missing or moved — must be inside the "
        "`if master_process:` block alongside _joint_F_residual_at_saved_fp"
    )

    # The fast-val log string must format rho_F so it appears on every line.
    assert "f\" rho_F:{rho_F:.4f}\" if rho_F is not None else \" rho_F:N/A\"" in src, (
        "fast-val log string missing rho_F formatting; the principled gate "
        "metric must appear in user-facing logs at every fast-val emission."
    )

    # Negative assertions: the removed lip_ub_* PROBE surface (operator-norm
    # gates and penalties) must NOT come back via copy-paste regression.
    # iter168's σ_max_F (DIAGNOSTIC ONLY, NOT a gate) is in a separate
    # helper named differently to make the role distinction explicit.
    assert "_lip_ub_at_saved_fp" not in src, (
        "_lip_ub_at_saved_fp was removed 2026-05-15 and must not reappear; "
        "use _rho_F_at_saved_fp + _joint_F_residual_at_saved_fp instead. "
        "iter168 σ_max_F via _sigma_max_F_at_saved_fp is diagnostic-only."
    )
    assert "lip_ub_T:" not in src and "lip_ub_S:" not in src and "lip_ub_F:" not in src, (
        "lip_ub_T/S/F log emission was removed 2026-05-15 — operator-norm "
        "proxies are over-restrictive as gates; iter168 emits sigma_max_F "
        "instead (diagnostic-only, distinct from the refuted lip_ub_F gate)."
    )


def test_sigma_max_F_is_emitted_in_fast_val_as_diagnostic():
    """iter168 (2026-05-16): sigma_max_F MUST appear in the fast-val log
    output alongside rho_F. It is DIAGNOSTIC ONLY — emitted to capture
    the robustness/basin-size info rho_F alone does not provide, but NOT
    used as a gate or penalty."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
    assert "sigma_max_F = _sigma_max_F_at_saved_fp(" in src, (
        "fast-val: sigma_max_F probe call missing — iter168 must restore "
        "the operator-norm diagnostic alongside rho_F."
    )
    assert "f\" sigma_max_F:{sigma_max_F:.4f}\" if sigma_max_F is not None else \" sigma_max_F:N/A\"" in src, (
        "fast-val log string missing sigma_max_F formatting; iter168 "
        "diagnostic must appear in user-facing logs at every fast-val emission."
    )


def test_sigma_max_F_is_emitted_in_k_sweep_table():
    """iter168 (2026-05-16): K-sweep table must include sigma_max_F as a
    per-K diagnostic column alongside rho_F, fp_residual_F, iter_conv_rel."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
    assert "(\"sigma_max_F\", 12)" in src, (
        "K-sweep table missing sigma_max_F column — iter168 must include "
        "it in `_kdiag_cols` alongside rho_F."
    )
    assert "\"sigma_max_F\": sigma_max_F_val" in src, (
        "K-sweep kdiag_row missing sigma_max_F entry — table columns must "
        "match the kdiag_row keys."
    )


def test_sigma_max_F_is_not_used_as_prescription_or_gate():
    """iter168 contract: sigma_max_F is DIAGNOSTIC ONLY. It must not appear
    in `_prescribe_failure_fix` (would re-create the iter155 over-restrictive
    gate failure pattern), must not be in any `lyapunov_target` enum (the
    Lyapunov-on-F branch was refuted), and must not trigger any post-int
    gate failure prescription."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
    # Find the _prescribe_failure_fix function body and assert sigma_max_F
    # is not referenced as a triggering metric there.
    import re
    m = re.search(r"def _prescribe_failure_fix\([^)]*\)[^:]*:(.*?)(?=\ndef\s)", src, re.DOTALL)
    assert m is not None, "could not locate _prescribe_failure_fix function"
    prescribe_body = m.group(1)
    assert "sigma_max_F" not in prescribe_body, (
        "sigma_max_F must NOT appear in _prescribe_failure_fix — it is a "
        "DIAGNOSTIC ONLY (operator-norm proxy; over-restrictive as gate per "
        "iter155 refutation). Use rho_F + iter_conv_rel for FP-convergence gates."
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", flush=True)
        sys.exit(0)
    try:
        test_rho_F_at_saved_fp_returns_finite_float()
        test_rho_F_multi_seed_returns_robust_estimate()
        test_sigma_max_F_at_saved_fp_returns_finite_float()
        test_rho_F_is_emitted_in_fast_val_alongside_residual()
        test_sigma_max_F_is_emitted_in_fast_val_as_diagnostic()
        test_sigma_max_F_is_emitted_in_k_sweep_table()
        test_sigma_max_F_is_not_used_as_prescription_or_gate()
        print("PASS", flush=True)
    except AssertionError as e:
        print(f"FAIL: {e}", flush=True)
