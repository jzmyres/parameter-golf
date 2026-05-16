"""rho_F (spectral radius) probe tests at the saved DEQ FP.

rho_F = |lambda_max(J_F)| via straight power iteration on J_F (not on
J_F^T J_F). Necessary AND sufficient for asymptotic local FP convergence
per Hartman-Grobman; this is the principled gate metric.

The lip_ub_T/S/F operator-norm probes and the legacy `_lip_ub_at_saved_fp`
helper were removed 2026-05-15 — over-restrictive for non-symmetric J_F
(iter152 has rho ≈ 0.85 yet sigma_max ≈ 17). This file was renamed from
test_lip_ub_fix.py at the same time.

Pytest-discoverable; CUDA-required (skipped on CPU-only CI).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_arch import _make_model
from train_gpt import _rho_F_at_saved_fp


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


def test_rho_F_is_emitted_in_fast_val_alongside_residual():
    """rho_F (necessary AND sufficient FP-convergence gate per Hartman-Grobman)
    MUST be probed and emitted in the train-time fast-val site. The lip_ub_*
    operator-norm proxies were removed 2026-05-15 — rho_F is now the sole
    spectral gate metric.

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

    # Negative assertions: the removed lip_ub_* probe surface must NOT
    # come back via copy-paste regression.
    assert "_lip_ub_at_saved_fp" not in src, (
        "_lip_ub_at_saved_fp was removed 2026-05-15 and must not reappear; "
        "use _rho_F_at_saved_fp + _joint_F_residual_at_saved_fp instead."
    )
    assert "lip_ub_T:" not in src and "lip_ub_S:" not in src and "lip_ub_F:" not in src, (
        "lip_ub_T/S/F log emission was removed 2026-05-15 — operator-norm "
        "proxies are over-restrictive; rho_F is the principled gate."
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", flush=True)
        sys.exit(0)
    try:
        test_rho_F_at_saved_fp_returns_finite_float()
        test_rho_F_is_emitted_in_fast_val_alongside_residual()
        print("PASS", flush=True)
    except AssertionError as e:
        print(f"FAIL: {e}", flush=True)
