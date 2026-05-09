"""Verify the saved-FP `lip_ub` probe returns a finite float.

Runs a single forward pass to populate `_lyapunov_z_star` /
`_lyapunov_x0`, then calls the sliced saved-FP local-contraction
probe and asserts a finite, plausibly-bounded float is returned.
Pytest-discoverable; CUDA-required (skipped on CPU-only CI).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_arch import _make_model
from train_gpt import _lip_ub_at_saved_fp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lip_ub_at_saved_fp_returns_finite_float():
    # Small config — the probe API does not depend on capacity, so a
    # 2-layer / model_dim=128 / num_experts=2 model gives equivalent
    # coverage in <1 s of CUDA time vs a full Hyperparameters() build.
    model = _make_model(
        num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
        num_experts=2, num_shared_experts=0,
        attn_expert_rank=8, mlp_expert_rank=12,
        bigram_vocab_size=0, bigram_dim=8,
    )

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

    lip_ub = _lip_ub_at_saved_fp(model, n_iters=1, B_probe=1)
    assert lip_ub is not None, "probe returned None — should yield a finite float"
    assert isinstance(lip_ub, float), f"expected float, got {type(lip_ub).__name__}"
    assert 0.0 < lip_ub < 1e6, f"lip_ub out of plausible range: {lip_ub}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", flush=True)
        sys.exit(0)
    try:
        test_lip_ub_at_saved_fp_returns_finite_float()
    except AssertionError as e:
        print(f"FAIL: {e}", flush=True)
        sys.exit(1)
    print("PASS: lip_ub probe verified", flush=True)
