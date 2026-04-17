"""Tests for Block construction defaults under the iter 30+ contraction shell.

The pre-iter-30 `inj_gate` / `gg_gate` bias tests are gone: those modules were
removed by the τ-shell refactor (`T_x(z) = (1-τ) b(x_0) + τ G_θ(z, x_0)` with
exogenous `b(x_0) = x_0 + U·rms_norm(x_0)`).  The tests below assert the new
invariants that keep that contraction shell honest.
"""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import Block  # noqa: E402


def _fresh_block(dim: int = 32) -> Block:
    torch.manual_seed(0)
    return Block(
        dim=dim, num_heads=4, num_kv_heads=2, rope_base=10000.0,
        qk_gain_init=1.0, mlp_mult=2.0,
    )


class TestContractionShellDefaults(unittest.TestCase):
    def test_tau_param_init_gives_small_tau(self) -> None:
        """τ = τ_max · sigmoid(tau_param).  tau_param=-1 → τ ≈ 0.9 · 0.27 ≈ 0.24.
        Small initial τ gives the model room to grow the shell weight during
        training without starting at the contraction boundary."""
        b = _fresh_block()
        self.assertTrue(hasattr(b, "_tau_max"))
        self.assertGreater(b._tau_max, 0.0, "τ_max must be > 0")
        self.assertLess(b._tau_max, 1.0, "τ_max must be < 1 for strict contraction")
        # Input-dependent τ: test with dummy input
        u = torch.randn(2, 5, 32)
        with torch.no_grad():
            tau = b._tau(u)
        self.assertEqual(tuple(tau.shape), (2, 5, 1), "τ must be per-token (B,T,1)")
        self.assertTrue(torch.all(tau > 0).item(), "τ must be positive")
        self.assertTrue(torch.all(tau <= b._tau_max + 1e-6).item(),
                        "τ must be ≤ τ_max for Banach guarantee")

    def test_inj_lin_is_spectrally_parameterized(self) -> None:
        """U in b(x_0) = x_0 + U·rms_norm(x_0) must be spectral-norm parametrized
        so ‖U‖_2 ≤ 1 is enforced every forward (doc §6.1)."""
        b = _fresh_block()
        self.assertTrue(hasattr(b, "inj_lin"))
        # spectral_norm registers parametrizations; attribute is present
        # under `torch.nn.utils.parametrize.is_parametrized` API.
        from torch.nn.utils import parametrize
        self.assertTrue(parametrize.is_parametrized(b.inj_lin, "weight"))

    def test_b_x0_shape_and_identity_path(self) -> None:
        """b(x_0) = x_0 + U·rms_norm(x_0) must return same shape as x_0 AND
        include the identity path.  Structural check: b(x_0) - inj_term == x_0
        for any U, confirming the '+ x_0' term is literally in the formula
        (not replaced by something that happens to be close)."""
        b = _fresh_block(dim=32)
        b.train(False)
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            bx0 = b._compute_b_x0(x0)
            inj_term = b.inj_lin(b.attn_norm(x0)).to(dtype=x0.dtype)
        self.assertEqual(tuple(bx0.shape), tuple(x0.shape))
        self.assertTrue(
            torch.allclose(bx0 - inj_term, x0, atol=1e-5),
            "b(x_0) - U·rms_norm(x_0) must equal x_0 (identity path)",
        )


if __name__ == "__main__":
    unittest.main()
