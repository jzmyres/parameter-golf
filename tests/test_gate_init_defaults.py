"""Tests for Block construction defaults under the iter 30+ contraction shell.

The pre-iter-30 `inj_gate` / `gg_gate` bias tests are gone: those modules were
removed by the τ-shell refactor (`T_x(z) = (1-τ) b(x_0) + τ G_θ(z, x_0)` with
exogenous `b(x_0) = x_0 + U·rms_norm(x_0)`).  The tests below assert the new
invariants that keep that contraction shell honest.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import Block, _rms_norm  # noqa: E402


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
        self.assertTrue(hasattr(b, "tau_max"))
        self.assertGreater(b.tau_max, 0.0, "τ_max must be > 0")
        self.assertLess(b.tau_max, 1.0, "τ_max must be < 1 for strict contraction")
        with torch.no_grad():
            tau = b._tau()
        self.assertGreater(tau.item(), 0.0, "τ must be positive")
        self.assertLessEqual(tau.item(), b.tau_max + 1e-6,
                             "τ must be ≤ τ_max for Banach guarantee")

    def test_inj_lin_is_orthogonally_parameterized(self) -> None:
        """U in b(x_0) = x_0 + U·rms_norm(x_0) must be orthogonally
        parametrized (all σ = 1, exact isometry) via OrthogonalParametrization
        (iter 40, doc §6.1)."""
        b = _fresh_block()
        self.assertTrue(hasattr(b, "inj_lin"))
        from torch.nn.utils import parametrize
        self.assertTrue(parametrize.is_parametrized(b.inj_lin, "weight"))
        from train_gpt import OrthogonalParametrization
        param_list = b.inj_lin.parametrizations.weight
        self.assertTrue(
            any(isinstance(p, OrthogonalParametrization) for p in param_list),
            "inj_lin must use OrthogonalParametrization",
        )

    def test_b_x0_shape_and_identity_path(self) -> None:
        """b(x_0) = x_0 + U·rms_norm(x_0) must return same shape as x_0 AND
        include the identity path.  Structural check: b(x_0) - inj_term == x_0
        for any U, confirming the '+ x_0' term is literally in the formula."""
        b = _fresh_block(dim=32)
        b.train(False)
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            bx0 = b._compute_b_x0(x0)
            # _compute_b_x0 uses _rms_norm, not attn_norm (BallProjection)
            inj_term = b.inj_lin(_rms_norm(x0)).to(dtype=x0.dtype)
        self.assertEqual(tuple(bx0.shape), tuple(x0.shape))
        self.assertTrue(
            torch.allclose(bx0 - inj_term, x0, atol=1e-5),
            "b(x_0) - U·rms_norm(x_0) must equal x_0 (identity path)",
        )

    def test_expert_banks_are_orthogonal(self) -> None:
        """All expert banks must use OrthogonalParametrization (iter 40)."""
        from torch.nn.utils import parametrize
        from train_gpt import OrthogonalParametrization
        b = _fresh_block()
        for name, mod in [("expert_out", b.attn), ("expert_proj", b.attn),
                          ("expert_gate", b.mlp), ("expert_fc", b.mlp),
                          ("expert_down", b.mlp)]:
            self.assertTrue(
                parametrize.is_parametrized(mod, name),
                f"{name} must be parametrized",
            )
            param_list = getattr(mod.parametrizations, name)
            self.assertTrue(
                any(isinstance(p, OrthogonalParametrization) for p in param_list),
                f"{name} must use OrthogonalParametrization",
            )


if __name__ == "__main__":
    unittest.main()
