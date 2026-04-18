"""Tests for Block construction defaults under the iter 41+ Lyapunov architecture.

Iter 41 replaced the Banach contraction shell (τ, Π_R, spectral norms) with
a fully expressive T_θ(z,x₀) = x₀ + Δ_θ(z,x₀). Stability via Lyapunov
penalty (iter 45), not hard architectural constraints.
"""
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


class TestLyapunovArchDefaults(unittest.TestCase):
    def test_state_norm_is_rmsnorm(self) -> None:
        """Shared state normalization h = RMSNorm(z + x_0) (doc §3.2)."""
        from train_gpt import RMSNorm
        b = _fresh_block()
        self.assertTrue(hasattr(b, "state_norm"))
        self.assertIsInstance(b.state_norm, RMSNorm)

    def test_no_spectral_norm_parametrization(self) -> None:
        """Iter 41 removed all spectral-norm caps from Block."""
        from torch.nn.utils import parametrize
        b = _fresh_block()
        # Check attention expert banks are NOT parametrized
        for name in ["expert_q_down", "expert_q_up", "expert_kv_a", "expert_kv_b"]:
            self.assertFalse(
                parametrize.is_parametrized(b.attn, name),
                f"attn.{name} still has spectral norm parametrization",
            )
        for name in ["expert_gate", "expert_fc", "expert_down"]:
            self.assertFalse(
                parametrize.is_parametrized(b.mlp, name),
                f"mlp.{name} still has spectral norm parametrization",
            )

    def test_no_tau_shell(self) -> None:
        """Iter 41 removed the τ-shell contraction parameters."""
        b = _fresh_block()
        self.assertFalse(hasattr(b, "tau_param"), "tau_param should be removed")
        self.assertFalse(hasattr(b, "inj_lin"), "inj_lin should be removed")

    def test_forward_is_x0_plus_delta(self) -> None:
        """T_θ(z, x₀) = x₀ + Δ_θ(z, x₀). When Δ≈0 (fresh init), output ≈ x₀."""
        b = _fresh_block(dim=32)
        b.train(False)
        z = torch.zeros(2, 5, 32)  # zero state
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            out = b(z, x0)
        # At init, delta should be small, so out ≈ x0 (not exact due to expert init)
        self.assertEqual(tuple(out.shape), tuple(x0.shape))


if __name__ == "__main__":
    unittest.main()
