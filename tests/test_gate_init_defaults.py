"""Tests for Block construction defaults under the iter 41+ Lyapunov architecture
with the iter 66b Parcae-paper-faithful input injection.

Block map (iter 66b):
    T_θ(z, x₀) = B̄ ⊙ RMSNorm_learn(x₀) + Δ_θ(z, x₀)

where B̄ = Δ·B is the Parcae ZOH input gain (set by GPT._deq_solve per step via
the `_parcae_b_bar` attribute; `None` ⇒ ones(dim) fallback for direct Block
calls). Stability via Lyapunov penalty (iter 45), not hard architectural
constraints.
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

    def test_forward_output_shape_iter66b(self) -> None:
        """T_θ output still has the same shape as z and x₀."""
        b = _fresh_block(dim=32)
        b.train(False)
        z = torch.zeros(2, 5, 32)
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            out = b(z, x0)
        self.assertEqual(tuple(out.shape), tuple(x0.shape))

    def test_zero_expert_delta_returns_b_bar_times_norm_x0(self) -> None:
        """If Δ=0 and B̄ is threaded in, block output == B̄ ⊙ RMSNorm_learn(x₀).

        This is the iter 66b invariant: the input injection is explicit and
        modulated by the Parcae B̄ = Δ·B, not by an unconditional x₀ residual.
        """
        b = _fresh_block(dim=32)
        b.train(False)
        z = torch.randn(2, 5, 32)
        x0 = torch.randn(2, 5, 32)

        def zero_attn(h):
            return h.new_zeros(*h.shape[:-1], b.num_experts, h.shape[-1])

        def zero_mlp(h, w, *, num_shared=0, shared_gate=None):
            return h.new_zeros(h.shape)

        b.attn.forward_experts = zero_attn
        b.mlp.mix_experts = zero_mlp

        b_bar = torch.full((32,), 0.25)
        b._parcae_b_bar = b_bar

        with torch.no_grad():
            out = b(z, x0)
            rms = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            expected = b_bar * (x0 * rms) * b.x0_inject_norm_weight

        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5),
                        "Block with Δ=0 must return B̄ ⊙ RMSNorm_learn(x₀).")

    def test_direct_block_without_b_bar_uses_ones_fallback(self) -> None:
        """Calling Block() without GPT.setup uses ones(dim) so the block still runs."""
        b = _fresh_block(dim=32)
        b.train(False)
        self.assertIsNone(b._parcae_b_bar,
                          "_parcae_b_bar default must be None so direct Block calls see the fallback.")

        def zero_attn(h):
            return h.new_zeros(*h.shape[:-1], b.num_experts, h.shape[-1])

        def zero_mlp(h, w, *, num_shared=0, shared_gate=None):
            return h.new_zeros(h.shape)

        b.attn.forward_experts = zero_attn
        b.mlp.mix_experts = zero_mlp

        z = torch.randn(2, 5, 32)
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            out = b(z, x0)
            rms = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            # Fallback: no B̄ scaling, so inject = RMSNorm(x₀) * x0_inject_norm_weight.
            expected = (x0 * rms) * b.x0_inject_norm_weight
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
