"""Tests for Block construction defaults.

Block map: T_θ(z, x₀) = B̄ ⊙ RMSUnit(x₀) ⊙ x0_inject_norm_weight + Δ_θ(z, x₀).
B̄ = Δ·B is passed explicitly through RevDEQFunction's autograd boundary so
task-loss gradients reach parcae_raw_b; Block(z, x0) with no b_bar arg means
B̄ = ones(D) (for direct unit-test calls outside the solver).
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
    def test_block_prenorm_is_parameter_free(self) -> None:
        """Block's state preconditioner is parameter-free; learned scales are projection-local."""
        b = _fresh_block(dim=16)
        # No state_norm module; RMS stat is applied inline as _rms_unit(x).
        self.assertFalse(hasattr(b, "state_norm"),
                         "state_norm module should be removed; use _rms_unit directly.")
        # Smoke check: a Block forward with zero experts produces only the x0-injection term,
        # which confirms no extra learned preconditioner got re-introduced.
        b.train(False)
        z = torch.zeros(1, 4, 16)
        x0 = torch.randn(1, 4, 16)
        b.attn.forward_experts = lambda h: h.new_zeros(*h.shape[:-1], b.num_experts, h.shape[-1])
        b.mlp.mix_experts = lambda h, w, *, num_shared=0, shared_gate=None: h.new_zeros(h.shape)
        with torch.no_grad():
            out = b(z, x0)
            rms = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            expected = (x0 * rms) * b.x0_inject_norm_weight
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

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

        with torch.no_grad():
            out = b(z, x0, b_bar)
            rms = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            expected = b_bar * (x0 * rms) * b.x0_inject_norm_weight

        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5),
                        "Block with Δ=0 must return B̄ ⊙ RMSNorm_learn(x₀).")

    def test_direct_block_without_b_bar_skips_b_bar_term(self) -> None:
        """Block(z, x0) without b_bar arg ⇒ ones(D) fallback (no B̄ scaling)."""
        b = _fresh_block(dim=32)
        b.train(False)

        b.attn.forward_experts = lambda h: h.new_zeros(*h.shape[:-1], b.num_experts, h.shape[-1])
        b.mlp.mix_experts = lambda h, w, *, num_shared=0, shared_gate=None: h.new_zeros(h.shape)

        z = torch.randn(2, 5, 32)
        x0 = torch.randn(2, 5, 32)
        with torch.no_grad():
            out = b(z, x0)
            rms = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            expected = (x0 * rms) * b.x0_inject_norm_weight
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_revdeq_default_trains_parcae_b_bar(self) -> None:
        """Default RevDEQ path must expose B̄ to autograd, not hide it in module state."""
        from train_gpt import GPT

        torch.manual_seed(0)
        model = GPT(
            vocab_size=64,
            num_layers=2,
            model_dim=32,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            rope_base=10000.0,
            qk_gain_init=1.0,
            bigram_vocab_size=0,
            kv_latent_dim=16,
            num_refinements=0,
            attn_expert_rank=4,
            mlp_expert_rank=4,
            num_experts=2,
            num_shared_experts=0,
            use_parcae=True,
            deq_backward="revdeq",
        )
        model.train()
        x = torch.randint(0, 64, (1, 6))
        y = torch.randint(0, 64, (1, 6))

        loss = model(x, y)
        loss.backward()

        grad = model.parcae_raw_b.grad
        self.assertIsNotNone(grad, "parcae_raw_b.grad must not be hidden by RevDEQFunction.")
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
