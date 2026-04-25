"""Tests for the post-int6 assertion helpers.

Covers:
- SoftDenseRouter._materialize_diag_lists: lazy, idempotent, respects GPU source
- Block.forward output shape under the current T_θ semantics

The post-iter-41 block map is T_θ(z, x₀) = Δ_θ(z, x₀); iter 66b extends this
to T_θ(z, x₀) = B̄ ⊙ RMSNorm_learn(x₀) + Δ_θ(z, x₀) (Parcae-paper-faithful
input injection). The "output must include x₀" invariant is enforced by
tests/test_gate_init_defaults.py::test_zero_expert_delta_does_not_return_x0
(iter 66a) and its iter 66b successor.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import SoftDenseRouter, Block, router_diagnostics  # noqa: E402


class TestSoftDenseRouterDiagMaterialization(unittest.TestCase):
    def _fresh_router(self, num_experts: int = 4) -> SoftDenseRouter:
        torch.manual_seed(0)
        r = SoftDenseRouter(dim=32, num_experts=num_experts)
        return r

    def test_record_populates_gpu_tensors(self):
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        self.assertIsNotNone(r._expert_usage_gpu)
        self.assertIsNotNone(r._expert_balance_cv_gpu)
        self.assertIsNotNone(r._expert_entropy_gpu)

    def test_standalone_forward_auto_materializes_lists(self):
        r = self._fresh_router(num_experts=4)
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        self.assertIsInstance(r._expert_usage, list)
        self.assertEqual(len(r._expert_usage), 4)
        self.assertIsInstance(r._expert_balance_cv, float)

    def test_materialize_is_idempotent(self):
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        first_list = r._expert_usage
        self.assertIsInstance(first_list, list)
        r._materialize_diag_lists()
        self.assertIs(r._expert_usage, first_list)

    def test_disabled_diagnostics_skips_materialization(self):
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        r.train()
        _ = r(x)
        self.assertIsNone(r._expert_usage)


class TestBlockForwardShape(unittest.TestCase):
    """Basic shape and structural tests for the iter 41 Lyapunov Block."""

    def _fresh_block(self, dim: int = 16) -> Block:
        torch.manual_seed(0)
        return Block(
            dim, num_heads=2, num_kv_heads=1, mlp_mult=2.0,
            rope_base=1000.0, qk_gain_init=1.0,
            num_experts=2, num_shared_experts=0, router_scoring="linear",
            attn_bottleneck_r=8, mlp_bottleneck_r=8, expert_proj_rank=4,
            attn_inner_heads=2, attn_inner_kv_heads=1, mlp_inner_mult=2.0,
        )

    def test_forward_output_shape(self):
        blk = self._fresh_block()
        B, T, D = 2, 3, 16
        z = torch.randn(B, T, D)
        x0 = torch.randn(B, T, D)
        with torch.no_grad():
            out = blk(z, x0)
        self.assertEqual(tuple(out.shape), (B, T, D))

if __name__ == "__main__":
    unittest.main()
