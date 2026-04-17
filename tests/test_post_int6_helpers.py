"""Tests for the post-int6 assertion helpers.

Covers:
- SoftDenseRouter._materialize_diag_lists: lazy, idempotent, respects GPU source

The old b(x_0) chunking-invariance tests were for the Banach contraction shell
(iter 30). Iter 41 replaced that with T_θ = x₀ + Δ — no inj_lin, no b(x_0).
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
        )

    def test_forward_output_shape(self):
        blk = self._fresh_block()
        B, T, D = 2, 3, 16
        z = torch.randn(B, T, D)
        x0 = torch.randn(B, T, D)
        with torch.no_grad():
            out = blk(z, x0)
        self.assertEqual(tuple(out.shape), (B, T, D))

    def test_forward_includes_x0(self):
        """T_θ = x₀ + Δ: output should include x₀ component."""
        import torch.nn.functional as F
        blk = self._fresh_block()
        z = torch.zeros(2, 3, 16)
        x0 = torch.randn(2, 3, 16) * 10.0  # large x0
        with torch.no_grad():
            out = blk(z, x0)
        # With large x0 and zero z, output should have significant x0 component
        cos_sim = F.cosine_similarity(out.flatten(), x0.flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.5,
                           "Output should have significant x0 component")


if __name__ == "__main__":
    unittest.main()
