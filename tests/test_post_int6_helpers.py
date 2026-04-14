"""Tests for the post-int6 assertion helpers.

Covers:
- SoftDenseRouter._materialize_diag_lists: lazy, idempotent, respects GPU source
- Token-local injection gate shape invariant (H29 principle)

The DDP-specific helpers (_ddp_mean_scalar / _ddp_mean_vec / _ddp_mean_tensor_vec)
are defined inside main() so they can't be unit-tested in isolation without a
live world_size>1 process group.  The non-distributed (world_size=1) path is
trivial; the soft-degrade on length mismatch is verified by the smoke test
manifesting a WARN rather than hanging/crashing under DDP.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import SoftDenseRouter, Block, router_diagnostics  # noqa: E402


class TestSoftDenseRouterDiagMaterialization(unittest.TestCase):
    """_materialize_diag_lists is a lazy sync helper for master-rank log sites.
    When outside a DEQ solve, the router forward auto-materializes for backward
    compat with standalone-router tests; inside a DEQ solve, materialization is
    deferred to log sites to avoid K× per-call .cpu() syncs."""

    def _fresh_router(self, num_experts: int = 4) -> SoftDenseRouter:
        torch.manual_seed(0)
        r = SoftDenseRouter(dim=32, num_experts=num_experts)
        return r

    def test_record_populates_gpu_tensors(self):
        """_record_diagnostics always stores GPU tensors (on every rank)."""
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        self.assertIsNotNone(r._expert_usage_gpu)
        self.assertIsNotNone(r._expert_balance_cv_gpu)
        # In single-process / master, entropy GPU tensor also populated
        self.assertIsNotNone(r._expert_entropy_gpu)

    def test_standalone_forward_auto_materializes_lists(self):
        """Outside a DEQ solve, forward auto-materializes the Python list view
        so existing tests (test_router_diagnostics, etc.) keep working."""
        r = self._fresh_router(num_experts=4)
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        self.assertIsInstance(r._expert_usage, list)
        self.assertEqual(len(r._expert_usage), 4)
        self.assertIsInstance(r._expert_balance_cv, float)
        for u in r._expert_usage:
            self.assertGreaterEqual(u, 0.0)
            self.assertLessEqual(u, 1.0)

    def test_materialize_is_idempotent(self):
        """Calling twice must short-circuit — no duplicate CPU sync."""
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        first_list = r._expert_usage
        self.assertIsInstance(first_list, list)
        r._materialize_diag_lists()  # should short-circuit since already materialized
        self.assertIs(r._expert_usage, first_list)

    def test_new_record_invalidates_cached_list(self):
        """A fresh _record_diagnostics call clears the cached list so the next
        auto-materialize (or explicit call) re-syncs from the new GPU tensor."""
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        with router_diagnostics(enabled=True, step_tag=0):
            _ = r(x)
        first_list = r._expert_usage
        # Another forward with different input — this will clear the list
        # inside _record_diagnostics and then auto-re-materialize.
        x2 = torch.randn(2, 5, 32) * 5.0
        with router_diagnostics(enabled=True, step_tag=1):
            _ = r(x2)
        self.assertIsInstance(r._expert_usage, list)
        self.assertIsNot(r._expert_usage, first_list)

    def test_disabled_diagnostics_skips_materialization(self):
        """If diagnostics are disabled and we're training, no list should be
        produced — keeps training-path hot loop cheap."""
        r = self._fresh_router()
        x = torch.randn(2, 5, 32)
        # No router_diagnostics context; training mode
        r.train()
        _ = r(x)
        # Training without the context shouldn't populate the list
        self.assertIsNone(r._expert_usage)


class TestTokenLocalInjectionGate(unittest.TestCase):
    """H29 principle: every gate must be input-dependent AND token-local."""

    def _fresh_block(self, dim: int = 16) -> Block:
        torch.manual_seed(0)
        return Block(
            dim, num_heads=2, num_kv_heads=1, mlp_mult=2.0,
            rope_base=1000.0, qk_gain_init=1.0, kv_latent_dim=0,
            attn_expert_rank=0, mlp_expert_rank=0,
        )

    def test_gate_shape_is_per_token(self):
        blk = self._fresh_block()
        B, T, D = 2, 3, 16
        z_in = torch.randn(B, T, D)
        with torch.no_grad():
            g = blk._inj_gate_from(z_in)
        self.assertEqual(tuple(g.shape), (B, T, 1))

    def test_gate_is_chunking_invariant(self):
        """Appending tokens to the end of the sequence must not change the
        gate values for the earlier positions — streaming / prefix-caching
        invariance that the old batch-mean gate violated."""
        blk = self._fresh_block()
        # Give the model non-trivial weights so the gate isn't constant.
        for p in blk.inj_gate.parameters():
            with torch.no_grad():
                p.normal_(std=0.5)
        B, T, D = 2, 3, 16
        z_in = torch.randn(B, T, D)
        z_ext = torch.cat([z_in, torch.randn(B, 4, D)], dim=1)
        with torch.no_grad():
            g_short = blk._inj_gate_from(z_in)
            g_long = blk._inj_gate_from(z_ext)
        self.assertTrue(
            torch.allclose(g_short, g_long[:, :T, :], atol=1e-6),
            "Token-local gate depends on later tokens — H29 violation",
        )

    def test_gate_bounds(self):
        blk = self._fresh_block()
        z_in = torch.randn(4, 8, 16)
        with torch.no_grad():
            g = blk._inj_gate_from(z_in)
        self.assertTrue(torch.all(g >= 0.0).item())
        self.assertTrue(torch.all(g <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
