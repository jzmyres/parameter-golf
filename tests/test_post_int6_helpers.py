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


class TestExogenousInjectionChunkingInvariance(unittest.TestCase):
    """H29 principle: the contraction-shell injection b(x_0) = x_0 + U·rms_norm(x_0)
    must be TOKEN-LOCAL (no reduction over batch or sequence).  The iter 30 shell
    replaces the old `_inj_gate_from` token gate; the chunking-invariance check
    migrates to the new `_compute_b_x0`."""

    def _fresh_block(self, dim: int = 16) -> Block:
        torch.manual_seed(0)
        return Block(
            dim, num_heads=2, num_kv_heads=1, mlp_mult=2.0,
            rope_base=1000.0, qk_gain_init=1.0, kv_latent_dim=0,
            attn_expert_rank=0, mlp_expert_rank=0,
        )

    def test_b_x0_shape_is_same_as_input(self):
        blk = self._fresh_block()
        B, T, D = 2, 3, 16
        x0 = torch.randn(B, T, D)
        with torch.no_grad():
            bx0 = blk._compute_b_x0(x0)
        self.assertEqual(tuple(bx0.shape), (B, T, D))

    def test_b_x0_is_chunking_invariant(self):
        """Appending tokens to the end of the sequence must not change the
        b(x_0) values at earlier positions — streaming / prefix-caching
        invariance, because rms_norm operates on the last (feature) dim only.
        Block put in inference mode to freeze the spectral_norm power iteration
        (otherwise U changes between the two forwards — a separate correctness
        issue fixed in Phase 6a.2)."""
        blk = self._fresh_block()
        with torch.no_grad():
            orig = getattr(blk.inj_lin, "weight_orig", None)
            if orig is not None:
                orig.normal_(std=0.3)
            else:
                blk.inj_lin.weight.normal_(std=0.3)
        blk.train(False)
        B, T, D = 2, 3, 16
        x0_short = torch.randn(B, T, D)
        x0_long = torch.cat([x0_short, torch.randn(B, 4, D)], dim=1)
        with torch.no_grad():
            b_short = blk._compute_b_x0(x0_short)
            b_long = blk._compute_b_x0(x0_long)
        self.assertTrue(
            torch.allclose(b_short, b_long[:, :T, :], atol=1e-6),
            "b(x_0) depends on later tokens — chunking / H29 violation",
        )

    def test_b_x0_structural_identity_path(self):
        """Structural check: b(x_0) - U·rms_norm(x_0) == x_0.  Confirms the
        identity term '+ x_0' is literally in the formula so input-dependence
        cannot vanish if U collapses during training."""
        blk = self._fresh_block()
        blk.train(False)
        x0 = torch.randn(4, 8, 16)
        with torch.no_grad():
            bx0 = blk._compute_b_x0(x0)
            inj_term = blk.inj_lin(blk.attn_norm(x0)).to(dtype=x0.dtype)
        self.assertTrue(
            torch.allclose(bx0 - inj_term, x0, atol=1e-5),
            "b(x_0) - U·rms_norm(x_0) must equal x_0",
        )


if __name__ == "__main__":
    unittest.main()
