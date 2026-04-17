"""Regression test for Phase 6a.2: PerExpertSpectralNormCap.forward is stateless.

RevDEQ's O(1) backward reconstructs the forward during backward; if σ varies
across successive calls on the same weight `W`, reconstruction diverges from
the original forward and gradients are silently corrupted.  The invariant:
`forward(W)` MUST produce bitwise-identical output on repeated calls (and
across forward vs reconstruction).  Writes to `self.u`, `self.v` now live in
an explicit `update_uv_()` method.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import PerExpertSpectralNormCap, SpectralNormCap, refresh_spectral_norms  # noqa: E402


class TestSpectralNormDeterminism(unittest.TestCase):
    def _fresh(self, shape=(3, 5, 7)):
        torch.manual_seed(0)
        cap = PerExpertSpectralNormCap(torch.Size(shape))
        W = torch.randn(*shape) * 0.5
        return cap, W

    def test_forward_does_not_mutate_buffers(self) -> None:
        cap, W = self._fresh()
        u_before = cap.u.detach().clone()
        v_before = cap.v.detach().clone()
        _ = cap(W)
        self.assertTrue(torch.equal(cap.u, u_before), "forward() mutated u")
        self.assertTrue(torch.equal(cap.v, v_before), "forward() mutated v")

    def test_forward_is_deterministic(self) -> None:
        """Two consecutive forwards on the same W must produce identical output.
        This is the invariant RevDEQ reconstruction depends on."""
        cap, W = self._fresh()
        out1 = cap(W)
        out2 = cap(W)
        self.assertTrue(
            torch.equal(out1, out2),
            "forward() output drifts across calls — RevDEQ reversibility broken",
        )

    def test_update_uv_writes_buffers(self) -> None:
        cap, W = self._fresh()
        u_before = cap.u.detach().clone()
        v_before = cap.v.detach().clone()
        cap.update_uv_(W)
        self.assertFalse(torch.equal(cap.u, u_before), "update_uv_ did not update u")
        self.assertFalse(torch.equal(cap.v, v_before), "update_uv_ did not update v")

    def test_update_then_forward_stable_across_calls(self) -> None:
        """After one update_uv_, successive forwards are deterministic."""
        cap, W = self._fresh()
        cap.update_uv_(W)
        out1 = cap(W)
        out2 = cap(W)
        out3 = cap(W)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(out2, out3))

    def test_refresh_spectral_norms_on_model(self) -> None:
        """Module-level helper walks all parametrizations and refreshes them."""
        import torch.nn as nn
        import torch.nn.utils.parametrize as parametrize

        class _DummyBank(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(3, 5, 7))

        m = _DummyBank()
        parametrize.register_parametrization(
            m, "W", PerExpertSpectralNormCap(m.W.shape),
        )
        cap = m.parametrizations.W[0]
        u_before = cap.u.detach().clone()
        refresh_spectral_norms(m)
        self.assertFalse(
            torch.equal(cap.u, u_before),
            "refresh_spectral_norms did not update u on registered cap",
        )


class TestSpectralNormCap2D(unittest.TestCase):
    """Mirror tests for the 2D SpectralNormCap (used by Block.inj_lin)."""

    def _fresh(self, shape=(5, 7)):
        torch.manual_seed(0)
        cap = SpectralNormCap(torch.Size(shape))
        W = torch.randn(*shape) * 0.5
        return cap, W

    def test_forward_does_not_mutate_buffers(self) -> None:
        cap, W = self._fresh()
        u_before = cap.u.detach().clone()
        _ = cap(W)
        self.assertTrue(torch.equal(cap.u, u_before), "forward() mutated u")

    def test_forward_is_deterministic(self) -> None:
        cap, W = self._fresh()
        out1 = cap(W)
        out2 = cap(W)
        self.assertTrue(torch.equal(out1, out2), "output drifts across calls")

    def test_update_uv_writes_buffers(self) -> None:
        cap, W = self._fresh()
        u_before = cap.u.detach().clone()
        cap.update_uv_(W)
        self.assertFalse(torch.equal(cap.u, u_before), "update_uv_ did not update u")

    def test_inj_lin_b_x0_deterministic_in_train_mode(self) -> None:
        """Block._compute_b_x0 must return identical results across two calls
        in train mode — the bug that PyTorch's spectral_norm caused."""
        from train_gpt import Block
        torch.manual_seed(0)
        blk = Block(dim=16, num_heads=2, num_kv_heads=1, mlp_mult=2.0,
                    rope_base=1000.0, qk_gain_init=1.0)
        blk.train(True)
        refresh_spectral_norms(blk)
        x0 = torch.randn(2, 3, 16)
        with torch.no_grad():
            b1 = blk._compute_b_x0(x0)
            b2 = blk._compute_b_x0(x0)
        self.assertTrue(
            torch.equal(b1, b2),
            "b(x_0) differs across calls in train mode — spectral norm nondeterminism",
        )


if __name__ == "__main__":
    unittest.main()
