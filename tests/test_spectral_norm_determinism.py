"""Tests for orthogonal parametrization (iter 40) and spectral-norm determinism.

Iter 40 replaced spectral-norm caps with Newton-Schulz orthogonal
parametrization.  The key invariants:
1. forward(W) is deterministic (no buffers, stateless) → RevDEQ-safe
2. Output has all singular values ≈ 1 (isometry)
3. Block._compute_b_x0 is deterministic in train mode
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import OrthogonalParametrization  # noqa: E402


class TestOrthogonalParametrization3D(unittest.TestCase):
    def _fresh(self, shape=(3, 8, 5)):
        torch.manual_seed(0)
        param = OrthogonalParametrization(torch.Size(shape))
        W = torch.randn(*shape) * 0.5
        return param, W

    def test_forward_is_deterministic(self) -> None:
        """Two consecutive forwards must produce identical output (RevDEQ)."""
        param, W = self._fresh()
        out1 = param(W)
        out2 = param(W)
        self.assertTrue(
            torch.equal(out1, out2),
            "forward() output drifts — RevDEQ reversibility broken",
        )

    def test_output_is_orthogonal_tall(self) -> None:
        """For tall (M≥N) experts, W^T W ≈ I_N."""
        param, W = self._fresh(shape=(3, 8, 5))  # tall: 8×5
        out = param(W)
        E, M, N = out.shape
        WtW = torch.bmm(out.transpose(-2, -1), out)  # (E, N, N)
        I = torch.eye(N).unsqueeze(0).expand(E, -1, -1)
        self.assertTrue(
            torch.allclose(WtW, I, atol=1e-4),
            f"W^T W not close to I: max err = {(WtW - I).abs().max().item():.6f}",
        )

    def test_output_is_orthogonal_wide(self) -> None:
        """For wide (M<N) experts, W W^T ≈ I_M."""
        param = OrthogonalParametrization(torch.Size((3, 4, 10)))
        W = torch.randn(3, 4, 10) * 0.5
        out = param(W)
        E, M, N = out.shape
        WWt = torch.bmm(out, out.transpose(-2, -1))  # (E, M, M)
        I = torch.eye(M).unsqueeze(0).expand(E, -1, -1)
        self.assertTrue(
            torch.allclose(WWt, I, atol=1e-4),
            f"W W^T not close to I: max err = {(WWt - I).abs().max().item():.6f}",
        )

    def test_no_buffers(self) -> None:
        """OrthogonalParametrization must be stateless (no buffers)."""
        param, _ = self._fresh()
        buffers = list(param.buffers())
        self.assertEqual(len(buffers), 0, "should have no buffers (stateless)")


class TestOrthogonalParametrization2D(unittest.TestCase):
    def _fresh(self, shape=(8, 8)):
        torch.manual_seed(0)
        param = OrthogonalParametrization(torch.Size(shape))
        W = torch.randn(*shape) * 0.5
        return param, W

    def test_forward_is_deterministic(self) -> None:
        param, W = self._fresh()
        out1 = param(W)
        out2 = param(W)
        self.assertTrue(torch.equal(out1, out2), "output drifts across calls")

    def test_output_is_orthogonal_square(self) -> None:
        """For square matrices, W^T W ≈ I."""
        param, W = self._fresh(shape=(8, 8))
        out = param(W)
        WtW = out.t() @ out
        I = torch.eye(8)
        self.assertTrue(
            torch.allclose(WtW, I, atol=1e-4),
            f"W^T W not close to I: max err = {(WtW - I).abs().max().item():.6f}",
        )

    def test_output_is_orthogonal_tall(self) -> None:
        param = OrthogonalParametrization(torch.Size((10, 6)))
        W = torch.randn(10, 6) * 0.5
        out = param(W)
        WtW = out.t() @ out
        I = torch.eye(6)
        self.assertTrue(
            torch.allclose(WtW, I, atol=1e-4),
            f"W^T W not close to I: max err = {(WtW - I).abs().max().item():.6f}",
        )

    def test_inj_lin_b_x0_deterministic_in_train_mode(self) -> None:
        """Block._compute_b_x0 must be deterministic in train mode."""
        from train_gpt import Block
        torch.manual_seed(0)
        blk = Block(dim=16, num_heads=2, num_kv_heads=1, mlp_mult=2.0,
                    rope_base=1000.0, qk_gain_init=1.0)
        blk.train(True)
        x0 = torch.randn(2, 3, 16)
        with torch.no_grad():
            b1 = blk._compute_b_x0(x0)
            b2 = blk._compute_b_x0(x0)
        self.assertTrue(
            torch.equal(b1, b2),
            "b(x_0) differs across calls in train mode",
        )


if __name__ == "__main__":
    unittest.main()
