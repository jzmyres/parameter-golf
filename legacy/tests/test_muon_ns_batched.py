"""Tests for batched Newton-Schulz in Muon optimizer."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from legacy.train_gpt_rich import _ns5_2d, _ns5_batched


class TestBatchedNS(unittest.TestCase):
    def test_batched_matches_loop(self):
        """Batched NS matches per-matrix loop (m < n)."""
        torch.manual_seed(42)
        G = torch.randn(8, 128, 768)
        ref = torch.stack([_ns5_2d(G[i], steps=5) for i in range(8)])
        batched = _ns5_batched(G, steps=5)
        diff = (ref.float() - batched.float()).abs().max().item()
        self.assertLess(diff, 1e-2, f"max_diff={diff}")

    def test_batched_matches_loop_transposed(self):
        """Batched NS matches per-matrix loop (m > n, triggers internal transpose)."""
        torch.manual_seed(123)
        G = torch.randn(8, 768, 128)
        ref = torch.stack([_ns5_2d(G[i], steps=5) for i in range(8)])
        batched = _ns5_batched(G, steps=5)
        diff = (ref.float() - batched.float()).abs().max().item()
        self.assertLess(diff, 1e-2, f"max_diff={diff}")

    def test_per_matrix_normalization(self):
        """Scaling one matrix should not affect others."""
        torch.manual_seed(99)
        G = torch.randn(4, 64, 128)
        G_scaled = G.clone()
        G_scaled[2] *= 1000.0
        out_orig = _ns5_batched(G, steps=5)
        out_scaled = _ns5_batched(G_scaled, steps=5)
        for i in [0, 1, 3]:
            diff = (out_orig[i].float() - out_scaled[i].float()).abs().max().item()
            self.assertLess(diff, 1e-3, f"Matrix {i} affected by scaling: diff={diff}")


if __name__ == "__main__":
    unittest.main()
