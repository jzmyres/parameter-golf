import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from legacy.train_gpt_rich import max_mean_abs_offdiag_cosine  # noqa: E402


class TestMaxAbsOffdiagCosine(unittest.TestCase):
    def test_e_lt_2_returns_zero(self):
        x = torch.randn(1, 8)
        self.assertEqual(float(max_mean_abs_offdiag_cosine(x).item()), 0.0)

    def test_identical_vectors_max_is_one(self):
        g = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        v = max_mean_abs_offdiag_cosine(g)
        self.assertAlmostEqual(float(v.item()), 1.0, places=6)

    def test_orthogonal_vectors_max_is_zero(self):
        g = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        v = max_mean_abs_offdiag_cosine(g)
        self.assertAlmostEqual(float(v.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
