import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Block  # noqa: E402


class TestBlockForwardShape(unittest.TestCase):
    def test_forward_preserves_shape(self):
        """Block(z, x0) → same shape as z."""
        torch.manual_seed(0)
        d = 16
        blk = Block(
            d,
            num_heads=2,
            num_kv_heads=1,
            mlp_mult=2.0,
            rope_base=1000.0,
            qk_gain_init=1.0,
        )
        B, T = 2, 3
        z_in = torch.randn(B, T, d)
        x0 = torch.randn(B, T, d)
        out = blk(z_in, x0)
        self.assertEqual(tuple(out.shape), tuple(z_in.shape))


if __name__ == "__main__":
    unittest.main()
