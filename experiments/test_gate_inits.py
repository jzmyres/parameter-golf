import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Block


class TestGateInits(unittest.TestCase):
    def test_block_global_gate_init_midpoint(self):
        torch.manual_seed(0)
        b = Block(
            dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            rope_base=1000.0,
            qk_gain_init=1.5,
        )
        # Global gate should start unsaturated to preserve gradients.
        self.assertTrue(hasattr(b, "gg_gate"))
        self.assertIsNotNone(getattr(b.gg_gate, "bias", None))
        self.assertAlmostEqual(float(b.gg_gate.bias.detach().float().mean().item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

