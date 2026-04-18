import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Block


class TestGateInits(unittest.TestCase):
    def test_block_attn_gate_bias_init_zero(self):
        """Attention gate biases start at 0 → sigmoid(0) = 0.5 (midpoint)."""
        torch.manual_seed(0)
        b = Block(
            dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            rope_base=1000.0,
            qk_gain_init=1.5,
        )
        self.assertTrue(hasattr(b.attn, "gate_bias"))
        self.assertAlmostEqual(float(b.attn.gate_bias.detach().float().mean().item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
