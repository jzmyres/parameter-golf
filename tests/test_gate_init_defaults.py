import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch


class TestGateInitDefaults(unittest.TestCase):
    def test_injection_gate_bias_is_conservative(self) -> None:
        from train_gpt import Block

        torch.manual_seed(0)
        b = Block(dim=32, num_heads=4, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.0, mlp_mult=2.0)
        # inj_gate has shape [1, dim]; bias is length-1.
        bias = float(b.inj_gate.bias.detach().float().item())
        self.assertAlmostEqual(bias, -2.1972246, places=4)

    def test_gg_gate_bias_is_midpoint(self) -> None:
        from train_gpt import Block

        torch.manual_seed(0)
        b = Block(dim=32, num_heads=4, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.0, mlp_mult=2.0)
        bias = float(b.gg_gate.bias.detach().float().item())
        self.assertAlmostEqual(bias, 1.5, places=4)  # sigmoid(1.5) ≈ 0.82


if __name__ == "__main__":
    unittest.main()
