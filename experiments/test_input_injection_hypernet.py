import os
import sys
import unittest
import math
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Block  # noqa: E402


class TestInputInjectionHypernet(unittest.TestCase):
    def test_injection_gate_bounds_and_init(self):
        torch.manual_seed(0)
        d = 16
        blk = Block(
            d,
            num_heads=2,
            num_kv_heads=1,
            mlp_mult=2.0,
            rope_base=1000.0,
            qk_gain_init=1.0,
            kv_latent_dim=0,
            attn_expert_rank=0,
            mlp_expert_rank=0,
            moe_level="component",
        )
        z_in = torch.randn(2, 3, d)
        x0 = torch.randn(2, 3, d)
        with torch.no_grad():
            g = blk._inj_gate_from(z_in)
        self.assertEqual(tuple(g.shape), (d,))
        self.assertTrue(torch.all(g >= 0.0).item())
        self.assertTrue(torch.all(g <= 1.0).item())
        # Default init: w_inj=0, b_inj=-6 → g = sigmoid(-6) regardless of z_in.
        expected = 1.0 / (1.0 + math.exp(6.0))
        self.assertAlmostEqual(float(g.mean().item()), expected, places=4)

        # Smoke: forward preserves shape.
        out = blk(z_in, x0)
        self.assertEqual(tuple(out.shape), tuple(z_in.shape))


if __name__ == "__main__":
    unittest.main()

