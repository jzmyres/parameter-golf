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
        )
        B, T = 2, 3
        z_in = torch.randn(B, T, d)
        x0 = torch.randn(B, T, d)
        with torch.no_grad():
            g = blk._inj_gate_from(z_in)
        # H29 (iter 21): token-local injection gate — shape [B, T, 1], not scalar.
        # Previously (1, 1) because the gate averaged across batch+seq first;
        # that was a DEQ violation (FP became batch-dependent).
        self.assertEqual(tuple(g.shape), (B, T, 1))
        self.assertTrue(torch.all(g >= 0.0).item())
        self.assertTrue(torch.all(g <= 1.0).item())
        # Default init: sigmoid(-2.2) ≈ 0.1 per token.  With zero-init weights
        # the gate is approximately constant at init, so all entries are close
        # to the same value — but they live in a [B, T, 1] tensor, not scalar.
        self.assertGreater(float(g.mean().item()), 0.0)
        # H29 principle check: chunking a sequence must not change the gate
        # for unchanged tokens.  With random weights this would differentiate;
        # with zero-init it's approximately flat but we still verify the shape.
        z_alt = torch.cat([z_in, torch.randn(B, T, d)], dim=1)  # longer sequence
        with torch.no_grad():
            g_alt = blk._inj_gate_from(z_alt)
        # First T positions should match exactly (token-local means no cross-seq coupling).
        self.assertTrue(torch.allclose(g[:, :T, :], g_alt[:, :T, :], atol=1e-6),
                        "Token-local gate must not depend on later tokens in the sequence")

        # Smoke: forward preserves shape.
        out = blk(z_in, x0)
        self.assertEqual(tuple(out.shape), tuple(z_in.shape))


if __name__ == "__main__":
    unittest.main()
