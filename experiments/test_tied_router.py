import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTiedAttnMlpRouter(unittest.TestCase):
    def test_tied_router_shares_module_and_diagnostics(self):
        """Constructs GPT against the post-iter146 NTP-only default (`use_ctp=False`).
        The test inspects router fields only — CTP coverage is exercised by
        `experiments/test_arch.py::_make_model` which pins `use_ctp=True`."""
        from train_gpt import GPT, router_diagnostics

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for this test")

        model = GPT(
            vocab_size=64,
            num_layers=2,
            model_dim=96,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            rope_base=1000.0,
            qk_gain_init=1.2,
            bigram_vocab_size=0,
            bigram_dim=0,
            kv_latent_dim=0,
            num_refinements=1,
        ).cuda().bfloat16()
        model.train(True)

        attn_r = model.shared_block.attn.attn_router
        mlp_r = model.shared_block.mlp.mlp_router
        self.assertIs(attn_r, mlp_r)

        x = torch.randint(0, 64, (1, 32), device="cuda")
        y = torch.randint(0, 64, (1, 32), device="cuda")
        with router_diagnostics(True, step_tag=7):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        loss.backward()

        self.assertIsInstance(attn_r._expert_usage, list)
        self.assertIsInstance(mlp_r._expert_usage, list)
        self.assertEqual(attn_r._expert_usage, mlp_r._expert_usage)
        self.assertEqual(attn_r._diag_step, 7)
        self.assertEqual(mlp_r._diag_step, 7)


if __name__ == "__main__":
    unittest.main()
