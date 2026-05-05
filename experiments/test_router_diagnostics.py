import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRouterDiagnostics(unittest.TestCase):
    def test_soft_dense_router_collects_diagnostics_in_train_when_enabled(self):
        from train_gpt import SoftDenseRouter, router_diagnostics

        router = SoftDenseRouter(dim=8, num_experts=3)
        router.train(True)
        x = torch.randn(2, 4, 8)

        # Default: training mode does not populate diagnostic fields.
        _ = router(x)
        self.assertIsNone(router._expert_usage)

        # When enabled: diagnostics are populated (usage, entropy, cv).
        with router_diagnostics(True, step_tag=123):
            _ = router(x)
        self.assertIsInstance(router._expert_usage, list)
        self.assertEqual(len(router._expert_usage), 3)
        self.assertIsInstance(router._expert_entropy, float)
        self.assertIsInstance(router._expert_balance_cv, float)
        self.assertEqual(router._diag_step, 123)

        # Non-diagnostic forwards must not wipe diagnostics (RevDEQ backward replay safety).
        _ = router(x)
        self.assertIsInstance(router._expert_usage, list)
        self.assertEqual(router._diag_step, 123)

    def test_revdeq_shared_block_routers_log_and_have_grad_regularizers(self):
        """RevDEQ forward runs the shared block under no_grad.

        Ensure we still get:
        - dense (train-step) mlp/attn router diagnostics when enabled
        - CV load regularizers that carry gradients (not detached)
        """
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

        x = torch.randint(0, 64, (1, 32), device="cuda")
        y = torch.randint(0, 64, (1, 32), device="cuda")

        with router_diagnostics(True, step_tag=1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        loss.backward()

        mlp_r = model.shared_block.mlp.mlp_router
        attn_r = model.shared_block.attn.attn_router
        self.assertIsInstance(mlp_r._expert_usage, list)
        self.assertIsInstance(attn_r._expert_usage, list)
        self.assertIsNotNone(mlp_r._cv_loss_raw)
        self.assertIsNotNone(attn_r._cv_loss_raw)
        self.assertTrue(getattr(mlp_r._cv_loss_raw, "requires_grad", False))
        self.assertTrue(getattr(attn_r._cv_loss_raw, "requires_grad", False))


if __name__ == "__main__":
    unittest.main()
