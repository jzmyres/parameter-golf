import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDEQSupervisionSnaps(unittest.TestCase):
    def test_collect_snaps_shape_and_count(self):
        from train_gpt import GPT

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = GPT(
            vocab_size=1024,
            num_layers=12,
            model_dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=20.0,
            rope_base=1000.0,
            qk_gain_init=1.0,
            bigram_vocab_size=0,
            bigram_dim=0,
            kv_latent_dim=0,
            num_refinements=2,
            deq_backward="autograd",
        ).to(dev)
        if dev.type == "cuda":
            m = m.bfloat16()
        m.train(True)
        m.deq_sup_enabled = True
        m.deq_sup_ks = [2, 6, 12]
        m.deq_sup_weights = [0.2, 0.3, 0.5]
        m.deq_sup_coef = 0.25

        # Run one encode/backbone to populate snaps.
        x = torch.randint(0, 1024, (2, 16), device=dev)
        y = torch.randint(0, 1024, (2, 16), device=dev)
        loss = m(x, y)
        self.assertTrue(torch.isfinite(loss).item())
        snaps = m._deq_sup_snaps
        self.assertIsNotNone(snaps)
        self.assertEqual(len(snaps), 3)
        for z in snaps:
            self.assertEqual(tuple(z.shape), (2, 16, 64))

    def test_sup_schedule_clips_and_renormalizes(self):
        from train_gpt import GPT

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = GPT(
            vocab_size=1024,
            num_layers=12,
            model_dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=20.0,
            rope_base=1000.0,
            qk_gain_init=1.0,
            bigram_vocab_size=0,
            bigram_dim=0,
            kv_latent_dim=0,
            num_refinements=2,
            deq_backward="autograd",
        ).to(dev)
        if dev.type == "cuda":
            m = m.bfloat16()
        m.train(True)
        m.deq_sup_enabled = True
        m.deq_sup_ks = [2, 6, 12]
        m.deq_sup_weights = [0.2, 0.3, 0.5]
        m.deq_sup_coef = 0.25

        # Force a smaller K so k=12 is clipped.
        m._deq_k_override = 6
        x = torch.randint(0, 1024, (2, 16), device=dev)
        y = torch.randint(0, 1024, (2, 16), device=dev)
        loss = m(x, y)
        self.assertTrue(torch.isfinite(loss).item())
        snaps = m._deq_sup_snaps
        self.assertIsNotNone(snaps)
        self.assertEqual(len(snaps), 2)
        ws = m._deq_sup_weights_eff
        self.assertEqual(len(ws), 2)
        self.assertAlmostEqual(float(sum(ws)), 1.0, places=4)
        # Renormalized from [0.2,0.3] -> [0.4,0.6]
        self.assertAlmostEqual(float(ws[0]), 0.4, places=2)
        self.assertAlmostEqual(float(ws[1]), 0.6, places=2)


if __name__ == "__main__":
    unittest.main()
