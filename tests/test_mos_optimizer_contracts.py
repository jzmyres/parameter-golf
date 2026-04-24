import os
import sys
import types
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch


class TestMoSProjectionLayout(unittest.TestCase):
    def test_project_a_matches_index_explicit_einsum(self) -> None:
        from train_gpt import MoSHead

        torch.manual_seed(0)
        head = MoSHead(d_model=5, vocab_size=7, rank=3,
                       num_shared=2, num_specialized=1, fsq_levels=0)
        x = torch.randn(4, 5)
        A_all = torch.cat([head.A_ctp_shared, head.A_ctp], dim=0)

        got = head._project_A(x, A_all)
        expected = torch.einsum("nd,edr->ner", x.to(A_all.dtype), A_all)

        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-6))


class TestOptimizerCoverage(unittest.TestCase):
    def test_all_trainable_parameters_are_grouped_once(self) -> None:
        from train_gpt import GPT, _build_optimizer_param_lists, _flatten_param_groups

        model = GPT(
            vocab_size=1024,
            num_layers=2,
            model_dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2.0,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            rope_base=10000.0,
            qk_gain_init=1.0,
            bigram_vocab_size=32,
            bigram_dim=8,
            kv_latent_dim=32,
            num_refinements=0,
            attn_expert_rank=8,
            mlp_expert_rank=8,
            num_experts=3,
            num_shared_experts=1,
            use_parcae=True,
        )
        args = types.SimpleNamespace(tie_embeddings=True, tied_embed_lr=0.03, embed_lr=0.6)

        tok_groups, matrix_params, scalar_params, parcae_params = _build_optimizer_param_lists(model, args)
        grouped = _flatten_param_groups(tok_groups) + matrix_params + scalar_params + parcae_params
        grouped_ids = [id(p) for p in grouped if p.requires_grad]

        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIn(id(param), grouped_ids, f"{name} missing from optimizer groups")

        named = dict(model.named_parameters())
        for name in [
            "bigram.proj_norm.weight",
            "mos_head.input_norm.weight",
            "final_norm.weight",
            "embed_norm.weight",
            # iter 66b: Parcae-paper-faithful input gain + per-dim input norm.
            "parcae_raw_a",
            "parcae_raw_delta",
            "parcae_raw_b",
            "shared_block.x0_inject_norm_weight",
        ]:
            self.assertIn(id(named[name]), grouped_ids,
                          f"{name} missing from optimizer groups")


if __name__ == "__main__":
    unittest.main()
