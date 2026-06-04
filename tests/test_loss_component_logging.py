import os
import sys
import unittest

import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestLossComponentLogging(unittest.TestCase):
    def test_router_cv_uses_combined_allocation_and_gate_mass(self) -> None:
        from train_gpt import SoftDenseRouter

        # `use_router_sigmoid_gate` is default-OFF in the iter146 rescue
        # stack — without it the sigmoid gate is forced to 1.0 and the
        # `router_gate.bias` mutations below would have no effect on the
        # combined `p = simplex(allocation) * sigmoid(gate)` mass. The
        # test's *purpose* is to verify the CV computation uses combined
        # mass *when* the gate is on, so we instantiate the router with
        # the gate enabled regardless of the current default.
        router = SoftDenseRouter(dim=4, num_experts=4, use_router_sigmoid_gate=True)
        router.train()
        x = torch.zeros(2, 3, 4)
        with torch.no_grad():
            router.router.weight.zero_()
            router.expert_bias.zero_()
            router.router_gate.weight.zero_()
            router.router_gate.bias.fill_(-5.0)
        _ = router(x)
        uniform_low_mass = float(router._cv_loss_raw.detach().item())

        with torch.no_grad():
            router.router_gate.bias[:] = torch.tensor([5.0, 5.0, 5.0, -5.0])
        _ = router(x)
        gated_imbalance = float(router._cv_loss_raw.detach().item())

        self.assertLess(uniform_low_mass, 1e-8)
        self.assertGreater(gated_imbalance, 1e-3)

    def test_gpt_exposes_detached_loss_component_tensors(self) -> None:
        from train_gpt import GPT

        torch.manual_seed(0)
        # Pass-through test: every coef is passed explicitly so the
        # assertion does not silently drift if a Hyperparameters default
        # changes (iter146 promotion changed several defaults from 1.0
        # to 0.0/0.15 — this test must lock the *pass-through* property,
        # not the defaults).
        model = GPT(
            vocab_size=32, num_layers=1, model_dim=32, num_heads=4,
            num_kv_heads=2, mlp_mult=1.0, tie_embeddings=False,
            tied_embed_init_std=0.01, rope_base=10000.0, qk_gain_init=1.0,
            bigram_vocab_size=0, bigram_dim=8, kv_latent_dim=0,
            num_refinements=0, attn_expert_rank=4, mlp_expert_rank=4,
            num_experts=4, num_shared_experts=1, use_ctp=False,
            router_pertoken_entropy_coef=0.01, expert_output_diversity_coef=0.1,
            mos_output_diversity_coef=0.0,
            expert_diversity_max_tokens=4,
        )
        model.train()
        model._expert_diversity_aux_enabled = True
        model._expert_diversity_coef_scale = 0.5
        model._aux_grad_accum_scale = 2.0
        model._expert_diversity_token_start = 1

        input_ids = torch.randint(0, 32, (1, 6))
        target_ids = torch.randint(0, 32, (1, 6))
        loss = model(input_ids, target_ids)

        self.assertTrue(loss.requires_grad)
        # _router_cv_loss_t and _mos_cv_loss_t are still emitted as diagnostic
        # tensors (CV is computed for logging) but neither contributes to the
        # loss after router_load_cv_coef and mos_load_cv_coef were removed
        # 2026-05-15. The finite-horizon scale-hinge tensor is a detached log
        # copy of the with-grad hinge term.
        required = [
            "_router_cv_loss_t", "_router_pertoken_entropy_loss_t", "_mos_cv_loss_t",
            "_expert_diversity_loss_t", "_mos_diversity_loss_t", "_router_reg_loss_t",
            "_router_pertoken_entropy_coef_eff_t",
            "_expert_diversity_coef_eff_t",
            "_mos_diversity_coef_eff_t", "_scale_hinge_loss_t",
        ]
        optional = []
        for name in required + optional:
            self.assertTrue(hasattr(model, name), name)
            t = getattr(model, name)
            if name in optional and t is None:
                continue
            self.assertIsInstance(t, torch.Tensor, name)
            self.assertEqual(tuple(t.shape), (), name)
            self.assertFalse(t.requires_grad, name)
            self.assertTrue(torch.isfinite(t.detach()).item(), name)

        self.assertAlmostEqual(float(model._router_pertoken_entropy_coef_eff_t.item()), 0.01)
        self.assertAlmostEqual(float(model._expert_diversity_coef_eff_t.item()), 0.1)
        self.assertAlmostEqual(float(model._mos_diversity_coef_eff_t.item()), 0.0)
        # Router AND MoS CV terms removed from router_reg_loss 2026-05-15.
        expected_router_reg = (
            float(model._router_pertoken_entropy_loss_t.item()) * float(model._router_pertoken_entropy_coef_eff_t.item())
            + float(model._expert_diversity_loss_t.item()) * float(model._expert_diversity_coef_eff_t.item())
            + float(model._mos_diversity_loss_t.item()) * float(model._mos_diversity_coef_eff_t.item())
        )
        # places=4 (≈1e-4 absolute) accommodates bf16 accumulation in the
        # regularizer assembly path. The semantics — `_router_reg_loss_t`
        # equals the linear combination of per-component losses and coefs
        # — is what's locked here, not float32-equality.
        self.assertAlmostEqual(float(model._router_reg_loss_t.item()), expected_router_reg, places=4)

    def test_deterministic_token_window_start_is_stable_and_bounded(self) -> None:
        from train_gpt import _deterministic_token_window_start

        first = _deterministic_token_window_start(seed=42, step=8, seqlen=128, max_tokens=64)
        second = _deterministic_token_window_start(seed=42, step=8, seqlen=128, max_tokens=64)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 64)
        self.assertEqual(_deterministic_token_window_start(42, 8, seqlen=32, max_tokens=64), 0)
        starts = {
            _deterministic_token_window_start(seed=42, step=step, seqlen=128, max_tokens=64)
            for step in range(8, 80, 8)
        }
        self.assertGreater(len(starts), 1)


if __name__ == "__main__":
    unittest.main()
