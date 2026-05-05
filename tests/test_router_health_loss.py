import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch


class TestRouterHealthLoss(unittest.TestCase):
    def test_cv_hinge_penalizes_collapse(self) -> None:
        from train_gpt import SoftDenseRouter

        torch.manual_seed(0)
        E = 6
        router = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6, cv_target=0.20)
        router.train()

        x = torch.ones(2, 3, 4, dtype=torch.float32)

        # Uniform routing: logits = 0 => p = 1/E
        with torch.no_grad():
            router.router.weight.zero_()
            router.expert_bias.zero_()
        _ = router(x)
        loss_uniform = float(router._cv_loss_raw.detach().cpu().item())

        # Collapsed routing: push almost all mass to expert 0
        with torch.no_grad():
            router.router.weight.fill_(-10.0)
            router.router.weight[0].fill_(+10.0)
            router.expert_bias.zero_()
        _ = router(x)
        loss_collapse = float(router._cv_loss_raw.detach().cpu().item())

        self.assertGreater(loss_collapse, loss_uniform + 1e-6)

    def test_uniform_submass_is_healthy(self) -> None:
        from train_gpt import SoftDenseRouter, router_diagnostics

        torch.manual_seed(0)
        E = 6
        router = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6, cv_target=0.20)
        router.train()

        x = torch.zeros(2, 3, 4, dtype=torch.float32)
        with torch.no_grad():
            router.router.weight.zero_()
            router.expert_bias.zero_()
            router.router_gate.weight.zero_()
            router.router_gate.bias.zero_()  # sigmoid=0.5, total mass=0.5

        with router_diagnostics(True, step_tag=1):
            p = router(x)

        self.assertAlmostEqual(float(p.detach().sum(dim=-1).mean().item()), 0.5, places=5)
        self.assertLess(float(router._cv_loss_raw.detach().cpu().item()), 1e-8)
        self.assertFalse(hasattr(router, "_health_loss"))
        self.assertFalse(hasattr(router, "_balance_loss"))
        self.assertAlmostEqual(sum(router._expert_usage), 1.0, places=5)
        self.assertAlmostEqual(float(router._expert_total_mass), 0.5, places=5)

    def test_pooled_router_health_normalizes_each_component(self) -> None:
        from train_gpt import SoftDenseRouter

        torch.manual_seed(0)
        router = SoftDenseRouter(dim=4, num_experts=4, health_slices=(2, 2))
        router.train()

        x = torch.zeros(2, 3, 4, dtype=torch.float32)
        with torch.no_grad():
            router.router.weight.zero_()
            router.expert_bias.zero_()
        _ = router(x)

        share = router._mean_share_last.detach().cpu()
        self.assertAlmostEqual(float(share[:2].sum().item()), 1.0, places=5)
        self.assertAlmostEqual(float(share[2:].sum().item()), 1.0, places=5)

    def test_bias_update_boosts_underused_experts(self) -> None:
        from train_gpt import SoftDenseRouter

        torch.manual_seed(0)
        E = 6
        router = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6)
        router.train()

        # Fake a terminal-state mean share with dead experts.
        ms = torch.tensor([1.0] + [0.0] * (E - 1), dtype=torch.float32)
        router._mean_share_last = ms
        with torch.no_grad():
            router.expert_bias.zero_()
            router.bias_update(lr=1.0, clip=0.0, distributed=False)

        # Underused experts should receive a positive bias relative to expert 0.
        self.assertLess(float(router.expert_bias[0].item()), float(router.expert_bias[1].item()))


if __name__ == "__main__":
    unittest.main()
