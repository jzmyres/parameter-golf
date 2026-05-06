import os
import sys
import unittest

import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_gpt import SoftDenseRouter, router_diagnostics


class TestRouterHealthLoss(unittest.TestCase):
    E = 6

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.router = SoftDenseRouter(dim=4, num_experts=self.E, min_share_frac=0.6)
        self.router.train()

    def test_cv_squared_penalizes_collapse(self) -> None:
        x = torch.ones(2, 3, 4, dtype=torch.float32)

        with torch.no_grad():
            self.router.router.weight.zero_()
            self.router.expert_bias.zero_()
        _ = self.router(x)
        loss_uniform = float(self.router._cv_loss_raw.detach().cpu().item())

        with torch.no_grad():
            self.router.router.weight.fill_(-10.0)
            self.router.router.weight[0].fill_(+10.0)
        _ = self.router(x)
        loss_collapse = float(self.router._cv_loss_raw.detach().cpu().item())

        self.assertGreater(loss_collapse, loss_uniform + 1e-6)

    def test_cv_squared_has_gradient_below_old_target(self) -> None:
        # Regression guard for the relu(cv − 0.20)² → cv² loss-form change
        # (iter 142b promotion 2026-05-06): under mild imbalance with cv ≈ 0.1
        # — i.e. inside the OLD hinge's silent region — the new continuous
        # cv² form must still produce nonzero pressure and a nonzero gradient
        # back to the router weights.
        with torch.no_grad():
            self.router.router.weight.zero_()
            self.router.router.weight[0].fill_(0.2)  # small bias toward expert 0
            self.router.expert_bias.zero_()

        x = torch.ones(2, 3, 4, dtype=torch.float32)
        _ = self.router(x)
        loss = self.router._cv_loss_raw

        self.assertGreater(float(loss.detach().cpu().item()), 0.0)
        loss.backward()
        grad_norm = float(self.router.router.weight.grad.norm().item())
        self.assertGreater(grad_norm, 1e-8)

    def test_uniform_submass_is_healthy(self) -> None:
        x = torch.zeros(2, 3, 4, dtype=torch.float32)
        with torch.no_grad():
            self.router.router.weight.zero_()
            self.router.expert_bias.zero_()
            self.router.router_gate.weight.zero_()
            self.router.router_gate.bias.zero_()  # sigmoid=0.5, total mass=0.5

        with router_diagnostics(True, step_tag=1):
            p = self.router(x)

        self.assertAlmostEqual(float(p.detach().sum(dim=-1).mean().item()), 0.5, places=5)
        self.assertLess(float(self.router._cv_loss_raw.detach().cpu().item()), 1e-8)
        self.assertFalse(hasattr(self.router, "_health_loss"))
        self.assertFalse(hasattr(self.router, "_balance_loss"))
        self.assertAlmostEqual(sum(self.router._expert_usage), 1.0, places=5)
        self.assertAlmostEqual(float(self.router._expert_total_mass), 0.5, places=5)

    def test_pooled_router_health_normalizes_each_component(self) -> None:
        # Different topology than setUp (4 experts split 2+2); build a local router.
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
        # Fake a terminal-state mean share with dead experts.
        ms = torch.tensor([1.0] + [0.0] * (self.E - 1), dtype=torch.float32)
        self.router._mean_share_last = ms
        with torch.no_grad():
            self.router.expert_bias.zero_()
            self.router.bias_update(lr=1.0, clip=0.0, distributed=False)

        self.assertLess(
            float(self.router.expert_bias[0].item()),
            float(self.router.expert_bias[1].item()),
        )


if __name__ == "__main__":
    unittest.main()
