import os
import sys
import unittest

import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestRouterCvTarget(unittest.TestCase):
    def test_cv_target_controls_hinge_strength(self) -> None:
        from train_gpt import SoftDenseRouter

        torch.manual_seed(0)
        E = 6
        r_low = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6, cv_target=0.20)
        r_high = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6, cv_target=10.0)
        r_low.train()
        r_high.train()
        x = torch.ones(2, 3, 4, dtype=torch.float32)

        # Force collapse onto expert 0.
        for r in (r_low, r_high):
            with torch.no_grad():
                r.router.weight.fill_(-10.0)
                r.router.weight[0].fill_(+10.0)
                r.expert_bias.zero_()

        _ = r_low(x)
        loss_low_target = float(r_low._cv_loss_raw.detach().cpu().item())

        _ = r_high(x)
        loss_high_target = float(r_high._cv_loss_raw.detach().cpu().item())

        self.assertGreater(loss_low_target, 1e-6)
        self.assertLess(loss_high_target, 1e-8)
        self.assertFalse(hasattr(r_low, "health_scale"))


if __name__ == "__main__":
    unittest.main()
