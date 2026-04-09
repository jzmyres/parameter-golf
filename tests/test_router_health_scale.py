import os
import sys
import unittest

import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestRouterHealthScale(unittest.TestCase):
    def test_health_scale_increases_barrier_strength(self) -> None:
        from train_gpt import SoftDenseRouter

        torch.manual_seed(0)
        E = 6
        r = SoftDenseRouter(dim=4, num_experts=E, min_share_frac=0.6, cv_target=0.20, min_share_loss_weight=1.0, cv_loss_weight=1.0)
        r.train()
        x = torch.ones(2, 3, 4, dtype=torch.float32)

        # Force collapse onto expert 0.
        with torch.no_grad():
            r.router.weight.fill_(-10.0)
            r.router.weight[0].fill_(+10.0)
            r.expert_bias.zero_()

        r.health_scale = 1.0
        _ = r(x)
        loss1 = float(r._health_loss.detach().cpu().item())

        r.health_scale = 5.0
        _ = r(x)
        loss5 = float(r._health_loss.detach().cpu().item())

        self.assertGreater(loss5, loss1 + 1e-6)


if __name__ == "__main__":
    unittest.main()
