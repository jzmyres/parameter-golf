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


if __name__ == "__main__":
    unittest.main()
