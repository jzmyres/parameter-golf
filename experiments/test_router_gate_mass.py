import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import SoftDenseRouter  # noqa: E402


class TestRouterGateMass(unittest.TestCase):
    def test_gated_router_mass_in_unit_interval(self):
        torch.manual_seed(0)
        r = SoftDenseRouter(16, 6, enable_gate=True)
        x = torch.randn(2, 3, 16)
        with torch.no_grad():
            w = r(x)  # [B,T,E]
            mass = w.sum(dim=-1)
        self.assertTrue(torch.all(mass >= -1e-6).item())
        self.assertTrue(torch.all(mass <= 1.0 + 1e-6).item())


if __name__ == "__main__":
    unittest.main()

