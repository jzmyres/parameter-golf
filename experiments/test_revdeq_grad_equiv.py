import os
import sys
import unittest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import RevDEQFunction  # noqa: E402


class _FTiny(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.wz = torch.nn.Linear(dim, dim, bias=False)
        self.wx = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.normal_(self.wz.weight, std=0.02)
        torch.nn.init.normal_(self.wx.weight, std=0.02)

    def forward(self, z, x0):
        return torch.tanh(self.wz(z) + self.wx(x0))


def _explicit_unroll(f, x0, z_init, beta: float, k: int):
    y = z_init
    z = z_init
    beta_inv = 1.0 - beta
    for _ in range(k):
        y = beta_inv * y + beta * f(z, x0)
        z = beta_inv * z + beta * f(y, x0)
    return z


class TestRevDEQGradEquivalence(unittest.TestCase):
    def test_revdeq_backward_matches_autograd_unroll(self):
        torch.manual_seed(0)
        dim = 16
        b, t = 2, 3
        beta = 0.2
        k = 4

        f = _FTiny(dim)
        # Make sure both paths start from identical params.
        f2 = _FTiny(dim)
        f2.load_state_dict(f.state_dict(), strict=True)

        x0 = torch.randn(b, t, dim, dtype=torch.float32, requires_grad=True)
        z0 = torch.randn(b, t, dim, dtype=torch.float32, requires_grad=True)
        x0_2 = x0.detach().clone().requires_grad_(True)
        z0_2 = z0.detach().clone().requires_grad_(True)

        # Path A: RevDEQ custom backward
        params = tuple(p for p in f.parameters() if p.requires_grad)
        z_term, _ = RevDEQFunction.apply(f, x0, z0, beta, k, *params)
        loss = (z_term ** 2).mean()
        loss.backward()

        # Path B: explicit unroll with standard autograd
        z_term2 = _explicit_unroll(f2, x0_2, z0_2, beta, k)
        loss2 = (z_term2 ** 2).mean()
        loss2.backward()

        # Compare grads (relative error).
        def _rel_err(a, b, eps=1e-8):
            return float((a - b).abs().max().item() / (b.abs().max().item() + eps))

        self.assertLess(_rel_err(x0.grad, x0_2.grad), 5e-3)
        self.assertLess(_rel_err(z0.grad, z0_2.grad), 5e-3)

        for (n1, p1), (n2, p2) in zip(f.named_parameters(), f2.named_parameters(), strict=True):
            self.assertEqual(n1, n2)
            self.assertLess(_rel_err(p1.grad, p2.grad), 5e-3)


if __name__ == "__main__":
    unittest.main()

