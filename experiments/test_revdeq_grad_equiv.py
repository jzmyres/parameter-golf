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

    def forward(self, z, x0, b_bar=None):
        if b_bar is not None:
            x0 = x0 * b_bar.to(dtype=x0.dtype)
        return torch.tanh(self.wz(z) + self.wx(x0))


def _explicit_unroll(f, x0, z_init, beta: float, b_bar, k: int):
    y = z_init
    z = z_init
    beta_inv = 1.0 - beta
    for _ in range(k):
        y = beta_inv * y + beta * f(z, x0, b_bar)
        z = beta_inv * z + beta * f(y, x0, b_bar)
    return z


class TestRevDEQGradEquivalence(unittest.TestCase):
    def test_revdeq_backward_matches_autograd_unroll(self):
        torch.manual_seed(0)
        dim = 16
        b, t = 2, 3
        k = 4

        f = _FTiny(dim)
        # Make sure both paths start from identical params.
        f2 = _FTiny(dim)
        f2.load_state_dict(f.state_dict(), strict=True)

        x0 = torch.randn(b, t, dim, dtype=torch.float32, requires_grad=True)
        z0 = torch.randn(b, t, dim, dtype=torch.float32, requires_grad=True)
        x0_2 = x0.detach().clone().requires_grad_(True)
        z0_2 = z0.detach().clone().requires_grad_(True)
        beta = torch.full((dim,), 0.2, dtype=torch.float32, requires_grad=True)
        beta_2 = beta.detach().clone().requires_grad_(True)
        b_bar = torch.rand(dim, dtype=torch.float32, requires_grad=True)
        b_bar_2 = b_bar.detach().clone().requires_grad_(True)

        # Path A: RevDEQ custom backward
        params = tuple(p for p in f.parameters() if p.requires_grad)
        z_term, _ = RevDEQFunction.apply(f, x0, z0, beta, b_bar, k, 0, *params)
        loss = (z_term ** 2).mean()
        loss.backward()

        # Path B: explicit unroll with standard autograd
        z_term2 = _explicit_unroll(f2, x0_2, z0_2, beta_2, b_bar_2, k)
        loss2 = (z_term2 ** 2).mean()
        loss2.backward()

        # Compare grads (relative error).
        def _rel_err(a, b, eps=1e-8):
            return float((a - b).abs().max().item() / (b.abs().max().item() + eps))

        self.assertLess(_rel_err(x0.grad, x0_2.grad), 5e-3)
        self.assertLess(_rel_err(z0.grad, z0_2.grad), 5e-3)
        self.assertIsNotNone(beta.grad)
        self.assertTrue(torch.isfinite(beta.grad).all())
        self.assertLess(_rel_err(beta.grad, beta_2.grad), 5e-3)
        self.assertIsNotNone(b_bar.grad)
        self.assertTrue(torch.isfinite(b_bar.grad).all())
        self.assertLess(_rel_err(b_bar.grad, b_bar_2.grad), 5e-3)

        for (n1, p1), (n2, p2) in zip(f.named_parameters(), f2.named_parameters(), strict=True):
            self.assertEqual(n1, n2)
            self.assertLess(_rel_err(p1.grad, p2.grad), 5e-3)


if __name__ == "__main__":
    unittest.main()
