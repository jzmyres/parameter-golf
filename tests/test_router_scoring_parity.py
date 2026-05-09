"""Phase 6a.3 (reviews 1, 2, 10): verify the matmul-based L2/SIPS router path
matches a broadcast reference (up to fp32 tolerance) and is NaN-robust under
extreme input.  Also checks the bounded-prototype projection and the
fp32-softmax correctness.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import SoftDenseRouter  # noqa: E402


def _broadcast_l2_logits(x_n: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Reference broadcast implementation of L2-distance router logits."""
    diff = x_n.unsqueeze(-2) - c                     # (..., E, D)
    dist_sq = diff.pow(2).sum(dim=-1)                 # (..., E)
    return torch.tanh(-gamma * dist_sq)


class TestRouterMatmulParity(unittest.TestCase):
    def test_l2_matmul_matches_broadcast_fp32(self) -> None:
        torch.manual_seed(0)
        D, E, B, T = 32, 4, 2, 5
        # Test whatever the iter146 default is for `use_router_sigmoid_gate`
        # (currently OFF — `gate = 1.0`, mass = simplex(allocation)). The
        # reference branches on the actual flag so the parity holds whether
        # the default flips back to ON in a future iter.
        r = SoftDenseRouter(dim=D, num_experts=E, scoring="l2")
        r.train(False)
        x = torch.randn(B, T, D)
        x_n = x  # we'll pass pre_normed=True so no internal RMS norm
        p = r(x_n, pre_normed=True)
        with torch.no_grad():
            c = r.prototypes.float()
        ref_route = _broadcast_l2_logits(x_n.float(), c, r.l2_gamma)
        ref_route = ref_route + r.expert_bias.float()
        if r.use_router_sigmoid_gate:
            gate_act = torch.sigmoid(r.router_gate(x_n).float())
        else:
            gate_act = torch.ones_like(ref_route)
        ref_p = torch.softmax(ref_route, dim=-1) * gate_act
        self.assertTrue(
            torch.allclose(p, ref_p.to(p.dtype), atol=1e-3, rtol=1e-3),
            f"L2 matmul parity failed: max abs err = {(p - ref_p.to(p.dtype)).abs().max().item()}",
        )

    def test_no_nan_on_extreme_input(self) -> None:
        """Large-magnitude inputs used to produce NaN in bf16 softmax edge cases;
        the fp32 softmax path is stable."""
        torch.manual_seed(2)
        D, E = 32, 4
        for scoring in ("linear", "l2"):
            r = SoftDenseRouter(dim=D, num_experts=E, scoring=scoring)
            r.train(False)
            x = torch.randn(4, 16, D) * 50.0  # deliberately large
            p = r(x, pre_normed=True)
            self.assertFalse(
                torch.isnan(p).any().item(),
                f"NaN in softmax output for scoring={scoring}",
            )
            # Sigmoid gate: rows sum to ≤ 1 (not necessarily 1)
            row_sum = p.sum(dim=-1)
            self.assertTrue(
                (row_sum <= 1.0 + 1e-4).all().item(),
                f"rows exceed 1 for scoring={scoring}, max sum = {row_sum.max().item()}",
            )
            self.assertTrue(
                (row_sum > 0.0).all().item(),
                f"rows are zero for scoring={scoring}",
            )

    # test_prototype_ball_caps_norm removed: BallProjection on prototypes
    # was for 1-Lip enforcement (Banach). Under Lyapunov, prototypes are unbounded.


if __name__ == "__main__":
    unittest.main()
