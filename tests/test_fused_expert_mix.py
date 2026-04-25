"""Tests for the iter 90 bottleneck expert architecture.

Replaces the previous low-rank-MLA fused-matmul tests; the new design uses
BottleneckIn → ExpertMLABody (full-rank at r) → BottleneckOut + router.
"""
import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F


def _make_attn(D=16, E=3, r=8, R_proj=4, H_in=2, H_kv_in=1):
    from train_gpt import CausalSelfAttention
    return CausalSelfAttention(
        dim=D,
        num_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        qk_gain_init=1.0,
        num_experts=E,
        num_inner_heads=H_in,
        num_inner_kv_heads=H_kv_in,
        attn_bottleneck_r=r,
        expert_proj_rank=R_proj,
    )


def _make_mlp(D=16, E=4, r=8, R_proj=4, mlp_inner_mult=2.0):
    from train_gpt import MLP
    return MLP(
        dim=D,
        mlp_mult=2.0,  # legacy SSOT mirror; ignored under bottleneck
        num_experts=E,
        mlp_bottleneck_r=r,
        expert_proj_rank=R_proj,
        mlp_inner_mult=mlp_inner_mult,
    )


class TestBottleneckCausalSelfAttention(unittest.TestCase):
    def test_forward_experts_output_shape(self) -> None:
        torch.manual_seed(0)
        B, T, D, E = 2, 5, 16, 3
        attn = _make_attn(D=D, E=E)
        attn.eval()
        x = torch.randn(B, T, D, dtype=torch.float32)
        out = attn.forward_experts(x)
        self.assertEqual(out.shape, (B, T, E, D))

    def test_forward_experts_gradient_flow(self) -> None:
        torch.manual_seed(0)
        B, T, D, E = 1, 4, 16, 2
        attn = _make_attn(D=D, E=E)
        attn.train()
        x = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
        out = attn.forward_experts(x)
        loss = out.sum()
        loss.backward()
        # Bottleneck-in / bottleneck-out matrix params
        for module, name in [
            (attn.in_proj, "in_down"), (attn.in_proj, "in_up"),
            (attn.out_proj, "out_down"), (attn.out_proj, "out_up"),
        ]:
            param = getattr(module, name)
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")
        # Inner MLA params
        for name in ("expert_q", "expert_kv_a", "expert_k_nope", "expert_v",
                     "expert_kr", "expert_wo"):
            param = getattr(attn.expert_body, name)
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_experts_produce_different_outputs(self) -> None:
        torch.manual_seed(42)
        B, T, D, E = 1, 8, 16, 3
        attn = _make_attn(D=D, E=E)
        attn.eval()
        x = torch.randn(B, T, D, dtype=torch.float32)
        out = attn.forward_experts(x)  # (B, T, E, D)
        for i in range(E):
            for j in range(i + 1, E):
                cos_sim = F.cosine_similarity(
                    out[0, :, i, :].flatten(),
                    out[0, :, j, :].flatten(),
                    dim=0,
                )
                self.assertLess(abs(cos_sim.item()), 0.99,
                                f"Experts {i} and {j} produce near-identical outputs")


class TestBottleneckMLP(unittest.TestCase):
    def test_mix_experts_shape_and_grad(self) -> None:
        torch.manual_seed(0)
        B, T, D, E = 2, 7, 16, 4
        mlp = _make_mlp(D=D, E=E)
        mlp.train()
        x = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
        w = torch.softmax(torch.randn(B, T, E, dtype=torch.float32), dim=-1)
        out = mlp.mix_experts(x, w)
        self.assertEqual(out.shape, (B, T, D))
        out.sum().backward()
        for module, name in [
            (mlp.in_proj, "in_down"), (mlp.in_proj, "in_up"),
            (mlp.out_proj, "out_down"), (mlp.out_proj, "out_up"),
        ]:
            param = getattr(module, name)
            self.assertIsNotNone(param.grad, f"No gradient for mlp.{name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for mlp.{name}")
        for name in ("expert_gate", "expert_fc", "expert_down"):
            param = getattr(mlp.expert_body, name)
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_routing_zero_weights_produce_zero_output(self) -> None:
        """Setting all routing weights to zero should zero the un-shared MLP output."""
        torch.manual_seed(0)
        B, T, D, E = 1, 4, 16, 4
        mlp = _make_mlp(D=D, E=E)
        mlp.eval()
        x = torch.randn(B, T, D, dtype=torch.float32)
        w = torch.zeros(B, T, E, dtype=torch.float32)
        with torch.no_grad():
            out = mlp.mix_experts(x, w)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6),
                        "All-zero routing should give zero MLP output (no shared experts)")

    def test_mean_expert_outputs_for_ortho_shape(self) -> None:
        torch.manual_seed(0)
        B, T, D, E = 2, 6, 16, 3
        mlp = _make_mlp(D=D, E=E)
        mlp.eval()
        x = torch.randn(B, T, D, dtype=torch.float32)
        with torch.no_grad():
            mu = mlp.mean_expert_outputs_for_ortho(x, max_tokens=4)
        self.assertEqual(mu.shape, (E, D))


if __name__ == "__main__":
    unittest.main()
