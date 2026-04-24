import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F


class TestIndependentExpertAttention(unittest.TestCase):
    def test_forward_experts_output_shape(self) -> None:
        """forward_experts returns (B, T, E, D) per-expert attention outputs."""
        from train_gpt import CausalSelfAttention

        torch.manual_seed(0)
        B, T, D = 2, 5, 16
        E = 3

        attn = CausalSelfAttention(
            dim=D,
            num_heads=4,
            num_kv_heads=2,
            rope_base=10000.0,
            qk_gain_init=1.0,
            kv_latent_dim=8,
            num_experts=E,
            expert_rank=4,
        )
        attn.eval()

        x = torch.randn(B, T, D, dtype=torch.float32)
        out = attn.forward_experts(x)

        self.assertEqual(out.shape, (B, T, E, D))

    def test_forward_experts_gradient_flow(self) -> None:
        """Gradients flow through forward_experts to all expert params."""
        from train_gpt import CausalSelfAttention

        torch.manual_seed(0)
        B, T, D = 1, 4, 16
        E = 2

        attn = CausalSelfAttention(
            dim=D,
            num_heads=4,
            num_kv_heads=2,
            rope_base=10000.0,
            qk_gain_init=1.0,
            kv_latent_dim=8,
            num_experts=E,
            expert_rank=4,
        )
        attn.train()

        x = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
        out = attn.forward_experts(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for all per-expert params (full MLA pipeline)
        for name in ("expert_q_down", "expert_q_up", "expert_kv_a", "expert_kv_b",
                      "expert_k_nope", "expert_v"):
            param = getattr(attn, name)
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_experts_produce_different_outputs(self) -> None:
        """Different experts should produce different attention outputs."""
        from train_gpt import CausalSelfAttention

        torch.manual_seed(42)
        B, T, D = 1, 8, 16
        E = 3

        attn = CausalSelfAttention(
            dim=D,
            num_heads=4,
            num_kv_heads=2,
            rope_base=10000.0,
            qk_gain_init=1.0,
            kv_latent_dim=8,
            num_experts=E,
            expert_rank=4,
        )
        attn.eval()

        x = torch.randn(B, T, D, dtype=torch.float32)
        out = attn.forward_experts(x)  # (B, T, E, D)

        # Check that different experts produce different outputs
        for i in range(E):
            for j in range(i + 1, E):
                cos_sim = F.cosine_similarity(
                    out[0, :, i, :].flatten(),
                    out[0, :, j, :].flatten(),
                    dim=0,
                )
                self.assertLess(abs(cos_sim.item()), 0.99,
                                f"Experts {i} and {j} produce near-identical outputs")


class TestMLPFusedExpertMix(unittest.TestCase):
    def test_mlp_fused_mix_matches_explicit(self) -> None:
        from train_gpt import MLP

        torch.manual_seed(0)
        B, T, D = 2, 7, 16
        E, R = 4, 6

        mlp = MLP(dim=D, mlp_mult=2.0, num_experts=E, expert_rank=R)
        mlp.eval()

        x = torch.randn(B, T, D, dtype=torch.float32)
        w = torch.softmax(torch.randn(B, T, E, dtype=torch.float32), dim=-1)

        out_fused = mlp.mix_experts(x, w)

        x_n = x
        gate_w = mlp.expert_gate.to(dtype=x_n.dtype) * mlp.gate_in_norm_weight.to(dtype=x_n.dtype).unsqueeze(1)
        fc_w = mlp.expert_fc.to(dtype=x_n.dtype) * mlp.fc_in_norm_weight.to(dtype=x_n.dtype).unsqueeze(1)
        gate_h = torch.einsum("btd,esd->btes", x_n, gate_w)
        fc_h = torch.einsum("btd,esd->btes", x_n, fc_w)
        B_, T_, _, _ = gate_h.shape
        h_act = F.silu(gate_h) * fc_h  # [B,T,E,R]  — SwiGLU
        h_flat = h_act.reshape(B_ * T_, mlp.num_experts, mlp.expert_rank)
        # Per-expert RMSNorm (matches fused path)
        h_rms = h_flat.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        h_flat = h_flat * h_rms * mlp.hidden_norm_weight.to(dtype=h_flat.dtype)
        h = h_flat.reshape(B_, T_, mlp.num_experts, mlp.expert_rank)
        out_e = torch.einsum("btes,eds->bted", h, mlp.expert_down.to(dtype=x_n.dtype))
        out_explicit = (w.unsqueeze(-1) * out_e).sum(dim=2)

        self.assertTrue(torch.allclose(out_fused, out_explicit, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
