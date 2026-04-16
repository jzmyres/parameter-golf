import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F


class TestFusedExpertMix(unittest.TestCase):
    def test_attention_fused_mix_matches_explicit(self) -> None:
        from train_gpt import CausalSelfAttention

        torch.manual_seed(0)
        B, T, D = 2, 5, 16
        E, R = 3, 4

        attn = CausalSelfAttention(
            dim=D,
            num_heads=4,
            num_kv_heads=2,
            rope_base=10000.0,
            qk_gain_init=1.0,
            kv_latent_dim=8,
            num_experts=E,
            expert_rank=R,
        )
        attn.eval()

        y = torch.randn(B, T, D, dtype=torch.float32)
        w = torch.softmax(torch.randn(B, T, E, dtype=torch.float32), dim=-1)

        out_fused = attn.mix_experts_from_shared(y, w)

        proj = attn.expert_proj.to(dtype=y.dtype)  # [E,R,D]
        out = attn.expert_out.to(dtype=y.dtype)    # [E,D,R]
        h = torch.einsum("btd,erd->bter", y, proj)
        out_e = torch.einsum("bter,edr->bted", h, out)  # [B,T,E,D]
        out_explicit = (w.unsqueeze(-1) * out_e).sum(dim=2)

        self.assertTrue(torch.allclose(out_fused, out_explicit, atol=1e-5, rtol=1e-5))

    def test_mlp_fused_mix_matches_explicit(self) -> None:
        from train_gpt import MLP, _rms_norm

        torch.manual_seed(0)
        B, T, D = 2, 7, 16
        E, R = 4, 6

        mlp = MLP(dim=D, mlp_mult=2.0, num_experts=E, expert_rank=R)
        mlp.eval()

        x = torch.randn(B, T, D, dtype=torch.float32)
        w = torch.softmax(torch.randn(B, T, E, dtype=torch.float32), dim=-1)

        out_fused = mlp.mix_experts(x, w)

        x_n = _rms_norm(x)
        gate_h = torch.einsum("btd,esd->btes", x_n, mlp.expert_gate.to(dtype=x_n.dtype))
        fc_h = torch.einsum("btd,esd->btes", x_n, mlp.expert_fc.to(dtype=x_n.dtype))
        # Phase 4.5 iter 22-add-all: fused path applies `hidden_post_norm` on the
        # per-expert hidden (shape (N, E, R)) after leaky_relu².  Mirror that in
        # the explicit path so the parity check holds.
        B_, T_, _, _ = gate_h.shape
        h_act = F.leaky_relu(gate_h, negative_slope=0.5).square() * fc_h  # [B,T,E,R]
        h_flat = h_act.reshape(B_ * T_, mlp.num_experts, mlp.expert_rank)
        h_flat = mlp.hidden_post_norm(h_flat)
        h = h_flat.reshape(B_, T_, mlp.num_experts, mlp.expert_rank)
        out_e = torch.einsum("btes,eds->bted", h, mlp.expert_down.to(dtype=x_n.dtype))
        out_explicit = (w.unsqueeze(-1) * out_e).sum(dim=2)

        self.assertTrue(torch.allclose(out_fused, out_explicit, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
