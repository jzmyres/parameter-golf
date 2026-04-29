"""Numerical equivalence between inline RMSNorm and F.rms_norm.

CLAUDE.md §7 + project_phase4_throughput_first.md require numerical
equivalence for any throughput micro-opt. Commit 68b0983 migrated 5
inline RMSNorm sites to ``F.rms_norm``; this test pins the equivalence
within fp tolerance for both float32 and bfloat16 paths and for both
shared-D and per-expert-(E,D) weight shapes.

The reference inline form is the canonical Llama / RWKV expression:

    y = x * rsqrt(mean(x*x) + eps) * weight

``F.rms_norm`` does the same fp32 statistic with eps and applies the
weight inside the kernel. We compare the two forms element-wise.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inline_rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference inline form, fp32 statistic, weight broadcast on the trailing dim."""
    dtype = x.dtype
    x32 = x.float()
    rsqrt = (x32.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    out = x32 * rsqrt
    return (out * weight.float()).to(dtype)


def _fused_rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """F.rms_norm with weight applied externally (mirrors the per-expert sites
    in train_gpt.py — F.rms_norm's `weight` arg requires the same shape as
    `normalized_shape`, so we apply the per-expert (E, D) weight outside).
    """
    D = x.size(-1)
    base = F.rms_norm(x, normalized_shape=(D,), eps=eps)
    return base * weight.to(base.dtype)


class TestRmsNormEquivalence(unittest.TestCase):
    def _check(self, shape: tuple[int, ...], weight_shape: tuple[int, ...],
               dtype: torch.dtype, atol: float, rtol: float) -> None:
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype)
        # Weight close to 1 (matches `nn.Parameter(torch.ones(...))` init).
        w = 1.0 + 0.05 * torch.randn(*weight_shape, dtype=torch.float32)
        ref = _inline_rms(x, w)
        got = _fused_rms(x, w)
        max_diff = (ref.float() - got.float()).abs().max().item()
        self.assertLess(max_diff, atol, f"max diff {max_diff:.3e} >= {atol:.0e}")
        self.assertTrue(torch.allclose(ref, got, atol=atol, rtol=rtol))

    def test_shared_d_float32(self):
        # x0 inject site (Block.forward): shape (B, T, D), weight (D,).
        self._check((4, 16, 64), (64,), torch.float32, atol=1e-5, rtol=1e-5)

    def test_shared_d_bfloat16(self):
        # bf16 has only ~7 mantissa bits; max abs delta on this size is ~1.5e-2.
        # The test pins that the two paths agree within bf16's representable
        # precision — anything tighter would fail on legitimate bf16 noise.
        self._check((4, 16, 64), (64,), torch.bfloat16, atol=3e-2, rtol=3e-2)

    def test_per_expert_d_float32(self):
        # MLA q_down site (CausalSelfAttention): shape (E, B, T, D), weight (E, D).
        E, B, T, D = 3, 2, 8, 32
        torch.manual_seed(1)
        x = torch.randn(E, B, T, D, dtype=torch.float32)
        w = 1.0 + 0.05 * torch.randn(E, D, dtype=torch.float32)
        # Per-expert: each expert e gets its own weight slice.
        ref_parts = [_inline_rms(x[e], w[e]) for e in range(E)]
        got_parts = [_fused_rms(x[e], w[e]) for e in range(E)]
        ref = torch.stack(ref_parts, dim=0)
        got = torch.stack(got_parts, dim=0)
        self.assertTrue(torch.allclose(ref, got, atol=1e-5, rtol=1e-5))

    def test_per_expert_d_bfloat16(self):
        E, B, T, D = 3, 2, 8, 32
        torch.manual_seed(2)
        x = torch.randn(E, B, T, D, dtype=torch.bfloat16)
        w = 1.0 + 0.05 * torch.randn(E, D, dtype=torch.float32)
        ref_parts = [_inline_rms(x[e], w[e]) for e in range(E)]
        got_parts = [_fused_rms(x[e], w[e]) for e in range(E)]
        ref = torch.stack(ref_parts, dim=0)
        got = torch.stack(got_parts, dim=0)
        # bf16 has only ~7 mantissa bits; per-expert max delta runs ~1.5e-2.
        max_diff = (ref.float() - got.float()).abs().max().item()
        self.assertLess(max_diff, 3e-2, f"bf16 max diff {max_diff:.3e}")

    def test_zero_input_safe(self):
        # eps must keep the kernel finite when x is exactly zero — this was
        # the canonical "RMSNorm crashes on cold start" failure mode.
        x = torch.zeros(2, 4, 8, dtype=torch.float32)
        w = torch.ones(8, dtype=torch.float32)
        out = _fused_rms(x, w)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
