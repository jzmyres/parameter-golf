"""Iter 122 / H93: Logit softcap (Gemma2-style).

Drop-in for the LM head's final logits in `train_gpt.py::MoSLowRankOutputHead.forward`
(or wherever final logits are produced). Bounds extreme logit values via tanh
to limit gradient spikes and bf16 numerical issues.

# Hypothesis (from experiments/hypotheses.md H93)

Records use `logits = softcap * tanh(logits / softcap)` with softcap=30 on
every submission since 2026-04+. While our grad_norm history is healthy (0.05-
0.30) and grad_clip=1.0 absorbs spikes, the regularization effect on training
dynamics is real and consistent in records (~−0.01 BPB).

# Math

For softcap τ > 0:
    softcap_logits(z, τ) = τ · tanh(z / τ)

Properties:
- |output| ≤ τ (hard upper bound on logit magnitude)
- output ≈ z when |z| ≪ τ (linear regime)
- output → ±τ as |z| → ∞ (saturated regime)
- Differentiable everywhere; gradient = sech²(z/τ) ∈ (0, 1]
- For τ = 0 (or negative): no-op (identity)

# Strict-generalization

`softcap = 0` (or unset) recovers identity exactly. Promotion under §11
unconditional-promote rule when integrated with default 0 → recovers iter
117 v5 baseline.

# Integration into train_gpt.py

Two touchpoints:
1. `Hyperparameters.logit_softcap = 30.0` (or 0.0 default for strict-gen).
2. After the final logits computation in `MoSLowRankOutputHead.forward`, wrap:
       logits = softcap_logits(logits, args.logit_softcap)

Smoke-test with `python experiments/components/archive/logit_softcap.py`.
"""
import torch
from torch import Tensor


def softcap_logits(logits: Tensor, softcap: float = 30.0) -> Tensor:
    """Apply Gemma2-style tanh softcap to logits.

    Args:
        logits: any-shape tensor of pre-softcap logits.
        softcap: maximum |output|. Set to 0 (or negative) to disable (identity).

    Returns:
        softcap * tanh(logits / softcap), or `logits` unchanged if softcap ≤ 0.
    """
    if softcap <= 0:
        return logits
    return softcap * torch.tanh(logits / softcap)


def _smoke_test() -> None:
    print("logit_softcap.py smoke test:")

    # Case 1: identity at softcap=0 (disabled)
    z = torch.randn(4, 8, requires_grad=True)
    out = softcap_logits(z, softcap=0.0)
    assert torch.equal(out, z), "softcap=0 must be identity"
    print("  softcap=0 identity:        PASS")

    # Case 2: |output| <= softcap (hard bound)
    z = torch.tensor([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    out = softcap_logits(z, softcap=30.0)
    assert (out.abs() <= 30.0 + 1e-6).all(), f"|output| must be <= softcap, got {out.abs().max()}"
    print(f"  hard bound |out|<=30:      PASS (max |out| = {out.abs().max().item():.4f})")

    # Case 3: linear regime — output ≈ z when |z| ≪ softcap
    z = torch.tensor([-0.1, -0.01, 0.0, 0.01, 0.1])
    out = softcap_logits(z, softcap=30.0)
    rel_err = ((out - z).abs() / (z.abs() + 1e-9)).max().item()
    # tanh(x) ≈ x − x³/3 for small x; with x = z/30, error ~ (z/30)²/3
    # For |z| = 0.1, relative error ≈ (0.1/30)²/3 ≈ 4e-6
    assert rel_err < 1e-4, f"linear regime error {rel_err} too large"
    print(f"  linear regime |z|<=0.1:    PASS (max rel_err = {rel_err:.2e})")

    # Case 4: saturation — output → ±softcap as |z| → ∞
    z = torch.tensor([-1e8, 1e8])
    out = softcap_logits(z, softcap=30.0)
    assert torch.allclose(out, torch.tensor([-30.0, 30.0]), atol=1e-3), \
        f"saturation expected ±30, got {out}"
    print(f"  saturation |z|=1e8:        PASS (out = {out.tolist()})")

    # Case 5: gradient flows
    z = torch.randn(16, requires_grad=True)
    out = softcap_logits(z, softcap=30.0)
    loss = out.sum()
    loss.backward()
    assert z.grad is not None and z.grad.abs().max().item() > 0, \
        "gradient must flow through softcap"
    # Gradient = sech²(z/τ) ∈ (0, 1]; for |z| ≤ τ it's close to 1.
    assert z.grad.abs().max().item() <= 1.0 + 1e-6, \
        f"|grad| must be <=1, got {z.grad.abs().max().item()}"
    print(f"  gradient flow:             PASS (max |grad| = {z.grad.abs().max().item():.4f})")

    # Case 6: dtype preservation
    for dt in [torch.float32, torch.float64, torch.bfloat16]:
        z = torch.randn(8, dtype=dt) * 50.0
        out = softcap_logits(z, softcap=30.0)
        assert out.dtype == dt, f"dtype mismatch for {dt}"
    print("  dtype preservation:        PASS (fp32, fp64, bf16)")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
