"""Iter 118a / H88 v2 Phase A: Fused routing+bmm primitive.

Drop-in component for `train_gpt.py::Block.forward`'s soft-MoE expert
dispatch path when `use_unified_routed_bmm=True`. Replaces the eager
sequence (materialize routing weights, per-expert linear, weighted sum)
with a single fused operation: `out[t] = Σ_e w[t,e] · expert_e(x[t])`.

# Hypothesis (from experiments/hypotheses.md H88 v2 Phase A)

A Triton kernel `fused_routed_bmm` fuses (a) the routing transform
(softmax / entmax-blend / sigmoid-gate combine) with (b) the per-expert
linear projection and (c) the routing-weighted sum into ONE GPU kernel.

Memory bandwidth + kernel-launch savings: avoids materializing the
(N, E) routing tensor and the (N, E, D_out) per-expert intermediate
to HBM. Predicted 10-30% per-layer wallclock speedup.

# RevDEQ-safety (H101)

Phase A operates at eps=0 — NO skip, NO truncation, NO discrete decisions.
The fused kernel computes the SAME mathematical operation as eager,
just in registers without intermediate HBM writes. Forward map is
preserved bit-identically up to bf16 reduction-order noise.

This iter is forbidden from introducing top-K, magnitude-skip with
ε > ε_bf16, capacity-bounded gather, or argmax-style decisions —
those are H101-forbidden because they break the RevDEQ reverse-pass
reconstruction (forward map kink + reconstruction floor 1e-3 >
order-statistic gap 5e-4 → silently corrupted gradients).

Phase B (iter 118b) will explore RevDEQ-safe sparsity primitives
(Sinkhorn / Gumbel-softmax / polysparse Lipschitz) on top of this kernel.

# Math (eager reference)

Given:
    scores  : (N, E)              — pre-routing-transform logits
    gate    : (N, 1) or (N, E)    — pre-sigmoid gate logits
    x       : (N, D)              — input tokens (flattened batch+seq)
    W       : (E, D, D')          — per-expert linear weights

Compute:
    p_alloc = softmax(scores, dim=-1)                   # (N, E)
    weights = p_alloc * sigmoid(gate)                   # (N, E)
    out[t]  = Σ_e weights[t, e] * (x[t] @ W[e])         # (N, D')

The kernel fuses these three operations.

# Strict-generalization

`use_unified_routed_bmm=False` (default) keeps the eager path intact.
At `use_unified_routed_bmm=True`, the forward map is mathematically
identical to eager up to bf16 reduction-order noise (~1e-3). Promotion
under §11 strict-gen rule is unconditional on val_bpb non-regression.

# Integration into train_gpt.py

Three touchpoints (deferred to integration commit, not this scaffold):

1. `Hyperparameters.use_unified_routed_bmm = False` field.
2. `--use-unified-routed-bmm` CLI flag (add to `_CLI_TUNABLE_KNOBS`).
3. In `Block.forward` expert-dispatch sites (attn + mlp), conditional
   dispatch:
        if use_unified_routed_bmm:
            out = fused_routed_bmm(scores, gate, x, expert_W)
        else:
            # existing eager path

# Status

Phase A1 (this file): eager reference + smoke test asserting bit-identity
to current `Block.forward` eager path. Confirms the math.

Phase A2 (next): Triton kernel implementing the fused operation. Wraps
via `torch.library.custom_op` + `register_fake` + `register_autograd`
(NOT `torch.autograd.Function` — that path is incompatible with
torch.compile's AOTAutograd functionalization; iter 104 AdaSplash
DROPPED 2026-04-29 for this reason).

Phase A3 (last): wire into Block.forward + full 1000-step run.
"""

from __future__ import annotations

import torch
from torch import Tensor


def fused_routed_bmm_eager(
    scores: Tensor,
    gate: Tensor,
    x: Tensor,
    expert_W: Tensor,
) -> Tensor:
    """Eager reference for the fused routed-bmm primitive.

    Args:
        scores:   (N, E) pre-softmax allocation logits.
        gate:     (N, 1) or (N, E) pre-sigmoid gate logits.
        x:        (N, D) input tokens.
        expert_W: (E, D, D_out) per-expert linear weights.

    Returns:
        out: (N, D_out) routing-weighted sum of per-expert linear projections.
            out[t, d_out] = Σ_e [softmax(scores)[t,e] · sigmoid(gate)[t,e] ·
                                 (x[t] @ expert_W[e])[d_out]]

    Strict-gen: this function is mathematically identical to the current
    eager dispatch path; the Triton kernel must match this output
    bit-identically (up to bf16 reduction order).
    """
    p_alloc = torch.softmax(scores.float(), dim=-1)
    gate_act = torch.sigmoid(gate.float())
    weights = (p_alloc * gate_act).to(dtype=x.dtype)
    expert_outs = torch.einsum("nd,edk->nek", x, expert_W)
    out = torch.einsum("ne,nek->nk", weights, expert_outs)
    return out


def fused_routed_bmm(
    scores: Tensor,
    gate: Tensor,
    x: Tensor,
    expert_W: Tensor,
) -> Tensor:
    """Fused routing+bmm. Currently dispatches to eager reference.

    Phase A2 (TODO): replace with Triton kernel registered via
    `torch.library.custom_op` + `register_fake` + `register_autograd`.
    """
    return fused_routed_bmm_eager(scores, gate, x, expert_W)


def _smoke_test() -> None:
    """Smoke: forward bit-identity vs explicit eager soft-MoE."""
    torch.manual_seed(0)
    N, E, D, D_out = 64, 4, 16, 24
    scores = torch.randn(N, E, dtype=torch.float32)
    gate = torch.randn(N, 1, dtype=torch.float32)
    x = torch.randn(N, D, dtype=torch.bfloat16)
    expert_W = torch.randn(E, D, D_out, dtype=torch.bfloat16) * 0.1

    fused = fused_routed_bmm(scores, gate, x, expert_W)

    p_alloc = torch.softmax(scores, dim=-1)
    gate_act = torch.sigmoid(gate)
    weights = (p_alloc * gate_act).to(dtype=x.dtype)
    eager_outs = torch.stack([x @ expert_W[e] for e in range(E)], dim=1)
    eager_ref = (weights.unsqueeze(-1) * eager_outs).sum(dim=1)

    diff = (fused.float() - eager_ref.float()).abs().max().item()
    assert diff < 1e-2, f"fused vs eager max-abs diff {diff:.3e} > 1e-2 (bf16 floor)"
    print(f"PASS: fused_routed_bmm matches eager soft-MoE (max-abs diff {diff:.3e})")

    grad_x = torch.randn_like(fused)
    x_ad = x.float().detach().requires_grad_()
    eW_ad = expert_W.float().detach().requires_grad_()
    out_ad = fused_routed_bmm_eager(scores.detach(), gate.detach(), x_ad, eW_ad)
    out_ad.backward(grad_x.float())
    assert x_ad.grad is not None and eW_ad.grad is not None
    assert torch.isfinite(x_ad.grad).all() and torch.isfinite(eW_ad.grad).all()
    print(f"PASS: backward gradients finite (x.grad shape {tuple(x_ad.grad.shape)}, "
          f"eW.grad shape {tuple(eW_ad.grad.shape)})")


if __name__ == "__main__":
    _smoke_test()
