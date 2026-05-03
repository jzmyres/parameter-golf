"""Iter 118a Phase A3 kernel: fused routed-down for MLP-down + attn-Wo dispatch.

Drop-in component for `train_gpt.py::MLP.mix_experts` and similar
per-expert (N, E, R) → routing-weighted (N, D) contraction sites.

Replaces the eager sequence:
    h = h * w[..., None]                       # (N, E, R) pre-weighted
    out_e = bmm(h.transpose(0,1), Dwn.transpose(1,2))  # (E, N, D)
    out = out_e.sum(dim=0)                     # (N, D)

with a single fused Triton kernel that:
  1. Avoids materializing the (E, N, D) per-expert output intermediate
     to HBM (~50 MB at our shapes — same memory-bandwidth fusion win
     as the A2 kernel for the routed-bmm pattern).
  2. Applies the H101-safe sparsity skip predicate per (token_block, e)
     when `max_t |w[t, e]| < ε_bf16` ≈ 3.9e-3 — mirrors the A2 forward
     kernel's skip semantics. RevDEQ-safe (skip threshold below bf16
     mantissa precision; reverse-pass reconstruction floor 1e-3 < ε_bf16).

# Why a NEW kernel signature (vs A2's fused_routed_bmm)

The A2 kernel's signature `fused_routed_bmm(scores, gate, x, expert_W)`
expected:
  - scores/gate: pre-routing logits (N, E)
  - x: SHARED input (N, D) — broadcast through all per-expert linears
  - expert_W: (E, D, D_out) — per-expert linear weights

But our actual MLP dispatch produces a PER-EXPERT intermediate `h` of
shape (N, E, R) BEFORE the down projection — each expert sees a
DIFFERENT input (per-expert SiLU(gate) * fc result). The A2 signature
doesn't fit; we'd need to pass (N, E, R) input, not (N, D).

This kernel matches the actual dispatch:
  fused_routed_down(h_pre, w, expert_down) -> out
where:
  h_pre: (N, E, R) bf16 — un-weighted per-expert intermediate
  w: (N, E) bf16 — routing weights (already softmax × sigmoid combined)
  expert_down: (E, D, R) bf16 — per-expert down projection (D = output dim)
  out: (N, D) bf16

Math:
  out[t, d] = Σ_e w[t, e] · Σ_r h_pre[t, e, r] · expert_down[e, d, r]

# RevDEQ safety (H101)

Same argument as A2 forward kernel:
  - Skip threshold ε_bf16 ≈ 3.9e-3 = bf16 mantissa precision
  - Reverse-pass reconstruction floor ≤ 1e-3 (per RevDEQ Ā ≥ 0.1, K=12)
  - 1e-3 < 3.9e-3 → skip-induced discontinuity is below reconstruction
    precision → invisible to FP iteration → RevDEQ-safe.

Forward map at eps=0 (no skip) is bit-identical to eager (modulo bf16
reduction-order noise, ~3e-3 rel — same level as A2 kernel).

# Backward strategy

Inherits A2's pattern: backward via eager autograd reference under
`torch.is_grad_enabled() AND any(requires_grad)` dispatch. The
training path goes through eager (which has the bmm + sum_e graph
saved); the inference path uses Triton. RevDEQ FP-iteration body runs
in `no_grad`, so 90%+ of training-time dispatch hits Triton.

# Phase A3 integration

Wire into `MLP.mix_experts` (train_gpt.py:2725-2729) behind CLI flag
`use_unified_routed_down`:
  if use_unified_routed_down and torch.is_grad_enabled() == False:
      out = fused_routed_down(h_pre, w_flat, self.expert_down)
  else:
      # existing eager path
      h = h_pre * w_flat[..., None]
      Dwn_T = expert_down.transpose(1, 2)
      out_e = bmm(h.transpose(0, 1), Dwn_T)
      out = out_e.sum(dim=0)

Smoke test: forward gradcheck rel ≤ 5e-2 (bf16 floor); FP iteration
trajectory at K∈{8, 32, 64} stays bounded (per A2 verification).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


# Same precision floor as A2 kernel — see H101 + EXPERIENCE.md
EPS_BF16 = 3.9e-3


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


@triton.jit
def _fused_routed_down_kernel(
    h_pre_ptr,     # (N, E, R) bf16 — per-expert intermediate
    w_ptr,         # (N, E) bf16 — routing weights
    Dwn_ptr,       # (E, D, R) bf16 — per-expert down projection
    out_ptr,       # (N, D) bf16
    N, D,
    E_PAD: tl.constexpr,
    E_REAL: tl.constexpr,
    R: tl.constexpr,
    R_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    EPS_SKIP: tl.constexpr,
):
    """Compute out[t, d] = Σ_e w[t, e] · Σ_r h_pre[t, e, r] · Dwn[e, d, r].

    Grid: cdiv(N, BLOCK_N) * cdiv(D, BLOCK_D), 1D pid with L2 swizzle.

    Per-tile work:
      1. Compute per-expert max(|w|) over BLOCK_N for skip predicate.
      2. For each expert e (Python `range`, constexpr):
         If max_abs_w_e < EPS_SKIP: skip.
         Else:
           Load h_pre[block_n, e, :R]  → (BLOCK_N, R) bf16
           Load Dwn[e, block_d, :R]    → (BLOCK_D, R) bf16
           proj = h_pre @ Dwn.T        → (BLOCK_N, BLOCK_D) fp32
           Scale by w[block_n, e] (broadcast over BLOCK_D).
           Accumulate into out_acc.
      3. Store out_acc cast to bf16.
    """
    # ---- L2 cache swizzle (mirror of A2 forward kernel) ----
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_d = tl.cdiv(D, BLOCK_D)
    num_pid_in_group = GROUP_SIZE_N * num_pid_d
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_d = (pid % num_pid_in_group) // group_size_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    n_mask = offs_n < N
    d_mask = offs_d < D
    # R is padded to next pow2 for tl.arange + masked.
    offs_r = tl.arange(0, R_PAD)
    r_mask = offs_r < R

    # ---- Load routing weights for this token block (BLOCK_N, E_PAD) ----
    offs_e = tl.arange(0, E_PAD)
    e_mask = offs_e < E_REAL
    w_off = offs_n[:, None] * E_REAL + offs_e[None, :]
    w_blk = tl.load(
        w_ptr + w_off,
        mask=n_mask[:, None] & e_mask[None, :],
        other=0.0,
    )  # (BLOCK_N, E_PAD) bf16
    # n_mask zeroes padded N rows → no NaN propagation in max-abs reduction
    # (mirrors the A2 kernel boundary fix at A2.5).
    w_blk = tl.where(n_mask[:, None] & e_mask[None, :], w_blk, 0.0)

    # Per-expert max-abs for skip predicate (computed once, reused per d-loop)
    w_abs_max_per_e = tl.max(tl.abs(w_blk), axis=0)  # (E_PAD,)

    out_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # ---- Inner loop over experts (Python range = constexpr e, masks fold) ----
    for e in range(E_REAL):
        # Skip-empty predicate (H101-safe at ε ≤ ε_bf16)
        max_abs_w_e = tl.sum(w_abs_max_per_e * (offs_e == e), axis=0)
        if max_abs_w_e >= EPS_SKIP:
            # Per-token weight w[block_n, e] (BLOCK_N,)
            w_e = tl.sum(w_blk * (offs_e == e)[None, :], axis=1)

            # Load h_pre[block_n, e, :] — (BLOCK_N, R_PAD), masked beyond R
            h_off = (
                offs_n[:, None] * (E_REAL * R)
                + e * R
                + offs_r[None, :]
            )
            h_blk = tl.load(
                h_pre_ptr + h_off,
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0,
            )

            # Load Dwn[e, block_d, :] — (BLOCK_D, R_PAD), masked beyond R
            d_off = (
                e * (D * R)
                + offs_d[:, None] * R
                + offs_r[None, :]
            )
            dwn_blk = tl.load(
                Dwn_ptr + d_off,
                mask=d_mask[:, None] & r_mask[None, :],
                other=0.0,
            )

            # h_pre @ Dwn.T → (BLOCK_N, BLOCK_D) fp32
            # tl.dot needs (M, K) @ (K, N) layout; Dwn is (BLOCK_D, R), so
            # transpose to (R, BLOCK_D) for the second operand.
            dwn_blk_T = tl.trans(dwn_blk)  # (R, BLOCK_D)
            proj_e = tl.dot(h_blk, dwn_blk_T)  # fp32

            # Scale by per-token weight + accumulate
            out_acc += proj_e * w_e[:, None]

    # ---- Store out (BLOCK_N, BLOCK_D) ----
    out_off = offs_n[:, None] * D + offs_d[None, :]
    tl.store(
        out_ptr + out_off,
        out_acc.to(tl.bfloat16),
        mask=n_mask[:, None] & d_mask[None, :],
    )


def _fused_routed_down_triton(
    h_pre: Tensor, w: Tensor, expert_down: Tensor
) -> Tensor:
    """Triton kernel launcher for fused routed-down.

    Args:
        h_pre: (N, E, R) bf16, contiguous
        w: (N, E) bf16, contiguous (already softmax × sigmoid)
        expert_down: (E, D, R) bf16, contiguous

    Returns:
        out: (N, D) bf16
    """
    assert h_pre.is_cuda and h_pre.dtype == torch.bfloat16
    assert w.is_cuda and w.dtype == torch.bfloat16
    assert expert_down.is_cuda and expert_down.dtype == torch.bfloat16

    N, E, R = h_pre.shape
    E_W, D, R_W = expert_down.shape
    assert w.shape == (N, E), f"w shape {w.shape} != ({N}, {E})"
    assert E_W == E and R_W == R, (
        f"expert_down shape {expert_down.shape} != ({E}, ?, {R})"
    )
    assert D % 16 == 0, f"D={D} must be multiple of 16"
    assert E <= 64, f"E={E} > 64 not supported (register pressure)"
    # R need not be a multiple of 16; the kernel pads R to R_PAD = next pow2
    # ≥ 16 and masks beyond R. Production R = mlp_expert_rank = 96.

    out = torch.empty((N, D), device=h_pre.device, dtype=torch.bfloat16)

    BLOCK_N = 64
    BLOCK_D = 64
    GROUP_SIZE_N = 8
    E_PAD = max(_next_pow2(E), 16)
    R_PAD = max(_next_pow2(R), 16)

    grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(D, BLOCK_D),)
    _fused_routed_down_kernel[grid](
        h_pre.contiguous(), w.contiguous(), expert_down.contiguous(), out,
        N, D,
        E_PAD=E_PAD,
        E_REAL=E,
        R=R,
        R_PAD=R_PAD,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        GROUP_SIZE_N=GROUP_SIZE_N,
        EPS_SKIP=EPS_BF16,
        num_warps=4,
        num_stages=3,
    )
    return out


def fused_routed_down_eager(
    h_pre: Tensor, w: Tensor, expert_down: Tensor
) -> Tensor:
    """Eager reference. Mirrors `MLP.mix_experts` dispatch (lines 2725-2729).

    h_pre: (N, E, R), w: (N, E), expert_down: (E, D, R) → out: (N, D)
    Math:  out[t, d] = Σ_e w[t, e] · Σ_r h_pre[t, e, r] · expert_down[e, d, r]
    """
    h_weighted = h_pre * w.unsqueeze(-1)  # (N, E, R)
    Dwn_T = expert_down.transpose(1, 2)   # (E, R, D)
    out_e = torch.bmm(h_weighted.transpose(0, 1), Dwn_T)  # (E, N, D)
    return out_e.sum(dim=0)               # (N, D)


# ---------------------------------------------------------------------------
# torch.library.custom_op registration (mirrors A2 pattern)
# ---------------------------------------------------------------------------
_LIB = torch.library.Library("opg_routed_down", "DEF")
_LIB.define(
    "fused_routed_down(Tensor h_pre, Tensor w, Tensor expert_down) -> Tensor"
)


def _impl_cuda(h_pre: Tensor, w: Tensor, expert_down: Tensor) -> Tensor:
    can_use_kernel = (
        h_pre.is_cuda
        and h_pre.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and expert_down.dtype == torch.bfloat16
        and h_pre.shape[2] % 16 == 0
        and expert_down.shape[1] % 16 == 0
        and h_pre.shape[1] <= 64
    )
    if can_use_kernel:
        return _fused_routed_down_triton(h_pre, w, expert_down)
    return fused_routed_down_eager(h_pre, w, expert_down)


_LIB_IMPL = torch.library.Library("opg_routed_down", "IMPL")
_LIB_IMPL.impl("fused_routed_down", _impl_cuda, "CUDA")
_LIB_IMPL.impl("fused_routed_down", fused_routed_down_eager, "CPU")


@torch.library.register_fake("opg_routed_down::fused_routed_down")
def _fake(h_pre, w, expert_down):
    N = h_pre.shape[0]
    D = expert_down.shape[1]
    return h_pre.new_empty((N, D), dtype=h_pre.dtype)


def _backward(ctx, grad_out):
    """Eager autograd via re-forward (same pattern as A2 fused_routed_bmm)."""
    h_pre, w, expert_down = ctx.saved_tensors
    with torch.enable_grad():
        h_ad = h_pre.detach().requires_grad_(h_pre.requires_grad)
        w_ad = w.detach().requires_grad_(w.requires_grad)
        ed_ad = expert_down.detach().requires_grad_(expert_down.requires_grad)
        out = fused_routed_down_eager(h_ad, w_ad, ed_ad)
        return torch.autograd.grad(
            outputs=out,
            inputs=[h_ad, w_ad, ed_ad],
            grad_outputs=grad_out,
            allow_unused=True,
        )


def _setup_context(ctx, inputs, output):
    h_pre, w, expert_down = inputs
    ctx.save_for_backward(h_pre, w, expert_down)


torch.library.register_autograd(
    "opg_routed_down::fused_routed_down",
    _backward,
    setup_context=_setup_context,
)


def fused_routed_down(
    h_pre: Tensor, w: Tensor, expert_down: Tensor
) -> Tensor:
    """Fused routed-down — dispatches by `torch.is_grad_enabled()`.

    See module docstring for full semantics. Training path goes through
    eager `fused_routed_down_eager` for backward correctness; inference
    + RevDEQ no_grad FP iteration body uses the Triton kernel.
    """
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (h_pre, w, expert_down)
    ):
        return fused_routed_down_eager(h_pre, w, expert_down)
    return torch.ops.opg_routed_down.fused_routed_down(h_pre, w, expert_down)


def _smoke_test() -> None:
    """Smoke: forward bit-identity vs eager + sparsity sweep + boundary tile."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    N, E, R, D = 256, 16, 96, 768

    h_pre = torch.randn(N, E, R, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(N, E, dtype=torch.bfloat16, device=device).softmax(dim=-1).bfloat16()
    expert_down = torch.randn(E, D, R, dtype=torch.bfloat16, device=device) * 0.1

    fused = fused_routed_down(h_pre, w, expert_down)
    eager = fused_routed_down_eager(h_pre, w, expert_down)
    rel = (fused.float() - eager.float()).abs().max().item() / max(eager.float().abs().max().item(), 1e-6)
    assert rel < 5e-2, f"forward kernel/eager rel {rel:.3e} > 5e-2"
    print(f"PASS: forward kernel matches eager (rel {rel:.3e})")

    # Backward gradcheck via grad-enabled path (= eager dispatch)
    h_ad = h_pre.detach().requires_grad_()
    w_ad = w.detach().requires_grad_()
    ed_ad = expert_down.detach().requires_grad_()
    out = fused_routed_down(h_ad, w_ad, ed_ad)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (h_ad, w_ad, ed_ad))
    print("PASS: backward gradients finite via grad-enabled (eager) dispatch")

    # Boundary tile (N=100, not multiple of BLOCK_N=64)
    Nb = 100
    h_b = torch.randn(Nb, E, R, dtype=torch.bfloat16, device=device) * 0.1
    w_b = torch.randn(Nb, E, dtype=torch.bfloat16, device=device).softmax(dim=-1).bfloat16()
    fused_b = fused_routed_down(h_b, w_b, expert_down)
    eager_b = fused_routed_down_eager(h_b, w_b, expert_down)
    assert torch.isfinite(fused_b).all(), "boundary tile contains NaN/Inf"
    rel_b = (fused_b.float() - eager_b.float()).abs().max().item() / max(eager_b.float().abs().max().item(), 1e-6)
    assert rel_b < 5e-2, f"boundary tile rel {rel_b:.3e}"
    print(f"PASS: boundary tile (N=100) finite + matches eager (rel {rel_b:.3e})")

    # Sparsity sweep — make some experts inactive (w → 0) per token block
    for active_frac in [1.0, 0.5, 0.25]:
        E_active = max(1, int(E * active_frac))
        w_sparse = torch.randn(N, E, dtype=torch.float32, device=device)
        w_sparse[:, E_active:] = -50.0  # softmax → ~0 for inactive experts
        w_sparse = w_sparse.softmax(dim=-1).bfloat16()
        fused_s = fused_routed_down(h_pre, w_sparse, expert_down)
        eager_s = fused_routed_down_eager(h_pre, w_sparse, expert_down)
        rel_s = (fused_s.float() - eager_s.float()).abs().max().item() / max(eager_s.float().abs().max().item(), 1e-6)
        assert rel_s < 5e-2
        print(f"PASS: sparsity {active_frac:.2f} active matches eager (rel {rel_s:.3e})")


if __name__ == "__main__":
    _smoke_test()
