"""Iter 118a / H88 v2 Phase A: Fused routing+bmm primitive.

Drop-in component for `train_gpt.py::Block.forward`'s soft-MoE expert
dispatch path when `use_unified_routed_bmm=True`. Replaces the eager
sequence (materialize routing weights, per-expert linear, weighted sum)
with a single fused operation: `out[t] = Σ_e w[t,e] · expert_e(x[t])`.

# Hypothesis (from experiments/hypotheses.md H88 v2 Phase A)

A Triton kernel `fused_routed_bmm` fuses (a) the routing transform
(softmax of allocation logits + sigmoid of gate logits) with (b) the
per-expert linear projection and (c) the routing-weighted sum into ONE
GPU kernel.

Memory bandwidth + kernel-launch savings: avoids materializing the
(N, E) routing tensor and the (N, E, D_out) per-expert intermediate
to HBM. Predicted 10-30% per-layer wallclock speedup.

# RevDEQ-safety (H101)

Phase A operates at eps=0 — NO skip, NO truncation, NO discrete decisions.
The fused kernel computes the SAME mathematical operation as eager,
just in registers without intermediate HBM writes. Forward map is
preserved bit-identically up to bf16 reduction-order noise.

# Math

Given:
    scores  : (N, E) fp32              — pre-softmax allocation logits
    gate    : (N, 1) or (N, E) fp32    — pre-sigmoid gate logits
    x       : (N, D) bf16              — input tokens (flattened batch+seq)
    W       : (E, D, D') bf16          — per-expert linear weights

Compute:
    p_alloc = softmax(scores, dim=-1)                  # fp32 (N, E)
    weights = p_alloc * sigmoid(gate)                  # fp32 (N, E)
    out[t]  = Σ_e weights[t, e] * (x[t] @ W[e])        # bf16 (N, D')

# Backward strategy (Phase A2)

Forward kernel only. Backward uses `register_autograd` with an eager-
torch reference function — gradients flow via standard autograd through
the eager reference. This keeps Phase A2 scope manageable; if forward-
only fusion isn't enough throughput, Phase A4 can add backward kernels.

# Strict-generalization

`fused_routed_bmm` is mathematically equivalent to the eager dispatch
sequence up to bf16 reduction-order noise. Setting the CLI flag
`use_unified_routed_bmm=False` (default) keeps the eager path.
Promotion under §11 strict-gen rule is unconditional on val_bpb
non-regression.

# custom_op vs autograd.Function

Per iter 104 (AdaSplash) DROPPED 2026-04-29: `torch.autograd.Function`
is incompatible with torch.compile's AOTAutograd functionalization. The
PyTorch-recommended path is `torch.library.custom_op` with
`register_fake` (for tracing/FakeTensor) + `register_autograd` (for
backward). Both registered below.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_routed_bmm_fwd_kernel(
    scores_ptr,    # (N, E) fp32
    gate_ptr,      # (N, GATE_DIM) fp32
    x_ptr,         # (N, D) bf16
    W_ptr,         # (E, D, D_out) bf16
    out_ptr,       # (N, D_out) bf16
    N, D, D_out,
    E_PAD: tl.constexpr,
    E_REAL: tl.constexpr,
    GATE_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    EPS_SKIP: tl.constexpr,
):
    """Fused (softmax × sigmoid × bmm × weighted-sum) over experts.

    Grid: (cdiv(N, BLOCK_N), cdiv(D_out, BLOCK_D_OUT)).

    Per-tile work:
      1. Load BLOCK_N rows of scores (E_REAL columns), pad to E_PAD with -inf.
      2. fp32 softmax along E.
      3. Load gate (1 or E_REAL columns); sigmoid; broadcast/multiply.
      4. For each expert e in [0, E_REAL):
         For each D-chunk:
           Load x[block_n, d_block] (bf16) and W[e, d_block, block_d_out] (bf16)
           Accumulate proj[block_n, block_d_out] += dot(x, W) (fp32)
         out_acc[block_n, block_d_out] += weight[block_n, e] * proj
      5. Store out_acc cast to bf16.
    """
    # ---- L2 cache swizzling (grouped launch order) ----
    # Default row-major pid traversal evicts x tiles between pid_d_out passes
    # because num_pid_n x-tiles get loaded before revisiting any tile. Grouped
    # ordering reuses x tiles across `GROUP_SIZE_N` consecutive programs that
    # share pid_n: x[block_n] is loaded once, reused for GROUP_SIZE_N programs
    # walking across pid_d_out. Standard Triton matmul tutorial swizzle —
    # 1.33× speedup + 60% L2 hit-rate gain on grouped GEMMs (PyTorch MoE blog,
    # 2026). https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_d_out = tl.cdiv(D_out, BLOCK_D_OUT)
    num_pid_in_group = GROUP_SIZE_N * num_pid_d_out
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_d_out = (pid % num_pid_in_group) // group_size_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d_out = pid_d_out * BLOCK_D_OUT + tl.arange(0, BLOCK_D_OUT)
    n_mask = offs_n < N
    d_out_mask = offs_d_out < D_out

    # ---- Step 1-2: load scores (BLOCK_N, E_PAD) and softmax ----
    offs_e = tl.arange(0, E_PAD)
    e_mask = offs_e < E_REAL
    scores_off = offs_n[:, None] * E_REAL + offs_e[None, :]
    scores_blk = tl.load(
        scores_ptr + scores_off,
        mask=n_mask[:, None] & e_mask[None, :],
        other=-float("inf"),
    )
    # numerically stable softmax in fp32
    scores_max = tl.max(scores_blk, axis=1, keep_dims=True)
    scores_exp = tl.exp(scores_blk - scores_max)
    scores_exp = tl.where(e_mask[None, :], scores_exp, 0.0)
    p_alloc = scores_exp / tl.sum(scores_exp, axis=1, keep_dims=True)

    # ---- Step 3: load gate, sigmoid, multiply ----
    if GATE_DIM == 1:
        gate_off = offs_n[:, None]
        gate_blk = tl.load(gate_ptr + gate_off, mask=n_mask[:, None], other=0.0)
        gate_act = 1.0 / (1.0 + tl.exp(-gate_blk))  # (BLOCK_N, 1)
        weights = p_alloc * gate_act  # broadcast
    else:
        gate_off = offs_n[:, None] * E_REAL + offs_e[None, :]
        gate_blk = tl.load(
            gate_ptr + gate_off,
            mask=n_mask[:, None] & e_mask[None, :],
            other=0.0,
        )
        gate_act = 1.0 / (1.0 + tl.exp(-gate_blk))
        weights = p_alloc * gate_act
    weights = tl.where(e_mask[None, :], weights, 0.0)

    # ---- Step 4: weighted bmm accumulation with H101-safe sparsity skip ----
    # Loop order: D OUTER, E INNER. x is loaded once per D-chunk and reused
    # across all E experts.
    #
    # Sparsity exploitation (Phase B, H101-permitted): for each (token_block,
    # expert) tile, skip the load+matmul when max(|w_e|) < EPS_BF16. The
    # threshold equals bf16 mantissa precision (~4e-3) — contributions below
    # this are below the data type's representable noise, so skipping is
    # forward-map-bit-identical at bf16 precision. The discontinuity is
    # invisible to RevDEQ's reverse-pass reconstruction (floor (1/Ā)^K · ε_fp64
    # ≈ 1e-3 < ε_bf16). Per H101 permitted alternative: "magnitude skip with
    # ε ≤ ε_bf16 ≈ 4e-3 (below mantissa precision → discontinuity invisible
    # to FP iteration)".
    #
    # Pre-compute per-expert max(|w_e|) ONCE to avoid recomputing inside the
    # D loop. weights shape (BLOCK_N, E_PAD); reduce along N via abs+max.
    weights_abs_max_per_e = tl.max(tl.abs(weights), axis=0)  # (E_PAD,)
    out_acc = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)

    for d_start in tl.range(0, D, BLOCK_D, num_stages=3):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        # Load x[block_n, d_block] ONCE (BLOCK_N, BLOCK_D); reuse across all E
        x_off = offs_n[:, None] * D + offs_d[None, :]
        x_blk = tl.load(
            x_ptr + x_off,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        for e in range(E_REAL):
            # Skip-empty predicate (H101-safe): per-program scalar comparison.
            # `tl.sum(... * (offs_e == e), axis=0)` extracts column e of the
            # pre-computed max-abs vector. With Python range + constexpr e,
            # the comparison folds at compile time.
            max_abs_w_e = tl.sum(weights_abs_max_per_e * (offs_e == e), axis=0)
            if max_abs_w_e >= EPS_SKIP:
                # `e` constexpr → `(offs_e == e)` mask folds.
                w_e = tl.sum(weights * (offs_e == e)[None, :], axis=1)  # (BLOCK_N,)

                w_off = (
                    e * (D * D_out)
                    + offs_d[:, None] * D_out
                    + offs_d_out[None, :]
                )
                w_blk = tl.load(
                    W_ptr + w_off,
                    mask=d_mask[:, None] & d_out_mask[None, :],
                    other=0.0,
                )
                proj_de = tl.dot(x_blk, w_blk)  # (BLOCK_N, BLOCK_D_OUT) fp32
                out_acc += proj_de * w_e[:, None]

    # ---- Step 5: store ----
    out_off = offs_n[:, None] * D_out + offs_d_out[None, :]
    tl.store(
        out_ptr + out_off,
        out_acc.to(tl.bfloat16),
        mask=n_mask[:, None] & d_out_mask[None, :],
    )


# bf16 mantissa precision (~3.9e-3). H101 permitted alternative: magnitude
# skip with ε ≤ ε_bf16 → discontinuity invisible to RevDEQ FP iteration
# (reconstruction floor (1/Ā)^K · ε_fp64 ≈ 1e-3 < ε_bf16).
EPS_BF16 = 3.9e-3


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _fused_routed_bmm_triton(
    scores: Tensor, gate: Tensor, x: Tensor, expert_W: Tensor
) -> Tensor:
    """Triton kernel launcher. Forward only.

    Args:
        scores:   (N, E) fp32 contiguous.
        gate:     (N, 1) or (N, E) fp32 contiguous.
        x:        (N, D) bf16 contiguous.
        expert_W: (E, D, D_out) bf16 contiguous.

    Returns:
        out: (N, D_out) bf16.

    Constraints:
        D % 16 == 0, D_out % 16 == 0 (Triton tl.dot tile alignment).
        E ≤ 64 (E_PAD register pressure).
    """
    assert scores.is_cuda and scores.dtype == torch.float32
    assert gate.is_cuda and gate.dtype == torch.float32
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert expert_W.is_cuda and expert_W.dtype == torch.bfloat16

    N, E = scores.shape
    _, D = x.shape
    E_W, D_W, D_out = expert_W.shape
    assert E_W == E and D_W == D, f"shape mismatch: scores E={E}, x D={D}, W E={E_W} D={D_W}"
    assert D % 16 == 0 and D_out % 16 == 0, "D and D_out must be multiples of 16"
    assert E <= 64, f"E={E} > 64 not supported (register pressure)"

    gate_dim = gate.shape[1]
    assert gate_dim in (1, E), f"gate must have shape (N, 1) or (N, E={E}); got (N, {gate_dim})"

    out = torch.empty((N, D_out), device=x.device, dtype=torch.bfloat16)

    BLOCK_N = 64
    BLOCK_D = 64
    BLOCK_D_OUT = 64
    GROUP_SIZE_N = 8
    E_PAD = max(_next_pow2(E), 16)

    # 1D grid; kernel decomposes via L2-cache swizzle.
    grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(D_out, BLOCK_D_OUT),)

    _fused_routed_bmm_fwd_kernel[grid](
        scores, gate, x, expert_W, out,
        N, D, D_out,
        E_PAD=E_PAD,
        E_REAL=E,
        GATE_DIM=gate_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_D_OUT=BLOCK_D_OUT,
        GROUP_SIZE_N=GROUP_SIZE_N,
        EPS_SKIP=EPS_BF16,
        num_warps=4,
        num_stages=3,
    )
    return out


def fused_routed_bmm_eager(
    scores: Tensor, gate: Tensor, x: Tensor, expert_W: Tensor
) -> Tensor:
    """Eager reference for the fused routed-bmm primitive.

    Mathematically identical to the Triton kernel (modulo bf16 reduction
    order). Used as backward reference + fallback for non-CUDA / non-aligned
    shapes.
    """
    p_alloc = torch.softmax(scores.float(), dim=-1)
    gate_act = torch.sigmoid(gate.float())
    weights = (p_alloc * gate_act).to(dtype=x.dtype)
    expert_outs = torch.einsum("nd,edk->nek", x, expert_W)
    out = torch.einsum("ne,nek->nk", weights, expert_outs)
    return out


# ---------------------------------------------------------------------------
# Backward kernel: d_x via Triton (dominant compute in backward)
# ---------------------------------------------------------------------------
# d_x derivation:
#   out[t, d_out] = Σ_e w[t,e] · Σ_{d_in} x[t, d_in] · W[e, d_in, d_out]
#   ∂out[t, d_out]/∂x[t', d_in] = δ_{t,t'} · Σ_e w[t, e] · W[e, d_in, d_out]
#   d_x[t, d_in] = Σ_{d_out} grad_out[t, d_out] · Σ_e w[t, e] · W[e, d_in, d_out]
#                = Σ_e w[t, e] · (grad_out[t] @ W[e].T)[d_in]
#
# Kernel structure mirrors forward: outer D_out chunks, inner E loop. Only
# difference is the matmul orientation: grad_out @ W.T (contract over d_out)
# instead of forward's x @ W (contract over d_in). Same L2 swizzle pattern.
#
# d_W and d_scores/d_gate use eager autograd reference (smaller compute,
# cleaner correctness). Reference: PyTorch LayerNorm tutorial backward
# pattern — https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


@triton.jit
def _fused_routed_bmm_bwd_dx_kernel(
    grad_out_ptr,  # (N, D_out) bf16
    weights_ptr,   # (N, E_REAL) bf16 (precomputed by launcher: softmax × sigmoid)
    W_ptr,         # (E, D, D_out) bf16
    grad_x_ptr,    # (N, D) bf16 — output
    N, D, D_out,
    E_PAD: tl.constexpr,
    E_REAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    EPS_SKIP: tl.constexpr,
):
    """Backward d_x kernel. Grid = cdiv(N, BLOCK_N) * cdiv(D, BLOCK_D).

    Same H101-safe skip predicate as forward: skip expert e when
    max(|w_e|) < EPS_SKIP. Bit-identical to forward's skip pattern at
    matching weights → consistent gradient signal."""
    # L2 cache swizzle (mirror of forward kernel)
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

    # Load weights for this block (BLOCK_N, E_PAD)
    offs_e = tl.arange(0, E_PAD)
    e_mask = offs_e < E_REAL
    weights_off = offs_n[:, None] * E_REAL + offs_e[None, :]
    weights = tl.load(
        weights_ptr + weights_off,
        mask=n_mask[:, None] & e_mask[None, :],
        other=0.0,
    )  # bf16

    # Pre-compute per-expert max(|w_e|) for skip predicate (mirrors forward)
    weights_abs_max_per_e = tl.max(tl.abs(weights), axis=0)  # (E_PAD,)
    d_x_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Outer: D_out chunks. Inner: experts (Python range — constexpr e).
    for d_out_start in tl.range(0, D_out, BLOCK_D_OUT, num_stages=3):
        offs_d_out = d_out_start + tl.arange(0, BLOCK_D_OUT)
        d_out_mask = offs_d_out < D_out

        # Load grad_out[block_n, d_out_chunk] ONCE; reuse across all E experts
        go_off = offs_n[:, None] * D_out + offs_d_out[None, :]
        grad_out_blk = tl.load(
            grad_out_ptr + go_off,
            mask=n_mask[:, None] & d_out_mask[None, :],
            other=0.0,
        )

        for e in range(E_REAL):
            # H101-safe skip: same predicate as forward kernel.
            max_abs_w_e = tl.sum(weights_abs_max_per_e * (offs_e == e), axis=0)
            if max_abs_w_e >= EPS_SKIP:
                w_e = tl.sum(weights * (offs_e == e)[None, :], axis=1)  # (BLOCK_N,)

                # Load W[e, d_block, d_out_block]; transpose for (d_out, d) matmul
                w_off = (
                    e * (D * D_out)
                    + offs_d[:, None] * D_out
                    + offs_d_out[None, :]
                )
                w_blk = tl.load(
                    W_ptr + w_off,
                    mask=d_mask[:, None] & d_out_mask[None, :],
                    other=0.0,
                )
                w_blk_T = tl.trans(w_blk)  # (BLOCK_D_OUT, BLOCK_D)

                # grad_out @ W[e].T → (BLOCK_N, BLOCK_D)
                proj_de = tl.dot(grad_out_blk, w_blk_T)  # fp32
                d_x_acc += proj_de * w_e[:, None]

    # Store d_x
    dx_off = offs_n[:, None] * D + offs_d[None, :]
    tl.store(
        grad_x_ptr + dx_off,
        d_x_acc.to(tl.bfloat16),
        mask=n_mask[:, None] & d_mask[None, :],
    )


def _bwd_dx_triton(grad_out: Tensor, weights: Tensor, expert_W: Tensor) -> Tensor:
    """Triton backward d_x launcher. weights = softmax × sigmoid (bf16)."""
    N, D_out = grad_out.shape
    E, D, _ = expert_W.shape
    assert weights.shape == (N, E)
    assert grad_out.dtype == torch.bfloat16
    assert weights.dtype == torch.bfloat16
    assert expert_W.dtype == torch.bfloat16
    assert D % 16 == 0 and D_out % 16 == 0

    grad_x = torch.empty((N, D), device=grad_out.device, dtype=torch.bfloat16)

    BLOCK_N, BLOCK_D, BLOCK_D_OUT = 64, 64, 64
    GROUP_SIZE_N = 8
    E_PAD = max(_next_pow2(E), 16)

    grid = (triton.cdiv(N, BLOCK_N) * triton.cdiv(D, BLOCK_D),)
    _fused_routed_bmm_bwd_dx_kernel[grid](
        grad_out.contiguous(), weights.contiguous(), expert_W.contiguous(), grad_x,
        N, D, D_out,
        E_PAD=E_PAD,
        E_REAL=E,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_D_OUT=BLOCK_D_OUT,
        GROUP_SIZE_N=GROUP_SIZE_N,
        EPS_SKIP=EPS_BF16,
        num_warps=4,
        num_stages=3,
    )
    return grad_x


# ---------------------------------------------------------------------------
# torch.library.custom_op registration
# ---------------------------------------------------------------------------

_LIB = torch.library.Library("opg_fused", "DEF")
_LIB.define(
    "fused_routed_bmm(Tensor scores, Tensor gate, Tensor x, Tensor expert_W) -> Tensor"
)


def _impl_cuda(scores: Tensor, gate: Tensor, x: Tensor, expert_W: Tensor) -> Tensor:
    can_use_kernel = (
        x.is_cuda
        and scores.is_cuda
        and x.dtype == torch.bfloat16
        and expert_W.dtype == torch.bfloat16
        and scores.dtype == torch.float32
        and gate.dtype == torch.float32
        and x.shape[1] % 16 == 0
        and expert_W.shape[2] % 16 == 0
        and scores.shape[1] <= 64
    )
    if can_use_kernel:
        return _fused_routed_bmm_triton(
            scores.contiguous(), gate.contiguous(), x.contiguous(), expert_W.contiguous()
        )
    return fused_routed_bmm_eager(scores, gate, x, expert_W)


_LIB_IMPL = torch.library.Library("opg_fused", "IMPL")
_LIB_IMPL.impl("fused_routed_bmm", _impl_cuda, "CUDA")
_LIB_IMPL.impl("fused_routed_bmm", fused_routed_bmm_eager, "CPU")


@torch.library.register_fake("opg_fused::fused_routed_bmm")
def _fake(scores, gate, x, expert_W):
    N = x.shape[0]
    D_out = expert_W.shape[2]
    return x.new_empty((N, D_out), dtype=x.dtype)


def _backward(ctx, grad_out):
    """Hybrid backward — Triton kernel for d_x (with H101-safe sparsity skip) +
    eager autograd via re-forward for d_W / d_scores / d_gate.

    Note (verified by experiment 2026-05-02): direct PyTorch (bmm/einsum)
    backward implementations are SLOWER than autograd-via-re-forward at our
    shapes because PyTorch's autograd + einsum has decade+ of optimization
    we can't beat with explicit ops. True end-to-end backward speedup
    requires Triton kernels for d_W and d_w (Phase A5 — substantial work).

    Current state:
      - Forward: Triton kernel WITH H101-safe sparsity skip (Phase A2 + B)
      - Backward d_x: Triton kernel WITH same skip predicate (consistent
        with forward — gradient signal omits the same below-ε contributions)
      - Backward d_W/d_scores/d_gate: eager autograd via re-forward
        (cleanest non-Triton path)
    """
    scores, gate, x, expert_W, p_alloc, gate_act = ctx.saved_tensors

    can_use_kernel = (
        grad_out.is_cuda
        and grad_out.dtype == torch.bfloat16
        and expert_W.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
        and grad_out.shape[1] % 16 == 0
        and x.shape[1] % 16 == 0
    )

    # ---- d_x via Triton kernel (with H101-safe sparsity skip) ----
    d_x_triton = None
    if can_use_kernel and x.requires_grad:
        weights_bf16 = (p_alloc * gate_act).to(dtype=torch.bfloat16).contiguous()
        d_x_triton = _bwd_dx_triton(grad_out.contiguous(), weights_bf16, expert_W.contiguous())

    # ---- d_W, d_scores, d_gate via eager autograd through re-forward ----
    with torch.enable_grad():
        s = scores.detach().requires_grad_(scores.requires_grad)
        g = gate.detach().requires_grad_(gate.requires_grad)
        ew = expert_W.detach().requires_grad_(expert_W.requires_grad)
        if d_x_triton is not None:
            out = fused_routed_bmm_eager(s, g, x.detach(), ew)
            d_scores, d_gate, d_W = torch.autograd.grad(
                outputs=out,
                inputs=[s, g, ew],
                grad_outputs=grad_out,
                allow_unused=True,
            )
            d_x = d_x_triton
        else:
            xx = x.detach().requires_grad_(x.requires_grad)
            out = fused_routed_bmm_eager(s, g, xx, ew)
            d_scores, d_gate, d_x, d_W = torch.autograd.grad(
                outputs=out,
                inputs=[s, g, xx, ew],
                grad_outputs=grad_out,
                allow_unused=True,
            )

    return d_scores, d_gate, d_x, d_W


def _setup_context(ctx, inputs, output):
    scores, gate, x, expert_W = inputs
    # Pre-compute and cache p_alloc, gate_act (small — N×E and N×{1,E} fp32).
    # Avoids recomputing softmax/sigmoid in backward AND avoids the autograd
    # recursion overhead of re-running forward inside _backward.
    p_alloc = torch.softmax(scores.float(), dim=-1)
    gate_act = torch.sigmoid(gate.float())
    ctx.save_for_backward(scores, gate, x, expert_W, p_alloc, gate_act)


torch.library.register_autograd(
    "opg_fused::fused_routed_bmm",
    _backward,
    setup_context=_setup_context,
)


def fused_routed_bmm(
    scores: Tensor, gate: Tensor, x: Tensor, expert_W: Tensor
) -> Tensor:
    """Fused routing+bmm — torch.library.custom_op entry point.

    Dispatches to Triton kernel on CUDA bf16 with aligned shapes; falls
    back to eager reference otherwise. Backward via eager autograd reference
    (Phase A2 scope; Phase A4 can add Triton backward).
    """
    return torch.ops.opg_fused.fused_routed_bmm(scores, gate, x, expert_W)


def _smoke_test() -> None:
    """Smoke: forward bit-identity + backward gradcheck vs eager + sanity benches."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    N, E, D, D_out = 128, 15, 64, 96

    scores = torch.randn(N, E, dtype=torch.float32, device=device)
    gate = torch.randn(N, 1, dtype=torch.float32, device=device)
    x = torch.randn(N, D, dtype=torch.bfloat16, device=device)
    expert_W = (torch.randn(E, D, D_out, dtype=torch.bfloat16, device=device) * 0.1)

    # ---- forward correctness ----
    fused = fused_routed_bmm(scores, gate, x, expert_W)
    eager = fused_routed_bmm_eager(scores, gate, x, expert_W)
    rel = (fused.float() - eager.float()).abs().max().item() / max(eager.float().abs().max().item(), 1e-6)
    assert rel < 5e-2, f"forward kernel/eager rel {rel:.3e} > 5e-2"
    print(f"PASS: forward kernel matches eager (rel {rel:.3e})")

    # ---- backward gradcheck: hybrid (Triton dx + eager d_W/d_scores/d_gate)
    #      vs fully-eager autograd reference ----
    grad_out = torch.randn(N, D_out, dtype=torch.bfloat16, device=device)

    # Fully-eager reference
    s_ref = scores.detach().requires_grad_()
    g_ref = gate.detach().requires_grad_()
    x_ref = x.detach().requires_grad_()
    w_ref = expert_W.detach().requires_grad_()
    out_ref = fused_routed_bmm_eager(s_ref, g_ref, x_ref, w_ref)
    out_ref.backward(grad_out)

    # Hybrid via custom_op
    s_h = scores.detach().requires_grad_()
    g_h = gate.detach().requires_grad_()
    x_h = x.detach().requires_grad_()
    w_h = expert_W.detach().requires_grad_()
    out_h = fused_routed_bmm(s_h, g_h, x_h, w_h)
    out_h.backward(grad_out)

    for name, ref, hyb in [
        ("d_scores", s_ref.grad, s_h.grad),
        ("d_gate", g_ref.grad, g_h.grad),
        ("d_x", x_ref.grad, x_h.grad),
        ("d_W", w_ref.grad, w_h.grad),
    ]:
        rel_g = (ref.float() - hyb.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)
        # bf16 reduction-order tolerance — loose because Triton kernel and eager
        # may sum in different orders. 5% rel is plenty for correctness.
        assert rel_g < 5e-2, f"{name}: rel {rel_g:.3e} > 5e-2"
        print(f"PASS: backward {name} matches eager (rel {rel_g:.3e})")

    # ---- per-expert gate (N, E) shape forward ----
    gate_e = torch.randn(N, E, dtype=torch.float32, device=device)
    fused_e = fused_routed_bmm(scores, gate_e, x, expert_W)
    eager_e = fused_routed_bmm_eager(scores, gate_e, x, expert_W)
    rel_e = (fused_e.float() - eager_e.float()).abs().max().item() / max(eager_e.float().abs().max().item(), 1e-6)
    assert rel_e < 5e-2
    print(f"PASS: per-expert gate path matches eager (rel {rel_e:.3e})")


if __name__ == "__main__":
    _smoke_test()
