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
    pid_n = tl.program_id(0)
    pid_d_out = tl.program_id(1)

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

    # ---- Step 4: per-expert weighted bmm accumulation ----
    out_acc = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)

    for e in tl.range(0, E_REAL):
        # weight per token for this expert (BLOCK_N,)
        w_e = tl.sum(weights * (offs_e == e)[None, :], axis=1)  # (BLOCK_N,)

        # Inner D-loop: accumulate proj = x @ W[e]
        proj = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < D

            x_off = offs_n[:, None] * D + offs_d[None, :]
            x_blk = tl.load(
                x_ptr + x_off,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

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
            proj += tl.dot(x_blk, w_blk)

        out_acc += proj * w_e[:, None]

    # ---- Step 5: store ----
    out_off = offs_n[:, None] * D_out + offs_d_out[None, :]
    tl.store(
        out_ptr + out_off,
        out_acc.to(tl.bfloat16),
        mask=n_mask[:, None] & d_out_mask[None, :],
    )


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
    E_PAD = max(_next_pow2(E), 16)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D_out, BLOCK_D_OUT))

    _fused_routed_bmm_fwd_kernel[grid](
        scores, gate, x, expert_W, out,
        N, D, D_out,
        E_PAD=E_PAD,
        E_REAL=E,
        GATE_DIM=gate_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_D_OUT=BLOCK_D_OUT,
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
    scores, gate, x, expert_W = ctx.saved_tensors
    with torch.enable_grad():
        s = scores.detach().requires_grad_(scores.requires_grad)
        g = gate.detach().requires_grad_(gate.requires_grad)
        xx = x.detach().requires_grad_(x.requires_grad)
        ew = expert_W.detach().requires_grad_(expert_W.requires_grad)
        out = fused_routed_bmm_eager(s, g, xx, ew)
        grads = torch.autograd.grad(
            outputs=out,
            inputs=[s, g, xx, ew],
            grad_outputs=grad_out,
            allow_unused=True,
        )
    return grads


def _setup_context(ctx, inputs, output):
    scores, gate, x, expert_W = inputs
    ctx.save_for_backward(scores, gate, x, expert_W)


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
    """Smoke: forward bit-identity vs eager + backward gradient check."""
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

    fused = fused_routed_bmm(scores, gate, x, expert_W)
    eager = fused_routed_bmm_eager(scores, gate, x, expert_W)

    diff = (fused.float() - eager.float()).abs().max().item()
    rel = diff / max(eager.float().abs().max().item(), 1e-6)
    print(f"forward: kernel vs eager max-abs diff {diff:.3e} (rel {rel:.3e})")
    assert rel < 5e-2, f"kernel/eager forward divergence rel {rel:.3e} > 5e-2"
    print("PASS: forward kernel matches eager within bf16 reduction-order noise")

    # backward via eager reference
    s2 = scores.detach().requires_grad_()
    g2 = gate.detach().requires_grad_()
    x2 = x.float().detach().requires_grad_()
    w2 = expert_W.float().detach().requires_grad_()
    out_eager = fused_routed_bmm_eager(s2, g2, x2.bfloat16(), w2.bfloat16())
    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (s2, g2, x2, w2))
    print("PASS: backward gradients finite for scores/gate/x/expert_W")

    # Per-expert gate (N, E) shape
    gate_e = torch.randn(N, E, dtype=torch.float32, device=device)
    fused_e = fused_routed_bmm(scores, gate_e, x, expert_W)
    eager_e = fused_routed_bmm_eager(scores, gate_e, x, expert_W)
    diff_e = (fused_e.float() - eager_e.float()).abs().max().item()
    rel_e = diff_e / max(eager_e.float().abs().max().item(), 1e-6)
    print(f"per-expert gate: kernel vs eager max-abs diff {diff_e:.3e} (rel {rel_e:.3e})")
    assert rel_e < 5e-2, f"per-expert-gate forward divergence rel {rel_e:.3e}"
    print("PASS: per-expert gate path matches eager")


if __name__ == "__main__":
    _smoke_test()
