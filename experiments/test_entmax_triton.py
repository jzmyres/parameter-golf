"""Standalone Triton entmax-1.5 kernel + correctness test (iter 117b-2 dev).

Validates a Triton-fused entmax-1.5 forward + backward kernel against the
pure-PyTorch closed-form implementation in train_gpt.py::entmax_1p5.

ALGORITHM VERIFICATION (2026-04-30, deep-spin/entmax cross-check):
  - Forward formulation matches train_gpt.py::entmax_1p5, which is
    mathematically equivalent to deep-spin/entmax Entmax15Function.forward
    (the reference uses X' = X/2 + tau'; we use the un-halved form with
    `0.5*(z - tau)` inside the square; algebraically tau = 2*tau').
  - Backward formula MATCHES deep-spin/entmax Entmax15Function.backward
    line-for-line:
        Reference:  gppr = sqrt(Y); dX = dY*gppr;
                    q = dX.sum(dim)/gppr.sum(dim); dX -= q * gppr
        This kernel: s = sqrt(w); c = sum(s*grad_w)/sum(s);
                     grad_z = s * (grad_w - c)
    Identical (note dX.sum = sum(dY*gppr) = sum(grad_w*sqrt(w)) = sum(s*grad_w)).
  - Reference: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py
  - Paper: Peters, Niculae, Martins (2019) "Sparse Sequence-to-Sequence Models"
    https://arxiv.org/pdf/1905.05702 (Algorithm 2 + Proposition 2 backward).
  - Numerical stability: our `discr.clamp_min(1e-6)` is STRICTER than the
    reference's `clamp(delta, 0)`; this is the iter 117 v3 NaN fix
    (sqrt(0) backward = Inf → 0×Inf = NaN propagation; ε=1e-6 caps the
    sqrt-gradient at 500, fixing a NaN bug not present in the reference).


The kernel is designed for the small-E regime (E=16 routed experts) where
all E values fit in registers and a single program block handles one row.

Forward algorithm (matches train_gpt.py::entmax_1p5):
  1. Shift z so max(z) = 0  (numerical stability)
  2. Sort z descending → z_sorted
  3. For each k ∈ {1..E}: compute prefix sums S_k, S2_k, then
     τ_k = (S_k - sqrt(max(S_k² - k·(S2_k - 4), eps))) / k
  4. Pick largest k where τ_k < z_sorted_k  (support size)
  5. w_i = max(0.5·(z_i - τ), 0)²

Backward (closed form, MUCH simpler than forward):
  dL/dz_j = sqrt(w_j) · (dL/dw_j - c)
  where c = Σ_supp sqrt(w_i)·dL/dw_i / Σ_supp sqrt(w_i)
  Derivation: differentiate w_i = 0.25(z_i - τ)² subject to Σw=1 gives
  ∂τ/∂z_j = sqrt(w_j) / Σ_supp sqrt(w_i) for j in support.

Run from repo root:
  ~/.conda/envs/opg/bin/python experiments/test_entmax_triton.py
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton entmax kernel tests require CUDA",
)


# ---------------------------------------------------------------------------
# Triton kernels

# Mirror train_gpt.py::_ENTMAX_TRITON_EPS — same singularity floor for both
# forward (`sqrt(discr)` radicand) and backward (`Σ√w` divisor).
_ENTMAX_TRITON_EPS: float = 1e-6


@triton.jit
def _entmax_1p5_fwd_kernel(
    z_ptr,
    w_ptr,
    B,
    eps: tl.constexpr,
    E_BLOCK: tl.constexpr,
):
    """One program per row; loads all E values into registers."""
    pid = tl.program_id(axis=0)
    if pid >= B:
        return
    offs = tl.arange(0, E_BLOCK)
    row_off = pid * E_BLOCK + offs
    z = tl.load(z_ptr + row_off).to(tl.float32)

    # Numerical stability: shift so max = 0.
    z = z - tl.max(z, axis=0)

    # Sort descending (Triton tl.sort default is ascending; use negation).
    z_sorted = -tl.sort(-z, dim=0)  # descending

    # Prefix sums.
    S = tl.cumsum(z_sorted, axis=0)
    z_sq = z_sorted * z_sorted
    S2 = tl.cumsum(z_sq, axis=0)

    # Candidate τ_k for each support size k = 1..E.
    k = (offs + 1).to(tl.float32)
    discr = S * S - k * (S2 - 4.0)
    discr = tl.maximum(discr, eps)
    tau_k = (S - tl.sqrt(discr)) / tl.maximum(k, 1.0)

    # Largest k where τ_k < z_sorted_k.
    valid = tau_k < z_sorted
    support = tl.sum(valid.to(tl.int32), axis=0)
    support = tl.maximum(support, 1)

    # Gather τ at index (support - 1). Mask trick: only one offs equals support-1.
    target_idx = support - 1
    tau = tl.sum(tl.where(offs == target_idx, tau_k, 0.0), axis=0)

    # w_i = max(0.5(z_i - τ), 0)²
    half = 0.5 * (z - tau)
    half = tl.maximum(half, 0.0)
    w = half * half

    tl.store(w_ptr + row_off, w.to(z.dtype))


@triton.jit
def _entmax_1p5_bwd_kernel(
    w_ptr,
    grad_w_ptr,
    grad_z_ptr,
    B,
    eps: tl.constexpr,
    E_BLOCK: tl.constexpr,
):
    """Backward closed form: dL/dz = sqrt(w) · (dL/dw - <dL/dw, sqrt(w)>/Σsqrt(w)).

    Outside support, w_i = 0 so sqrt(w_i) = 0 → dL/dz_i = 0 automatically.
    """
    pid = tl.program_id(axis=0)
    if pid >= B:
        return
    offs = tl.arange(0, E_BLOCK)
    row_off = pid * E_BLOCK + offs

    w = tl.load(w_ptr + row_off).to(tl.float32)
    g = tl.load(grad_w_ptr + row_off).to(tl.float32)

    s = tl.sqrt(tl.maximum(w, 0.0))
    s_sum = tl.sum(s, axis=0)
    sg_sum = tl.sum(s * g, axis=0)
    c = sg_sum / tl.maximum(s_sum, eps)

    grad_z = s * (g - c)
    tl.store(grad_z_ptr + row_off, grad_z.to(w.dtype))


# ---------------------------------------------------------------------------
# Python wrappers (will be wrapped in torch.library.custom_op when merged).


def entmax_1p5_triton_fwd(z: torch.Tensor) -> torch.Tensor:
    orig_shape = z.shape
    E = orig_shape[-1]
    if E & (E - 1) != 0:
        raise ValueError(f"entmax_1p5_triton requires E to be a power of 2, got {E}")
    z_flat = z.contiguous().view(-1, E)
    w_flat = torch.empty_like(z_flat)
    B = z_flat.shape[0]
    grid = (B,)
    _entmax_1p5_fwd_kernel[grid](z_flat, w_flat, B, eps=_ENTMAX_TRITON_EPS, E_BLOCK=E)
    return w_flat.view(orig_shape)


def entmax_1p5_triton_bwd(w: torch.Tensor, grad_w: torch.Tensor) -> torch.Tensor:
    orig_shape = w.shape
    E = orig_shape[-1]
    w_flat = w.contiguous().view(-1, E)
    g_flat = grad_w.contiguous().view(-1, E)
    grad_z = torch.empty_like(w_flat)
    B = w_flat.shape[0]
    grid = (B,)
    _entmax_1p5_bwd_kernel[grid](w_flat, g_flat, grad_z, B, eps=_ENTMAX_TRITON_EPS, E_BLOCK=E)
    return grad_z.view(orig_shape)


# ---------------------------------------------------------------------------
# Reference: import the pure-PyTorch entmax_1p5 from train_gpt.py.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# train_gpt.py has heavy module-level side effects; import only the function via
# AST extraction would be cleanest, but for a test we can re-implement inline.


def entmax_1p5_pyref(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pure-PyTorch reference (copy of train_gpt.py::entmax_1p5)."""
    z = z - z.amax(dim=dim, keepdim=True)
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    K = z.shape[dim]
    S = z_sorted.cumsum(dim=dim)
    S2 = (z_sorted * z_sorted).cumsum(dim=dim)
    k_range = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    k_shape = [1] * z.ndim
    k_shape[dim] = K
    k = k_range.view(*k_shape)
    discr = (S * S - k * (S2 - 4.0)).clamp_min(1e-6)
    tau_k = (S - discr.sqrt()) / k.clamp_min(1.0)
    valid = tau_k < z_sorted
    support = valid.to(dtype=torch.long).sum(dim=dim, keepdim=True).clamp_min(1)
    tau = tau_k.gather(dim, support - 1)
    return (0.5 * (z - tau)).clamp_min(0.0).square()


# ---------------------------------------------------------------------------
# Tests.


def _max_abs(x, y):
    return (x.float() - y.float()).abs().max().item()


def test_forward_correctness():
    """Compare Triton forward against pure-PyTorch reference."""
    torch.manual_seed(0)
    cases = [
        (8, 16),    # tiny
        (512, 16),  # batch
        (4096, 16), # full token-batch
    ]
    print(f"{'B':>6} {'E':>4} {'max_abs_err':>14} {'sum_err':>14} {'status':>8}")
    print("-" * 50)
    failures = 0
    for B, E in cases:
        z = torch.randn(B, E, device="cuda", dtype=torch.float32) * 2.0
        w_ref = entmax_1p5_pyref(z, dim=-1)
        w_triton = entmax_1p5_triton_fwd(z)
        err = _max_abs(w_ref, w_triton)
        sum_err = abs(w_triton.sum(-1).mean().item() - 1.0)
        status = "PASS" if err < 1e-4 else "FAIL"
        if err >= 1e-4:
            failures += 1
        print(f"{B:>6} {E:>4} {err:>14.2e} {sum_err:>14.2e} {status:>8}")
    return failures


def test_backward_correctness():
    """Compare Triton backward against autograd through pure-PyTorch reference."""
    torch.manual_seed(1)
    print(f"{'B':>6} {'E':>4} {'max_abs_err':>14} {'rel_err':>14} {'status':>8}")
    print("-" * 50)
    failures = 0
    for B, E in [(8, 16), (512, 16), (4096, 16)]:
        z = torch.randn(B, E, device="cuda", dtype=torch.float32, requires_grad=True) * 2.0
        # Autograd through reference for grad_z_ref.
        z_ref = z.detach().clone().requires_grad_(True)
        w_ref = entmax_1p5_pyref(z_ref, dim=-1)
        grad_w = torch.randn_like(w_ref)
        w_ref.backward(grad_w)
        grad_z_ref = z_ref.grad.detach()
        # Triton path.
        w_triton = entmax_1p5_triton_fwd(z.detach())
        grad_z_triton = entmax_1p5_triton_bwd(w_triton, grad_w)
        err = _max_abs(grad_z_ref, grad_z_triton)
        rel = err / (grad_z_ref.abs().max().item() + 1e-8)
        status = "PASS" if rel < 1e-3 else "FAIL"
        if rel >= 1e-3:
            failures += 1
        print(f"{B:>6} {E:>4} {err:>14.2e} {rel:>14.2e} {status:>8}")
    return failures


def test_throughput():
    """Compare per-call wallclock for Triton vs pure PyTorch."""
    import time

    torch.manual_seed(2)
    B, E = 8192, 16  # representative for B=4 × T=2048 × E=16 routing
    z = torch.randn(B, E, device="cuda", dtype=torch.float32) * 2.0
    n_iter = 100

    # Warmup.
    for _ in range(10):
        _ = entmax_1p5_pyref(z)
        _ = entmax_1p5_triton_fwd(z)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = entmax_1p5_pyref(z)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1000 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = entmax_1p5_triton_fwd(z)
    torch.cuda.synchronize()
    t_tri = (time.perf_counter() - t0) * 1000 / n_iter

    print(f"\nForward throughput (B={B}, E={E}, fp32, n_iter={n_iter}):")
    print(f"  pure PyTorch: {t_ref:.3f} ms/call")
    print(f"  Triton:       {t_tri:.3f} ms/call ({t_ref/t_tri:.2f}× speedup)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; cannot run Triton kernel.")
        sys.exit(1)
    print("=" * 50)
    print("Forward correctness")
    print("=" * 50)
    fwd_fail = test_forward_correctness()
    print()
    print("=" * 50)
    print("Backward correctness")
    print("=" * 50)
    bwd_fail = test_backward_correctness()
    if fwd_fail or bwd_fail:
        print(f"\nFAIL: fwd_fail={fwd_fail} bwd_fail={bwd_fail}")
        sys.exit(1)
    test_throughput()
    print("\nALL TESTS PASSED")
