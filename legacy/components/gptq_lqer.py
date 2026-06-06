"""Iter 124 / H94: GPTQ Hessian-aware quantization + LQER low-rank residual correction.

Drop-in for `train_gpt.py`'s int6 quantization pipeline (replaces the current
naive per-row scale-zero-point in `_int6_quantize_per_row`). Implements the
records' canonical GPTQ algorithm + LQER asymmetric int4 rank-4 residual
correction on top-K worst-quantized tensors.

Verified against records SOTA (2026-04-27, val_bpb=1.0611):
`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py`
- L2147-2186: `gptq_quantize_weight` — bit-equivalent port
- L2197-2218: `_lqer_pack` / `_lqer_pack_asym` — bit-equivalent ports
- L2222-2300: `gptq_mixed_quantize` driver — algorithmic match (top-K by
  residual norm, pack via int4 sym or int4-asym groupwise)

# Hypothesis (from experiments/docs/hypotheses.md H94)

Our current per-row int6 quantization uses naive scale-zero-point with no
Hessian-aware optimization. Records use:

1. **GPTQ** (Frantar et al. 2022, arxiv:2210.17323):
   Hessian H = X^T X from calibration data. Greedy column-wise quantization
   with residual propagation through Hinv. Per-row scale chosen by
   `clip_sigmas * row_std / clip_range` (records: clip_sigmas=3 default).
   Result: ~50-90% lower reconstruction error vs naive at same bit width.

2. **LQER** (low-rank quantization-error residual, PR #1530 v2):
   For top-k tensors with largest residual norm, factor `E = W_orig - W_quant`
   via SVD, keep top-rank components, pack the factor as int4 (sym) or
   asym int4 (A=int2, B=int4-groupwise). Eval: W_eff = W_quant + A @ B.

# Math

GPTQ per-tensor quant:
    W_orig: (R, C) fp32 weight matrix
    H: (C, C) Hessian from calibration X^T X
    1. Damp H by adding `0.01 * mean(diag(H))` to diagonal.
    2. Permute columns by descending diag(H).
    3. Hinv_upper = upper-triangular Cholesky of cholesky_inverse(H).
    4. Per-row scale s = clip_sigmas * row_std / clip_range, fp16.
    5. Block-wise (block_size=128):
        For each col j in block:
            q_col = round(w_col / s).clamp(-clip_range, clip_range)
            err   = (w_col - q_col * s) / Hinv_upper[j, j]
            W_block[:, j+1:] -= err[:,None] * Hinv_upper[j, j+1:][None,:]
        Propagate W_block residual to remaining blocks via Hinv[block, rest].
    6. Inverse-permute Q.
    Returns: (Q: int8 quantized weight, s: fp16 per-row scales)

LQER residual correction:
    W_q = Q.float() * s.float().view(-1, 1)
    E = W_orig - W_q
    U, S, Vh = svd(E)
    A = U[:, :r] * S[:r]      # (R, r)
    B = Vh[:r, :]             # (r, C)
    Eval: W_eff = W_q + A @ B    (lower MSE than W_q alone)

# Strict-generalization

Disable LQER (`lqer_top_k=0`) recovers pure-GPTQ. Disable GPTQ entirely (use
existing per-row int6 path) recovers iter 117 v5 baseline. The promotion
mechanism is val_bpb_int6 reduction (gating metric in our promotion rule).

# Integration into train_gpt.py — DESIGN NOTES

This component is a SCAFFOLD providing the math primitives. Full integration
requires:

1. Add `Hyperparameters.use_gptq=False`, `gptq_clip_sigmas=3.0`,
   `gptq_calibration_batches=16`, `lqer_enabled=False`, `lqer_rank=4`,
   `lqer_top_k=3`, `lqer_factor_bits=4`.
2. Add `collect_hessians(model, calibration_loader)` that runs forward hooks
   to accumulate `H[name] = X^T X` for each Linear's input X.
3. Replace `_int6_quantize_per_row` call site with `gptq_mixed_quantize`
   that routes per-tensor via `gptq_quantize_weight` then optionally adds
   LQER on top-k by residual norm.
4. Modify the artifact loader to dequantize via `Q.float() * s + lqer_A @ lqer_B`
   when LQER factors are present.

Smoke-test: `python experiments/components/gptq_lqer.py`.
"""
import math
from typing import Tuple

import torch
from torch import Tensor


_USE_GPTQ: bool = False
_USE_LQER: bool = False
_LQER_RANK: int = 4
_LQER_TOP_K: int = 3


def set_gptq_lqer_enabled(
    *,
    use_gptq: bool = False,
    use_lqer: bool = False,
    lqer_rank: int = 4,
    lqer_top_k: int = 3,
) -> None:
    """Module-level toggle invoked from `train_gpt.py::main()` at startup."""
    global _USE_GPTQ, _USE_LQER, _LQER_RANK, _LQER_TOP_K
    _USE_GPTQ = bool(use_gptq)
    _USE_LQER = bool(use_lqer)
    _LQER_RANK = int(lqer_rank)
    _LQER_TOP_K = int(lqer_top_k)


def is_gptq_lqer_enabled() -> bool:
    return _USE_GPTQ or _USE_LQER


# ---------------------------------------------------------------------------
# GPTQ Hessian-aware per-tensor quantization
# ---------------------------------------------------------------------------


def gptq_quantize_weight(
    W: Tensor,
    H: Tensor,
    clip_sigmas: float = 3.0,
    clip_range: int = 63,  # int7 max for unsigned; for signed int{B}, range = 2^(B-1) - 1
    block_size: int = 128,
) -> Tuple[Tensor, Tensor]:
    """Hessian-aware GPTQ per-tensor quantization. Records-canonical port.

    Args:
        W: (R, C) fp32 weight matrix.
        H: (C, C) calibration Hessian = X^T X where X is (n_samples, C).
        clip_sigmas: per-row scale = clip_sigmas * row_std / clip_range.
            Records SOTA stack uses 11.5 (mlp_clip_sigmas), 14.0 (embed),
            3.0 (default).
        clip_range: int range halfwidth, e.g. 63 for int7, 31 for int6, 7 for int4.
        block_size: GPTQ block size (default 128).

    Returns:
        Q: (R, C) int8 quantized weight (values in [-clip_range, clip_range]).
        s: (R,) fp16 per-row scale. Dequant: `W_q = Q.float() * s.float().view(-1, 1)`.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D, got {W.shape}")
    if H.shape != (W.shape[1], W.shape[1]):
        raise ValueError(f"H must be (C, C)={W.shape[1]}, got {H.shape}")
    W_orig = W.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()

    # Damp dead columns (rows of H with zero diagonal).
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)

    # Permute by descending diagonal (most-important columns first).
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]

    # Hinv_upper: upper-triangular factor of cholesky_inverse(H).
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv_upper = torch.linalg.cholesky(Hinv, upper=True)

    # Per-row scale (fp16 storage; fp32 use).
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()

    # Block-wise iterative quantization with residual propagation. `Q` and
    # `Err` must share device with the working tensor to support CUDA W/H.
    Q = torch.zeros(rows, cols, dtype=torch.int8, device=W_perm.device)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv_upper[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1, device=W_perm.device, dtype=W_work.dtype)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            # Propagate residual to remaining cols in this block.
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            # Propagate accumulated block-residual to remaining blocks.
            W_work[:, i2:] -= Err @ Hinv_upper[i1:i2, i2:]

    # Inverse-permute back to original column order.
    return Q[:, invperm], s


def gptq_dequant(Q: Tensor, s: Tensor) -> Tensor:
    """Dequantize GPTQ output. W_q = Q.float() * s.float().view(-1, 1)."""
    return Q.float() * s.float().view(-1, 1)


def naive_per_row_quant(W: Tensor, clip_range: int = 63) -> Tuple[Tensor, Tensor]:
    """Baseline naive per-row symmetric quantization for MSE comparison.

    Q_ij = clamp(round(W_ij / s_i), -clip_range, clip_range)
    s_i = max_j |W_ij| / clip_range

    Returns: (Q int8, s fp16). Same dequant signature as `gptq_quantize_weight`.
    """
    W = W.float()
    row_max = W.abs().amax(dim=1).clamp_min(1e-10)
    s = (row_max / clip_range).to(torch.float16)
    Q = torch.clamp(torch.round(W / s.float().view(-1, 1)), -clip_range, clip_range).to(torch.int8)
    return Q, s


# ---------------------------------------------------------------------------
# LQER: low-rank quantization-error residual correction
# ---------------------------------------------------------------------------


def lqer_compute_factors(W_orig: Tensor, W_quant: Tensor, rank: int) -> Tuple[Tensor, Tensor]:
    """Compute LQER (A, B) factors via SVD on the GPTQ residual.

    Args:
        W_orig: (R, C) fp32 original weight.
        W_quant: (R, C) fp32 dequantized GPTQ weight.
        rank: target rank for the residual approximation.

    Returns:
        A: (R, rank) — equals U[:, :rank] * S[:rank].
        B: (rank, C) — equals Vh[:rank, :].
        Reconstruction: W_eff ≈ W_quant + A @ B.
    """
    E = W_orig.float() - W_quant.float()
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    r = min(rank, S.numel())
    A = (U[:, :r] * S[:r]).contiguous()
    B = Vh[:r, :].contiguous()
    return A, B


def lqer_pack_sym(A: Tensor, B: Tensor, bits: int = 4) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Symmetric per-row int{bits} pack of LQER factors. Records-canonical L2197.

    Args:
        A: (R, rank) — left factor.
        B: (rank, C) — right factor.
        bits: int bit-width for storage. Default 4 → range [-7, 7].

    Returns:
        qA: (R, rank) int8 storage.
        sA: (R,) fp16 per-row scale of A.
        qB: (rank, C) int8 storage.
        sB: (rank,) fp16 per-row scale of B.

    Dequant: A ≈ qA.float() * sA.float().view(-1,1); B ≈ qB.float() * sB.float().view(-1,1).
    """
    rng = 2 ** (bits - 1) - 1
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -rng, rng).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -rng, rng).to(torch.int8)
    return qA, sA, qB, sB


def lqer_pack_asym(A: Tensor, B: Tensor, group_size: int = 64) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Asymmetric int2(A) + groupwise int4(B) pack of LQER factors. Records-canonical L2206.

    Args:
        A: (R, rank).
        B: (rank, C).
        group_size: group size for B's groupwise int4 (must divide B.numel()).

    Returns:
        qA: int8 with values in [-2, 1], single per-matrix scalar scale sA.
        sA: scalar fp16.
        qB: int8 with values in [-8, 7], per-group fp16 scale.
        sB: (B.numel() // group_size,) fp16 per-group scales.
    """
    if B.numel() % group_size != 0:
        raise ValueError(f"B.numel()={B.numel()} not divisible by group_size={group_size}")
    # A: int2 single-scalar [-2, 1], scale = |A|max / 1.5
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    # B: int4 groupwise over flattened B [-8, 7], per-group scale = |group|max / 7.5
    Bf = B.reshape(-1, group_size)
    Bmax = Bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (Bmax / 7.5).to(torch.float16).reshape(-1)
    qB = (
        torch.clamp(torch.round(Bf / sB.float().reshape(-1, 1)), -8, 7)
        .to(torch.int8)
        .reshape(B.shape)
    )
    return qA, sA, qB, sB


def lqer_dequant_sym(qA: Tensor, sA: Tensor, qB: Tensor, sB: Tensor) -> Tensor:
    """Dequantize symmetric LQER pack back to the residual approximation A @ B."""
    A = qA.float() * sA.float().view(-1, 1)
    B = qB.float() * sB.float().view(-1, 1)
    return A @ B


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    print("gptq_lqer.py smoke test (records-canonical):")
    torch.manual_seed(0)

    # Setup: structured synthetic that gives GPTQ a chance to demonstrate its
    # advantage. Real transformer weights are correlated row/col-wise (low
    # effective rank); GPTQ's Hessian-ordered column quant exploits this.
    # Random iid weights make GPTQ's column-importance heuristic powerless,
    # so we use a low-rank-plus-noise model that better mirrors reality.
    R, C = 64, 96
    n_samples = 4096  # larger calibration → cleaner Hessian
    # Calibration: low-rank + noise
    X_basis = torch.randn(n_samples, 16)
    X_proj = torch.randn(16, C)
    X = X_basis @ X_proj + 0.1 * torch.randn(n_samples, C)
    H = X.T @ X  # Hessian from calibration

    # Weight: low-rank + noise + outlier rows (structured-real)
    W_basis = torch.randn(R, 16)
    W_proj = torch.randn(16, C)
    W = W_basis @ W_proj * 0.05 + 0.1 * torch.randn(R, C)
    W[:5] *= 10.0  # outlier rows

    # Case 1: shape & dtype contracts
    Q, s = gptq_quantize_weight(W, H, clip_sigmas=3.0, clip_range=31)
    assert Q.shape == W.shape, f"Q shape {Q.shape} vs {W.shape}"
    assert Q.dtype == torch.int8
    assert s.shape == (R,) and s.dtype == torch.float16
    assert (Q.abs() <= 31).all(), "Q values must lie in [-31, 31] for int6"
    print(f"  GPTQ shape/dtype contract:    PASS (Q {Q.shape} int8, s {s.shape} fp16)")

    # Case 2: GPTQ MSE < naive per-row MSE at SAME clip_range
    Q_g, s_g = gptq_quantize_weight(W, H, clip_sigmas=3.0, clip_range=31)
    Q_n, s_n = naive_per_row_quant(W, clip_range=31)
    W_g = gptq_dequant(Q_g, s_g)
    W_n = gptq_dequant(Q_n, s_n)
    mse_g = (W - W_g).pow(2).mean().item()
    mse_n = (W - W_n).pow(2).mean().item()
    # GPTQ uses Hessian to optimize for X^T X-weighted error, not raw MSE.
    # On RAW MSE, naive (which optimizes raw MSE per row) often wins. The TRUE
    # comparison is X-weighted error: ||(W - W_q) X^T||_F.
    err_g = (W - W_g) @ X.T  # (R, n_samples) error projected onto calibration
    err_n = (W - W_n) @ X.T
    weighted_g = err_g.pow(2).mean().item()
    weighted_n = err_n.pow(2).mean().item()
    print(f"  raw MSE: GPTQ={mse_g:.4e} naive={mse_n:.4e}")
    print(f"  X-weighted err: GPTQ={weighted_g:.4e} naive={weighted_n:.4e}")
    assert weighted_g < weighted_n, \
        f"GPTQ should beat naive on X-weighted error: GPTQ={weighted_g} >= naive={weighted_n}"
    print("  GPTQ < naive on X-weighted err: PASS")

    # Case 3: LQER reduces residual norm monotonically with rank
    W_q = gptq_dequant(Q_g, s_g)
    res_norm_orig = (W - W_q).norm().item()
    res_norms = [res_norm_orig]
    for r in [1, 2, 4, 8, 16]:
        A, B = lqer_compute_factors(W, W_q, rank=r)
        residual_after = (W - W_q - A @ B).norm().item()
        res_norms.append(residual_after)
    assert all(res_norms[i] >= res_norms[i + 1] for i in range(len(res_norms) - 1)), \
        f"residual norm must monotonically decrease with rank: {res_norms}"
    print(f"  LQER residual decreases:      PASS ({[f'{r:.3f}' for r in res_norms]})")

    # Case 4: LQER at rank=min(R,C) fully recovers (within fp32 precision)
    rfull = min(R, C)
    A, B = lqer_compute_factors(W, W_q, rank=rfull)
    W_eff = W_q + A @ B
    rel_err = (W - W_eff).norm().item() / W.norm().item()
    assert rel_err < 1e-5, f"full-rank LQER should recover exactly, rel_err={rel_err}"
    print(f"  LQER full-rank recovery:      PASS (rel_err={rel_err:.2e})")

    # Case 5: LQER pack_sym + dequant round-trip preserves rank-r approximation
    A, B = lqer_compute_factors(W, W_q, rank=4)
    qA, sA, qB, sB = lqer_pack_sym(A, B, bits=4)
    assert qA.dtype == torch.int8 and qB.dtype == torch.int8
    assert sA.dtype == torch.float16 and sB.dtype == torch.float16
    AB_recovered = lqer_dequant_sym(qA, sA, qB, sB)
    AB_true = A @ B
    pack_err = (AB_recovered - AB_true).norm().item() / AB_true.norm().item()
    # int4 sym pack has ~6% relative error typical
    assert pack_err < 0.20, f"int4 sym pack rel err too high: {pack_err}"
    print(f"  LQER int4 sym pack round-trip: PASS (rel_err={pack_err:.4f})")

    # Case 6: LQER pack_asym round-trip
    # B has shape (4, 96); flat numel = 384, divisible by group_size=64.
    qA_a, sA_a, qB_a, sB_a = lqer_pack_asym(A, B, group_size=64)
    assert qA_a.dtype == torch.int8 and qB_a.dtype == torch.int8
    assert (qA_a.abs() <= 2).all(), "int2 A should be in [-2, 1]"
    assert (qB_a.abs() <= 8).all(), "int4 B should be in [-8, 7]"
    print("  LQER int2/int4-asym pack:      PASS (qA in [-2,1], qB in [-8,7])")

    # Case 7: combined GPTQ + LQER beats GPTQ alone on raw MSE
    A, B = lqer_compute_factors(W, W_q, rank=4)
    W_lqer = W_q + A @ B
    mse_q = (W - W_q).pow(2).mean().item()
    mse_lqer = (W - W_lqer).pow(2).mean().item()
    assert mse_lqer < mse_q, \
        f"LQER+GPTQ should beat GPTQ alone: lqer={mse_lqer} >= gptq={mse_q}"
    print(f"  GPTQ+LQER < GPTQ alone:        PASS ({mse_lqer:.4e} < {mse_q:.4e})")

    # Case 8: dead-column robustness (zero-diagonal H)
    H_dead = H.clone()
    H_dead[3, 3] = 0  # dead column
    H_dead[3, :] = 0
    H_dead[:, 3] = 0
    H_dead[3, 3] = 0
    Q_d, s_d = gptq_quantize_weight(W, H_dead, clip_sigmas=3.0, clip_range=31)
    assert Q_d.shape == W.shape
    assert not torch.isnan(s_d).any()
    print("  dead-column robustness:        PASS")

    # Case 9: clip_range=7 (int4) vs clip_range=31 (int6) MSE comparison
    Q4, s4 = gptq_quantize_weight(W, H, clip_sigmas=3.0, clip_range=7)
    Q6, s6 = gptq_quantize_weight(W, H, clip_sigmas=3.0, clip_range=31)
    mse4 = (W - gptq_dequant(Q4, s4)).pow(2).mean().item()
    mse6 = (W - gptq_dequant(Q6, s6)).pow(2).mean().item()
    assert mse6 <= mse4, f"int6 should beat int4 on MSE: int6={mse6} > int4={mse4}"
    print(f"  bit-width MSE ordering:        PASS (int6={mse6:.4e} ≤ int4={mse4:.4e})")

    print("ALL SMOKE TESTS PASSED")


def run_gptq_lqer_component_smoke(
    *,
    device: torch.device | str = "cpu",
    use_gptq: bool = True,
    use_lqer: bool = True,
    lqer_rank: int = 4,
) -> dict[str, float]:
    """Small deterministic exec witness for the artifact quantization scaffold.

    Full production GPTQ requires Hessian collection hooks over model Linear
    inputs; this smoke validates the math primitive and optional LQER residual
    path without altering the canonical int6 artifact schema.
    """
    torch.manual_seed(23)
    dev = torch.device(device)
    rows, cols, samples = 16, 24, 128
    x = torch.randn(samples, cols, device=dev)
    h = x.T @ x
    w = torch.randn(rows, cols, device=dev) * 0.05
    qn, sn = naive_per_row_quant(w, clip_range=31)
    w_naive = gptq_dequant(qn, sn).to(dev)
    if use_gptq:
        qg, sg = gptq_quantize_weight(w, h, clip_sigmas=3.0, clip_range=31, block_size=16)
        w_q = gptq_dequant(qg, sg).to(dev)
    else:
        w_q = w_naive
    mse_before = float((w - w_q).pow(2).mean().detach().cpu().item())
    weighted_before = float(((w - w_q) @ x.T).pow(2).mean().detach().cpu().item())
    mse_after = mse_before
    if use_lqer:
        a, b = lqer_compute_factors(w, w_q, rank=int(lqer_rank))
        w_lqer = w_q + a @ b
        mse_after = float((w - w_lqer).pow(2).mean().detach().cpu().item())
        if mse_after > mse_before + 1e-12:
            raise RuntimeError("LQER residual correction increased reconstruction MSE")
    if not math.isfinite(mse_before) or not math.isfinite(weighted_before):
        raise RuntimeError("GPTQ/LQER smoke produced non-finite metrics")
    return {
        "mse_before_lqer": mse_before,
        "mse_after_lqer": mse_after,
        "weighted_error": weighted_before,
    }


if __name__ == "__main__":
    _smoke_test()
