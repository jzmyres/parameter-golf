"""Progressive correctness tests for sparse-aware MoE dispatch (iter 117b-3 dev).

Built up incrementally per "start simple, work progressively, comprehensive
and rigorous testing" directive (2026-04-30).

Phase ladder:
  A.0  Verify existing entmax_1p5 in train_gpt.py matches deep-spin reference
       (sanity check — grounds all later sparsity work)
  A.1  Pure-PyTorch capacity-padded sparse MoE dispatch — `gather → expert → scatter`
       Test numerical equivalence vs dense reference (C=∞: bit-identical;
       C=1.5: bf16-floor approximation under entmax sparsity)
  A.2  Gradient correctness for the sparse dispatch (autograd, then gradcheck)
  A.3  Edge cases: uniform routing, single-spike, empty support, capacity overflow
  A.4  Wrap in `torch.library.custom_op`, verify torch.compile traces it
  B.0  Triton entmax kernel (already written in test_entmax_triton.py — re-test here under custom_op)
  B.1  Triton fused entmax + grouped-GEMM
  C    Throughput micro-benchmark vs dense; smoke test full training

Each phase has focused tests with clear pass/fail criteria. Run from repo root:
  ~/.conda/envs/opg/bin/python experiments/test_sparse_dispatch.py
"""
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
# Phase A.0: verify existing entmax_1p5 matches deep-spin reference
# ===========================================================================


def _entmax_1p5_pyref(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pure-PyTorch reference matching train_gpt.py::entmax_1p5.

    Algorithm: shift z so max=0, sort descending, find largest k where
    candidate τ_k = (S_k - sqrt(S_k² - k(S2_k - 4)))/k satisfies τ_k <
    z_sorted_k. Output w_i = max(0.5(z_i - τ), 0)².
    """
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


def _entmax_1p5_deepspin(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reference impl matching deep-spin/entmax Entmax15Function.forward exactly.

    Includes the X/2 factoring (instead of our 0.5*(z-tau) factoring); the
    two are mathematically equivalent (their tau' = our tau / 2). This is a
    cross-check against the upstream library's published algorithm.
    """
    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val
    X = X / 2  # the canonical deep-spin scaling

    # _entmax_threshold_and_support
    Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    K = X.shape[dim]
    rho = torch.arange(1, K + 1, device=X.device, dtype=X.dtype)
    rho_shape = [1] * X.ndim
    rho_shape[dim] = K
    rho = rho.view(*rho_shape)
    mean = Xsrt.cumsum(dim=dim) / rho
    mean_sq = (Xsrt * Xsrt).cumsum(dim=dim) / rho
    ss = rho * (mean_sq - mean * mean)
    delta = (1 - ss) / rho
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= Xsrt).sum(dim=dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    Y = torch.clamp(X - tau_star, min=0) ** 2
    return Y


def test_phase_A0_entmax_matches_reference():
    """Phase A.0: our entmax_1p5 should match deep-spin reference within fp32 floor."""
    print("\n" + "=" * 60)
    print("Phase A.0: entmax_1p5 matches deep-spin reference")
    print("=" * 60)

    torch.manual_seed(0)
    failures = 0
    cases = [
        ("uniform-random", lambda B, E: torch.randn(B, E)),
        ("scaled-random",  lambda B, E: torch.randn(B, E) * 4.0),
        ("near-uniform",   lambda B, E: torch.randn(B, E) * 0.01),
        ("one-spike",      lambda B, E: torch.zeros(B, E).scatter_(1, torch.randint(0, E, (B, 1)), 5.0)),
    ]
    for name, gen in cases:
        for B, E in [(32, 16), (256, 16)]:
            z = gen(B, E)
            w_ours = _entmax_1p5_pyref(z, dim=-1)
            w_ref  = _entmax_1p5_deepspin(z, dim=-1)
            err_max = (w_ours - w_ref).abs().max().item()
            sum_err = (w_ours.sum(-1) - 1.0).abs().max().item()
            sum_err_ref = (w_ref.sum(-1) - 1.0).abs().max().item()
            status = "PASS" if err_max < 1e-5 else "FAIL"
            if err_max >= 1e-5:
                failures += 1
            print(f"  {name:<18} B={B:>4} E={E:>3}  "
                  f"max_diff={err_max:.2e}  Σw_ours-1={sum_err:.2e}  "
                  f"Σw_ref-1={sum_err_ref:.2e}  {status}")
    return failures


# ===========================================================================
# Phase A.1: pure-PyTorch capacity-padded sparse MoE dispatch
# ===========================================================================


def dense_moe_dispatch(
    x: torch.Tensor,           # (T, D)
    w: torch.Tensor,           # (T, E)  routing weights, may be sparse
    expert_W: torch.Tensor,    # (E, D, R)  per-expert linear (rank-R bottleneck)
    expert_V: torch.Tensor,    # (E, R, D)  per-expert linear back to D
) -> torch.Tensor:
    """Dense reference: every expert processes every token; weighted sum.

    out[t] = Σ_e w[t,e] · (expert_V[e] @ expert_W[e].T @ x[t])

    With routing weights w (shape (T, E)) — softmax dense or entmax sparse,
    same interface. This is the OPERATOR-LEVEL semantics that any sparse
    dispatch must match within bf16 floor.
    """
    # Compute all expert outputs: (E, T, D) → reshape for matmul
    # x_per: (E, T, D); each expert sees the same x but applies its own W, V.
    T, D = x.shape
    E, _, R = expert_W.shape
    # (E, T, D) @ (E, D, R) → (E, T, R)
    x_b = x.unsqueeze(0).expand(E, T, D)
    h = torch.einsum("etd,edr->etr", x_b, expert_W)
    out_per_e = torch.einsum("etr,erd->etd", h, expert_V)  # (E, T, D)
    # Weight + sum: (E, T, D) * (E, T, 1) → (T, D)
    weighted = out_per_e * w.t().unsqueeze(-1)  # (E, T, D) * (E, T, 1)
    return weighted.sum(dim=0)


def sparse_moe_dispatch_capacity(
    x: torch.Tensor,           # (T, D)
    w: torch.Tensor,           # (T, E)
    expert_W: torch.Tensor,    # (E, D, R)
    expert_V: torch.Tensor,    # (E, R, D)
    capacity_factor: float = 1.5,
) -> torch.Tensor:
    """Capacity-padded sparse dispatch: each expert processes its top-K tokens
    by routing weight; output scattered back.

    K = ceil(capacity_factor * T / E) is a static compile-time-known dimension
    given T and E. Tokens not in expert_e's top-K contribute zero from that
    expert (capacity overflow). Tokens with weight=0 are guaranteed to fall
    outside any expert's top-K when sparsity > 1 - 1/capacity_factor.

    Returns the same operator-level output as dense_moe_dispatch within bf16
    floor, modulo capacity-overflow truncation.
    """
    T, D = x.shape
    E, _, R = expert_W.shape
    K = int(math.ceil(capacity_factor * T / E))

    output = torch.zeros_like(x)
    for e in range(E):
        w_e = w[:, e]                            # (T,)
        topk_w, topk_idx = w_e.topk(K, dim=0)    # (K,), (K,)
        x_e = x.index_select(0, topk_idx)        # (K, D)
        # expert_W[e]: (D, R); expert_V[e]: (R, D)
        h_e = x_e @ expert_W[e]                  # (K, R)
        y_e = h_e @ expert_V[e]                  # (K, D)
        y_e = y_e * topk_w.unsqueeze(-1)         # weighted
        output.index_add_(0, topk_idx, y_e)
    return output


def test_phase_A1_sparse_dispatch_equivalence():
    """Phase A.1: capacity-padded dispatch correctness + capacity-error trade-off.

    Capacity math: with E experts, T tokens, sparsity s (fraction of zero
    entries in routing matrix), each expert serves an average of (1-s)·T
    nonzero tokens. Capacity-padded dispatch with K = ceil(C·T/E) tokens per
    expert is BIT-EXACT iff K ≥ max-per-expert-utilization, i.e., when

        C ≥ (1-s) · E + headroom-for-imbalance

    For uniform per-expert utilization the bound is C* = (1-s)·E. Lower C
    truncates nonzero weights; the error scales with the truncated mass.
    """
    print("\n" + "=" * 60)
    print("Phase A.1: capacity-padded sparse dispatch")
    print("=" * 60)

    torch.manual_seed(1)
    failures = 0
    T, D, E, R = 256, 32, 16, 8
    x = torch.randn(T, D)
    expert_W = torch.randn(E, D, R) * 0.1
    expert_V = torch.randn(E, R, D) * 0.1
    logits = torch.randn(T, E) * 2.0

    # --- Test 1: CORRECTNESS — sufficient capacity gives bit-identical output.
    print("\n  Correctness (capacity ≥ utilization → bit-identical):")
    # Softmax: all entries nonzero, each expert serves ALL T tokens. C=E gives
    # K=T → covers everything. Expected: max_abs=0 (modulo float associativity).
    w_softmax = logits.softmax(dim=-1)
    out_dense  = dense_moe_dispatch(x, w_softmax, expert_W, expert_V)
    out_sparse = sparse_moe_dispatch_capacity(x, w_softmax, expert_W, expert_V, capacity_factor=E)
    err = (out_dense - out_sparse).abs().max().item()
    rel = err / (out_dense.abs().max().item() + 1e-12)
    status = "PASS" if rel < 1e-5 else "FAIL"
    failures += int(rel >= 1e-5)
    print(f"    softmax + C=E (K=T):       max_abs={err:.2e}  rel={rel:.2e}  {status}")

    # Entmax: ~80% sparse at random init. Each expert serves ≈ (1-s)·T = 51
    # tokens. Need C ≥ (1-s)·E + headroom = 0.20·16 + headroom = 3.2 + ε.
    # With C = 8 (≈2.5× the bound), all per-expert utilizations are covered.
    w_entmax = _entmax_1p5_pyref(logits, dim=-1)
    sparsity = (w_entmax == 0).float().mean().item()
    util_max = (w_entmax > 0).sum(dim=0).max().item()  # max nonzero tokens for any expert
    out_dense  = dense_moe_dispatch(x, w_entmax, expert_W, expert_V)
    out_sparse = sparse_moe_dispatch_capacity(x, w_entmax, expert_W, expert_V, capacity_factor=8.0)
    err = (out_dense - out_sparse).abs().max().item()
    rel = err / (out_dense.abs().max().item() + 1e-12)
    K = int(math.ceil(8.0 * T / E))
    status = "PASS" if rel < 5e-5 else "FAIL"
    failures += int(rel >= 5e-5)
    print(f"    entmax  + C=8 (K={K}):    max_abs={err:.2e}  rel={rel:.2e}  "
          f"sparsity={sparsity:.2f} util_max={util_max} (K must ≥ util_max)  {status}")

    # --- Test 2: APPROXIMATION trade-off — error monotone in (1/C).
    # As C decreases below (1-s)·E, more nonzero weights get truncated; error rises.
    print("\n  Capacity sweep (entmax, sparsity={:.2f}, util_max={}):".format(sparsity, util_max))
    print(f"    {'C':>6}  {'K':>5}  {'rel':>10}  monotonicity")
    prev_rel = float("inf")
    rels = []
    for C in [E, 8.0, 5.0, 3.5, 2.0, 1.5, 1.0]:
        out_sparse = sparse_moe_dispatch_capacity(x, w_entmax, expert_W, expert_V, capacity_factor=C)
        err = (out_dense - out_sparse).abs().max().item()
        rel = err / (out_dense.abs().max().item() + 1e-12)
        K = int(math.ceil(C * T / E))
        rels.append((C, K, rel))
        print(f"    {C:>6.1f}  {K:>5}  {rel:>10.2e}")
    # Verify monotonicity: as C decreases, rel error should not decrease.
    monotone = all(rels[i][2] <= rels[i+1][2] + 1e-10 for i in range(len(rels)-1))
    status = "PASS" if monotone else "FAIL"
    failures += int(not monotone)
    print(f"    monotone error in 1/C: {status}")

    # --- Test 3: spike test — extreme sparsity gives small error at small C.
    # When most tokens use only K_active << E experts, capacity = K_active is enough.
    print("\n  Extreme-sparsity stress (logits scaled 5×):")
    logits_sharp = logits * 5.0
    w_sharp = _entmax_1p5_pyref(logits_sharp, dim=-1)
    sparsity_sharp = (w_sharp == 0).float().mean().item()
    util_max_sharp = (w_sharp > 0).sum(dim=0).max().item()
    out_dense_sharp = dense_moe_dispatch(x, w_sharp, expert_W, expert_V)
    for C in [3.0, 2.0, 1.5]:
        out_sparse = sparse_moe_dispatch_capacity(x, w_sharp, expert_W, expert_V, capacity_factor=C)
        err = (out_dense_sharp - out_sparse).abs().max().item()
        rel = err / (out_dense_sharp.abs().max().item() + 1e-12)
        K = int(math.ceil(C * T / E))
        sufficient = K >= util_max_sharp
        target_rel = 1e-5 if sufficient else 5e-2
        status = "PASS" if rel < target_rel else "FAIL"
        failures += int(rel >= target_rel)
        suffix = " (capacity sufficient)" if sufficient else " (capacity overflow)"
        print(f"    C={C:>3.1f} K={K:>3}  rel={rel:.2e}  sparsity={sparsity_sharp:.2f} "
              f"util_max={util_max_sharp}{suffix}  {status}")

    return failures


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    failures = 0

    # Phase A.0 runs on CPU.
    failures += test_phase_A0_entmax_matches_reference()

    # Phase A.1 runs on CPU; later phases will need GPU.
    failures += test_phase_A1_sparse_dispatch_equivalence()

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALL PHASE A.0 + A.1 TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s)")
        sys.exit(1)
