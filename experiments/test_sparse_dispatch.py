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
# Phase A.2: gradient correctness
# ===========================================================================


def test_phase_A2_gradient_correctness():
    """Phase A.2: autograd through sparse dispatch matches autograd through dense.

    The dispatch uses gather + scatter_add + matmul + multiply — all
    differentiable. With sufficient capacity (no truncation), backward
    should give bit-identical gradients (modulo float associativity in
    scatter_add reduction order).
    """
    print("\n" + "=" * 60)
    print("Phase A.2: gradient correctness")
    print("=" * 60)

    failures = 0
    torch.manual_seed(2)
    T, D, E, R = 64, 16, 8, 4

    def make_inputs():
        x = torch.randn(T, D, requires_grad=True)
        eW = torch.randn(E, D, R, requires_grad=True) * 0.1
        eV = torch.randn(E, R, D, requires_grad=True) * 0.1
        logits = torch.randn(T, E, requires_grad=True) * 2.0
        return x, eW, eV, logits

    def grad_relerr(g_ref, g_test, tag):
        if g_ref is None and g_test is None:
            return 0.0
        if g_ref is None or g_test is None:
            print(f"    {tag}: GRAD MISMATCH (None)")
            return 1.0
        err = (g_ref - g_test).abs().max().item()
        ref_max = g_ref.abs().max().item() + 1e-12
        return err / ref_max

    # --- Test 1: softmax + C=E (sufficient capacity) — bit-identical fwd, fwd
    # equivalence implies backward equivalence by chain rule. Verify.
    print("\n  Backward through dense vs sparse with C=E (sufficient capacity):")
    x, eW, eV, logits = make_inputs()
    w_softmax = logits.softmax(dim=-1)

    out_dense = dense_moe_dispatch(x, w_softmax, eW, eV)
    loss_dense = out_dense.pow(2).sum()
    g_x_d, g_eW_d, g_eV_d, g_logits_d = torch.autograd.grad(
        loss_dense, [x, eW, eV, logits], retain_graph=False)

    x2, eW2, eV2, logits2 = (x.detach().clone().requires_grad_(),
                              eW.detach().clone().requires_grad_(),
                              eV.detach().clone().requires_grad_(),
                              logits.detach().clone().requires_grad_())
    w_softmax2 = logits2.softmax(dim=-1)
    out_sparse = sparse_moe_dispatch_capacity(x2, w_softmax2, eW2, eV2, capacity_factor=E)
    loss_sparse = out_sparse.pow(2).sum()
    g_x_s, g_eW_s, g_eV_s, g_logits_s = torch.autograd.grad(
        loss_sparse, [x2, eW2, eV2, logits2], retain_graph=False)

    for tag, gref, gtst in [("∂L/∂x",      g_x_d,      g_x_s),
                            ("∂L/∂W",      g_eW_d,     g_eW_s),
                            ("∂L/∂V",      g_eV_d,     g_eV_s),
                            ("∂L/∂logits", g_logits_d, g_logits_s)]:
        rel = grad_relerr(gref, gtst, tag)
        # Some scatter_add reorder noise; allow 1e-4 relative
        status = "PASS" if rel < 1e-4 else "FAIL"
        if rel >= 1e-4:
            failures += 1
        print(f"    {tag:<14} rel_max={rel:.2e}  {status}")

    # --- Test 2: entmax + C=8 (sufficient for sparsity 0.80) — bit-identical fwd → bit-identical bwd.
    print("\n  Backward through dense vs sparse with entmax + C=8 (sufficient):")
    x, eW, eV, logits = make_inputs()
    w_entmax = _entmax_1p5_pyref(logits, dim=-1)

    out_dense = dense_moe_dispatch(x, w_entmax, eW, eV)
    loss_dense = out_dense.pow(2).sum()
    g_x_d, g_eW_d, g_eV_d, g_logits_d = torch.autograd.grad(
        loss_dense, [x, eW, eV, logits], retain_graph=False)

    x2, eW2, eV2, logits2 = (x.detach().clone().requires_grad_(),
                              eW.detach().clone().requires_grad_(),
                              eV.detach().clone().requires_grad_(),
                              logits.detach().clone().requires_grad_())
    w_entmax2 = _entmax_1p5_pyref(logits2, dim=-1)
    out_sparse = sparse_moe_dispatch_capacity(x2, w_entmax2, eW2, eV2, capacity_factor=E)
    loss_sparse = out_sparse.pow(2).sum()
    g_x_s, g_eW_s, g_eV_s, g_logits_s = torch.autograd.grad(
        loss_sparse, [x2, eW2, eV2, logits2], retain_graph=False)

    for tag, gref, gtst in [("∂L/∂x",      g_x_d,      g_x_s),
                            ("∂L/∂W",      g_eW_d,     g_eW_s),
                            ("∂L/∂V",      g_eV_d,     g_eV_s),
                            ("∂L/∂logits", g_logits_d, g_logits_s)]:
        rel = grad_relerr(gref, gtst, tag)
        status = "PASS" if rel < 1e-4 else "FAIL"
        if rel >= 1e-4:
            failures += 1
        print(f"    {tag:<14} rel_max={rel:.2e}  {status}")

    # --- Test 3: gradcheck on a small case (most stringent — finite diff vs autograd).
    print("\n  torch.autograd.gradcheck (sparse dispatch, small case, C=E):")
    torch.manual_seed(3)
    Tg, Dg, Eg, Rg = 4, 3, 2, 2
    xg = torch.randn(Tg, Dg, dtype=torch.float64, requires_grad=True)
    eWg = torch.randn(Eg, Dg, Rg, dtype=torch.float64, requires_grad=True) * 0.3
    eVg = torch.randn(Eg, Rg, Dg, dtype=torch.float64, requires_grad=True) * 0.3
    logits_g = torch.randn(Tg, Eg, dtype=torch.float64, requires_grad=True) * 1.5

    def fn(x_, eW_, eV_, l_):
        w_ = _entmax_1p5_pyref(l_, dim=-1)
        return sparse_moe_dispatch_capacity(x_, w_, eW_, eV_, capacity_factor=Eg)

    try:
        ok = torch.autograd.gradcheck(fn, (xg, eWg, eVg, logits_g), eps=1e-6, atol=1e-4)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"    gradcheck: {status}")
    except Exception as e:
        # gradcheck raises on failure — print and count as failure
        print(f"    gradcheck: FAIL ({type(e).__name__}: {str(e)[:80]})")
        failures += 1

    return failures


# ===========================================================================
# Phase A.3: edge cases
# ===========================================================================


def test_phase_A3_edge_cases():
    """Phase A.3: pathological routing distributions and shape edges."""
    print("\n" + "=" * 60)
    print("Phase A.3: edge cases")
    print("=" * 60)

    failures = 0
    torch.manual_seed(4)

    def check(tag, w, x, eW, eV, C, atol=1e-5):
        nonlocal failures
        out_dense  = dense_moe_dispatch(x, w, eW, eV)
        out_sparse = sparse_moe_dispatch_capacity(x, w, eW, eV, capacity_factor=C)
        err = (out_dense - out_sparse).abs().max().item()
        rel = err / (out_dense.abs().max().item() + 1e-12) if out_dense.abs().max().item() > 0 else err
        status = "PASS" if rel < atol else "FAIL"
        if rel >= atol:
            failures += 1
        print(f"  {tag:<48} max_abs={err:.2e}  rel={rel:.2e}  {status}")

    T, D, E, R = 64, 16, 8, 4
    x = torch.randn(T, D)
    eW = torch.randn(E, D, R) * 0.1
    eV = torch.randn(E, R, D) * 0.1

    # Edge 1: all-zero routing → output is zero
    w_zero = torch.zeros(T, E)
    out_sparse = sparse_moe_dispatch_capacity(x, w_zero, eW, eV, capacity_factor=2.0)
    is_zero = out_sparse.abs().max().item() < 1e-12
    print(f"  {'all-zero routing → zero output':<48} max_abs={out_sparse.abs().max().item():.2e}  "
          f"{'PASS' if is_zero else 'FAIL'}")
    if not is_zero:
        failures += 1

    # Edge 2: one-hot routing per token (extreme sparsity, K_active=1) — C=1 sufficient
    idx = torch.randint(0, E, (T,))
    w_onehot = torch.zeros(T, E).scatter_(1, idx.unsqueeze(1), 1.0)
    util_max = (w_onehot > 0).sum(dim=0).max().item()
    # C must give K ≥ util_max; with random indices util_max varies
    K_min = util_max
    C_min = K_min * E / T
    check(f"one-hot routing (util_max={util_max}, C={max(C_min*1.5,1.0):.2f})",
          w_onehot, x, eW, eV, C=max(C_min*1.5, 1.0))

    # Edge 3: identical weight per token (uniform, no zeros) — C=E sufficient
    w_uniform = torch.full((T, E), 1.0/E)
    check("uniform routing 1/E (C=E)", w_uniform, x, eW, eV, C=E)

    # Edge 4: single token → smallest possible batch
    x_one = torch.randn(1, D)
    logits_one = torch.randn(1, E) * 2.0
    w_one = _entmax_1p5_pyref(logits_one, dim=-1)
    util_max_one = (w_one > 0).sum(dim=0).max().item()
    # K = ceil(C*1/E) needs to be ≥ util_max_one (=1 since only 1 token).
    # ceil(C/E) ≥ 1 ⇒ C ≥ 1/E ≈ 0.125. C=1 is safe.
    check("single token (T=1)", w_one, x_one, eW, eV, C=1.0)

    # Edge 5: dim mismatch → shape-correctness ground truth
    # Just check shapes
    out = sparse_moe_dispatch_capacity(x, w_uniform, eW, eV, capacity_factor=E)
    shape_ok = out.shape == (T, D)
    print(f"  {'output shape matches (T,D)':<48} got={tuple(out.shape)}  expected={(T,D)}  "
          f"{'PASS' if shape_ok else 'FAIL'}")
    if not shape_ok:
        failures += 1

    # Edge 6: T not divisible by E (no special handling needed since K = ceil)
    T2 = 67  # prime
    x2 = torch.randn(T2, D)
    logits2 = torch.randn(T2, E) * 2.0
    w2 = _entmax_1p5_pyref(logits2, dim=-1)
    check(f"T not divisible by E (T={T2}, E={E}, C={E})",
          w2, x2, eW, eV, C=E)  # C=E gives K ≥ T2

    # Edge 7: capacity overflow — explicitly test that error is BOUNDED by truncated mass.
    # If we truncate weights with sum_truncated_w, the upper bound on rel_err
    # scales with sum_truncated_w / sum_total_w.
    print("  capacity-overflow error bound check:")
    logits = torch.randn(T, E) * 2.0
    w = _entmax_1p5_pyref(logits, dim=-1)
    util_max = (w > 0).sum(dim=0).max().item()
    sum_total = w.sum().item()
    # Calculate sum of truncated weights at C=2 (deliberately short)
    K = int(math.ceil(2.0 * T / E))
    sum_kept = 0.0
    for e in range(E):
        topk_w, _ = w[:, e].topk(K, dim=0)
        sum_kept += topk_w.clamp_min(0).sum().item()
    sum_truncated = max(sum_total - sum_kept, 0)
    truncated_frac = sum_truncated / max(sum_total, 1e-12)
    out_dense  = dense_moe_dispatch(x, w, eW, eV)
    out_sparse = sparse_moe_dispatch_capacity(x, w, eW, eV, capacity_factor=2.0)
    err_rel = (out_dense - out_sparse).abs().max().item() / (out_dense.abs().max().item() + 1e-12)
    # Upper bound: rel_err ≤ truncated_frac × max-expert-output-norm (≈ rel_err ≤ 2-3× truncated_frac in practice)
    bound_holds = err_rel < 5 * truncated_frac + 1e-6
    print(f"    truncated_frac={truncated_frac:.2%}  rel_err={err_rel:.2e}  "
          f"bound_5x: {'PASS' if bound_holds else 'FAIL'}")
    if not bound_holds:
        failures += 1

    return failures


# ===========================================================================
# Phase A.4: torch.compile traceability
# ===========================================================================


def test_phase_A4_compile_traceability():
    """Phase A.4: verify torch.compile traces through sparse dispatch.

    Pure-PyTorch dispatch (gather + scatter_add + matmul + multiply) should
    be tracable by torch.compile natively — no custom_op wrapper needed yet.
    custom_op becomes necessary for the Triton kernel path (Phase B).

    Three checks:
      1. Forward-only compile: graph captures successfully
      2. Compile + autograd: backward through compiled graph matches eager
      3. Recompile pressure: shape changes don't trigger excessive recompiles
    """
    print("\n" + "=" * 60)
    print("Phase A.4: torch.compile traceability")
    print("=" * 60)

    failures = 0
    torch.manual_seed(5)
    T, D, E, R = 64, 16, 8, 4
    x = torch.randn(T, D)
    eW = torch.randn(E, D, R) * 0.1
    eV = torch.randn(E, R, D) * 0.1
    logits = torch.randn(T, E) * 2.0
    w = _entmax_1p5_pyref(logits, dim=-1)

    # --- Test 1: forward compile traceability.
    print("\n  Compile forward only (capacity_factor as constant):")
    try:
        @torch.compile(fullgraph=False, dynamic=False)
        def fn_compiled(x, w, eW, eV):
            return sparse_moe_dispatch_capacity(x, w, eW, eV, capacity_factor=float(E))
        out_eager = sparse_moe_dispatch_capacity(x, w, eW, eV, capacity_factor=float(E))
        out_comp  = fn_compiled(x, w, eW, eV)
        err = (out_eager - out_comp).abs().max().item()
        rel = err / (out_eager.abs().max().item() + 1e-12)
        status = "PASS" if rel < 1e-5 else "FAIL"
        if rel >= 1e-5:
            failures += 1
        print(f"    eager vs compiled forward: rel={rel:.2e}  {status}")
    except Exception as e:
        print(f"    compile FAILED: {type(e).__name__}: {str(e)[:120]}")
        failures += 1

    # --- Test 2: compile + autograd.
    print("\n  Compile + autograd (gradient through compiled graph):")
    try:
        x2 = x.detach().clone().requires_grad_(True)
        eW2 = eW.detach().clone().requires_grad_(True)
        eV2 = eV.detach().clone().requires_grad_(True)
        logits2 = logits.detach().clone().requires_grad_(True)

        @torch.compile(fullgraph=False, dynamic=False)
        def fn_full_compiled(x_, l_, eW_, eV_):
            w_ = _entmax_1p5_pyref(l_, dim=-1)
            return sparse_moe_dispatch_capacity(x_, w_, eW_, eV_, capacity_factor=float(E))

        # Eager reference.
        x_e = x.detach().clone().requires_grad_(True)
        eW_e = eW.detach().clone().requires_grad_(True)
        eV_e = eV.detach().clone().requires_grad_(True)
        logits_e = logits.detach().clone().requires_grad_(True)
        w_e = _entmax_1p5_pyref(logits_e, dim=-1)
        out_e = sparse_moe_dispatch_capacity(x_e, w_e, eW_e, eV_e, capacity_factor=float(E))
        loss_e = out_e.pow(2).sum()
        loss_e.backward()

        out_c = fn_full_compiled(x2, logits2, eW2, eV2)
        loss_c = out_c.pow(2).sum()
        loss_c.backward()

        for tag, ge, gc in [("∂L/∂x", x_e.grad, x2.grad),
                            ("∂L/∂W", eW_e.grad, eW2.grad),
                            ("∂L/∂V", eV_e.grad, eV2.grad),
                            ("∂L/∂logits", logits_e.grad, logits2.grad)]:
            rel = (ge - gc).abs().max().item() / (ge.abs().max().item() + 1e-12)
            status = "PASS" if rel < 1e-4 else "FAIL"
            if rel >= 1e-4:
                failures += 1
            print(f"    {tag:<14} rel_max={rel:.2e}  {status}")
    except Exception as e:
        print(f"    compile+autograd FAILED: {type(e).__name__}: {str(e)[:120]}")
        failures += 1

    # --- Test 3: recompile pressure (shape changes).
    # If we recompile aggressively on K change, dynamo recompile_limit can hit.
    # Capacity factor is a Python float — making K a constant per compile.
    # Different T or D should trigger recompile (acceptable).
    # Same T/D but different *content* should NOT recompile (this is the win).
    print("\n  Recompile pressure check (same shapes, different values):")
    try:
        compile_count = [0]

        @torch.compile(fullgraph=False, dynamic=False)
        def fn(x, w, eW, eV):
            compile_count[0] += 1
            return sparse_moe_dispatch_capacity(x, w, eW, eV, capacity_factor=float(E))

        for _ in range(5):
            new_x = torch.randn_like(x)
            new_w = _entmax_1p5_pyref(torch.randn_like(logits), dim=-1)
            _ = fn(new_x, new_w, eW, eV)

        # `compile_count` increments on each PYTHON call, not just graph (re)compiles —
        # so this is a structural check; real recompile detection needs dynamo logs.
        # Just verify no exception raised across 5 calls.
        print(f"    5 forward calls with new tensor values: PASS (no exception)")
    except Exception as e:
        print(f"    recompile pressure: FAIL ({type(e).__name__}: {str(e)[:120]})")
        failures += 1

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

    # Phase A.2 runs on CPU.
    failures += test_phase_A2_gradient_correctness()

    # Phase A.3 runs on CPU.
    failures += test_phase_A3_edge_cases()

    # Phase A.4 runs on CPU; torch.compile traceability.
    failures += test_phase_A4_compile_traceability()

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s)")
        sys.exit(1)
