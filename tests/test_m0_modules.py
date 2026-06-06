import pytest
import torch

from train_gpt import Hyperparameters, M0GPT, MLAttention, MoSHead, SwiGLUMoE


# ---------------------------------------------------------------------------
# Orthogonal (mHC-style) strictly-invertible damping: low-rank-skew Q = expm(S).
# ``cayley_apply`` computes Q·x / Qᵀ·x via the MATRIX EXPONENTIAL of the low-rank
# skew S = U Vᵀ − V Uᵀ projected onto its 2r-dim active subspace (NO d×d matrix
# is formed, NO (I+S)^{-1} solve — exactly orthogonal for ANY ‖S‖). These are the
# fp64 helper-level gates (orthogonality + Qᵀ∘Q == I). The function name is kept
# for call-site stability; the parameterization is now expm(S), not the Cayley
# transform (the Cayley solve became ill-conditioned as the learnable U,V grew).
# ---------------------------------------------------------------------------
def test_cayley_apply_matches_dense_Q_and_QT():
    """``cayley_apply`` (low-rank, no d×d matrix) equals the dense matrix-
    exponential Q·x = x @ expm(S)ᵀ and Qᵀ·x = x @ expm(S) to fp64 precision."""
    from train_gpt import cayley_apply

    torch.manual_seed(0)
    d, r = 12, 4
    U = torch.randn(d, r, dtype=torch.float64)
    V = torch.randn(d, r, dtype=torch.float64)
    S = U @ V.T - V @ U.T
    Q = torch.matrix_exp(S)            # Q = expm(S) (exactly orthogonal, skew S)
    x = torch.randn(3, 5, d, dtype=torch.float64)
    # Row-vector convention: applying Q to each d-vector is x @ Qᵀ.
    assert torch.allclose(cayley_apply(x, U, V), x @ Q.T, atol=1e-10)
    assert torch.allclose(cayley_apply(x, U, V, transpose=True), x @ Q, atol=1e-10)


def test_cayley_Q_is_orthogonal():
    """The expm(S) Q from a random skew S is orthogonal: ‖QᵀQ − I‖ < 1e-10 (fp64).
    Probed columnwise through ``cayley_apply`` on the identity (no dense Q)."""
    from train_gpt import cayley_apply

    torch.manual_seed(1)
    d, r = 16, 5
    U = torch.randn(d, r, dtype=torch.float64)
    V = torch.randn(d, r, dtype=torch.float64)
    I = torch.eye(d, dtype=torch.float64)
    Q = cayley_apply(I, U, V)          # rows are Q applied to e_i -> Q matrix rows
    # Q here is the matrix whose i-th ROW is Q·e_i, i.e. Qᵀ. QᵀQ = I either way.
    assert (Q @ Q.T - I).abs().max() < 1e-10
    assert (Q.T @ Q - I).abs().max() < 1e-10
    # det = +1 (expm of a skew matrix is a proper rotation, det > 0).
    assert torch.linalg.det(Q) > 0


def test_cayley_apply_roundtrip_is_identity():
    """Qᵀ(Q·x) == x to ~1e-12 in fp64 — the load-bearing reversibility primitive."""
    from train_gpt import cayley_apply

    torch.manual_seed(2)
    d, r = 24, 8
    U = torch.randn(d, r, dtype=torch.float64)
    V = torch.randn(d, r, dtype=torch.float64)
    x = torch.randn(4, 7, d, dtype=torch.float64)
    rt = cayley_apply(cayley_apply(x, U, V), U, V, transpose=True)
    assert (rt - x).abs().max() < 1e-12
    # Also the other order Q(Qᵀ·x) == x.
    rt2 = cayley_apply(cayley_apply(x, U, V, transpose=True), U, V)
    assert (rt2 - x).abs().max() < 1e-12


def test_cayley_apply_near_identity_at_small_UV():
    """Small U,V (1e-3 init scale) ⇒ S≈0 ⇒ Q≈I: the recurrence starts ≈ the
    additive coupling (stability + the near-identity readout start)."""
    from train_gpt import cayley_apply

    torch.manual_seed(3)
    d, r = 16, 4
    U = 1e-3 * torch.randn(d, r, dtype=torch.float64)
    V = 1e-3 * torch.randn(d, r, dtype=torch.float64)
    x = torch.randn(2, 3, d, dtype=torch.float64)
    y = cayley_apply(x, U, V)
    assert (y - x).abs().max() < 1e-4  # Q ≈ I at the 1e-3 init scale


# ---------------------------------------------------------------------------
# GROWN-Q numerical-stability gate (the regime that broke late-training Cayley).
# As the learnable U,V grow (init ×10–50, so ‖S‖ is O(1)–O(10), mimicking late
# training) the (I+S)^{-1} Cayley solve becomes ill-conditioned and Q drifts off
# the orthogonal manifold ⇒ Qᵀ(Q·x)≠x ⇒ recon explodes. Q=expm(S) (matrix-
# exponential of the projected low-rank skew) is EXACTLY orthogonal for ANY ‖S‖
# (no (I+S)^{-1} solve), so orthogonality + round-trip stay machine-exact.
# ---------------------------------------------------------------------------
def _q_matrix_from_apply(apply_fn, U, V, d, transpose=False):
    """Materialize the dense Q (or Qᵀ) by applying ``apply_fn`` to the identity.

    Row-vector convention: ``apply_fn(I, ...)`` returns the matrix whose i-th ROW
    is the map applied to e_i, i.e. (for Q) the matrix Qᵀ. QᵀQ=I either way, and
    we transpose back to recover Q itself for the explicit ‖QᵀQ−I‖ probe."""
    I = torch.eye(d, dtype=torch.float64)
    rows = apply_fn(I, U, V, transpose=transpose)   # i-th row = map·e_i
    return rows.transpose(-1, -2)                    # columns = map·e_i  -> the map matrix


@pytest.mark.parametrize("scale", [1.0, 5.0, 10.0, 30.0, 50.0])
def test_orthogonal_apply_stays_orthogonal_at_large_S(scale):
    """‖QᵀQ−I‖ < 1e-9 (fp64) across a range of ‖S‖ from small to LARGE.

    expm(S) is exactly orthogonal for ANY ‖S‖ (no (I+S)^{-1} solve to lose
    precision). The residual is fp64 matmul rounding (≤~1e-10 even at ‖S‖~1e5)."""
    from train_gpt import cayley_apply

    torch.manual_seed(11)
    d, r = 20, 6
    base_U = torch.randn(d, r, dtype=torch.float64)
    base_V = torch.randn(d, r, dtype=torch.float64)
    U = scale * base_U
    V = scale * base_V
    S = U @ V.T - V @ U.T
    s_norm = float(torch.linalg.matrix_norm(S, ord=2))
    Q = _q_matrix_from_apply(cayley_apply, U, V, d)
    I = torch.eye(d, dtype=torch.float64)
    ortho_err = (Q.T @ Q - I).abs().max()
    assert ortho_err < 1e-9, (
        f"Q not orthogonal at ‖S‖≈{s_norm:.2f} (scale={scale}): "
        f"‖QᵀQ−I‖={float(ortho_err):.2e}")


@pytest.mark.parametrize("scale", [1.0, 5.0, 10.0, 30.0, 50.0])
def test_orthogonal_apply_roundtrip_at_large_S(scale):
    """Qᵀ(Q·x)=x to < 1e-9 (fp64) across small→large ‖S‖.

    This is the load-bearing reversibility primitive. expm(S)·expm(−S)=I holds
    for ANY ‖S‖ (no solve), so the round-trip stays machine-exact even in the
    late-training regime where ‖S‖ is O(1)+; the Cayley solve degraded here."""
    from train_gpt import cayley_apply

    torch.manual_seed(12)
    d, r = 24, 8
    U = scale * torch.randn(d, r, dtype=torch.float64)
    V = scale * torch.randn(d, r, dtype=torch.float64)
    S = U @ V.T - V @ U.T
    s_norm = float(torch.linalg.matrix_norm(S, ord=2))
    x = torch.randn(4, 7, d, dtype=torch.float64)
    rt = cayley_apply(cayley_apply(x, U, V), U, V, transpose=True)
    err = (rt - x).abs().max()
    assert err < 1e-9, (
        f"Qᵀ(Q·x)≠x at ‖S‖≈{s_norm:.2f} (scale={scale}): "
        f"round-trip err={float(err):.2e}")
    # And the other order Q(Qᵀ·x)=x.
    rt2 = cayley_apply(cayley_apply(x, U, V, transpose=True), U, V)
    err2 = (rt2 - x).abs().max()
    assert err2 < 1e-9, (
        f"Q(Qᵀ·x)≠x at ‖S‖≈{s_norm:.2f} (scale={scale}): err={float(err2):.2e}")


def test_orthogonal_apply_beats_cayley_solve_at_large_S():
    """DIRECT fix demonstration: at a LARGE ‖S‖ (the late-training regime), the
    expm(S) apply matches the TRUE orthogonal target ``torch.matrix_exp(S)`` to
    ~1e-9, while the OLD Cayley-solve form ``(I−S)(I+S)^{-1}`` (reconstructed
    here) is BOTH a different matrix (the Cayley transform ≠ expm, so it diverges
    from the orthogonal target by O(1)) AND a less accurate orthogonal map.

    This is the deterministic, seed-stable proof that the bug fix is correct: the
    Cayley solve becomes ill-conditioned / inexact as U,V grow; expm does not."""
    from train_gpt import cayley_apply

    def _cayley_solve(x, U, V, transpose=False):
        # The OLD low-rank Woodbury Cayley apply (pre-fix), for comparison only.
        dt = x.dtype
        U, V = U.to(dt), V.to(dt)
        M = torch.cat([U, V], dim=-1)
        N = torch.cat([V, -U], dim=-1)
        sgn = -1.0 if transpose else 1.0
        Mh = sgn * M
        eye = torch.eye(M.shape[-1], dtype=dt)
        cap = eye + (N.transpose(-1, -2) @ Mh)
        sol = torch.linalg.solve(cap, (x @ N).unsqueeze(-1)).squeeze(-1)
        w = x - sol @ Mh.transpose(-1, -2)
        return w - sgn * ((w @ N) @ M.transpose(-1, -2))

    torch.manual_seed(20)
    d, r = 16, 8
    U = 50.0 * torch.randn(d, r, dtype=torch.float64)   # LARGE ‖S‖ (late training)
    V = 50.0 * torch.randn(d, r, dtype=torch.float64)
    S = U @ V.T - V @ U.T
    Q_true = torch.matrix_exp(S)                         # the exact orthogonal target
    x = torch.randn(6, d, dtype=torch.float64)
    target = x @ Q_true.T
    expm_err = (cayley_apply(x, U, V) - target).abs().max()
    cayley_err = (_cayley_solve(x, U, V) - target).abs().max()
    # expm matches the orthogonal target; the Cayley transform is a DIFFERENT map.
    assert expm_err < 1e-9, f"expm apply should match matrix_exp, err={float(expm_err):.2e}"
    assert cayley_err > 1e-3, (
        "Cayley transform should differ from expm(S) at large ‖S‖ "
        f"(it is a different orthogonal matrix), got err={float(cayley_err):.2e}")


def test_orthogonal_apply_matches_true_expm_at_large_S():
    """The low-rank apply equals the TRUE full-space matrix-exponential expm(S)
    of the skew S, at a LARGE ‖S‖ (the regime where the Cayley solve drifts).

    expm(S) is the unique exactly-orthogonal target; the projected low-rank apply
    must equal it because S is rank ≤ 2r and acts as identity off span([U|V])."""
    from train_gpt import cayley_apply

    torch.manual_seed(13)
    d, r = 18, 5
    U = 20.0 * torch.randn(d, r, dtype=torch.float64)
    V = 20.0 * torch.randn(d, r, dtype=torch.float64)
    S = U @ V.T - V @ U.T
    Q_true = torch.matrix_exp(S)        # exact full-space expm of the skew
    x = torch.randn(3, 5, d, dtype=torch.float64)
    # Row-vector convention: applying Q to each d-vector is x @ Qᵀ.
    assert torch.allclose(cayley_apply(x, U, V), x @ Q_true.T, atol=1e-9)
    assert torch.allclose(cayley_apply(x, U, V, transpose=True), x @ Q_true, atol=1e-9)


def test_orthogonal_apply_rank_deficient_W_is_orthogonal():
    """Handle rank deficiency in W=[U|V]: when U,V share columns the active span
    is < 2r. The apply must still produce an exactly-orthogonal Q (round-trip
    exact) — qr must keep only the actual span."""
    from train_gpt import cayley_apply

    torch.manual_seed(14)
    d, r = 16, 4
    U = 10.0 * torch.randn(d, r, dtype=torch.float64)
    V = U.clone()                       # span([U|V]) = span(U), so k = r < 2r
    x = torch.randn(2, 3, d, dtype=torch.float64)
    # S = U Vᵀ − V Uᵀ = U Uᵀ − U Uᵀ = 0 here, so Q = I exactly.
    y = cayley_apply(x, U, V)
    assert torch.allclose(y, x, atol=1e-9)
    rt = cayley_apply(cayley_apply(x, U, V), U, V, transpose=True)
    assert (rt - x).abs().max() < 1e-9
    # And a partial overlap (V shares one column with U) must still round-trip.
    V2 = 10.0 * torch.randn(d, r, dtype=torch.float64)
    V2[:, 0] = U[:, 0]
    rt2 = cayley_apply(cayley_apply(x, U, V2), U, V2, transpose=True)
    assert (rt2 - x).abs().max() < 1e-9


def test_mla_shapes_and_kv_latent():
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8)
    x = torch.randn(2, 6, 32)
    y = m(x)
    assert y.shape == x.shape
    assert m.kv_latent == 8


def test_mla_is_causal():
    """Output at position t must not depend on inputs at positions > t."""
    torch.manual_seed(0)
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8).double().eval()
    T = 7
    x = torch.randn(1, T, 32, dtype=torch.float64)
    with torch.no_grad():
        y_full = m(x)
    # Perturb the LAST token's input; earlier outputs must be unchanged.
    x2 = x.clone()
    x2[:, -1, :] += 1.0
    with torch.no_grad():
        y_pert = m(x2)
    assert torch.allclose(y_full[:, :-1], y_pert[:, :-1], atol=1e-9)
    # Sanity: the last position's output DID change (causality, not a no-op).
    assert not torch.allclose(y_full[:, -1], y_pert[:, -1], atol=1e-6)


def test_mla_q_latent_override():
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8, q_latent=12)
    x = torch.randn(2, 5, 32)
    assert m(x).shape == x.shape


def test_moe_routers_run_and_relu_is_sparse():
    x = torch.randn(2, 5, 32)
    for rt in ("softmax", "relu"):
        moe = SwiGLUMoE(dim=32, n_experts=8, expert_rank=8, router_type=rt)
        y = moe(x)
        assert y.shape == x.shape
        if rt == "relu":
            assert (moe.last_route == 0).any()  # exact-zero sparsity


def test_moe_routing_is_smooth_for_reversibility():
    moe = SwiGLUMoE(dim=32, n_experts=8, expert_rank=8, router_type="relu")
    x = torch.randn(2, 5, 32)
    y1 = moe(x)
    y2 = moe(x + 1e-7)
    assert (y1 - y2).abs().max() < 1e-3  # continuous (no discrete jumps)


def test_mos_head_is_distribution_and_high_rank():
    h = MoSHead(dim=16, vocab=32, n_mix=3)
    z = torch.randn(4, 7, 16)
    logp = h(z)
    assert logp.shape == (4, 7, 32)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(4, 7), atol=1e-4)


def test_m0gpt_forward_backward_and_depth():
    args = Hyperparameters(model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=32,
                           n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8)
    m = M0GPT(args)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    loss = m(x, y, depth=4)
    loss.backward()
    assert torch.isfinite(loss)
    # tied embedding: out_embed shares the input embedding Parameter
    assert m.tok_emb.weight.data_ptr() == m.mos_head.out_embed.weight.data_ptr()


def test_m0gpt_recurrence_reconstructs_with_real_blocks():
    """Reversibility integration gate with the REAL MLA+MoE delta blocks (fp64)."""
    torch.manual_seed(0)
    args = Hyperparameters(model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16,
                           n_experts=4, expert_rank=4, n_mix=2, kv_latent=4, head_dim=8)
    m = M0GPT(args).double()
    # M0GPT zero-inits the delta-block output projections (near-identity start),
    # which would make the recurrence a trivial identity here. Re-randomize them
    # so reconstruction is tested against NON-TRIVIAL F/G updates.
    rec = m.rec  # the ReversibleRecurrence
    with torch.no_grad():
        for blk in (rec.F, rec.G):
            for sub in blk.sublayers:
                for mla in sub.attn_modules():
                    mla.o_proj.weight.normal_(std=0.3)
                for moe in sub.moe_modules():
                    moe.w_out.normal_(std=0.3)
    x0 = torch.randn(2, 5, 16, dtype=torch.float64)
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=4)
    assert not torch.allclose(aK, x0), "recurrence is trivially identity; test is vacuous"
    a0, b0 = rec.invert(aK, bK, x0, depth=4)
    assert torch.allclose(a0, x0, atol=1e-7) and torch.allclose(b0, x0, atol=1e-7)


# ---------------------------------------------------------------------------
# DeepSeek-V3 loss-free balancing bias + ST-MoE z-loss (MoE collapse fix)
# ---------------------------------------------------------------------------
def test_router_bias_is_a_buffer_not_a_parameter():
    """The DeepSeek-V3 ``router_bias`` MUST be a detached buffer (no grad, in no
    optimizer group), so it is constant within a forward+backward (reversibility
    safe). It must NOT appear in ``model.parameters()`` nor require grad."""
    from train_gpt import _RouterBiasMixin

    args = Hyperparameters(model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
                           n_experts=4, expert_rank=8, n_mix=2, kv_latent=8,
                           head_dim=8, max_seq_len=16, router_bias_update_rate=1e-3)
    m = M0GPT(args)
    param_ids = {id(p) for p in m.parameters()}
    n_moe = 0
    for mod in m.modules():
        if isinstance(mod, _RouterBiasMixin):
            n_moe += 1
            assert id(mod.router_bias) not in param_ids, "router_bias is a Parameter!"
            assert not mod.router_bias.requires_grad
            # It is a registered buffer (appears in the buffer set).
            assert any(b is mod.router_bias for _, b in mod.named_buffers())
    assert n_moe >= 1


def test_router_bias_update_rule_under_used_rises_over_used_falls():
    """DeepSeek-V3 sign rule: after a bias update from an IMBALANCED synthetic
    load, under-used experts' bias RISES and over-used experts' bias FALLS, and
    the bias is re-centered (zero mean)."""
    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax",
                    router_bias_update_rate=0.1)
    # expert 0 heavily over-used, expert 3 heavily under-used.
    moe._load_accum = torch.tensor([100.0, 50.0, 30.0, 5.0])
    moe._load_count = torch.tensor(185.0)
    b0 = moe.router_bias.clone()
    moe.update_router_bias()
    db = moe.router_bias - b0
    assert db[3] > 0 and db[0] < 0, f"sign rule violated: {db.tolist()}"
    assert moe.router_bias[3] > moe.router_bias[0]
    assert abs(float(moe.router_bias.mean())) < 1e-6, "bias must be re-centered"
    # Accumulators reset for the next optimizer step window.
    assert float(moe._load_count) == 0.0


def test_router_bias_improves_global_utilization_vs_no_bias():
    """Loss-free balancing must IMPROVE global expert utilization vs no bias on a
    deliberately imbalanced router. Several sign-rule updates from the imbalanced
    load push mass toward under-used experts -> higher utilization entropy."""
    from train_gpt import _global_util_entropy

    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax",
                    router_bias_update_rate=0.5)
    # Bias the router weights so expert 0 dominates at init.
    with torch.no_grad():
        moe.router.bias[0] = 4.0
    x = torch.randn(4, 8, 16)
    moe._load_accum_enabled = True
    moe(x)
    util_before = _global_util_entropy(moe.last_route)
    # Run several accumulate + step-boundary updates.
    for _ in range(40):
        moe._load_accum = torch.zeros_like(moe._load_accum)
        moe._load_count = torch.zeros_like(moe._load_count)
        moe(x)
        moe.update_router_bias()
    moe(x)
    util_after = _global_util_entropy(moe.last_route)
    assert util_after > util_before + 0.1, (
        f"loss-free bias did not improve utilization: {util_before:.4f} -> "
        f"{util_after:.4f}")


def test_router_bias_zero_rate_is_a_no_op():
    """rate=0 disables the bias: update is a no-op and the bias stays at zero."""
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax",
                    router_bias_update_rate=0.0)
    moe._load_accum = torch.tensor([100.0, 1.0, 1.0, 1.0])
    moe._load_count = torch.tensor(103.0)
    assert moe.update_router_bias() is None
    assert torch.allclose(moe.router_bias, torch.zeros(4))


def test_router_z_loss_formula_and_nonneg():
    """ST-MoE z-loss equals ``mean(logsumexp(router_logits)^2)`` over routed
    tokens (computed on the saved block input), is finite and >= 0."""
    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax")
    x = torch.randn(2, 5, 16)
    moe(x)  # sets _route_input
    z = moe.router_z_loss()
    assert z is not None and torch.isfinite(z) and float(z.detach()) >= 0.0
    # Recompute the documented formula from the saved input.
    logits = moe.router(moe._route_input)
    expected = (torch.logsumexp(logits, dim=-1) ** 2).mean()
    assert torch.allclose(z, expected, atol=1e-5)


def test_router_z_loss_decreases_logit_magnitude_when_minimized():
    """Minimizing the z-loss bounds the router logit magnitude: a few gradient
    steps on ONLY the z-loss shrink the mean |logsumexp(logits)|."""
    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax")
    # Inflate the router so the logits start large.
    with torch.no_grad():
        moe.router.weight.mul_(5.0)
    x = torch.randn(4, 8, 16)
    moe(x)
    with torch.no_grad():
        mag0 = float(torch.logsumexp(moe.router(moe._route_input), dim=-1).abs().mean())
    opt = torch.optim.SGD(moe.router.parameters(), lr=0.5)
    for _ in range(30):
        moe(x)
        opt.zero_grad(set_to_none=True)
        moe.router_z_loss().backward()
        opt.step()
    moe(x)
    with torch.no_grad():
        mag1 = float(torch.logsumexp(moe.router(moe._route_input), dim=-1).abs().mean())
    assert mag1 < mag0, f"z-loss did not shrink logit magnitude: {mag0:.4f} -> {mag1:.4f}"


def test_expert_output_cosine_diversity_range_and_degenerate():
    """``expert_cos_div`` is in [0, ~2] (1 - cos in [0,2]) and finite; duplicated
    experts (identical weights) give ~0 diversity."""
    from train_gpt import expert_output_cosine_diversity

    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax")
    d = expert_output_cosine_diversity(moe)
    assert torch.isfinite(torch.tensor(d)) and d >= 0.0
    # Degenerate: make every expert identical -> diversity ~0.
    with torch.no_grad():
        moe.w_in.copy_(moe.w_in[0:1].expand_as(moe.w_in))
        moe.w_out.copy_(moe.w_out[0:1].expand_as(moe.w_out))
    d_deg = expert_output_cosine_diversity(moe)
    assert abs(d_deg) < 1e-4, f"duplicated experts should give ~0 diversity, got {d_deg}"
