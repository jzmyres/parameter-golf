"""Task 8 — two-goal metric functions (known-input correctness).

The project tracks TWO goals with principled metrics:

* Resource: activation-memory scaling ``R_act(K)``, KV bytes/token, active-expert
  fraction (MoE sparsity), parameter count, artifact bytes.
* Expressiveness: ``val_bpb`` (already covered by the trainer), effective rank
  (``erank``, Roy & Vetterli 2007 spectral entropy of the singular values), and
  the recurrence-equivalence exponent ``phi`` (Iso-Depth scaling-law exponent:
  does looping a block ``r`` times buy the capacity of ``r`` unique blocks?).

Task 8 adds the *pure metric functions* plus per-step logging of the cheap
metrics (``erank`` / ``active_frac`` / ``kv_bytes`` / ``params``). The
control-experiment runner (Task 9) computes ``phi`` over ``r`` and ``R_act``
over ``K`` from these primitives. Here we lock in known-input correctness and
deterministic behaviour for the primitives themselves.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (  # noqa: E402
    Hyperparameters,
    M0GPT,
    active_expert_fraction,
    effective_rank,
    fit_phi,
    fit_phi_isodepth,
    kv_bytes_per_token,
    recurrence_param_counts,
    _fit_slope,
    vram_vs_batch_scaling,
)


# --- effective_rank (Roy & Vetterli spectral entropy) ---------------------
def test_effective_rank_known():
    # Isotropic spectrum -> erank == dim.
    assert abs(effective_rank(torch.eye(4)) - 4.0) < 1e-4
    # Rank-1 spectrum -> erank == 1.
    s2 = torch.zeros(4, 4)
    s2[0, 0] = 1.0
    assert abs(effective_rank(s2) - 1.0) < 1e-4


def test_effective_rank_singular_value_vector_input():
    # 1D input is treated as the singular-value spectrum directly.
    assert abs(effective_rank(torch.ones(8)) - 8.0) < 1e-4
    assert abs(effective_rank(torch.tensor([1.0, 0.0, 0.0])) - 1.0) < 1e-4


def test_effective_rank_two_equal_values():
    # Two equal nonzero singular values -> erank == 2 (independent of magnitude).
    assert abs(effective_rank(torch.tensor([3.0, 3.0, 0.0, 0.0])) - 2.0) < 1e-4


def test_effective_rank_is_float_and_deterministic():
    m = torch.randn(6, 5, generator=torch.Generator().manual_seed(0))
    a = effective_rank(m)
    b = effective_rank(m)
    assert isinstance(a, float)
    assert a == b
    # erank is bounded by the number of singular values (min dim).
    assert 1.0 <= a <= 5.0 + 1e-4


# --- fit_phi (Iso-Depth recurrence-equivalence exponent) ------------------
def test_fit_phi_in_range():
    losses = {1: 3.0, 2: 2.7, 4: 2.5, 8: 2.4}
    phi = fit_phi(losses)
    assert 0.0 <= phi <= 1.0


def test_fit_phi_full_equivalence_is_one():
    # If looping r times improves loss exactly like r UNIQUE blocks would, phi=1.
    # The unique-block reference slope is b_ref=1.0, so construct loss = a - log(r).
    import math
    a, b = 3.0, 1.0
    losses = {r: a - b * math.log(r) for r in (1, 2, 4, 8)}
    phi = fit_phi(losses)
    assert abs(phi - 1.0) < 1e-3


def test_fit_phi_no_improvement_is_zero():
    # Looping buys nothing (loss flat in r) -> phi == 0.
    losses = {1: 2.5, 2: 2.5, 4: 2.5, 8: 2.5}
    assert abs(fit_phi(losses) - 0.0) < 1e-6


def test_fit_phi_deterministic_and_float():
    losses = {1: 3.0, 2: 2.7, 4: 2.5, 8: 2.4}
    assert isinstance(fit_phi(losses), float)
    assert fit_phi(losses) == fit_phi(losses)


def test_fit_phi_degenerate_single_point():
    # Fewer than two depths -> no slope to estimate -> defined as 0.0.
    assert fit_phi({4: 2.5}) == 0.0
    assert fit_phi({}) == 0.0


# --- fit_phi_isodepth (PRINCIPLED Iso-Depth train-r scaling-law phi) -------
# Fits L(r) = E + A * (N_once + r^phi * N_rec)^(-alpha) to a {r: trained loss}
# sweep, recovering the recurrence-equivalence exponent phi as a FITTED law
# parameter (arXiv:2604.21106). The synthetic-recovery tests below generate
# L(r) from KNOWN (phi, alpha, E, A, N_once, N_rec) and assert phi is recovered.
_ISO_R = [1, 2, 4, 8, 16]
_ISO_N_ONCE = 2_000_000
_ISO_N_REC = 6_000_000


def _isodepth_law(r, phi, alpha, E, A, n_once, n_rec):
    return E + A * (n_once + (r ** phi) * n_rec) ** (-alpha)


def _isodepth_losses(phi, alpha=0.3, E=2.5, A=50.0,
                     n_once=_ISO_N_ONCE, n_rec=_ISO_N_REC, rs=_ISO_R):
    return {r: _isodepth_law(r, phi, alpha, E, A, n_once, n_rec) for r in rs}


def test_fit_phi_isodepth_recovers_known_phi_noiseless():
    # Noiseless recovery within +/-0.05 for representative phi in (0, 1).
    for phi_true in (0.3, 0.6, 0.9):
        losses = _isodepth_losses(phi_true)
        out = fit_phi_isodepth(losses, _ISO_N_ONCE, _ISO_N_REC)
        assert math.isfinite(out["phi"]), out
        assert abs(out["phi"] - phi_true) < 0.05, (phi_true, out)
        assert 0.0 <= out["phi"] <= 1.0
        assert out["n_points"] == len(_ISO_R)
        assert out["rmse"] < 1e-3  # near-exact fit on noiseless data


def test_fit_phi_isodepth_recovers_known_phi_with_noise():
    # Small additive noise -> recovery within +/-0.1 (deterministic seed).
    #
    # NOTE on noise scale: the 4-parameter law over only 5 r is a small-sample,
    # weakly-identified fit (phi/alpha/A partially trade off), so its tolerance to
    # noise is SMALL — empirically ~1e-5 absolute (≈0.05% of the inter-r loss gap)
    # keeps every phi within +/-0.1; an order of magnitude more already pushes the
    # shallow phi=0.3 case past 0.1. This fragility is a real property of the
    # estimator (documented in fit_phi_isodepth and the harness header): use more
    # r and/or seeds for tight confidence intervals on real sweeps.
    g = torch.Generator().manual_seed(0)
    for phi_true in (0.3, 0.6, 0.9):
        clean = _isodepth_losses(phi_true)
        noisy = {
            r: v + 1e-5 * float(torch.randn((), generator=g))
            for r, v in clean.items()
        }
        out = fit_phi_isodepth(noisy, _ISO_N_ONCE, _ISO_N_REC)
        assert math.isfinite(out["phi"]), out
        assert abs(out["phi"] - phi_true) < 0.1, (phi_true, out)


def test_fit_phi_isodepth_edge_phi_zero_flat_loss():
    # phi = 0 -> r^phi = 1 -> loss is FLAT in r (looping buys nothing).
    losses = _isodepth_losses(0.0)
    vals = list(losses.values())
    assert max(vals) - min(vals) < 1e-9  # confirm the construction is flat
    out = fit_phi_isodepth(losses, _ISO_N_ONCE, _ISO_N_REC)
    assert math.isfinite(out["phi"])
    assert out["phi"] < 0.05


def test_fit_phi_isodepth_edge_phi_one_full_equivalence():
    # phi = 1 -> each loop is worth a full unique block.
    out = fit_phi_isodepth(_isodepth_losses(1.0), _ISO_N_ONCE, _ISO_N_REC)
    assert math.isfinite(out["phi"])
    assert abs(out["phi"] - 1.0) < 0.05


def test_fit_phi_isodepth_too_few_points_is_nan():
    # <3 finite points -> phi is NaN with a reason, no crash.
    out = fit_phi_isodepth({1: 3.0, 2: 2.9}, _ISO_N_ONCE, _ISO_N_REC)
    assert math.isnan(out["phi"])
    assert out["n_points"] == 2
    assert isinstance(out.get("reason"), str) and out["reason"]


def test_fit_phi_isodepth_drops_nonfinite_points():
    # A diverged r (NaN/inf loss) is DROPPED before fitting; the fit proceeds on
    # the remaining finite points and stays finite + in range. (With only the 3
    # surviving r the 4-parameter law is under-identified, so we assert the drop +
    # a sane finite phi, not tight recovery — that needs the full sweep.)
    losses = _isodepth_losses(0.6)
    losses[8] = float("nan")        # diverged depth
    losses[16] = float("inf")       # diverged depth
    out = fit_phi_isodepth(losses, _ISO_N_ONCE, _ISO_N_REC)
    assert out["n_points"] == 3     # r in {1, 2, 4} survive the finite filter
    assert math.isfinite(out["phi"])
    assert 0.0 <= out["phi"] <= 1.0


def test_fit_phi_isodepth_returns_full_param_dict():
    out = fit_phi_isodepth(_isodepth_losses(0.5), _ISO_N_ONCE, _ISO_N_REC)
    for key in ("phi", "alpha", "E", "A", "rmse", "n_points"):
        assert key in out, key
    assert all(isinstance(out[k], float) for k in ("phi", "alpha", "E", "A", "rmse"))
    assert isinstance(out["n_points"], int)


# --- recurrence_param_counts (N_once / N_rec split for the Iso-Depth law) --
def _tiny_m0gpt(step_conditioning=True):
    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, step_conditioning=step_conditioning,
    )
    return M0GPT(args)


def test_recurrence_param_counts_split_matches_named_parameters():
    model = _tiny_m0gpt()
    n_once, n_rec = recurrence_param_counts(model)
    # n_rec is exactly the params under model.rec (F/G blocks + step_emb).
    expected_rec = sum(p.numel() for p in model.rec.parameters())
    assert n_rec == expected_rec
    assert n_rec > 0 and n_once > 0


def test_recurrence_param_counts_sum_is_total_trainable():
    model = _tiny_m0gpt()
    n_once, n_rec = recurrence_param_counts(model)
    # Tied embedding is one tensor object -> named_parameters dedupes it, so the
    # split sums to the total trainable params (counted once).
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    assert n_once + n_rec == total


def test_recurrence_param_counts_tied_embedding_counted_once():
    model = _tiny_m0gpt()
    assert model.tok_emb.weight is model.mos_head.out_embed.weight
    n_once, _n_rec = recurrence_param_counts(model)
    # The tied (vocab x dim) weight is in n_once exactly once, not twice.
    vocab, dim = model.tok_emb.weight.shape
    # n_once contains tok_emb + pos_emb + final_norm + mos_head (ctx/gate),
    # and the out_embed alias must NOT double-count the embedding.
    non_rec = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("rec.")
    )
    assert n_once == non_rec
    # Sanity: the embedding block (vocab*dim) appears once in the non-rec sum.
    assert non_rec >= vocab * dim


def test_recurrence_param_counts_unwraps_ddp_module():
    model = _tiny_m0gpt()

    class _FakeDDP:
        def __init__(self, m):
            self.module = m

    direct = recurrence_param_counts(model)
    wrapped = recurrence_param_counts(_FakeDDP(model))
    assert direct == wrapped


# --- active_expert_fraction (MoE sparsity) --------------------------------
def test_active_expert_fraction_softmax_is_full():
    # softmax weights are strictly positive -> ~all experts active.
    from train_gpt import SwiGLUMoE

    moe = SwiGLUMoE(dim=8, n_experts=4, expert_rank=4, router_type="softmax")
    moe(torch.randn(2, 5, 8))
    assert abs(active_expert_fraction(moe) - 1.0) < 1e-6


def test_active_expert_fraction_route_tensor():
    # Direct route-tensor input: half the weights are exact zeros -> 0.5.
    route = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
    assert abs(active_expert_fraction(route) - 0.5) < 1e-6


def test_active_expert_fraction_relu_in_unit_interval():
    from train_gpt import SwiGLUMoE

    moe = SwiGLUMoE(dim=8, n_experts=4, expert_rank=4, router_type="relu")
    moe(torch.randn(2, 5, 8))
    frac = active_expert_fraction(moe)
    assert 0.0 <= frac <= 1.0


# --- kv_bytes_per_token (MLA cache footprint) -----------------------------
def test_kv_bytes_per_token_positive_int():
    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    kvb = kv_bytes_per_token(args)
    assert isinstance(kvb, int)
    assert kvb > 0


def test_kv_bytes_per_token_scales_with_latent():
    base = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    big = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=16, head_dim=8,
        max_seq_len=16,
    )
    assert kv_bytes_per_token(big) > kv_bytes_per_token(base)


# --- _fit_slope (OLS slope/intercept for VRAM-vs-batch scaling) ------------
def test_fit_slope_proportional():
    # ys = 10 * xs (zero intercept) -> slope == 10, intercept == 0.
    slope, intercept = _fit_slope([1, 2, 4, 8], [10, 20, 40, 80])
    assert abs(slope - 10.0) < 1e-9
    assert abs(intercept - 0.0) < 1e-9


def test_fit_slope_flat_is_zero_slope():
    # Constant ys (memory-efficiency / flat scaling goal) -> slope == 0.
    slope, intercept = _fit_slope([1, 2, 4, 8], [10, 10, 10, 10])
    assert abs(slope - 0.0) < 1e-9
    assert abs(intercept - 10.0) < 1e-9


def test_fit_slope_affine_with_intercept():
    # ys = 3 * xs + 5 -> slope == 3, intercept == 5.
    slope, intercept = _fit_slope([0, 1, 2, 3], [5, 8, 11, 14])
    assert abs(slope - 3.0) < 1e-9
    assert abs(intercept - 5.0) < 1e-9


def test_fit_slope_degenerate_single_point():
    # Fewer than two distinct xs -> no slope -> (0.0, mean(ys)).
    slope, intercept = _fit_slope([4], [7])
    assert slope == 0.0
    assert abs(intercept - 7.0) < 1e-9
    slope, intercept = _fit_slope([2, 2, 2], [3, 5, 7])
    assert slope == 0.0
    assert abs(intercept - 5.0) < 1e-9


def test_fit_slope_returns_floats():
    slope, intercept = _fit_slope([1, 2], [1, 2])
    assert isinstance(slope, float)
    assert isinstance(intercept, float)


# --- vram_vs_batch_scaling (CPU: available=False, slope-fit exercisable) ----
def test_vram_vs_batch_scaling_cpu_unavailable():
    # On CPU (no CUDA) the helper must NOT crash; it reports unavailability so
    # the caller can still exercise the pure slope-fit on injected points.
    import torch

    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    out = vram_vs_batch_scaling(model=None, args=args, batch_sizes=[1, 2],
                                device=torch.device("cpu"))
    assert out["available"] is False


def test_vram_vs_batch_scaling_slope_fit_on_synthetic_points():
    # The slope math used by vram_vs_batch_scaling is _fit_slope; verify it on
    # known (batch, vram) points so the CPU test covers the fit logic a real GPU
    # run would feed it. Flat VRAM across batch -> slope ~ 0 (reversibility win).
    slope_flat, _ = _fit_slope([1, 2, 4, 8], [4000.0, 4000.0, 4000.0, 4000.0])
    assert abs(slope_flat) < 1e-6
    # Linear growth -> positive slope == per-sample MB.
    slope_lin, intercept_lin = _fit_slope([1, 2, 4, 8], [100.0, 200.0, 400.0, 800.0])
    assert abs(slope_lin - 100.0) < 1e-6
    assert abs(intercept_lin) < 1e-6


# --- artifact save no longer hard-gates on 16 MB --------------------------
def test_save_int6_artifact_has_no_budget_gate():
    """The resource goal is memory-efficiency, not an artifact-size gate.

    ``save_int6_artifact`` must just quantize+serialize+compress and return the
    bytes; there is no ``MAX_ARTIFACT_BYTES`` raise in the M0 save path. We
    assert (a) the symbol is gone from the module and (b) saving a model never
    raises regardless of size.
    """
    import train_gpt
    from train_gpt import M0GPT, save_int6_artifact

    assert not hasattr(train_gpt, "MAX_ARTIFACT_BYTES"), (
        "M0 trainer should not carry a 16 MB artifact budget constant"
    )

    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    model = M0GPT(args)
    compressed, _qsd, _meta = save_int6_artifact(model.state_dict())
    assert isinstance(compressed, (bytes, bytearray))
    assert len(compressed) > 0


# --- recurrence_displacement (effective-depth signal) ---------------------
class _ConstUpdateRec(torch.nn.Module):
    """Toy recurrence whose per-step midpoint moves by a CONSTANT vector.

    ``forward_states`` is the only surface ``recurrence_displacement`` uses. Here
    each step adds a fixed ``delta`` to both streams, so ``z_k = x0 + k*delta``
    grows linearly; the absolute step is constant but the RELATIVE displacement
    ``||z_{k+1}-z_k|| / ||z_k||`` DECAYS as ``z`` grows. To get a *constant*
    relative displacement we instead scale the update by the current state.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward_states(self, a, b, x0, depth):
        states = []
        for _ in range(int(depth)):
            # Multiplicative (geometric) growth -> constant RELATIVE displacement.
            a = a + 0.5 * a
            b = b + 0.5 * b
            states.append(0.5 * (a + b))
        return (a, b), states


class _ContractRec(torch.nn.Module):
    """Toy recurrence that saturates: each step adds a geometrically SHRINKING
    update, so the midpoint converges and the relative displacement DECAYS."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward_states(self, a, b, x0, depth):
        states = []
        step = torch.ones_like(a)
        for _ in range(int(depth)):
            step = 0.3 * step           # geometrically shrinking update
            a = a + step
            b = b + step
            states.append(0.5 * (a + b))
        return (a, b), states


class _ToyModel(torch.nn.Module):
    """Minimal model exposing the (tok_emb, pos_emb, rec) surface that
    ``recurrence_displacement`` reads, wrapping a toy recurrence."""

    def __init__(self, rec, vocab=8, d=4, max_seq_len=16):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab, d)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, max_seq_len, d))
        self.rec = rec
        torch.nn.init.ones_(self.tok_emb.weight)  # nonzero x0 so ||z_0|| > 0


def test_recurrence_displacement_constant_update_is_flat():
    """A geometric (constant-relative-step) recurrence yields a flat per-step
    relative displacement: ``||z_{k+1}-z_k|| / ||z_k||`` is constant across k."""
    from train_gpt import recurrence_displacement

    m = _ToyModel(_ConstUpdateRec(4))
    tokens = torch.zeros(2, 5, dtype=torch.long)
    disp = recurrence_displacement(m, tokens, depth=6)
    assert len(disp) == 6
    # All steps share (nearly) the same relative displacement (= 0.5 here).
    assert max(disp) - min(disp) < 1e-4, disp
    assert abs(disp[0] - 0.5) < 1e-4, disp


def test_recurrence_displacement_contracting_decays():
    """A saturating recurrence (shrinking updates) yields a DECAYING per-step
    relative displacement — the recurrence stops doing work (low effective depth)."""
    from train_gpt import recurrence_displacement

    m = _ToyModel(_ContractRec(4))
    tokens = torch.zeros(2, 5, dtype=torch.long)
    disp = recurrence_displacement(m, tokens, depth=6)
    assert len(disp) == 6
    # Strictly decaying displacement (each step moves less than the previous).
    for k in range(1, len(disp)):
        assert disp[k] < disp[k - 1], disp
    assert disp[-1] < disp[0] * 0.5, disp


def test_recurrence_displacement_runs_on_real_m0gpt():
    """Smoke on a real M0GPT: returns ``depth`` finite non-negative floats."""
    from train_gpt import Hyperparameters, M0GPT, recurrence_displacement

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16, n_experts=4,
        expert_rank=4, n_mix=2, kv_latent=4, head_dim=8, max_seq_len=16,
    )
    m = M0GPT(args)
    tokens = torch.randint(0, 16, (2, 5))
    disp = recurrence_displacement(m, tokens, depth=4)
    assert len(disp) == 4
    assert all(isinstance(v, float) and v == v and v >= 0.0 for v in disp)


def test_displacement_tail_summary_helper():
    """``displacement_tail`` summarizes the per-step displacement as the mean over
    the last half of the steps (the ``disp_tail`` effective-depth headline)."""
    from train_gpt import displacement_tail

    # Flat list -> tail mean equals the flat value.
    assert abs(displacement_tail([0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9
    # Decaying list -> tail mean is the mean of the LAST half.
    vals = [1.0, 0.5, 0.25, 0.125]
    assert abs(displacement_tail(vals) - (0.25 + 0.125) / 2) < 1e-9
    # Empty -> NaN (no forward yet).
    assert displacement_tail([]) != displacement_tail([])


# --- route_step_div / expert_cos_div (MoE-basis-depth diagnostics) ---------
def test_route_step_diversity_in_range_on_real_m0gpt():
    """``route_step_div`` is a finite float in ``[1/K, 1]`` on a real M0GPT."""
    from train_gpt import Hyperparameters, M0GPT, route_step_diversity

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16, n_experts=4,
        expert_rank=4, n_mix=2, kv_latent=4, head_dim=8, max_seq_len=16,
    )
    m = M0GPT(args)
    tokens = torch.randint(0, 16, (2, 5))
    K = 6
    rsd = route_step_diversity(m, tokens, depth=K)
    assert isinstance(rsd, float) and rsd == rsd  # finite (not NaN)
    assert 1.0 / K - 1e-6 <= rsd <= 1.0 + 1e-6, rsd


def test_route_step_diversity_degenerate_single_expert_is_one_over_K():
    """A DEGENERATE router (one expert always dominant every step) gives
    ``route_step_div`` == 1/K (the collapse mode this fix targets)."""
    from train_gpt import Hyperparameters, M0GPT, route_step_diversity, SwiGLUMoE

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16, n_experts=4,
        expert_rank=4, n_mix=2, kv_latent=4, head_dim=8, max_seq_len=16,
    )
    m = M0GPT(args)
    # Force the FFN MoE routers to ALWAYS pick expert 0 (huge bias on logit 0):
    # the argmax is then expert 0 at every step -> 1 distinct expert / K.
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, SwiGLUMoE):
                mod.router.weight.zero_()
                mod.router.bias.zero_()
                mod.router.bias[0] = 1e3
    tokens = torch.randint(0, 16, (2, 5))
    K = 5
    rsd = route_step_diversity(m, tokens, depth=K)
    assert abs(rsd - 1.0 / K) < 1e-6, f"degenerate router should give 1/K, got {rsd}"


def test_expert_output_cosine_diversity_metric():
    """``expert_output_cosine_diversity`` is finite and >= 0; <2 experts -> NaN."""
    from train_gpt import SwiGLUMoE, expert_output_cosine_diversity

    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="softmax")
    d = expert_output_cosine_diversity(moe)
    assert d == d and d >= 0.0
    moe1 = SwiGLUMoE(dim=16, n_experts=1, expert_rank=8, router_type="softmax")
    assert expert_output_cosine_diversity(moe1) != expert_output_cosine_diversity(moe1)


# --- reconstruction_error (reversible round-trip / BPTT gradient-correctness gate) -
def _small_m0_args(**overrides):
    """Tiny CPU M0 config for the recon_rel probe tests."""
    from train_gpt import Hyperparameters

    base = dict(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16, n_experts=4,
        expert_rank=4, n_mix=2, kv_latent=4, head_dim=8, max_seq_len=16,
    )
    base.update(overrides)
    return Hyperparameters(**base)


def test_reconstruction_error_fp64_near_exact_depth8_and_16():
    """The gradient-correctness gate: in fp64 the reversible inverse round-trips
    the seed (a0, b0) to ~machine precision, so recon_rel is ~0 (< 1e-10) at both
    a shallow and a deep budget — confirming the backward reconstructs the true
    forward graph (and hence the true gradients)."""
    from train_gpt import M0GPT, reconstruction_error

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args()).double()  # fp64 model -> near-exact round-trip
    tokens = torch.randint(0, 16, (2, 5))
    for depth in (8, 16):
        rel = reconstruction_error(m, tokens, depth)
        assert isinstance(rel, float)
        assert rel >= 0.0
        assert rel < 1e-10, (depth, rel)


def test_reconstruction_error_fp32_finite_nonneg():
    """fp32 model: the probe returns a finite non-negative float (nonzero round-
    trip drift is allowed; the gate only requires a measurable finite value)."""
    from train_gpt import M0GPT, reconstruction_error

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args())  # default fp32
    tokens = torch.randint(0, 16, (2, 5))
    rel = reconstruction_error(m, tokens, depth=8)
    assert isinstance(rel, float)
    assert rel == rel and rel >= 0.0  # finite (not NaN) and non-negative


def test_reconstruction_error_random_init_finite_nonneg():
    """random-init model: round-trip fidelity is SEED-INDEPENDENT — the SAME
    (a0, b0) drawn for the forward are inverted from (aK, bK), so the relative
    round-trip error is still a finite non-negative float."""
    from train_gpt import M0GPT, reconstruction_error

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args(init_state="random"))
    tokens = torch.randint(0, 16, (2, 5))
    rel = reconstruction_error(m, tokens, depth=8)
    assert isinstance(rel, float)
    assert rel == rel and rel >= 0.0


def test_reconstruction_error_step_conditioning_fp64_near_exact():
    """step_conditioning model: e_k is recomputed deterministically from the step
    index inside ``invert``, so the round-trip stays near-exact in fp64 (< 1e-10).
    """
    from train_gpt import M0GPT, reconstruction_error

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args(step_conditioning=True)).double()
    tokens = torch.randint(0, 16, (2, 5))
    rel = reconstruction_error(m, tokens, depth=8)
    assert isinstance(rel, float)
    assert rel >= 0.0
    assert rel < 1e-10, rel


def test_m0gpt_forward_readout_dtype_fp32_under_bf16_autocast():
    """The recurrence now accumulates the stream in fp64 internally, but the
    READOUT path (final_norm/mos_head/loss) must stay at the model dtype (fp32):
    the midpoint z_K is cast back BEFORE final_norm. Under bf16 autocast the loss
    must be finite and the model parameters/output path remain fp32 (no fp64
    leak into the head)."""
    from train_gpt import M0GPT

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args())  # fp32 params
    tokens = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = m(tokens, targets, depth=8)
    assert torch.isfinite(loss)
    # final_norm/mos_head weights are fp32 — the fp64 accumulation is internal.
    assert m.final_norm.w.dtype == torch.float32
    assert m.tok_emb.weight.dtype == torch.float32


def test_reconstruction_error_bf16_autocast_near_exact_x0_and_random():
    """REPRODUCE-THEN-FIX gate: under CPU bf16 autocast the reversible round-trip
    must stay near-exact (recon_rel < 1e-6) for BOTH init modes at BOTH a shallow
    and a deep budget.

    Before the fp64-stream-accumulation + autocast-replay fix this FAILS: bf16
    coupling ± drifts (random-init grew > 1.0, x0-init ~1e-2..4e-2), meaning the
    O(1) reversible backward reconstructs activations that diverge from the true
    forward — i.e. the BPTT gradients are wrong. The fp64 stream accumulation
    keeps the ± exact while F/G still run their matmuls under bf16 autocast.
    """
    from train_gpt import M0GPT, reconstruction_error

    for init_state in ("x0", "random"):
        torch.manual_seed(0)
        m = M0GPT(_small_m0_args(init_state=init_state))  # fp32 params
        tokens = torch.randint(0, 16, (2, 5))
        for depth in (16, 32):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                rel = reconstruction_error(m, tokens, depth)
            assert isinstance(rel, float)
            assert rel >= 0.0
            assert rel < 1e-6, (init_state, depth, rel)


def test_reconstruction_error_bf16_float32_accum_is_finite():
    """``recurrence_accum_dtype='float32'`` option: under bf16 autocast the
    round-trip is still a finite non-negative float. fp32 accumulation is far
    better than ambient bf16 ± (it removes the catastrophic random-init blow-up)
    but is NOT exact like fp64, so we only assert finiteness + a loose bound."""
    from train_gpt import M0GPT, reconstruction_error

    torch.manual_seed(0)
    m = M0GPT(_small_m0_args(recurrence_accum_dtype="float32"))
    tokens = torch.randint(0, 16, (2, 5))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        rel = reconstruction_error(m, tokens, depth=16)
    assert isinstance(rel, float)
    assert rel == rel and rel >= 0.0
    assert rel < 1e-1, rel  # fp32 ± is well-behaved (no random-init blow-up)


def test_format_metrics_line_includes_recon_rel_when_present():
    """``recon_rel`` is rendered (scientific notation) when supplied and omitted
    when absent (mirrors the ``disp_tail`` formatter contract)."""
    from train_gpt import format_metrics_line

    base = {"erank": 1.0, "peak_vram": 0.0, "kv_bytes": 1, "params": 1}
    with_recon = format_metrics_line({**base, "recon_rel": 5e-16})
    assert "recon_rel:" in with_recon
    # Absent key -> field omitted.
    without = format_metrics_line(base)
    assert "recon_rel:" not in without
    # None value -> field omitted (same guard as disp_tail).
    none_val = format_metrics_line({**base, "recon_rel": None})
    assert "recon_rel:" not in none_val
