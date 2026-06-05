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
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (  # noqa: E402
    Hyperparameters,
    active_expert_fraction,
    effective_rank,
    fit_phi,
    kv_bytes_per_token,
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
