"""Iter 112 / H84: Orthogonal-expansion routing — Gram-matrix penalty.

Drop-in component for `train_gpt.py::SoftDenseRouter` when
`use_orthogonal_expansion_routing=True`.

# Hypothesis (from experiments/hypotheses.md H84)

Replace the current per-token soft routing with one that produces literal
**orthogonal per-token weight vectors across tokens**: enforce
    G = (1/N) · Σ_t w_t w_t^T  ≈  I/E
across N=B·T tokens in a batch. This forces the routing weight matrix
W = [w_1, …, w_N]^T (shape N×E) to have approximately orthogonal columns
(per-expert utilization vectors) when row-normalized.

# Math

Given routing weight matrix W ∈ R^(N × E):
    G = (1/N) · W^T · W ∈ R^(E × E)
    L_ortho_route = ||G - I/E||²_F

Properties:
- Uniform routing (all w_t = 1/E·1): G = (1/E²)·1·1^T (rank 1) — penalty large.
- One-hot balanced (each token uses 1 expert, N/E tokens per expert):
  G = (1/E)·I — penalty ZERO.
- The penalty thus jointly drives:
  (a) per-token sparsity (concentrated routing, each token few experts)
  (b) global balance (no expert dominates batch-wide).

# Implementation

The Gram matrix is computed over the FULL (B*T, E) routing weight tensor
WITHOUT renormalization (entmax/softmax outputs already in [0,1]). The
penalty is added to `router_reg_loss` with its own coefficient, ramped
from 0 over the first 30% of training (cold-start trap mitigation per
`feedback_anneal_sparsity_coefs.md`).

# Strict-generalization

`routing_gram_coef = 0` recovers iter 117b-1 baseline exactly. Promotion
under §11 standard rule (val_bpb-primary).

# Risks (per H84)

- Over-constraint: iter 99 sparsemax (similar over-constraint) cost
  +0.16 capacity. Mitigation: anneal from 0; small target coef (0.01).
- Interaction with existing entropy reg: both push toward sparsity, but
  along different axes (entropy = per-token; gram = joint per-token +
  global balance). May be redundant or synergistic — tested empirically.
- Routing depth interacts with DEQ: gram penalty is computed at
  SoftDenseRouter forward, propagated through the FP iteration via the
  loss term. Should not affect FP convergence directly.

# Integration into train_gpt.py (after smoke test)

Three touchpoints:

1. Add `Hyperparameters.use_orthogonal_expansion_routing = False`,
   `Hyperparameters.routing_gram_coef = 0.01`,
   `Hyperparameters.routing_gram_warmup_delay_frac = 0.3` fields.
2. Add `--use-orthogonal-expansion-routing` CLI flag.
3. In `SoftDenseRouter.forward`, after computing routing weights `p`,
   call `compute_gram_penalty(p)` and store as
   `self._gram_penalty_loss`. Then in `_collect_routing_losses`, fold
   it into `router_reg_loss` analogous to `_pertoken_entropy_loss`.

This file documents the design and provides the standalone helper
function. Smoke-test with `python experiments/components/orthogonal_expansion_routing.py`.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level toggle (set by main() before model construction)
# ---------------------------------------------------------------------------

_USE_GRAM_PENALTY: bool = False
_GRAM_COEF: float = 0.01
_GRAM_WARMUP_DELAY_FRAC: float = 0.3


def set_orthogonal_expansion_routing(
    enabled: bool,
    coef: float = 0.01,
    warmup_delay_frac: float = 0.3,
) -> None:
    """Module-level toggle invoked from train_gpt.py::main() at startup."""
    global _USE_GRAM_PENALTY, _GRAM_COEF, _GRAM_WARMUP_DELAY_FRAC
    _USE_GRAM_PENALTY = bool(enabled)
    _GRAM_COEF = float(coef)
    _GRAM_WARMUP_DELAY_FRAC = float(warmup_delay_frac)


def is_gram_penalty_enabled() -> bool:
    return _USE_GRAM_PENALTY


# ---------------------------------------------------------------------------
# Core penalty
# ---------------------------------------------------------------------------


def compute_gram_penalty(
    p: Tensor,
    coef_scale: Optional[Tensor] = None,
) -> Tensor:
    """Compute the Gram-matrix orthogonality penalty.

    Args:
        p: routing weights, shape (..., E). Reshapes leading dims into N
           tokens. Each row of W = p.reshape(N, E) is one token's routing
           weight vector.
        coef_scale: optional scalar tensor in [0, 1] for annealing
           (typically buffer set by training loop). If None, returns
           the raw penalty (caller multiplies by coef).

    Returns:
        Scalar tensor: ||G - I/E||²_F where G = (1/N) · W^T W.
    """
    if p.ndim < 2:
        raise ValueError(f"Expected p with at least 2 dims, got {p.shape}")
    E = p.shape[-1]
    W = p.reshape(-1, E).float()
    N = W.shape[0]
    # G = (1/N) W^T W
    G = (W.t() @ W) / max(N, 1)
    target = torch.eye(E, device=W.device, dtype=W.dtype) / E
    diff = G - target
    penalty = diff.pow(2).sum()
    if coef_scale is not None:
        penalty = penalty * coef_scale.to(penalty.dtype)
    return penalty


# ---------------------------------------------------------------------------
# Anneal helper
# ---------------------------------------------------------------------------


def anneal_scale(progress: float, warmup_delay_frac: float) -> float:
    """Linear ramp from 0 to 1 starting at `warmup_delay_frac`.

    Identical math to train_gpt.py's existing entropy-coef ramp. Returns:
        0                                   if progress < warmup_delay_frac
        (progress - delay) / (1 - delay)    otherwise (clamped to [0, 1])
    """
    if progress < warmup_delay_frac:
        return 0.0
    if warmup_delay_frac >= 1.0:
        return 0.0
    return min(max((progress - warmup_delay_frac) / (1.0 - warmup_delay_frac), 0.0), 1.0)


# ---------------------------------------------------------------------------
# Smoke test (CPU only — math correctness)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Verify penalty behaves correctly on canonical inputs."""
    print("orthogonal_expansion_routing.py smoke test:")
    torch.manual_seed(0)
    E = 8
    N = 64

    # Case 1: uniform routing (all w = 1/E). G = (1/E²)·1·1^T ≠ I/E. Large penalty.
    W_uniform = torch.full((N, E), 1.0 / E)
    p_uniform = compute_gram_penalty(W_uniform)
    print(f"  uniform routing (w=1/E):   penalty = {p_uniform.item():.6f}  (expect > 0)")
    assert p_uniform.item() > 0.0, "uniform should have nonzero penalty"

    # Case 2: balanced one-hot (N/E tokens per expert, each token uses one).
    # G_ee = 1/E, G_ef = 0 for e≠f. Penalty = 0.
    idx = torch.arange(N) % E
    W_onehot = torch.zeros(N, E)
    W_onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    p_onehot = compute_gram_penalty(W_onehot)
    print(f"  balanced one-hot:           penalty = {p_onehot.item():.6f}  (expect ≈ 0)")
    assert p_onehot.item() < 1e-10, f"balanced one-hot should give zero penalty, got {p_onehot.item()}"

    # Case 3: imbalanced one-hot (all tokens to expert 0). G_00 = 1, rest 0.
    # G - I/E = diag(1-1/E, -1/E, ..., -1/E) → penalty = (1-1/E)² + (E-1)·(1/E²)
    W_collapsed = torch.zeros(N, E)
    W_collapsed[:, 0] = 1.0
    p_collapsed = compute_gram_penalty(W_collapsed)
    expected_collapsed = (1.0 - 1.0 / E) ** 2 + (E - 1) * (1.0 / E) ** 2
    print(f"  collapsed (all → expert 0): penalty = {p_collapsed.item():.6f}  (expect ≈ {expected_collapsed:.6f})")
    assert abs(p_collapsed.item() - expected_collapsed) < 1e-6, "collapsed penalty mismatch"

    # Case 4: gradient flows through W
    W = torch.randn(N, E, requires_grad=True)
    p = compute_gram_penalty(W.softmax(dim=-1))
    p.backward()
    assert W.grad is not None and W.grad.abs().max().item() > 0, "gradient should flow"
    print(f"  gradient flow check:        max(|∂L/∂W|) = {W.grad.abs().max().item():.6f}  (expect > 0)")

    # Case 5: anneal helper
    assert anneal_scale(0.0, 0.3) == 0.0
    assert anneal_scale(0.3, 0.3) == 0.0  # Just at delay → 0
    assert abs(anneal_scale(0.65, 0.3) - 0.5) < 1e-9  # halfway through ramp
    assert anneal_scale(1.0, 0.3) == 1.0
    print(f"  anneal helper:              PASS")

    # Case 6: toggle setter
    assert not is_gram_penalty_enabled()
    set_orthogonal_expansion_routing(True, coef=0.05, warmup_delay_frac=0.4)
    assert is_gram_penalty_enabled()
    assert _GRAM_COEF == 0.05
    set_orthogonal_expansion_routing(False)
    assert not is_gram_penalty_enabled()
    print(f"  toggle setter:              PASS")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
