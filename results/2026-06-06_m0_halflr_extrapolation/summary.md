# 2026-06-06 — M0 half-LR DDP smoke (LR-promotion verification + depth extrapolation)

**Purpose:** verify the promoted (halved) LR defaults restore monotone training
on deep-horizon recurrent depth, and characterize depth-utility **beyond the
trained max K** (test-time extrapolation).

**Config:** GPUs 6,7 (DDP, `nproc_per_node=2`); 600 steps; `k_set={32,64}`
(`k_hi` sampled per step), `k_lo=8`; `--matrix-lr 0.01 --embed-lr 0.05
--scalar-lr 0.01` (promoted/halved); `--k-eval-sweep 4,8,16,32,64,128,256`
(128,256 are extrapolation > trained 64); `--depth-sweep-every 200`;
`damping=none`; `init_state=x0`; `recurrence_accum_dtype=float64`.
`params=9,666,338`. Evidence: [`run_metrics.txt`](./run_metrics.txt).

## Two-goal metrics — all improving over training

| metric | s200 → s600 / final | trend |
|---|---|---|
| `val_bpb` | 2.131 → **1.866** (final **1.828**) | ✅ monotone ↓ |
| `I_V`(K=64) usable info | 3.494 → **4.178** bits | ✅ rising at every K |
| `I_V`(K=4→64) in-range gain | 0.92 → **0.97** bits | ✅ depth-gain ↑ |
| `phi_eval` (eval-K proxy) | 0.095 → **0.116** | ✅ rising |
| `depth_gain_GT` | 0.258 → **0.324** | ✅ ↑ |
| `iv_total_bits` (K=4→256) | 0.619 → **0.797** | ✅ ↑ |
| `recon_rel` | **0.00e+00** throughout | ✅ exact (gradient-correctness gate) |
| `expert_util` | ~2.70 | ✅ healthy |
| `erank` | 636 → 550 | ✅ healthy (non-degenerate) |
| `peak_vram` | ~30,468 MB, flat in K | ✅ resource flat |
| `route_step_div` | ~0.034–0.067 | (MoE-basis unengaged — expected; shelved) |

## Depth extrapolation — `I_V(K)` final curve (bits)

| K | 4 | 8 | 16 | 32 | **64 (trained max)** | 128 (extrap) | 256 (extrap) |
|---|---|---|---|---|---|---|---|
| `I_V` | 3.20 | 3.76 | 4.04 | 4.15 | **4.17 (peak)** | 4.12 ↓ | 4.00 ↓ |

**Finding:** the curve rises monotonically to the trained max K=64, then *declines*
at K=128, 256 — the **finite-horizon signature** (depth-utility caps at the
trained horizon; no Huginn-style convergent test-time extrapolation, by design).
`expressiveness_rho=0.5357` (flat across checkpoints) reflects this: the full-range
Spearman includes the post-K=64 extrapolation dip, so it cannot reach 1.0 for a
finite-horizon model; **in-range (K≤64) the curve is monotone (ρ≈1.0).**

## Verdict

The LR promotion is verified: train_loss is monotone and **all two-goal eval
metrics improve over training** (val_bpb ↓, I_V/φ/depth-gain ↑ at all trained K,
recon_rel=0, resource flat). The one flat metric (`expressiveness_rho`, full-range)
is the by-design finite-horizon extrapolation cap, not a training regression. To
*use* deeper K, train deeper (extend `k_set`); test-time extrapolation would
require convergence/anytime readout (a separate design axis).
