"""
Simplified RevDEQ training script for Parameter Golf.
Single-file, self-contained submission artifact.

Architecture: RevDEQ + Soft Dense MoE + MLA + Gated Attention + FSQ/MoS + Diffusion-AR
"""

from __future__ import annotations

import contextlib
import glob
import io
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import warnings
import zlib
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

try:
    from experiments.plotting_hook import maybe_update_experiment_plots as _update_experiment_plots
except Exception:
    _update_experiment_plots = None

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
# donated_buffer left ENABLED (default). Lyapunov uses two-forward approach
# with retain_graph=False to avoid conflict with compiled donated buffers.

# ---------------------------------------------------------------------------
# DIAGNOSTICS CONTROL
# ---------------------------------------------------------------------------
_ROUTER_DIAGNOSTICS_ACTIVE = False
_ROUTER_DIAGNOSTICS_STEP: int | None = None
_DEQ_SOLVE_ACTIVE = False


# Iter 104 (AdaSplash α-entmax attention) DROPPED 2026-04-29 user directive
# after two systematic-debug rounds confirmed the upstream incompatibility
# between AdaSplash's `torch.autograd.Function` (line 936 of adasplash_block_mask.py)
# and torch.compile's AOTAutograd. Per PyTorch dev-discuss "Custom Ops
# Under torch.compile": "torch.autograd.Function is not the recommended one
# if you need torch.compile integration." Wrapping with custom_op + FakeTensor
# does not bypass the inner Function — both register, view-meta replay corrupts.
# Sliding-window attention (iter 104 v2) also dropped — global reach loss
# fundamentally hurts long-range modeling. NSA (iter 106, see H77-region in
# experiments/docs/hypotheses.md) is the principled forward path: torch.compile-native via
# flex_attention, preserves global reach via 3-branch (compression + selection
# + sliding), natively trainable from scratch.


def _unwrap_compiled_module(m: nn.Module) -> nn.Module:
    """Return the semantic owner behind compile/DDP-style wrappers.

    `torch.compile(nn.Module)` returns a wrapper (e.g., OptimizedModule) whose
    `forward` dispatches into the original module. DDP/DataParallel wrappers
    similarly store the underlying module on `.module`. Attribute writes
    against wrappers do not reliably affect the owner, so code that toggles
    module-local flags or eval knobs must write to the fully unwrapped module.
    """
    cur = m
    seen: set[int] = set()
    while isinstance(cur, nn.Module):
        cur_id = id(cur)
        if cur_id in seen:
            break
        seen.add(cur_id)

        orig = getattr(cur, "_orig_mod", None)
        if isinstance(orig, nn.Module) and orig is not cur:
            cur = orig
            continue

        wrapped = getattr(cur, "module", None)
        if isinstance(wrapped, nn.Module) and wrapped is not cur:
            cur = wrapped
            continue

        break
    return cur


def _iter_unique_routers(sb: nn.Module) -> Iterator["SoftDenseRouter"]:
    """Yield each routing module owned by `sb` exactly once, deduping by id().

    Under unified routing the attn and MLP routers are the same instance
    (one shared SoftDenseRouter); under chained routing each stage owns its
    own router. Schedulers, EMA updates, bias updates, and diagnostic
    materialization all need every distinct router visited once — this helper
    is the single source of that iteration so adding a new router-iter site
    cannot drift in dedup logic. Falls back to `sb.router` when `sb` has no
    `active_routers` (e.g., a wrapper or stub module).
    """
    if not hasattr(sb, "active_routers"):
        yield sb.router
        return
    seen: set[int] = set()
    for r in sb.active_routers():
        rid = id(r)
        if rid in seen:
            continue
        seen.add(rid)
        yield r


def _deterministic_token_window_start(seed: int, step: int, seqlen: int, max_tokens: int) -> int:
    """Deterministically choose a contiguous token window start for aux losses."""
    seqlen_i = max(int(seqlen), 0)
    if seqlen_i <= 0:
        return 0
    window = min(max(1, int(max_tokens)), seqlen_i)
    span = seqlen_i - window + 1
    if span <= 1:
        return 0
    mask = (1 << 64) - 1
    x = (int(seed) & mask) ^ ((int(step) * 0x9E3779B97F4A7C15) & mask) ^ 0xD1B54A32D192ED03
    x &= mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x = (x ^ (x >> 31)) & mask
    return int(x % span)


@contextlib.contextmanager
def _temporary_deq_k_override(model: nn.Module, k: int):
    base_m = _unwrap_compiled_module(model)
    missing = object()
    prev_k = getattr(base_m, "_deq_k_override", missing)
    base_m._deq_k_override = int(k)
    try:
        yield base_m
    finally:
        if prev_k is missing:
            try:
                delattr(base_m, "_deq_k_override")
            except AttributeError:
                pass
        else:
            base_m._deq_k_override = prev_k


def dynamo_disable(fn):
    try:
        import torch._dynamo as dynamo
        return dynamo.disable(fn)
    except Exception:
        return fn


@dynamo_disable
def _should_diag(training: bool) -> bool:
    """Return True if this rank should record diagnostics right now.

    Training: rank 0 only (master-logged, other ranks save the compute).
    Eval: ALL ranks — the post-eval DDP-global assertions need every rank to
    have populated diagnostics so dist.all_reduce() has matching participants.
    Rank-gating still happens at the log sites, not here.

    `@dynamo_disable`: this function reads `_ROUTER_DIAGNOSTICS_ACTIVE` (Python
    bool global) and `dist.get_rank()`. Without dynamo_disable, any call from a
    torch.compile region creates guards on these values that recompile when the
    flag toggles (at every diagnostic-emission boundary, ~every 10 train steps).
    Disabling makes the function opaque to dynamo: callers see a Python-bool
    return without internal guards. (Profile 2026-04-28.)
    """
    if training and not _ROUTER_DIAGNOSTICS_ACTIVE:
        return False
    if training:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    # Eval path: every rank populates local diagnostics.
    return True


def _safe_sum(values, floor: float = 1e-8) -> float:
    """sum(values) clamped to a positive floor to keep downstream divisions safe.

    Replaces the ``sum(values) or 1e-8`` pattern (which only triggers when the
    sum is exactly 0) with an explicit max(_, floor) so the intent — divide-by-
    zero protection in degenerate empty/all-zero cases — is the literal code.
    """
    return max(sum(values), floor)


def _safe_mean(values, floor: float = 1e-8) -> float:
    """sum(values)/len(values) clamped to a positive floor — companion to _safe_sum.

    Used by the routing-health diagnostics, where ``mu`` is the denominator of
    a CV computation and the surrounding `usage` list can be empty during a
    cold-start window.
    """
    n = len(values) if hasattr(values, "__len__") else 0
    return max(sum(values) / max(n, 1), floor)


def _share_cv(p: list[float]) -> float:
    """Coefficient-of-variation of a normalized share list.

    Companion to ``_safe_sum`` / ``_safe_mean``. The router-health log
    formatter computes this for `attn/mlp/pool` raw shares and again for
    EMA shares; keep the definition in one place.
    """
    mu = _safe_mean(p)
    var = sum((x - mu) ** 2 for x in p) / max(len(p), 1)
    return (var ** 0.5) / mu


def _share_entropy(p: list[float]) -> float:
    """Shannon entropy (nats) of a normalized share list.

    The +1e-8 floor inside ``log`` keeps the term finite for x=0 (the 0·log
    contribution is then 0, so no explicit guard is needed).
    """
    return -sum(x * math.log(x + 1e-8) for x in p)


def _diag_scalar(value) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# Single source of truth for Dirichlet-UCB router confidence diagnostics.
# Each entry: (log_field, router_attr, kdiag_short_name). Adding a new
# diagnostic is a one-line registry change — train logger, val logger,
# K-sweep table, plot parser, and SoftDenseRouter cache zeroing all
# iterate this tuple instead of duplicating the field list. Audit gate:
# CLAUDE.md "Sibling-fanout DRY gate".
ROUTER_DIRICHLET_DIAG_TERMS: tuple[tuple[str, str, str], ...] = (
    ("router_dir_strength_mean",    "_dirichlet_strength_last",         "dir_S"),
    ("router_dir_uncertainty_mass", "_dirichlet_uncertainty_mass_last", "dir_U"),
    ("router_dir_sigma_mean",       "_dirichlet_uncertainty_last",      "dir_sigma"),
    ("router_dir_evidence_mean",    "_dirichlet_evidence_mean_last",    "dir_evid"),
    ("router_dir_mu_entropy_norm",  "_dirichlet_mu_entropy_norm_last",  "dir_Hmu"),
)
# Beta is sourced from a different router attribute (`_dirichlet_ucb_beta`)
# and is per-config rather than per-token; tracked separately so that the
# main registry stays homogeneous.
ROUTER_DIRICHLET_BETA_TERM: tuple[str, str, str] = (
    "router_ucb_beta_current", "_dirichlet_ucb_beta", "ucb_beta",
)


def _router_confidence_stats(
    routers,
    *,
    step: int | None = None,
    require_step_match: bool = False,
) -> dict[str, float]:
    """Aggregate Dirichlet-router confidence diagnostics across unique routers."""
    acc: dict[str, list[float]] = {name: [] for name, _, _ in ROUTER_DIRICHLET_DIAG_TERMS}
    beta_name, beta_attr, _ = ROUTER_DIRICHLET_BETA_TERM
    beta_vals: list[float] = []
    for router in routers:
        if router is None or getattr(router, "scoring", None) != "dirichlet_ucb":
            continue
        if require_step_match and getattr(router, "_dirichlet_diag_step", None) != step:
            continue
        for name, attr, _ in ROUTER_DIRICHLET_DIAG_TERMS:
            v = _diag_scalar(getattr(router, attr, None))
            if v is not None:
                acc[name].append(v)
        beta_v = _diag_scalar(getattr(router, beta_attr, None))
        if beta_v is not None:
            beta_vals.append(beta_v)
    out = {name: sum(vals) / len(vals) for name, vals in acc.items() if vals}
    if beta_vals:
        out[beta_name] = sum(beta_vals) / len(beta_vals)
    return out


def _format_router_confidence_parts(
    routers,
    *,
    step: int | None = None,
    require_step_match: bool = False,
) -> list[str]:
    stats = _router_confidence_stats(routers, step=step, require_step_match=require_step_match)
    order = tuple(name for name, _, _ in ROUTER_DIRICHLET_DIAG_TERMS) + (
        ROUTER_DIRICHLET_BETA_TERM[0],
    )
    return [f"{name}:{stats[name]:.4f}" for name in order if name in stats]


@contextlib.contextmanager
def router_diagnostics(enabled: bool = True, *, step_tag: int | None = None):
    global _ROUTER_DIAGNOSTICS_ACTIVE, _ROUTER_DIAGNOSTICS_STEP
    prev, prev_step = _ROUTER_DIAGNOSTICS_ACTIVE, _ROUTER_DIAGNOSTICS_STEP
    _ROUTER_DIAGNOSTICS_ACTIVE = bool(enabled)
    _ROUTER_DIAGNOSTICS_STEP = step_tag if enabled else None
    try:
        yield
    finally:
        _ROUTER_DIAGNOSTICS_ACTIVE = prev
        _ROUTER_DIAGNOSTICS_STEP = prev_step


# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

class Hyperparameters:
    config_profile = "fast_default"
    eval_profile = "diagnostic"
    # `advisory` keeps the iter142 BPB-primary policy: any successful full
    # validation produces `run_valid=True` regardless of health gates. `hard`
    # makes `run_valid` follow `health_valid`, so health failures block
    # promotion even when val_bpb improved.
    diagnostic_gate_policy = "advisory"

    data_path = "./data/datasets/fineweb10B_sp1024"
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id = ""
    seed = 42

    val_batch_size = 524_288
    # Throughput-only eval cap on per-forward chunking (NOT eval coverage —
    # `eval_batch_seqs` controls coverage). Calibrated for the dev
    # `world_size=2, grad_accum=4` profile where the derived local train
    # microbatch is 32, so 48 ≈ 1.5× train. The literal 48 may not be
    # 1.5× on submission hardware (`world_size=8`); revisit if eval-time
    # VRAM headroom changes. 0 ⇒ derive from the training microbatch
    # (see `_resolve_val_micro_batch_seqs` ~ line 1147).
    val_micro_batch_seqs = 48
    val_loss_every = 200
    # iter171 (2026-05-16): mini K-sweep at every fast-val event. When set,
    # after the default-K val_bpb is computed, the trainer re-runs validation
    # on the same fast-val subset at each additional K in this tuple and emits
    # `fast_k_sweep:k4=X,k16=Y,k128=Z` to the log. Diagnostic ONLY — lets us
    # detect degenerate K-sweep patterns (e.g. K=4 < K=16 < K=128, a signature
    # of consistency-loss over-pressure causing F to become flat in z) at every
    # 200-step val instead of waiting for end-of-training K-sweep. Default
    # empty = feature disabled. Recommended for iter171: (4, 128) — adds ~65s
    # per val event (~+1 % wallclock per 1000-step run). The default K (=16)
    # is automatically excluded if listed.
    fast_val_k_sweep_set: tuple[int, ...] = ()
    train_log_every = 10  # log every 10 steps (~85s at 8.5s/step) for better progress visibility
    auto_plot_on_val = True
    # Fixed-point spectral probe: rho_F = |lambda_max(J_F)| at the saved DEQ
    # FP, via power iteration on J_F directly (NOT J_F^T J_F). rho_F < 1 is
    # necessary AND sufficient for asymptotic local convergence per
    # Hartman-Grobman; this is the principled gate metric. lip_ub_T/S/F
    # (operator-norm proxies) were removed 2026-05-15 — over-restrictive
    # for non-symmetric J_F (iter152 has rho ≈ 0.85 yet sigma_max ≈ 17).
    fp_rho_power_iters = 8
    checkpoint_dir = "experiments/checkpoints"
    checkpoint_every = 200
    checkpoint_keep = 2
    resume_from = ""
    resume_latest = False

    iterations = 1000  # default training budget: step-count-governed, DDP with all GPUs; submission runs override via --max-training-seconds=600
    warmdown_frac = 0.72  # fraction of total steps for warmdown
    train_batch_tokens = 524_288
    train_seq_len = 2048
    # `max_training_seconds` is the total *process wallclock* budget (the
    # value submitters pass on the command line). The training loop runs
    # for at most max_training_seconds - eval_reservation_seconds so that
    # post-loop val + int6 + K-sweep + sliding val fit under the same
    # process wallclock. Project invariant: submission must fit 600 s
    # 8×H100. See `_compute_training_budget_ms` and audit gate
    # `EXPERIENCE.md#scalar-semantic-shift`.
    max_training_seconds = 0
    max_wallclock_seconds = 0  # deprecated one-cycle alias for max_training_seconds
    eval_reservation_seconds = 120
    # Promotion/submission metadata must use the full validation split. Fast
    # validation remains useful for step logs and smoke loops, but a final
    # `val_bpb` proxy is not promotable.
    final_full_validation = True
    # Iter 98b (2026-04-28) NOT PROMOTED ✗ (closed 2026-04-29). The
    # micro-batch-halving rescue successfully fit D=1024 (peak VRAM 23 GiB
    # vs iter 98's 47 GiB OOM), but val_bpb int6 = 1.5018 vs iter 100b
    # 1.4893 (Δ +0.0125 regression) AND step_avg 25.1s vs 24.5s (+2.6%
    # slower per-wallclock). Refinement at D=1024 amplified to +70% step
    # cost (vs iter 100b's +6%) — the activation-memory pressure during
    # the extra DEQ pass dominates. D=1024 deferred to 8× H100 submission
    # hardware where activation memory isn't the binding constraint.
    # Field retained as default=1 (no behavior change) for future use; can
    # be enabled via CLI for any iter that wants halved micro-batch.
    grad_accum_multiplier = 1

    # Model architecture
    vocab_size = 1024
    num_layers = 12  # DEQ solver max K
    # 2026-04-29 user directive: disable refinement by DEFAULT after iter 98b
    # showed refinement adds +6% step cost at D=768 (iter 100b) → +70% at
    # D=1024 (iter 98b) — the 85% ramp activates a full extra DEQ pass that
    # becomes a major cost driver and the val_bpb benefit is unverified
    # (never directly ablated). iter 110 (queued) tests re-enable on top of
    # the current baseline as a clean ablation. Set to 1 + ramp_frac<1 to
    # restore the prior diffusion-AR refinement loop.
    num_refinements = 0
    num_refinements_ramp_frac = 0.85  # only matters if num_refinements > 0
    num_kv_heads = 4
    # Iter 96 baseline restored (iter 98b D=1024 NOT PROMOTED, see
    # grad_accum_multiplier docstring + H73). D-scaling deferred to 8× H100.
    model_dim = 768
    num_heads = 8
    num_experts = 16  # iter 96 baseline (PROMOTED ★, H71): 8 → 16 paired with attn/mlp_expert_rank halving. Iter 97 (E=20) NOT PROMOTED on per-wallclock grounds; H72 documents axis saturation past E=16 / R=64 on D=768.
    # Shared expert disabled by default. The DeepSeek-style always-on
    # bypass-routing expert (Phase 9 iter 51) remains a code path but
    # is opt-in: every routed expert is full-D LoRA-style under the
    # iter146 rescue stack, and a separately-parameterized always-on
    # expert is redundant when the routing-EMA balance + alive-hinge
    # already prevent expert collapse. To re-enable for a specific
    # ablation, pass --num-shared-experts=1.
    num_shared_experts = 0
    # Iter 94 (2026-04-24): disable CTP head entirely. When False, MoS head only
    # emits NTP log-probs; CTP param banks (gate_ctp, A_ctp_shared, A_ctp,
    # B_denoise, ctp_*_norm_weight) are not allocated, CTP loss is skipped, and
    # the refinement soft-embedding mix uses p_ntp only. Tests whether the
    # dual-head denoising gradient is still load-bearing under iter 66b's
    # Parcae + learnable-norm landscape.
    use_ctp = False
    # Direct coefficient on ctp_loss (default 0 since use_ctp=False and refinements=0).
    # Reach back into the refinement-tied schedule (0.05 * num_refinements * refine_strength)
    # only if a future iter wires refinement back on; until then this knob is the
    # single source of CTP gradient strength.
    ctp_weight = 0.0
    mlp_mult = 3.0
    tie_embeddings = True
    rope_base = 10000.0
    qk_gain_init = 5.0
    deq_beta = 0.50  # Phase 9: β=0.7 caused high recon_err (1.2 vs 0.88) → RevDEQ unstable. Keep 0.5.
    deq_beta_jitter = True   # Phase 9 iter 49: sample β from {0.3, 0.5, 0.7} per step (like K-jitter)
    deq_beta_jitter_set = (0.3, 0.5, 0.7)  # β values to sample from
    # Phase 9 iter 66a: Parcae per-dim learned damping (arXiv:2604.12946).
    # Replaces scalar β with per-dim Ā = exp(Δ·(-exp(log_a))) ∈ (0,1), β = 1-Ā.
    # When active, supersedes deq_beta / deq_beta_jitter.
    use_parcae = True
    parcae_init_a_bar = 0.7  # initial Ā per dim (0.7 → β=0.3, tested sweet spot)
    # B̄₀ at step 0; defaults to 1 − Ā₀ for continuity with iter 66a tied β = 1 − Ā.
    # Promoted to Hyperparameters per §9 single-source-of-truth (was derived inline
    # in _parcae_init_raw_values).
    parcae_init_b_bar = 0.3
    parcae_lr = 0.002  # 10× slower than scalar_lr — Parcae params control DEQ mixing
    # iter 117 v2 (H87): same precedent as parcae_lr. The blend_logit
    # controls how much entmax-1.5 sparsity contributes to routing — drift
    # too fast and routing collapses (NaN at iter 117 v1 step 60). 10×
    # slower LR keeps drift bounded once anneal ramps.
    entmax_blend_lr = 0.002

    # Iter 106 (NSA — Native Sparse Attention; arxiv:2502.11089). Three-branch
    # hybrid that preserves O(T) reachability while remaining sparse:
    #   compression — mean-pool K/V over fixed-size sliding blocks, attend to
    #     the downsampled stream (global, coarse).
    #   selection — top-K block selection per query (sparse, precise).
    #     DEFERRED to iter 106b: `nsa_num_selected_blocks=0` disables it.
    #   sliding-window — last W tokens (local recency).
    # Mixed via per-expert-per-head learnable softmax gate. Strict-generalization:
    # `nsa_compress_block_size=1, nsa_compress_block_sliding_stride=1,
    # nsa_sliding_window_size=T, nsa_branch_gate_init=0` recovers full causal
    # SDPA exactly. See H86 in experiments/docs/hypotheses.md for the design spec.
    use_nsa_attention = False
    nsa_compress_block_size = 32
    nsa_compress_block_sliding_stride = 16
    nsa_selection_block_size = 64
    nsa_num_selected_blocks = 0  # 0 disables selection branch (iter 106 = 2-branch); 4 enables it (iter 106b)
    nsa_sliding_window_size = 256
    nsa_branch_gate_init = 0.0   # zero → softmax uniform mix at init

    # Optimizer
    tied_embed_lr = 0.03
    embed_lr = 0.6
    matrix_lr = 0.022
    scalar_lr = 0.02
    muon_momentum = 0.99
    # iter 121b PE-NS DEFAULT (2026-04-30 user clarification, supersedes
    # 2026-04-29 diagnosis): PE-NS NaN root cause was the iter 111 H83
    # variance regularizer (`routing_variance_coef = -λ · Σ_e Var_token(w(e|t))`),
    # NOT PE-NS itself. Diagnostic chain:
    #   iter 117 v1: stock NS @ 5 + variance reg + entropy reg + entmax → NaN s60
    #   iter 117 v2: PE-NS @ 7 + variance reg + entropy reg + entmax → NaN s30
    #     (earlier — PE-NS amplifies the variance-reg gradient cascade, but
    #      doesn't originate it)
    #   iter 117 v3: REMOVE variance reg, KEEP everything else → clean run
    #   iter 117 v5 (PROMOTED): same v3 config, ran 1000 steps clean
    # The variance regularizer + entmax produce exact-zero routing weights,
    # whose Var_token gradient is undefined and amplifies into a NaN cascade.
    # Removing variance reg removes the cascade source; PE-NS @ 7 is fine.
    # PE-NS preserved as default ON @ steps=7 (empirical elbow). The
    # `use_polar_express_ns` flag is kept for future runs that need stock NS
    # for clean A/B comparison via `--use-polar-express-ns=0`.
    muon_backend_steps = 7
    use_polar_express_ns = True
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 800  # absolute cap; effective = min(this, frac × iterations)
    # Adaptive cap: shorter runs auto-scale the ramp so we don't spend training
    # at the start-of-warmup momentum the entire short trajectory. 0.8 = the
    # historical 800/1000 ratio at the canonical 1000-iter run.
    muon_momentum_warmup_frac = 0.8
    beta1 = 0.85
    beta2 = 0.90
    adam_eps = 1e-8
    grad_clip_norm = 1.0  # iter 73: relax from 0.3 — Lyapunov provides soft contraction (H56)
    weight_decay = 0.015  # iter 145r PROMOTED_WITH_TECH_DEBT (2026-05-08): 0.01 -> 0.015. Improved BPB with Dirichlet-UCB/EMA routing, but did not solve the transition-Jacobian contraction gate; future fixes should target T_theta directly, not scalar Parcae beta.
    tied_embed_init_std = 0.005

    # Routing
    # Slow generic router-bias controller.  Default stays off for historical
    # parity, but when diagnostics identify a routed min-share root cause this
    # controller is the preferred first fix over adding per-component losses.
    router_bias_update = False
    router_bias_lr = 0.10
    router_bias_clip = 10.0
    # iter 145r PROMOTED_WITH_TECH_DEBT: evidential Dirichlet-UCB routing
    # replaces linear point-estimate routing. It improved BPB and liveness, but
    # did not enforce the strict post-int min-share/orthogonality/contraction
    # gates; those remain documented future-iteration targets.
    router_scoring = "dirichlet_ucb"
    # scoring="dirichlet_ucb" interprets
    # router logits as Softplus evidence, forms Dirichlet alpha=e+1, and routes
    # by the expected probability plus an annealed local Beta-variance UCB bonus.
    router_dirichlet_ucb_beta = 0.5
    # iter 145r removes the routed sigmoid gate so every token must use the
    # routed expert mixture rather than suppressing it through an extra gate.
    use_router_sigmoid_gate = False
    # EMA-anchored balance/liveness/specialization losses. EMA is persistent
    # state; the KL-balance term uses a straight-through EMA anchor so it can
    # train current routing while the forward value tracks historical usage.
    # Exact detached KL(EMA||U) would only be a diagnostic. The hard min-share
    # issue is not solved by KL alone, so the alive hinge below directly
    # matches the no-underuse gate.
    # Small strict liveness hinge: EMA balance improves average usage, but the
    # post-int min-share gate fails on worst under-use. This term directly
    # targets the same deficit without changing token-local routing.
    router_ema_alive_coef = 0.02
    # Coefficient-first rescue: double the promoted EMA terms to push closer to
    # ideal long-run balance and stronger token specialization.
    router_ema_balance_coef = 0.30
    router_ema_specialization_coef = 0.20
    # iter153 (provisional default-on, pending iter result): use reverse KL
    # `KL(U || EMA)` instead of forward `KL(EMA || U)` for the balance term.
    # Forward KL weights each per-expert term by EMA_i, so a dead expert at
    # tiny EMA_i contributes a tiny loss term even when log(EMA_i / U_i) is
    # large; the gradient on the tail vanishes. Reverse KL weights by U_i,
    # giving each expert an equal weight regardless of mass, and the gradient
    # on EMA_i diverges as EMA_i → 0 — exactly the asymmetry the min-share
    # gate needs. Revert to False if iter153 fails.
    use_reverse_kl_balance = True
    # iter 100b (2026-04-27): per-token entropy penalty with ANNEALED schedule
    # + min_share_loss decoupled. iter 100 (entropy_coef=0.02 from step 0)
    # hit train_loss instability — penalty fights min_share_loss penalty
    # (one pushes uniform, other pushes peaky). Decouple them:
    #   1. Lower coef target: 0.02 → 0.005 (4× weaker)
    #   2. Anneal from 0 over first warmup_frac of training (cold-start trap
    #      avoidance, same principle as α-annealing for iter 102).
    #   3. Drop min_share_loss penalty; keep min_share as DIAGNOSTIC metric only.
    # Per-token entropy MINIMIZATION (sign: +coef·H_pertoken added to total
    # loss, so the optimizer drives H_pertoken → 0 → per-token specialization).
    # CV loss operates on the orthogonal axis of global cross-batch balance
    # (prevents dead experts).
    #
    # iter 117b-1 RESULT (2026-04-30): tested 10× bump 0.005 → 0.05.
    # NOT PROMOTED — int6 +0.0007 vs iter 117 v5, hypothesis REFUTED:
    # in soft-dense MoE, the CV-redistribution dominates over the per-token
    # sparsity push at any reasonable entropy coef. Pertoken_entropy
    # stabilizes around 2.6 regardless. See H87b RESULT in experiments/docs/hypotheses.md.
    # iter 145r promoted a lower, warm-started regularization stack around the
    # evidential router. This supersedes the 2026-05-06 uniform-1.0 stack for
    # new default runs, while keeping the post-int health failures documented.
    # Direct coefficients in the flat regularization assembly. The full
    # objective is:
    #   loss = ntp + ctp_weight · ctp
    #        + router_pertoken_entropy_coef · Σ_r H_pertoken(r)
    #        + expert_output_diversity_coef · diversity(expert outputs)
    #        + mos_output_diversity_coef    · diversity(MoS low-rank states)
    # With the promoted default `regularizer_warmup_frac=0.07`, every
    # router/MoS/diversity regularizer ramps from 0 to its target over the
    # first 7% of optimizer steps. CV-as-loss removed entirely 2026-05-15:
    # router CV was 0.0 since iter146 (subsumed by EMA-anchored balance),
    # MoS CV contribution at iter163 step 1000 was 0.000242 (0.08% of
    # router_reg_loss = 0.31) — balance maintained for free by the MoS
    # routing softmax + small head count, no measurable loss-side pressure
    # needed. CV (router + MoS) remains computed and emitted as diagnostic
    # tensors `_router_cv_loss_t` / `_mos_cv_loss_t` and `attn_cv`/`mlp_cv`/
    # `pool_cv` step-log fields, but does not contribute to the loss.
    router_pertoken_entropy_coef = 0.1
    # Per-token expert-OUTPUT diversity (replaces block_ortho_aux + iter 141 expert_gram).
    # `expert_diversity_kind`:
    #   cosine    (default): Y' = normalize(Y); loss =
    #             mean_token max_pair(off_diag(Y' Y'^T)²). Scale-invariant
    #             worst-pair pressure (per-token argmax over off-diagonal
    #             pairs, then mean over tokens) — penalizes direction only
    #             at the most antagonistic expert pair, which targets the
    #             post-int orthogonality failure mode directly. CV handles
    #             usage and the optimizer handles norm (separation of
    #             concerns).
    #   frobenius: G = (Y Y^T)/D; loss = E_t[‖G − I/E‖²_F] / E². Couples
    #             direction AND per-expert norm; bundles a norm-target onto
    #             the diversity penalty. The /E² is for magnitude parity
    #             with cosine — without it, "same coef" means ~30× more
    #             gradient pressure than cosine.
    expert_diversity_kind = "cosine"
    # iter 145r promoted a weaker cosine expert-output diversity coefficient.
    # Final post-int gates still flagged attn/mlp output collinearity, so a
    # future fix should strengthen direction pressure or add a better
    # transition-output scale/Jacobian control rather than hiding the issue.
    # Root-cause rescue: 0.15 -> 0.30 and cosine loss uses worst-pair pressure
    # so training targets the post-int orthogonality failure mode directly.
    expert_output_diversity_coef = 0.30
    expert_diversity_every = 8       # cadence: aux fires every N optimizer steps
    expert_diversity_max_tokens = 64
    mos_output_diversity_coef = 0.0  # per-token Gram on MoS low-rank states; default off
    # iter 145r: 0.07 warmup avoided cold-start over-regularization while
    # preserving the BPB win.
    regularizer_warmup_frac = 0.07
    # iter 129 / H99 (NEW 2026-05-04): SmearGate — position-mixing memory channel.
    # `x[t] += g · x[t-1] · not_bos_mask` after embedding lookup. BOS-fixed
    # (mask suppresses leak across packed-doc boundaries; SP BOS_ID=1).
    # Default OFF; strict-gen at coef=0 → exact recovery (no x-mixing).
    use_smear_gate = False
    smear_gate_init = 0.0
    smear_gate_window = 12
    smear_gate_bos_id = 1  # SentencePiece BOS
    # Remaining-queue component toggles (2026-05-12): every promoted component
    # below is default-off and removable. Training smokes turn on one flag at a
    # time so failed iterations can be dropped by deleting the component file
    # plus its narrow hook.
    use_sparse_attn_head_gate = False
    sparse_attn_gate_window = 12
    sparse_attn_gate_scale = 1.0
    sparse_attn_gate_factor = 2.0
    sparse_attn_gate_init_std = 0.0
    use_rr_attention = False
    rr_stride = 8
    rr_block_size = 64
    rr_tau = 0.95
    use_ttt_eval = False
    use_gptq = False
    use_lqer = False
    use_grouped_artifact_compression = False
    use_caseops = False
    # iter 122 / H93 (2026-05-01): logit softcap (Gemma2-style).
    # `logits = softcap * tanh(logits / softcap)` bounds extreme logit values,
    # smoothing gradient spikes and reducing bf16 numerical issues. Applied
    # in `MoSHead._head_forward` to per-expert logits BEFORE log_softmax — the
    # tanh saturates per-token-per-expert, the convex log-softmax mixture is
    # unaffected by an additive shift but the shape of the logit cloud is
    # reshaped (extreme positives/negatives saturate). Default 0 = disabled
    # (strict-gen recovery: tanh(x)*softcap → x as softcap→∞; we treat 0 as
    # the "off" sentinel via early-return). Records use 30.0 since 2026-04+;
    # archived smoke test in `experiments/components/archive/logit_softcap.py` (6/6 PASS).
    logit_softcap = 0.0
    # iter 117 v3 (2026-04-29): routing-variance penalty REMOVED. iter 111 H83
    # introduced `routing_variance_coef = -λ · sum_e Var_token(w(e|t))` to break
    # the symmetric trap of per-token entropy at uniform routing. iter 117 v1/v2
    # NaN'd at s60/s30 with variance active (combined with entmax-1.5 blend);
    # iter 117 v3 with variance REMOVED + entropy KEPT crossed both NaN points
    # cleanly with BETTER metrics than v1 (s40 train_loss 4.79 vs v1 5.04, cv
    # 0.35 vs 0.67). Conclusion: variance + entmax is mutually amplifying (both
    # push toward concentration; entmax produces exact zeros that variance
    # gradient amplifies into cascade). Entropy + entmax is mutually balancing.
    # Variance regularization removed per user direction; entropy regularizer
    # alone is sufficient under the entmax-blend regime.
    # iter 117a (H87 Phase 1): entmax-α routing replacement. When True, the
    # post-softmax routing weights are computed via entmax-α (closed-form
    # sort-and-threshold) instead of softmax. α is a learnable scalar per
    # router with `α = 1.0 + softplus(alpha_logit)`; init `alpha_logit = −7.0`
    # gives `softplus(−7) ≈ 9e-4` → `α ≈ 1.0009` ≈ softmax (strict-gen at
    # init). Sparsity emerges as gradient pushes alpha_logit upward (variance
    # penalty from iter 111 provides the gradient signal).
    # Phase 1 (117a): KEEP fused BMM, no compute-skip yet — this is the
    # cheap val_bpb signal for whether sparse routing helps. Path D
    # capacity-padded dispatch (true compute-skip, ~4-15 hr refactor) is
    # iter 117b conditional on 117a's val_bpb preservation.
    use_entmax_routing = False
    # Init the blend logit to +5 so `sigmoid(5) ≈ 0.9933` → at step 0
    # routing is ≈ pure softmax (strict-gen recovery within 1% in bf16).
    # The blend is annealed in over the first
    # `entmax_blend_warmup_delay_frac=0.3` of training; once anneal ramps to
    # 1.0 the gradient drives the LOGIT DOWN if entmax-1.5's exact-zeros
    # routing benefits val_bpb.
    entmax_blend_init_logit = 5.0
    # iter 117b-2 (2026-04-30): Triton-fused entmax-1.5 kernel. Default OFF
    # (pure-PyTorch closed form via train_gpt.py::entmax_1p5). When True AND
    # Triton is available, SoftDenseRouter's entmax path uses the custom_op
    # registered Triton kernel (forward + closed-form backward, verified
    # against deep-spin/entmax reference at fp32 noise floor). Throughput-
    # only change; numerical equivalence within bf16 floor is required for
    # promotion. Smoke test under --use-entmax-triton=1 must pass before
    # full training launch (custom_op + DDP + compile + RevDEQ is fragile;
    # see CLAUDE.md Fix #2 NOT VIABLE for the AdaSplash precedent).
    use_entmax_triton = False
    # iter 117b-3 (2026-04-30): capacity-padded sparse MoE dispatch scaffold.
    # Default OFF and rejected by startup validation for training: hard top-k
    # capacity dispatch is discrete and not RevDEQ-safe inside T_theta. Keep
    # helper tests as archived kernel evidence, not an active train-time flag.
    use_sparse_dispatch = False
    sparse_dispatch_capacity_factor = 4.0
    # iter 118a Phase A3 (2026-05-03): fused Triton routed-down kernel for
    # MLP-down dispatch. Default OFF; enable with --use-unified-routed-down=1.
    # When True AND torch.is_grad_enabled() is False, MLP.mix_experts calls
    # `fused_routed_down(h_pre, w_combined, expert_down)` instead of the eager
    # path. Kernel bench: 2.86× dense, 7.36× at 87.5% sparse (per H101 ε ≤ ε_bf16).
    # Training (grad-enabled) and TBPTT window unaffected (eager path).
    use_unified_routed_down = False
    # iter 103 / H77 (2026-04-30): chained 2-stage pooled routing.
    # Default OFF (single-stage routing as in iter 117 v5 / iter 117b-1).
    # When True, Block uses TWO sequentially-chained SoftDenseRouter
    # instances inside T_θ. Each stage drives `num_routed/2` attn experts
    # + `num_routed/2` mlp experts (independent params, no sharing across
    # stages). Stage 1 output is used as input to stage 2; final block
    # residual = T_2(T_1(z, x_0), x_0) + Parcae input injection.
    # Skip-connection across the chain: out = stage_1_out + stage_2_out
    # (preserves iter 100b strict-gen path when stage 2 zero-init).
    # See experiments/docs/hypotheses.md H77 for full spec + design questions.
    # Step 1 (this commit): Hyperparameter + CLI only — Block refactor
    # lands in step 2.
    use_chained_routing = False
    # iter 103 / H77 active component switch. "none" keeps the current unified
    # Block path with no extra registered params. Named presets are implemented
    # in experiments/components/chained_routing.py:
    # split_2stage, attn_first_2stage, mlp_first_2stage, split_4stage.
    chained_stages_preset = "none"
    # iter 117 v2 (post-NaN rescue 2026-04-29): the entmax blend itself is
    # ANNEALED from pure softmax (anneal=0 → blend forced to 1.0 = softmax)
    # to learnable (anneal=1 → blend = sigmoid(blend_logit)) over training.
    # Without this anneal, even a 0.7% entmax contribution at init produced
    # exact-zero routing for low-score experts, which combined with no entropy
    # cushion let CV concentration cascade to NaN at step 60. Strict-gen at
    # step 0 is now EXACT (anneal=0 → pure softmax = iter 100b forward map).
    # Mirrors the entropy/variance warmup_delay_frac=0.3 pattern.
    entmax_blend_warmup_delay_frac = 0.3

    # iter147/iter155 path: finite-perturbation expansion penalty. Per-call
    # form is `expansion = ‖M(z+ε·u) − M(z)‖_RMS / ε` with `u` a unit-RMS
    # random direction; the Hutchinson expectation is ‖J_M‖_F/√D, NOT operator
    # norm. The principled spectral signal is `rho_F` (Hartman-Grobman:
    # rho(J_F) < 1 is necessary AND sufficient for asymptotic local
    # convergence); the soft-Lyapunov FD penalty is a refuted lower-bound
    # proxy retained only as a default-off ablation path.
    # `M` is selected by `lyapunov_target`:
    #   - "transition_T" (iter147 default): M = T_θ(z, x₀); penalizes ‖J_T·u‖.
    #     Decomposition-only — T is one component of the iterated map.
    #   - "iteration_S"  (iter155 first attempt): M = S(z, x₀) = Ā·z+(1−Ā)·T_θ;
    #      penalizes ‖J_S·u‖.  Single-state convex blend — NOT what the solver
    #      iterates (the solver is two-state).  Advisory surrogate only.
    #   - "iteration_F"  (iter155 corrected): M = F(y, z) =
    #      (Ā·y + β·T(z), Ā·z + β·T(Ā·y + β·T(z))) with β = 1−Ā.  This is
    #      the actual Parcae cycle the solver iterates; penalizes
    #      ‖J_F·(u_y, u_z)‖ — empirical pressure toward contraction, not a
    #      formal certificate. iter155 refuted this path empirically.
    # Kept default-off; the principled FP-convergence mechanism is now the
    # iter163 multi-K consistency loss (which *learns* rho(J_F) < 1 naturally
    # by anchoring each TBPTT-K to the converged-FP target).
    lyapunov_coef = 0.0        # λ_jac: weight of relu(expansion - gamma)^2
    # γ is on the Frobenius/√D proxy (Hutchinson), not ‖J‖_2: ‖J‖_2 < 1
    # implies Frobenius/√D < 1 but not the reverse, so γ=0.97 is a soft
    # directional proxy. The lip_ub_T/S/F operator-norm probes that
    # historically accompanied this penalty were removed 2026-05-15 as
    # over-restrictive — for non-symmetric J_F the σ_max can be orders of
    # magnitude above ρ (iter152: σ_max ≈ 17, ρ ≈ 0.85, iter_conv_rel ≈ 0.02).
    # Pure IFT was removed after iter144; future contraction work should change
    # the transition parameterization or use a new hybrid finite-K design.
    lyapunov_gamma = 0.97
    lyapunov_every = 16
    lyapunov_max_tokens = 64
    # iter155 (corrected): which Jacobian the Lyapunov FD probe penalizes.
    # Default "transition_T" preserves iter147 behavior.  "iteration_S" routes
    # the FD probe to the single-state convex blend (advisory only — not the
    # iterated map); "iteration_F" routes to the actual two-state Parcae
    # cycle map.  The S/F branches do NOT detach Ā so gradients flow to Parcae
    # damping; B̄ remains detached because it appears inside the FD subtraction
    # and would carry second-order curvature otherwise.
    lyapunov_target = "transition_T"
    # iter155 (corrected): which direction the FD probe perturbs.  Currently
    # validator accepts only "random_fd" — the `power_jvp_F` worst-direction
    # branch was removed 2026-05-15 with the underlying lip_ub_* probes.
    lyapunov_estimator = "random_fd"
    # Phase 9 iter 55: Denoising regularization (HyDRA 2026, Efficient DEQ 2025).
    # ||f(z*+ε, x0) - z*||² penalizes contraction failure at finite perturbation.
    # Complements Hutchinson (which penalizes ||J||²_F at infinitesimal scale).
    # Phase 9 iter 89 (2026-04-25): HyDRA denoising disabled (0.01 → 0).
    # Same hypothesis as iter 88 for the finite-perturbation contraction probe:
    # Parcae-style damping improves solver reversibility/stability, but does not
    # certify contraction of the full nonlinear block; contraction remains
    # empirically monitored. If the probe is redundant, denoising contributes
    # only noise + one extra block forward per step.  Code path retained
    # (commented-out future cleanup permitted per user directive 2026-04-25).
    denoising_coef = 0.0       # weight of denoising loss (iter 89: disabled)
    denoising_noise_std = 0.01 # σ: Gaussian noise scale added to z*

    # DEQ solver: RevDEQFunction with fp64 accumulators (O(1) memory). Enables
    # torch.compile(shared_block) for fused permute+rms_norm kernels. The legacy
    # "unroll"/"autograd" backward modes were removed (user directive 2026-04-28)
    # — only revdeq is supported. Memory: 4.4 GB peak vs unroll's 43.7 GB at K=12.
    # iter 28-tbptt: Truncated BPTT. Backward reconstructs only the last
    # `deq_bptt_k` forward iterations; earlier iters contribute no gradient.
    # 0 (or >= num_layers) = full BPTT.  Rationale: for a contractive DEQ,
    # per-iter VJP magnitudes decay geometrically toward x0, so the last few
    # iters should dominate the total param gradient.  If the hypothesis
    # holds, throughput scales ~ K_fwd / (K_fwd + K_bwd) improvement.
    deq_bptt_k = 3  # iter 95 (2026-05-02): TBPTT=2 → 3 PROMOTED ★ under iter 112+122 baseline. Triggered by grad_norm=0.07 in mid-flight of iter 112+122 (well below clip=1.0 → headroom for deeper backward). Backward coverage at K=16 increases 12.5%→19%. Cost ~+10% wallclock; val_bpb int6 1.5001 (Δ-0.0164 vs iter 112+122) and K-sweep tightening confirmed.
    # Iter 85 enabled stochastic TBPTT {2,3,4} as a K-jitter analog (H63
    # PROMOTED ★ narrow margin). 2026-04-28 PROFILE-driven revert: H63 itself
    # noted +0.0054 val_bpb regression vs fixed k=2 AND +21% throughput cost,
    # and the dev profile run showed jitter (3 values) × K-jitter (3 values)
    # = 9 unique compiled-graph variants, exceeding `_dynamo.config.recompile_limit
    # = 8` and triggering per-step recompile thrash. Disabling jitter recovers
    # iter 84's val_bpb baseline, +21% throughput, AND eliminates the 3× cache-axis
    # pressure from the K × TBPTT cross product. The `deq_bptt_k_for_step`
    # sampler short-circuits to `args.deq_bptt_k` when `deq_bptt_k_jitter=False`
    # — the singleton set below is kept for state-dict / config compat.
    deq_bptt_k_jitter = False
    deq_bptt_k_jitter_set = (3,)  # tracks deq_bptt_k=3; jitter disabled, value is state-dict/config cosmetic
    # TBPTT investigation (28-28d) concluded; best point was 28c (val_bpb
    # 1.925, K=128 Δ=0.015 vs baseline 0.039).  Machinery retained in code
    # — re-enable via CLI --deq-bptt-k=N.  Deeper-K jitter (4,8,16,24) may
    # be re-combined with Phase 6 contraction shell in a follow-up iter once
    # the new architecture stabilizes val_bpb.
    # 2026-04-29 user directive: K-jitter re-enabled with set {16, 24}.
    # Rationale: iter 98b K-sweep showed val_bpb is essentially CONVERGED at
    # K=16 (K=16: 1.5018, K=128: 1.5039, Δ=+0.0021). The FP is found at K=16.
    # Adding K=24 to the jitter set tests whether wider FP-depth jitter
    # provides regularization gain (analog of H12 VERIFIED at the wider
    # {4,6,10}→{8,12,20} scale). The post-2026-04-28 K=16-fix established that
    # RevDEQ is O(1) in K — there's no OOM concern at K=24. K=24 is also
    # added to the K-sweep matrix for cross-K diagnostics at this depth.
    deq_k_jitter = True
    deq_k_min = 4
    # iter 146: keep most training at the promoted K16/K24 depths, but sample
    # K32/K64 at low probability for finite-depth robustness. Weighted sampling
    # is required: a plain 4-value tuple would sample K64 25% of the time.
    deq_k_max = 128  # extended 2026-05-13 to accommodate K=96 and K=128 in the K-jitter tail
    deq_k_step = 4
    # K-jitter set extended 2026-05-13 per user directive: add K=96 and K=128
    # at half-frequency cascading from K=64 (96 = 0.5 * w_64, 128 = 0.5 * w_96).
    # Rationale: low-frequency deeper-K samples expose the model to fixed-point
    # convergence at the gate-relevant K=128 evaluation depth without
    # significantly increasing average per-step cost
    # (E[K] = 16*0.489 + 24*0.391 + 32*0.069 + 64*0.029 + 96*0.0147 + 128*0.0073
    #     ≈ 22.9, vs prior 21.76, ~5% step-time increase).
    # Framework normalizes weights internally; raw values shown for clarity.
    deq_k_jitter_set = (16, 24, 32, 64, 96, 128)
    deq_k_jitter_weights = (0.50, 0.40, 0.07, 0.03, 0.015, 0.0075)
    # iter173 (2026-05-17): K-jitter weight annealing curriculum. Linearly
    # interpolates `deq_k_jitter_weights` → `deq_k_jitter_weights_final` over
    # the training window [anneal_start_frac, anneal_end_frac] · total_steps.
    # Empty `deq_k_jitter_weights_final` = annealing DISABLED (default OFF
    # preserves iter172 behavior). Hypothesis: iter172 achieved rho_F=0.77
    # at K=128 (vs iter163's 0.92) — model is "ready" for deeper K. By
    # gradually shifting probability mass toward K∈{64,96,128} over training,
    # the model practices the deep-K regime that K-sweep eval actually uses.
    # Final distribution (0.125,0.125,0.125,0.125,0.25,0.25) → E[K] ≈ 51 vs
    # the iter172 start E[K] ≈ 22.9 (~2.2× deeper). Anneal window [0.2, 1.0]
    # leaves the first 20% of training at iter172's biased weights (so basic
    # FP convergence is locked in before pushing depth) and spans the final
    # 80% smoothly — no settling buffer at the end because the deeper-K
    # regime is what the K-sweep eval measures, so the model should be at
    # the final distribution exactly when training stops. Cost: deep-K steps
    # are slower (K=128 ≈ 6× K=16 cost), so E[step_cost] increases ~2.2× by
    # end-of-training — expected total wallclock ~1.5× iter172 baseline.
    deq_k_jitter_weights_final: tuple[float, ...] = ()
    deq_k_jitter_anneal_start_frac: float = 0.2
    deq_k_jitter_anneal_end_frac: float = 1.0
    deq_k_eval = 16  # iter 30: baseline eval K (the converged FP)
    # iter152: conditional prefix-K multi-anchor supervision. Default OFF.
    # When enabled, a sampled K still performs one K-step solve, but the train
    # task loss is averaged over traversed prefix endpoints from the current
    # jitter set (e.g. K=32 supervises {16,24,32}). Router/MoS/diversity
    # regularizers remain attached to the final endpoint only, so the iter is a
    # pure finite-depth task-supervision change rather than another coefficient
    # bundle.
    deq_prefix_anchors = True  # iter152 promoted on BPB (1.4718 vs 1.4787); promotion-propagation completed 2026-05-13 after iter153/iter155 confound was diagnosed.
    # iter172 (2026-05-17, PROMOTED at full val_bpb=1.462898 vs iter163's
    # 1.471598 = −8.7 mBPB win, 39× iter163's margin over iter152). Explicit
    # prefix-anchor depths, decoupled from the K-jitter set. The K=8 shallow
    # anchor (gap=8 to next-deeper K=16) matches iter163's healthy regime
    # while giving a non-trivial consistency-pair at K_sampled=16 (which
    # samples ~88% of training steps). iter170's K=4 (gap=12) collapsed
    # rho_F→0 by anchoring an unconverged shallow state to the deep target;
    # iter172's K=8 preserves contraction (final rho_F=0.77 at K=128 vs
    # iter163's 0.92). Default falls back to deq_k_jitter_set if set to ().
    deq_prefix_anchor_set: tuple[int, ...] = (8, 16, 24, 32, 64, 128)

    # iter172 (2026-05-17, PROMOTED at val_bpb=1.462898 vs iter163's
    # 1.471598 = −8.7 mBPB): recursive nearest-neighbor consistency loss.
    # Each prefix-anchor z_i is paired with its NEXT-DEEPER z_{i+1}.detach():
    #   L_anchor = anchor_coef · mean_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²
    # This is literally iter163's promoted recursive anchor formula.
    #
    # Why recursive instead of iter170's all-to-deepest:
    # iter170 (all z_i → z_K_sampled) was REFUTED at step 200 val
    # (val_bpb=2.067 vs baseline 2.003, +65 mBPB regression). The
    # all-to-deepest pairing has ρ → 0 as its unique global minimum —
    # the model satisfies all pairs simultaneously by making F nearly
    # constant in z (flat iteration map = degenerate DEQ = experts
    # collapse, mlp_ortho +0.40). Recursive pairs only require contraction
    # over each LOCAL depth range, allowing ρ ≈ 0.85 (iter163's healthy
    # regime) without degenerating to ρ → 0.
    #
    # Trade-offs vs prior designs (REFUTED alternatives kept for the
    # `removal-symmetry-sweep` audit row; do NOT reintroduce as ablations
    # without re-reading the closure note in hypotheses.md):
    #   iter163  (recursive + Δ=K_train extension): prior champion;
    #     ~1.6× iter152 step time due to deep no-grad extension forward.
    #   iter163c v2 (Δ=1 no-grad iter): REFUTED at full val, +22 mBPB.
    #   iter167 (Δ=8 no-grad iters):   subsumed (z_K_sampled free).
    #   iter170 (all-to-deepest):      REFUTED at s200, +65 mBPB; ρ → 0
    #     degeneracy. K=4 anchor (gap=12 to K=16) further collapsed rho_F.
    #   iter171 (recursive, K=4 + K=256 anchors): superseded by iter172.
    #   iter172 (this design, PROMOTED): recursive only (no extension),
    #     anchor set (8, 16, 24, 32, 64, 128). K=8 gives non-trivial
    #     consistency pair at K_sampled=16 without iter170's gap=12
    #     rho_F collapse. Cost ~1.00× iter152 step time. Worst-case
    #     backward chain count at K_sampled=128: 6 anchors × 3 bptt = 18
    #     chains = iter163-baseline-safe memory.
    #
    # Architecture-agnostic per CLAUDE.md most-principled-simplest-general.
    # Disable for ablation with `--multi-k-consistency-anchor-coef=0`.
    multi_k_consistency_anchor_coef = 0.1

    # iter161-QAT-late (2026-05-16): deterministic STE int6-SDCLIP fake-quant
    # on CastedLinear weights (matrix tensors with numel > 8192) for the last
    # `(total_steps - qat_late_start_step)` training steps. The model adapts
    # to the EXACT artifact-time int6 quantizer (same per-row SDCLIP scaling
    # as encode_scored_artifact's quantize_int6_sdclip) so the fast→full BPB
    # gap (~22 mBPB at iter163 baseline, of which ~5-10 mBPB is quant tax)
    # closes via training-time root-cause adaptation rather than post-hoc PTQ
    # patches (refuted alternatives: GPTQ+LQER bundle is symptom-targeting,
    # not principled). Default -1 means OFF. Recommended value 800 (= last
    # 200/1000 steps = 20 %).
    qat_late_start_step: int = -1

    # Architecture knobs
    # iter 6: reduced bigram hash from 65536×208 (13.7M params = 71% of model!)
    # to 4096×128 (~590K params) to match records (2026-03-25 uses 2048×128,
    # 2026-03-20 SmearGate uses 4096×128).  Frees ~13.1M params for the
    # transformer body.  The freed capacity enables model_dim increase in
    # iter 7 and mlp_mult increase in iter 8.
    # Iter 93 (2026-04-24): disable BigramHash. Added in iter 6 under a very
    # different architecture (pre-DEQ, pre-experts, no Parcae B̄ input
    # injection). Under the current iter 85 baseline (NTP-only + learnable
    # norms everywhere + Parcae input forcing + split shared gates + TBPTT
    # jitter), the DEQ's x₀ re-injection already carries token-pair
    # information. Setting to 0 frees ~1 MB of artifact budget (4096 × 128
    # × 2 B + 128 × 768 proj) that can compound into iter 90–92's arch
    # scale-up. GPT.__init__ already guards `if bigram_vocab_size > 0`.
    bigram_vocab_size = 0
    bigram_dim = 128
    kv_latent_dim = 0  # auto: dim//2
    # iter 96 baseline (PROMOTED, H71): rank-halving paired with E×2 from
    # iter 89's R=128/192 / E=8. Iter 97 attempted further continuation
    # (E=20, R=51/77) but axis saturated past E=16 — see H72.
    attn_expert_rank = 64
    mlp_expert_rank = 96

    # iter176 (2026-05-18) — per-dim sigmoid gate per MLP expert. Each expert's
    # output (after expert_down) is multiplied element-wise by a per-expert
    # per-token sigmoid gate computed from the input x via a low-rank LoRA
    # projection: gate_e(x) = σ(V_e (U_e x) + b_e) ∈ (0, 1)^D.
    # iter176b (2026-05-19) — STRICT-GENERALIZATION init (LoRA convention,
    # Hu et al. 2021): U xavier (down), V zeros (up), bias +6.0 → at init
    # gate = σ(V·U·x + b) = σ(0·U·x + 6.0) ≈ 0.9975 ≈ 1.0 → forward
    # ≡ iter172 baseline. Optimizer is free to learn V ≠ 0 only if it
    # reduces loss; if redundant with the diversity loss, V stays near zero
    # and val_bpb matches iter172. Per CLAUDE.md strict-generalization:
    # at worst this matches iter172, never regresses.
    # (iter176 used the wrong init — xavier U/V + bias=0 → σ ≈ 0.5 →
    # halved expert outputs at init, violating strict-gen and showing
    # transient early lead followed by regression as warmup completed.)
    # Cost at rank=16: 16 (rank) * 768 (D) * 2 (U+V) * 16 (experts) +
    # 16 * 768 (bias) = 405K params per MLP layer. Step time +5-10%.
    # RevDEQ-safe: sigmoid is differentiable + bounded + monotonic;
    # reverse pass recomputes the gate via one matmul. No batch coupling.
    use_expert_perdim_gate = False
    expert_perdim_gate_rank = 16

    # Weight averaging
    # iter 1: disabled.  At 1h budget (~822 steps) ema_decay 0.997 leaves
    # ~8.6% of random init in the EMA average (0.997^822); SWA averages 4
    # checkpoints from a still-improving region.  Both dragged iter 0's gates
    # toward identity and added 0.83 BPB to the post-quant val.  Test whether
    # the DEQ fixed-point survives without weight averaging.
    swa_enabled = False
    swa_start_frac = 0.4
    swa_every = 50
    ema_enabled = False
    ema_decay = 0.997
    ema_update_every = 1

    eval_stride = 0
    eval_batch_seqs = 256  # fast eval: 256 seqs × 2048 = 512K tokens (~8x more representative than 32)


_CONFIG_PROFILES: dict[str, dict[str, object]] = {
    # Current source-of-truth defaults.  Kept explicit so launch logs and
    # tests can distinguish "default recipe" from hand-assembled CLI flags.
    "fast_default": {},
    # BPB-winning reference from iter152: conditional prefix-K anchors on the
    # iter149 coefficient base.  This is a profile, not the unconditional
    # default, because it costs more wallclock and still carries health debt.
    # Note: `eval_profile=submission` runs the local Lipschitz probe ONLY at
    # K=128, so reproducing this profile produces a shorter audit trail than
    # the iter152 promotion run (which used the full diagnostic K-sweep).
    # Switch to `--eval-profile=diagnostic` to recover the original coverage.
    "score_iter152": {
        "deq_prefix_anchors": True,
        "router_ema_balance_coef": 0.60,
        "router_ema_specialization_coef": 0.40,
        "router_pertoken_entropy_coef": 0.20,
        "expert_output_diversity_coef": 0.60,
        "router_ema_alive_coef": 0.04,
        "eval_profile": "submission",
    },
    # Local smoke/debug profile.  It changes validation scope only; callers
    # still choose their own training length explicitly.
    "debug": {
        "eval_profile": "debug",
        "final_full_validation": False,
    },
}


@dataclass(frozen=True)
class EvalProfile:
    k_sweep_values: tuple[int, ...]


_EVAL_PROFILES: dict[str, EvalProfile] = {
    "debug": EvalProfile(k_sweep_values=(16,)),
    "submission": EvalProfile(k_sweep_values=(16, 24, 64, 128)),
    "diagnostic": EvalProfile(
        # K=192 added 2026-05-17 for extrapolation perf test: 50% deeper than
        # max-trained K=128 in iter172/iter173, tests whether the rho_F<1
        # contraction basin extends beyond the trained-K cap. If val_bpb at
        # K=192 ≈ K=128, the model has wide-basin convergence (good); if
        # K=192 degrades, the consistency loss is overfitting to trained K.
        # For iter174 (K=256 in jitter), K=192 becomes an interpolation point
        # between trained K=128 and K=256.
        k_sweep_values=(4, 8, 16, 17, 24, 32, 37, 64, 113, 128, 192),
    ),
}


_HYPERPARAMETER_FIELDS: frozenset[str] = frozenset(
    name for name, value in vars(Hyperparameters).items()
    if not name.startswith("_") and not callable(value)
)


def _assert_known_hyperparameter(key: str, *, source: str) -> None:
    # Without this guard a typo in `_CONFIG_PROFILES` (e.g. `router_ema_balanced_coef`)
    # silently creates a stray attribute on `args`; the real Hyperparameter stays
    # at its default and the promotion log misreports the active config.
    if key not in _HYPERPARAMETER_FIELDS:
        raise SystemExit(
            f"unknown Hyperparameter {key!r} (source: {source}); "
            f"add the field to Hyperparameters or fix the typo"
        )


def _apply_config_profile(args, profile_name: str) -> None:
    profile = str(profile_name or "fast_default").strip().lower()
    if profile not in _CONFIG_PROFILES:
        valid = ", ".join(sorted(_CONFIG_PROFILES))
        raise SystemExit(f"config_profile={profile_name!r} must be one of: {valid}")
    args.config_profile = profile
    for key, value in _CONFIG_PROFILES[profile].items():
        _assert_known_hyperparameter(key, source=f"_CONFIG_PROFILES[{profile!r}]")
        setattr(args, key, value)


def _resolve_k_sweep_values(args) -> list[int]:
    profile = str(args.eval_profile).strip().lower()
    # `_validate_hyperparameters` owns the value-set check; if we got here with
    # an unknown profile something bypassed validation.
    assert profile in _EVAL_PROFILES, f"unknown eval_profile {profile!r}"
    return list(_EVAL_PROFILES[profile].k_sweep_values)


@dataclass(frozen=True)
class OptionalComponentCapability:
    py_name: str
    label: str
    state: str  # rejected | training_effect | eval_effect | artifact_effect


_OPTIONAL_COMPONENT_CAPABILITIES: tuple[OptionalComponentCapability, ...] = (
    OptionalComponentCapability("use_smear_gate", "smear_gate", "training_effect"),
    OptionalComponentCapability("use_sparse_attn_head_gate", "sparse_attn_head_gate", "training_effect"),
    OptionalComponentCapability("use_rr_attention", "rr_attention", "training_effect"),
    OptionalComponentCapability("use_ttt_eval", "ttt_eval", "rejected"),
    OptionalComponentCapability("use_gptq", "gptq", "rejected"),
    OptionalComponentCapability("use_lqer", "lqer", "rejected"),
    OptionalComponentCapability("use_grouped_artifact_compression", "grouped_artifact", "artifact_effect"),
    OptionalComponentCapability("use_caseops", "caseops", "rejected"),
    OptionalComponentCapability("use_sparse_dispatch", "sparse_dispatch", "rejected"),
    OptionalComponentCapability("deq_prefix_anchors", "deq_prefix_anchors", "training_effect"),
    OptionalComponentCapability("use_reverse_kl_balance", "reverse_kl_balance", "training_effect"),
    OptionalComponentCapability("use_expert_perdim_gate", "expert_perdim_gate", "training_effect"),
)


# Numeric / string knobs that are tunable from the command line. Single source
# of truth — `experiments/test_cli_parser.py::test_default_parity` iterates
# this tuple to verify every entry resolves to a `Hyperparameters` field with
# the documented default. Adding a new tunable knob? Append here AND add the
# CLAUDE.md §5 row in the same commit (EXPERIENCE.md#hyperparameter-fanout).
_CLI_TUNABLE_KNOBS: tuple[str, ...] = (
    "config-profile", "eval-profile", "diagnostic-gate-policy",
    "data-path", "tokenizer-path", "run-id", "seed", "iterations",
    "train-batch-tokens", "train-seq-len",
    "val-batch-size", "val-micro-batch-seqs", "val-loss-every", "train-log-every",
    "fp-rho-power-iters",
    "checkpoint-dir", "checkpoint-every", "checkpoint-keep", "resume-from",
    "max-training-seconds", "max-wallclock-seconds",  # max-wallclock-seconds is a deprecated alias
    "grad-accum-multiplier",
    "num-heads", "num-kv-heads",
    "bigram-vocab-size", "bigram-dim",
    "kv-latent-dim", "attn-expert-rank", "mlp-expert-rank",
    "swa-start-frac", "swa-every", "ema-decay", "ema-update-every",
    "deq-k-min", "deq-k-max", "deq-k-step", "deq-k-eval",
    "deq-k-jitter-set", "deq-k-jitter-weights", "deq-bptt-k",
    "deq-k-jitter-weights-final", "deq-k-jitter-anneal-start-frac",
    "deq-k-jitter-anneal-end-frac",
    "deq-prefix-anchor-set",
    "fast-val-k-sweep-set",
    "expert-perdim-gate-rank",
    "deq-beta", "deq-beta-jitter-set",
    "warmdown-frac", "num-refinements-ramp-frac",
    "weight-decay",
    "muon-momentum-warmup-frac", "muon-momentum-warmup-steps",
    "router-scoring", "router-pertoken-entropy-coef",
    "router-dirichlet-ucb-beta",
    "router-ema-alive-coef", "router-ema-balance-coef", "router-ema-specialization-coef",
    "entmax-blend-init-logit", "entmax-blend-warmup-delay-frac", "entmax-blend-lr",
    # iter 129 / H99 — SmearGate (default off; --use-smear-gate=1 to enable)
    "smear-gate-init", "smear-gate-window", "smear-gate-bos-id",
    "sparse-attn-gate-window", "sparse-attn-gate-scale",
    "sparse-attn-gate-factor", "sparse-attn-gate-init-std",
    "rr-stride", "rr-block-size", "rr-tau",
    # iter 122 H93 — logit softcap (Gemma2-style)
    "logit-softcap",
    # iter 117b-3 — sparse MoE dispatch capacity factor
    "sparse-dispatch-capacity-factor",
    "parcae-init-a-bar", "parcae-init-b-bar", "parcae-lr",
    "denoising-coef", "denoising-noise-std",
    "lyapunov-coef", "lyapunov-gamma", "lyapunov-every", "lyapunov-max-tokens",
    "lyapunov-target",
    "lyapunov-estimator",
    "eval-reservation-seconds",
    "ctp-weight",
    "multi-k-consistency-anchor-coef",
    "qat-late-start-step",
    "expert-diversity-kind",
    "expert-output-diversity-coef", "expert-diversity-every", "expert-diversity-max-tokens",
    "mos-output-diversity-coef",
    "regularizer-warmup-frac",
    "chained-stages-preset",
    # iter 106 NSA — Native Sparse Attention (H86)
    "use-nsa-attention",
    "nsa-compress-block-size", "nsa-compress-block-sliding-stride",
    "nsa-selection-block-size", "nsa-num-selected-blocks",
    "nsa-sliding-window-size", "nsa-branch-gate-init",
)


# Optional default-off component flags. Single source of truth that drives:
# - the CLI bool-flag list in `_parse_cli_overrides` (dashed names),
# - the `bool_keys` set in `_parse_cli_overrides` (underscored names),
# - the banner emission in `main()` (label seen in run.log).
# Adding a new optional component is a registry change with an explicit
# capability state.  `_OPTIONAL_COMPONENT_FLAGS` remains as the backwards-
# compatible `(py_name, label)` projection used by older call sites.
_OPTIONAL_COMPONENT_FLAGS: tuple[tuple[str, str], ...] = tuple(
    (cap.py_name, cap.label) for cap in _OPTIONAL_COMPONENT_CAPABILITIES
)


def _parse_cli_overrides(argv: list[str]) -> dict[str, object]:
    p = argparse.ArgumentParser(add_help=True)
    for name in _CLI_TUNABLE_KNOBS:
        py_name = name.replace("-", "_")
        field_val = getattr(Hyperparameters, py_name, None)
        if isinstance(field_val, float):
            p.add_argument(f"--{name}", type=float, default=None)
        elif isinstance(field_val, int):
            p.add_argument(f"--{name}", type=int, default=None)
        elif isinstance(field_val, tuple):
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=str, default=None)
    _core_bool_flag_names = [
        "auto-plot-on-val", "router-bias-update", "deq-k-jitter",
        "swa-enabled", "ema-enabled", "use-ctp", "use-entmax-routing",
        "use-router-sigmoid-gate",
        "use-polar-express-ns", "use-entmax-triton",
        "use-chained-routing",
        "use-unified-routed-down",  # iter 118a Phase A3
        "use-parcae", "deq-beta-jitter",
        "final-full-validation", "resume-latest",
    ]
    _optional_component_flag_names = [
        py_name.replace("_", "-") for py_name, _ in _OPTIONAL_COMPONENT_FLAGS
    ]
    for name in _core_bool_flag_names + _optional_component_flag_names:
        p.add_argument(f"--{name}", type=int, default=None, help="1/0")
    # iter 106: `use_nsa_attention` defaults to False (bool subclass of int)
    # which the loop above already routes through `add_argument(type=int)`. Add
    # it to bool_keys so 0/1 → False/True conversion happens at parse time.
    p.add_argument("--router-bias-lr", type=float, default=None)
    p.add_argument("--router-bias-clip", type=float, default=None)
    ns, unknown = p.parse_known_args(argv)
    # Reject only `--`-prefixed unknowns; bare positionals are passed through
    # so wrapper / profile harnesses can inject their own flags without breaking
    # this parser.
    bad = [u for u in unknown if u.startswith("--")]
    if bad:
        raise SystemExit(f"Unknown args: {bad}")
    out: dict[str, object] = {}
    bool_keys = {
        "auto_plot_on_val", "router_bias_update", "deq_k_jitter",
        "swa_enabled", "ema_enabled", "use_ctp", "use_nsa_attention",
        "use_entmax_routing", "use_router_sigmoid_gate",
        "use_polar_express_ns",
        "use_entmax_triton", "use_unified_routed_down",
        "use_chained_routing",
        "use_parcae", "deq_beta_jitter",
        "final_full_validation", "resume_latest",
    }
    bool_keys.update(py_name for py_name, _ in _OPTIONAL_COMPONENT_FLAGS)
    for k, v in vars(ns).items():
        if v is not None:
            key = k.replace("-", "_")
            default = getattr(Hyperparameters, key, None)
            if key in bool_keys:
                out[key] = bool(int(v))
            elif isinstance(default, tuple):
                raw = str(v).strip()
                if raw.startswith(("(", "[")) and raw.endswith((")", "]")):
                    raw = raw[1:-1]
                vals = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
                out[key] = vals
            else:
                out[key] = v
    return out


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def update_ema_state_(ema_state: dict[str, Tensor], model_state: dict[str, Tensor],
                      *, decay: float) -> None:
    alpha = 1.0 - float(decay)
    with torch.no_grad():
        for name, t in model_state.items():
            ema_state[name].mul_(decay).add_(t.detach().float().cpu(), alpha=alpha)


# ---------------------------------------------------------------------------
# MUON OPTIMIZER
# ---------------------------------------------------------------------------

# Polar-Express per-iteration minimax-optimized Newton-Schulz coefficients
# (Bernstein et al. 2024; adopted in records via PR #1344 → #1787).
# Each tuple is (a, b, c) for ONE iteration of the quintic NS recurrence
# `X' = aX + (bA + cA²)X` where `A = XX^T`. The coefficients are tuned per
# iteration: aggressive at the start (X far from polar factor), gentle at
# the end (X near polar factor). Same converged accuracy as the stock fixed
# `(3.4445, -4.7750, 2.0315)` tuple in ~half the matmul count (5 iter vs ~10).
# Outside `len(_PE_COEFFS)` iterations, fall back to the last (converged)
# tuple — strict equivalence to the standard NS limit.
_PE_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461,  -22.48329292557795,   15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
)


# Stock NS coefficients (Muon's original minimax-optimal triple, used in
# all iters preceding 121's PE-NS investigation).
_STOCK_NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

# Module-level flag toggled by `Hyperparameters.use_polar_express_ns` at GPT
# construction (`_apply_ns_coefficient_choice`). When False, _ns_iter_coeffs
# returns the stock triple repeated; when True, returns the per-iter PE table.
_USE_POLAR_EXPRESS_NS: bool = False


def _ns_iter_coeffs(steps: int) -> tuple[tuple[float, float, float], ...]:
    """Return `steps` (a, b, c) tuples for the NS iteration. Default uses
    the stock minimax-optimal triple repeated; setting `_USE_POLAR_EXPRESS_NS`
    True swaps to the per-iter Polar-Express table (with the last tuple
    repeated past `len(_PE_COEFFS)` for converged refinement)."""
    if _USE_POLAR_EXPRESS_NS:
        if steps <= len(_PE_COEFFS):
            return _PE_COEFFS[:steps]
        return _PE_COEFFS + (_PE_COEFFS[-1],) * (steps - len(_PE_COEFFS))
    return (_STOCK_NS_COEFFS,) * steps


def _apply_ns_coefficient_choice(use_pe: bool) -> None:
    """Update module-level `_USE_POLAR_EXPRESS_NS` flag. Called once at
    GPT construction from `Hyperparameters.use_polar_express_ns`. The
    NS iteration helpers (compiled at module load) read this flag at
    forward time."""
    global _USE_POLAR_EXPRESS_NS
    _USE_POLAR_EXPRESS_NS = bool(use_pe)


def _ns5_2d(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """NS preconditioner for a single 2D matrix.

    Uses Polar-Express per-iter coefficients: 5 iterations matches the
    convergence quality of stock NS at 10 iterations. `muon_backend_steps`
    default is 5 — exactly the length of `_PE_COEFFS`.
    """
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for a, b, c in _ns_iter_coeffs(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def _ns5_batched(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """NS preconditioner for a batch of independent matrices.

    Input: (..., m, n) — leading dims are batch, last two are the matrix.
    Each matrix is normalized and processed independently. Polar-Express
    coefficients (per-iter; see `_ns_iter_coeffs`).
    """
    X = G.bfloat16()
    norms = X.flatten(-2).norm(dim=-1)
    X = X / (norms[..., None, None] + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2).contiguous()
    for a, b, c in _ns_iter_coeffs(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X


# Compile NS backends for throughput.  Graceful fallback to eager if compile fails.
try:
    zeropower_via_newtonschulz5 = torch.compile(_ns5_2d)
    zeropower_via_newtonschulz5_batched = torch.compile(_ns5_batched)
except Exception:
    zeropower_via_newtonschulz5 = _ns5_2d
    zeropower_via_newtonschulz5_batched = _ns5_batched


class Muon(torch.optim.Optimizer):
    """Muon optimizer with batched NS for expert weight banks.

    For ndim>2 params (expert banks), NS operates on the last two dims
    independently per batch element.  All expert tensors are now stored
    natively as (E, R, D) — no transpose handling needed.
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))
        # Non-serialized cache for per-group flat update buffers + offsets.
        # Stored outside self.state to avoid breaking state_dict()/checkpointing.
        self._buf_cache: dict[int, tuple[Tensor, list[tuple[int, int]]]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            # Cached flat update buffer + offsets (non-serialized, won't break state_dict)
            group_idx = id(group["params"][0]) if params else 0
            if group_idx in self._buf_cache:
                updates_flat, offsets = self._buf_cache[group_idx]
                updates_flat.zero_()
            else:
                total_params = sum(int(p.numel()) for p in params)
                updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
                offsets = []
                curr = 0
                for p in params:
                    offsets.append((curr, curr + p.numel()))
                    curr += p.numel()
                self._buf_cache[group_idx] = (updates_flat, offsets)
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    orig_shape = g.shape
                    if g.ndim > 2:
                        g_b = g.reshape(-1, g.shape[-2], g.shape[-1])
                        g_b = zeropower_via_newtonschulz5_batched(g_b, steps=backend_steps)
                        m, n = g_b.size(-2), g_b.size(-1)
                        g_b *= max(1, m / n) ** 0.5
                        g = g_b.reshape(orig_shape)
                    else:
                        g = g.reshape(-1, g.shape[-1])
                        g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                        g = g.reshape(orig_shape)
                    start, end = offsets[i]
                    updates_flat[start:end] = g.reshape(-1)
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int,
                             device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def run_validation(args, model, rank, world_size, device, grad_accum_steps,
                   val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                   *, full_validation: bool, deq_k: int | None = None) -> tuple[float, float]:
    """Validate on the val set.  deq_k overrides the number of DEQ solver
    iterations; when None (default) uses args.deq_k_eval."""
    local_batch_tokens = args.val_batch_size // world_size
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    # T-opt 17: cap val micro-batch to training micro-batch for robustness.
    # When val_micro_batch_seqs=0, derive from training config (same B).
    # Eval uses inference_mode (no backward), so same B is always safe.
    _val_cap = getattr(args, "val_micro_batch_seqs", 0)
    if _val_cap > 0:
        local_batch_seqs = min(local_batch_seqs, _val_cap)
    else:
        # Derive: train_batch_tokens / seq_len / (world_size * grad_accum)
        _train_seqs = args.train_batch_tokens // args.train_seq_len
        _train_micro = max(1, _train_seqs // (world_size * grad_accum_steps))
        local_batch_seqs = min(local_batch_seqs, _train_micro)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if full_validation or args.eval_batch_seqs <= 0:
        global_seqs = total_seqs
    else:
        global_seqs = min(total_seqs, int(args.eval_batch_seqs))
    seq_start = (global_seqs * rank) // world_size
    seq_end = (global_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    # `model.train(...)` is called on the outer (possibly wrapped) module on
    # purpose: PyTorch cascades training-mode to all children via
    # `self.children()`, so toggling the wrapper toggles the underlying GPT.
    # `_unwrap_compiled_module(model)` is only used for attribute *writes*
    # (e.g. `_deq_k_override`), which must reach the semantic owner.
    was_training = bool(model.training)
    model.train(False)
    base_m = _unwrap_compiled_module(model)
    k = int(deq_k if deq_k is not None else getattr(args, "deq_k_eval", base_m.num_layers))
    # Option G2 attempt (2026-04-28) — REVERTED. Wrapping val in
    # `torch._dynamo.config.disable=True` would have eliminated the grad_mode
    # toggle as a recompile axis, but the global config flip disrupted the
    # compile-cache state across val→train transitions, forcing a fresh
    # compile of block.forward at training step 1 with K=24, which spiked peak
    # VRAM over the 44 GiB cap → OOM at the RevDEQ backward (profile_v8). Fix
    # #5a's 0 recompile_limit hits already provided sufficient cache headroom
    # (12 of 16 slots used), so G2's marginal benefit (further reduce to ~6
    # slots) wasn't worth the OOM risk. Keeping the val path as-is —
    # `torch.inference_mode()` triggers ONE cache slot for the eval graph,
    # which is bounded and fits comfortably under recompile_limit=16.
    try:
        with _temporary_deq_k_override(base_m, k):
            with torch.inference_mode():
                for batch_start in range(seq_start, seq_end, local_batch_seqs):
                    batch_end = min(batch_start + local_batch_seqs, seq_end)
                    raw_start = batch_start * args.train_seq_len
                    raw_end = batch_end * args.train_seq_len + 1
                    local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
                    x = local[:-1].reshape(-1, args.train_seq_len)
                    y = local[1:].reshape(-1, args.train_seq_len)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        _ = model(x, y)
                    loss_t = getattr(base_m, "_ntp_loss_t", None)
                    if isinstance(loss_t, torch.Tensor):
                        batch_loss = loss_t.detach().to(device=device, dtype=torch.float64)
                    else:
                        batch_loss = torch.tensor(
                            float(getattr(base_m, "_ntp_loss", 0.0)),
                            device=device,
                            dtype=torch.float64,
                        )
                    batch_token_count = float(y.numel())
                    val_loss_sum += batch_loss * batch_token_count
                    val_token_count += batch_token_count
                    prev_ids = x.reshape(-1)
                    tgt_ids = y.reshape(-1)
                    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                    val_byte_count += token_bytes.to(torch.float64).sum()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
            val_loss = val_loss_sum / val_token_count
            bits_per_token = val_loss.item() / math.log(2.0)
            tokens_per_byte = val_token_count.item() / val_byte_count.item()
            return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
    finally:
        model.train(was_training)


# Backwards-compatible alias used by smoke_test.py and other callers
eval_val = run_validation


# ---------------------------------------------------------------------------
# QUANTIZATION (uniform INT6 + SDClip)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_PATTERNS = (
    "q_gain", "gate_bias", "bigram.scale", "norm_weight", "nsa_branch_gate",
    "_entmax_blend_logit", "attn_gate_w", "smear_gate",
)
FP16_KEEP_PATTERNS = ("tok_emb",)
SDCLIP_K_MATRIX = 12.85
SDCLIP_K_EMBED = 20.0
INT6_CLIP = 31

def _sdclip_scale(t: Tensor, k: float) -> Tensor:
    """SDClip: clip = k * std(row), scale = clip / 31."""
    if t.ndim >= 2:
        row_std = t.float().std(dim=-1)
        clip_abs = k * row_std
        return (clip_abs / INT6_CLIP).clamp_min(1e-12).to(torch.float16)
    amax = t.float().abs().max().item()
    return torch.tensor(max(amax / INT6_CLIP, 1e-12), dtype=torch.float16)

def quantize_int6_sdclip(t: Tensor, k: float = SDCLIP_K_MATRIX) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim >= 2:
        s = _sdclip_scale(t32, k)
        s = s.clamp_min(torch.finfo(torch.float16).tiny)
        s_expand = s.float().view(-1, *([1] * (t32.ndim - 1)))
        t_2d = t32.reshape(-1, t32.shape[-1]) if t32.ndim > 2 else t32
        s_2d = s_expand.reshape(-1, 1) if t32.ndim > 2 else s_expand
        q = torch.clamp(torch.round(t_2d / s_2d), -(INT6_CLIP + 1), INT6_CLIP).to(torch.int8)
        if t32.ndim > 2:
            q = q.view(t32.shape)
        return q, s
    s = _sdclip_scale(t32, k)
    q = torch.clamp(torch.round(t32 / s.float()), -(INT6_CLIP + 1), INT6_CLIP).to(torch.int8)
    return q, s


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if "bigram" in name:
        return "bigram"
    return "matrix"


# iter161-QAT-late (2026-05-16): deterministic STE fake-quant matching the
# encode_scored_artifact int6 SDCLIP quantizer EXACTLY. Activated only for
# the last 20 % of training steps so the model adapts to the deployment
# quantization noise without slowing earlier training. Deterministic forward
# (no randomness) → RevDEQ-safe (unlike iter20's random quant-noise injection
# which broke reverse reconstruction). Per the most-principled-simplest-
# general directive, this is the root-cause attack on the ~22 mBPB fast→full
# gap: train the model to BE quant-robust, not patch frozen weights post-hoc.

class _FakeQuantInt6SDClipSTE(torch.autograd.Function):
    """STE-style int6 SDCLIP fake quantization matching
    :func:`quantize_int6_sdclip` exactly (per-row scale on ndim>=2 weights,
    sd-clip factor ``k=SDCLIP_K_MATRIX``).

    Forward: same dequantized output that the artifact codec produces.
    Backward: identity (straight-through estimator).

    Deterministic forward (only weight values matter; no RNG, no random
    noise) → RevDEQ reverse reconstruction works. This is the architectural
    distinction from iter20's stochastic noise injection (refuted 2026-04
    because random per-call noise made forward non-deterministic).
    """

    @staticmethod
    def forward(ctx, weight, k):
        if weight.ndim < 2:
            return weight  # too small for per-row quant; identity
        t32 = weight.float()
        s = _sdclip_scale(t32, k)
        s = s.clamp_min(torch.finfo(torch.float16).tiny)
        s_expand = s.float().view(-1, *([1] * (t32.ndim - 1)))
        t_2d = t32.reshape(-1, t32.shape[-1]) if t32.ndim > 2 else t32
        s_2d = s_expand.reshape(-1, 1) if t32.ndim > 2 else s_expand
        q_int = torch.clamp(torch.round(t_2d / s_2d), -(INT6_CLIP + 1), INT6_CLIP)
        deq = (q_int * s_2d)
        if t32.ndim > 2:
            deq = deq.view(weight.shape)
        return deq.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # STE: identity gradient; no grad on k


class _QATLateState:
    """Module-global QAT-late activation state. Training loop sets
    `current_step` per iteration; `start_step` is the step at which fake-
    quant turns on (set from `Hyperparameters.qat_late_start_step`, -1
    means OFF). `CastedLinear.forward` checks `active()` to decide whether
    to apply STE fake-quant on the weight.
    """
    start_step: int = -1
    current_step: int = -1
    sdclip_k: float = SDCLIP_K_MATRIX

    @classmethod
    def reset(cls) -> None:
        cls.start_step = -1
        cls.current_step = -1
        cls.sdclip_k = SDCLIP_K_MATRIX

    @classmethod
    def set_step(cls, step: int) -> None:
        cls.current_step = int(step)

    @classmethod
    def configure(cls, start_step: int, sdclip_k: float = SDCLIP_K_MATRIX) -> None:
        cls.start_step = int(start_step)
        cls.sdclip_k = float(sdclip_k)

    @classmethod
    def active(cls) -> bool:
        return cls.start_step >= 0 and cls.current_step >= cls.start_step


_QAT_LATE_STATE = _QATLateState()


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 8192:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(p in name for p in FP16_KEEP_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            k = SDCLIP_K_EMBED if cat == "embed" else SDCLIP_K_MATRIX
            q, s = quantize_int6_sdclip(t, k=k)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_int6_sdclip(t, k=SDCLIP_K_MATRIX)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
    return result, meta


def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta[name]
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        orig_shape = orig.shape
        if s.ndim > 0:
            q_2d = q.view(-1, q.shape[-1]) if q.ndim > 2 else q
            deq = q_2d.float() * s.float().view(q_2d.shape[0], *([1] * (q_2d.ndim - 1)))
            out[name] = deq.view(orig_shape).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).view(orig_shape).to(orig_dtype)
    return out


# Canonical int6 quantization category set used by the production save path
# (`main()` end-of-training).  Tests must use the same set or they verify a
# different artifact than the one actually scored.
INT6_CATEGORIES: set[str] = {"matrix", "embed", "bigram"}


def save_int6_artifact(state_dict: dict[str, Tensor]) -> tuple[bytes, dict, dict]:
    """Quantize + serialize + compress a state_dict the same way `main()` does.

    Single source of truth for the submission artifact format. Returns a
    `(compressed, qsd, meta)` tuple — `compressed` is the bytes that get
    written to `model.int6.ptz`; `qsd` and `meta` are the inputs to
    `dequantize_mixed_int6` and are returned for callers that want to
    diagnostic-roundtrip without re-decoding the blob.

    Format mirrors `main()` exactly:
      - int6_cats = INT6_CATEGORIES
      - serialized as `{"state_dict": qsd, "meta": meta}` via torch.save
      - compressed with zstd level 22 if zstandard is importable, else zlib 9
    """
    qsd, meta = mixed_quantize_int6(state_dict, INT6_CATEGORIES)
    buf = io.BytesIO()
    torch.save({"state_dict": qsd, "meta": meta}, buf)
    raw_bytes = buf.getvalue()
    if _COMPRESSOR == "zstd":
        compressed = zstandard.ZstdCompressor(level=22).compress(raw_bytes)
    else:
        compressed = zlib.compress(raw_bytes, 9)
    return compressed, qsd, meta


@dataclass(frozen=True)
class ArtifactCodecResult:
    compressed: bytes
    qsd: dict
    meta: dict
    codec: str
    baseline_bytes: int
    compressed_bytes: int
    stats: dict[str, object]


def encode_scored_artifact(
    state_dict: dict[str, Tensor],
    *,
    use_grouped_artifact_compression: bool = False,
    log_fn=None,
) -> ArtifactCodecResult:
    """Encode the scored artifact behind one byte-accounted codec interface.

    New codecs must compete here via measured bytes and roundtrip-compatible
    payloads, rather than growing one-off save branches in `main()`.
    """
    compressed, qsd, meta = save_int6_artifact(state_dict)
    baseline_bytes = len(compressed)
    codec = f"int6_{_COMPRESSOR}"
    stats: dict[str, object] = {
        "baseline_bytes": baseline_bytes,
        "compressed_bytes": baseline_bytes,
        "codec": codec,
    }
    if use_grouped_artifact_compression:
        from experiments.components.artifact_compression import grouped_compress_int6_payload
        grouped_compressed, grouped_stats = grouped_compress_int6_payload(
            qsd, meta, compressor=_COMPRESSOR,
        )
        grouped_bytes = int(grouped_stats["compressed_bytes"])
        stats["grouped_bytes"] = grouped_bytes
        # Strict `<`: on exact tie, keep the baseline so logs truthfully report
        # that the grouped path did not win.
        if grouped_bytes < baseline_bytes:
            compressed = grouped_compressed
            codec = f"int6_grouped_{_COMPRESSOR}"
            stats.update(grouped_stats)
            stats["codec"] = codec
            stats["compressed_bytes"] = grouped_bytes
            if log_fn is not None:
                log_fn(
                    f"grouped_artifact_compression:accepted bytes:{grouped_bytes} "
                    f"baseline_bytes:{baseline_bytes}"
                )
        elif log_fn is not None:
            log_fn(
                f"grouped_artifact_compression:kept_baseline bytes:{baseline_bytes} "
                f"grouped_bytes:{grouped_bytes}"
            )
    return ArtifactCodecResult(
        compressed=compressed,
        qsd=qsd,
        meta=meta,
        codec=codec,
        baseline_bytes=baseline_bytes,
        compressed_bytes=len(compressed),
        stats=stats,
    )


def load_int6_artifact(blob: bytes, template_state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Decompress + deserialize + dequantize an artifact produced by save_int6_artifact.

    `template_state_dict` is the un-quantized model's state_dict — needed by
    `dequantize_mixed_int6` for original shapes / dtypes. Returns a state_dict
    suitable for `model.load_state_dict(strict=True)`.
    """
    if _COMPRESSOR == "zstd":
        decompressed = zstandard.ZstdDecompressor().decompress(blob)
    else:
        decompressed = zlib.decompress(blob)
    payload = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    return dequantize_mixed_int6(payload["state_dict"], payload["meta"], template_state_dict)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    # Use memmap so multiple DDP ranks share OS page cache.
    # torch.from_numpy on a memmap returns a view (no copy) — the tensor
    # is backed by the file's page cache.  Slicing in TokenStream.take()
    # only materializes the accessed pages.
    #
    # PyTorch warns once per process when wrapping any read-only buffer,
    # because in general writing to such a tensor would be undefined
    # behavior.  Our flow never writes: TokenStream.take() only slices, and
    # DistributedTokenLoader.next_batch() .copy_()s into a separately
    # allocated pinned buffer before any GPU transfer.  Copying the shard
    # at load time would defeat the page-cache sharing this loader was
    # designed to provide (each rank would hold its own ~hundreds-of-MB
    # copy).  Suppress the specific warning at the wrap site only.
    tokens_mmap = np.memmap(file, dtype="<u2", mode="r", offset=header_bytes, shape=(num_tokens,))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        return torch.from_numpy(tokens_mmap.view(np.uint16))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def skip(self, n: int) -> None:
        """Advance position by n tokens without materializing them."""
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            self.pos += k
            remaining -= k

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def state_dict(self) -> dict[str, int]:
        return {"file_idx": int(self.file_idx), "pos": int(self.pos)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.file_idx = int(state.get("file_idx", self.file_idx)) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state.get("pos", 0))


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
        self._pinned_buf: Tensor | None = None  # pinned CPU staging buffer

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        denom = self.world_size * grad_accum_steps
        global_seqs = global_tokens // seq_len
        local_seqs = global_seqs // denom
        if local_seqs < 1:
            raise ValueError(f"TRAIN_BATCH_TOKENS too small: {global_tokens}")
        local_tokens = local_seqs * seq_len
        per_rank_span = local_tokens + 1
        # Each rank advances the stream by the full global span but only
        # materializes its own per_rank_span slice.  Avoids the old O(world_size)
        # read-then-slice pattern where every rank read all ranks' data.
        self.stream.skip(self.rank * per_rank_span)
        local_u16 = self.stream.take(per_rank_span)  # CPU uint16 (memmap-backed)
        self.stream.skip((self.world_size - 1 - self.rank) * per_rank_span)
        # Pinned memory staging: memmap→pinned copy is CPU-only; pinned→GPU
        # with non_blocking=True is truly async (overlaps with compute).
        if self._pinned_buf is None or self._pinned_buf.numel() < per_rank_span:
            try:
                self._pinned_buf = torch.empty(per_rank_span, dtype=torch.uint16, pin_memory=True)
            except RuntimeError:
                self._pinned_buf = torch.empty(per_rank_span, dtype=torch.uint16)
        self._pinned_buf[:per_rank_span].copy_(local_u16)
        local = self._pinned_buf[:per_rank_span].to(self.device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

    def state_dict(self) -> dict[str, object]:
        return {"stream": self.stream.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        stream_state = state.get("stream") if isinstance(state, dict) else None
        if isinstance(stream_state, dict):
            self.stream.load_state_dict(stream_state)


# ---------------------------------------------------------------------------
# TRANSFORMER MODULES
# ---------------------------------------------------------------------------

def _frob_normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    """Frobenius-normalize a tensor (flatten → unit-norm → reshape)."""
    flat = t.reshape(-1)
    return (flat / (flat.norm() + eps)).reshape(t.shape)


def _rms_unit(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Parameter-free RMS normalization over the last dimension."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNorm(nn.Module):
    """Learnable RMSNorm used where the learned scale is intentionally shared."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        y = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return y * self.weight.to(dtype=y.dtype)



# BallProjection, PerExpertSpectralNormCap, SpectralNormCap, and
# refresh_spectral_norms removed in iter 41 (Lyapunov replaced hard
# spectral caps). See git history for reference implementations.


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        bias = self.bias
        # iter161-QAT-late (2026-05-16): apply deterministic int6 SDCLIP STE
        # fake-quant on the weight when QAT-late is active. Per CLAUDE.md
        # most-principled-simplest-general: matches the artifact codec exactly
        # (per-row SDCLIP scale), deterministic so RevDEQ reverse reconstruction
        # works, no extra parameters or modules.
        if _QAT_LATE_STATE.active() and w.numel() > 8192:
            w = _FakeQuantInt6SDClipSTE.apply(w, _QAT_LATE_STATE.sdclip_k)
        if (x.dtype != w.dtype or (bias is not None and x.dtype != bias.dtype)) and not torch.is_autocast_enabled():
            w = w.to(dtype=x.dtype)
            if bias is not None:
                bias = bias.to(dtype=x.dtype)
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Cast control tensors and scalars to FP32 for stable accumulation.

    Does NOT cast expert banks (ndim>=3) — those stay in bf16 to avoid
    per-forward .to(bf16) copies in CastedLinear (huge throughput hit).
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    @dynamo_disable
    def _refresh_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self._cos_cached = freqs.cos()[None, None, :, :].contiguous()
        self._sin_cached = freqs.sin()[None, None, :, :].contiguous()
        self._seq_len_cached = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
            self._refresh_cache(seq_len, device)
        if not torch.is_inference_mode_enabled() and self._cos_cached is not None and self._cos_cached.is_inference():
            self._refresh_cache(seq_len, device)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def mean_abs_offdiag_cosine(groups: Tensor, eps: float = 1e-8) -> Tensor:
    e = groups.shape[0]
    if e < 2:
        return groups.new_zeros(())
    g = groups / (groups.norm(dim=-1, keepdim=True) + eps)
    cos = g @ g.T
    mask = ~torch.eye(e, dtype=torch.bool, device=cos.device)
    return cos[mask].abs().mean()


def max_mean_abs_offdiag_cosine(groups: Tensor, eps: float = 1e-8) -> Tensor:
    e = groups.shape[0]
    if e < 2:
        return groups.new_zeros(())
    g = groups / (groups.norm(dim=-1, keepdim=True) + eps)
    cos = (g @ g.T).abs()
    cos = cos.masked_fill(torch.eye(e, dtype=torch.bool, device=cos.device), 0.0)
    per_expert_mean = cos.sum(dim=-1) / float(e - 1)
    return per_expert_mean.max()


def max_pairwise_abs_cosine(groups: Tensor, eps: float = 1e-8) -> Tensor:
    """Max absolute off-diagonal cosine similarity across ALL expert pairs.

    Flags the single worst pair of near-duplicates in the expert group.
    Used by the post-int6 expert-ortho hard assertion: if any two experts
    within a component have |cos| > 0.9, those two are effectively the same
    expert — wasted capacity.  This is a much looser bar than max-mean
    (which averages across peers), but it's the right structural check:
    training can't trivially make one worst pair diverge, so a near-1
    value is a true capacity collapse.
    """
    e = groups.shape[0]
    if e < 2:
        return groups.new_zeros(())
    g = groups / (groups.norm(dim=-1, keepdim=True) + eps)
    cos = (g @ g.T).abs()
    cos = cos.masked_fill(torch.eye(e, dtype=torch.bool, device=cos.device), 0.0)
    return cos.max()


def _normalize_k_jitter_weights(values, weights) -> tuple[list[int], list[float]]:
    if values is None:
        raise ValueError("K jitter weights require explicit values")
    vals = [int(v) for v in values]
    ws = [float(w) for w in weights]
    if len(vals) != len(ws):
        raise ValueError("K jitter values and weights must have equal length")
    if len(set(vals)) != len(vals):
        raise ValueError("Weighted K jitter values must be unique; encode probability in weights")
    if any((not math.isfinite(w)) or w < 0.0 for w in ws):
        raise ValueError("K jitter weights must be finite and non-negative")
    weight_sum = sum(ws)
    if weight_sum <= 0.0:
        raise ValueError("K jitter weights must sum to a positive value")
    return vals, [w / weight_sum for w in ws]


def _compute_annealed_k_jitter_weights(
    initial: tuple[float, ...] | list[float],
    final: tuple[float, ...] | list[float],
    step: int,
    total_steps: int,
    start_frac: float,
    end_frac: float,
) -> tuple[float, ...]:
    """iter173 K-jitter weight annealing curriculum (2026-05-17).

    Returns weights interpolated linearly between `initial` and `final` over
    the training window [start_frac, end_frac] · total_steps. Returns
    `initial` unchanged when `final` is empty (annealing disabled), when
    `step` is before `start_frac · total_steps`, or when lengths don't match.
    Returns `final` after `end_frac · total_steps`.

    Hypothesis (iter173): iter172 achieved rho_F=0.77 at K=128 (vs iter163's
    0.92), so the model is "ready" for deeper K. Gradually shifting K-jitter
    probability mass toward larger K teaches the model to use the depth that
    K-sweep eval actually uses, without paying deep-K compute cost for the
    full training run."""
    init_t = tuple(float(w) for w in initial)
    if not final or len(final) != len(init_t) or total_steps <= 0:
        return init_t
    final_t = tuple(float(w) for w in final)
    start_step = int(total_steps * float(start_frac))
    end_step = int(total_steps * float(end_frac))
    if step <= start_step:
        return init_t
    if step >= end_step:
        return final_t
    span = max(1, end_step - start_step)
    alpha = float(step - start_step) / float(span)
    return tuple(
        (1.0 - alpha) * i + alpha * f
        for i, f in zip(init_t, final_t)
    )


def _prefix_anchor_depths(sampled_k: int, values) -> tuple[int, ...]:
    """Return traversed prefix anchors for iter152 multi-anchor supervision.

    Anchors are the sorted positive jitter depths no larger than the sampled
    solve depth, with the sampled endpoint always included. This preserves the
    one-solve forward budget: K=32 supervises {16,24,32}; it does not force a
    separate K=64 pass.
    """
    k = int(sampled_k)
    if k <= 0:
        raise ValueError(f"sampled_k must be positive, got {sampled_k}")
    candidates = (int(v) for v in (values or ()))
    anchors = sorted({v for v in candidates if 0 < v <= k})
    if k not in anchors:
        anchors.append(k)
    return tuple(anchors)


class KShuffleBagSampler:
    def __init__(self, k_min: int, k_max: int, rng: random.Random, *, step: int = 1,
                 values: list[int] | None = None,
                 weights: list[float] | None = None):
        # Explicit `values` overrides range(k_min, k_max+1, step) — lets us
        # express non-uniform K sets like {4, 8, 16} (skipping K=12 because
        # K=8 and K=16 bracket it well; ~7% throughput gain).
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.step = int(step)
        if weights is not None and not values:
            raise ValueError("KShuffleBagSampler weights require explicit values")
        if values and weights is not None:
            self.values, self.weights = _normalize_k_jitter_weights(values, weights)
        else:
            self.values = sorted(set(int(v) for v in values)) if values else None
            self.weights = None
        self.rng = rng
        self._bag: list[int] = []

    def _weighted_bag(self) -> list[int]:
        """Build one exact weighted cycle from normalized weights.

        Over each cycle the empirical K counts match the configured weights
        up to integer rounding (no min-count floor — a configured weight
        smaller than 1/cycle_len rounds down to zero in this cycle, which
        the docstring contract permits and which avoids ≥10× silent
        inflation of small weights).
        """
        assert self.values is not None and self.weights is not None
        cycle_len = max(100, len(self.values))
        raw = [float(w) * float(cycle_len) for w in self.weights]
        counts = [int(math.floor(r)) for r in raw]
        while sum(counts) < cycle_len:
            idx = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
            counts[idx] += 1
        while sum(counts) > cycle_len:
            candidates = [i for i, c in enumerate(counts) if c > 0]
            if not candidates:
                # Unreachable: cycle_len ≥ len(values) ≥ 1 and counts sum
                # ≥ floor(cycle_len * sum(weights)) = cycle_len, so every
                # over-budget loop has a non-zero count to decrement.
                raise AssertionError(
                    "_weighted_bag: no positive count available to decrement "
                    "(cycle_len smaller than the count sum should be impossible)"
                )
            idx = max(candidates, key=lambda i: counts[i] - raw[i])
            counts[idx] -= 1
        bag: list[int] = []
        for value, count in zip(self.values, counts):
            bag.extend([int(value)] * int(count))
        self.rng.shuffle(bag)
        return bag

    def sample(self) -> int:
        if self.weights is not None:
            if not self._bag:
                self._bag = self._weighted_bag()
            return self._bag.pop()
        if not self._bag:
            if self.values is not None:
                self._bag = list(self.values)
            else:
                self._bag = list(range(self.k_min, self.k_max + 1, self.step))
            self.rng.shuffle(self._bag)
        return int(self._bag.pop())

    def reset(self) -> None:
        self._bag.clear()

    def set_weights(self, weights: list[float] | tuple[float, ...]) -> None:
        """Update per-K weights and clear cached bag so next sample() rebuilds
        with the new distribution. Used by iter173 K-jitter annealing curriculum
        to gradually shift sampling mass toward larger K over training. No-op
        when the new weights equal the current ones (caller's responsibility
        to discretize if avoiding bag-rebuild churn matters)."""
        if self.values is None:
            raise ValueError("set_weights requires explicit values")
        self.values, self.weights = _normalize_k_jitter_weights(self.values, list(weights))
        self._bag.clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "k_min": self.k_min,
            "k_max": self.k_max,
            "step": self.step,
            "values": self.values,
            "weights": self.weights,
            "bag": list(self._bag),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.k_min = int(state.get("k_min", self.k_min))
        self.k_max = int(state.get("k_max", self.k_max))
        self.step = int(state.get("step", self.step))
        values = state.get("values", self.values)
        weights = state.get("weights")
        if weights is not None:
            self.values, self.weights = _normalize_k_jitter_weights(values, weights)
        else:
            self.values = [int(v) for v in values] if values is not None else None
            self.weights = None
        self._bag = [int(v) for v in state.get("bag", [])]
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.proj_norm = RMSNorm(bigram_dim) if self.proj is not None else None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(self.proj_norm(h))
        return h * self.scale.to(dtype=h.dtype)


# ---------------------------------------------------------------------------
# Entmax-1.5 (Peters et al. 2019, "Sparse Sequence-to-Sequence Models")
# ---------------------------------------------------------------------------
# Closed-form sparse softmax for α=1.5. Output sums to 1 with exact zeros for
# entries below threshold. Mathematically: p_i = max(0, 0.5*(z_i − τ))^2 with
# τ chosen by sort-and-threshold so the surviving entries sum to 1.
# Differentiable everywhere (Lipschitz Jacobian). Used by SoftDenseRouter via
# a learnable sigmoid blend with softmax (init blend ≈ 0.993 softmax → strict-
# generalization recovery of iter 100b).

def entmax_1p5(z: Tensor, dim: int = -1) -> Tensor:
    """α=1.5 entmax via closed-form sort-and-threshold.

    Numerically stable variant: shift z so max=0 along `dim` before solving.
    Each candidate k (size of support) gives a quadratic in τ:
        sum_{i≤k} (0.5(z_sort_i − τ))^2 = 1
        ⟹  k τ^2 − 2 S τ + (S2 − 4) = 0
    where S = sum z_sort_{1..k}, S2 = sum z_sort_{1..k}^2. Solve, pick the
    smaller root (the relevant one that satisfies τ < z_sort_k), then accept
    the largest k where τ < z_sort_k.
    """
    z = z - z.amax(dim=dim, keepdim=True)
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    K = z.shape[dim]

    # Cumulative sums (S, S2) along `dim`.
    S = z_sorted.cumsum(dim=dim)
    S2 = (z_sorted * z_sorted).cumsum(dim=dim)
    k_range = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    k_shape = [1] * z.ndim
    k_shape[dim] = K
    k = k_range.view(*k_shape)

    # quadratic k τ² − 2 S τ + (S2 − 4) = 0  → τ = (S ± sqrt(S² − k(S2 − 4))) / k
    # Smaller root is τ_low = (S − sqrt(...)) / k.
    # iter 117 v3 NaN fix (2026-04-29): clamp discr at small positive epsilon
    # rather than 0. Backward gradient of sqrt(0) is 1/(2·sqrt(0)) = Inf →
    # IEEE-754 0×Inf = NaN propagates through (1−blend)·p_entmax even when
    # blend=1.0 zeros the forward contribution. ε=1e-6 caps the sqrt gradient
    # at 1/(2·sqrt(1e-6)) = 500 (finite), preserving the chain rule (any
    # finite × 0 = 0, no NaN). Forward bias on sqrt(discr) is ~1e-3 → routing
    # weight bias ≤ 1e-4, BELOW bf16 precision floor (~1e-3). 1e-6 chosen over
    # 1e-8 for additional defensive margin: the smaller value's 5000 max
    # gradient could spike under chain-rule × (1−blend)≈0.5 to 2500, while
    # 1e-6's 500 max gives 250 worst case — 10× safer at no observable cost.
    discr = (S * S - k * (S2 - 4.0)).clamp_min(1e-6)
    tau_k = (S - discr.sqrt()) / k.clamp_min(1.0)

    # Pick largest k with tau_k < z_sorted_k (support size).
    valid = tau_k < z_sorted
    # Convert to support-count via sum along `dim`. (broadcast_dim_safe sum)
    support = valid.to(dtype=torch.long).sum(dim=dim, keepdim=True).clamp_min(1)
    # Gather τ at support − 1 (1-indexed → 0-indexed).
    tau = tau_k.gather(dim, support - 1)

    return (0.5 * (z - tau)).clamp_min(0.0).square()


# ---------------------------------------------------------------------------
# Iter 117b-2: Triton entmax-1.5 kernel + torch.library.custom_op wrapper.
# ---------------------------------------------------------------------------
# Default OFF (`use_entmax_triton: bool = False` in Hyperparameters). When
# enabled via `--use-entmax-triton=1`, the SoftDenseRouter dispatches entmax
# through a Triton kernel rather than the pure-PyTorch closed form above.
# Verified equivalent to deep-spin/entmax reference in
# experiments/test_entmax_triton.py + experiments/test_sparse_dispatch.py
# (both forward and backward closed-form, see Phase A.0/A.2).
# Backward formula (matching deep-spin): grad_z = sqrt(w) * (grad_w - c)
#   where c = sum(sqrt(w) * grad_w) / sum(sqrt(w)).
# torch.library.custom_op wrapper makes the kernel opaque to torch.compile
# (the modern equivalent of the older autograd.Function pattern that broke
# AdaSplash under DDP+RevDEQ+compile per CLAUDE.md Fix #2 NOT VIABLE).

_USE_ENTMAX_TRITON: bool = False  # set by main() from Hyperparameters


def _set_entmax_triton(enabled: bool) -> None:
    """Module-level toggle invoked from main() before model construction.

    Set ONCE at startup; reading is dynamo-folded into the compiled graph.
    """
    global _USE_ENTMAX_TRITON
    _USE_ENTMAX_TRITON = bool(enabled)


try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


if _TRITON_OK:

    # Single source of truth for the entmax-1.5 Triton numerical floor.
    # Forward uses it to clamp the `sqrt(discr)` radicand; backward uses it
    # as a divisor floor for `c = Σ(√w · g) / max(Σ √w, eps)`. Both bound
    # the same near-empty-support singularity — keeping a single constant
    # prevents forward/backward drift on future tuning.
    _ENTMAX_TRITON_EPS: float = 1e-6

    @triton.jit
    def _entmax_1p5_triton_fwd_kernel(
        z_ptr,
        w_ptr,
        eps: tl.constexpr,
        E_BLOCK: tl.constexpr,
    ):
        """One program per row; loads E values into registers (E small fixed).

        Forward: shift z so max=0; sort descending via -tl.sort(-z); prefix
        sums S, S2; candidate τ_k = (S_k - sqrt(max(S²_k - k(S2_k - 4), eps)))/k;
        pick largest k where τ_k < z_sorted_k; output w = max(0.5(z-τ), 0)².
        """
        pid = tl.program_id(axis=0)
        offs = tl.arange(0, E_BLOCK)
        row_off = pid * E_BLOCK + offs
        z = tl.load(z_ptr + row_off).to(tl.float32)
        z = z - tl.max(z, axis=0)
        z_sorted = -tl.sort(-z, dim=0)
        S = tl.cumsum(z_sorted, axis=0)
        S2 = tl.cumsum(z_sorted * z_sorted, axis=0)
        k = (offs + 1).to(tl.float32)
        discr = S * S - k * (S2 - 4.0)
        discr = tl.maximum(discr, eps)
        tau_k = (S - tl.sqrt(discr)) / tl.maximum(k, 1.0)
        valid = tau_k < z_sorted
        support = tl.sum(valid.to(tl.int32), axis=0)
        support = tl.maximum(support, 1)
        target_idx = support - 1
        tau = tl.sum(tl.where(offs == target_idx, tau_k, 0.0), axis=0)
        half = 0.5 * (z - tau)
        half = tl.maximum(half, 0.0)
        w = half * half
        tl.store(w_ptr + row_off, w)

    @triton.jit
    def _entmax_1p5_triton_bwd_kernel(
        w_ptr,
        grad_w_ptr,
        grad_z_ptr,
        eps: tl.constexpr,
        E_BLOCK: tl.constexpr,
    ):
        """Backward closed-form: grad_z = sqrt(w) * (grad_w - c).

        Outside support, w=0 → sqrt(w)=0 → grad_z=0 automatically.
        """
        pid = tl.program_id(axis=0)
        offs = tl.arange(0, E_BLOCK)
        row_off = pid * E_BLOCK + offs
        w = tl.load(w_ptr + row_off).to(tl.float32)
        g = tl.load(grad_w_ptr + row_off).to(tl.float32)
        s = tl.sqrt(tl.maximum(w, 0.0))
        s_sum = tl.sum(s, axis=0)
        sg_sum = tl.sum(s * g, axis=0)
        c = sg_sum / tl.maximum(s_sum, eps)
        grad_z = s * (g - c)
        tl.store(grad_z_ptr + row_off, grad_z)

    def _next_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    @torch.library.custom_op("opg::entmax_1p5_triton", mutates_args=())
    def _entmax_1p5_triton_op(z: torch.Tensor) -> torch.Tensor:
        """Triton-fused entmax-1.5 forward. Input shape (..., E). Pads E to the
        next power of 2 with -inf so non-power-of-2 vocabularies (e.g. E=30 for
        attn+mlp pooled router with num_routed=15 each) work too — entmax of -inf
        is 0, so padding doesn't bias the support. Returns (..., E) original
        shape, weights summing to ≤1 (with exact zeros outside support).
        Iter 117b-2-fix (2026-05-04): pad E=30→32 (was raising ValueError).
        """
        orig_shape = z.shape
        E = orig_shape[-1]
        E_pad = _next_pow2(E) if E > 0 else 1
        z32 = z.detach().to(torch.float32).contiguous().view(-1, E)
        B = z32.shape[0]
        if B == 0:
            w_out = torch.empty_like(z32)
            return w_out.view(orig_shape).to(z.dtype)
        if E_pad != E:
            # Pad with -inf along last dim; entmax(-inf) = 0.
            pad_cols = E_pad - E
            pad_buf = torch.full((B, pad_cols), float("-inf"),
                                  dtype=torch.float32, device=z32.device)
            z32_pad = torch.cat([z32, pad_buf], dim=1)
            w_pad = torch.empty_like(z32_pad)
            _entmax_1p5_triton_fwd_kernel[(B,)](z32_pad, w_pad,
                                                 eps=_ENTMAX_TRITON_EPS,
                                                 E_BLOCK=E_pad)
            w_flat = w_pad[:, :E].contiguous()
        else:
            w_flat = torch.empty_like(z32)
            _entmax_1p5_triton_fwd_kernel[(B,)](z32, w_flat,
                                                 eps=_ENTMAX_TRITON_EPS,
                                                 E_BLOCK=E)
        return w_flat.view(orig_shape).to(z.dtype)

    @_entmax_1p5_triton_op.register_fake
    def _entmax_1p5_triton_op_fake(z: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(z)

    def _entmax_1p5_triton_setup_ctx(ctx, inputs, output):
        ctx.save_for_backward(output)

    def _entmax_1p5_triton_backward(ctx, grad_w: torch.Tensor) -> torch.Tensor:
        (w,) = ctx.saved_tensors
        orig_shape = w.shape
        E = orig_shape[-1]
        E_pad = _next_pow2(E) if E > 0 else 1
        w32 = w.detach().to(torch.float32).contiguous().view(-1, E)
        g32 = grad_w.detach().to(torch.float32).contiguous().view(-1, E)
        B = w32.shape[0]
        if B == 0:
            grad_z = torch.empty_like(w32)
            return grad_z.view(orig_shape).to(grad_w.dtype)
        if E_pad != E:
            # Pad both w and grad with zeros — outside-support contributes nothing.
            pad_cols = E_pad - E
            zeros = torch.zeros((B, pad_cols), dtype=torch.float32, device=w32.device)
            w_pad = torch.cat([w32, zeros], dim=1)
            g_pad = torch.cat([g32, zeros], dim=1)
            grad_z_pad = torch.empty_like(w_pad)
            _entmax_1p5_triton_bwd_kernel[(B,)](
                w_pad, g_pad, grad_z_pad,
                eps=_ENTMAX_TRITON_EPS, E_BLOCK=E_pad)
            grad_z = grad_z_pad[:, :E].contiguous()
        else:
            grad_z = torch.empty_like(w32)
            _entmax_1p5_triton_bwd_kernel[(B,)](
                w32, g32, grad_z, eps=_ENTMAX_TRITON_EPS, E_BLOCK=E)
        return grad_z.view(orig_shape).to(grad_w.dtype)

    _entmax_1p5_triton_op.register_autograd(
        _entmax_1p5_triton_backward,
        setup_context=_entmax_1p5_triton_setup_ctx,
    )


def entmax_1p5_dispatch(z: Tensor, dim: int = -1) -> Tensor:
    """Dispatch wrapper used by SoftDenseRouter. Default = pure-PyTorch
    `entmax_1p5`; when `_USE_ENTMAX_TRITON=True` and Triton is available AND
    `dim == -1` (the only routing case), uses the Triton kernel.
    """
    if _USE_ENTMAX_TRITON and _TRITON_OK and dim == -1:
        return _entmax_1p5_triton_op(z)
    return entmax_1p5(z, dim=dim)


# ---------------------------------------------------------------------------
# Iter 117b-3: capacity-padded sparse MoE dispatch (pure-PyTorch).
# ---------------------------------------------------------------------------
# Default OFF and rejected by `_validate_hyperparameters` for training.
# Replaces dense (E, T, D) expert evaluation with a capacity-padded
# gather/scatter path: each expert processes only its top-K tokens by routing
# weight, K = ceil(C * T / E) static. That hard top-k is not smooth enough for
# the RevDEQ transition, so this remains kernel evidence rather than an active
# train-time optimization.
#
# Equivalence:
#   - C ≥ (1-s)·E + headroom (for sparsity s): bit-identical to dense
#   - C smaller: monotone approximation error bounded by truncated mass
#
# Pure-PyTorch ops (gather + scatter_add + matmul + topk) — torch.compile
# traces natively without custom_op wrapping (Phase A.4 verified).

_USE_SPARSE_DISPATCH: bool = False  # set by main() from Hyperparameters
_SPARSE_DISPATCH_C: float = 4.0     # default capacity factor


def _set_sparse_dispatch(enabled: bool, capacity_factor: float = 4.0) -> None:
    """Module-level toggle invoked from main() before model construction."""
    global _USE_SPARSE_DISPATCH, _SPARSE_DISPATCH_C
    _USE_SPARSE_DISPATCH = bool(enabled)
    _SPARSE_DISPATCH_C = float(capacity_factor)


# iter 118a Phase A3: Triton fused routed-down kernel for MLP-down dispatch.
# When True AND torch.is_grad_enabled() is False (RevDEQ FP iter, eval, K-sweep),
# `MLP.mix_experts` calls `fused_routed_down(h_pre, w_combined, expert_down)`
# instead of the eager (h * w + bmm + sum_e) path. Bench: 2.86× dense, up to
# 7.36× at 87.5% sparse. RevDEQ-safe per H101 (skip ε ≤ ε_bf16).
_USE_UNIFIED_ROUTED_DOWN: bool = False


def _set_unified_routed_down(enabled: bool) -> None:
    """Module-level toggle invoked from main() before model construction."""
    global _USE_UNIFIED_ROUTED_DOWN
    _USE_UNIFIED_ROUTED_DOWN = bool(enabled)


def sparse_moe_dispatch_capacity(
    x: Tensor,            # (T, D)
    w: Tensor,            # (T, E)
    expert_W: Tensor,     # (E, D, R)
    expert_V: Tensor,     # (E, R, D)
    capacity_factor: float,
) -> Tensor:
    """Capacity-padded sparse MoE forward via gather → expert → scatter_add.

    out[t] = Σ_e w[t,e] · (x[t] @ W_e @ V_e) computed with each expert
    evaluated on only its top-K tokens by w[:, e]. K = ceil(C·T/E) is
    constant after graph construction (compile-friendly).

    Bit-identical to dense when K ≥ max-per-expert-utilization. Lower K
    truncates low-weight tokens; error bounded by truncated mass.
    """
    T, D = x.shape
    E, _, R = expert_W.shape
    K = int(math.ceil(float(capacity_factor) * T / E))
    output = torch.zeros_like(x)
    for e in range(E):
        w_e = w[:, e]
        topk_w, topk_idx = w_e.topk(K, dim=0)
        x_e = x.index_select(0, topk_idx)
        h_e = x_e @ expert_W[e]
        y_e = h_e @ expert_V[e]
        y_e = y_e * topk_w.unsqueeze(-1)
        output.index_add_(0, topk_idx, y_e)
    return output


# ---------------------------------------------------------------------------
# SOFT DENSE ROUTER
# ---------------------------------------------------------------------------

# Single source of truth for the EMA-anchored routing-regularizer family
# introduced in iter145 / promoted in iter145r. Each entry drives nine parallel
# code sites — Hyperparameters defaults stay explicit (public knob surface),
# but every internal init / accumulator / cache / annealer / log column that
# fans out across the family iterates over this registry. Adding a future
# member (e.g., a future strict alive-hinge) MUST be a one-line addition here,
# not a multi-site grep-and-paste. See `EXPERIENCE.md#sibling-fanout-dry-gate`.
#
# Convention: for `name`, the trainer expects the matching:
#   - SoftDenseRouter attribute   `_ema_<name>_raw_loss`
#   - GPT coefficient field       `router_ema_<name>_coef`
#   - GPT target store            `_router_ema_<name>_coef_target`
#   - GPT loss-tensor cache       `_router_ema_<name>_loss_t`
#   - GPT coef-eff cache          `_router_ema_<name>_coef_eff_t`
# `sign` is the multiplier applied in the assembled `router_reg_loss` —
# `-1.0` for `specialization` because we maximize KL(token_share || EMA_ref)
# to push routing AWAY from the historical mean, bounded by `log(E)`.
ROUTER_EMA_LOSS_TERMS: tuple[tuple[str, float], ...] = (
    ("alive",          +1.0),
    ("balance",        +1.0),
    ("specialization", -1.0),
)


class SoftDenseRouter(nn.Module):
    """Dense token-local routing over experts (no top-k, no dropping).

    Active default scoring is `dirichlet_ucb`: router logits parameterize
    Softplus evidence for a Dirichlet mean plus an annealed uncertainty bonus.
    Legacy `linear`, `l2`, and `sips` scoring remain available for ablations;
    prototype parameters are allocated only for the prototype-based modes.
    The extra sigmoid gate is also optional and default-off.
    """
    def __init__(self, dim: int, num_experts: int, *,
                 min_share_frac: float = 0.6,
                 # Scoring/gate defaults align with Hyperparameters
                 # (current rescue stack, 2026-05-08).
                 scoring: str = "dirichlet_ucb", health_slices: tuple[int, ...] | None = None,
                 entropy_coef: float = 0.1,
                 dirichlet_ucb_beta: float = 0.5,
                 use_router_sigmoid_gate: bool = False,
                 use_entmax_routing: bool = False,
                 entmax_blend_init_logit: float = 5.0,
                 use_reverse_kl_balance: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.min_share_frac = float(min_share_frac)
        self.scoring = str(scoring)
        self._use_reverse_kl_balance = bool(use_reverse_kl_balance)
        assert self.scoring in ("linear", "l2", "sips", "dirichlet_ucb"), f"unknown scoring: {scoring}"
        self.use_router_sigmoid_gate = bool(use_router_sigmoid_gate)
        self.health_slices = tuple(int(v) for v in (health_slices or (num_experts,)))
        if sum(self.health_slices) != int(num_experts) or any(v <= 0 for v in self.health_slices):
            raise ValueError(f"health_slices={self.health_slices} must partition {num_experts} experts")
        # Tensor buffer for entropy coef so the training loop's annealer can
        # update it without triggering a per-step dynamo recompile. Set via the
        # `entropy_coef` @property so legacy callers can write Python floats.
        self.register_buffer("_entropy_coef", torch.tensor(float(entropy_coef), dtype=torch.float32), persistent=False)
        self.register_buffer("_dirichlet_ucb_beta",
                              torch.tensor(float(dirichlet_ucb_beta), dtype=torch.float32),
                              persistent=False)
        # iter 117 (H87): learnable blend between softmax (init ≈ 1.0) and
        # entmax-1.5 (sparse with exact zeros). NOT a buffer — this is a
        # learnable nn.Parameter that gradient drives. Init logit=+5 →
        # sigmoid(+5) ≈ 0.9933 ≈ pure softmax (strict-gen recovery within
        # bf16 floor). Gradient pushes logit DOWN when sparse routing helps.
        # Stored as fp32 (matches q_gain/gate_bias pattern); placed under
        # CONTROL_TENSOR_PATTERNS for AdamW + scalar_lr routing.
        self.use_entmax_routing = bool(use_entmax_routing)
        if self.use_entmax_routing:
            self._entmax_blend_logit = nn.Parameter(
                torch.tensor(float(entmax_blend_init_logit), dtype=torch.float32))
        else:
            # Register a buffer so the attribute exists but never has gradient.
            # Forward path branches on `use_entmax_routing` so this is unused
            # when entmax routing is off, but keeps `getattr(...)` safe.
            self.register_buffer("_entmax_blend_logit",
                                  torch.tensor(float(entmax_blend_init_logit), dtype=torch.float32),
                                  persistent=False)
        # iter 117 v2: anneal scale, written by training-loop annealer. anneal=0
        # forces blend=1.0 (pure softmax, strict-gen exact); anneal=1 lets
        # blend = sigmoid(blend_logit) (learnable). Tensor-gated to avoid
        # per-step dynamo recompile (mirrors entropy/variance pattern).
        self.register_buffer("_entmax_blend_anneal",
                              torch.tensor(0.0, dtype=torch.float32),
                              persistent=False)
        # Dirichlet evidence follows e_i = Softplus(W_i h + b_i). Keep the
        # legacy no-bias router for the promoted linear path, and add the bias
        # only when the evidential router is explicitly selected.
        self.router = CastedLinear(dim, num_experts, bias=(self.scoring == "dirichlet_ucb"))
        nn.init.normal_(self.router.weight, std=0.01)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        self.score_norm_weight = nn.Parameter(torch.ones(dim))
        if self.use_router_sigmoid_gate:
            self.gate_norm_weight = nn.Parameter(torch.ones(dim))
        else:
            self.gate_norm_weight = None
        # Prototype scoring is an ablation path; the promoted default should
        # not carry unused trainable state or optimizer slots.
        if self.scoring in ("l2", "sips"):
            self.prototypes = nn.Parameter(torch.empty(num_experts, dim))
            with torch.no_grad():
                nn.init.normal_(self.prototypes, std=0.02)
            self.l2_gamma = 1.0
        else:
            self.prototypes = None
        # Lyapunov: no prototype bounding needed (was BallProjection for 1-Lip).
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        # Input-dependent sigmoid gate on routing weights (iter 17, H14).
        # Init fully open: weight=0, bias=5.0 → sigmoid(5)≈0.993.
        # The model can learn to suppress specific experts per-token.
        if self.use_router_sigmoid_gate:
            self.router_gate = CastedLinear(dim, num_experts, bias=True)
            with torch.no_grad():
                self.router_gate.weight.zero_()
                self.router_gate.bias.fill_(5.0)
        else:
            self.router_gate = None
        # 0-d GPU tensor (or None) — see comment at the assignment site in `forward`.
        # Stored on GPU to avoid the dynamo Python-float value-guard that triggered
        # a per-step recompile of the compiled block.forward (profile, 2026-04-28).
        self._router_gate_last_mean: Tensor | None = None
        self._mean_share_last: Tensor | None = None
        self._cv_loss_raw: Tensor | None = None
        self._expert_usage = None
        self._expert_usage_ema = None
        self._expert_usage_ema_min_share = None
        self._expert_entropy = None
        self._expert_sparsity = None
        self._expert_balance_cv = None
        self._expert_total_mass = None
        # Iter145r EMA-anchored routing-regularizer family (alive / balance /
        # specialization). Driven by ROUTER_EMA_LOSS_TERMS — see registry.
        for _name, _ in ROUTER_EMA_LOSS_TERMS:
            setattr(self, f"_ema_{_name}_raw_loss", None)
        # Dirichlet-UCB confidence cache; driven by
        # ROUTER_DIRICHLET_DIAG_TERMS so adding a new diagnostic is a
        # single registry edit.
        self._dirichlet_diag_step: int | None = None
        self._clear_dirichlet_diag()
        # GPU-resident views used by DDP reductions — populated by
        # _record_diagnostics on every rank during eval, avoiding per-forward
        # cpu() syncs on non-master ranks.
        self._expert_usage_gpu: Tensor | None = None
        self.register_buffer("_expert_usage_ema_gpu", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        self.register_buffer("_expert_usage_ema_min_share_gpu", torch.tensor(float("nan"), dtype=torch.float32), persistent=True)
        self.register_buffer("_expert_usage_ema_initialized", torch.tensor(False), persistent=True)
        self._expert_entropy_gpu: Tensor | None = None
        self._expert_balance_cv_gpu: Tensor | None = None
        self._expert_total_mass_gpu: Tensor | None = None
        self._diag_step: int | None = None
        self._expert_usage_ema_decay = 0.99

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        # Silently drop legacy optional-param keys when the current router did
        # not allocate them (Dirichlet-UCB + gate-off default). PyTorch's
        # public `load_state_dict` shallow-copies the input dict before
        # recursing here, so popping from `state_dict` does not mutate the
        # caller's original; no restore needed.
        if self.prototypes is None:
            state_dict.pop(prefix + "prototypes", None)
        if self.gate_norm_weight is None:
            state_dict.pop(prefix + "gate_norm_weight", None)
        if self.router_gate is None:
            state_dict.pop(prefix + "router_gate.weight", None)
            state_dict.pop(prefix + "router_gate.bias", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _component_ranges(self) -> list[tuple[int, int]]:
        out = []
        start = 0
        for width in self.health_slices:
            end = start + int(width)
            out.append((start, end))
            start = end
        return out

    def _normalized_component_shares(self, mean_mass: Tensor) -> Tensor:
        chunks = []
        for start, end in self._component_ranges():
            mass = mean_mass[..., start:end]
            chunks.append(mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        return torch.cat(chunks, dim=-1)

    def _component_uniform_target(self, ref: Tensor) -> Tensor:
        chunks = []
        for start, end in self._component_ranges():
            width = max(end - start, 1)
            chunks.append(torch.full_like(ref[..., start:end], 1.0 / float(width)))
        return torch.cat(chunks, dim=-1)

    def _component_balance_cv(self, share: Tensor) -> Tensor:
        vals = []
        for start, end in self._component_ranges():
            s = share[..., start:end]
            vals.append(s.std(dim=-1, unbiased=False) / s.mean(dim=-1).clamp_min(1e-8))
        return torch.stack(vals).mean()

    def _component_kl(self, p: Tensor, q: Tensor) -> Tensor:
        vals = []
        for start, end in self._component_ranges():
            pp = p[..., start:end].float().clamp_min(1e-8)
            qq = q[..., start:end].float().clamp_min(1e-8)
            vals.append((pp * (pp.log() - qq.log())).sum(dim=-1).mean())
        return torch.stack(vals).sum()

    @property
    def entropy_coef(self) -> float:
        return float(self._entropy_coef.item())

    @entropy_coef.setter
    def entropy_coef(self, value: float) -> None:
        with torch.no_grad():
            self._entropy_coef.fill_(float(value))

    @property
    def entmax_blend_anneal(self) -> float:
        return float(self._entmax_blend_anneal.item())

    @entmax_blend_anneal.setter
    def entmax_blend_anneal(self, value: float) -> None:
        with torch.no_grad():
            self._entmax_blend_anneal.fill_(float(value))

    @property
    def dirichlet_ucb_beta(self) -> float:
        return float(self._dirichlet_ucb_beta.item())

    @dirichlet_ucb_beta.setter
    def dirichlet_ucb_beta(self, value: float) -> None:
        with torch.no_grad():
            self._dirichlet_ucb_beta.fill_(float(value))

    def _clear_dirichlet_diag(self) -> None:
        for _, _attr, _ in ROUTER_DIRICHLET_DIAG_TERMS:
            setattr(self, _attr, None)
        self._dirichlet_diag_step = None

    @torch.no_grad()
    def bias_update(self, *, lr: float, clip: float, distributed: bool) -> None:
        ms = self._mean_share_last
        if ms is None:
            return
        ms = ms.detach().to(dtype=torch.float32)
        if distributed and dist.is_available() and dist.is_initialized():
            ms = ms.clone()
            dist.all_reduce(ms, op=dist.ReduceOp.SUM)
            ms /= float(dist.get_world_size())
        target = self._component_uniform_target(ms)
        lb_vec = self._component_uniform_target(ms) * float(self.min_share_frac)
        if self.min_share_frac > 0.0:
            boost = (lb_vec - ms).clamp_min(0.0)
            # Pure-tensor: conditionally apply boost without .item() GPU→CPU sync.
            boosted = (target + boost).clamp_min(1e-8)
            boosted = self._normalized_component_shares(boosted)
            target = torch.where(boost.sum() > 0.0, boosted, target)
        self.expert_bias.add_(lr * (target - ms))
        if clip > 0:
            self.expert_bias.clamp_(min=-clip, max=clip)

    @torch.no_grad()
    def usage_ema_update(self, *, decay: float | None = None, distributed: bool) -> None:
        """Update persistent expert-usage EMA once per optimizer iteration."""
        ms = self._mean_share_last
        if ms is None:
            return
        ms = ms.detach().to(dtype=torch.float32)
        if distributed and dist.is_available() and dist.is_initialized():
            ms = ms.clone()
            dist.all_reduce(ms, op=dist.ReduceOp.SUM)
            ms /= float(dist.get_world_size())
        if self._expert_usage_ema_gpu.shape != ms.shape or self._expert_usage_ema_gpu.device != ms.device:
            self._expert_usage_ema_gpu = torch.zeros_like(ms)
            self._expert_usage_ema_min_share_gpu = torch.tensor(float("nan"), device=ms.device)
            self._expert_usage_ema_initialized = torch.tensor(False, device=ms.device)
        # Branchless update: per-step CPU sync (`.item()` on the init flag) is
        # cheap individually but pays a D2H stall every optimizer step. Use
        # `torch.where` so the kernel stays on-device; first call lands `ms`
        # exactly, subsequent calls do EMA blend.
        d = self._expert_usage_ema_decay if decay is None else float(decay)
        ema_blend = self._expert_usage_ema_gpu * d + ms * (1.0 - d)
        new_ema = torch.where(self._expert_usage_ema_initialized, ema_blend, ms)
        self._expert_usage_ema_gpu.copy_(new_ema)
        self._expert_usage_ema_initialized.fill_(True)
        self._expert_usage_ema_min_share_gpu.copy_(self._expert_usage_ema_gpu.min())
        self._expert_usage_ema = None
        self._expert_usage_ema_min_share = None

    def forward(self, x: Tensor, *, pre_normed: bool = True) -> Tensor:
        x_n = x  # Caller supplies parameter-free RMS-normalized activations.
        score_weight = self.score_norm_weight.to(dtype=x_n.dtype)
        x_score = x_n * score_weight
        x_gate: Tensor | None = None
        if self.use_router_sigmoid_gate:
            assert self.gate_norm_weight is not None
            gate_weight = self.gate_norm_weight.to(dtype=x_n.dtype)
            x_gate = x_n * gate_weight
        D = x_score.shape[-1]
        p_base: Tensor | None = None
        if self.scoring in ("l2", "sips"):
            assert self.prototypes is not None
            # Phase 6a.3 (reviews 1, 10): replace the O(B·T·E·D) broadcast
            # (x_n.unsqueeze(-2) - c) with a matmul-based distance / cosine,
            # and apply the bounded-prototype projection.  Compute the
            # kernel in fp32 to avoid catastrophic cancellation of
            # ‖x‖² + ‖c‖² − 2·x·c in bf16.
            leading = x_score.shape[:-1]
            x_flat_32 = x_score.reshape(-1, D).float()
            c_32 = self.prototypes.float()
            if self.scoring == "l2":
                # ‖x − c‖² = ‖x‖² + ‖c‖² − 2·x·cᵀ  (fp32-safe GEMM path)
                xc = x_flat_32 @ c_32.t()  # (N, E)
                x_sq = x_flat_32.pow(2).sum(dim=-1, keepdim=True)  # (N, 1)
                c_sq = c_32.pow(2).sum(dim=-1)  # (E,)
                dist_sq = (x_sq + c_sq - 2.0 * xc).clamp_min(0.0)
                route_logits = torch.tanh(-self.l2_gamma * dist_sq)
            else:  # "sips" — cosine similarity via normalized matmul
                x_norm = F.normalize(x_flat_32, dim=-1, eps=1e-12)
                c_norm = F.normalize(c_32, dim=-1, eps=1e-12)
                route_logits = self.l2_gamma * (x_norm @ c_norm.t())
            route_logits = route_logits.reshape(*leading, -1).to(dtype=x.dtype)
            route_logits = route_logits + self.expert_bias.to(dtype=x.dtype)
            self._clear_dirichlet_diag()
        elif self.scoring == "dirichlet_ucb":
            # Evidential routing: logits parameterize non-negative evidence,
            # alpha=e+1 gives a Dirichlet over the expert simplex, and the UCB
            # bonus uses the marginal Beta variance per expert. Routing remains
            # token-local; EMA/balance terms never assign capacity across tokens.
            evidence_logits = self.router(x_score) + self.expert_bias.to(dtype=x.dtype)
            evidence = F.softplus(evidence_logits.float())
            alpha = evidence + 1.0
            strength = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mu = alpha / strength
            var_denom = (strength.square() * (strength + 1.0)).clamp_min(1e-8)
            sigma = torch.sqrt((alpha * (strength - alpha) / var_denom).clamp_min(0.0))
            beta = self._dirichlet_ucb_beta.to(device=x.device, dtype=mu.dtype)
            acq = (mu + beta * sigma).clamp_min(1e-8)
            p_base = acq / acq.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            route_logits = torch.log(acq)
            # Confidence reductions are log-only (consumed by
            # `_router_confidence_stats` which gates on
            # `_dirichlet_diag_step`). Skipping when diagnostics are off
            # eliminates 5+ per-layer reductions on the DEQ unroll hot path.
            if _ROUTER_DIAGNOSTICS_ACTIVE:
                self._dirichlet_strength_last = strength.detach().mean()
                self._dirichlet_uncertainty_last = sigma.detach().mean()
                self._dirichlet_uncertainty_mass_last = (float(alpha.shape[-1]) / strength).detach().mean()
                self._dirichlet_evidence_mean_last = evidence.detach().mean()
                mu_entropy = -(mu * (mu + 1e-8).log()).sum(dim=-1)
                self._dirichlet_mu_entropy_norm_last = (
                    mu_entropy / math.log(max(int(alpha.shape[-1]), 2))
                ).detach().mean()
                self._dirichlet_diag_step = _ROUTER_DIAGNOSTICS_STEP
            else:
                self._clear_dirichlet_diag()
        else:
            # Linear scoring (iter 30-33b baseline).
            route_logits = self.router(x_score) + self.expert_bias.to(dtype=x.dtype)
            self._clear_dirichlet_diag()
        # Softmax allocation (iter 100b promoted baseline). Sparsemax/entmax15/
        # entmax_anneal variants tested in iters 99/101/102 NOT PROMOTED — see
        # H74/H75/H77 in experiments/docs/hypotheses.md. Sigmoid gate is applied
        # multiplicatively (NOT renormalized) so total mass can be < 1, letting
        # the model suppress the mixture near fixed point.
        # iter 117 (H87): when use_entmax_routing=True, p_alloc is a learnable
        # blend of softmax (always-dense) and entmax-1.5 (sparse with exact
        # zeros). Init blend_logit=+5 → sigmoid(5)≈0.993 → essentially softmax
        # at init (strict-gen recovery within bf16 floor). Gradient drives
        # blend_logit DOWN if entmax-1.5's sparse routing benefits val_bpb;
        # variance penalty (iter 111 H83) provides the bootstrap signal that
        # rewards token specialization (which higher entmax weight expresses).
        logits_f32 = route_logits.float()
        p_softmax = torch.softmax(logits_f32, dim=-1) if p_base is None else p_base.to(dtype=logits_f32.dtype)
        if self.use_entmax_routing:
            # iter 117 v2: annealed blend. anneal=0 → effective_blend=1.0 (pure
            # softmax, strict-gen exact). anneal=1 → effective_blend = sigmoid(
            # blend_logit) (learnable). Linear interpolation between the two:
            # effective = 1.0 * (1 − anneal) + sigmoid(blend_logit) * anneal.
            blend_learned = torch.sigmoid(self._entmax_blend_logit)
            anneal = self._entmax_blend_anneal
            blend = (1.0 - anneal) + blend_learned * anneal
            # iter 117b-2: dispatch routes to Triton kernel when
            # `_USE_ENTMAX_TRITON` is set (CLI flag --use-entmax-triton=1);
            # otherwise pure-PyTorch entmax_1p5. Default OFF preserves
            # iter 117b-1 behavior bit-identically.
            p_entmax = entmax_1p5_dispatch(logits_f32, dim=-1)
            p_alloc = blend * p_softmax + (1.0 - blend) * p_entmax
        else:
            p_alloc = p_softmax
        if self.use_router_sigmoid_gate:
            assert self.router_gate is not None and x_gate is not None
            gate_act = torch.sigmoid(self.router_gate(x_gate).float())
            self._router_gate_last_mean = gate_act.detach().mean()
        else:
            gate_act = torch.ones_like(p_alloc, dtype=torch.float32)
            # `_router_gate_last_mean` was set to None in __init__ and stays
            # None on the gate-off path — skip the redundant per-forward write.
        p = (p_alloc * gate_act).to(dtype=x.dtype)
        # When the optional sigmoid gate is enabled, its mean is kept as a 0-d
        # GPU tensor. Materialization to Python happens only at log time.
        # Background on the 0-d-GPU diagnostic pattern (avoids dynamo
        # value-guard recompiles): EXPERIENCE.md#hot-path-sync (iter 84 profile).
        if self.training:
            reduce_dims = tuple(range(p.ndim - 1))
            mean_mass = p.mean(dim=reduce_dims)
            mean_share = self._normalized_component_shares(mean_mass.float()).to(dtype=mean_mass.dtype)
            # Routing CV²: continuous balance pressure (no relu / no target).
            # iter 142b promotion follow-up 2026-05-06 user directive: drop the
            # relu(cv − cv_target)² hinge so the term keeps pushing CV → 0
            # rather than going silent at the old cv_target=0.20. Per-axis story:
            # CV² → balance, pertoken_entropy → sparsity, no antagonism.
            cv_sum = mean_mass.new_zeros(())
            for start, end in self._component_ranges():
                share_s = mean_mass[..., start:end].float()
                cv_s = (share_s.std(dim=-1, unbiased=False)
                        / share_s.mean(dim=-1).clamp_min(1e-8))
                cv_sum = cv_sum + cv_s.pow(2)
            self._cv_loss_raw = cv_sum
            # Per-token entropy minimization. Always-compute (no `.item()` gate
            # so the FP iter stays sync-free). Annealer writes `_entropy_coef`
            # in-place; multiply by 0 is gradient-free.
            p_f = p.float()
            p_norm = p_f / p_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pertoken_ent = -(p_norm * (p_norm + 1e-8).log()).sum(dim=-1).mean()
            self._pertoken_entropy_raw_loss = pertoken_ent
            self._pertoken_entropy_loss = self._entropy_coef.to(dtype=pertoken_ent.dtype) * pertoken_ent
            token_share = self._normalized_component_shares(p_f)
            uniform = self._component_uniform_target(mean_share.float())
            ema_ready = self._expert_usage_ema_initialized.to(device=x.device)
            ema_default = uniform.detach()
            ema_ref = torch.where(ema_ready, self._expert_usage_ema_gpu.to(device=x.device), ema_default)
            ema_ref = self._normalized_component_shares(ema_ref.float()).detach()
            # EMA-anchored balance. `ema_ref` is persistent and detached; the
            # straight-through correction preserves the forward divergence value
            # while routing the gradient through the current token-local router.
            # iter153 default (use_reverse_kl_balance=True): use reverse KL
            # `KL(U || anchor)` so each expert contributes ~U_i regardless of
            # its mass — the gradient on a dead expert (tiny EMA_i) diverges as
            # `-U_i/EMA_i`, matching the min-share gate's asymmetric need.
            # Legacy path (forward KL(anchor || U)) is recoverable via the flag.
            ema_balance_anchor = ema_ref + (mean_share.float() - mean_share.float().detach())
            ema_balance_anchor = self._normalized_component_shares(
                ema_balance_anchor.clamp_min(1e-8))
            if self._use_reverse_kl_balance:
                self._ema_balance_raw_loss = self._component_kl(uniform, ema_balance_anchor)
            else:
                self._ema_balance_raw_loss = self._component_kl(ema_balance_anchor, uniform)
            target_min = (uniform * float(self.min_share_frac)).detach()
            ema_deficit = (target_min - ema_ref).clamp_min(0.0)
            current_deficit = (target_min - mean_share.float()).clamp_min(0.0)
            # EMA-gated current-deficit²: per-expert ema_deficit weight on
            # relu(τ−share)². Experts whose persistent EMA share has recovered
            # contribute zero penalty even if the instant share dips, so the
            # term chases truly persistent under-utilization rather than
            # transient routing noise. NOT the simpler Σ relu(τ−share)² hinge.
            self._ema_alive_raw_loss = (ema_deficit * current_deficit.pow(2)).sum()
            # Loss assembles as `- coef * KL(token || ema_ref)`: minimizing the
            # negative KL ⇔ MAXIMIZING KL ⇔ pushing per-token routing AWAY from
            # the historical EMA mean (specialization). Bounded above by log(E)
            # since ema_ref is a normalized distribution clamped at 1e-8. Sign
            # lives in ROUTER_EMA_LOSS_TERMS, not at the call site.
            self._ema_specialization_raw_loss = self._component_kl(token_share, ema_ref)
            self._mean_share_last = mean_share.detach()
            self._maybe_record_diag(p.detach(), reduce_dims, clear_on_skip=False)
        else:
            self._cv_loss_raw = torch.tensor(0.0, device=x.device)
            self._pertoken_entropy_raw_loss = torch.tensor(0.0, device=x.device)
            self._pertoken_entropy_loss = torch.tensor(0.0, device=x.device)
            zero = torch.tensor(0.0, device=x.device)
            for _name, _ in ROUTER_EMA_LOSS_TERMS:
                setattr(self, f"_ema_{_name}_raw_loss", zero)
            self._mean_share_last = None
            self._maybe_record_diag(p.detach(), tuple(range(p.ndim - 1)),
                                    clear_on_skip=True)
        return p

    @dynamo_disable
    def _record_diagnostics(self, routed_mass: Tensor, reduce_dims: tuple[int, ...]) -> None:
        # GPU-only recording — no .cpu()/.item() sync on any rank during the
        # forward pass.  Both usage + CV stay as detached GPU tensors; master
        # additionally keeps the per-token entropy GPU tensor for log time.
        # The Python list/float views used by format_expert_info are
        # materialized lazily via _materialize_diag_lists(), which should be
        # called exactly once per log site (not per forward).
        mean_mass = routed_mass.mean(dim=reduce_dims).float()
        mean_share = self._normalized_component_shares(mean_mass)
        self._expert_usage_gpu = mean_share.detach()
        self._expert_balance_cv_gpu = self._component_balance_cv(mean_share).detach()
        self._expert_total_mass_gpu = mean_mass.sum().detach()
        distributed = dist.is_available() and dist.is_initialized()
        is_master = (not distributed) or dist.get_rank() == 0
        if is_master:
            token_share = routed_mass.float() / routed_mass.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
            per_token_ent = -(token_share * (token_share + 1e-8).log()).sum(-1)
            self._expert_entropy_gpu = per_token_ent.mean().detach()
        else:
            self._expert_entropy_gpu = None
        # Clear any previously-materialized list-form diagnostics.  Log sites
        # that need Python-native values call _materialize_diag_lists() first.
        self._expert_usage = None
        self._expert_usage_ema = None
        self._expert_usage_ema_min_share = None
        self._expert_entropy = None
        self._expert_sparsity = None
        self._expert_balance_cv = None
        self._expert_total_mass = None
        self._diag_step = _ROUTER_DIAGNOSTICS_STEP

        # Do not materialize list-form diagnostics here: this function can be
        # called 2*K times per DEQ solve. Materialization happens either at
        # log sites (format_expert_info) or at the end of the DEQ solve.

    @dynamo_disable
    def _materialize_diag_lists(self) -> None:
        """Lazily convert GPU-resident diagnostic tensors to Python values.

        One .cpu() sync per call covers usage + entropy + CV.  Safe to call
        multiple times — subsequent calls short-circuit if the list is already
        materialized for the current _diag_step.
        """
        if self._expert_usage_gpu is not None and self._expert_usage is None:
            self._expert_usage = self._expert_usage_gpu.detach().float().cpu().tolist()
        if bool(self._expert_usage_ema_initialized.item()):
            if self._expert_usage_ema is None:
                self._expert_usage_ema = self._expert_usage_ema_gpu.detach().float().cpu().tolist()
            if self._expert_usage_ema_min_share is None:
                self._expert_usage_ema_min_share = float(self._expert_usage_ema_min_share_gpu.item())
        if self._expert_balance_cv_gpu is not None and self._expert_balance_cv is None:
            self._expert_balance_cv = float(self._expert_balance_cv_gpu.item())
        if self._expert_total_mass_gpu is not None and self._expert_total_mass is None:
            self._expert_total_mass = float(self._expert_total_mass_gpu.item())
        if self._expert_entropy_gpu is not None and self._expert_entropy is None:
            ent = float(self._expert_entropy_gpu.item())
            self._expert_entropy = ent
            self._expert_sparsity = 1.0 - (ent / max(math.log(float(self.num_experts)), 1e-8))

    @dynamo_disable
    def _maybe_record_diag(self, routed_mass: Tensor,
                           reduce_dims: tuple[int, ...],
                           clear_on_skip: bool) -> None:
        """Eager-only diagnostic record/clear — moves the
        `_ROUTER_DIAGNOSTICS_ACTIVE` flag check OUT of the compiled forward.

        Pre-fix the forward had:
            if _should_diag(self.training):
                self._record_diagnostics(...)
                if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and not bool(_DEQ_SOLVE_ACTIVE):
                    self._materialize_diag_lists()
            else:
                <clear python-list views>
        Both `_should_diag` and `_ROUTER_DIAGNOSTICS_ACTIVE` are read inside the
        compiled SoftDenseRouter.forward; dynamo guards on each global, and any
        toggle (every ~10 train steps for log emission) invalidates the cache
        slot → recompile of the parent compiled block. Profile (2026-04-28)
        showed this was a dominant residual recompile vector even after value-
        guard fixes (see commit 7343c06 H78). Hoisting the gate into a
        `@dynamo_disable` helper makes the call opaque to dynamo, so the flag
        toggle has no effect on the compiled graph cache.

        `clear_on_skip=True`: in eval mode (training=False) the original code
        cleared the python-list-form diagnostics when not active.
        """
        with torch.no_grad():
            if _should_diag(self.training):
                self._record_diagnostics(routed_mass, reduce_dims)
                if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and not bool(_DEQ_SOLVE_ACTIVE):
                    self._materialize_diag_lists()
            elif clear_on_skip:
                self._expert_usage = None
                self._expert_usage_ema = None
                self._expert_usage_ema_min_share = None
                self._expert_entropy = None
                self._expert_sparsity = None
                self._expert_balance_cv = None
                self._expert_total_mass = None
                self._expert_usage_gpu = None
                self._expert_entropy_gpu = None
                self._expert_balance_cv_gpu = None
                self._expert_total_mass_gpu = None
                self._diag_step = None


# ---------------------------------------------------------------------------
# NSA — Native Sparse Attention (iter 106; arxiv:2502.11089)
# ---------------------------------------------------------------------------
#
# Two branches are implemented here (selection branch deferred to iter 106b):
#   compression — mean-pool K/V over fixed-size sliding blocks, attend to the
#     downsampled stream (global, coarse coverage).
#   sliding-window — last W tokens (local recency).
#
# Mixed via a per-expert-per-head learnable softmax gate stored on
# CausalSelfAttention as `nsa_branch_gate` (shape `(E*H, 2)`). Init at zero →
# uniform softmax → equal mixing of the two branches.
#
# Strict-generalization (CLAUDE.md §11): with `nsa_compress_block_size=1,
# nsa_compress_block_sliding_stride=1, nsa_sliding_window_size=T,
# nsa_branch_gate_init=0` both branches collapse to full causal SDPA and the
# uniform mixer averages two identical outputs → the iter 100b forward map is
# recovered exactly.

def _nsa_compression_branch(
    q_full: Tensor, k_full: Tensor, v_full: Tensor, *,
    block_size: int, stride: int,
) -> Tensor:
    """Compression branch: mean-pool K/V over sliding blocks, then SDPA against
    the (B, H_kv, n_blocks_pad, d) compressed K/V with a rectangular causal
    mask. A leading zero-block is prepended so queries at t < block_size − 1
    (which observe no completed compression block) still see one valid key
    column — keeps softmax well-defined without per-row masking gymnastics.
    """
    B, H, T, d = q_full.shape
    H_kv = k_full.shape[1]
    device = q_full.device
    dtype = q_full.dtype

    # Edge case: sequence shorter than one compression block. Compression has
    # nothing to summarize; return zeros so the mixer drops this branch's
    # contribution. Sliding-window branch handles all attention in this regime.
    if T < block_size:
        return torch.zeros_like(q_full)

    # n_blocks: (T - block_size) // stride + 1 sliding windows fit in T tokens.
    n_blocks = (T - block_size) // stride + 1
    # unfold last-but-one dim → (B, H_kv, n_blocks, d, block_size); mean over
    # block_size axis.
    k_unf = k_full.unfold(dimension=2, size=block_size, step=stride).mean(dim=-1)
    v_unf = v_full.unfold(dimension=2, size=block_size, step=stride).mean(dim=-1)

    # Prepend a zero K/V column so every query has at least one finite-score
    # key (avoids softmax(-inf) → NaN on early rows that observe no completed
    # block). The zero column's score is biased by `_NSA_NEG` so it dominates
    # the softmax ONLY when every real block is masked out (early rows when
    # block_size > 1) — otherwise its weight is numerically zero, leaving the
    # forward map unchanged from a pure "real-keys-only" SDPA.
    zero_kv = torch.zeros(B, H_kv, 1, d, device=device, dtype=dtype)
    k_pad = torch.cat([zero_kv, k_unf], dim=2)
    v_pad = torch.cat([zero_kv, v_unf], dim=2)

    # Rectangular causal mask: query t observes compressed block i iff the
    # block has fully ended (last source position i*stride+block_size-1 ≤ t).
    block_ends = torch.arange(n_blocks, device=device) * stride + block_size - 1
    q_pos = torch.arange(T, device=device)
    valid = q_pos[:, None] >= block_ends[None, :]                  # (T, n_blocks)

    # Mask convention chosen so the zero column has zero weight when ANY real
    # block is observable (its logit −1e9 is dominated by any real q·k score)
    # but FULL weight when every real block is masked (the masked reals get
    # true −inf, so the zero column's −1e9 is the only finite logit).
    NEG_INF = torch.tensor(float("-inf"), device=device, dtype=dtype)
    NEG_LARGE = torch.tensor(-1e9, device=device, dtype=dtype)
    ZERO = torch.tensor(0.0, device=device, dtype=dtype)
    real_bias = torch.where(valid, ZERO, NEG_INF)                  # (T, n_blocks)
    zero_col_bias = NEG_LARGE.expand(T, 1)                         # (T, 1) — finite but tiny
    attn_mask = torch.cat([zero_col_bias, real_bias], dim=1)[None, None, :, :]  # (1, 1, T, 1+n_blocks)

    return F.scaled_dot_product_attention(
        q_full, k_pad, v_pad,
        attn_mask=attn_mask, is_causal=False,
        enable_gqa=(H_kv != H),
    )


def _nsa_sliding_branch(
    q_full: Tensor, k_full: Tensor, v_full: Tensor, *, window_size: int,
) -> Tensor:
    """Sliding-window branch: SDPA with a band-causal mask
    `M[t, k] = (max(0, t-W+1) ≤ k ≤ t)`. The first row (t=0) always has at
    least one valid key (k=0) so softmax is well-defined.
    """
    B, H, T, d = q_full.shape
    H_kv = k_full.shape[1]
    device = q_full.device
    dtype = q_full.dtype

    # Fast path: if W ≥ T the band degenerates to full causal — let SDPA's
    # built-in is_causal=True path take over (FA backend, no explicit mask).
    if window_size >= T:
        return F.scaled_dot_product_attention(
            q_full, k_full, v_full, attn_mask=None, is_causal=True,
            enable_gqa=(H_kv != H),
        )

    q_pos = torch.arange(T, device=device)[:, None]                # (T, 1)
    k_pos = torch.arange(T, device=device)[None, :]                # (1, T)
    valid = (k_pos <= q_pos) & (k_pos >= q_pos - window_size + 1)  # (T, T)
    attn_mask = torch.where(
        valid,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )[None, None, :, :]                                            # (1, 1, T, T)

    return F.scaled_dot_product_attention(
        q_full, k_full, v_full, attn_mask=attn_mask, is_causal=False,
        enable_gqa=(H_kv != H),
    )


def _nsa_attention(
    q_full: Tensor, k_full: Tensor, v_full: Tensor, *,
    branch_gate: Tensor,
    compress_block_size: int, compress_stride: int,
    sliding_window_size: int,
) -> Tensor:
    """Two-branch NSA mixer. `branch_gate` has shape `(H, 2)` (logits per head
    over [compression, sliding]). Selection branch deferred to iter 106b — the
    CausalSelfAttention call site sets `nsa_num_selected_blocks=0` for now.
    """
    out_c = _nsa_compression_branch(
        q_full, k_full, v_full,
        block_size=compress_block_size, stride=compress_stride,
    )
    out_s = _nsa_sliding_branch(
        q_full, k_full, v_full, window_size=sliding_window_size,
    )

    # Per-head softmax mixer. branch_gate (H, 2) → (H, 2) softmax.
    gate = F.softmax(branch_gate.to(q_full.dtype), dim=-1)
    w_c = gate[:, 0][None, :, None, None]                          # (1, H, 1, 1)
    w_s = gate[:, 1][None, :, None, None]
    return w_c * out_c + w_s * out_s


# ---------------------------------------------------------------------------
# MLA + GATED ATTENTION
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Fully independent expert MLA: per-expert Q/K/V/K_rope/Wo.

    Each expert has its own complete MLA pipeline with zero shared params:
      1. Per-expert Q (low-rank): dim → rank → H*d_head + H (gate logits)
      2. Per-expert KV compression (low-rank): dim → kv_rank → kv_latent
      3. Per-expert KV decompression: RMS statistic + separate K/V scales → K_nope, V
      4. Per-expert K_rope (low-rank): dim → kr_rank → H_kv*rope_dim
      5. Per-expert Wo (low-rank): D → wo_rank → D (mixes heads per expert)

    Expert index is packed into the head dimension (E×H query heads,
    E×H_kv KV heads) for a single FlashAttention call.  GQA ratio is
    H/H_kv (same as before).
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, kv_latent_dim: int = 0, num_experts: int = 16,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None,
                 use_nsa_attention: bool = False,
                 nsa_compress_block_size: int = 32,
                 nsa_compress_block_sliding_stride: int = 16,
                 nsa_sliding_window_size: int = 256,
                 nsa_branch_gate_init: float = 0.0,
                 use_sparse_attn_head_gate: bool = False,
                 sparse_attn_gate_window: int = 12,
                 sparse_attn_gate_scale: float = 1.0,
                 sparse_attn_gate_factor: float = 2.0,
                 sparse_attn_gate_init_std: float = 0.0,
                 use_rr_attention: bool = False,
                 rr_stride: int = 8,
                 rr_block_size: int = 64,
                 rr_tau: float = 0.95,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(dim // max(num_experts, 1), 1)
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else dim // 2
        # iter 106 NSA settings — see H86 + module-level NSA helpers above.
        self.use_nsa_attention = bool(use_nsa_attention)
        self.nsa_compress_block_size = int(nsa_compress_block_size)
        self.nsa_compress_block_sliding_stride = int(nsa_compress_block_sliding_stride)
        self.nsa_sliding_window_size = int(nsa_sliding_window_size)
        self.use_rr_attention = bool(use_rr_attention)
        self.rr_stride = int(rr_stride)
        self.rr_block_size = int(rr_block_size)
        self.rr_tau = float(rr_tau)
        self.use_sparse_attn_head_gate = bool(use_sparse_attn_head_gate)
        self.rope_dim = self.head_dim // 2
        self.nope_dim = self.head_dim - self.rope_dim
        self.kv_rank = max(self.kv_latent_dim // 8, 32)

        # Per-expert Q: low-rank dim → expert_rank → (H*d_head + H).
        # The +H appended to Q output provides per-head gate logits (gated attn).
        q_out_dim = num_heads * self.head_dim + num_heads
        self.expert_q_down = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_q_up = nn.Parameter(torch.empty(num_experts, q_out_dim, self.expert_rank))
        self.q_down_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.q_up_norm_weight = nn.Parameter(torch.ones(num_experts, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_q_down.data[e])
            # Zero-init the gate logit rows so gates start at sigmoid(0) = 0.5.
            nn.init.xavier_uniform_(self.expert_q_up.data[e, :num_heads * self.head_dim, :])
            self.expert_q_up.data[e, num_heads * self.head_dim:, :].zero_()

        # Per-expert KV: low-rank dim → kv_rank → kv_latent_dim.
        self.expert_kv_a = nn.Parameter(torch.empty(num_experts, self.kv_rank, dim))
        self.expert_kv_b = nn.Parameter(torch.empty(num_experts, self.kv_latent_dim, self.kv_rank))
        self.kv_a_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.kv_b_norm_weight = nn.Parameter(torch.ones(num_experts, self.kv_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_kv_a.data[e])
            nn.init.xavier_uniform_(self.expert_kv_b.data[e])

        # Per-expert KV decompression from latent (full MLA per expert).
        # K and V own separate learned input scales after the shared RMS statistic.
        self.k_nope_in_norm_weight = nn.Parameter(torch.ones(num_experts, self.kv_latent_dim))
        self.v_in_norm_weight = nn.Parameter(torch.ones(num_experts, self.kv_latent_dim))
        self.expert_k_nope = nn.Parameter(
            torch.empty(num_experts, num_kv_heads * self.nope_dim, self.kv_latent_dim))
        self.expert_v = nn.Parameter(
            torch.empty(num_experts, num_kv_heads * self.head_dim, self.kv_latent_dim))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_k_nope.data[e])
            nn.init.xavier_uniform_(self.expert_v.data[e])

        # Per-expert K_rope: low-rank dim → rank_kr → H_kv*rope_dim.
        # Position-dependent but per-expert — each expert attends to positions
        # differently.  Low-rank (rank=32) keeps params manageable.
        self.kr_rank = max(num_kv_heads * self.rope_dim // 6, 16)
        # Iter 104 (AdaSplash α-entmax) DROPPED 2026-04-29 — see module-level
        # comment near `_unwrap_compiled_module` for the full root-cause +
        # decision rationale. The `attn_alpha_niter` and `_attn_alpha`
        # attributes were removed in the same cleanup; the SDPA call site
        # below now goes straight to dense `F.scaled_dot_product_attention`.
        self.expert_kr_a = nn.Parameter(torch.empty(num_experts, self.kr_rank, dim))
        self.expert_kr_b = nn.Parameter(
            torch.empty(num_experts, num_kv_heads * self.rope_dim, self.kr_rank))
        self.kr_a_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.kr_b_norm_weight = nn.Parameter(torch.ones(num_experts, self.kr_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_kr_a.data[e])
            nn.init.xavier_uniform_(self.expert_kr_b.data[e])

        # Per-expert output projection Wo: low-rank D → rank_wo → D.
        # Mixes heads per expert (DeepSeek MLA Wo equivalent).
        self.wo_rank = max(dim // 12, 32)
        self.expert_wo_down = nn.Parameter(torch.empty(num_experts, self.wo_rank, dim))
        self.expert_wo_up = nn.Parameter(torch.empty(num_experts, dim, self.wo_rank))
        self.wo_down_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.wo_up_norm_weight = nn.Parameter(torch.ones(num_experts, self.wo_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_wo_down.data[e])
            nn.init.xavier_uniform_(self.expert_wo_up.data[e])

        # Per-expert-per-head gains and gates (E*H entries each).
        self.q_gain = nn.Parameter(torch.full((num_experts * num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dim, base=rope_base)
        self.gate_bias = nn.Parameter(torch.zeros(num_experts * num_heads, dtype=torch.float32))
        if self.use_sparse_attn_head_gate:
            from experiments.components.sparse_attn_head_gate import SparseAttnHeadGate
            self.sparse_attn_head_gate = SparseAttnHeadGate(
                num_heads=num_experts * num_heads,
                gate_window=int(sparse_attn_gate_window),
                scale=float(sparse_attn_gate_scale),
                gate_factor=float(sparse_attn_gate_factor),
                init_std=float(sparse_attn_gate_init_std),
            )
        # iter 106 NSA: per-expert-per-head two-branch (compression, sliding)
        # softmax mixer logits. Allocated only when NSA is on; zero-param when
        # off so disabled-default has no opt-coverage / quantization fallout.
        # CLAUDE.md §6.2 hard constraint: every learned per-head param has
        # leading dim E to keep experts fully independent. Stored as float32
        # (matches q_gain / gate_bias) — auto-routes to scalar AdamW group.
        if self.use_nsa_attention:
            self.nsa_branch_gate = nn.Parameter(torch.full(
                (num_experts * num_heads, 2), float(nsa_branch_gate_init),
                dtype=torch.float32,
            ))

        self.attn_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        # Per-expert learned pre-RMS scales for Q/K components. RMS math is
        # shared and parameter-free; every learned scale stays expert-local.
        self.q_rope_norm_weight = nn.Parameter(torch.ones(num_experts, self.rope_dim))
        self.q_nope_norm_weight = nn.Parameter(torch.ones(num_experts, self.nope_dim))
        self.k_rope_norm_weight = nn.Parameter(torch.ones(num_experts, self.rope_dim))
        self.k_nope_norm_weight = nn.Parameter(torch.ones(num_experts, self.nope_dim))
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None
        # 0-d GPU tensor (or None) — same rationale as
        # SoftDenseRouter._router_gate_last_mean (avoid dynamo Python-float guards).
        self._attn_gate_last_mean: Tensor | None = None

    @staticmethod
    def _rms_scale(x: Tensor, weight: Tensor, *, expert_dim: int = 0, eps: float = 1e-6) -> Tensor:
        # 2026-04-28 Fix #1 (profile-driven): replace inline `x.pow(2).mean(-1)
        # .add(eps).rsqrt() * x` with the fused F.rms_norm kernel. The inline
        # pattern produced 6 distinct triton_per_fused__to_copy_add_mean_mul_pow_rsqrt
        # variants in profile_v12 totalling ~31% of CUDA time. F.rms_norm dispatches
        # to a single well-tuned fused kernel (PyTorch 2.5+).
        # Per-expert weight is applied as a separate elementwise mul because the
        # weight shape (E, D) does not match F.rms_norm's `weight` arg requirement
        # of broadcasting to `normalized_shape` (D,).
        y = F.rms_norm(x, (x.size(-1),), eps=eps)
        shape = [1] * y.ndim
        shape[expert_dim] = weight.shape[0]
        shape[-1] = weight.shape[-1]
        return y * weight.to(dtype=y.dtype).reshape(*shape)

    def forward_experts(self, x_n: Tensor) -> Tensor:
        """Compute per-expert attention outputs via head-packed SDPA.

        Args:
            x_n: Normalized input (B, T, D).

        Returns:
            Per-expert attention outputs (B, T, E, D).
        """
        B, T, D = x_n.shape
        E = self.num_experts
        H, H_kv = self.num_heads, self.num_kv_heads
        d = self.head_dim
        R_q, R_kv = self.expert_rank, self.kv_rank
        N = B * T
        dtype = x_n.dtype

        # --- Per-expert Q (low-rank) ---
        x_flat = x_n.reshape(N, D)
        q_down = self.expert_q_down.to(dtype=dtype) * self.q_down_norm_weight.to(dtype=dtype).unsqueeze(1)
        q_h = (x_flat @ q_down.reshape(E * R_q, D).t()).view(N, E, R_q).permute(1, 0, 2)
        q_h = self._rms_scale(q_h, self.q_up_norm_weight)
        q_up = self.expert_q_up.to(dtype=dtype).transpose(1, 2)
        q_and_gate = torch.bmm(q_h, q_up)  # (E, N, H*d+H)

        q_raw = q_and_gate[:, :, :H * d].reshape(E, B, T, H, d)
        gate_logits = q_and_gate[:, :, H * d:].reshape(E, B, T, H, 1)

        q_rope = q_raw[..., :self.rope_dim]
        q_nope = q_raw[..., self.rope_dim:]
        q_rope = self._rms_scale(q_rope, self.q_rope_norm_weight)
        q_nope = self._rms_scale(q_nope, self.q_nope_norm_weight)

        # --- Per-expert KV (low-rank latent) ---
        kv_a = self.expert_kv_a.to(dtype=dtype) * self.kv_a_norm_weight.to(dtype=dtype).unsqueeze(1)
        kv_b = self.expert_kv_b.to(dtype=dtype).transpose(1, 2)
        kv_h = (x_flat @ kv_a.reshape(E * R_kv, D).t()).view(N, E, R_kv).permute(1, 0, 2)
        kv_h = self._rms_scale(kv_h, self.kv_b_norm_weight)
        kv_latent = torch.bmm(kv_h, kv_b)  # (E, N, kv_lat)

        # Per-expert MLA decompress: kv_latent → K_nope, V
        # Parameter-free RMS statistic with separate learned K and V input scales.
        kv_flat = kv_latent.reshape(E * N, self.kv_latent_dim)
        # Fix #1 (profile-driven, see _rms_scale comment): fused F.rms_norm.
        kv_unit = F.rms_norm(kv_flat, (self.kv_latent_dim,), eps=1e-6).reshape(E, N, self.kv_latent_dim)
        kv_k = kv_unit * self.k_nope_in_norm_weight.to(dtype=dtype).unsqueeze(1)
        kv_v = kv_unit * self.v_in_norm_weight.to(dtype=dtype).unsqueeze(1)
        ek = self.expert_k_nope.to(dtype=dtype)
        ev = self.expert_v.to(dtype=dtype)
        k_nope = torch.bmm(kv_k, ek.transpose(1, 2)).reshape(E, B, T, H_kv, self.nope_dim)
        v = torch.bmm(kv_v, ev.transpose(1, 2)).reshape(E, B, T, H_kv, d)

        # Per-expert K_rope (low-rank): each expert has independent position attention
        kr_a = self.expert_kr_a.to(dtype=dtype) * self.kr_a_norm_weight.to(dtype=dtype).unsqueeze(1)
        kr_b = self.expert_kr_b.to(dtype=dtype).transpose(1, 2)
        kr_h = (x_flat @ kr_a.reshape(E * self.kr_rank, D).t()).view(N, E, self.kr_rank).permute(1, 0, 2)
        kr_h = self._rms_scale(kr_h, self.kr_b_norm_weight)
        k_rope_raw = torch.bmm(kr_h, kr_b)  # (E, N, H_kv*rope)
        k_rope = k_rope_raw.reshape(E, B, T, H_kv, self.rope_dim)

        k_rope = self._rms_scale(k_rope, self.k_rope_norm_weight)
        k_nope = self._rms_scale(k_nope, self.k_nope_norm_weight)

        # --- RoPE ---
        cos, sin = self.rotary(T, x_n.device, q_rope.dtype)
        q_rope_p = q_rope.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, self.rope_dim)
        q_rope_p = apply_rotary_emb(q_rope_p, cos, sin)

        k_rope_p = k_rope.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, self.rope_dim)
        k_rope_p = apply_rotary_emb(k_rope_p, cos, sin)

        # --- Assemble full Q, K ---
        q_nope_p = q_nope.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, self.nope_dim)
        q_full = torch.cat([q_rope_p, q_nope_p], dim=-1)
        q_full = q_full * self.q_gain.to(dtype=dtype)[None, :, None, None]

        k_nope_p = k_nope.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, self.nope_dim)
        k_full = torch.cat([k_rope_p, k_nope_p], dim=-1)

        v_full = v.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, d)

        # --- Head-packed SDPA / NSA ---
        if self.use_nsa_attention:
            # iter 106: 2-branch NSA replaces dense causal SDPA.
            y = _nsa_attention(
                q_full, k_full, v_full,
                branch_gate=self.nsa_branch_gate,
                compress_block_size=self.nsa_compress_block_size,
                compress_stride=self.nsa_compress_block_sliding_stride,
                sliding_window_size=self.nsa_sliding_window_size,
            )
        elif self.use_rr_attention:
            from experiments.components.rr_attention import rr_attention
            k_use, v_use = k_full, v_full
            if H_kv != H:
                rep = H // H_kv
                k_use = k_full.repeat_interleave(rep, dim=1)
                v_use = v_full.repeat_interleave(rep, dim=1)
            y = rr_attention(
                q_full, k_use, v_use,
                stride=self.rr_stride,
                block_size=self.rr_block_size,
                tau=self.rr_tau,
                causal=True,
            )
        else:
            try:
                y = F.scaled_dot_product_attention(
                    q_full, k_full, v_full, attn_mask=None, is_causal=True,
                    enable_gqa=(H_kv != H),
                )
            except TypeError:
                k_use, v_use = k_full, v_full
                if H_kv != H:
                    rep = H // H_kv
                    k_use = k_full.repeat_interleave(rep, dim=1)
                    v_use = v_full.repeat_interleave(rep, dim=1)
                    y = F.scaled_dot_product_attention(q_full, k_use, v_use, attn_mask=None, is_causal=True)

        if self.use_sparse_attn_head_gate:
            y = self.sparse_attn_head_gate(x_n, y.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # --- Gated attention ---
        gate_logits_p = gate_logits.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, 1)
        gate_act = torch.sigmoid(
            gate_logits_p.to(dtype=y.dtype)
            + self.gate_bias.to(dtype=y.dtype)[None, :, None, None]
        )
        y = y * gate_act

        # 0-d GPU tensor; same pattern as SoftDenseRouter._router_gate_last_mean
        # at L1496 — avoids dynamo per-step value-guard recompiles. Materialization
        # to Python float happens at log time.
        self._attn_gate_last_mean = gate_act.detach().float().mean()

        # --- Per-expert output projection Wo (low-rank, mixes heads) ---
        y = y.reshape(B, E, H, T, d).permute(0, 3, 1, 2, 4).reshape(B * T, E, D)
        # (E, N, D) → down (E, N, rank_wo) → up (E, N, D)
        wo_d = self.expert_wo_down.to(dtype=y.dtype)
        wo_u = self.expert_wo_up.to(dtype=y.dtype).transpose(1, 2)  # (E, rank_wo, D)
        y_e = y.permute(1, 0, 2)  # (E, N, D)
        y_e = self._rms_scale(y_e, self.wo_down_norm_weight)
        y_h = torch.bmm(y_e, wo_d.transpose(1, 2))  # (E, N, rank_wo)
        y_h = self._rms_scale(y_h, self.wo_up_norm_weight)
        y_out = torch.bmm(y_h, wo_u)  # (E, N, D)
        y = y_out.permute(1, 0, 2).reshape(B, T, E, D)

        # Eager-only diagnostic — see helper docstring.
        self._capture_attn_out_ortho(y)

        return y

    @dynamo_disable
    def _capture_attn_out_ortho(self, y: Tensor) -> None:
        """Eager-only capture of attention-expert output orthogonality
        (`_out_ortho_cos_sim`).

        Profile run (2026-04-28, post-Fix-#1) identified THIS site as the
        dominant remaining recompile vector — dynamo's [9/0] graph break at
        L1892 with reason `_ROUTER_DIAGNOSTICS_ACTIVE` flag-flip. The flag
        toggles every ~10 train steps (log boundary) AND the `.item()` materi-
        alization is itself a sync + value-guard. Wrapping in `@dynamo_disable`
        keeps the compiled `forward_experts` graph stable: dynamo sees this
        method call as an opaque op (one fixed graph break per forward, no
        guards on internal state), and the flag check + cosine sim run in pure
        eager mode when actually needed.
        """
        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            with torch.no_grad():
                mu_out = y.detach().float().mean(dim=(0, 1))  # (E, D)
                self._out_ortho_cos_sim = float(max_pairwise_abs_cosine(mu_out).item())

    def forward(self, x: Tensor) -> Tensor:
        raise RuntimeError("Use Block.forward()")

    @property
    def attn_gate(self) -> Tensor:
        return self.gate_bias

    def get_expert_diagnostics(self) -> dict:
        diag: dict = {}
        r = self.attn_router
        if r._expert_usage is not None:
            diag["usage"] = r._expert_usage
            diag["entropy"] = r._expert_entropy
            diag["balance_cv"] = r._expert_balance_cv
        if self._out_ortho_cos_sim is not None:
            diag["ortho_cos_sim"] = self._out_ortho_cos_sim
        return diag


# ---------------------------------------------------------------------------
# MLP with SwiGLU
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU-gated MLP expert bank (doc §3.4)."""
    def __init__(self, dim: int, mlp_mult: float, num_experts: int = 16,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None,
                 *, use_perdim_gate: bool = False, perdim_gate_rank: int = 16):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(hidden // max(num_experts, 1), 1)
        self.expert_gate = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_fc = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.gate_in_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.fc_in_norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        # Layout (E, D, R) matches repo tests/experiments; computation uses
        # a transpose view to (E, R, D) for batched GEMMs.
        self.expert_down = nn.Parameter(torch.empty(num_experts, dim, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_gate.data[e])
            nn.init.xavier_uniform_(self.expert_fc.data[e])
            nn.init.xavier_uniform_(self.expert_down.data[e])
        self.mlp_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None
        # Per-expert RMSNorm on hidden — no shared learned weights across experts.
        # Shape: (E, R) scale weight, applied after normalization.
        self.hidden_norm_weight = nn.Parameter(torch.ones(num_experts, self.expert_rank))
        # iter176 (2026-05-18) per-dim sigmoid gate per expert. When enabled,
        # each expert's output is multiplied element-wise by a per-expert per-
        # token gate computed from the input x via a low-rank LoRA projection:
        #   gate_e(x) = σ(V_e (U_e x) + b_e) ∈ (0, 1)^D
        # Default OFF preserves iter172 behavior; non-zero rank materializes
        # the U / V / b parameters and triggers the eager gated path.
        self.use_perdim_gate = bool(use_perdim_gate)
        self.perdim_gate_rank = int(perdim_gate_rank) if self.use_perdim_gate else 0
        if self.use_perdim_gate:
            R_g = self.perdim_gate_rank
            # iter176b (2026-05-19) — STRICT-GENERALIZATION init (CLAUDE.md rule):
            #   U init xavier (LoRA "down" projection, like LoRA A)
            #   V init ZEROS  (LoRA "up" projection, like LoRA B)
            #   bias init +6.0
            # Forward: gate = σ(V·(U·x) + b) = σ(0·U·x + 6.0) = σ(6.0) ≈ 0.9975
            # → gate ≈ 1.0 at init → e_i_gated ≈ e_i → forward ≡ iter172 baseline
            # Gradient at init: ∂L/∂V = ∂L/∂pre · U·x ≠ 0 (V trains immediately);
            # ∂L/∂U = ∂L/∂pre · V = 0 (U trains only after V drifts from zero)
            # — matches the standard LoRA convention (Hu et al. 2021).
            # The previous iter176 init (xavier U/V + bias=0 → σ ≈ 0.5) HALVED
            # each expert's output at init, violating strict-generalization: the
            # functional class contained iter172 but the INIT did not match it.
            # iter176 ran with that bug and showed transient early lead followed
            # by lead evaporation as the regularizer-warmup completed (the gate
            # was double-counted with diversity loss). iter176b: optimizer is
            # free to learn V ≠ 0 ONLY if it reduces loss; if redundant with
            # the diversity loss, V stays near zero and val_bpb matches iter172.
            # Per CLAUDE.md strict-generalization: at worst this matches iter172,
            # never regresses.
            self.perdim_gate_u = nn.Parameter(torch.empty(num_experts, R_g, dim))
            self.perdim_gate_v = nn.Parameter(torch.zeros(num_experts, dim, R_g))
            self.perdim_gate_bias = nn.Parameter(torch.full((num_experts, dim), 6.0))
            for e in range(num_experts):
                nn.init.xavier_uniform_(self.perdim_gate_u.data[e])

    def mix_experts(self, x: Tensor, w: Tensor, *,
                    num_shared: int = 0, shared_gate: Tensor | None = None) -> Tensor:
        # Caller must pass x already parameter-free RMS-normalized by Block.
        B, T, D = x.shape
        E, R = self.num_experts, self.expert_rank
        N = B * T
        x_flat = x.reshape(N, D)
        # iter 117b-3 X2 step 3: when _USE_SPARSE_DISPATCH is set, the routed
        # expert path takes the capacity-padded gather/scatter route instead
        # of the dense bmm-then-sum below. Shared experts (if any) still run
        # densely. Module-level flag is read once; dynamo constant-folds.
        # See sparse_moe_dispatch_capacity (module-level helper) for the
        # standalone reference implementation; this in-line variant is
        # specialized to the SwiGLU MLP layout (separate gate/fc projections
        # with per-expert RMS norm on the hidden) rather than the simpler
        # (W, V) two-matrix form of the standalone helper.
        if _USE_SPARSE_DISPATCH:
            return self._mix_experts_sparse(
                x_flat, w, num_shared=num_shared, shared_gate=shared_gate,
                B=B, T=T, D=D, E=E, R=R, N=N,
            )
        G = (
            self.expert_gate.to(dtype=x_flat.dtype)
            * self.gate_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(E * R, D)
        Fm = (
            self.expert_fc.to(dtype=x_flat.dtype)
            * self.fc_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(E * R, D)
        gate = x_flat @ G.t()
        fc = x_flat @ Fm.t()
        h = F.silu(gate) * fc
        h = h.view(N, E, R)
        # Per-expert RMSNorm (no shared weights across experts).
        # Fix #1 (profile-driven, see _rms_scale comment): fused F.rms_norm.
        h = F.rms_norm(h, (R,), eps=1e-6) * self.hidden_norm_weight.to(dtype=h.dtype)  # (E, R) broadcasts over (N, E, R)
        # Phase 9 iter 51: shared experts (sigmoid-gated) + routed experts
        S = int(num_shared)

        # iter 118a Phase A3: when _USE_UNIFIED_ROUTED_DOWN flag is set AND
        # autograd is disabled (RevDEQ no_grad FP iter, eval, K-sweep), call
        # the fused Triton kernel instead of (h * w + bmm + sum_e). Combines
        # shared sigmoid gate + routed router weights into one (N, E) weight
        # tensor; kernel handles the per-expert weighted contraction with
        # H101-safe sparsity skip. Bench: 2.86× dense, 7.36× at 87.5% sparse.
        # Training (grad enabled) and any non-bf16 path fall through to eager.
        # iter176: per-dim gate forces eager (kernel doesn't support per-expert
        # per-dim output mask); only kicks in when use_perdim_gate=False.
        if (
            _USE_UNIFIED_ROUTED_DOWN
            and not torch.is_grad_enabled()
            and h.dtype == torch.bfloat16
            and self.expert_down.dtype == torch.bfloat16
            and D % 16 == 0
            and h.shape[2] <= 128       # R_PAD register-pressure bound
            and E <= 64
            and not self.use_perdim_gate
        ):
            from experiments.components.fused_routed_down import fused_routed_down
            if S > 0 and shared_gate is not None:
                num_routed = E - S
                w_combined = torch.empty(N, E, dtype=h.dtype, device=h.device)
                w_combined[:, :S] = shared_gate.reshape(N, S).to(dtype=h.dtype)
                w_combined[:, S:] = w.reshape(N, num_routed).to(dtype=h.dtype)
            else:
                w_combined = w.reshape(N, E).to(dtype=h.dtype)
            out = fused_routed_down(h, w_combined, self.expert_down)
            # Diagnostic capture is eager-only; skipped in fused path.
            return out.reshape(B, T, D)

        # Eager fallback (training path with grad, or non-aligned shapes).
        if S > 0 and shared_gate is not None:
            num_routed = E - S
            # Shared: gated by per-token sigmoid (same gate as attention path)
            g_s_flat = shared_gate.reshape(N, S).to(dtype=h.dtype)
            h_shared = h[:, :S, :] * g_s_flat.unsqueeze(-1)  # (N, S, R)
            # Routed: weighted by router output
            w_flat = w.reshape(N, num_routed).to(dtype=h.dtype)
            h_routed = h[:, S:, :] * w_flat.unsqueeze(-1)  # (N, num_routed, R)
            h = torch.cat([h_shared, h_routed], dim=1)  # (N, E, R)
        else:
            w_flat = w.reshape(N, E).to(dtype=x_flat.dtype)
            h = h * w_flat.unsqueeze(-1)
        Dwn_T = self.expert_down.to(dtype=x_flat.dtype).transpose(1, 2)  # (E, R, D)
        out_e = torch.bmm(h.transpose(0, 1), Dwn_T)  # (E, N, D)
        # iter176: per-dim sigmoid gate per expert. Each expert's output is
        # multiplied element-wise by a per-token per-dim gate computed from
        # the (pre-RMS-normalized) input x via a low-rank LoRA projection.
        # Both the routing weight (already applied to h above) and this gate
        # are multiplicative on out_e — they commute and we apply gate post-
        # routing so the dtype-matched bmm above is unchanged. Per CLAUDE.md
        # "Expert independence is hard": U/V/b are per-expert parameters.
        if self.use_perdim_gate:
            U = self.perdim_gate_u.to(dtype=x_flat.dtype)        # (E, R_g, D)
            V = self.perdim_gate_v.to(dtype=x_flat.dtype)        # (E, D, R_g)
            b = self.perdim_gate_bias.to(dtype=x_flat.dtype)     # (E, D)
            # x_flat: (N, D); inter: (E, N, R_g); pre: (E, N, D)
            inter = torch.einsum("nd,erd->enr", x_flat, U)
            pre = torch.einsum("enr,edr->end", inter, V) + b.unsqueeze(1)
            gate_mask = torch.sigmoid(pre)
            out_e = out_e * gate_mask
        out = out_e.sum(dim=0)  # (N, D)

        # Eager-only diagnostic — see helper docstring.
        self._capture_mlp_out_ortho(h, N, E, R)

        return out.reshape(B, T, D)

    def _mix_experts_sparse(
        self,
        x_flat: Tensor,         # (N, D) — pre-RMS-normalized input
        w: Tensor,              # router output for routed experts (N or B*T, num_routed)
        *,
        num_shared: int,
        shared_gate: Tensor | None,
        B: int, T: int, D: int, E: int, R: int, N: int,
    ) -> Tensor:
        """Capacity-padded sparse MoE forward (iter 117b-3 X2 step 3).

        Replaces the dense (E, N, D) bmm of `mix_experts` with a per-expert
        gather → expert → scatter_add pattern. Each routed expert processes
        only its top-K tokens by routing weight (K = ceil(C·N/num_routed)).
        Shared experts (if any) still run densely on all tokens, then the
        two outputs sum.

        Bit-equivalence: when K ≥ max-per-expert-utilization, output is
        operator-identical to the dense path within fp32 reorder noise.
        Capacity formula: C ≥ (1-s)·E + headroom for sparsity s. With
        entmax routing at s ≈ 0.80, default C=4.0 gives K = N/E·4 = 4N/16
        → ~25% of tokens per expert vs all N in dense (~4× compute saving
        on routed-expert linears).

        Diagnostic capture (`_capture_mlp_out_ortho`) is skipped in sparse
        mode — it requires the full per-expert h tensor that doesn't exist
        here. The router-level diagnostics (cv, entropy, usage) come from
        the SoftDenseRouter forward, which is unchanged.
        """
        S = int(num_shared)
        num_routed = E - S
        out = torch.zeros(N, D, dtype=x_flat.dtype, device=x_flat.device)

        # Path A: shared experts — dense compute on all N tokens (always-on).
        if S > 0 and shared_gate is not None:
            G_s = (
                self.expert_gate[:S].to(dtype=x_flat.dtype)
                * self.gate_in_norm_weight[:S].to(dtype=x_flat.dtype).unsqueeze(1)
            ).reshape(S * R, D)
            F_s = (
                self.expert_fc[:S].to(dtype=x_flat.dtype)
                * self.fc_in_norm_weight[:S].to(dtype=x_flat.dtype).unsqueeze(1)
            ).reshape(S * R, D)
            gate_s = x_flat @ G_s.t()
            fc_s = x_flat @ F_s.t()
            h_s = F.silu(gate_s) * fc_s
            h_s = h_s.view(N, S, R)
            h_s = F.rms_norm(h_s, (R,), eps=1e-6) * self.hidden_norm_weight[:S].to(dtype=h_s.dtype)
            g_s_flat = shared_gate.reshape(N, S).to(dtype=h_s.dtype)
            h_s = h_s * g_s_flat.unsqueeze(-1)  # (N, S, R)
            Dwn_s_T = self.expert_down[:S].to(dtype=x_flat.dtype).transpose(1, 2)  # (S, R, D)
            out_s_e = torch.bmm(h_s.transpose(0, 1), Dwn_s_T)  # (S, N, D)
            out = out + out_s_e.sum(dim=0)  # (N, D)

        # Path B: routed experts — sparse capacity-padded compute.
        if num_routed > 0:
            K = int(math.ceil(_SPARSE_DISPATCH_C * N / max(num_routed, 1)))
            K = max(min(K, N), 1)  # clamp to [1, N]
            w_flat = w.reshape(N, num_routed).to(dtype=x_flat.dtype)
            for e_idx in range(num_routed):
                e_global = e_idx + S  # offset for shared experts
                w_e = w_flat[:, e_idx]
                topk_w, topk_idx = w_e.topk(K, dim=0)
                x_e = x_flat.index_select(0, topk_idx)  # (K, D)
                # Per-expert weights with prenorm scale folded in
                ge = (
                    self.expert_gate[e_global].to(dtype=x_e.dtype)
                    * self.gate_in_norm_weight[e_global].to(dtype=x_e.dtype).unsqueeze(0)
                )  # (R, D)
                fe = (
                    self.expert_fc[e_global].to(dtype=x_e.dtype)
                    * self.fc_in_norm_weight[e_global].to(dtype=x_e.dtype).unsqueeze(0)
                )  # (R, D)
                gate_e = x_e @ ge.t()  # (K, R)
                fc_e = x_e @ fe.t()    # (K, R)
                h_e = F.silu(gate_e) * fc_e  # (K, R)
                h_e = F.rms_norm(h_e, (R,), eps=1e-6) * self.hidden_norm_weight[e_global].to(dtype=h_e.dtype)
                h_e = h_e * topk_w.unsqueeze(-1)  # (K, R) routing-weighted
                de = self.expert_down[e_global].to(dtype=x_e.dtype).t()  # (R, D)
                out_e = h_e @ de  # (K, D)
                out.index_add_(0, topk_idx, out_e)

        return out.reshape(B, T, D)

    @dynamo_disable
    def _capture_mlp_out_ortho(self, h: Tensor, N: int, E: int, R: int) -> None:
        """Eager-only capture of MLP-expert output orthogonality.
        Same dynamo-disable rationale as
        CausalSelfAttention._capture_attn_out_ortho — moves the
        `_ROUTER_DIAGNOSTICS_ACTIVE` flag check + `.item()` materialization
        out of the compiled `mix_experts` graph.
        """
        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            with torch.no_grad():
                mu_h = h.reshape(N, E, R).mean(dim=0).to(dtype=torch.float32)
                down_T = self.expert_down.to(dtype=mu_h.dtype).transpose(1, 2)
                mu_out = torch.einsum("er,erd->ed", mu_h, down_T)
                # Max pairwise |cos| across expert pairs — near-duplicate check.
                self._out_ortho_cos_sim = float(max_pairwise_abs_cosine(mu_out).item())

    def forward(self, x: Tensor) -> Tensor:
        raise RuntimeError("Use Block.forward()")

    @property
    def router(self) -> SoftDenseRouter:
        return self.mlp_router

    def get_expert_diagnostics(self) -> dict:
        diag: dict = {}
        r = self.mlp_router
        if r._expert_usage is not None:
            diag["usage"] = r._expert_usage
            diag["entropy"] = r._expert_entropy
            diag["balance_cv"] = r._expert_balance_cv
        if self._out_ortho_cos_sim is not None:
            diag["ortho_cos_sim"] = self._out_ortho_cos_sim
        return diag


# ---------------------------------------------------------------------------
# FSQ
# ---------------------------------------------------------------------------

def _fsq_ste(x: Tensor, num_levels: int, training: bool) -> Tensor:
    x_bounded = torch.tanh(x)
    step = 2.0 / (num_levels - 1)
    if training:
        x_q = torch.round(x_bounded / step) * step
        return x_bounded + (x_q - x_bounded).detach()
    return torch.round(x_bounded / step) * step


# ---------------------------------------------------------------------------
# MoS HEAD (CTP + NTP)
# ---------------------------------------------------------------------------

class MoSHead(nn.Module):
    """Mixture-of-Softmaxes dual head with FSQ.

    Iter 94: when `use_ctp=False`, the CTP head's parameter banks are not
    allocated; `forward()` returns `(log_p_ntp, log_p_ntp)` and the CTP-side
    balance / ortho diagnostics stay at identity defaults. This is the clean
    one-variable ablation of dual-head MoS.
    """
    def __init__(self, d_model: int, vocab_size: int, rank: int = 256,
                 num_shared: int = 2, num_specialized: int = 1, fsq_levels: int = 8,
                 use_ctp: bool = False, logit_softcap: float = 0.0,
                 mos_output_diversity_kind: str = "cosine"):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.num_shared = num_shared
        self.num_specialized = num_specialized
        self.num_experts = num_shared + num_specialized
        self.fsq_levels = fsq_levels
        self.use_ctp = bool(use_ctp)
        self.logit_softcap = float(logit_softcap)
        self.mos_output_diversity_kind = str(mos_output_diversity_kind)
        if self.mos_output_diversity_kind not in ("frobenius", "cosine"):
            raise ValueError(
                f"mos_output_diversity_kind must be 'frobenius' or 'cosine'; "
                f"got {self.mos_output_diversity_kind!r}"
            )
        self.gate_ntp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
        self.gate_ntp_norm_weight = nn.Parameter(torch.ones(d_model))
        self.ntp_a_norm_weight = nn.Parameter(torch.ones(self.num_experts, d_model))
        self.A_ntp_shared = nn.Parameter(torch.empty(num_shared, d_model, rank))
        self.A_ntp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        self.B_NTP = nn.Parameter(torch.empty(self.num_experts, vocab_size, rank))
        self.ntp_rank_norm_weight = nn.Parameter(torch.ones(self.num_experts, rank))
        if self.use_ctp:
            self.gate_ctp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
            self.gate_ctp_norm_weight = nn.Parameter(torch.ones(d_model))
            self.ctp_a_norm_weight = nn.Parameter(torch.ones(self.num_experts, d_model))
            self.A_ctp_shared = nn.Parameter(torch.empty(num_shared, d_model, rank))
            self.A_ctp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
            self.B_denoise = nn.Parameter(torch.empty(self.num_experts, vocab_size, rank))
            self.ctp_rank_norm_weight = nn.Parameter(torch.ones(self.num_experts, rank))
        self._diag_step: int | None = None
        self._ctp_ortho_out: Tensor | None = None
        self._ntp_ortho_out: Tensor | None = None
        self._diversity_aux_enabled = False
        # GPU-resident diagnostics for DDP-safe reductions (avoid per-forward CPU sync).
        self._ctp_expert_usage_gpu: Tensor | None = None
        self._ntp_expert_usage_gpu: Tensor | None = None
        self._ctp_expert_balance_cv_gpu: Tensor | None = None
        self._ntp_expert_balance_cv_gpu: Tensor | None = None
        self._init_params()

    def _init_params(self):
        gates = [self.gate_ntp] + ([self.gate_ctp] if self.use_ctp else [])
        for gate in gates:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.zeros_(gate.bias)
        A_mats = [self.A_ntp_shared, self.A_ntp]
        if self.use_ctp:
            A_mats += [self.A_ctp_shared, self.A_ctp]
        for A in A_mats:
            for e in range(A.shape[0]):
                nn.init.xavier_uniform_(A.data[e])
        B_mats = [self.B_NTP] + ([self.B_denoise] if self.use_ctp else [])
        for B in B_mats:
            for e in range(B.shape[0]):
                nn.init.xavier_uniform_(B.data[e])

    def get_head_orthogonality(self, head: str) -> float:
        # GATE metric: max pairwise |cos| across MoS head experts (shared + specialized),
        # computed on the A weight matrices directly.  We DO NOT fall back to the
        # cached ortho_out (which is now a max_mean LOSS value, used for smoother
        # gradient signal during training).  Loss and gate are deliberately decoupled:
        #   - LOSS: max_mean (smoother gradient, every pair contributes)
        #   - GATE: max_pairwise (clean duplicate detection at threshold 0.9)
        if self.num_shared + self.num_specialized < 2:
            return 0.0
        if head == "ctp" and not self.use_ctp:
            return 0.0
        A_shared = self.A_ctp_shared if head == "ctp" else self.A_ntp_shared
        A_spec = self.A_ctp if head == "ctp" else self.A_ntp
        with torch.no_grad():
            groups = torch.cat([A_shared, A_spec], dim=0).float().reshape(self.num_shared + self.num_specialized, -1)
            return float(max_pairwise_abs_cosine(groups).item())

    def _fsq(self, x: Tensor) -> Tensor:
        if self.fsq_levels <= 1:
            return x  # Phase 9 iter 62 (H53): disabled FSQ, keep low-rank only
        return _fsq_ste(x, self.fsq_levels, self.training)

    def _project_A(self, x: Tensor, A_all: Tensor) -> Tensor:
        E = A_all.shape[0]
        # A_all is stored as (E, D, R). Flatten as (E*R, D) so each
        # contiguous rank row matches einsum("nd,edr->ner").
        A_flat = A_all.permute(0, 2, 1).reshape(E * self.rank, self.d_model)
        return (x.to(A_all.dtype) @ A_flat.t()).view(x.shape[0], E, self.rank)

    def _head_forward(self, x: Tensor, gate: nn.Linear, gate_norm_weight: Tensor,
                      A_shared: Tensor, A_spec: Tensor, a_norm_weight: Tensor,
                      B: Tensor, rank_norm_weight: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        N = x.shape[0]
        E_s, E_p = A_shared.shape[0], A_spec.shape[0]
        E = E_s + E_p
        x = _rms_unit(x)
        alpha = F.softmax(gate(x * gate_norm_weight.to(dtype=x.dtype)).float(), dim=-1)  # (N, E)
        log_w = alpha.clamp(min=1e-8).log()          # (N, E)

        # Vectorized: concat all A matrices → single batched GEMM
        A_all = torch.cat([A_shared, A_spec], dim=0)  # (E, D, R)
        A_all = A_all * a_norm_weight.to(dtype=A_all.dtype).unsqueeze(-1)
        t_all = self._project_A(x, A_all)  # (N, E, R)

        # Ortho diagnostic from mean expert projections (kept as readout only).
        mu_groups = t_all.float().mean(dim=0)  # (E, R)
        ortho_out = max_mean_abs_offdiag_cosine(mu_groups) if E >= 2 else x.new_zeros(())

        # Per-token expert-state diversity loss. Same kind as block-side
        # (cosine default). Coef applied externally as `mos_output_diversity_coef`.
        if self.training and E >= 2 and bool(getattr(self, "_diversity_aux_enabled", False)):
            t_f = t_all.float()  # (N, E, R)
            R = t_f.shape[-1]
            if self.mos_output_diversity_kind == "cosine":
                t_n = F.normalize(t_f, dim=-1, eps=1e-6)
                G = torch.einsum("ner,nfr->nef", t_n, t_n)
                off = G - torch.eye(E, device=G.device, dtype=G.dtype)
                diversity = off.pow(2).sum(dim=(-2, -1)).mean() / float(max(E * (E - 1), 1))
            else:  # frobenius (E²-normalized for magnitude parity with cosine)
                G = torch.einsum("ner,nfr->nef", t_f, t_f) / float(R)
                target = torch.eye(E, device=G.device, dtype=G.dtype) / float(E)
                diversity = (G - target).pow(2).sum(dim=(-2, -1)).mean() / float(E * E)
        else:
            diversity = x.new_zeros(())

        # FSQ + per-expert rank prenorm + logits:
        # (N, E, R) → FSQ/RMS → per-expert B → (N, E, V)
        u_all = self._fsq(t_all)
        # Fix #1 (profile-driven, see _rms_scale comment): fused F.rms_norm.
        u_all = F.rms_norm(u_all, (u_all.size(-1),), eps=1e-6) * rank_norm_weight.to(dtype=u_all.dtype).unsqueeze(0)
        logits_all = torch.bmm(
            u_all.permute(1, 0, 2).to(B.dtype),
            B.transpose(1, 2),
        ).permute(1, 0, 2).float()  # (N, E, V)

        # Iter 122 / H93: logit softcap (Gemma2-style). Reshape extreme tails
        # via tanh saturation. Strict-gen recovery at logit_softcap == 0.
        if self.logit_softcap > 0:
            sc = self.logit_softcap
            logits_all = sc * torch.tanh(logits_all / sc)

        # Mixture of softmaxes in log space (vectorized logaddexp)
        log_p_experts = F.log_softmax(logits_all, dim=-1)  # (N, E, V)
        log_p_weighted = log_w.unsqueeze(-1) + log_p_experts  # (N, E, V)
        log_p_unnorm = torch.logsumexp(log_p_weighted, dim=1)  # (N, V)
        log_p = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=-1, keepdim=True)

        return log_p, alpha, ortho_out, diversity

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        orig_shape = h.shape[:-1]
        x = h.reshape(-1, self.d_model)
        log_p_n, alpha_n, ortho_ntp, div_ntp = self._head_forward(
            x, self.gate_ntp, self.gate_ntp_norm_weight,
            self.A_ntp_shared, self.A_ntp, self.ntp_a_norm_weight,
            self.B_NTP, self.ntp_rank_norm_weight,
        )
        if self.use_ctp:
            log_p_d, alpha_d, ortho_ctp, div_ctp = self._head_forward(
                x, self.gate_ctp, self.gate_ctp_norm_weight,
                self.A_ctp_shared, self.A_ctp, self.ctp_a_norm_weight,
                self.B_denoise, self.ctp_rank_norm_weight,
            )
        else:
            log_p_d = log_p_n
            alpha_d = alpha_n
            ortho_ctp = x.new_zeros(())
            div_ctp = x.new_zeros(())
        self._ctp_ortho_out = ortho_ctp
        self._ntp_ortho_out = ortho_ntp

        if self.training:
            # MoS load CV²: continuous balance pressure (no relu / no target).
            # iter 142b promotion follow-up 2026-05-06: same change as the
            # router-side CV — drop the relu hinge for uniform always-on
            # balance pressure under the unified coef=1.0 reg stack.
            cv_sum = torch.tensor(0.0, device=x.device)
            alphas = [alpha_n] if not self.use_ctp else [alpha_d, alpha_n]
            for alpha_soft in alphas:
                mean_a = alpha_soft.mean(dim=0).float()
                cv_a = mean_a.std(unbiased=False) / mean_a.mean().clamp_min(1e-8)
                cv_sum = cv_sum + cv_a.pow(2)
            self._cv_loss_raw = cv_sum
            # MoS per-token expert-state diversity (replaces mean-projection ortho).
            self._diversity_loss = (div_ntp + div_ctp) if self.use_ctp else div_ntp
        else:
            self._cv_loss_raw = torch.tensor(0.0, device=x.device)
            self._diversity_loss = torch.tensor(0.0, device=x.device)

        # Eager-only diagnostic capture — see helper docstring.
        self._capture_mos_diagnostics(alpha_d, alpha_n)
        return log_p_d.view(*orig_shape, -1), log_p_n.view(*orig_shape, -1)

    @dynamo_disable
    def _capture_mos_diagnostics(self, alpha_d: Tensor, alpha_n: Tensor) -> None:
        """Eager-only MoS expert-usage diagnostic capture.

        The compiled `mos_head.forward` (5.45× speedup, L3847) was guarded on
        `_ROUTER_DIAGNOSTICS_ACTIVE` and `_should_diag`. Each toggle of the
        global flag invalidated the cache slot. Hoisting into a dynamo-disabled
        helper makes the call opaque (one fixed graph break per forward, no
        guards on internal state). Same pattern as
        SoftDenseRouter._maybe_record_diag and
        CausalSelfAttention._capture_attn_out_ortho.
        """
        with torch.no_grad():
            if _should_diag(self.training):
                a_d = alpha_d.detach()
                a_n = alpha_n.detach()
                mean_d = a_d.mean(dim=0).float().detach()
                mean_n = a_n.mean(dim=0).float().detach()
                self._ctp_expert_usage_gpu = mean_d
                self._ntp_expert_usage_gpu = mean_n
                self._ctp_expert_balance_cv_gpu = (mean_d.std() / mean_d.mean().clamp_min(1e-8)).detach()
                self._ntp_expert_balance_cv_gpu = (mean_n.std() / mean_n.mean().clamp_min(1e-8)).detach()
                self._diag_step = _ROUTER_DIAGNOSTICS_STEP
                # Default: no Python-native diagnostics (avoid CPU sync).
                self._ctp_expert_usage = None
                self._ntp_expert_usage = None
                self._ctp_expert_balance_cv = None
                self._ntp_expert_balance_cv = None
                distributed = dist.is_available() and dist.is_initialized()
                is_master = (not distributed) or dist.get_rank() == 0
                if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and is_master:
                    # Backward-compat: retain list-form usage on master when explicitly enabled.
                    self._ctp_expert_usage = mean_d.cpu().tolist()
                    self._ntp_expert_usage = mean_n.cpu().tolist()
                    self._ctp_expert_balance_cv = float(self._ctp_expert_balance_cv_gpu.float().item())
                    self._ntp_expert_balance_cv = float(self._ntp_expert_balance_cv_gpu.float().item())
            else:
                self._ctp_expert_usage_gpu = None
                self._ntp_expert_usage_gpu = None
                self._ctp_expert_balance_cv_gpu = None
                self._ntp_expert_balance_cv_gpu = None
                self._ctp_expert_usage = None
                self._ntp_expert_usage = None
                self._ctp_expert_balance_cv = None
                self._ntp_expert_balance_cv = None
                self._diag_step = None


# ---------------------------------------------------------------------------
# BLOCK (Attention + MLP)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    # iter 96 baseline: num_experts=16 (H71 PROMOTED ★).
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, kv_latent_dim: int = 0,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 num_experts: int = 16, num_shared_experts: int = 0,
                 # Defaults below mirror Hyperparameters (current rescue stack,
                 # 2026-05-08).
                 router_scoring: str = "dirichlet_ucb",
                 router_pertoken_entropy_coef: float = 0.1,
                 router_dirichlet_ucb_beta: float = 0.5,
                 use_router_sigmoid_gate: bool = False,
                 use_entmax_routing: bool = False,
                 entmax_blend_init_logit: float = 5.0,
                 use_reverse_kl_balance: bool = True,
                 use_nsa_attention: bool = False,
                 nsa_compress_block_size: int = 32,
                 nsa_compress_block_sliding_stride: int = 16,
                 nsa_sliding_window_size: int = 256,
                 nsa_branch_gate_init: float = 0.0,
                 use_sparse_attn_head_gate: bool = False,
                 sparse_attn_gate_window: int = 12,
                 sparse_attn_gate_scale: float = 1.0,
                 sparse_attn_gate_factor: float = 2.0,
                 sparse_attn_gate_init_std: float = 0.0,
                 use_rr_attention: bool = False,
                 rr_stride: int = 8,
                 rr_block_size: int = 64,
                 rr_tau: float = 0.95,
                 chained_stages_preset: str | None = None,
                 use_expert_perdim_gate: bool = False,
                 expert_perdim_gate_rank: int = 16,
                 **kwargs):
        super().__init__()
        # T_θ(z, x₀) = B̄ ⊙ RMSUnit(x₀) ⊙ x0_inject_norm_weight + Δ_θ(z, x₀).
        # State preconditioner is parameter-free RMSUnit; every projection-local
        # scale is owned by its downstream linear (see Prenorm Scale Independence
        # Rule in CLAUDE.md). B̄ is passed explicitly through RevDEQFunction's
        # autograd boundary; a None b_bar means ones(D) (direct Block() in tests).
        self.x0_inject_norm_weight = nn.Parameter(torch.ones(dim))
        # Shared experts (DeepSeek-style): always-on with per-token sigmoid gate.
        # T_θ = g_s·E_shared(h) + Σ w_j E_routed_j(h)
        self.num_experts = num_experts
        self.num_shared_experts = int(num_shared_experts)
        num_routed = num_experts - self.num_shared_experts
        self.chained_stages_preset = chained_stages_preset
        self.chained_stack = None
        # Diagnostic tracking for per-DEQ-iteration gate trajectories.
        self._diag_track_enabled = False
        # Call-track lists hold 0-d GPU tensors (one per block.forward invocation
        # under DEQ solve). Materialization to Python floats happens in
        # `_run_solver_kernel`'s finally block, OUTSIDE the compiled forward —
        # mirrors the `_attn_expert_weights_per_iter` pattern at L3041.
        self._attn_gate_call_track: list[Tensor] = []
        self._router_gate_call_track: list[Tensor] = []
        # Pooled router (iter 35): attn track is the canonical source — the
        # legacy `_mlp_router_gate_call_track` was a value-equal duplicate.
        self._attn_router_gate_call_track: list[Tensor] = []
        # Per-expert routing weight per iteration (shows if experts specialize across
        # iters).  Stored as detached GPU tensors of shape (E,) — materialized to
        # Python lists once after the DEQ solve, NOT per-iter (CLAUDE.md §9
        # "Hot-path sync prohibition" forbids .item() inside the micro_step loop).
        self._attn_expert_weights_per_iter: list[Tensor] = []
        self._mlp_expert_weights_per_iter: list[Tensor] = []
        # Shared-gate scalars are stored as GPU 0-d tensors during the solve;
        # the .item() materialization happens at log time (outside the hot
        # range) via the `float(...)` calls in `format_expert_info`.
        self._shared_gate_mean: Tensor | float | None = None
        self._shared_gate_min: Tensor | float | None = None
        self._shared_gate_std: Tensor | float | None = None
        self._shared_gate_diag_step: int | None = None

        if chained_stages_preset is not None and str(chained_stages_preset).strip().lower() not in ("", "none"):
            from experiments.components.chained_routing import ChainedExpertStack, parse_stages

            stages = parse_stages(
                str(chained_stages_preset),
                num_experts=int(num_experts),
                num_shared_experts=int(self.num_shared_experts),
            )
            if stages is None:
                raise ValueError("chained_stages_preset parsed to None after non-empty input")
            self.chained_stack = ChainedExpertStack(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
                kv_latent_dim=kv_latent_dim,
                attn_expert_rank=attn_expert_rank,
                mlp_expert_rank=mlp_expert_rank,
                stages=stages,
                router_scoring=router_scoring,
                router_pertoken_entropy_coef=router_pertoken_entropy_coef,
                router_dirichlet_ucb_beta=router_dirichlet_ucb_beta,
                use_router_sigmoid_gate=use_router_sigmoid_gate,
                use_entmax_routing=use_entmax_routing,
                entmax_blend_init_logit=entmax_blend_init_logit,
                use_nsa_attention=use_nsa_attention,
                nsa_compress_block_size=nsa_compress_block_size,
                nsa_compress_block_sliding_stride=nsa_compress_block_sliding_stride,
                nsa_sliding_window_size=nsa_sliding_window_size,
                nsa_branch_gate_init=nsa_branch_gate_init,
                use_sparse_attn_head_gate=use_sparse_attn_head_gate,
                sparse_attn_gate_window=sparse_attn_gate_window,
                sparse_attn_gate_scale=sparse_attn_gate_scale,
                sparse_attn_gate_factor=sparse_attn_gate_factor,
                sparse_attn_gate_init_std=sparse_attn_gate_init_std,
                use_rr_attention=use_rr_attention,
                rr_stride=rr_stride,
                rr_block_size=rr_block_size,
                rr_tau=rr_tau,
                soft_dense_router_cls=SoftDenseRouter,
                causal_self_attention_cls=CausalSelfAttention,
                mlp_cls=MLP,
                rms_norm_cls=RMSNorm,
            )
            self._init_chained_stage0_aliases()
            return

        self.attn_post_mix_norm = RMSNorm(dim)
        self.mlp_post_mix_norm = RMSNorm(dim)
        # Iter 84 (2026-04-24): independent shared-expert sigmoid gates for
        # attention and MLP paths.  Previously a single `shared_gate` produced
        # one (B,T,S) gate applied to BOTH the attention shared expert and the
        # MLP shared expert — an accidental symmetry forcing the two paths to
        # open/close together.  Splitting the gates lets each path learn its
        # own modulation (trivial param cost: 2 × num_shared_experts × dim +
        # 2 × num_shared_experts bias + 2 × dim for the prenorm scales).
        if self.num_shared_experts > 0:
            self.shared_gate_attn = nn.Linear(dim, self.num_shared_experts, bias=True)
            self.shared_gate_mlp = nn.Linear(dim, self.num_shared_experts, bias=True)
            self.shared_gate_norm_weight_attn = nn.Parameter(torch.ones(dim))
            self.shared_gate_norm_weight_mlp = nn.Parameter(torch.ones(dim))
            for g in (self.shared_gate_attn, self.shared_gate_mlp):
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, 1.0)  # init near-open (matches pre-iter-84)
        # Router only covers routed experts (not shared). Routing-balance
        # regularization is the per-slice CV² loss inside SoftDenseRouter
        # (no relu hinge / no target — continuous balance pressure),
        # aggregated by GPT._collect_routing_losses.
        self.router = SoftDenseRouter(dim, 2 * num_routed,
                                      scoring=router_scoring,
                                      health_slices=(num_routed, num_routed),
                                      entropy_coef=router_pertoken_entropy_coef,
                                      dirichlet_ucb_beta=router_dirichlet_ucb_beta,
                                      use_router_sigmoid_gate=use_router_sigmoid_gate,
                                      use_entmax_routing=use_entmax_routing,
                                      entmax_blend_init_logit=entmax_blend_init_logit,
                                      use_reverse_kl_balance=use_reverse_kl_balance)
        self.attn_router = self.router  # alias for backward-compat diagnostics
        self.mlp_router = self.router   # alias (same instance → dedup via id())
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         kv_latent_dim=kv_latent_dim, num_experts=num_experts,
                                         expert_rank=attn_expert_rank, router=self.router,
                                         use_nsa_attention=use_nsa_attention,
                                         nsa_compress_block_size=nsa_compress_block_size,
                                         nsa_compress_block_sliding_stride=nsa_compress_block_sliding_stride,
                                         nsa_sliding_window_size=nsa_sliding_window_size,
                                         nsa_branch_gate_init=nsa_branch_gate_init,
                                         use_sparse_attn_head_gate=use_sparse_attn_head_gate,
                                         sparse_attn_gate_window=sparse_attn_gate_window,
                                         sparse_attn_gate_scale=sparse_attn_gate_scale,
                                         sparse_attn_gate_factor=sparse_attn_gate_factor,
                                         sparse_attn_gate_init_std=sparse_attn_gate_init_std,
                                         use_rr_attention=use_rr_attention,
                                         rr_stride=rr_stride,
                                         rr_block_size=rr_block_size,
                                         rr_tau=rr_tau)
        self.mlp = MLP(dim, mlp_mult, num_experts=num_experts,
                       expert_rank=mlp_expert_rank, router=self.router,
                       use_perdim_gate=use_expert_perdim_gate,
                       perdim_gate_rank=expert_perdim_gate_rank)

    def _init_chained_stage0_aliases(self) -> None:
        """Stage-0-only compatibility aliases for legacy diagnostics and tests
        that read `shared_block.attn` / `.mlp` / `.router`.

        Under chained mode these point to stage-0 ONLY. Every consumer that
        needs all stages MUST go through `active_routers()` /
        `active_attn_modules()` / `active_mlp_modules()` (or iterate
        `chained_stack.iter_*`). Forward, ortho-aux, gate tracking, and the
        optimizer-step bias_update already use those iterators; new diagnostics
        that touch `block.attn` / `.mlp` / `.router` directly MUST gate on
        `chained_stack is None`.
        """
        stack = self.chained_stack
        assert stack is not None
        self.attn = stack.attns[0] if len(stack.attns) > 0 else None
        self.mlp = stack.mlps[0] if len(stack.mlps) > 0 else None
        self.router = stack.routers[0] if len(stack.routers) > 0 else None
        self.attn_router = self.router
        self.mlp_router = self.router

    def active_routers(self) -> list[SoftDenseRouter]:
        if self.chained_stack is not None:
            return list(self.chained_stack.iter_routers())
        return [self.router]

    def active_attn_modules(self):
        if self.chained_stack is not None:
            return list(self.chained_stack.iter_attn_modules())
        return [("attn", self.attn)]

    def active_mlp_modules(self):
        if self.chained_stack is not None:
            return list(self.chained_stack.iter_mlp_modules())
        return [("mlp", self.mlp)]

    @staticmethod
    def _mlp_expert_outputs(mlp: MLP, h: Tensor) -> Tensor:
        bsz, seqlen, dim = h.shape
        n_tok = bsz * seqlen
        x_flat = h.reshape(n_tok, dim)
        e_count, rank = mlp.num_experts, mlp.expert_rank
        gate_w = (
            mlp.expert_gate.to(dtype=x_flat.dtype)
            * mlp.gate_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(e_count * rank, dim)
        fc_w = (
            mlp.expert_fc.to(dtype=x_flat.dtype)
            * mlp.fc_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(e_count * rank, dim)
        gate = x_flat @ gate_w.t()
        fc = x_flat @ fc_w.t()
        h_mlp = F.silu(gate) * fc
        h_mlp_per_expert = h_mlp.reshape(n_tok, e_count, rank)
        h_mlp_per_expert = (
            F.rms_norm(h_mlp_per_expert, (rank,), eps=1e-6)
            * mlp.hidden_norm_weight.to(dtype=h_mlp_per_expert.dtype)
        )
        down_t = mlp.expert_down.to(dtype=h_mlp_per_expert.dtype).transpose(1, 2)
        return torch.einsum("ner,erd->ned", h_mlp_per_expert.float(), down_t.float())

    def _ortho_aux_chained(
        self,
        z_sub: Tensor,
        x0_sub: Tensor,
        *,
        compute_per_token_gram: bool,
        per_token_gram_kind: str,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        assert self.chained_stack is not None
        stack = self.chained_stack
        cumulative_state = z_sub
        attn_outputs: list[Tensor] = []
        mlp_outputs: list[Tensor] = []
        kv_flats: list[Tensor] = []

        for stage_idx, spec in enumerate(stack.stages):
            h = _rms_unit(cumulative_state + x0_sub)
            w_attn, w_mlp = stack._route(stage_idx, h)
            stage_delta = torch.zeros_like(z_sub)

            attn_idx = stack._stage_attn_indices[stage_idx]
            if attn_idx is not None:
                attn = stack.attns[attn_idx]
                attn_expert_out = attn.forward_experts(h)
                attn_outputs.append(attn_expert_out)
                kv_flats.append(attn.expert_kv_a.float().reshape(attn.expert_kv_a.shape[0], -1))
                g_s_attn = stack._shared_gate(
                    h,
                    gate_list=stack.shared_gate_attn,
                    norm_list=stack.shared_gate_norm_weight_attn,
                    gate_idx=stack._stage_shared_attn_gate_indices[stage_idx],
                    norm_idx=stack._stage_shared_attn_norm_indices[stage_idx],
                )
                parts = []
                if spec.shared_attn > 0 and g_s_attn is not None:
                    parts.append((attn_expert_out[:, :, :spec.shared_attn, :] * g_s_attn.unsqueeze(-1)).sum(dim=2))
                if spec.attn_experts > 0:
                    assert w_attn is not None
                    parts.append((attn_expert_out[:, :, spec.shared_attn:, :] * w_attn.unsqueeze(-1)).sum(dim=2))
                attn_mix = sum(parts) if len(parts) > 1 else parts[0]
                norm_idx = stack._stage_attn_norm_indices[stage_idx]
                assert norm_idx is not None
                stage_delta = stage_delta + stack.attn_post_norms[norm_idx](attn_mix)

            mlp_idx = stack._stage_mlp_indices[stage_idx]
            if mlp_idx is not None:
                mlp = stack.mlps[mlp_idx]
                mlp_outputs.append(self._mlp_expert_outputs(mlp, h))
                g_s_mlp = stack._shared_gate(
                    h,
                    gate_list=stack.shared_gate_mlp,
                    norm_list=stack.shared_gate_norm_weight_mlp,
                    gate_idx=stack._stage_shared_mlp_gate_indices[stage_idx],
                    norm_idx=stack._stage_shared_mlp_norm_indices[stage_idx],
                )
                if w_mlp is None:
                    w_mlp = h.new_empty((*h.shape[:-1], 0))
                mlp_mix = mlp.mix_experts(h, w_mlp, num_shared=spec.shared_mlp, shared_gate=g_s_mlp)
                norm_idx = stack._stage_mlp_norm_indices[stage_idx]
                assert norm_idx is not None
                stage_delta = stage_delta + stack.mlp_post_norms[norm_idx](mlp_mix)

            cumulative_state = cumulative_state + stage_delta

        if not attn_outputs or not mlp_outputs:
            zero = z_sub.new_zeros(())
            return zero, zero, None, None

        y_attn = torch.cat(attn_outputs, dim=2)
        y_mlp = torch.cat(mlp_outputs, dim=1)
        e_attn = y_attn.shape[2]
        e_mlp = y_mlp.shape[1]
        dim = z_sub.shape[-1]

        mu_attn = y_attn.detach().float().mean(dim=(0, 1))
        attn_ortho = mean_abs_offdiag_cosine(mu_attn)
        if kv_flats:
            attn_ortho = 0.5 * attn_ortho + 0.5 * mean_abs_offdiag_cosine(torch.cat(kv_flats, dim=0))
        mlp_ortho = mean_abs_offdiag_cosine(y_mlp.detach().float().mean(dim=0))

        attn_gram_pt: Tensor | None = None
        mlp_gram_pt: Tensor | None = None
        if compute_per_token_gram:
            if per_token_gram_kind == "cosine":
                y_attn_n = F.normalize(y_attn.float(), dim=-1, eps=1e-6)
                g_attn = torch.einsum("bted,btfd->btef", y_attn_n, y_attn_n)
                off_attn = g_attn - torch.eye(e_attn, device=g_attn.device, dtype=g_attn.dtype)
                # Strict health rescue: train against the worst off-diagonal
                # pair per sampled token, not the mean over all pairs.
                attn_gram_pt = off_attn.pow(2).amax(dim=(-2, -1)).mean()

                y_mlp_n = F.normalize(y_mlp.float(), dim=-1, eps=1e-6)
                g_mlp = torch.einsum("ned,nfd->nef", y_mlp_n, y_mlp_n)
                off_mlp = g_mlp - torch.eye(e_mlp, device=g_mlp.device, dtype=g_mlp.dtype)
                mlp_gram_pt = off_mlp.pow(2).amax(dim=(-2, -1)).mean()
            else:
                target_attn = torch.eye(e_attn, device=y_attn.device, dtype=torch.float32) / float(e_attn)
                g_attn = torch.einsum("bted,btfd->btef", y_attn.float(), y_attn.float()) / float(dim)
                attn_gram_pt = (g_attn - target_attn).pow(2).sum(dim=(-2, -1)).mean() / float(e_attn * e_attn)

                target_mlp = torch.eye(e_mlp, device=y_mlp.device, dtype=torch.float32) / float(e_mlp)
                g_mlp = torch.einsum("ned,nfd->nef", y_mlp.float(), y_mlp.float()) / float(dim)
                mlp_gram_pt = (g_mlp - target_mlp).pow(2).sum(dim=(-2, -1)).mean() / float(e_mlp * e_mlp)

        return attn_ortho, mlp_ortho, attn_gram_pt, mlp_gram_pt

    def ortho_aux(self, z_in: Tensor, x0: Tensor, *, max_tokens: int = 256,
                   token_start: int | None = None,
                   compute_per_token_gram: bool = False,
                   per_token_gram_kind: str = "frobenius"
                   ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        bsz, seqlen, dim = z_in.shape
        t = int(min(max(1, int(max_tokens)), seqlen))
        if t >= int(seqlen):
            start = 0
        else:
            max_start = int(seqlen) - t
            start = int(token_start or 0) % (max_start + 1)
        z_sub = z_in[:, start:start + t]
        x0_sub = x0[:, start:start + t]
        if self.chained_stack is not None:
            return self._ortho_aux_chained(
                z_sub, x0_sub,
                compute_per_token_gram=compute_per_token_gram,
                per_token_gram_kind=per_token_gram_kind,
            )
        x = z_sub + x0_sub
        h = _rms_unit(x)

        # Attention ortho: per-expert outputs from independent expert SDPA.
        attn_expert_out = self.attn.forward_experts(h)  # (B, t, E, D)
        mu_attn = attn_expert_out.detach().float().mean(dim=(0, 1))  # (E, D)
        attn_ortho = mean_abs_offdiag_cosine(mu_attn)

        # MLP ortho (unchanged: compute mean expert outputs via existing path).
        E = self.num_experts
        N = bsz * t
        x_flat = h.reshape(N, dim)
        E2, R2 = self.mlp.num_experts, self.mlp.expert_rank
        G = (
            self.mlp.expert_gate.to(dtype=x_flat.dtype)
            * self.mlp.gate_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(E2 * R2, dim)
        Fm = (
            self.mlp.expert_fc.to(dtype=x_flat.dtype)
            * self.mlp.fc_in_norm_weight.to(dtype=x_flat.dtype).unsqueeze(1)
        ).reshape(E2 * R2, dim)
        gate = x_flat @ G.t()
        fc = x_flat @ Fm.t()
        h_mlp = F.silu(gate) * fc
        h_mlp_per_expert = h_mlp.reshape(N, E2, R2)  # (N, E, R) — per-token, per-expert hidden
        h_mlp_per_expert = (
            F.rms_norm(h_mlp_per_expert, (R2,), eps=1e-6)
            * self.mlp.hidden_norm_weight.to(dtype=h_mlp_per_expert.dtype)
        )
        mu_h2 = h_mlp_per_expert.mean(dim=0).to(dtype=torch.float32)
        down_T = self.mlp.expert_down.to(dtype=mu_h2.dtype).transpose(1, 2)  # (E, R, D)
        mu_mlp = torch.einsum("er,erd->ed", mu_h2, down_T)
        mlp_ortho = mean_abs_offdiag_cosine(mu_mlp)

        # Phase 9 iter 52: KV latent subspace orthogonalization.
        # Penalize off-diagonal cosine similarity of KV down-projection weights.
        # Forces expert KV compressions to span distinct subspaces.
        # Weight-space penalty (structural) vs output-space penalty (input-dependent).
        kv_a = self.attn.expert_kv_a.float()  # (E, kv_rank, dim)
        kv_flat = kv_a.reshape(kv_a.shape[0], -1)  # (E, kv_rank*dim)
        kv_subspace_ortho = mean_abs_offdiag_cosine(kv_flat)
        # Blend: 50% output-level ortho + 50% weight-level KV subspace ortho
        attn_ortho = 0.5 * attn_ortho + 0.5 * kv_subspace_ortho

        # iter 141 (NEW 2026-05-04): per-token expert-OUTPUT Gram penalty.
        # G_t = (Y_t Y_t^T) / D where Y_t ∈ ℝ^{E×D} stacks expert outputs at
        # token t. Penalty = E_t[‖G_t − I/E‖²_F]. Pulls per-token expert
        # computations toward orthogonality with bounded norm. Subsumes
        # block_ortho via Jensen (E_t[G_t] mean → block_ortho is the lower-
        # bound objective). Compute only when requested (gated on coef>0
        # at the annealer, then propagated by GPT.forward).
        attn_gram_pt: Tensor | None = None
        mlp_gram_pt: Tensor | None = None
        if compute_per_token_gram:
            # iter 142 refactor: two diversity formulations selectable via
            # `per_token_gram_kind`.
            #   frobenius (iter 141): G = (Y Y^T)/D; loss = ‖G − I/E‖²_F.
            #     Penalizes both orthogonality (off-diag → 0) AND norm balance
            #     (diag → 1/E ⇔ ‖y_e‖² → D/E). Subsumes block_ortho via Jensen.
            #   cosine: Y' = normalize(Y); G = Y' Y'^T; loss =
            #     mean_token max_pair(off_diag²). Scale-invariant and aligned
            #     with the worst-pair post-int health failure.
            E_attn = attn_expert_out.shape[2]
            Y_attn = attn_expert_out.float()  # (B, t, E, D)
            Y_mlp = torch.einsum("ner,erd->ned",
                                  h_mlp_per_expert.float(),
                                  down_T.float())  # (N, E, D)
            if per_token_gram_kind == "cosine":
                Y_attn_n = F.normalize(Y_attn, dim=-1, eps=1e-6)
                G_attn = torch.einsum("bted,btfd->btef", Y_attn_n, Y_attn_n)
                off_attn = G_attn - torch.eye(E_attn, device=G_attn.device, dtype=G_attn.dtype)
                attn_gram_pt = off_attn.pow(2).amax(dim=(-2, -1)).mean()

                Y_mlp_n = F.normalize(Y_mlp, dim=-1, eps=1e-6)
                G_mlp = torch.einsum("ned,nfd->nef", Y_mlp_n, Y_mlp_n)
                off_mlp = G_mlp - torch.eye(E2, device=G_mlp.device, dtype=G_mlp.dtype)
                mlp_gram_pt = off_mlp.pow(2).amax(dim=(-2, -1)).mean()
            else:  # frobenius
                # Per-entry mean (divide by E²) so the loss magnitude is
                # comparable to cosine's per-pair-mean rather than ~30× larger.
                # Without this, "same coef" between the two kinds means ~30×
                # more frobenius gradient pressure — apples vs oranges.
                target_attn = torch.eye(E_attn, device=attn_expert_out.device,
                                         dtype=torch.float32) / float(E_attn)
                G_attn = torch.einsum("bted,btfd->btef", Y_attn, Y_attn) / float(dim)
                attn_gram_pt = (G_attn - target_attn).pow(2).sum(dim=(-2, -1)).mean() / float(E_attn * E_attn)

                target_mlp = torch.eye(E2, device=h_mlp.device, dtype=torch.float32) / float(E2)
                G_mlp = torch.einsum("ned,nfd->nef", Y_mlp, Y_mlp) / float(dim)
                mlp_gram_pt = (G_mlp - target_mlp).pow(2).sum(dim=(-2, -1)).mean() / float(E2 * E2)

        return attn_ortho, mlp_ortho, attn_gram_pt, mlp_gram_pt

    def _route_pooled(self, u_proj: Tensor) -> tuple[Tensor, Tensor]:
        """Compute pooled routing weights for ROUTED experts only.

        Returns (w_attn, w_mlp) each of shape (..., num_routed).
        Shared experts (indices 0:num_shared) bypass routing entirely.
        """
        num_routed = self.num_experts - self.num_shared_experts
        w_all = self.router(u_proj, pre_normed=True)  # (..., 2*num_routed)
        return w_all[..., :num_routed].contiguous(), w_all[..., num_routed:].contiguous()

    @dynamo_disable
    def _capture_shared_gate_diag(self, g_s_attn: Tensor, g_s_mlp: Tensor) -> None:
        """Eager-only capture of shared-gate stats. Called from forward().

        iter 117 v5 (2026-04-29): hoisted from inline `if _should_diag(...):`
        to a `@dynamo_disable` helper to break the recompile-pressure cycle
        observed in v4 (16-recompile budget hit at ~s30, eager fallback for
        one resume frame, slow throughput). dynamo treats this as a single
        opaque call — no guards on _should_diag / _ROUTER_DIAGNOSTICS_ACTIVE
        / grad_mode toggles inside.

        Inside the helper, `.detach()` already prevents gradient flow, so the
        previous `with torch.no_grad():` was redundant — removed for clarity.
        """
        if _should_diag(self.training):
            g_sf = torch.cat([g_s_attn.detach().float(), g_s_mlp.detach().float()], dim=-1)
            self._shared_gate_mean = g_sf.mean().detach()
            self._shared_gate_min = g_sf.min().detach()
            self._shared_gate_std = g_sf.std(unbiased=False).detach()
            self._shared_gate_diag_step = _ROUTER_DIAGNOSTICS_STEP
        else:
            self._shared_gate_mean = None
            self._shared_gate_min = None
            self._shared_gate_std = None
            self._shared_gate_diag_step = None

    def forward(self, z_in: Tensor, x0: Tensor, b_bar: Tensor | None = None) -> Tensor:
        if self.chained_stack is not None:
            self.chained_stack._diag_track_enabled = bool(self._diag_track_enabled)
            delta = self.chained_stack(z_in, x0, _rms_unit)
            self._shared_gate_mean = self.chained_stack._shared_gate_mean
            self._shared_gate_min = self.chained_stack._shared_gate_min
            self._shared_gate_std = self.chained_stack._shared_gate_std
            self._shared_gate_diag_step = self.chained_stack._shared_gate_diag_step
            x0_rms = F.rms_norm(x0, (x0.size(-1),), eps=1e-6) * self.x0_inject_norm_weight.to(x0.dtype)
            if b_bar is not None:
                x0_rms = b_bar.to(x0_rms.dtype) * x0_rms
            return x0_rms + delta

        # h = RMSUnit(z + x_0); Δ = Σ g_s·E_shared(h) + Σ w_j·E_routed_j(h).
        # Output injection (x_0 term) is applied below via B̄ ⊙ RMSUnit(x_0).
        u = z_in + x0
        h = _rms_unit(u)

        E = self.num_experts
        S = self.num_shared_experts
        # Route only the non-shared experts.
        w_attn, w_mlp = self._route_pooled(h)
        # Eager-only per-iter expert-weight tracking — see helper docstring.
        self._maybe_track_expert_weights(w_attn, w_mlp)

        # All experts compute outputs together (shared + routed).
        attn_expert_out = self.attn.forward_experts(h)  # (B, T, E, D)
        if S > 0:
            # Iter 84: independent per-path sigmoid gates for attn vs mlp.
            h_shared_gate_attn = h * self.shared_gate_norm_weight_attn.to(dtype=h.dtype)
            h_shared_gate_mlp = h * self.shared_gate_norm_weight_mlp.to(dtype=h.dtype)
            g_s_attn = torch.sigmoid(self.shared_gate_attn(h_shared_gate_attn))  # (B, T, S)
            g_s_mlp = torch.sigmoid(self.shared_gate_mlp(h_shared_gate_mlp))      # (B, T, S)
            # iter 117 v5 (2026-04-29): hoist this diagnostic block into a
            # `@dynamo_disable` helper. Original inline pattern hit dynamo's
            # recompile_limit (16) at ~s30: `_should_diag` is a Python-bool
            # guard that toggles every ~10 steps for log emission, AND the
            # `with torch.no_grad():` context flips GLOBAL_STATE grad_mode.
            # Every log boundary burned 2 recompiles → 8 boundaries hit the
            # 16-budget → eager fallback for one resume frame → 30-50%
            # slower steady-state. Wrapping in `@dynamo_disable` makes it
            # opaque to dynamo — one fixed graph break per forward (no
            # guards on internal state). Mirrors the iter 28 `_capture_attn_out_ortho`
            # pattern at L1860.
            self._capture_shared_gate_diag(g_s_attn, g_s_mlp)
            attn_shared = (attn_expert_out[:, :, :S, :] * g_s_attn.unsqueeze(-1)).sum(dim=2)
            # Routed: weighted by router
            attn_routed = (attn_expert_out[:, :, S:, :] * w_attn.unsqueeze(-1)).sum(dim=2)
            attn_mix = attn_shared + attn_routed
        else:
            g_s_mlp = None
            self._shared_gate_mean = None
            self._shared_gate_min = None
            self._shared_gate_std = None
            self._shared_gate_diag_step = None
            attn_mix = (attn_expert_out * w_attn.unsqueeze(-1)).sum(dim=2)
        attn_mix = self.attn_post_mix_norm(attn_mix)

        # MLP experts (same split: shared gated + routed).
        # h is already parameter-free RMS-normalized above, which mix_experts assumes.
        mlp_mix = self.mlp.mix_experts(h, w_mlp,
                                        num_shared=S, shared_gate=g_s_mlp if S > 0 else None)
        mlp_mix = self.mlp_post_mix_norm(mlp_mix)

        # Dense mixture Δ = attn_mix + mlp_mix.
        # Fix #3 (profile-driven, 2026-04-28): drop `.to(dtype=z_in.dtype)`
        # on `(attn_mix + mlp_mix)` — both come from `*_post_mix_norm`
        # (RMSNorm, dtype-preserving) whose inputs are bf16 (z_in is bf16, h is
        # bf16-derived), so the cast is provably a no-op. PyTorch handles
        # same-dtype `.to()` as a return-self, but emitting it as a graph node
        # adds a fixed boundary that may inhibit Inductor fusion with the
        # downstream `+ delta`. Removing also drops one fused-Triton kernel
        # invocation (`triton_poi_fused__to_copy__unsafe_view_add_clone_mul_*`
        # was 3.96% × 2 in profile_v13, partly from this cast pair).
        delta = attn_mix + mlp_mix

        # Parcae input injection: T_θ = B̄ ⊙ RMSUnit(x_0) ⊙ g + Δ_θ.
        # b_bar=None ⇒ ones(D) (direct Block() in tests).  At the fixed point
        # β=1-Ā cancels and y* = B̄ ⊙ RMSUnit(x_0) ⊙ g + Δ*.
        # Fix #1 (profile-driven, see _rms_scale comment): fused F.rms_norm.
        x0_rms = F.rms_norm(x0, (x0.size(-1),), eps=1e-6) * self.x0_inject_norm_weight.to(x0.dtype)
        if b_bar is not None:
            x0_rms = b_bar.to(x0_rms.dtype) * x0_rms
        # Fix #3 (profile-driven): same dtype-preservation argument — x0 is
        # bf16 (input from _run_solver_kernel), F.rms_norm preserves it, the
        # x0_inject_norm_weight cast preserves it, b_bar is fp32 cast to
        # x0_rms.dtype (= bf16). So x0_rms.dtype == z_in.dtype already.
        raw_out = x0_rms + delta

        # Eager-only gate-call tracking — see helper docstring.
        self._maybe_track_gate_calls()
        return raw_out

    @dynamo_disable
    def _maybe_track_expert_weights(self, w_attn: Tensor, w_mlp: Tensor) -> None:
        """Eager-only per-iter expert-weight tracking.

        Profile_v3 (post Fix #3, commit d7996da) showed the residual recompile
        vector at 23.45s/step was `len(self._mlp_expert_weights_per_iter) == N`
        guards inside the compiled `Block.forward`. With K-jitter (8,12,20)
        and incremental list growth per iter (lengths 0,1,2,...,2K), dynamo
        sees 2*8 + 2*12 + 2*20 = 80 unique list-length values across the
        K-loop — even with `recompile_limit=16`, the cache thrashes.

        Wrapping the append in this `@dynamo_disable` helper makes the call
        opaque to dynamo (one fixed graph break per forward, no list-length
        guards). Same pattern as `_capture_attn_out_ortho` from Fix #3, applied
        to the second graph-break vector identified in profile_v3.
        """
        if not self._diag_track_enabled:
            return
        with torch.no_grad():
            reduce_dims = tuple(range(w_attn.dim() - 1))
            self._attn_expert_weights_per_iter.append(
                w_attn.detach().float().mean(dim=reduce_dims))
            self._mlp_expert_weights_per_iter.append(
                w_mlp.detach().float().mean(dim=reduce_dims))

    @dynamo_disable
    def _maybe_track_gate_calls(self) -> None:
        """Eager-only per-call gate-stat tracking (attn gate, router gate).
        Same dynamo-disable rationale as `_maybe_track_expert_weights`: removes
        list-length guards on `_attn_gate_call_track`, `_router_gate_call_track`,
        and `_*_router_gate_call_track` from the compiled Block.forward.
        """
        if not self._diag_track_enabled:
            return
        ag = getattr(self.attn, "_attn_gate_last_mean", None)
        if ag is not None:
            self._attn_gate_call_track.append(ag)
        attn_rg = getattr(self.router, "_router_gate_last_mean", None)
        if attn_rg is not None:
            self._attn_router_gate_call_track.append(attn_rg)
        # Iter 35: attn and mlp share the same router instance, so the mlp
        # router-gate value equals the attn router-gate value. The legacy
        # `_mlp_router_gate_call_track` was dropped (value-equal duplicate
        # of `_attn_router_gate_call_track`); downstream readers consume the
        # attn track and treat it as the pooled-router value.
        # Combined router-gate pair-mean (kept for log-format compat — attn
        # value alone equals the pair mean since the two are identical).
        rg_vals: list[Tensor] = []
        if attn_rg is not None:
            rg_vals.append(attn_rg)
        if rg_vals:
            avg_rg = rg_vals[0] if len(rg_vals) == 1 else 0.5 * (rg_vals[0] + rg_vals[1])
            self._router_gate_call_track.append(avg_rg)


# ---------------------------------------------------------------------------
# RevDEQ FUNCTION
# ---------------------------------------------------------------------------

class RevDEQFunction(torch.autograd.Function):
    """O(1) memory backward for RevDEQ coupled-state solver."""
    @staticmethod
    def _autocast_off_ctx(device_type: str):
        try:
            return torch.autocast(device_type=device_type, enabled=False)
        except Exception:
            return contextlib.nullcontext()

    @staticmethod
    def _autocast_like_ctx(device_type: str, dtype: torch.dtype):
        if device_type == "cpu" or dtype not in (torch.float16, torch.bfloat16):
            return RevDEQFunction._autocast_off_ctx(device_type)
        try:
            return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
        except Exception:
            return contextlib.nullcontext()

    @staticmethod
    def forward(ctx, f_theta, x0, z_init, beta, b_bar, K, bptt_k, *params):
        acc_dtype = torch.float64
        state_dtype = torch.float32
        compute_dtype = z_init.dtype
        device_type = x0.device.type
        # Support both scalar and tensor (per-dim Parcae) beta.
        # Track whether upstream graph needs grad_beta (for Parcae learning).
        ctx.beta_requires_grad = isinstance(beta, torch.Tensor) and beta.requires_grad
        ctx.beta_input_dtype = beta.dtype if isinstance(beta, torch.Tensor) else None
        ctx.b_bar_requires_grad = isinstance(b_bar, torch.Tensor) and b_bar.requires_grad
        ctx.b_bar_input_dtype = b_bar.dtype if isinstance(b_bar, torch.Tensor) else None
        ctx.has_b_bar = isinstance(b_bar, torch.Tensor)
        # Store the detached B̄ on ctx directly rather than embedding an empty
        # sentinel tensor in save_for_backward — the sentinel pattern is brittle
        # for anything that iterates ctx.saved_tensors expecting real shapes.
        ctx.b_bar_saved = b_bar.detach() if isinstance(b_bar, torch.Tensor) else None
        if isinstance(beta, torch.Tensor):
            beta = beta.detach().to(torch.float64)
            beta_inv = 1.0 - beta
        else:
            beta_inv = 1.0 - beta
        ctx.do_recon_diag = bool(_ROUTER_DIAGNOSTICS_ACTIVE)

        y_state = z_init.to(state_dtype)
        z_state = z_init.to(state_dtype)
        z_prev_state = z_state
        z_init_state = z_state.detach()

        # TBPTT recon snapshot — captures (y, z) at the iter index where
        # backward will land after K_bwd reverse steps, so the diagnostic can
        # measure true reversibility within the window. CLAUDE.md §6.1.
        K_bwd = int(bptt_k) if bptt_k else 0
        if K_bwd <= 0 or K_bwd >= K:
            K_bwd = K
        snap_iter = K - K_bwd
        y_snap_state: Tensor | None = None
        z_snap_state: Tensor | None = None

        with torch.no_grad():
            if bool(ctx.do_recon_diag) and snap_iter == 0:
                # Full BPTT: snapshot is the initial state — recover today's
                # legacy comparison without an extra clone.
                y_snap_state = z_init_state
                z_snap_state = z_init_state
            out_y = None
            for i in range(K):
                z_prev_state = z_state
                y_acc = y_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_z = f_theta(z_state.to(compute_dtype), x0, b_bar)
                y_acc = y_acc + out_z.to(acc_dtype) * beta
                y_state = y_acc.to(state_dtype)

                z_acc = z_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_state.to(compute_dtype), x0, b_bar)
                z_acc = z_acc + out_y.to(acc_dtype) * beta
                z_state = z_acc.to(state_dtype)

                if bool(ctx.do_recon_diag) and (i + 1) == snap_iter:
                    y_snap_state = y_state.detach().clone()
                    z_snap_state = z_state.detach().clone()

            if out_y is not None:
                if bool(ctx.do_recon_diag):
                    try:
                        # Write to unwrapped module — compiled wrapper may not
                        # propagate attribute writes to the original module.
                        _target = _unwrap_compiled_module(f_theta)
                        setattr(
                            _target,
                            "_deq_residual_proxy_t",
                            (z_state - out_y.to(state_dtype)).norm().detach(),
                        )
                        # FP travel = ||z_K - z_init|| / ||z_init||. Distance
                        # from initial embedding to converged FP — proxy for
                        # the expressive transformation the DEQ block applies.
                        # Larger = more state movement (within reason; growing
                        # alongside diverging iter_conv_rel signals runaway).
                        z0 = z_init_state.to(state_dtype)
                        denom0 = z0.norm().clamp(min=1.0)
                        setattr(
                            _target,
                            "_deq_fp_travel_last_fwd",
                            ((z_state - z0).norm() / denom0).detach(),
                        )
                    except Exception:
                        pass

        ctx.save_for_backward(x0.detach(), y_state.detach(), z_state.detach(),
                              z_prev_state.detach())
        ctx.z_init_state = z_init_state
        ctx.y_snap_state = y_snap_state
        ctx.z_snap_state = z_snap_state
        ctx.f_theta = f_theta
        # Store as fp64 tensors for backward precision.
        ctx.beta = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta, dtype=torch.float64)
        ctx.beta_inv = beta_inv if isinstance(beta_inv, torch.Tensor) else torch.tensor(beta_inv, dtype=torch.float64)
        ctx.K = K
        ctx.bptt_k = int(bptt_k) if bptt_k else 0
        ctx.compute_dtype = compute_dtype
        ctx.device_type = device_type
        ctx.params = params
        return z_state.to(compute_dtype), z_prev_state.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_z, _grad_z_prev_ignored):
        x0, y_terminal, z_terminal, _z_prev = (t.detach() for t in ctx.saved_tensors)
        b_bar_saved = ctx.b_bar_saved
        z_init_state = getattr(ctx, "z_init_state", None)
        f_theta = ctx.f_theta
        beta, beta_inv = ctx.beta, ctx.beta_inv
        K = ctx.K
        compute_dtype = ctx.compute_dtype
        device_type = ctx.device_type
        acc_dtype = torch.float64
        state_dtype = torch.float32

        params_all = tuple(ctx.params)
        req_indices = [i for i, p in enumerate(params_all) if getattr(p, "requires_grad", False)]
        params_req = tuple(params_all[i] for i in req_indices)

        bar_z = grad_z.to(state_dtype)
        bar_y = torch.zeros_like(bar_z)
        cur_param_grads_req: list[torch.Tensor | None] = [None] * len(params_req)
        cur_x_grad = torch.zeros_like(x0, dtype=torch.float32)
        b_bar_requires_grad = bool(getattr(ctx, "b_bar_requires_grad", False))
        b_bar_local_base: Tensor | None = None
        grad_b_bar: torch.Tensor | None = None
        if bool(getattr(ctx, "has_b_bar", False)):
            b_dtype = getattr(ctx, "b_bar_input_dtype", None) or compute_dtype
            b_bar_local_base = b_bar_saved.detach().to(dtype=b_dtype)
            if b_bar_requires_grad:
                grad_b_bar = torch.zeros_like(b_bar_local_base, dtype=torch.float32)

        # Phase 9 iter 66a: accumulate dL/dβ for Parcae gradient flow.
        beta_requires_grad = ctx.beta_requires_grad
        grad_beta: torch.Tensor | None = None
        if beta_requires_grad:
            grad_beta = torch.zeros_like(beta)  # (D,) in fp64

        y_next64 = y_terminal.to(acc_dtype)
        z_next64 = z_terminal.to(acc_dtype)

        bptt_k = int(getattr(ctx, "bptt_k", 0) or 0)
        K_bwd = K if (bptt_k <= 0 or bptt_k >= K) else bptt_k
        truncated = K_bwd < K
        diag_vjp_per_iter: list[tuple[float, float]] = []
        do_vjp_diag = bool(getattr(ctx, "do_recon_diag", False))

        for _ in range(K_bwd):
            y_local = y_next64.detach().to(compute_dtype).requires_grad_()
            x_local = x0.detach().to(x0.dtype).requires_grad_()
            b_bar_y = None
            if b_bar_local_base is not None:
                # .clone() (not .detach()) so the y-leg and z-leg cannot alias
                # storage within the same iter.  See Custom Autograd Input Rule.
                b_bar_y = b_bar_local_base.clone().requires_grad_(b_bar_requires_grad)
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_local, x_local, b_bar_y)
            z_n64 = (z_next64 - out_y.detach().to(acc_dtype) * beta) / beta_inv

            # Parcae β gradient from z update: ∂z_{k+1}/∂β = f(y_{k+1}) - z_k
            if grad_beta is not None:
                innovation_z = out_y.detach().to(acc_dtype) - z_n64
                grad_beta += (bar_z.to(acc_dtype) * innovation_z).sum(dim=(0, 1))

            grad_seed_y = (beta * bar_z).to(out_y.dtype)
            if b_bar_requires_grad:
                grads_y = torch.autograd.grad(out_y, (y_local, x_local, b_bar_y, *params_req),
                                              grad_outputs=grad_seed_y, allow_unused=True)
                gy_b = grads_y[2]
                y_param_offset = 3
            else:
                grads_y = torch.autograd.grad(out_y, (y_local, x_local, *params_req),
                                              grad_outputs=grad_seed_y, allow_unused=True)
                gy_b = None
                y_param_offset = 2
            vjp_y = grads_y[0].to(state_dtype)
            bar_y_acc = bar_y + vjp_y

            z_local = z_n64.detach().to(compute_dtype).requires_grad_()
            x_local2 = x0.detach().to(x0.dtype).requires_grad_()
            b_bar_z = None
            if b_bar_local_base is not None:
                # .clone() so the z-leg has an independent leaf; see above.
                b_bar_z = b_bar_local_base.clone().requires_grad_(b_bar_requires_grad)
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_z = f_theta(z_local, x_local2, b_bar_z)
            y_n64 = (y_next64 - out_z.detach().to(acc_dtype) * beta) / beta_inv

            # Parcae β gradient from y update: ∂y_{k+1}/∂β = f(z_k) - y_k
            if grad_beta is not None:
                innovation_y = out_z.detach().to(acc_dtype) - y_n64
                grad_beta += (bar_y_acc.to(acc_dtype) * innovation_y).sum(dim=(0, 1))

            grad_seed_z = (beta * bar_y_acc).to(out_z.dtype)
            if b_bar_requires_grad:
                grads_z = torch.autograd.grad(out_z, (z_local, x_local2, b_bar_z, *params_req),
                                              grad_outputs=grad_seed_z, allow_unused=True)
                gz_b = grads_z[2]
                z_param_offset = 3
            else:
                grads_z = torch.autograd.grad(out_z, (z_local, x_local2, *params_req),
                                              grad_outputs=grad_seed_z, allow_unused=True)
                gz_b = None
                z_param_offset = 2
            vjp_z = grads_z[0].to(state_dtype)

            if do_vjp_diag:
                # Phase 6a.5 (review 9): store as GPU scalar tensors, not
                # Python floats.  Avoids K synchronous GPU→CPU roundtrips
                # per backward; the .item() sync happens once at log time.
                diag_vjp_per_iter.append((
                    vjp_y.detach().norm(),
                    vjp_z.detach().norm(),
                ))

            bar_z = beta_inv * bar_z + vjp_z
            bar_y = beta_inv * bar_y_acc

            if grad_b_bar is not None and (gy_b is not None or gz_b is not None):
                g_b = (gy_b if gy_b is not None else 0.0) + (gz_b if gz_b is not None else 0.0)
                grad_b_bar = grad_b_bar + g_b.detach().float()

            for j in range(len(params_req)):
                gy = grads_y[y_param_offset + j]
                gz = grads_z[z_param_offset + j]
                if gy is None and gz is None:
                    continue
                g = (gy if gy is not None else 0.0) + (gz if gz is not None else 0.0)
                g = g.detach()
                cur_param_grads_req[j] = g if cur_param_grads_req[j] is None else cur_param_grads_req[j] + g
            if grads_y[1] is not None:
                cur_x_grad += grads_y[1].detach().float()
            if grads_z[1] is not None:
                cur_x_grad += grads_z[1].detach().float()

            y_next64, z_next64 = y_n64, z_n64

        y_snap_state = getattr(ctx, "y_snap_state", None)
        z_snap_state = getattr(ctx, "z_snap_state", None)
        if bool(getattr(ctx, "do_recon_diag", False)) and isinstance(z_init_state, torch.Tensor):
            try:
                # `_deq_recon_error_last_bwd` (tbptt_recon): always emitted —
                # ||z_rec_{K-K_bwd} − z_snap|| / ||z_snap||, comparing the
                # K_bwd grad-reverse landing against the forward snapshot at
                # iter (K − K_bwd). This is the in-window reversibility check
                # for the gradient path that training actually consumes.
                #
                # `_deq_x0_recon_error_last_bwd` (deq_recon_err): only emitted
                # under full BPTT (K_bwd == K) — the grad-reverse already lands
                # at z_0 then, so it's free. Under TBPTT we skip x0 recon
                # entirely; rebuilding it via no_grad would cost K − K_bwd
                # extra f_theta forwards per backward solely for a diagnostic
                # the gradient path doesn't use. CLAUDE.md §6.1.
                _target = _unwrap_compiled_module(f_theta)

                if isinstance(y_snap_state, torch.Tensor) and isinstance(z_snap_state, torch.Tensor):
                    y_snap = y_snap_state.to(dtype=state_dtype)
                    z_snap = z_snap_state.to(dtype=state_dtype)
                    denom_s = z_snap.norm().clamp(min=1.0)
                    z_rec_in = z_next64.to(dtype=state_dtype)
                    y_rec_in = y_next64.to(dtype=state_dtype)
                    recon_err_t = ((z_rec_in - z_snap).norm() + (y_rec_in - y_snap).norm()) / denom_s
                    setattr(_target, "_deq_recon_error_last_bwd", recon_err_t.detach())

                if not truncated:
                    z0 = z_init_state.to(dtype=state_dtype)
                    denom_0 = z0.norm().clamp(min=1.0)
                    z_rec_full = z_next64.to(dtype=state_dtype)
                    y_rec_full = y_next64.to(dtype=state_dtype)
                    dist_t = ((z_rec_full - z0).norm() + (y_rec_full - z0).norm()) / denom_0
                    setattr(_target, "_deq_x0_recon_error_last_bwd", dist_t.detach())
            except Exception:
                pass

        # TBPTT: if we stopped before reaching iter 0, bar_y+bar_z live at an
        # intermediate iter, not at z_init — zero the grad rather than inject
        # a bogus signal into the embedding path.
        if truncated:
            z_init_grad = torch.zeros_like(x0)
        else:
            z_init_grad = (bar_y + bar_z).to(x0.dtype)
        if do_vjp_diag and diag_vjp_per_iter:
            try:
                _target = _unwrap_compiled_module(f_theta)
                setattr(_target, "_tbptt_vjp_iter_last_bwd", diag_vjp_per_iter)
                setattr(_target, "_tbptt_bwd_k_last", int(K_bwd))
                setattr(_target, "_tbptt_fwd_k_last", int(K))
            except Exception:
                pass
        param_grads_out: list[torch.Tensor | None] = [None] * len(params_all)
        for j, all_idx in enumerate(req_indices):
            g = cur_param_grads_req[j]
            if g is None:
                continue
            param_grads_out[all_idx] = g.to(dtype=params_all[all_idx].dtype)
        # grad_beta: (D,) gradient for Parcae β, or None for scalar β.
        # Autograd chain-rules from here through β = 1-exp(Δ·(-exp(log_a))) to parcae params.
        grad_beta_out = grad_beta.to(ctx.beta_input_dtype) if grad_beta is not None else None
        grad_b_bar_out = grad_b_bar.to(ctx.b_bar_input_dtype) if grad_b_bar is not None else None
        return (None, cur_x_grad.to(x0.dtype), z_init_grad, grad_beta_out,
                grad_b_bar_out, None, None, *param_grads_out)


class RevDEQPrefixAnchorFunction(torch.autograd.Function):
    """RevDEQ solve that returns several already-traversed prefix endpoints.

    Forward runs the solver once to the sampled K and stores only the requested
    prefix terminal states. Backward replays a TBPTT tail from each endpoint and
    sums the gradients; the loss-side averaging controls each anchor's weight.
    """

    @staticmethod
    def forward(ctx, f_theta, x0, z_init, beta, b_bar, K, bptt_k, anchor_depths, *params):
        acc_dtype = torch.float64
        state_dtype = torch.float32
        compute_dtype = z_init.dtype
        device_type = x0.device.type
        anchors = tuple(int(k) for k in anchor_depths)
        if not anchors:
            anchors = (int(K),)
        if anchors[-1] != int(K):
            raise ValueError(f"last prefix anchor {anchors[-1]} must equal sampled K={int(K)}")
        ctx.beta_requires_grad = isinstance(beta, torch.Tensor) and beta.requires_grad
        ctx.beta_input_dtype = beta.dtype if isinstance(beta, torch.Tensor) else None
        ctx.b_bar_requires_grad = isinstance(b_bar, torch.Tensor) and b_bar.requires_grad
        ctx.b_bar_input_dtype = b_bar.dtype if isinstance(b_bar, torch.Tensor) else None
        ctx.has_b_bar = isinstance(b_bar, torch.Tensor)
        ctx.b_bar_saved = b_bar.detach() if isinstance(b_bar, torch.Tensor) else None
        if isinstance(beta, torch.Tensor):
            beta = beta.detach().to(torch.float64)
            beta_inv = 1.0 - beta
        else:
            beta_inv = 1.0 - beta
        ctx.do_recon_diag = bool(_ROUTER_DIAGNOSTICS_ACTIVE)

        y_state = z_init.to(state_dtype)
        z_state = z_init.to(state_dtype)
        z_prev_state = z_state
        z_init_state = z_state.detach()
        anchor_set = set(anchors)
        y_anchors: list[Tensor] = []
        z_anchors: list[Tensor] = []
        # Only the final anchor's z_prev is consumed (caller reads [-1]) and
        # backward ignores it. Capturing the full list would be O(A·B·T·D)
        # dead state; keep just the last.
        last_z_prev: Tensor | None = None

        K_bwd_final = int(bptt_k) if bptt_k else 0
        if K_bwd_final <= 0 or K_bwd_final >= int(K):
            K_bwd_final = int(K)
        snap_iter = int(K) - K_bwd_final
        y_snap_state: Tensor | None = None
        z_snap_state: Tensor | None = None

        with torch.no_grad():
            if bool(ctx.do_recon_diag) and snap_iter == 0:
                y_snap_state = z_init_state
                z_snap_state = z_init_state
            out_y = None
            for i in range(int(K)):
                z_prev_state = z_state
                y_acc = y_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_z = f_theta(z_state.to(compute_dtype), x0, b_bar)
                y_acc = y_acc + out_z.to(acc_dtype) * beta
                y_state = y_acc.to(state_dtype)

                z_acc = z_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_state.to(compute_dtype), x0, b_bar)
                z_acc = z_acc + out_y.to(acc_dtype) * beta
                z_state = z_acc.to(state_dtype)

                step_i = i + 1
                if bool(ctx.do_recon_diag) and step_i == snap_iter:
                    y_snap_state = y_state.detach().clone()
                    z_snap_state = z_state.detach().clone()
                if step_i in anchor_set:
                    y_anchors.append(y_state.detach().clone())
                    z_anchors.append(z_state.detach().clone())
                    last_z_prev = z_prev_state.detach().clone()

            if out_y is not None and bool(ctx.do_recon_diag):
                # Best-effort recon-diag capture: writes are diagnostic-only,
                # so a failure must never crash training. `AttributeError` is
                # the paranoid swallow for future compile-wrapper-structure
                # changes (today `_unwrap_compiled_module` does not raise, but
                # private torch APIs shift); any other exception is a real bug
                # surfaced via `warnings.warn` so it lands in run.log instead
                # of being silently dropped.
                try:
                    _target = _unwrap_compiled_module(f_theta)
                    setattr(
                        _target,
                        "_deq_residual_proxy_t",
                        (z_state - out_y.to(state_dtype)).norm().detach(),
                    )
                    denom0 = z_init_state.to(state_dtype).norm().clamp(min=1.0)
                    setattr(
                        _target,
                        "_deq_fp_travel_last_fwd",
                        ((z_state - z_init_state.to(state_dtype)).norm() / denom0).detach(),
                    )
                except AttributeError:
                    pass
                except Exception as e:
                    warnings.warn(f"deq_recon_diag_capture_failed[prefix]: {type(e).__name__}: {e}")

        if len(z_anchors) != len(anchors):
            raise RuntimeError(f"captured {len(z_anchors)} anchors for requested {anchors}")
        if last_z_prev is None:
            raise RuntimeError(f"no anchor captured for K={K} anchors={anchors}")
        y_stack = torch.stack(y_anchors, dim=0)
        z_stack = torch.stack(z_anchors, dim=0)
        z_prev_stack = last_z_prev.unsqueeze(0)
        ctx.save_for_backward(x0.detach(), y_stack.detach(), z_stack.detach())
        ctx.z_init_state = z_init_state
        ctx.y_snap_state = y_snap_state
        ctx.z_snap_state = z_snap_state
        ctx.f_theta = f_theta
        ctx.beta = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta, dtype=torch.float64)
        ctx.beta_inv = beta_inv if isinstance(beta_inv, torch.Tensor) else torch.tensor(beta_inv, dtype=torch.float64)
        ctx.K = int(K)
        ctx.anchor_depths = anchors
        ctx.bptt_k = int(bptt_k) if bptt_k else 0
        ctx.compute_dtype = compute_dtype
        ctx.device_type = device_type
        ctx.params = params
        return z_stack.to(compute_dtype), z_prev_stack.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_z_stack, _grad_z_prev_stack_ignored):
        x0, y_stack, z_stack = (t.detach() for t in ctx.saved_tensors)
        f_theta = ctx.f_theta
        beta, beta_inv = ctx.beta, ctx.beta_inv
        compute_dtype = ctx.compute_dtype
        device_type = ctx.device_type
        acc_dtype = torch.float64
        state_dtype = torch.float32
        anchors = tuple(int(k) for k in getattr(ctx, "anchor_depths", (ctx.K,)))

        params_all = tuple(ctx.params)
        req_indices = [i for i, p in enumerate(params_all) if getattr(p, "requires_grad", False)]
        params_req = tuple(params_all[i] for i in req_indices)
        param_grads_req: list[torch.Tensor | None] = [None] * len(params_req)
        cur_x_grad = torch.zeros_like(x0, dtype=torch.float32)
        z_init_grad = torch.zeros_like(x0)

        b_bar_saved = ctx.b_bar_saved
        b_bar_requires_grad = bool(getattr(ctx, "b_bar_requires_grad", False))
        b_bar_local_base: Tensor | None = None
        grad_b_bar: torch.Tensor | None = None
        if bool(getattr(ctx, "has_b_bar", False)):
            b_dtype = getattr(ctx, "b_bar_input_dtype", None) or compute_dtype
            b_bar_local_base = b_bar_saved.detach().to(dtype=b_dtype)
            if b_bar_requires_grad:
                grad_b_bar = torch.zeros_like(b_bar_local_base, dtype=torch.float32)

        beta_requires_grad = ctx.beta_requires_grad
        grad_beta: torch.Tensor | None = torch.zeros_like(beta) if beta_requires_grad else None
        diag_vjp_per_iter: list[tuple[float, float]] = []
        do_vjp_diag = bool(getattr(ctx, "do_recon_diag", False))

        if grad_z_stack is None:
            grad_z_stack = torch.zeros_like(z_stack)

        for anchor_idx, K_anchor in enumerate(anchors):
            bar_z = grad_z_stack[anchor_idx].to(state_dtype)
            bar_y = torch.zeros_like(bar_z)
            y_next64 = y_stack[anchor_idx].to(acc_dtype)
            z_next64 = z_stack[anchor_idx].to(acc_dtype)
            bptt_k = int(getattr(ctx, "bptt_k", 0) or 0)
            K_bwd = K_anchor if (bptt_k <= 0 or bptt_k >= K_anchor) else bptt_k
            truncated = K_bwd < K_anchor
            collect_diag = do_vjp_diag and anchor_idx == len(anchors) - 1

            for _ in range(K_bwd):
                y_local = y_next64.detach().to(compute_dtype).requires_grad_()
                x_local = x0.detach().to(x0.dtype).requires_grad_()
                b_bar_y = None
                if b_bar_local_base is not None:
                    b_bar_y = b_bar_local_base.clone().requires_grad_(b_bar_requires_grad)
                with torch.enable_grad():
                    with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                        out_y = f_theta(y_local, x_local, b_bar_y)
                z_n64 = (z_next64 - out_y.detach().to(acc_dtype) * beta) / beta_inv
                if grad_beta is not None:
                    grad_beta += (bar_z.to(acc_dtype) * (out_y.detach().to(acc_dtype) - z_n64)).sum(dim=(0, 1))

                grad_seed_y = (beta * bar_z).to(out_y.dtype)
                if b_bar_requires_grad:
                    grads_y = torch.autograd.grad(out_y, (y_local, x_local, b_bar_y, *params_req),
                                                  grad_outputs=grad_seed_y, allow_unused=True)
                    gy_b = grads_y[2]
                    y_param_offset = 3
                else:
                    grads_y = torch.autograd.grad(out_y, (y_local, x_local, *params_req),
                                                  grad_outputs=grad_seed_y, allow_unused=True)
                    gy_b = None
                    y_param_offset = 2
                vjp_y = grads_y[0].to(state_dtype)
                bar_y_acc = bar_y + vjp_y

                z_local = z_n64.detach().to(compute_dtype).requires_grad_()
                x_local2 = x0.detach().to(x0.dtype).requires_grad_()
                b_bar_z = None
                if b_bar_local_base is not None:
                    b_bar_z = b_bar_local_base.clone().requires_grad_(b_bar_requires_grad)
                with torch.enable_grad():
                    with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                        out_z = f_theta(z_local, x_local2, b_bar_z)
                y_n64 = (y_next64 - out_z.detach().to(acc_dtype) * beta) / beta_inv
                if grad_beta is not None:
                    grad_beta += (bar_y_acc.to(acc_dtype) * (out_z.detach().to(acc_dtype) - y_n64)).sum(dim=(0, 1))

                grad_seed_z = (beta * bar_y_acc).to(out_z.dtype)
                if b_bar_requires_grad:
                    grads_z = torch.autograd.grad(out_z, (z_local, x_local2, b_bar_z, *params_req),
                                                  grad_outputs=grad_seed_z, allow_unused=True)
                    gz_b = grads_z[2]
                    z_param_offset = 3
                else:
                    grads_z = torch.autograd.grad(out_z, (z_local, x_local2, *params_req),
                                                  grad_outputs=grad_seed_z, allow_unused=True)
                    gz_b = None
                    z_param_offset = 2
                vjp_z = grads_z[0].to(state_dtype)

                if collect_diag:
                    diag_vjp_per_iter.append((vjp_y.detach().norm(), vjp_z.detach().norm()))

                bar_z = beta_inv * bar_z + vjp_z
                bar_y = beta_inv * bar_y_acc
                if grad_b_bar is not None and (gy_b is not None or gz_b is not None):
                    grad_b_bar = grad_b_bar + ((gy_b if gy_b is not None else 0.0)
                                               + (gz_b if gz_b is not None else 0.0)).detach().float()
                for j in range(len(params_req)):
                    gy = grads_y[y_param_offset + j]
                    gz = grads_z[z_param_offset + j]
                    if gy is None and gz is None:
                        continue
                    g = (gy if gy is not None else 0.0) + (gz if gz is not None else 0.0)
                    param_grads_req[j] = g.detach() if param_grads_req[j] is None else param_grads_req[j] + g.detach()
                if grads_y[1] is not None:
                    cur_x_grad += grads_y[1].detach().float()
                if grads_z[1] is not None:
                    cur_x_grad += grads_z[1].detach().float()
                y_next64, z_next64 = y_n64, z_n64

            if truncated:
                z_init_grad = z_init_grad + torch.zeros_like(x0)
            else:
                z_init_grad = z_init_grad + (bar_y + bar_z).to(x0.dtype)

            if collect_diag:
                try:
                    _target = _unwrap_compiled_module(f_theta)
                    setattr(_target, "_tbptt_vjp_iter_last_bwd", diag_vjp_per_iter)
                    setattr(_target, "_tbptt_bwd_k_last", int(K_bwd))
                    setattr(_target, "_tbptt_fwd_k_last", int(K_anchor))
                    y_snap_state = getattr(ctx, "y_snap_state", None)
                    z_snap_state = getattr(ctx, "z_snap_state", None)
                    if isinstance(y_snap_state, torch.Tensor) and isinstance(z_snap_state, torch.Tensor):
                        denom_s = z_snap_state.to(dtype=state_dtype).norm().clamp(min=1.0)
                        recon_err_t = (
                            (z_next64.to(dtype=state_dtype) - z_snap_state.to(dtype=state_dtype)).norm()
                            + (y_next64.to(dtype=state_dtype) - y_snap_state.to(dtype=state_dtype)).norm()
                        ) / denom_s
                        setattr(_target, "_deq_recon_error_last_bwd", recon_err_t.detach())
                except Exception:
                    pass

        param_grads_out: list[torch.Tensor | None] = [None] * len(params_all)
        for j, all_idx in enumerate(req_indices):
            g = param_grads_req[j]
            if g is not None:
                param_grads_out[all_idx] = g.to(dtype=params_all[all_idx].dtype)
        grad_beta_out = grad_beta.to(ctx.beta_input_dtype) if grad_beta is not None else None
        grad_b_bar_out = grad_b_bar.to(ctx.b_bar_input_dtype) if grad_b_bar is not None else None
        return (None, cur_x_grad.to(x0.dtype), z_init_grad, grad_beta_out,
                grad_b_bar_out, None, None, None, *param_grads_out)


# ---------------------------------------------------------------------------
# GPT MODEL
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: float, tie_embeddings: bool,
                 tied_embed_init_std: float, rope_base: float,
                 qk_gain_init: float, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 kv_latent_dim: int = 0, num_refinements: int = 0,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 deq_beta: float = 0.50,
                 deq_bptt_k: int = 3,
                 # Defaults below mirror Hyperparameters (current rescue stack,
                 # 2026-05-08). Keep these aligned: experiments/test_arch.py
                 # constructs GPT() without overrides and asserts the values.
                 router_scoring: str = "dirichlet_ucb",
                 router_pertoken_entropy_coef: float = 0.1,
                 router_dirichlet_ucb_beta: float = 0.5,
                 use_router_sigmoid_gate: bool = False,
                 use_entmax_routing: bool = False,
                 entmax_blend_init_logit: float = 5.0,
                 entmax_blend_warmup_delay_frac: float = 0.3,
                 use_smear_gate: bool = False,
                 smear_gate_init: float = 0.0,
                 smear_gate_window: int = 12,
                 smear_gate_bos_id: int = 1,
                 logit_softcap: float = 0.0,
                 num_experts: int = 16, num_shared_experts: int = 0,
                 lyapunov_coef: float = 0.0,
                 lyapunov_gamma: float = 0.97,
                 lyapunov_every: int = 16,
                 lyapunov_max_tokens: int = 64,
                 lyapunov_target: str = "transition_T",
                 lyapunov_estimator: str = "random_fd",
                 deq_prefix_anchors: bool = True,
                 deq_prefix_anchor_set: tuple[int, ...] | None = (8, 16, 24, 32, 64, 128),
                 use_parcae: bool = True,
                 parcae_init_a_bar: float = 0.7,
                 parcae_init_b_bar: float | None = None,
                 use_ctp: bool = False,
                 ctp_weight: float = 0.0,
                 router_ema_alive_coef: float = 0.02,
                 router_ema_balance_coef: float = 0.30,
                 router_ema_specialization_coef: float = 0.20,
                 use_reverse_kl_balance: bool = True,
                 use_expert_perdim_gate: bool = False,
                 expert_perdim_gate_rank: int = 16,
                 multi_k_consistency_anchor_coef: float = 0.1,
                 expert_diversity_kind: str = "cosine",
                 expert_output_diversity_coef: float = 0.30,
                 expert_diversity_every: int = 8,
                 expert_diversity_max_tokens: int = 64,
                 mos_output_diversity_coef: float = 0.0,
                 regularizer_warmup_frac: float = 0.07,
                 use_nsa_attention: bool = False,
                 nsa_compress_block_size: int = 32,
                 nsa_compress_block_sliding_stride: int = 16,
                 nsa_sliding_window_size: int = 256,
                 nsa_branch_gate_init: float = 0.0,
                 use_sparse_attn_head_gate: bool = False,
                 sparse_attn_gate_window: int = 12,
                 sparse_attn_gate_scale: float = 1.0,
                 sparse_attn_gate_factor: float = 2.0,
                 sparse_attn_gate_init_std: float = 0.0,
                 use_rr_attention: bool = False,
                 rr_stride: int = 8,
                 rr_block_size: int = 64,
                 rr_tau: float = 0.95,
                 chained_stages_preset: str | None = None):
        super().__init__()
        self.use_ctp = bool(use_ctp)
        self.ctp_weight = float(ctp_weight)
        self.router_ema_alive_coef = float(router_ema_alive_coef)
        self.router_ema_balance_coef = float(router_ema_balance_coef)
        self.router_ema_specialization_coef = float(router_ema_specialization_coef)
        self.multi_k_consistency_anchor_coef = float(multi_k_consistency_anchor_coef)
        # Stash for diagnostics + readback in compute_loss; populated in _run_backbone.
        self._consistency_anchor_loss_t: Tensor | None = None
        self.expert_diversity_kind = str(expert_diversity_kind)
        if self.expert_diversity_kind not in ("frobenius", "cosine"):
            raise ValueError(
                f"expert_diversity_kind must be 'frobenius' or 'cosine'; "
                f"got {self.expert_diversity_kind!r}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.num_layers = num_layers
        self.num_refinements = num_refinements
        # All regularization coefficients are annealed by a single
        # `regularizer_warmup_frac` schedule in the training loop. With the
        # promoted default warmup=0, this is behavior-identical to step-0
        # regularization; iter145 uses warmup=0.07 for a gentler cold start.
        self._router_ema_alive_coef_target = float(router_ema_alive_coef)
        self._router_ema_balance_coef_target = float(router_ema_balance_coef)
        self._router_ema_specialization_coef_target = float(router_ema_specialization_coef)
        self._router_pertoken_entropy_coef_target = float(router_pertoken_entropy_coef)
        self._router_dirichlet_ucb_beta_target = float(router_dirichlet_ucb_beta)
        self.regularizer_warmup_frac = float(regularizer_warmup_frac)
        # entmax_blend_anneal still has its own delay (it's a routing-form
        # behavior, not a loss-coefficient ramp; keep separate).
        self._use_entmax_routing = bool(use_entmax_routing)
        self._entmax_blend_warmup_delay_frac = float(entmax_blend_warmup_delay_frac)
        # Per-token expert-output diversity targets + scratch.
        self._expert_diversity_coef_target = float(expert_output_diversity_coef)
        self._expert_diversity_every = int(expert_diversity_every)
        self._expert_diversity_max_tokens = int(expert_diversity_max_tokens)
        self._expert_diversity_coef_scale = 1.0  # set per-step by annealer
        self._expert_diversity_loss: Tensor | None = None
        self._expert_diversity_token_start = 0
        # MoS per-token output-diversity (mirrors block-side; default off via coef=0).
        self.mos_output_diversity_coef = float(mos_output_diversity_coef)
        self._mos_diversity_loss: Tensor | None = None
        self._router_cv_loss_t: Tensor | None = None
        self._router_pertoken_entropy_loss_t: Tensor | None = None
        # iter145r EMA-anchored family loss/coef caches — see ROUTER_EMA_LOSS_TERMS.
        for _name, _ in ROUTER_EMA_LOSS_TERMS:
            setattr(self, f"_router_ema_{_name}_loss_t", None)
            setattr(self, f"_router_ema_{_name}_coef_eff_t", None)
        self._mos_cv_loss_t: Tensor | None = None
        self._expert_diversity_loss_t: Tensor | None = None
        self._mos_diversity_loss_t: Tensor | None = None
        self._router_pertoken_entropy_coef_eff_t: Tensor | None = None
        self._expert_diversity_coef_eff_t: Tensor | None = None
        self._mos_diversity_coef_eff_t: Tensor | None = None
        self._router_reg_loss_t: Tensor | None = None
        # iter 129 / H99: SmearGate — single learnable scalar shared across the
        # entire backbone (we have one shared Block; per-layer doesn't apply).
        # The active component implements the records-style input-dependent
        # narrow gate, applied after token embedding lookup with BOS masking.
        # Strict-gen at use_smear_gate=False -> no parameter, no compute.
        self.use_smear_gate = bool(use_smear_gate)
        self.smear_gate_bos_id = int(smear_gate_bos_id)
        if self.use_smear_gate:
            from experiments.components.smear_gate import SmearGate
            self.smear_gate = SmearGate(
                dim=model_dim,
                window=int(smear_gate_window),
                bos_id=int(smear_gate_bos_id),
            )
            with torch.no_grad():
                self.smear_gate.lam.fill_(float(smear_gate_init))
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        # Invariant: GPT.num_experts == shared_block.{attn,mlp}.num_experts
        # (threaded from Hyperparameters; verified by experiments/test_arch.py).
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.shared_block = Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                                   rope_base, qk_gain_init, kv_latent_dim=kv_latent_dim,
                                   attn_expert_rank=attn_expert_rank, mlp_expert_rank=mlp_expert_rank,
                                   num_experts=self.num_experts,
                                   num_shared_experts=self.num_shared_experts,
                                   router_scoring=router_scoring,
                                   router_pertoken_entropy_coef=router_pertoken_entropy_coef,
                                   router_dirichlet_ucb_beta=router_dirichlet_ucb_beta,
                                   use_router_sigmoid_gate=use_router_sigmoid_gate,
                                   use_entmax_routing=use_entmax_routing,
                                   entmax_blend_init_logit=entmax_blend_init_logit,
                                   use_reverse_kl_balance=use_reverse_kl_balance,
                                   use_nsa_attention=use_nsa_attention,
                                   nsa_compress_block_size=nsa_compress_block_size,
                                   nsa_compress_block_sliding_stride=nsa_compress_block_sliding_stride,
                                   nsa_sliding_window_size=nsa_sliding_window_size,
                                   nsa_branch_gate_init=nsa_branch_gate_init,
                                   use_sparse_attn_head_gate=use_sparse_attn_head_gate,
                                   sparse_attn_gate_window=sparse_attn_gate_window,
                                   sparse_attn_gate_scale=sparse_attn_gate_scale,
                                   sparse_attn_gate_factor=sparse_attn_gate_factor,
                                   sparse_attn_gate_init_std=sparse_attn_gate_init_std,
                                   use_rr_attention=use_rr_attention,
                                   rr_stride=rr_stride,
                                   rr_block_size=rr_block_size,
                                   rr_tau=rr_tau,
                                   chained_stages_preset=chained_stages_preset,
                                   use_expert_perdim_gate=use_expert_perdim_gate,
                                   expert_perdim_gate_rank=expert_perdim_gate_rank,
                                   )
        self.deq_beta = float(deq_beta)
        # Phase 9 iter 66b: Parcae-style per-dim diagonal damping with
        # independent input gain (arXiv:2604.12946, Mamba-style ZOH),
        # adapted to RevDEQ's reversible two-state solver.
        #   Δ  = softplus(parcae_raw_delta) + ε_min       (step size)
        #   A  = -(softplus(parcae_raw_a)   + ε_min)      (decay, < 0)
        #   B  =   softplus(parcae_raw_b)   + ε_min       (input gain, > 0)
        #   Ā_core = exp(Δ · A)                           (paper form, in (0,1))
        #   Ā = ε_rev + (1 − ε_rev) · Ā_core              (RevDEQ safety)
        #   B̄ = Δ · B                                     (paper form, no floor)
        #   β = 1 − Ā                                     (solver blend, tied to Ā)
        # Gradient flows: loss → RevDEQFunction.backward(grad_beta) → autograd
        # → parcae_raw_a, parcae_raw_delta, parcae_raw_b (via Block.forward).
        self.use_parcae = bool(use_parcae)
        # ε_min: softplus lower bound; prevents exact-zero Δ, |A|, B at init.
        self.parcae_min_rate = 1e-3
        # ε_rev: CORRECTNESS CONSTANT — NOT a tuning knob. RevDEQ backward
        # reconstructs via y_n ← (y_{n+1} − β·T)/(1 − β) where (1 − β) = Ā.
        # Error amplifies by 1/Ā per backward step, so over K iterations the
        # worst-case bound is (1/Ā)^K · ε_fp64. For K=num_layers=12 and fp64
        # (ε_fp64 ≈ 1e-15), Ā ≥ 0.1 keeps backward error ≤ 10^12 · 1e-15 = 1e-3,
        # which the smoke-test tolerance (1e-1) expects. Smaller ε_rev
        # compounds per-iteration error through K backward steps — not safe.
        # Changing it requires updating the reversibility contract test +
        # smoke-test tolerance together.
        self.parcae_reversibility_floor = 0.1
        if self.use_parcae:
            raw_a_init, raw_delta_init, raw_b_init = self._parcae_init_raw_values(
                float(parcae_init_a_bar), parcae_init_b_bar,
            )
            self.parcae_raw_a = nn.Parameter(torch.full((model_dim,), raw_a_init))
            self.parcae_raw_delta = nn.Parameter(torch.full((model_dim,), raw_delta_init))
            self.parcae_raw_b = nn.Parameter(torch.full((model_dim,), raw_b_init))
        self.deq_bptt_k = int(deq_bptt_k)
        self.deq_prefix_anchors = bool(deq_prefix_anchors)
        self.deq_prefix_anchor_set = tuple(int(k) for k in (deq_prefix_anchor_set or ()))
        self._deq_prefix_anchor_depths_last: tuple[int, ...] = ()
        # iter147/iter155 path: finite-perturbation Hutchinson Frobenius/√D
        # probe on T_θ (default), the single-state convex blend
        # S = Ā·z+(1−Ā)·T_θ (when `lyapunov_target=iteration_S`; advisory
        # only — NOT the iterated map), or the actual two-state Parcae cycle
        # F (when `lyapunov_target=iteration_F`).  Default OFF, run
        # low-cadence from the training loop. NOT an operator-norm probe —
        # the principled gate is `rho_F` (spectral radius via power
        # iteration on J_F; Hartman-Grobman necessary AND sufficient).
        self.lyapunov_coef = float(lyapunov_coef)
        self.lyapunov_gamma = float(lyapunov_gamma)
        self.lyapunov_every = int(lyapunov_every)
        self.lyapunov_max_tokens = int(lyapunov_max_tokens)
        # iter155 (corrected): enum {"transition_T","iteration_S","iteration_F"}; selects which Jacobian
        # the FD probe penalizes. Validated in `_validate_hyperparameters`.
        self.lyapunov_target = str(lyapunov_target)
        # iter155 (corrected): enum currently only accepts {"random_fd"} — the
        # `power_jvp_F` worst-direction branch was removed 2026-05-15 alongside
        # the lip_ub_* operator-norm probes (refuted by iter155/iter152). The
        # field is retained because the validator still names it, but should be
        # collapsed to a constant in a follow-up cleanup.
        self.lyapunov_estimator = str(lyapunov_estimator)
        self.logit_softcap = float(logit_softcap)
        self.mos_head = MoSHead(
            model_dim, vocab_size, rank=256,
            num_shared=2, num_specialized=1, fsq_levels=0,
            use_ctp=self.use_ctp, logit_softcap=self.logit_softcap,
            mos_output_diversity_kind=expert_diversity_kind,  # follow same kind
        )
        self.final_norm = RMSNorm(model_dim)
        # Embedding/final norms remain learnable shared scales outside T_theta.
        self.embed_norm = RMSNorm(model_dim)
        self._init_weights()

    def _parcae_init_raw_values(self, init_a_bar: float, init_b_bar: float | None = None) -> tuple[float, float, float]:
        """Invert the paper forms to choose raw params at initialization.

        Picks Δ₀, |A|₀, B₀ so Ā₀ ≈ init_a_bar and B̄₀ ≈ init_b_bar.
        ``init_b_bar=None`` defaults to ``1 − init_a_bar`` (iter 66a continuity
        with tied β = 1 − Ā). Returns raw values that pass through softplus+ε_min
        to recover the targets exactly (modulo the safety ε_min offset on |A|, B).
        """
        if init_b_bar is None:
            init_b_bar = 1.0 - float(init_a_bar)
        eps_min = float(self.parcae_min_rate)
        eps_rev = float(self.parcae_reversibility_floor)

        def inv_softplus(y: float) -> float:
            return math.log(math.expm1(max(float(y), 1e-12)))

        # Δ₀ = 1 (unit step) via softplus(raw_delta) + ε_min = 1.
        raw_delta_init = inv_softplus(1.0 - eps_min)
        delta0 = 1.0

        # Ā₀ = ε_rev + (1 − ε_rev) · exp(Δ₀ · A₀).  Solve for |A|₀:
        #   exp(−Δ₀·|A|₀) = (init_a_bar − ε_rev) / (1 − ε_rev)
        clamped_a = min(max(float(init_a_bar), eps_rev + 1e-6), 1.0 - 1e-6)
        a_bar_core = (clamped_a - eps_rev) / max(1.0 - eps_rev, 1e-8)
        a_mag = -math.log(a_bar_core) / delta0  # |A|₀
        raw_a_init = inv_softplus(max(a_mag - eps_min, 1e-8))

        # B̄₀ = Δ₀ · B₀; solve for B₀ given target init_b_bar.
        b_mag = max(float(init_b_bar) / delta0, eps_min + 1e-8)
        raw_b_init = inv_softplus(b_mag - eps_min)
        return raw_a_init, raw_delta_init, raw_b_init

    def _parcae_delta(self) -> Tensor:
        # Δ = softplus(raw_delta) + ε_min — shared by Ā and B̄.
        return F.softplus(self.parcae_raw_delta.float()) + float(self.parcae_min_rate)

    def _parcae_a_bar(self) -> Tensor:
        # Paper form Ā_core = exp(Δ · A) with A = -(softplus(raw_a) + ε_min),
        # rescaled to (ε_rev, 1) for RevDEQ reversibility (β = 1 − Ā ≤ 1 − ε_rev,
        # so the backward reconstruction denominator never approaches zero).
        delta = self._parcae_delta()
        a_mag = F.softplus(self.parcae_raw_a.float()) + float(self.parcae_min_rate)
        a_bar_core = torch.exp(-(delta * a_mag))
        eps_rev = float(self.parcae_reversibility_floor)
        return eps_rev + (1.0 - eps_rev) * a_bar_core

    def _parcae_b_bar(self) -> Tensor:
        # Pure paper form B̄ = Δ · B (Mamba ZOH). No floor — B̄ never appears
        # in the solver reconstruction, only inside T(z, x₀), so reversibility
        # does not constrain it and adding a floor would silently bias training.
        delta = self._parcae_delta()
        b_mag = F.softplus(self.parcae_raw_b.float()) + float(self.parcae_min_rate)
        return delta * b_mag

    def parcae_diagnostics(self, k_override: int | None = None) -> dict[str, Tensor]:
        """Detached scalar diagnostics for the Parcae-style RevDEQ damping state."""
        if not self.use_parcae:
            return {}
        delta = self._parcae_delta().detach().float()
        a_mag = F.softplus(self.parcae_raw_a.detach().float()) + float(self.parcae_min_rate)
        a_bar_core = torch.exp(-(delta * a_mag))
        eps_rev = float(self.parcae_reversibility_floor)
        a_bar = eps_rev + (1.0 - eps_rev) * a_bar_core
        beta = 1.0 - a_bar
        b_mag = F.softplus(self.parcae_raw_b.detach().float()) + float(self.parcae_min_rate)
        b_bar = delta * b_mag
        k = int(k_override or getattr(self, "_deq_k_override", 0) or self.num_layers)
        a_bar_min = a_bar.min().clamp_min(1e-12)
        return {
            "parcae_a_bar_min": a_bar.min(),
            "parcae_a_bar_mean": a_bar.mean(),
            "parcae_a_bar_max": a_bar.max(),
            "parcae_a_bar_core_max": a_bar_core.max(),
            "parcae_beta_mean": beta.mean(),
            "parcae_beta_max": beta.max(),
            "parcae_b_bar_mean": b_bar.mean(),
            "parcae_b_bar_max": b_bar.max(),
            "parcae_delta_mean": delta.mean(),
            "parcae_delta_max": delta.max(),
            "parcae_recon_amp_log10": a_bar_min.reciprocal().log10() * float(max(k, 1)),
        }

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * self.num_layers))
        # Gate logits are zero-initialized in CausalSelfAttention.__init__
        # (expert_q_up gate rows zeroed → sigmoid(0) = 0.5 at init).

    def _get_soft_embedding(self, z: Tensor, topk: int = 64) -> Tensor:
        def _logp_to_prob(log_p: Tensor) -> Tensor:
            p = log_p.float().exp()
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.no_grad():
            h = self.final_norm(z)
            was_training = self.mos_head.training
            self.mos_head.train(False)
            log_p_ctp, log_p_ntp = self.mos_head(h)
            self.mos_head.train(was_training)
            torch.clear_autocast_cache()

            log_p_ntp_shifted = torch.cat([log_p_ntp[:, :1], log_p_ntp[:, :-1]], dim=1)
            p_ntp = _logp_to_prob(log_p_ntp_shifted)
            if self.use_ctp:
                p_ctp = _logp_to_prob(log_p_ctp)
                p_mix = 0.5 * (p_ctp + p_ntp)
                p_mix[:, 0] = p_ctp[:, 0]
            else:
                p_mix = p_ntp
            p_mix = torch.nan_to_num(p_mix, nan=0.0, posinf=0.0, neginf=0.0)

            k = min(int(topk), int(p_mix.shape[-1]))
            topk_probs, topk_idx = p_mix.topk(k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            W = self.tok_emb.weight.data
            B, T, K = topk_idx.shape
            d = W.shape[1]
            flat_idx = topk_idx.reshape(B * T, K)
            flat_p = topk_probs.reshape(B * T, K)
            flat_out = W.new_empty((B * T, d))
            chunk = 512
            for s in range(0, B * T, chunk):
                e = min(s + chunk, B * T)
                emb = F.embedding(flat_idx[s:e], W)
                flat_out[s:e] = (flat_p[s:e].unsqueeze(-1) * emb).sum(dim=1)
            soft_embed = flat_out.reshape(B, T, d)
            soft_embed = soft_embed.to(dtype=z.dtype)
        return soft_embed

    def _deq_solve(self, x0: Tensor, z_init: Tensor):
        global _DEQ_SOLVE_ACTIVE
        # Phase 9 iter 66b: Parcae-style per-dim damping + Parcae input gain.
        #   β = 1 − Ā  (solver blend, carries grad → parcae_raw_a / raw_delta)
        #   B̄ = Δ · B (Block.forward input gain, carries grad → parcae_raw_b / raw_delta)
        if self.use_parcae:
            a_bar = self._parcae_a_bar()  # bounded in (parcae_reversibility_floor, 1)
            beta = 1.0 - a_bar  # per-dim β, shape (D,), requires_grad=True
            b_bar = self._parcae_b_bar()  # per-dim B̄ for Block.forward input injection
        else:
            beta = self.deq_beta
            b_bar = None
        dtype = x0.dtype
        K = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
        track_diag = _should_diag(self.training) and bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        sb = _unwrap_compiled_module(self.shared_block)
        sb._diag_track_enabled = bool(track_diag)
        stack = getattr(sb, "chained_stack", None)
        if stack is not None:
            stack._diag_track_enabled = bool(track_diag)
            stack._attn_gate_call_track = []
            stack._router_gate_call_track = []
            stack._attn_router_gate_call_track = []
            stack._attn_expert_weights_per_iter = []
            stack._mlp_expert_weights_per_iter = []
        sb._attn_gate_call_track = []
        sb._router_gate_call_track = []
        sb._attn_router_gate_call_track = []
        prev_deq_flag = bool(_DEQ_SOLVE_ACTIVE)
        _DEQ_SOLVE_ACTIVE = True
        try:
            f_theta = self.shared_block
            if self.training:
                params = tuple(p for p in sb.parameters() if p.requires_grad)
                bptt_k = int(getattr(self, "deq_bptt_k", 0) or 0)
                z, z_prev = RevDEQFunction.apply(
                    f_theta, x0, z_init, beta, b_bar, K, bptt_k, *params
                )
                return z, z_prev, None, None

            # FP32 accumulators for the unrolled solver (eval path only — train uses revdeq).
            # FP64 is only needed inside RevDEQFunction for exact reversibility.
            acc_dtype = torch.float32
            y_acc = z_init.to(acc_dtype)
            z_acc = z_init.to(acc_dtype)
            z = z_init
            z_prev = z
            for _ in range(K):
                z_prev = z
                f_z = f_theta(z, x0, b_bar)
                y_acc = (1 - beta) * y_acc + beta * f_z.to(acc_dtype)
                y = y_acc.to(dtype)
                f_y = f_theta(y, x0, b_bar)
                z_acc = (1 - beta) * z_acc + beta * f_y.to(acc_dtype)
                z = z_acc.to(dtype)
            return z, z_prev, y_acc, z_acc
        finally:
            _DEQ_SOLVE_ACTIVE = prev_deq_flag
            sb._diag_track_enabled = False
            if stack is not None:
                stack._diag_track_enabled = False
            # Aggregate per-iteration gate stats. Producers (Block.forward,
            # SoftDenseRouter.forward, CausalSelfAttention.forward) now store 0-d
            # GPU tensors instead of Python floats — the materialization to floats
            # happens HERE, outside the compiled forward, with one .item() per
            # iteration (CLAUDE.md §9 "Hot-path sync prohibition"-compliant since
            # this finally block runs after the K-iteration solver loop exits,
            # not inside it).
            def _materialize_pairs(track: list) -> list[float]:
                if len(track) == 2 * K:
                    pairs = [0.5 * (track[2*i] + track[2*i+1]) for i in range(K)]
                    return [float(p.item()) if torch.is_tensor(p) else float(p) for p in pairs]
                if len(track) > 0 and len(track) % (2 * K) == 0:
                    per_solve_call = len(track) // (2 * K)
                    vals = []
                    for i in range(K):
                        chunk = track[(2 * i * per_solve_call):(2 * (i + 1) * per_solve_call)]
                        avg = sum(chunk) / float(len(chunk))
                        vals.append(float(avg.item()) if torch.is_tensor(avg) else float(avg))
                    return vals
                else:
                    return []

            self._attn_gate_iter_last_solve = _materialize_pairs(
                list(getattr(stack if stack is not None else sb, "_attn_gate_call_track", []) or []))
            self._router_gate_iter_last_solve = _materialize_pairs(
                list(getattr(stack if stack is not None else sb, "_router_gate_call_track", []) or []))
            self._attn_router_gate_iter_last_solve = _materialize_pairs(
                list(getattr(stack if stack is not None else sb, "_attn_router_gate_call_track", []) or []))
            # mlp track was a value-equal duplicate; mlp readers consume the
            # attn track (pooled router post iter 35).
            self._mlp_router_gate_iter_last_solve = self._attn_router_gate_iter_last_solve

            # Per-expert routing weights per iteration (2 calls per iter: y-update,
            # z-update).  The producer at Block.forward stored each entry as a
            # detached GPU tensor of shape (E,); we stack + pair-mean here but
            # KEEP THE RESULT ON GPU as `_attn_expert_weights_iter_t` — the .cpu()
            # materialization is deferred to the log-emission site
            # (`format_iter_dynamics_info`), which runs OUTSIDE the micro_step
            # hot range (CLAUDE.md §9 "Hot-path sync prohibition").
            track_owner = stack if stack is not None else sb
            attn_ew = list(getattr(track_owner, "_attn_expert_weights_per_iter", []) or [])
            if stack is not None:
                self._attn_expert_weights_iter_t = None
            elif len(attn_ew) == 2 * K:
                # (2K, E) → reshape (K, 2, E) → mean over the 2-call axis → (K, E)
                self._attn_expert_weights_iter_t = (
                    torch.stack(attn_ew, dim=0).reshape(K, 2, -1).mean(dim=1)
                )
            else:
                self._attn_expert_weights_iter_t = None
            self._attn_expert_weights_iter: list[list[float]] | None = None  # lazy materialization
            track_owner._attn_expert_weights_per_iter = []
            track_owner._mlp_expert_weights_per_iter = []

            # Backward compat: after a DEQ solve with router_diagnostics enabled,
            # materialize list-form router diagnostics exactly once (not per-iter).
            # The `hasattr` guard below already filters routers that lack the
            # optional helper, so `AttributeError` here is a paranoid swallow
            # for race-window changes; any other exception is a real bug —
            # surface via warnings.warn.
            if track_diag and bool(_ROUTER_DIAGNOSTICS_ACTIVE):
                try:
                    for r in _iter_unique_routers(sb):
                        if hasattr(r, "_materialize_diag_lists"):
                            r._materialize_diag_lists()
                except AttributeError:
                    pass
                except Exception as e:
                    warnings.warn(f"router_diag_materialize_failed: {type(e).__name__}: {e}")

    def _deq_solve_prefix_anchors(self, x0: Tensor, z_init: Tensor, anchor_depths: tuple[int, ...]):
        global _DEQ_SOLVE_ACTIVE
        if self.use_parcae:
            a_bar = self._parcae_a_bar()
            beta = 1.0 - a_bar
            b_bar = self._parcae_b_bar()
        else:
            beta = self.deq_beta
            b_bar = None
        K = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
        track_diag = _should_diag(self.training) and bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        sb = _unwrap_compiled_module(self.shared_block)
        sb._diag_track_enabled = bool(track_diag)
        stack = getattr(sb, "chained_stack", None)
        if stack is not None:
            stack._diag_track_enabled = bool(track_diag)
            stack._attn_gate_call_track = []
            stack._router_gate_call_track = []
            stack._attn_router_gate_call_track = []
            stack._attn_expert_weights_per_iter = []
            stack._mlp_expert_weights_per_iter = []
        sb._attn_gate_call_track = []
        sb._router_gate_call_track = []
        sb._attn_router_gate_call_track = []
        prev_deq_flag = bool(_DEQ_SOLVE_ACTIVE)
        _DEQ_SOLVE_ACTIVE = True
        try:
            params = tuple(p for p in sb.parameters() if p.requires_grad)
            bptt_k = int(getattr(self, "deq_bptt_k", 0) or 0)
            return RevDEQPrefixAnchorFunction.apply(
                self.shared_block, x0, z_init, beta, b_bar, K, bptt_k, anchor_depths, *params
            )
        finally:
            _DEQ_SOLVE_ACTIVE = prev_deq_flag
            sb._diag_track_enabled = False
            if stack is not None:
                stack._diag_track_enabled = False

            def _materialize_pairs(track: list) -> list[float]:
                if len(track) == 2 * K:
                    pairs = [0.5 * (track[2*i] + track[2*i+1]) for i in range(K)]
                    return [float(p.item()) if torch.is_tensor(p) else float(p) for p in pairs]
                if len(track) > 0 and len(track) % (2 * K) == 0:
                    per_solve_call = len(track) // (2 * K)
                    vals = []
                    for i in range(K):
                        chunk = track[(2 * i * per_solve_call):(2 * (i + 1) * per_solve_call)]
                        avg = sum(chunk) / float(len(chunk))
                        vals.append(float(avg.item()) if torch.is_tensor(avg) else float(avg))
                    return vals
                return []

            track_owner = stack if stack is not None else sb
            self._attn_gate_iter_last_solve = _materialize_pairs(
                list(getattr(track_owner, "_attn_gate_call_track", []) or []))
            self._router_gate_iter_last_solve = _materialize_pairs(
                list(getattr(track_owner, "_router_gate_call_track", []) or []))
            self._attn_router_gate_iter_last_solve = _materialize_pairs(
                list(getattr(track_owner, "_attn_router_gate_call_track", []) or []))
            self._mlp_router_gate_iter_last_solve = self._attn_router_gate_iter_last_solve

            attn_ew = list(getattr(track_owner, "_attn_expert_weights_per_iter", []) or [])
            if stack is not None:
                self._attn_expert_weights_iter_t = None
            elif len(attn_ew) == 2 * K:
                self._attn_expert_weights_iter_t = (
                    torch.stack(attn_ew, dim=0).reshape(K, 2, -1).mean(dim=1)
                )
            else:
                self._attn_expert_weights_iter_t = None
            self._attn_expert_weights_iter = None
            track_owner._attn_expert_weights_per_iter = []
            track_owner._mlp_expert_weights_per_iter = []
            if track_diag and bool(_ROUTER_DIAGNOSTICS_ACTIVE):
                try:
                    for r in _iter_unique_routers(sb):
                        if hasattr(r, "_materialize_diag_lists"):
                            r._materialize_diag_lists()
                except Exception:
                    pass

    def _run_backbone(self, x: Tensor) -> Tensor:
        x0 = x
        z = x
        self._deq_residuals: list[float] = []
        # Distance travelled during un-reconstructed iterations under TBPTT;
        # equals true reconstruction error only when deq_bptt_k == 0 (full BPTT).
        # See RevDEQFunction.backward for the math; CLAUDE.md §6.1 for the rule.
        self._deq_x0_recon_error = None
        self._deq_recon_error = None
        self._deq_fp_travel = None
        self._deq_z_init_last: Tensor | None = None
        self._deq_k_last = None
        prev_soft_embed = x0
        x0_refined = x0
        self._expert_diversity_loss = None
        prefix_mode = bool(self.training and self.deq_prefix_anchors)

        for r in range(1 + self.num_refinements):
            if r > 0:
                new_soft_embed = self._get_soft_embedding(z)
                alpha = float(getattr(self, "_refine_mix_alpha", 0.5))
                alpha = float(min(max(alpha, 0.0), 1.0))
                x0_refined = alpha * new_soft_embed + (1.0 - alpha) * prev_soft_embed
                prev_soft_embed = x0_refined.detach()
                z = x0_refined
            else:
                x0_refined = x0

            self._deq_k_last = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
            self._deq_z_init_last = z.detach()
            if prefix_mode:
                anchors = _prefix_anchor_depths(self._deq_k_last, self.deq_prefix_anchor_set)
                self._deq_prefix_anchor_depths_last = anchors
                z_stack, z_prev_stack = self._deq_solve_prefix_anchors(x0_refined, z, anchors)
                z = z_stack[-1]
                z_prev = z_prev_stack[-1]
            else:
                self._deq_prefix_anchor_depths_last = ()
                z, z_prev, y_acc, z_acc = self._deq_solve(x0_refined, z)

        # iter172 (2026-05-17, PROMOTED at val_bpb=1.462898 vs iter163's
        # 1.471598 = −8.7 mBPB): recursive nearest-neighbor consistency loss.
        # Each prefix-anchor z_i is paired with its NEXT-DEEPER z_{i+1}.detach():
        #
        #   L_anchor = anchor_coef · mean_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²
        #
        # Zero extra forward compute — target is the next z already produced
        # by the gradient-carrying prefix-anchor forward. Default anchor set
        # (8, 16, 24, 32, 64, 128) gives a non-trivial (z_8, z_16.det) pair at
        # K_sampled=16 (~88 % of steps) while keeping the deepest backward
        # chain count at 6 anchors (iter163-baseline-safe memory).
        #
        # Recursive (not all-to-deepest) avoids iter170's degeneracy: all-to-
        # deepest pressures the iteration map to be FLAT in z (ρ → 0) so all
        # pairs collapse to a pseudo-FP — REFUTED at step 200 (val_bpb=2.07
        # vs baseline 2.00, +65 mBPB). Recursive pairs require only LOCAL
        # contraction over each depth range, allowing ρ ≈ 0.85 (iter163's
        # healthy regime). iter170's K=4 anchor (gap=12 to K=16) further
        # collapsed rho_F → 0 by anchoring an unconverged shallow state to
        # the deep target; iter172's K=8 (gap=8) preserves contraction
        # (final rho_F=0.77 at K=128 vs iter163's 0.92).
        #
        # The `_*_loss_raw` field holds the with-grad tensor that crosses the
        # `_run_backbone` → `forward` boundary; the `_*_loss_t` field is the
        # detached log copy. Same separation as `_ntp_loss_t` / `ntp_loss`.
        self._consistency_anchor_loss_raw = None
        self._consistency_anchor_loss_t = None
        if self.training and prefix_mode and self.multi_k_consistency_anchor_coef > 0.0 and z_stack.shape[0] >= 2:
            anchor_raw = (z_stack[:-1] - z_stack[1:].detach()).pow(2).mean()
            self._consistency_anchor_loss_raw = anchor_raw
            self._consistency_anchor_loss_t = anchor_raw.detach()

        # DEQ diagnostics: keep tensor fields for low-overhead logging, but
        # also maintain legacy float/list fields for existing experiments.
        self._deq_residual_t = None
        self._deq_iter_convergence_t = None
        self._deq_iter_convergence_rel_t = None

        abs_conv_t = None
        if z_prev is not None:
            abs_conv_t = (z - z_prev).float().norm().detach()
            z_norm_t = z.detach().float().norm().clamp_min(1.0).detach()
            self._deq_iter_convergence_t = abs_conv_t
            self._deq_iter_convergence_rel_t = (abs_conv_t / z_norm_t).detach()

        proxy_t = getattr(self.shared_block, "_deq_residual_proxy_t", None)
        if isinstance(proxy_t, torch.Tensor):
            self._deq_residual_t = proxy_t.detach()
        elif isinstance(abs_conv_t, torch.Tensor):
            self._deq_residual_t = abs_conv_t

        # iter 41: removed extra shared_block() call for diagnostics.
        # Use abs_conv_t (last-iteration ‖z_K - z_{K-1}‖) as residual proxy.

        distributed = dist.is_available() and dist.is_initialized()
        is_master = (not distributed) or dist.get_rank() == 0

        # Legacy scalar convergence/residuals (used by experiments/*).
        legacy_diag = (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        # Avoid GPU→CPU sync in the hot training path by only materializing
        # floats when diagnostics are enabled (or in eval mode).
        self._deq_iter_convergence = 0.0 if is_master else None
        self._deq_iter_convergence_rel = 0.0 if is_master else None
        self._deq_residuals = [0.0] if is_master else []
        if is_master and legacy_diag and isinstance(abs_conv_t, torch.Tensor):
            try:
                self._deq_iter_convergence = float(abs_conv_t.float().item())
            except Exception:
                self._deq_iter_convergence = 0.0
        if is_master and legacy_diag and isinstance(self._deq_iter_convergence_rel_t, torch.Tensor):
            try:
                self._deq_iter_convergence_rel = float(self._deq_iter_convergence_rel_t.float().item())
            except Exception:
                self._deq_iter_convergence_rel = 0.0
        if is_master and legacy_diag and isinstance(self._deq_residual_t, torch.Tensor):
            try:
                self._deq_residuals = [float(self._deq_residual_t.float().item())]
            except Exception:
                self._deq_residuals = [0.0]

        # iter 45: store z* and x0 for Lyapunov penalty computation
        # OUTSIDE the compiled forward (see training loop).
        # Calling shared_block with detached inputs inside the compiled graph
        # corrupts the compiled tensor metadata — must be done externally.
        self._lyapunov_z_star = z.detach()
        self._lyapunov_x0 = x0_refined.detach()

        if self.training:
            # Per-token expert-output diversity: gated by training-loop annealer
            # via `_expert_diversity_aux_enabled` (every-N + last-micro-step).
            need_diversity = (
                getattr(self, "_expert_diversity_aux_enabled", False)
                and self._expert_diversity_coef_target > 0.0
            )
            if need_diversity:
                max_tokens = int(self._expert_diversity_max_tokens)
                token_start = int(getattr(self, "_expert_diversity_token_start", 0))
                attn_o, mlp_o, attn_gram_pt, mlp_gram_pt = self.shared_block.ortho_aux(
                    z, x0_refined,
                    max_tokens=max_tokens,
                    token_start=token_start,
                    compute_per_token_gram=True,
                    per_token_gram_kind=str(self.expert_diversity_kind),
                )
                # Diagnostic mean-cosine readout for log time (no sync here).
                self.shared_block.attn._out_ortho_cos_sim_t = attn_o.detach()
                self.shared_block.mlp._out_ortho_cos_sim_t = mlp_o.detach()
                if attn_gram_pt is not None and mlp_gram_pt is not None:
                    self._expert_diversity_loss = 0.5 * (attn_gram_pt + mlp_gram_pt)

        return z_stack if prefix_mode else z

    def _encode(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.use_smear_gate:
            x = self.smear_gate(x, input_ids, bos_id=self.smear_gate_bos_id)
        # Phase 9 iter 71g: learnable embed norm.
        x = self.embed_norm(x)
        x = self._run_backbone(x)
        return self.final_norm(x)

    def _collect_routing_losses(
        self,
        device: torch.device,
        diversity_loss: Tensor | None = None,
        eff_diversity_coef: float = 0.0,
        mos_diversity_loss: Tensor | None = None,
        eff_mos_diversity_coef: float = 0.0,
    ) -> Tensor:
        """Flat router-side regularization assembly:

            router_reg_loss
              = router_pertoken_entropy_coef_eff   × Σ_r H_pertoken(r)
              + router_ema_alive_coef      × Σ_r EMA_alive(r)
              + router_ema_balance_coef    × Σ_r KL_bal(r)
                where KL_bal = KL(U || ST(EMA_r)) when use_reverse_kl_balance
                (iter153 default), else KL(ST(EMA_r) || U). Reverse KL
                gives unbounded gradient on dead experts; forward KL
                under-weights the tail.
              - router_ema_specialization_coef × Σ_r KL(P_token(r) || stopgrad(EMA_r))
              + eff_diversity_coef         × per_token_expert_diversity
              + eff_mos_diversity_coef     × mos_per_token_expert_diversity

        CV-as-loss (router_load_cv_coef / mos_load_cv_coef) was removed
        2026-05-15 — router CV is subsumed by the EMA-anchored balance
        term, and MoS CV is subsumed by the routing softmax + small head
        count (contribution was 0.08% noise at iter163 step 1000). CV is
        still computed for the `_router_cv_loss_t` / `_mos_cv_loss_t`
        diagnostic tensors (and `attn_cv` / `mlp_cv` / `pool_cv` step-log
        fields); the multiplication into total loss is gone.

        `eff_*_coef` already includes the grad_accum_steps compensation for
        last-micro-step gating (multiplied by `_aux_grad_accum_scale`).
        """
        zero = torch.tensor(0.0, device=device)
        sb = _unwrap_compiled_module(self.shared_block)
        routers: list[SoftDenseRouter] = list(_iter_unique_routers(sb))
        cv_sum = zero
        ent_sum = zero
        ent_raw_sum = zero
        ema_sums: dict[str, Tensor] = {name: zero for name, _ in ROUTER_EMA_LOSS_TERMS}
        ent_coef_sum = zero
        ent_count = 0
        for r in routers:
            # cv_sum / mos_cv are diagnostic-only (CV-as-loss removed
            # 2026-05-15) — aggregate as-is; the autograd graph is broken at
            # the `_router_cv_loss_t = cv_sum.detach()` / `_mos_cv_loss_t =
            # mos_cv.detach()` assignments below, which is the single source
            # of truth for the diagnostic boundary.
            cv_sum = cv_sum + getattr(r, "_cv_loss_raw", zero)
            ent_sum = ent_sum + getattr(r, "_pertoken_entropy_loss", zero)
            ent_raw_sum = ent_raw_sum + getattr(r, "_pertoken_entropy_raw_loss", zero)
            for name, _ in ROUTER_EMA_LOSS_TERMS:
                ema_sums[name] = ema_sums[name] + getattr(r, f"_ema_{name}_raw_loss", zero)
            ent_coef_sum = ent_coef_sum + r._entropy_coef.to(device=device, dtype=torch.float32)
            ent_count += 1
        mos_cv = getattr(self.mos_head, "_cv_loss_raw", zero)
        # Router + MoS CV-as-loss removed entirely 2026-05-15: routed-usage
        # balance is fully covered by the EMA-anchored balance loss
        # (router_ema_balance_coef, reverse-KL form since iter153). MoS head
        # balance is maintained for free by the routing softmax + small head
        # count (iter163 step 1000 mos_cv contribution was 0.000242 = 0.08%
        # of router_reg_loss — noise). cv_sum and mos_cv are still computed
        # for the `_router_cv_loss_t` / `_mos_cv_loss_t` diagnostic tensors
        # logged below; only the loss multiplication is removed.
        router_reg_loss = ent_sum
        # Iter145r EMA-anchored terms (`alive`/`balance` add, `specialization`
        # subtracts — KL maximization pushes routing OFF the historical mean).
        # Sign comes from ROUTER_EMA_LOSS_TERMS so adding a future term cannot
        # silently flip direction at a different call site.
        for name, sign in ROUTER_EMA_LOSS_TERMS:
            coef = getattr(self, f"router_ema_{name}_coef")
            router_reg_loss = router_reg_loss + sign * coef * ema_sums[name]
        if diversity_loss is not None and eff_diversity_coef > 0.0:
            router_reg_loss = router_reg_loss + eff_diversity_coef * diversity_loss
        if mos_diversity_loss is not None and eff_mos_diversity_coef > 0.0:
            router_reg_loss = router_reg_loss + eff_mos_diversity_coef * mos_diversity_loss
        self._router_cv_loss_t = cv_sum.detach()
        self._router_pertoken_entropy_loss_t = ent_raw_sum.detach()
        for name, _ in ROUTER_EMA_LOSS_TERMS:
            setattr(self, f"_router_ema_{name}_loss_t", ema_sums[name].detach())
            setattr(
                self,
                f"_router_ema_{name}_coef_eff_t",
                zero.new_tensor(float(getattr(self, f"router_ema_{name}_coef"))).detach(),
            )
        self._mos_cv_loss_t = mos_cv.detach()
        self._expert_diversity_loss_t = (
            diversity_loss.detach() if isinstance(diversity_loss, torch.Tensor) else zero.detach()
        )
        self._mos_diversity_loss_t = (
            mos_diversity_loss.detach() if isinstance(mos_diversity_loss, torch.Tensor) else zero.detach()
        )
        self._router_pertoken_entropy_coef_eff_t = (
            (ent_coef_sum / float(max(ent_count, 1))).detach()
        )
        self._expert_diversity_coef_eff_t = zero.new_tensor(float(eff_diversity_coef)).detach()
        self._mos_diversity_coef_eff_t = zero.new_tensor(float(eff_mos_diversity_coef)).detach()
        self._router_reg_loss_t = router_reg_loss.detach()
        return router_reg_loss

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        if self.training and self.deq_prefix_anchors:
            # x is (A,B,T,D): task loss over all prefix anchors; auxiliary
            # regularizers remain final-endpoint-only for attribution.
            A = int(x.shape[0])
            self.mos_head._diversity_aux_enabled = False
            log_p_ctp, log_p_ntp = self.mos_head(x)
            V = self.tok_emb.num_embeddings
            target_stack = target_ids.unsqueeze(0).expand(A, *target_ids.shape)
            ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_stack.reshape(-1))
            if self.use_ctp:
                input_stack = input_ids.unsqueeze(0).expand(A, *input_ids.shape)
                ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_stack.reshape(-1))
            else:
                ctp_loss = torch.tensor(0.0, device=ntp_loss.device)

            final_x = x[-1]
            self.mos_head._diversity_aux_enabled = bool(
                self.training
                and self.mos_output_diversity_coef > 0.0
                and float(self._expert_diversity_coef_scale) > 0.0
            )
            # Populate final-endpoint MoS CV/diversity diagnostics/losses.
            self.mos_head(final_x)

            aux_gas = float(getattr(self, "_aux_grad_accum_scale", 1.0))
            diversity_loss = torch.tensor(0.0, device=ntp_loss.device)
            eff_diversity_coef = 0.0
            if getattr(self, "_expert_diversity_aux_enabled", False) and isinstance(self._expert_diversity_loss, torch.Tensor):
                diversity_loss = self._expert_diversity_loss.to(device=ntp_loss.device)
                eff_diversity_coef = (
                    float(self._expert_diversity_coef_target)
                    * float(self._expert_diversity_coef_scale)
                    * aux_gas
                )
            mos_diversity_loss = torch.tensor(0.0, device=ntp_loss.device)
            eff_mos_diversity_coef = 0.0
            mos_div_t = getattr(self.mos_head, "_diversity_loss", None)
            if self.mos_output_diversity_coef > 0.0 and isinstance(mos_div_t, torch.Tensor):
                mos_diversity_loss = mos_div_t.to(device=ntp_loss.device)
                eff_mos_diversity_coef = (
                    float(self.mos_output_diversity_coef)
                    * float(self._expert_diversity_coef_scale)
                )
            router_reg_loss = self._collect_routing_losses(
                ntp_loss.device,
                diversity_loss, eff_diversity_coef,
                mos_diversity_loss, eff_mos_diversity_coef,
            )
            self._ntp_loss_t = ntp_loss.detach()
            self._ctp_loss_t = ctp_loss.detach()
            self._ntp_loss = 0.0
            self._ctp_loss = 0.0
            ctp_weight = float(getattr(self, "ctp_weight", 0.0)) if self.use_ctp else 0.0
            total = ntp_loss + ctp_weight * ctp_loss + router_reg_loss
            if isinstance(self._consistency_anchor_loss_raw, torch.Tensor) and self.multi_k_consistency_anchor_coef > 0.0:
                total = total + self.multi_k_consistency_anchor_coef * self._consistency_anchor_loss_raw
            return total

        self.mos_head._diversity_aux_enabled = bool(
            self.training
            and self.mos_output_diversity_coef > 0.0
            and float(self._expert_diversity_coef_scale) > 0.0
        )
        log_p_ctp, log_p_ntp = self.mos_head(x)
        V = self.tok_emb.num_embeddings
        ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_ids.reshape(-1))
        if self.use_ctp:
            ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_ids.reshape(-1))
        else:
            ctp_loss = torch.tensor(0.0, device=ntp_loss.device)
        # Per-token expert-output diversity: gated to last-micro-step + every-N
        # by the training loop. Multiplied by `_aux_grad_accum_scale` so the
        # documented coefficient holds despite the global `loss / grad_accum_steps`
        # division before backward.
        aux_gas = float(getattr(self, "_aux_grad_accum_scale", 1.0))
        diversity_loss = torch.tensor(0.0, device=ntp_loss.device)
        eff_diversity_coef = 0.0
        if (self.training and getattr(self, "_expert_diversity_aux_enabled", False)
                and isinstance(self._expert_diversity_loss, torch.Tensor)):
            diversity_loss = self._expert_diversity_loss.to(device=ntp_loss.device)
            eff_diversity_coef = (
                float(self._expert_diversity_coef_target)
                * float(self._expert_diversity_coef_scale)
                * aux_gas
            )
        # MoS per-token diversity (mirrors expert side; default off via coef=0).
        mos_diversity_loss = torch.tensor(0.0, device=ntp_loss.device)
        eff_mos_diversity_coef = 0.0
        mos_div_t = getattr(self.mos_head, "_diversity_loss", None)
        if (self.training and self.mos_output_diversity_coef > 0.0
                and isinstance(mos_div_t, torch.Tensor)):
            mos_diversity_loss = mos_div_t.to(device=ntp_loss.device)
            # Scaled by warmup (annealer writes _expert_diversity_coef_scale).
            # Computed every micro-step (coef=0 gates the work upstream), so no
            # grad-accum compensation is needed here.
            eff_mos_diversity_coef = (
                float(self.mos_output_diversity_coef)
                * float(self._expert_diversity_coef_scale)
            )
        router_reg_loss = self._collect_routing_losses(
            ntp_loss.device,
            diversity_loss, eff_diversity_coef,
            mos_diversity_loss, eff_mos_diversity_coef,
        )
        self._ntp_loss_t = ntp_loss.detach()
        self._ctp_loss_t = ctp_loss.detach()
        # Back-compat scalar fields used by experiments/*.
        if (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE):
            try:
                self._ntp_loss = float(self._ntp_loss_t.float().item())
                self._ctp_loss = float(self._ctp_loss_t.float().item())
            except Exception:
                self._ntp_loss = float(ntp_loss.detach().float().mean().item())
                self._ctp_loss = float(ctp_loss.detach().float().mean().item())
        else:
            self._ntp_loss = 0.0
            self._ctp_loss = 0.0
        ctp_weight = float(getattr(self, "ctp_weight", 0.0)) if self.use_ctp else 0.0
        # iter163c (2026-05-15): consistency anchor requires prefix_mode, which
        # is only active in the prefix-anchor branch of `_run_backbone`. In the
        # final-endpoint mode reached here, there is no z_stack to compute
        # pairwise differences from, so the anchor term is silently skipped.
        total = ntp_loss + ctp_weight * ctp_loss + router_reg_loss
        # Lyapunov + denoising auxiliaries are added in the training loop.
        return total

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        self.mos_head._diversity_aux_enabled = False
        _, log_p_ntp = self.mos_head(x)
        return log_p_ntp


# ---------------------------------------------------------------------------
# SLIDING WINDOW VALIDATION
# ---------------------------------------------------------------------------

def sliding_window_validation(args, base_model, rank, world_size, device, val_tokens,
                              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                              stride: int, batch_seqs: int = 32) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs = base_model.forward_logits(x_batch)
            nll = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)).float(),
                             y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte

# Backwards-compatible alias
eval_val_sliding = sliding_window_validation


def _flatten_param_groups(groups: list[dict[str, object]]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for group in groups:
        for p in group.get("params", []):
            if isinstance(p, nn.Parameter):
                params.append(p)
    return params


def _assert_optimizer_param_coverage(model: nn.Module,
                                     named_groups: list[tuple[str, list[nn.Parameter]]]) -> None:
    seen: dict[int, str] = {}
    duplicates: list[str] = []
    for group_name, params in named_groups:
        for p in params:
            if not isinstance(p, nn.Parameter) or not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                duplicates.append(f"{group_name} duplicates {seen[pid]}")
            else:
                seen[pid] = group_name
    missing = [
        f"{name}{tuple(p.shape)}"
        for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in seen
    ]
    if missing or duplicates:
        msg_parts = []
        if missing:
            msg_parts.append("missing trainable params: " + ", ".join(missing))
        if duplicates:
            msg_parts.append("duplicate trainable params: " + ", ".join(duplicates))
        raise RuntimeError("Optimizer parameter coverage error: " + "; ".join(msg_parts))


@dataclass(frozen=True)
class OptimizerParamLists:
    """Named return for _build_optimizer_param_lists — eliminates positional
    unpack drift between the producer, consumers, and tests."""
    tok: list[dict[str, object]]
    matrix: list[nn.Parameter]
    scalar: list[nn.Parameter]
    parcae: list[nn.Parameter]
    entmax_blend: list[nn.Parameter]


def _build_optimizer_param_lists(base_model: nn.Module, args) -> OptimizerParamLists:
    sb = _unwrap_compiled_module(base_model.shared_block)
    block_named_params = list(sb.named_parameters())
    # All ndim >= 2 block params (including active router weights/prototypes) -> Muon;
    # all ndim < 2 / control params -> AdamW scalar.
    matrix_params = [p for name, p in block_named_params
                     if p.ndim >= 2 and not any(pat in name for pat in CONTROL_TENSOR_PATTERNS)]
    # iter 117 v2 (H87): carve out `_entmax_blend_logit` to a slow-LR group
    # (entmax_blend_lr=0.002, 10× smaller than scalar_lr) so the blend drift
    # rate is bounded once anneal ramps up. Mirrors the parcae_lr precedent
    # (system-dynamics-sensitive params get a slower LR than other scalars).
    entmax_blend_params = [p for name, p in block_named_params
                            if "_entmax_blend_logit" in name and p.requires_grad]
    scalar_params = [p for name, p in block_named_params
                     if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_PATTERNS))
                     and "_entmax_blend_logit" not in name]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params: list[dict[str, object]] = [
        {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
    ]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        scalar_params.append(base_model.bigram.scale)
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
        if base_model.bigram.proj_norm is not None:
            scalar_params.append(base_model.bigram.proj_norm.weight)

    mos = base_model.mos_head
    mos_params = [mos.A_ntp_shared, mos.A_ntp, mos.B_NTP,
                  mos.ntp_rank_norm_weight, mos.gate_ntp_norm_weight,
                  mos.ntp_a_norm_weight,
                  mos.gate_ntp.weight, mos.gate_ntp.bias,
                  ]
    if mos.use_ctp:
        mos_params += [mos.A_ctp_shared, mos.A_ctp, mos.B_denoise,
                       mos.ctp_rank_norm_weight, mos.gate_ctp_norm_weight,
                       mos.ctp_a_norm_weight,
                       mos.gate_ctp.weight, mos.gate_ctp.bias]
    scalar_params.extend(mos_params)
    scalar_params.append(base_model.final_norm.weight)
    scalar_params.append(base_model.embed_norm.weight)
    # iter 129 / H99: SmearGate lives at GPT level (single module shared across
    # the entire backbone). Route its narrow gate + lambda through AdamW scalar
    # handling rather than Muon.
    if getattr(base_model, "use_smear_gate", False):
        scalar_params.extend(p for p in base_model.smear_gate.parameters() if p.requires_grad)

    parcae_params: list[nn.Parameter] = []
    if getattr(base_model, "use_parcae", False):
        parcae_params = [
            base_model.parcae_raw_a,
            base_model.parcae_raw_delta,
            base_model.parcae_raw_b,   # iter 66b: independent Parcae input gain
        ]

    _assert_optimizer_param_coverage(base_model, [
        ("token", _flatten_param_groups(tok_params)),
        ("matrix", matrix_params),
        ("scalar", scalar_params),
        ("parcae", parcae_params),
        ("entmax_blend", entmax_blend_params),
    ])
    return OptimizerParamLists(
        tok=tok_params,
        matrix=matrix_params,
        scalar=scalar_params,
        parcae=parcae_params,
        entmax_blend=entmax_blend_params,
    )


def _prescribe_failure_fix(failure: str) -> dict:
    """Map a post-int6 diagnostic failure string to a fix prescription.

    Diagnostics identify root causes; prescriptions prefer reusable mechanisms
    over symptom-specific losses:
      - router_bias_update          — slow generic usage-prior controller
      - expert-bank geometry        — preferred expert-collapse fix
      - transition parameterization — preferred contraction fix
      - multi-K consistency loss    — principled FP-convergence mechanism
    Router/MoS CV-as-loss was removed 2026-05-15 (EMA-anchored balance fully
    covers routed-usage balance, and MoS head balance is maintained for free
    by the routing softmax + small head count); the CV-floor fallback is no
    longer a valid prescription lever.
    """
    low = failure.lower()
    first_token = low.split("=", 1)[0].split()[0] if low else ""

    if "min_share" in first_token:
        if first_token.startswith("mos_"):
            return {
                "failure": failure,
                "category": "mos_router_collapse_advisory",
                "hypothesis": "MoS CV-as-loss removed 2026-05-15 (subsumed by routing softmax + small head count, contribution was 0.08% of router_reg_loss at iter163 step 1000). FAILED Principled test once removed: there is no remaining loss-side lever for MoS head balance. Treat as informational only; if MoS heads genuinely collapse in a future iter, the principled fix is a new architectural mechanism (e.g. MoS-side EMA balance, hard head dispatch outside RevDEQ), not re-adding a per-symptom CV penalty.",
                "fix": "Treat as informational only — no actionable config change. The CV-floor lever was removed for the same reason as router_load_cv_coef: subsumed by other mechanisms and contributing only noise.",
                "config_change": {},
            }
        return {
            "failure": failure,
            "category": "router_collapse_advisory",
            "hypothesis": "EMPIRICALLY REFUTED 2026-05-15 by 3-iter closure (iter158/162/165): "
                          "pushing on attn_min_share via router_bias_update or doubled "
                          "regularizers does NOT move BPB at iter152's operating point. "
                          "FAILED Principled test (per CLAUDE.md most-principled-simplest-general "
                          "directive): attn_min_share is a symptom correlated with collapse, "
                          "not a root-cause invariant for BPB. The fair-share threshold pressures "
                          "the wrong objective.",
            "fix": ("Treat as informational only. iter158 (reverse-KL alone, Δ +0.003), "
                    "iter162 (entropy-only push, killed mid-run), iter165 (bundled "
                    "router_bias_update + doubled coefs, Δ +0.013) all confirmed "
                    "regression. The 0.0375 fair-share threshold is an arbitrary "
                    "diagnostic, not a correctable BPB defect. A fundamentally new "
                    "routing-balance mechanism (e.g. hard dispatch outside RevDEQ, "
                    "dynamic expert pruning + respawn) is required to re-open this "
                    "prescription class."),
            "config_change": {},
        }
    if first_token.startswith("mos_") and "ortho" in first_token:
        return {
            "failure": failure,
            "category": "mos_head_collapse",
            "hypothesis": "MoS expert-bank geometry permits parallel MoS directions",
            "fix": ("Prefer a shared MoS expert-bank geometry constraint "
                    "(normalized/orthogonalized low-rank state vectors or retraction). "
                    "mos_output_diversity_coef=0.05 is a temporary ablation only."),
            "config_change": {"needs_mos_output_geometry_constraint": True,
                              "mos_output_diversity_coef": 0.05},
        }
    if "ortho" in first_token:
        cur = float(Hyperparameters.expert_output_diversity_coef)
        return {
            "failure": failure,
            "category": "expert_collapse",
            "hypothesis": "Expert-bank geometry permits parallel expert directions",
            "fix": ("Prefer a shared expert-bank geometry constraint "
                    "(normalized/orthogonalized low-rank deltas or retraction). "
                    f"Use expert_output_diversity_coef ×1.5 (e.g. {cur:g}→{cur * 1.5:g}) "
                    "only as a temporary ablation."),
            "config_change": {"needs_expert_bank_geometry_constraint": True,
                              "expert_output_diversity_coef_mult": 1.5},
        }
    if low.startswith("k-sweep"):
        return {
            "failure": failure,
            "category": "fp_quality_loss",
            "hypothesis": "H12 VERIFIED (wider K jitter → better FP)",
            "fix": "Gross FP-quality loss at deep K (Δ > 0.1).  Try widening K jitter: increase "
                   "deq_k_max by 4.  Note: minor non-monotonicity (K=64 vs K=32 ±0.01) "
                   "is no longer gated; it's within finite-K noise.",
            "config_change": {"deq_k_max_delta": 4},
        }
    # `first_token` is lowercased from `failure.lower()` above, so comparisons
    # below are all lowercase.  iter155 (corrected) split the contraction
    # metric into three:
    # rho_F (spectral radius on the actual two-state Parcae cycle F) is
    # the principled FP-convergence gate per Hartman-Grobman. lip_ub_T/S/F
    # operator-norm proxies and fp_bound (Banach error bound) were removed
    # 2026-05-15 as over-restrictive. Legacy lip_ub_* / fp_bound failure
    # strings from pre-2026-05-15 logs are routed to advisory.
    if first_token in ("lip_ub_f", "lip_ub", "lip_ub_t", "lip_ub_s") or first_token.startswith("fp_bound"):
        return {
            "failure": failure,
            "category": "operator_norm_advisory",
            "hypothesis": "Operator-norm proxy (lip_ub_*) and fp_bound were removed 2026-05-15 — over-restrictive for non-symmetric J_F. Check rho_F (spectral radius) and iter_conv_rel for the principled FP-convergence signals.",
            "fix": ("Legacy operator-norm / Banach-bound prescription. The "
                    "necessary-and-sufficient condition for asymptotic local "
                    "contraction is rho(J_F) < 1 (spectral radius), NOT "
                    "sigma_max(J_F) < 1. For non-symmetric J_F the gap can be "
                    "huge (iter152 has sigma_max ~ 17 with iter_conv_rel ~ 0.02 "
                    "-- clearly converging). Check rho_F and iter_conv_rel "
                    "diagnostics; do NOT enable lyapunov_coef (refuted by iter155). "
                    "If rho_F >= 1 or iter_conv_rel >= 0.05, follow those prescriptions."),
            "config_change": {},
        }
    if first_token == "rho_f":
        # rho_F = |lambda_max(J_F)| via straight power iteration on J_F.
        # Necessary AND sufficient for asymptotic local FP convergence
        # (Hartman-Grobman). Architecture-agnostic.
        return {
            "failure": failure,
            "category": "fp_convergence_failed",
            "hypothesis": "Spectral radius rho(J_F) >= 1 -- the iteration is locally non-contractive and the fixed point is not asymptotically attractive.",
            "fix": ("rho(J_F) >= 1 means asymptotic local convergence is genuinely "
                    "threatened (Hartman--Grobman: rho < 1 is necessary AND "
                    "sufficient).  Soft Lyapunov penalties were refuted by iter155 "
                    "as ineffective at constraining spectral properties.  "
                    "Formal-tier mechanism required: spectral normalization on "
                    "T_theta's component layers (caps sigma_max per-layer, bounds "
                    "rho via composition), bounded-Lipschitz block parameterization, "
                    "or learned `c*Delta` gain with hard projection.  Architecture "
                    "constraint: any new iteration mechanism must satisfy "
                    "rho(J) < 1 for asymptotic local contraction."),
            "config_change": {"needs_formal_tier_contraction": True},
        }
    if first_token.startswith("iter_conv_rel"):
        # Tier 1 (2026-05-13): iter_conv_rel is the EMPIRICAL FP
        # convergence signal -- the operational evidence that
        # rho(J_F) < 1 holds.  Promoted from advisory to gate-relevant
        # because it is the actually-required condition for asymptotic
        # local contraction (the architecture-agnostic principled gate).
        # Prescription is solver-tuning rather than spectral-control:
        # if the iteration is empirically not converging, we adjust K
        # depth or solver knobs, NOT add Lyapunov penalties (refuted).
        return {
            "failure": failure,
            "category": "fp_convergence_empirical_failed",
            "hypothesis": "Empirical FP convergence signal weak: ||F^K(z) - F^{K-1}(z)|| / ||F^{K-1}(z)|| does not shrink with K.  This is the actually-required condition for asymptotic local contraction (architecture-agnostic).",
            "fix": ("iter_conv_rel large at deepest K means the solver isn't reaching "
                    "a fixed point in practice.  First-order remediations: increase "
                    "weight_decay 1.5x (regularize transition Jacobian), increase "
                    "deq_k_max by 4 (more solver iterations); under Parcae the scalar "
                    "deq_beta is fallback-only.  Do NOT enable lyapunov_coef -- "
                    "iter155 confirmed soft penalties on operator-norm proxies are "
                    "not principled for asymptotic convergence.  If iter_conv_rel "
                    "remains stuck after solver tuning, escalate to formal-tier "
                    "spectral control (see rho_F prescription)."),
            "config_change": {"weight_decay_mult": 1.5, "deq_k_max_delta": 4},
        }
    if first_token.startswith("deq_recon_err"):
        return {
            "failure": failure,
            "category": "reversibility_broken",
            "hypothesis": "RevDEQ reversibility requires f(z, x0, W) be deterministic and solver in stable contraction region",
            "fix": "1) Check for any random/non-deterministic op in the block (quant-noise, dropout etc. "
                   "— see H15 REFUTED).  2) Check Parcae A_bar/reversibility-floor diagnostics. "
                   "3) Increase weight_decay 1.5× to shrink Jacobian spectral norm.",
            "config_change": {"weight_decay_mult": 1.5},
        }
    return {
        "failure": failure,
        "category": "unknown",
        "hypothesis": "none",
        "fix": "Manual analysis required — check experiments/docs/hypotheses.md for related observations.",
        "config_change": {},
    }


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------


def _slice_for_fp_probe(z_star: Tensor, x0_lyap: Tensor, B_probe: int) -> tuple[Tensor, Tensor]:
    """Slice (z_star, x0_lyap) to the first ``B_probe`` sequences along the
    batch dim. The Hutchinson estimator and finite-direction Lipschitz probe
    are both unbiased at any batch size — for an LM with translation-invariant
    attention, B=1 yields a representative spectral estimate. Slicing bounds
    the SharedBlock-forward activation graph from ~32 GiB (B=val_micro≈32)
    down to ~1 GiB (B=1), which is what unblocks the probes on the 44 GiB
    dev cap.

    Returns the original tensors if ``B_probe <= 0`` or already smaller.
    """
    if B_probe <= 0 or z_star.shape[0] <= B_probe:
        return z_star, x0_lyap
    return z_star[:B_probe].contiguous(), x0_lyap[:B_probe].contiguous()


def _parcae_cycle_F(
    y: Tensor,
    z: Tensor,
    x0: Tensor,
    b_bar: Tensor | None,
    a_bar_d: Tensor,
    sb,
) -> tuple[Tensor, Tensor]:
    """Single application of the actual two-state Parcae cycle map F.

    ``F(y, z) = (Ā·y + β·T_θ(z),  Ā·z + β·T_θ(Ā·y + β·T_θ(z)))``
    with ``β = 1 − Ā`` and per-dim Ā stored in ``a_bar_d`` (broadcast-
    compatible with ``y`` / ``z``, e.g. shape ``(1, 1, D)``).

    This is the iteration map probed by ``rho_F`` (the principled spectral
    gate per Hartman--Grobman), measured by ``fp_residual_F``, and used by
    the iter163 multi-K consistency loss extension term — the single source
    of truth for the F-cycle algebra so the probe / residual / training
    paths cannot silently diverge. The operator-norm probes (``lip_ub_F``,
    ``lip_ub_S``, ``lip_ub_T``) and Banach error bound (``fp_bound``) were
    removed 2026-05-15 as over-restrictive.

    Returns ``(y_new, z_new)`` with the same shape and dtype as ``y, z``.
    Two ``sb`` calls per invocation (one for ``T(z)``, one for ``T(y_new)``).
    """
    t_z = sb(z, x0, b_bar)
    y_new = a_bar_d * y + (1.0 - a_bar_d) * t_z
    t_y_new = sb(y_new, x0, b_bar)
    z_new = a_bar_d * z + (1.0 - a_bar_d) * t_y_new
    return y_new, z_new


def _run_spectral_radius_power(
    z_star: Tensor,
    x0_lyap: Tensor,
    b_bar_d: Tensor | None,
    sb,
    target_dtype: torch.dtype,
    n_iters: int,
    ctx_factory,
    map_kind: str = "F",
    a_bar_d: Tensor | None = None,
) -> float | None:
    """Power iteration on J directly for ``|lambda_max(J)|`` (spectral radius).

    For the iterated map M, ``rho(J_M) < 1`` is the necessary AND sufficient
    condition for asymptotic local fixed-point convergence
    (Hartman--Grobman).  This probe returns an estimate of
    ``rho(J_M) = |lambda_max(J_M)|`` via straight power iteration on ``J``:
    ``v_{k+1} = J v_k / ||J v_k||`` converges to the dominant eigendirection,
    and ``||J v|| / ||v||`` at convergence equals ``|lambda_max|`` (real case;
    for complex eigenvalues the per-step ratio oscillates around
    ``|lambda_max|`` and we average the last few iterations).

    Compare with :func:`_run_spectral_norm_power` which iterates
    ``J^T J`` and returns ``sigma_max(J)`` (operator norm, upper bound on
    ``|lambda_max|`` but not equivalent).  For non-symmetric ``J`` (the
    Parcae cycle Jacobian is non-symmetric in general), ``rho`` can be
    much smaller than ``sigma_max`` -- pursuing ``sigma_max < 1`` (the
    Lyapunov-on-F target in iter155) is therefore over-restrictive and
    not principled for asymptotic convergence.  This probe targets the
    actually-required quantity.

    The framework is **architecture-agnostic**: ``rho(J_F) < 1`` is the
    sufficient-and-necessary condition for asymptotic local contraction
    of any iteration map ``M``, not specifically the Parcae cycle.  Any
    future iteration mechanism (alternative DEQ solvers, refinement
    loops, etc.) must satisfy the same condition.

    ``map_kind`` selects the Jacobian probed (same enum as
    :func:`_run_spectral_norm_power`); typically used with ``"F"`` to
    measure the actual two-state Parcae cycle.
    """
    if map_kind in ("S", "F") and a_bar_d is None:
        raise ValueError(f"map_kind={map_kind!r} requires a_bar_d (per-dim Ā) for the Parcae blend")

    torch.cuda.empty_cache()
    state_factor = 2 if map_kind == "F" else 1
    free_b, _ = torch.cuda.mem_get_info(z_star.device)
    need_b = int(z_star.numel() * z_star.element_size() * 24 * state_factor)
    if free_b < int(need_b * 1.25):
        print(
            f"rho_{map_kind} skip: oom_pred need={need_b/1e9:.2f}GiB "
            f"free={free_b/1e9:.2f}GiB B_probe={z_star.shape[0]}",
            flush=True,
        )
        return None

    a_bar_t = a_bar_d.to(device=z_star.device, dtype=target_dtype) if a_bar_d is not None else None

    if map_kind == "F":
        D = z_star.shape[-1]

        def _apply_map(state: Tensor) -> Tensor:
            y, z = torch.split(state, D, dim=-1)
            y_new, z_new = _parcae_cycle_F(y, z, x0_lyap, b_bar_d, a_bar_t, sb)
            return torch.cat([y_new, z_new], dim=-1)

        seed_state = torch.cat([z_star, z_star], dim=-1).contiguous()
    else:
        def _apply_map(z_arg: Tensor) -> Tensor:
            out_T = sb(z_arg, x0_lyap, b_bar_d)
            if map_kind == "S":
                return a_bar_t * z_arg + (1.0 - a_bar_t) * out_T
            return out_T

        seed_state = z_star

    v = torch.randn_like(seed_state, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-8)

    def _jvp(v_float: Tensor) -> Tensor:
        z_b = seed_state.detach().clone().requires_grad_(True)
        v_b = v_float.to(device=z_b.device, dtype=z_b.dtype)
        with ctx_factory(), torch.enable_grad(), torch.autocast(device_type="cuda", dtype=target_dtype):
            _, jv = torch.autograd.functional.jvp(
                _apply_map, (z_b,), (v_b,), create_graph=False, strict=False
            )
        del z_b, v_b
        return jv.detach()

    # Power iteration on J directly: v_{k+1} = Jv_k / ||Jv_k||.
    # The per-step ratio ||Jv||/||v|| at convergence equals |lambda_max|.
    # For complex dominant eigenvalues the ratio oscillates; we average
    # the last `avg_window` iterations for robustness.
    n = max(2, int(n_iters))
    avg_window = min(4, n)
    ratios: list[float] = []
    for k in range(n):
        v_norm = v.float().norm().clamp(min=1e-8)
        jv = _jvp(v)
        jv_norm = jv.float().norm()
        ratio = float((jv_norm / v_norm).item())
        if k >= n - avg_window:
            ratios.append(ratio)
        v = (jv.float() / jv_norm.clamp(min=1e-8)).detach()
        del jv
        torch.cuda.empty_cache()

    # Free the post-loop v before returning (mirrors per-iter discipline;
    # otherwise v lives until frame teardown — ~12 MiB at default shapes).
    del v
    torch.cuda.empty_cache()
    # Geometric mean of the last `avg_window` ratios is the |lambda_max|
    # estimate robust to oscillation from complex eigenvalues.
    if not ratios:
        return None
    log_mean = sum(math.log(max(r, 1e-12)) for r in ratios) / len(ratios)
    return float(math.exp(log_mean))


def _rho_F_at_saved_fp(
    base_m,
    n_iters: int = 8,
    B_probe: int = 1,
    log_label: str = "rho_F",
    n_seeds: int = 4,
) -> float | None:
    """Spectral-radius estimate ``rho(J_F)`` at the model's saved DEQ FP.

    Returns ``|lambda_max(J_F)|`` (necessary AND sufficient for asymptotic
    local FP convergence: ``rho(J_F) < 1`` iff the fixed point is locally
    attractive). Architecture-agnostic: same condition applies to any
    iteration map M (alternative DEQ solvers, refinement loops, recurrent
    layers — all subject to the same gate).

    iter168 (2026-05-16): multi-seed power iteration. Single-seed power
    iteration on non-symmetric J_F can give noisy estimates when the
    starting vector has small projection onto the dominant eigenspace
    (iter163 K-sweep observed rho_F = 1.04 at K=24, 1.15 at K=32 with
    K=128 = 0.92 — single-seed estimator artifacts). Running power
    iteration with ``n_seeds`` independent random starts and returning
    the geometric median is robust to this noise (the dominant eigenvalue
    is invariant; only the per-seed convergence speed depends on init).

    Returns ``None`` if no saved FP is available (e.g., before any forward
    pass) or the OOM predictor blocks the probe.
    """
    prepared = _prepare_saved_fp_probe(base_m, B_probe)
    if prepared is None:
        return None
    z_star, x0_lyap, b_bar_d, a_bar_d, sb, target_dtype, ctx_factory = prepared
    # Parcae-disabled fallback: when there is no Ā, the iteration map IS
    # the transition map T (the solver does z_{k+1} = T(z_k) directly),
    # so rho(J_T) is the principled spectral-radius value.
    map_kind = "F" if a_bar_d is not None else "T"
    n_seeds = max(1, int(n_seeds))
    estimates: list[float] = []
    for _ in range(n_seeds):
        try:
            est = _run_spectral_radius_power(
                z_star, x0_lyap, b_bar_d, sb, target_dtype,
                n_iters=n_iters,
                ctx_factory=ctx_factory,
                map_kind=map_kind,
                a_bar_d=a_bar_d,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{log_label} skip: oom during power iteration", flush=True)
            return None
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"{log_label} skip: cuda oom during power iteration", flush=True)
                return None
            raise
        if est is not None:
            estimates.append(est)
    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]  # median; geo-median ≈ median for log-normal noise


def _run_spectral_norm_power(
    z_star: Tensor,
    x0_lyap: Tensor,
    b_bar_d: Tensor | None,
    sb,
    target_dtype: torch.dtype,
    n_iters: int,
    ctx_factory,
    a_bar_d: Tensor | None,
) -> float | None:
    """Operator-norm estimate ``sigma_max(J_F)`` via power iteration on ``J^T J``.

    iter168 (2026-05-16): restored as DIAGNOSTIC ONLY after the 2026-05-15
    over-aggressive removal. ``sigma_max(J_F) < 1`` is sufficient (but NOT
    necessary) for asymptotic convergence — refuted as a GATE by iter152's
    empirical convergence with sigma_max ≈ 17 and rho ≈ 0.85. However, the
    quantity itself carries useful operational information that is NOT
    captured by rho_F:

      - Per-step monotone contraction bound: ||F(z) − F(z*)|| <= sigma_max · ||z − z*||
      - Robustness to input/weight perturbations (operator-norm Lipschitz)
      - Basin of attraction size lower bound (smaller sigma_max → larger basin)

    These properties become operationally critical when training-time
    perturbations are introduced (iter161-QAT-late STE fake-quant on
    weights), so the diagnostic is restored as a non-gate, non-penalty,
    non-prescription signal emitted in fast-val + K-sweep.

    Power iteration: v_{k+1} = J^T J v_k / ||J^T J v_k|| converges to the
    dominant right-singular vector; ``||J v||/||v||`` at convergence equals
    ``sigma_max(J)``. Implementation mirrors :func:`_run_spectral_radius_power`
    on the same two-state Parcae cycle ``F``.
    """
    if a_bar_d is None:
        # Parcae-disabled fallback: probe σ_max(J_T) on the transition map.
        # In this branch ``F = T`` (no Ā damping), so naming is consistent.
        pass

    torch.cuda.empty_cache()
    state_factor = 2 if a_bar_d is not None else 1
    free_b, _ = torch.cuda.mem_get_info(z_star.device)
    # ~2x the rho_F probe budget because JVP + VJP both fire per power-iter step.
    need_b = int(z_star.numel() * z_star.element_size() * 32 * state_factor)
    if free_b < int(need_b * 1.25):
        print(
            f"sigma_max_F skip: oom_pred need={need_b/1e9:.2f}GiB "
            f"free={free_b/1e9:.2f}GiB B_probe={z_star.shape[0]}",
            flush=True,
        )
        return None

    a_bar_t = a_bar_d.to(device=z_star.device, dtype=target_dtype) if a_bar_d is not None else None

    if a_bar_d is not None:
        D = z_star.shape[-1]

        def _apply_map(state: Tensor) -> Tensor:
            y, z = torch.split(state, D, dim=-1)
            y_new, z_new = _parcae_cycle_F(y, z, x0_lyap, b_bar_d, a_bar_t, sb)
            return torch.cat([y_new, z_new], dim=-1)

        seed_state = torch.cat([z_star, z_star], dim=-1).contiguous()
    else:
        def _apply_map(z_arg: Tensor) -> Tensor:
            return sb(z_arg, x0_lyap, b_bar_d)

        seed_state = z_star

    v = torch.randn_like(seed_state, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-8)

    def _jvp(v_float: Tensor) -> Tensor:
        z_b = seed_state.detach().clone().requires_grad_(True)
        v_b = v_float.to(device=z_b.device, dtype=z_b.dtype)
        with ctx_factory(), torch.enable_grad(), torch.autocast(device_type="cuda", dtype=target_dtype):
            _, jv = torch.autograd.functional.jvp(
                _apply_map, (z_b,), (v_b,), create_graph=False, strict=False
            )
        del z_b, v_b
        return jv.detach()

    def _vjp(v_float: Tensor) -> Tensor:
        # J^T v via reverse-mode AD on _apply_map at seed_state.
        z_b = seed_state.detach().clone().requires_grad_(True)
        v_b = v_float.to(device=z_b.device, dtype=z_b.dtype)
        with ctx_factory(), torch.enable_grad(), torch.autocast(device_type="cuda", dtype=target_dtype):
            out = _apply_map(z_b)
            (jtv,) = torch.autograd.grad(
                outputs=out,
                inputs=z_b,
                grad_outputs=v_b,
                create_graph=False,
                retain_graph=False,
            )
        del z_b, v_b, out
        return jtv.detach()

    # Power iteration on J^T J: v <- J^T(Jv); sqrt(<v_old, J^T J v_old>) ≈ sigma_max.
    n = max(2, int(n_iters))
    avg_window = min(4, n)
    ratios: list[float] = []
    for k in range(n):
        v_norm = v.float().norm().clamp(min=1e-8)
        jv = _jvp(v)
        jv_norm = jv.float().norm()
        ratio = float((jv_norm / v_norm).item())  # ||Jv||/||v|| → sigma_max
        if k >= n - avg_window:
            ratios.append(ratio)
        jtjv = _vjp(jv.float())
        jtjv_norm = jtjv.float().norm().clamp(min=1e-8)
        v = (jtjv.float() / jtjv_norm).detach()
        del jv, jtjv
        torch.cuda.empty_cache()

    del v
    torch.cuda.empty_cache()
    if not ratios:
        return None
    return float(sum(ratios) / len(ratios))


def _sigma_max_F_at_saved_fp(
    base_m,
    n_iters: int = 8,
    B_probe: int = 1,
    log_label: str = "sigma_max_F",
) -> float | None:
    """Operator-norm estimate sigma_max(J_F) at the saved DEQ FP.

    DIAGNOSTIC ONLY (iter168, 2026-05-16). NOT a gate, NOT a penalty,
    NOT a prescription input. Restored after the 2026-05-15 over-aggressive
    removal because the operator norm carries operationally-relevant
    information (per-step contraction bound, perturbation robustness,
    basin size) that is not captured by rho_F alone. See
    :func:`_run_spectral_norm_power` for full rationale.

    Returns ``None`` if no saved FP is available or OOM blocks the probe.
    """
    prepared = _prepare_saved_fp_probe(base_m, B_probe)
    if prepared is None:
        return None
    z_star, x0_lyap, b_bar_d, a_bar_d, sb, target_dtype, ctx_factory = prepared
    try:
        return _run_spectral_norm_power(
            z_star, x0_lyap, b_bar_d, sb, target_dtype,
            n_iters=n_iters,
            ctx_factory=ctx_factory,
            a_bar_d=a_bar_d,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"{log_label} skip: oom during power iteration", flush=True)
        return None
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"{log_label} skip: cuda oom during power iteration", flush=True)
            return None
        raise


def _fp_probe_context_factory():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel_impl

        def _grad_safe_sdpa():
            return _sdpa_kernel_impl(
                [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )
        return _grad_safe_sdpa
    except Exception:
        from contextlib import nullcontext
        return nullcontext


def _prepare_saved_fp_probe(base_m, B_probe: int = 1):
    """Slice + dtype-prep the saved FP tensors and the Parcae per-dim
    coefficients so probes can run map-agnostically.

    Returns ``(z_star, x0_lyap, b_bar_d, a_bar_d, sb_call, target_dtype, ctx_factory)``.
    ``a_bar_d`` is the per-dim Ā tensor (broadcast as ``(1, 1, D)``) or ``None``
    when Parcae is disabled — callers selecting ``map_kind="S"`` must check.
    """
    z_star = getattr(base_m, "_lyapunov_z_star", None)
    x0_lyap = getattr(base_m, "_lyapunov_x0", None)
    if z_star is None or x0_lyap is None:
        return None
    sb = _unwrap_compiled_module(base_m.shared_block)
    sb_call = type(sb).forward.__get__(sb, type(sb)) if isinstance(sb, nn.Module) else sb
    try:
        target_dtype = next(sb.parameters()).dtype
    except StopIteration:
        target_dtype = z_star.dtype
    z_star, x0_lyap = _slice_for_fp_probe(z_star, x0_lyap, B_probe)
    z_star = z_star.to(target_dtype)
    x0_lyap = x0_lyap.to(target_dtype)
    b_bar = base_m._parcae_b_bar() if base_m.use_parcae else None
    b_bar_d = b_bar.detach().to(target_dtype) if b_bar is not None else None
    if base_m.use_parcae:
        a_bar = base_m._parcae_a_bar().detach().to(target_dtype)
        # Broadcast (D,) -> (1, 1, D) so a_bar_d * z works on (B, T, D).
        a_bar_d = a_bar.view(*([1] * (z_star.ndim - 1)), -1)
    else:
        a_bar_d = None
    return z_star, x0_lyap, b_bar_d, a_bar_d, sb_call, target_dtype, _fp_probe_context_factory()


def _clear_saved_fp_probe_tensors(base_m) -> None:
    """Release saved validation fixed-point tensors before returning to train."""
    for attr in ("_lyapunov_z_star", "_lyapunov_x0"):
        if hasattr(base_m, attr):
            setattr(base_m, attr, None)


def _to_cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(v) for v in obj)
    return obj


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _joint_F_residual_at_saved_fp(
    base_m,
    B_probe: int = 1,
    log_label: str = "fp_residual_F",
) -> float | None:
    """Joint two-state Parcae cycle residual at the saved FP.

    Computes ``r_F = ‖F(z*, z*) − (z*, z*)‖_RMS / ‖(z*, z*)‖_RMS`` for the
    actual cycle ``F(y, z) = (Ā·y + β·T(z), Ā·z + β·T(Ā·y + β·T(z)))``.
    At a true joint FP, ``r_F = 0``; in practice ``z*`` is from a finite-K
    solver, so ``r_F`` measures the distance to the joint FP and is reported
    as a standalone empirical convergence signal alongside ``rho_F`` and
    ``iter_conv_rel``. The earlier Banach pair ``fp_bound = r_F / (1 - lip_ub_F)``
    was removed 2026-05-15 — operator-norm bounds are over-restrictive for
    non-symmetric ``J_F``.

    Cost: 2 ``sb`` calls (one for ``T(z*)``, one for ``T(u')``); no autograd.
    Reuses the saved-FP probe preparation so dtype/slice choices match the
    spectral-norm probes.  Returns ``None`` when Parcae is disabled (the
    cycle reduces to a single-step iteration and the conv_rel z-only
    residual is the right object).
    """
    prepared = _prepare_saved_fp_probe(base_m, B_probe)
    if prepared is None:
        return None
    z_star, x0_lyap, b_bar_d, a_bar_d, sb, target_dtype, ctx_factory = prepared
    if a_bar_d is None:
        # Parcae disabled: F ≡ T (single step).  Caller should fall back
        # to the z-only conv_rel.
        return None
    a_bar_t = a_bar_d.to(device=z_star.device, dtype=target_dtype)
    try:
        with torch.no_grad():
            with ctx_factory(), torch.autocast(device_type="cuda", dtype=target_dtype):
                # Joint state at saved FP: (y*, z*) = (z_star, z_star).
                # Delegates to the shared cycle helper so the residual
                # measurement reuses `_parcae_cycle_F` (single source of truth).
                y_new, z_new = _parcae_cycle_F(
                    z_star, z_star, x0_lyap, b_bar_d, a_bar_t, sb,
                )
            diff_y = (y_new - z_star).float()
            diff_z = (z_new - z_star).float()
            num_sq = diff_y.pow(2).sum() + diff_z.pow(2).sum()
            # ‖(z*, z*)‖² = 2·‖z*‖²
            denom_sq = 2.0 * z_star.float().pow(2).sum().clamp_min(1e-12)
            return float((num_sq / denom_sq).sqrt().item())
    except torch.cuda.OutOfMemoryError as e:
        print(f"{log_label} skip: oom_runtime {e}", flush=True)
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        print(f"{log_label} skip: cuda_runtime {type(e).__name__}: {e}", flush=True)
        return None


def _resolve_training_seconds_alias(args, cli_overrides: dict[str, object]) -> None:
    """Bridge --max-wallclock-seconds (deprecated) onto args.max_training_seconds.
    Downstream timer logic reads only max_training_seconds after this."""
    legacy = float(getattr(args, "max_wallclock_seconds", 0.0))
    canonical = float(getattr(args, "max_training_seconds", 0.0))
    saw_legacy = "max_wallclock_seconds" in cli_overrides
    saw_canonical = "max_training_seconds" in cli_overrides
    if saw_legacy and not saw_canonical:
        # Defer the warning until log0 exists post-distributed-init.
        args.max_training_seconds = legacy
        args._max_wallclock_seconds_deprecated = True
    elif saw_legacy and saw_canonical and legacy != canonical:
        raise SystemExit(
            f"--max-wallclock-seconds={legacy} (deprecated) and "
            f"--max-training-seconds={canonical} disagree; pass only one."
        )


def _compute_training_budget_ms(
    max_training_seconds: float,
    eval_reservation_seconds: float,
) -> float | None:
    """Process wallclock budget minus eval reservation, in ms; None if uncapped.

    See `EXPERIENCE.md#scalar-semantic-shift` and CLAUDE.md
    "Project Invariants" (600 s 8×H100). Both args are required to make
    forgetting the reservation a type-checker error rather than a silent
    wallclock overrun.
    """
    max_train_s = float(max_training_seconds)
    if max_train_s <= 0.0:
        return None
    eval_res_s = max(float(eval_reservation_seconds), 0.0)
    train_budget_s = max(max_train_s - eval_res_s, 1.0)
    return 1000.0 * train_budget_s


def _compute_run_status(
    *,
    diagnostic_policy: str,
    health_valid: bool,
    full_val_completed: bool,
) -> tuple[bool, str, str | None]:
    # Pure decision table for `meta.json` `run_valid` / `status` / `non_promotable_reason`.
    # Extracted so the scalar-semantic shift in `diagnostic_gate_policy=hard` is
    # pinned by a focused test (truth table over policy × health_valid × full_val).
    if not full_val_completed:
        return False, "validated_fast_only", "final_full_validation_disabled"
    if health_valid:
        return True, "validated_clean", None
    if diagnostic_policy == "advisory":
        return True, "validated_with_tech_debt", None
    return False, "health_gate_failed", "diagnostic_gate_policy_hard"


def _validate_hyperparameters(args) -> None:
    """Fail-fast architecture/config validation — catches malformed combos
    at startup rather than as opaque reshape errors deep in SDPA forward.

    Tolerates `SimpleNamespace` test fixtures via `getattr`-with-default so
    focused tests can omit unrelated profile fields.
    """
    profile = str(getattr(args, "config_profile", "fast_default")).strip().lower()
    if profile not in _CONFIG_PROFILES:
        valid = ", ".join(sorted(_CONFIG_PROFILES))
        raise SystemExit(f"config_profile={profile!r} must be one of: {valid}")
    eval_profile = str(getattr(args, "eval_profile", "diagnostic")).strip().lower()
    if eval_profile not in _EVAL_PROFILES:
        valid = ", ".join(sorted(_EVAL_PROFILES))
        raise SystemExit(f"eval_profile={eval_profile!r} must be one of: {valid}")
    policy = str(getattr(args, "diagnostic_gate_policy", "advisory")).strip().lower()
    if policy not in ("advisory", "hard"):
        raise SystemExit("diagnostic_gate_policy must be one of: advisory, hard")

    md, nh, nkv = int(args.model_dim), int(args.num_heads), int(args.num_kv_heads)
    if md % nh != 0:
        raise SystemExit(f"model_dim ({md}) must be divisible by num_heads ({nh})")
    if nh % nkv != 0:
        raise SystemExit(f"num_heads ({nh}) must be divisible by num_kv_heads ({nkv}) for GQA")
    if int(args.num_layers) <= 0:
        raise SystemExit(f"num_layers ({args.num_layers}) must be positive")
    # iter161-QAT-late: start_step is either -1 (OFF, default) or a non-
    # negative integer (start QAT at that step). Negative values other
    # than -1 are nonsense — fail-fast rather than silently treat as OFF.
    qat_start = int(getattr(args, "qat_late_start_step", -1))
    if qat_start < -1:
        raise SystemExit(
            f"qat_late_start_step={qat_start} must be -1 (OFF) or a non-negative integer; "
            "negative values other than -1 are not allowed"
        )
    if qat_start > int(getattr(args, "iterations", 1)):
        raise SystemExit(
            f"qat_late_start_step={qat_start} exceeds total iterations "
            f"({args.iterations}); QAT would never activate"
        )
    if int(getattr(args, "lyapunov_every", 1)) <= 0:
        raise SystemExit(f"lyapunov_every ({args.lyapunov_every}) must be positive")
    if int(getattr(args, "lyapunov_max_tokens", 1)) <= 0:
        raise SystemExit(f"lyapunov_max_tokens ({args.lyapunov_max_tokens}) must be positive")
    lyap_target = str(getattr(args, "lyapunov_target", "transition_T"))
    if lyap_target not in ("transition_T", "iteration_S", "iteration_F"):
        raise SystemExit(
            f"lyapunov_target={lyap_target!r} must be one of: "
            "transition_T, iteration_S, iteration_F"
        )
    lyap_estimator = str(getattr(args, "lyapunov_estimator", "random_fd"))
    if lyap_estimator != "random_fd":
        raise SystemExit(
            f"lyapunov_estimator={lyap_estimator!r} must be 'random_fd'. "
            "The 'power_jvp_F' estimator was removed 2026-05-15 alongside the "
            "lip_ub_* operator-norm probes (both refuted by iter155/iter152 "
            "evidence). See CLAUDE.md for the principled rho_F gate."
        )
    nE, nS = int(args.num_experts), int(args.num_shared_experts)
    if nE <= 0:
        raise SystemExit(f"num_experts ({nE}) must be positive")
    if nS < 0:
        raise SystemExit(f"num_shared_experts ({nS}) must be non-negative")
    if nS >= nE:
        raise SystemExit(
            f"num_shared_experts ({nS}) must be < num_experts ({nE}) so at least "
            f"one routed expert remains"
        )
    if str(args.router_scoring) not in ("linear", "l2", "sips", "dirichlet_ucb"):
        raise SystemExit(f"router_scoring={args.router_scoring!r} must be one of linear,l2,sips,dirichlet_ucb")
    if str(args.router_scoring) == "dirichlet_ucb" and bool(getattr(args, "use_entmax_routing", False)):
        # entmax_1.5 is not invariant under the softmax→entmax substitution
        # `log(acq)`-as-logits assumes; the resulting blend would be ambiguous.
        raise SystemExit(
            "dirichlet_ucb scoring + use_entmax_routing is not a supported combination "
            "(entmax projection has no defined meaning on Dirichlet UCB acquisition scores)"
        )
    if bool(getattr(args, "use_sparse_dispatch", False)):
        raise SystemExit(
            "use_sparse_dispatch is not supported in train_gpt.py: the capacity/top-k "
            "dispatch path is discrete and is not RevDEQ-safe during training"
        )
    if int(getattr(args, "smear_gate_window", 1)) <= 0:
        raise SystemExit("smear_gate_window must be positive")
    if int(getattr(args, "smear_gate_window", 1)) > md:
        raise SystemExit(f"smear_gate_window ({args.smear_gate_window}) must be <= model_dim ({md})")
    if int(getattr(args, "sparse_attn_gate_window", 1)) <= 0:
        raise SystemExit("sparse_attn_gate_window must be positive")
    if int(getattr(args, "sparse_attn_gate_window", 1)) > md:
        raise SystemExit(
            f"sparse_attn_gate_window ({args.sparse_attn_gate_window}) must be <= model_dim ({md})"
        )
    if float(getattr(args, "sparse_attn_gate_factor", 1.0)) <= 0.0:
        raise SystemExit("sparse_attn_gate_factor must be positive")
    if float(getattr(args, "sparse_attn_gate_init_std", 0.0)) < 0.0:
        raise SystemExit("sparse_attn_gate_init_std must be non-negative")
    if bool(getattr(args, "use_rr_attention", False)) and bool(getattr(args, "use_nsa_attention", False)):
        raise SystemExit("use_rr_attention and use_nsa_attention are mutually exclusive attention dispatches")
    if int(getattr(args, "rr_stride", 1)) <= 0:
        raise SystemExit("rr_stride must be positive")
    if int(getattr(args, "rr_block_size", 1)) <= 0:
        raise SystemExit("rr_block_size must be positive")
    if int(getattr(args, "rr_block_size", 1)) % int(getattr(args, "rr_stride", 1)) != 0:
        raise SystemExit("rr_block_size must be a multiple of rr_stride")
    if int(getattr(args, "train_seq_len", 1)) % int(getattr(args, "rr_stride", 1)) != 0:
        raise SystemExit("train_seq_len must be divisible by rr_stride")
    if int(getattr(args, "train_seq_len", 1)) % int(getattr(args, "rr_block_size", 1)) != 0:
        raise SystemExit("train_seq_len must be divisible by rr_block_size")
    rr_tau = float(getattr(args, "rr_tau", 0.0))
    if not math.isfinite(rr_tau) or not (0.0 < rr_tau <= 1.0):
        raise SystemExit("rr_tau must be finite and in (0, 1]")
    # Flag-to-effect contract: rr_attention silently falls back to dense SDPA
    # above _RR_MAX_TOKEN_MASK_TOKENS, so enabling it at the project default
    # train_seq_len=2048 would be a no-op with a misleading banner. Checked
    # last in the rr block so the more specific divisibility errors fire first
    # when both conditions are violated.
    if bool(getattr(args, "use_rr_attention", False)):
        from experiments.components.rr_attention import _RR_MAX_TOKEN_MASK_TOKENS
        rr_T_cap = int(_RR_MAX_TOKEN_MASK_TOKENS)
        if int(getattr(args, "train_seq_len", 1)) > rr_T_cap:
            raise SystemExit(
                f"use_rr_attention requires train_seq_len <= {rr_T_cap} "
                f"(rr_attention silently falls back to dense SDPA above that cap)"
            )
    # Flag-to-effect contract: --use-gptq / --use-lqer currently only run a
    # synthetic-tensor smoke; the scored int6 artifact path is unchanged, so a
    # banner advertising gptq=1 / lqer=1 would misrepresent the run.
    if bool(getattr(args, "use_gptq", False)) or bool(getattr(args, "use_lqer", False)):
        raise SystemExit(
            "use_gptq / use_lqer are scaffolds: only run_gptq_lqer_component_smoke "
            "exercises them, and the scored int6 artifact path is unchanged. Disable "
            "the flag or wire the codec into save_int6_artifact first."
        )
    # Flag-to-effect contract: --use-ttt-eval has no consumer in train_gpt.py
    # (no forward_ttt; _evaluate_val_loss does not branch on the flag).
    if bool(getattr(args, "use_ttt_eval", False)):
        raise SystemExit(
            "use_ttt_eval has no training- or eval-path consumer in train_gpt.py. "
            "Disable the flag until the phased TTT eval driver is wired."
        )
    # Flag-to-effect contract: --use-caseops only runs a hardcoded-string fixture
    # print; the FineWeb shards are already tokenized and bypass the codec.
    if bool(getattr(args, "use_caseops", False)):
        raise SystemExit(
            "use_caseops only runs a fixture print over a hardcoded smoke string; "
            "the tokenized dataset bypasses it. Disable the flag until the codec "
            "is wired into the data pipeline."
        )
    if bool(getattr(args, "deq_prefix_anchors", False)) and int(getattr(args, "num_refinements", 0)) != 0:
        raise SystemExit("deq_prefix_anchors currently requires num_refinements=0")
    if int(args.train_seq_len) <= 0:
        raise SystemExit(f"train_seq_len ({args.train_seq_len}) must be positive")
    if int(args.train_batch_tokens) % int(args.train_seq_len) != 0:
        raise SystemExit(
            f"train_batch_tokens ({args.train_batch_tokens}) must be divisible "
            f"by train_seq_len ({args.train_seq_len})"
        )
    if int(args.grad_accum_multiplier) <= 0:
        raise SystemExit(f"grad_accum_multiplier ({args.grad_accum_multiplier}) must be positive")


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    cli_overrides = _parse_cli_overrides(sys.argv[1:])
    args = Hyperparameters()
    profile_name = str(cli_overrides.pop("config_profile", args.config_profile))
    _apply_config_profile(args, profile_name)
    for k, v in cli_overrides.items():
        _assert_known_hyperparameter(k, source="cli_override")
        setattr(args, k, v)
    _resolve_training_seconds_alias(args, cli_overrides)
    _validate_hyperparameters(args)
    # iter161-QAT-late (2026-05-16): configure the module-global QAT state
    # at startup so CastedLinear.forward can branch on it. set_step() is
    # called per training iteration in the main loop.
    _QAT_LATE_STATE.reset()
    _qat_start = int(getattr(args, "qat_late_start_step", -1))
    if _qat_start >= 0:
        _QAT_LATE_STATE.configure(start_step=_qat_start, sdclip_k=SDCLIP_K_MATRIX)
    if float(getattr(args, "max_training_seconds", 0.0)) > 0.0 and "iterations" not in cli_overrides:
        args.iterations = int(1_000_000_000)
    # iter 121 PE-NS gate: apply chosen coefficient set BEFORE building model
    # (_ns5_2d / _ns5_batched read the module-level flag at forward time, but
    # torch.compile inlines the iter_coeffs call so we want the flag set by
    # the time the model is constructed and compile fires).
    _apply_ns_coefficient_choice(getattr(args, "use_polar_express_ns", False))
    # iter 117b-2: Triton entmax kernel toggle. Same module-level pattern;
    # set BEFORE model construction so dynamo constant-folds the dispatch
    # branch into the compiled SoftDenseRouter graph.
    _set_entmax_triton(getattr(args, "use_entmax_triton", False))
    # iter 117b-3: sparse MoE dispatch toggle (capacity-padded gather/scatter).
    # Module-level flag read by helper at forward time; constant-folded by
    # dynamo when set BEFORE model construction.
    _set_sparse_dispatch(
        getattr(args, "use_sparse_dispatch", False),
        getattr(args, "sparse_dispatch_capacity_factor", 4.0),
    )
    # iter 118a Phase A3: fused routed-down kernel toggle. Same pattern.
    _set_unified_routed_down(getattr(args, "use_unified_routed_down", False))
    # iter 103: legacy boolean is superseded by the preset selector. Keep a
    # fail-fast guard so old launch scripts do not silently pick a variant.
    if getattr(args, "use_chained_routing", False):
        raise NotImplementedError(
            "use_chained_routing is deprecated; launch iter 103 with "
            "--chained-stages-preset={split_2stage,attn_first_2stage,mlp_first_2stage}."
        )
    from experiments.components.chained_routing import set_chained_routing_enabled
    _chained_preset = getattr(args, "chained_stages_preset", None)
    set_chained_routing_enabled(_chained_preset is not None and str(_chained_preset).strip().lower() not in ("", "none"))
    from experiments.components.artifact_compression import set_grouped_artifact_compression_enabled
    from experiments.components.caseops_tokenizer import set_caseops_enabled
    from experiments.components.gptq_lqer import set_gptq_lqer_enabled
    from experiments.components.phased_ttt import set_ttt_eval_enabled
    from experiments.components.rr_attention import set_rr_attention
    from experiments.components.smear_gate import set_smear_gate_enabled
    from experiments.components.sparse_attn_head_gate import set_sparse_attn_head_gate_enabled
    set_smear_gate_enabled(
        getattr(args, "use_smear_gate", False),
        window=getattr(args, "smear_gate_window", 12),
        bos_id=getattr(args, "smear_gate_bos_id", 1),
    )
    set_sparse_attn_head_gate_enabled(
        getattr(args, "use_sparse_attn_head_gate", False),
        gate_window=getattr(args, "sparse_attn_gate_window", 12),
        scale=getattr(args, "sparse_attn_gate_scale", 1.0),
        gate_factor=getattr(args, "sparse_attn_gate_factor", 2.0),
    )
    set_rr_attention(
        getattr(args, "use_rr_attention", False),
        stride=getattr(args, "rr_stride", 8),
        block_size=getattr(args, "rr_block_size", 64),
        tau=getattr(args, "rr_tau", 0.95),
    )
    set_ttt_eval_enabled(getattr(args, "use_ttt_eval", False))
    set_gptq_lqer_enabled(
        use_gptq=getattr(args, "use_gptq", False),
        use_lqer=getattr(args, "use_lqer", False),
    )
    set_grouped_artifact_compression_enabled(getattr(args, "use_grouped_artifact_compression", False))
    set_caseops_enabled(getattr(args, "use_caseops", False))
    args.train_files = os.path.join(args.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")
    if not getattr(args, "run_id", ""):
        args.run_id = str(uuid.uuid4())
    _resume_from_cli = str(getattr(args, "resume_from", "") or "").strip()
    resume_requested = bool(getattr(args, "resume_latest", False)) or _resume_from_cli not in ("", "none", "None")

    if int(args.deq_k_min) <= 0:
        raise ValueError("deq_k_min must be positive")
    if int(args.deq_k_max) < int(args.deq_k_min):
        raise ValueError("deq_k_max must be >= deq_k_min")
    _k_values_for_validation = getattr(args, "deq_k_jitter_set", None)
    _k_weights_for_validation = getattr(args, "deq_k_jitter_weights", None)
    if _k_weights_for_validation:
        _k_vals_norm, _k_weights_norm = _normalize_k_jitter_weights(
            _k_values_for_validation, _k_weights_for_validation)
        args.deq_k_jitter_set = tuple(_k_vals_norm)
        args.deq_k_jitter_weights = tuple(_k_weights_norm)
    # deq_k_max can exceed num_layers: the DEQ uses a shared block so the
    # solver can run any number of iterations.  num_layers is just the default K.

    # NS functions already compiled at module scope (L286-287). No recompile needed.

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Base grad_accum: 4 global microsteps / world_size (so per-rank microstep
    # count is modest). RevDEQ's O(1) backward memory makes the base sufficient.
    # k_rope permute fix (64ac616) doubled the Hutchinson VJP VRAM, so we
    # double grad_accum to halve per-microstep B.
    # Iter 98b: args.grad_accum_multiplier (default 1; 2 under iter 98b D=1024)
    # halves per-step activation memory by doubling micro-steps; effective batch
    # is invariant so LR / WD do NOT need rescaling.
    grad_accum_steps = max(1, math.ceil(4 / world_size)) * 2 * args.grad_accum_multiplier
    global_seqs = args.train_batch_tokens // args.train_seq_len
    while grad_accum_steps > 1 and global_seqs < world_size * grad_accum_steps:
        grad_accum_steps -= 1
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # TF32 for torch.compile-generated kernels
    # Phase 6a.4 (review 6): fail loud on math-SDPA fallback.  The math
    # kernel is ~10× slower than flash; a silent fallback invalidates
    # wallclock comparisons.  Disabling it forces SDPA to raise instead of
    # quietly slow-pathing — a promotion-gating regression we want to see.
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    sdp_kernel_policy = (
        "sdpa_kernels: flash=ON mem_eff=OFF math=OFF cudnn=OFF "
        "(math fallback disabled — SDPA will raise on unsupported shapes)"
    )

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        with open(logfile, "a" if resume_requested else "w", encoding="utf-8") as f:
            if resume_requested:
                f.write("\n")
        print(logfile, flush=True)

        # Clear stale artifacts from any previous run so a crash mid-way doesn't
        # leave a half-written experiments/weights/current/ that gets picked up
        # by update_results.sh --promote.  Also preemptively write meta.json
        # with run_valid=false so even if THIS run aborts before the final
        # assertion, promotion refuses.
        current_dir = Path("experiments/weights/current")
        if current_dir.exists():
            for f in current_dir.iterdir():
                if f.is_file():
                    f.unlink()
        current_dir.mkdir(parents=True, exist_ok=True)
        with open(current_dir / "meta.json", "w") as mf:
            # Write BOTH schema variants (step/steps, commit/git_commit) so the
            # log-rotation script's python3 json reads don't abort under set -e
            # regardless of which key it expects.
            json.dump(
                {"val_bpb": 0.0, "artifact_bytes": 0,
                 "step": 0, "steps": 0,
                 "commit": "", "git_commit": "",
                 "run_valid": False, "status": "in_progress"},
                mf,
            )
        if not resume_requested:
            ckpt_dir_for_run = Path(str(args.checkpoint_dir)) / str(args.run_id)
            if ckpt_dir_for_run.exists():
                for f in ckpt_dir_for_run.glob("*.pt*"):
                    if f.is_file() or f.is_symlink():
                        f.unlink()

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    if getattr(args, "_max_wallclock_seconds_deprecated", False):
        log0(
            f"deprecation_warning: --max-wallclock-seconds is now an alias for "
            f"--max-training-seconds (process wallclock budget; training loop "
            f"reserves {int(args.eval_reservation_seconds)}s for post-loop work). "
            f"Update record commands to --max-training-seconds={int(args.max_training_seconds)}."
        )

    # train_batch_tokens divisibility depends on world_size, so it lives here
    # rather than in _validate_hyperparameters which runs pre-distributed-init.
    _T = int(args.train_seq_len)
    _gam = int(getattr(args, "grad_accum_multiplier", 1))
    _per_step_tokens = _T * world_size * _gam
    if int(args.train_batch_tokens) % _per_step_tokens != 0:
        raise SystemExit(
            f"train_batch_tokens ({args.train_batch_tokens}) must be divisible by "
            f"train_seq_len * world_size * grad_accum_multiplier "
            f"({_T} * {world_size} * {_gam} = {_per_step_tokens})"
        )

    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(sdp_kernel_policy, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    k_rng = random.Random(args.seed + 12345)
    _k_jitter_set = getattr(args, "deq_k_jitter_set", None)
    _k_jitter_weights = getattr(args, "deq_k_jitter_weights", None)
    k_sampler = KShuffleBagSampler(args.deq_k_min, args.deq_k_max, k_rng,
                                    step=int(args.deq_k_step),
                                    values=list(_k_jitter_set) if _k_jitter_set else None,
                                    weights=list(_k_jitter_weights) if _k_jitter_weights else None)
    k_sample_counts: Counter[int] = Counter()
    # iter173 K-jitter annealing curriculum: re-discretize the current weights
    # at integer permille resolution and only rebuild the sampler bag when
    # the rounded weights actually changed. This keeps per-step overhead at
    # one short tuple comparison while still tracking the linear interpolation
    # smoothly enough (1000 distinct stages over the anneal window).
    _k_jitter_weights_final = tuple(getattr(args, "deq_k_jitter_weights_final", ()) or ())
    _k_jitter_anneal_active = bool(
        _k_jitter_weights_final
        and len(_k_jitter_weights_final) == len(_k_jitter_weights or ())
    )
    _k_jitter_anneal_start = float(getattr(args, "deq_k_jitter_anneal_start_frac", 0.3))
    _k_jitter_anneal_end = float(getattr(args, "deq_k_jitter_anneal_end_frac", 0.9))
    _k_jitter_anneal_total = int(args.iterations)
    _last_quantized_weights: tuple[int, ...] = ()

    def _maybe_anneal_k_jitter_weights(step_i: int) -> None:
        nonlocal _last_quantized_weights
        if not _k_jitter_anneal_active:
            return
        new_weights = _compute_annealed_k_jitter_weights(
            initial=_k_jitter_weights or (),
            final=_k_jitter_weights_final,
            step=step_i,
            total_steps=_k_jitter_anneal_total,
            start_frac=_k_jitter_anneal_start,
            end_frac=_k_jitter_anneal_end,
        )
        quantized = tuple(int(round(w * 1000.0)) for w in new_weights)
        if quantized != _last_quantized_weights:
            if rank == 0:
                k_sampler.set_weights(list(new_weights))
            _last_quantized_weights = quantized

    def deq_k_for_step(step_i: int) -> int:
        # Short-circuit when jitter is disabled — broadcasting a constant
        # value every step is wasteful and pulls a `.item()` sync into the
        # train hot path. The default (jitter=False) hits this fast path.
        if not args.deq_k_jitter:
            k_fixed = int(args.deq_k_max)
            k_sample_counts[k_fixed] += 1
            return k_fixed
        _maybe_anneal_k_jitter_weights(step_i)
        k = 0
        if rank == 0:
            k = int(k_sampler.sample())
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        k_sample_counts[int(k)] += 1
        return int(k)

    def format_k_jitter_info() -> str:
        total = sum(k_sample_counts.values())
        if total <= 0:
            return ""
        mean_k = sum(int(k) * int(n) for k, n in k_sample_counts.items()) / float(total)
        counts = ",".join(f"{int(k)}:{int(k_sample_counts[k])}" for k in sorted(k_sample_counts))
        return f" deq_k_sample_mean:{mean_k:.2f} deq_k_counts:[{counts}]"

    # Phase 9 iter 49: β jitter — sample β per step (like K-jitter, H30).
    # Forces model to be robust across solver dynamics. RevDEQ-safe because
    # β is constant within each step (all iterations use the same β).
    _beta_jitter_set = getattr(args, "deq_beta_jitter_set", None)
    _beta_bag: list[float] = []
    _beta_rng = random.Random(42 + 7)  # separate RNG from K-jitter

    def deq_beta_for_step(step_i: int) -> float:
        if not getattr(args, "deq_beta_jitter", False):
            return float(args.deq_beta)
        nonlocal _beta_bag
        b = 0.0
        if rank == 0:
            if not _beta_bag:
                _beta_bag = list(_beta_jitter_set) if _beta_jitter_set else [0.3, 0.5, 0.7]
                _beta_rng.shuffle(_beta_bag)
            b = float(_beta_bag.pop())
        if distributed:
            b_t = torch.tensor([b], device=device, dtype=torch.float32)
            dist.broadcast(b_t, src=0)
            b = float(b_t.item())
        return b

    # Iter 85: stochastic TBPTT sampler (shuffle-bag, mirroring β-jitter).
    _bptt_k_jitter_set = getattr(args, "deq_bptt_k_jitter_set", None)
    _bptt_k_bag: list[int] = []
    _bptt_k_rng = random.Random(args.seed + 31337)

    def deq_bptt_k_for_step(step_i: int) -> int:
        if not getattr(args, "deq_bptt_k_jitter", False):
            return int(args.deq_bptt_k)
        nonlocal _bptt_k_bag
        k = int(args.deq_bptt_k)
        if rank == 0:
            if not _bptt_k_bag:
                _bptt_k_bag = list(_bptt_k_jitter_set) if _bptt_k_jitter_set else [int(args.deq_bptt_k)]
                _bptt_k_rng.shuffle(_bptt_k_bag)
            k = int(_bptt_k_bag.pop())
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        return k

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab={int(sp.vocab_size())}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"run_id:{args.run_id}")
    optional_flags_banner = " ".join(
        f"{label}={int(bool(getattr(args, py_name, False)))}"
        for py_name, label in _OPTIONAL_COMPONENT_FLAGS
    )
    log0(
        f"config:"
        f" model_dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads}"
        f" mlp_mult={args.mlp_mult} beta={args.deq_beta:.3f}"
        f" deq_k_range={args.deq_k_min}-{args.deq_k_max} deq_k_eval={args.deq_k_eval}"
        f" deq_k_jitter_set={tuple(getattr(args, 'deq_k_jitter_set', ()) or ())}"
        f" deq_k_jitter_weights={tuple(getattr(args, 'deq_k_jitter_weights', ()) or ())}"
        f" deq_bptt_k={args.deq_bptt_k}"
        f" batch_tokens={args.train_batch_tokens} seq_len={args.train_seq_len}"
        f" refinements={args.num_refinements} refine_ramp_frac={args.num_refinements_ramp_frac}"
        f" ema={int(args.ema_enabled)} ema_decay={args.ema_decay:.4f}"
        f" config_profile={args.config_profile}"
        f" eval_profile={args.eval_profile}"
        f" diagnostic_gate_policy={args.diagnostic_gate_policy}"
        f" pooled_router=True router_scoring={args.router_scoring}"
        f" router_ucb_beta={float(args.router_dirichlet_ucb_beta):.4g}"
        f" router_sigmoid_gate={int(bool(args.use_router_sigmoid_gate))}"
        f" {optional_flags_banner}"
    )

    # MODEL
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim, num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank, mlp_expert_rank=args.mlp_expert_rank,
        deq_beta=args.deq_beta,
        deq_bptt_k=args.deq_bptt_k,
        deq_prefix_anchors=bool(getattr(args, "deq_prefix_anchors", False)),
        # iter172 default is the explicit anchor set (8, 16, 24, 32, 64, 128);
        # prefer a CLI override when supplied, else fall back to the jitter
        # set (legacy behavior — the anchor set may differ from the jitter
        # set, e.g. shallow K=8 anchor but no K=8 in jitter weights).
        deq_prefix_anchor_set=tuple(int(k) for k in (
            getattr(args, "deq_prefix_anchor_set", None)
            or getattr(args, "deq_k_jitter_set", ())
            or ()
        )),
        num_experts=args.num_experts, num_shared_experts=args.num_shared_experts,
        router_scoring=args.router_scoring,
        router_pertoken_entropy_coef=float(args.router_pertoken_entropy_coef),
        router_dirichlet_ucb_beta=float(args.router_dirichlet_ucb_beta),
        use_router_sigmoid_gate=bool(args.use_router_sigmoid_gate),
        use_entmax_routing=args.use_entmax_routing,
        entmax_blend_init_logit=float(args.entmax_blend_init_logit),
        entmax_blend_warmup_delay_frac=float(args.entmax_blend_warmup_delay_frac),
        use_smear_gate=bool(args.use_smear_gate),
        smear_gate_init=float(args.smear_gate_init),
        smear_gate_window=int(args.smear_gate_window),
        smear_gate_bos_id=int(args.smear_gate_bos_id),
        logit_softcap=float(args.logit_softcap),
        lyapunov_coef=args.lyapunov_coef,
        lyapunov_gamma=args.lyapunov_gamma,
        lyapunov_every=args.lyapunov_every,
        lyapunov_max_tokens=args.lyapunov_max_tokens,
        lyapunov_target=getattr(args, "lyapunov_target", "transition_T"),
        lyapunov_estimator=getattr(args, "lyapunov_estimator", "random_fd"),
        use_parcae=args.use_parcae,
        parcae_init_a_bar=args.parcae_init_a_bar,
        parcae_init_b_bar=args.parcae_init_b_bar,
        use_ctp=args.use_ctp,
        ctp_weight=float(args.ctp_weight),
        router_ema_alive_coef=float(args.router_ema_alive_coef),
        router_ema_balance_coef=float(args.router_ema_balance_coef),
        router_ema_specialization_coef=float(args.router_ema_specialization_coef),
        use_reverse_kl_balance=bool(args.use_reverse_kl_balance),
        use_expert_perdim_gate=bool(getattr(args, "use_expert_perdim_gate", False)),
        expert_perdim_gate_rank=int(getattr(args, "expert_perdim_gate_rank", 16)),
        multi_k_consistency_anchor_coef=float(getattr(args, "multi_k_consistency_anchor_coef", 0.0)),
        expert_diversity_kind=str(args.expert_diversity_kind),
        expert_output_diversity_coef=float(args.expert_output_diversity_coef),
        expert_diversity_every=int(args.expert_diversity_every),
        expert_diversity_max_tokens=int(args.expert_diversity_max_tokens),
        mos_output_diversity_coef=float(args.mos_output_diversity_coef),
        regularizer_warmup_frac=float(args.regularizer_warmup_frac),
        use_nsa_attention=args.use_nsa_attention,
        nsa_compress_block_size=args.nsa_compress_block_size,
        nsa_compress_block_sliding_stride=args.nsa_compress_block_sliding_stride,
        nsa_sliding_window_size=args.nsa_sliding_window_size,
        nsa_branch_gate_init=args.nsa_branch_gate_init,
        use_sparse_attn_head_gate=bool(args.use_sparse_attn_head_gate),
        sparse_attn_gate_window=int(args.sparse_attn_gate_window),
        sparse_attn_gate_scale=float(args.sparse_attn_gate_scale),
        sparse_attn_gate_factor=float(args.sparse_attn_gate_factor),
        sparse_attn_gate_init_std=float(args.sparse_attn_gate_init_std),
        use_rr_attention=bool(args.use_rr_attention),
        rr_stride=int(args.rr_stride),
        rr_block_size=int(args.rr_block_size),
        rr_tau=float(args.rr_tau),
        chained_stages_preset=args.chained_stages_preset,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # GPU-class compile policy: full shared_block compile on ≥60 GB GPUs
    # (H100), sub-module compile elsewhere. The legacy "unroll"/"autograd"
    # backward modes were removed (user directive) — only revdeq is supported,
    # and revdeq is always compile-compatible because its VJP backward is a
    # single block.forward call.
    _gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if _gpu_mem_gb >= 60:
        try:
            base_model.shared_block = torch.compile(base_model.shared_block, dynamic=False)
            log0(f"compiled shared_block (dynamic=False, {_gpu_mem_gb:.0f} GB GPU)")
        except Exception as e:
            log0(f"shared_block compile failed ({e}), falling back to sub-module compile")
    # Compile block.forward METHOD (not the module) to fuse router→attn→MLP.
    # Compiling the method avoids DDP graph expansion that caused the ~40 GB
    # workspace OOM with torch.compile(module).  Works with revdeq because
    # the VJP backward is a single block.forward call.  1.97× speedup, 4.4 GB.
    # 2026-04-28 PROFILE-driven: raise dynamo's compile-cache size. Default 8
    # was hit during dev runs by K-jitter (3 unique K values change the
    # `_*_expert_weights_per_iter` list lengths and thus the graph variant) ×
    # any other guard axis (e.g., `_diag_track_enabled` toggling at log
    # boundaries). Hitting recompile_limit causes dynamo to fall back to eager
    # mode for the offending frame, losing the `compiled block.forward` 1.97×
    # speedup. Setting to 16 gives headroom for the 3 K-variants × eval/train
    # mode × diagnostic-on/off = up to 12 expected slots without thrash.
    # Setting both attribute names — recompile_limit is the current name in
    # this PyTorch version (per the W0428 14:42:19 dynamo log message);
    # cache_size_limit is the older alias.
    for _attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, _attr):
            setattr(torch._dynamo.config, _attr, 16)

    if not hasattr(base_model.shared_block, '_orig_mod'):  # not already full-compiled
        sb = base_model.shared_block
        try:
            # Option A test (2026-04-28) FAILED — dynamic=True crashed at val_loss
            # eval under DDP with `BackendCompilerFailed: AttributeError: 'int'
            # object has no attribute 'meta'` (aot_autograd inference-compile
            # pre_compile pass at runtime_wrappers.py:577). Known pytorch issue
            # for the dynamic=True + DDP + aot_autograd inference path. Smoke
            # test PASSED on single-GPU (no DDP), masking the bug. Reverted to
            # dynamic=False — keeps the static-shape Inductor optimizations and
            # avoids the DDP interaction. The 12 cache slots (2 K × 2 grad_mode
            # × 3 graph types) fit comfortably under recompile_limit=16 since
            # Fix #5a, so cache simplification is no longer pressing.
            sb.forward = torch.compile(sb.forward, dynamic=False)
            log0("compiled block.forward (1.97× speedup, fused router→attn→MLP)")
        except Exception as e:
            log0(f"block.forward compile failed ({e}), running eager")

    # Compile MoS head forward (5.45× speedup: 34.7ms → 6.4ms at B=32).
    # Called once per micro-step (not inside DEQ loop), no chaining issue.
    try:
        base_model.mos_head.forward = torch.compile(base_model.mos_head.forward, dynamic=False)
        log0("compiled mos_head.forward (5.45× speedup)")
    except Exception as e:
        log0(f"mos_head compile failed ({e}), running eager")

    model: nn.Module = (
        DDP(base_model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False,
            bucket_cap_mb=50)  # T-opt 22: larger buckets → fewer all_reduce calls (~10M params fit in 1 bucket)
        if distributed else base_model
    )

    # OPTIMIZER SETUP
    _opt_groups = _build_optimizer_param_lists(base_model, args)
    tok_params = _opt_groups.tok
    matrix_params = _opt_groups.matrix
    scalar_params = _opt_groups.scalar
    parcae_params = _opt_groups.parcae
    entmax_blend_params = _opt_groups.entmax_blend

    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2),
                                       eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    # Expert banks use (E, R, D) or (E, D, R) layouts. Muon handles both via
    # the last-2-dims NS preconditioner; no special transpose param group needed.
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    # Phase 9 iter 66a: Parcae separate optimizer — NO weight decay, slow LR.
    if base_model.use_parcae:
        optimizer_parcae = torch.optim.AdamW(
            [{"params": parcae_params, "lr": args.parcae_lr, "base_lr": args.parcae_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.0, fused=True)
        optimizers.append(optimizer_parcae)
    # iter 117 v2 (H87): blend_logit gets a 10× slower LR (mirrors parcae_lr
    # precedent for sensitive system-dynamics params). Empty list when
    # use_entmax_routing=False — no optimizer added.
    if entmax_blend_params:
        optimizer_entmax_blend = torch.optim.AdamW(
            [{"params": entmax_blend_params, "lr": args.entmax_blend_lr, "base_lr": args.entmax_blend_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.0, fused=True)
        optimizers.append(optimizer_entmax_blend)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    checkpoint_run_dir = Path(str(args.checkpoint_dir)) / str(args.run_id)

    def _checkpoint_path_for_step(step_i: int) -> Path:
        return checkpoint_run_dir / f"step_{int(step_i):06d}.pt"

    def _latest_checkpoint_path() -> Path:
        return checkpoint_run_dir / "latest.pt"

    def _resolve_resume_checkpoint_path() -> Path | None:
        resume_from = str(getattr(args, "resume_from", "") or "").strip()
        if resume_from and resume_from not in ("none", "None"):
            return Path(resume_from)
        if bool(getattr(args, "resume_latest", False)):
            return _latest_checkpoint_path()
        return None

    def _save_training_checkpoint(step_i: int, training_time_i: float, stop_after_i: int | None) -> None:
        if not master_process:
            return
        payload = {
            "schema": 1,
            "run_id": str(args.run_id),
            "step": int(step_i),
            "training_time_ms": float(training_time_i),
            "stop_after_step": stop_after_i,
            "model": _to_cpu_tree(base_model.state_dict()),
            "optimizers": [_to_cpu_tree(opt.state_dict()) for opt in optimizers],
            "train_loader": train_loader.state_dict(),
            "k_sampler": k_sampler.state_dict(),
            "k_sample_counts": {int(k): int(v) for k, v in k_sample_counts.items()},
            "beta_bag": list(_beta_bag),
            "beta_rng_state": _beta_rng.getstate(),
            "bptt_k_bag": list(_bptt_k_bag),
            "bptt_k_rng_state": _bptt_k_rng.getstate(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "ema_state": _to_cpu_tree(ema_state) if ema_state is not None else None,
            "swa_state": _to_cpu_tree(swa_state) if swa_state is not None else None,
            "swa_count": int(swa_count),
            "step_dt_window": list(_step_dt_window),
        }
        step_path = _checkpoint_path_for_step(step_i)
        _atomic_torch_save(payload, step_path)
        latest_path = _latest_checkpoint_path()
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            os.link(step_path, latest_path)
        except OSError:
            shutil.copy2(step_path, latest_path)
        keep = max(1, int(getattr(args, "checkpoint_keep", 2)))
        old_paths = sorted(checkpoint_run_dir.glob("step_*.pt"))
        for old_path in old_paths[:-keep]:
            if old_path != step_path:
                old_path.unlink(missing_ok=True)
        log0(f"checkpoint:saved step:{int(step_i)} path:{step_path}")

    # `max_training_seconds` is the *process wallclock* budget; the training
    # loop runs for at most that minus `eval_reservation_seconds` so that
    # post-loop val + int6 roundtrip + K-sweep + sliding val fit under the
    # same process wallclock. Project invariant: submission must fit 600 s.
    max_training_ms: float | None = _compute_training_budget_ms(
        float(getattr(args, "max_training_seconds", 0.0)),
        float(getattr(args, "eval_reservation_seconds", 0.0)),
    )
    max_wallclock_ms = max_training_ms  # deprecated alias for downstream log-grep compat

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if max_training_ms is None:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_frac = float(getattr(args, "warmdown_frac", 0.72))
        total_est_steps = max_training_ms / max(step_ms, 1e-9)
        warmdown_steps = warmdown_frac * total_est_steps
        warmdown_ms = warmdown_steps * step_ms
        remaining_ms = max(max_training_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def format_deq_info(m: nn.Module) -> str:
        parts: list[str] = []
        if hasattr(m, "_deq_k_last"):
            parts.append(f"deq_k:{int(m._deq_k_last)}")
        prefix_anchors = getattr(m, "_deq_prefix_anchor_depths_last", None)
        if prefix_anchors:
            parts.append(f"deq_prefix_anchors:[{','.join(str(int(k)) for k in prefix_anchors)}]")
        resid_t = getattr(m, "_deq_residual_t", None)
        if isinstance(resid_t, torch.Tensor):
            parts.append(f"deq_residual:{float(resid_t.detach().float().item()):.6f}")
        # RevDEQ diagnostics (CLAUDE.md §6.1):
        #  - `tbptt_recon`: ||z_rec_K_bwd − z_snap|| / ||z_snap|| — in-window
        #    reversibility check against the forward snapshot at iter K − K_bwd.
        #    Always emitted; reflects the gradient path training consumes.
        #  - `deq_recon_err`: ||z_rec_0 − z_0|| / ||z_0|| — emitted only under
        #    full BPTT where the grad reverse already lands at z_0 (free).
        #    Skipped under TBPTT (rebuilding via no_grad would waste compute
        #    on a value the gradient path never uses).
        #  - `deq_fp_travel`: ||z_K − z_0|| / ||z_0|| — distance from initial
        #    embedding to converged FP, expressivity proxy.
        recon = getattr(m, "_deq_recon_error", None)
        if recon is not None:
            parts.append(f"tbptt_recon:{float(recon):.3e}")
        dist = getattr(m, "_deq_x0_recon_error", None)
        if dist is not None:
            parts.append(f"deq_recon_err:{float(dist):.3e}")
        fp_t = getattr(m, "_deq_fp_travel", None)
        if fp_t is not None:
            parts.append(f"deq_fp_travel:{float(fp_t):.3e}")
        conv_t = getattr(m, "_deq_iter_convergence_t", None)
        if isinstance(conv_t, torch.Tensor):
            parts.append(f"deq_iter_conv:{float(conv_t.detach().float().item()):.6f}")
        conv_rel_t = getattr(m, "_deq_iter_convergence_rel_t", None)
        if isinstance(conv_rel_t, torch.Tensor):
            parts.append(f"deq_iter_conv_rel:{float(conv_rel_t.detach().float().item()):.6f}")
        # TBPTT: per-backward-iter VJP norms (index 0 = last forward iter).
        # A steep decay confirms the gradient-scale hypothesis.
        sb_for_vjp = getattr(m, "shared_block", None)
        vjp_iter = getattr(sb_for_vjp, "_tbptt_vjp_iter_last_bwd", None) if sb_for_vjp is not None else None
        if vjp_iter and len(vjp_iter) > 0:
            # Phase 6a.5: values may be GPU tensors (deferred sync) or floats.
            sums = [float(a.item() if hasattr(a, 'item') else a)
                    + float(b.item() if hasattr(b, 'item') else b)
                    for a, b in vjp_iter]
            parts.append(f"tbptt_vjp:[{','.join(f'{v:.2e}' for v in sums)}]")
            k_bwd = getattr(sb_for_vjp, "_tbptt_bwd_k_last", None)
            k_fwd = getattr(sb_for_vjp, "_tbptt_fwd_k_last", None)
            if k_bwd is not None and k_fwd is not None:
                parts.append(f"tbptt_k:{int(k_bwd)}/{int(k_fwd)}")
        # Per-iteration attention gate trajectory
        ag_iter = getattr(m, "_attn_gate_iter_last_solve", None)
        if ag_iter is not None and len(ag_iter) > 0:
            parts.append(f"attn_gate_iter:[{','.join(f'{v:.3f}' for v in ag_iter)}]")
        # Per-iteration router gate trajectory (combined, backward compat)
        rg_iter = getattr(m, "_router_gate_iter_last_solve", None)
        if rg_iter is not None and len(rg_iter) > 0:
            parts.append(f"router_gate_iter:[{','.join(f'{v:.3f}' for v in rg_iter)}]")
        # Per-component router gate trajectories (attn vs FFN)
        attn_rg_iter = getattr(m, "_attn_router_gate_iter_last_solve", None)
        if attn_rg_iter is not None and len(attn_rg_iter) > 0:
            parts.append(f"attn_rg_iter:[{','.join(f'{v:.3f}' for v in attn_rg_iter)}]")
        mlp_rg_iter = getattr(m, "_mlp_router_gate_iter_last_solve", None)
        if mlp_rg_iter is not None and len(mlp_rg_iter) > 0:
            parts.append(f"mlp_rg_iter:[{','.join(f'{v:.3f}' for v in mlp_rg_iter)}]")
        # Per-expert routing weights per iteration (shows expert specialization
        # across iters).  The DEQ solver stored the per-iter (K, E) routing
        # weight tensor on GPU as `_attn_expert_weights_iter_t`; materialize it
        # here at the log-emission site (outside the micro_step hot range) via
        # a single .cpu() sync, and cache the resulting list on the module so
        # repeated reads within the same logging cycle don't re-sync.
        ew_iter = getattr(m, "_attn_expert_weights_iter", None)
        if ew_iter is None:
            ew_iter_t = getattr(m, "_attn_expert_weights_iter_t", None)
            if ew_iter_t is not None:
                ew_iter = ew_iter_t.cpu().tolist()
                m._attn_expert_weights_iter = ew_iter
        if ew_iter is not None and len(ew_iter) > 0:
            # Log std across iterations per expert (high std = specialist, low = uniform)
            ew_arr = np.array(ew_iter)  # (K, E)
            iter_std = ew_arr.std(axis=0)  # per-expert std across iters
            iter_range = ew_arr.max(axis=0) - ew_arr.min(axis=0)  # per-expert range
            parts.append(f"expert_iter_std:[{','.join(f'{v:.4f}' for v in iter_std)}]")
            parts.append(f"expert_iter_range:[{','.join(f'{v:.4f}' for v in iter_range)}]")
        return (" " + " ".join(parts)) if parts else ""

    def format_expert_info(m: nn.Module, *, step: int | None = None, require_step_match: bool = False) -> str:
        parts: list[str] = []
        if hasattr(m, "shared_block"):
            sb_fmt = _unwrap_compiled_module(m.shared_block)
            if getattr(sb_fmt, "chained_stack", None) is not None:
                cvs: list[float] = []
                ents: list[float] = []
                masses: list[float] = []
                mins: list[float] = []
                ema_mins: list[float] = []
                routers_for_conf = list(_iter_unique_routers(sb_fmt))
                for router in routers_for_conf:
                    if hasattr(router, "_materialize_diag_lists"):
                        router._materialize_diag_lists()
                    ok = getattr(router, "_expert_usage", None) is not None
                    if ok and require_step_match and getattr(router, "_diag_step", None) != step:
                        ok = False
                    if not ok:
                        continue
                    usage = list(router._expert_usage)
                    if usage:
                        mins.append(min(usage))
                    ema_min = getattr(router, "_expert_usage_ema_min_share", None)
                    if ema_min is not None:
                        ema_mins.append(float(ema_min))
                    cv = getattr(router, "_expert_balance_cv", None)
                    if cv is not None:
                        cvs.append(float(cv))
                    ent = getattr(router, "_expert_entropy", None)
                    if ent is not None:
                        ents.append(float(ent))
                    mass = getattr(router, "_expert_total_mass", None)
                    if mass is not None:
                        masses.append(float(mass))
                if cvs:
                    parts.append(f"chain_router_cv:{_safe_mean(cvs):.4f}")
                if ema_mins:
                    parts.append(f"chain_ema_expert_min_share:{min(ema_mins):.4f}")
                if mins:
                    parts.append(f"chain_router_batch_min:{min(mins):.4f}")
                if ents:
                    parts.append(f"chain_pertoken_entropy:{_safe_mean(ents):.4f}")
                if masses:
                    parts.append(f"chain_router_mass:{_safe_mean(masses):.4f}")
                parts.extend(_format_router_confidence_parts(
                    routers_for_conf, step=step, require_step_match=require_step_match))
                attn_vals: list[float] = []
                for _, comp in sb_fmt.active_attn_modules():
                    v_t = getattr(comp, "_out_ortho_cos_sim_t", None)
                    if isinstance(v_t, torch.Tensor):
                        attn_vals.append(float(v_t.float().item()))
                    else:
                        v = getattr(comp, "_out_ortho_cos_sim", None)
                        if v is not None:
                            attn_vals.append(float(v))
                mlp_vals: list[float] = []
                for _, comp in sb_fmt.active_mlp_modules():
                    v_t = getattr(comp, "_out_ortho_cos_sim_t", None)
                    if isinstance(v_t, torch.Tensor):
                        mlp_vals.append(float(v_t.float().item()))
                    else:
                        v = getattr(comp, "_out_ortho_cos_sim", None)
                        if v is not None:
                            mlp_vals.append(float(v))
                if attn_vals:
                    parts.append(f"attn_ortho:{_safe_mean(attn_vals):.4f}")
                if mlp_vals:
                    parts.append(f"mlp_ortho:{_safe_mean(mlp_vals):.4f}")
                sg_mean = getattr(sb_fmt, "_shared_gate_mean", None)
                sg_step = getattr(sb_fmt, "_shared_gate_diag_step", None)
                if sg_mean is not None and (not require_step_match or sg_step == step):
                    parts.append(f"shared_gate_mean:{float(sg_mean):.4f}")
                return (" " + " ".join(parts)) if parts else ""
            # Dedup routers by id — pooled router is aliased as attn_router AND
            # mlp_router.  Log pooled 2E usage once, then per-type normalized halves.
            seen_routers: set[int] = set()
            routers_for_conf = list(_iter_unique_routers(sb_fmt))
            for prefix, router in (("attn", getattr(getattr(sb_fmt, "attn", None), "attn_router", None)),
                                   ("mlp", getattr(getattr(sb_fmt, "mlp", None), "mlp_router", None))):
                if router is None or id(router) in seen_routers:
                    continue
                seen_routers.add(id(router))
                if hasattr(router, "_materialize_diag_lists"):
                    router._materialize_diag_lists()
                ok = getattr(router, "_expert_usage", None) is not None
                if ok and require_step_match and getattr(router, "_diag_step", None) != step:
                    ok = False
                if not ok:
                    continue
                usage = router._expert_usage
                usage_str = ",".join(f"{u:.3f}" for u in usage)
                E = m.shared_block.num_experts
                R = E - int(getattr(m.shared_block, "num_shared_experts", 0))
                # Pooled 2R router: split into attn (first R) and mlp (last R).
                # Shared bypass experts are tracked separately via shared_gate_*.
                if len(usage) == 2 * R:
                    attn_half = usage[:R]
                    mlp_half = usage[R:]
                    attn_sum = _safe_sum(attn_half)
                    mlp_sum = _safe_sum(mlp_half)
                    pool_sum = max(attn_sum + mlp_sum, 1e-8)
                    attn_norm = [u / attn_sum for u in attn_half]
                    mlp_norm = [u / mlp_sum for u in mlp_half]
                    pool_norm = [u / pool_sum for u in usage]
                    parts.append(f"attn_usage:[{','.join(f'{u:.3f}' for u in attn_norm)}]")
                    parts.append(f"mlp_usage:[{','.join(f'{u:.3f}' for u in mlp_norm)}]")
                    # Per-slice CV (within-role balance) + pool CV (cross-slice
                    # dominance: attn-vs-mlp). Pool CV is computed on the full
                    # 2R distribution WITHOUT slice renormalization, so it
                    # captures both within-slice imbalance AND any tilt of total
                    # mass toward attn or mlp. Per-slice CVs use the already-
                    # renormalized halves above so they are independent of the
                    # cross-slice tilt.
                    ema_usage = getattr(router, "_expert_usage_ema", None)
                    if ema_usage is not None and len(ema_usage) == 2 * R:
                        ema_attn_half = list(ema_usage[:R])
                        ema_mlp_half = list(ema_usage[R:])
                        ema_attn_sum = _safe_sum(ema_attn_half)
                        ema_mlp_sum = _safe_sum(ema_mlp_half)
                        ema_pool_sum = max(ema_attn_sum + ema_mlp_sum, 1e-8)
                        ema_norms = (
                            ("attn", [u / ema_attn_sum for u in ema_attn_half]),
                            ("mlp", [u / ema_mlp_sum for u in ema_mlp_half]),
                            ("pool", [u / ema_pool_sum for u in ema_usage]),
                        )
                        for _prefix, _norm in ema_norms:
                            parts.append(f"{_prefix}_ema_min:{min(_norm):.4f}")
                            parts.append(f"{_prefix}_ema_cv:{_share_cv(_norm):.4f}")
                    parts.append(f"attn_cv:{_share_cv(attn_norm):.4f}")
                    parts.append(f"mlp_cv:{_share_cv(mlp_norm):.4f}")
                    parts.append(f"pool_cv:{_share_cv(pool_norm):.4f}")
                    parts.append(f"attn_entropy:{_share_entropy(attn_norm):.4f}")
                    parts.append(f"mlp_entropy:{_share_entropy(mlp_norm):.4f}")
                    parts.append(f"pool_entropy:{_share_entropy(pool_norm):.4f}")
                    pertoken_ent = getattr(router, "_expert_entropy", None)
                    if pertoken_ent is not None:
                        parts.append(f"pertoken_entropy:{pertoken_ent:.4f}")
                    total_mass = getattr(router, "_expert_total_mass", None)
                    if total_mass is not None:
                        parts.append(f"router_mass:{float(total_mass):.4f}")
                else:
                    parts.append(f"{prefix}_usage:[{usage_str}]")
                    ent = getattr(router, "_expert_entropy", None)
                    if ent is not None:
                        parts.append(f"{prefix}_entropy:{ent:.4f}")
                    cv = getattr(router, "_expert_balance_cv", None)
                    if cv is not None:
                        parts.append(f"{prefix}_cv:{cv:.4f}")
                    total_mass = getattr(router, "_expert_total_mass", None)
                    if total_mass is not None:
                        parts.append(f"{prefix}_mass:{float(total_mass):.4f}")
            parts.extend(_format_router_confidence_parts(
                routers_for_conf, step=step, require_step_match=require_step_match))
            sg_mean = getattr(m.shared_block, "_shared_gate_mean", None)
            sg_step = getattr(m.shared_block, "_shared_gate_diag_step", None)
            if sg_mean is not None and (not require_step_match or sg_step == step):
                sg_min = getattr(m.shared_block, "_shared_gate_min", None)
                sg_std = getattr(m.shared_block, "_shared_gate_std", None)
                parts.append(f"shared_gate_mean:{float(sg_mean):.4f}")
                if sg_min is not None:
                    parts.append(f"shared_gate_min:{float(sg_min):.4f}")
                if sg_std is not None:
                    parts.append(f"shared_gate_std:{float(sg_std):.4f}")
            for attr, label in [("attn", "attn_ortho"), ("mlp", "mlp_ortho")]:
                comp = getattr(m.shared_block, attr, None)
                # T-opt 15: prefer GPU tensor (no sync until .item() here at log site).
                v_t = getattr(comp, "_out_ortho_cos_sim_t", None)
                if isinstance(v_t, torch.Tensor):
                    v = float(v_t.float().item())
                else:
                    v = getattr(comp, "_out_ortho_cos_sim", None)
                if v is not None:
                    parts.append(f"{label}:{float(v):.4f}")
        if hasattr(m, "mos_head"):
            mos = m.mos_head
            if hasattr(mos, "get_head_orthogonality"):
                # CTP head is opt-in (iter 94 NTP-only baseline); emitting a
                # constant 0.0 log row when CTP is disabled produces misleading
                # log spam.  See EXPERIENCE.md §1 `diagnostic-gate-component-awareness`.
                if getattr(mos, "use_ctp", False):
                    parts.append(f"mos_ctp_ortho:{mos.get_head_orthogonality('ctp'):.4f}")
                parts.append(f"mos_ntp_ortho:{mos.get_head_orthogonality('ntp'):.4f}")
        return (" " + " ".join(parts)) if parts else ""

    # MAIN TRAINING LOOP
    training_time_ms = 0.0
    # Rolling window of per-step latency. Appended once per completed training
    # step and reset alongside `t0` post-val so plotting/eval overhead stays
    # out of the measurement. The 50-step window absorbs CUDA-launch-vs-completion
    # noise without forcing a per-step `torch.cuda.synchronize()`.
    _step_dt_window: deque[float] = deque(maxlen=50)
    _step_t_prev: float | None = None
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state: dict[str, Tensor] | None = None
    if bool(args.ema_enabled):
        ema_state = {name: t.detach().float().cpu().clone() for name, t in base_model.state_dict().items()}
        log0(f"ema:enabled decay:{args.ema_decay:.4f} update_every:{args.ema_update_every}")
    # Initialize step BEFORE the resume block so `ckpt.get("step", step)` has a
    # valid fallback when no checkpoint key exists, and so the loop counter
    # below preserves the resumed value instead of being clobbered.
    step = 0
    resume_path = _resolve_resume_checkpoint_path()
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        if distributed:
            dist.barrier()
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt["model"], strict=True)
        optimizer_states = ckpt.get("optimizers", [])
        if len(optimizer_states) != len(optimizers):
            raise RuntimeError(
                f"checkpoint optimizer count {len(optimizer_states)} != current {len(optimizers)}"
            )
        for opt, opt_state in zip(optimizers, optimizer_states):
            opt.load_state_dict(opt_state)
        loader_state = ckpt.get("train_loader")
        if isinstance(loader_state, dict):
            train_loader.load_state_dict(loader_state)
        sampler_state = ckpt.get("k_sampler")
        if isinstance(sampler_state, dict):
            k_sampler.load_state_dict(sampler_state)
        k_counts_state = ckpt.get("k_sample_counts")
        if isinstance(k_counts_state, dict):
            k_sample_counts.clear()
            k_sample_counts.update({int(k): int(v) for k, v in k_counts_state.items()})
        _beta_bag = [float(v) for v in ckpt.get("beta_bag", [])]
        if ckpt.get("beta_rng_state") is not None:
            _beta_rng.setstate(ckpt["beta_rng_state"])
        _bptt_k_bag = [int(v) for v in ckpt.get("bptt_k_bag", [])]
        if ckpt.get("bptt_k_rng_state") is not None:
            _bptt_k_rng.setstate(ckpt["bptt_k_rng_state"])
        rng_state = ckpt.get("rng", {})
        if isinstance(rng_state, dict):
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"])
            cuda_states = rng_state.get("cuda")
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
        step = int(ckpt.get("step", step))
        training_time_ms = float(ckpt.get("training_time_ms", training_time_ms))
        stop_after_step = ckpt.get("stop_after_step", stop_after_step)
        if stop_after_step is not None:
            stop_after_step = int(stop_after_step)
        swa_state = ckpt.get("swa_state", swa_state)
        swa_count = int(ckpt.get("swa_count", swa_count))
        loaded_ema = ckpt.get("ema_state", None)
        if loaded_ema is not None:
            ema_state = loaded_ema
        _step_dt_window.clear()
        _step_dt_window.extend(float(v) for v in ckpt.get("step_dt_window", []))
        log0(f"checkpoint:loaded step:{step} path:{resume_path}")
        del ckpt
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _step_t_prev = t0
    _fast_val_count = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            # Timing-accounting boundary: stop the train clock BEFORE val so
            # validation time does NOT contribute to training_time_ms / step_avg.
            # `t0` is reset post-val below (after optional plot refresh),
            # so the next training step's clock starts AFTER val is done.
            # Post-training artifact write + K-sweep happen after the main loop
            # break, so they cannot pollute step_avg either.
            torch.cuda.synchronize()
            _step_dt_ms = 1000.0 * (time.perf_counter() - t0)
            training_time_ms += _step_dt_ms
            val_loss, val_bpb = run_validation(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                full_validation=False,
            )
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step) if master_process else ""
            _window_avg = (sum(_step_dt_window) / len(_step_dt_window)) if _step_dt_window else 0.0
            _fast_val_count += 1
            # Always-on FP spectral probe at the saved DEQ FP. rho_F is the
            # principled gate metric (necessary AND sufficient for asymptotic
            # local FP convergence per Hartman-Grobman). lip_ub_T/S/F and
            # fp_bound were removed 2026-05-15 — operator-norm proxies are
            # over-restrictive for non-symmetric J_F and add cost without
            # changing the gate decision.
            if master_process:
                fp_residual_F = _joint_F_residual_at_saved_fp(base_model)
                rho_F = _rho_F_at_saved_fp(
                    base_model,
                    n_iters=int(getattr(args, "fp_rho_power_iters", 8)),
                )
                # iter168 (2026-05-16): sigma_max_F restored as DIAGNOSTIC.
                # NOT a gate, NOT a penalty, NOT a prescription input.
                # Carries operational info (per-step contraction bound,
                # perturbation robustness, basin size) that rho_F alone
                # does not — see _sigma_max_F_at_saved_fp docstring.
                sigma_max_F = _sigma_max_F_at_saved_fp(
                    base_model,
                    n_iters=int(getattr(args, "fp_rho_power_iters", 8)),
                )
            else:
                fp_residual_F = None
                rho_F = None
                sigma_max_F = None
            conv_rel_t = getattr(base_model, "_deq_iter_convergence_rel_t", None)
            fp_residual_rel = (
                float(conv_rel_t.detach().float().item())
                if isinstance(conv_rel_t, torch.Tensor) else None
            )
            rho_str = f" rho_F:{rho_F:.4f}" if rho_F is not None else " rho_F:N/A"
            sigma_str = f" sigma_max_F:{sigma_max_F:.4f}" if sigma_max_F is not None else " sigma_max_F:N/A"
            fp_resid_str = (
                (f" fp_residual_rel:{fp_residual_rel:.6f}" if fp_residual_rel is not None else "")
                + (f" fp_residual_F:{fp_residual_F:.6f}" if fp_residual_F is not None else "")
            )
            # iter171 (2026-05-16): mini K-sweep on the same fast-val subset.
            # When `--fast-val-k-sweep-set "4,128"` is set, re-runs validation
            # at each extra K and logs the per-K val_bpb. Diagnostic ONLY —
            # lets us catch degenerate K-sweep patterns mid-training (e.g.
            # iter170's K=4 < K=16 < K=128 signature) without waiting for
            # end-of-training K-sweep. Cost: ~+65s per val event for
            # {4, 128} (rare K=128 is the dominant cost).
            default_k = int(getattr(args, "deq_k_eval", 16))
            extra_ks = tuple(int(k) for k in (getattr(args, "fast_val_k_sweep_set", ()) or ()))
            fast_k_sweep_pairs: list[tuple[int, float]] = [(default_k, float(val_bpb))]
            for k_extra in extra_ks:
                if k_extra == default_k or k_extra <= 0:
                    continue
                base_model._deq_k_override = k_extra
                try:
                    _, val_bpb_extra = run_validation(
                        args, model, rank, world_size, device, grad_accum_steps,
                        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        full_validation=False,
                    )
                    fast_k_sweep_pairs.append((k_extra, float(val_bpb_extra)))
                finally:
                    base_model._deq_k_override = default_k
            if len(fast_k_sweep_pairs) > 1:
                fast_k_sweep_pairs.sort(key=lambda kv: kv[0])
                sweep_str = " fast_k_sweep:" + ",".join(
                    f"k{k}={v:.4f}" for k, v in fast_k_sweep_pairs
                )
            else:
                sweep_str = ""
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"val_mode:fast "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms "
                f"step_avg_w50:{_window_avg:.2f}ms"
                f"{rho_str}{sigma_str}{fp_resid_str}"
                f"{sweep_str}"
                f"{deq_info}{expert_info}"
                f"{format_k_jitter_info()}"
            )
            if _update_experiment_plots is not None:
                _update_experiment_plots(logfile, enabled=master_process and getattr(args, "auto_plot_on_val", False))
            _clear_saved_fp_probe_tensors(base_model)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step_t_prev = t0

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_wallclock_ms is not None and max_wallclock_ms > 0:
            time_frac = min(max(elapsed_ms / max_wallclock_ms, 0.0), 1.0)
        else:
            time_frac = min(max(step / max(int(args.iterations), 1), 0.0), 1.0)

        # Compile-wrapper-safe: always write through unwrapped module.
        sb = _unwrap_compiled_module(base_model.shared_block)
        # Single shared regularizer ramp: 0 -> 1 linearly over the first
        # `regularizer_warmup_frac` fraction of training. With warmup=0 this
        # preserves the promoted iter142b step-0 behavior; with warmup>0 all
        # router/MoS/diversity regularizers share the same cold-start schedule.
        reg_warm = float(base_model.regularizer_warmup_frac)
        reg_scale = (
            min(max(time_frac / max(reg_warm, 1e-8), 0.0), 1.0) if reg_warm > 0 else 1.0
        )
        # iter145r EMA-anchored family — annealed identically; registry drives loop.
        for _name, _ in ROUTER_EMA_LOSS_TERMS:
            target = float(getattr(base_model, f"_router_ema_{_name}_coef_target"))
            setattr(base_model, f"router_ema_{_name}_coef", target * reg_scale)
        ent_target = float(base_model._router_pertoken_entropy_coef_target)
        sched_routers = list(_iter_unique_routers(sb))
        if ent_target > 0.0:
            for router_sched in sched_routers:
                router_sched.entropy_coef = ent_target * reg_scale
        ucb_target = float(getattr(base_model, "_router_dirichlet_ucb_beta_target", 0.0))
        if ucb_target > 0.0:
            ucb_beta = ucb_target * max(1.0 - time_frac, 0.0)
            for router_sched in sched_routers:
                router_sched.dirichlet_ucb_beta = ucb_beta
        if bool(getattr(base_model, "_use_entmax_routing", False)):
            blend_delay = float(base_model._entmax_blend_warmup_delay_frac)
            if time_frac < blend_delay:
                blend_anneal = 0.0
            else:
                blend_anneal = min(max((time_frac - blend_delay) / max(1.0 - blend_delay, 1e-8), 0.0), 1.0)
            for router_sched in sched_routers:
                router_sched.entmax_blend_anneal = blend_anneal
        # Per-token expert-output diversity ramp shares `reg_scale`.
        if base_model._expert_diversity_coef_target > 0.0:
            base_model._expert_diversity_coef_scale = reg_scale
        else:
            base_model._expert_diversity_coef_scale = 0.0

        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        if hasattr(sb, "_deq_x0_recon_error_last_bwd"):
            sb._deq_x0_recon_error_last_bwd = None
        if hasattr(sb, "_deq_recon_error_last_bwd"):
            sb._deq_recon_error_last_bwd = None
        if hasattr(sb, "_deq_fp_travel_last_fwd"):
            sb._deq_fp_travel_last_fwd = None

        train_loss = torch.zeros((), device=device)
        next_step = step + 1
        will_log_train = (
            args.train_log_every > 0
            and (next_step <= 10 or next_step % args.train_log_every == 0 or stop_after_step is not None)
        )
        base_model._deq_k_override = deq_k_for_step(next_step)
        # Scalar β jitter only when Parcae is disabled (Parcae supersedes scalar β).
        if not base_model.use_parcae:
            base_model.deq_beta = deq_beta_for_step(next_step)
        # Iter 85: stochastic TBPTT — sample deq_bptt_k per step from {2,3,4}.
        base_model.deq_bptt_k = deq_bptt_k_for_step(next_step)
        # iter161-QAT-late (2026-05-16): notify the global QAT state of the
        # current step. CastedLinear.forward consults `_QAT_LATE_STATE.active()`
        # to decide whether to apply the deterministic int6-SDCLIP STE on
        # weights (only fires when current_step >= start_step).
        _QAT_LATE_STATE.set_step(next_step)

        # Refinement gating: enable after ramp_frac of wallclock
        ramp_frac = float(getattr(args, "num_refinements_ramp_frac", 0.85))
        base_model.num_refinements = 0 if time_frac < ramp_frac else int(args.num_refinements)
        if base_model.num_refinements > 0 and ramp_frac > 0:
            prog = min(max((time_frac - ramp_frac) / max(1.0 - ramp_frac, 0.01), 0.0), 1.0)
            base_model._refine_mix_alpha = 0.5 * float(prog)
        else:
            base_model._refine_mix_alpha = 0.5 if base_model.num_refinements > 0 else 0.0

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                diag_enabled = master_process and will_log_train and micro_step == grad_accum_steps - 1
                # Compensates the global `loss * grad_scale = loss / grad_accum_steps`
                # for last-only auxiliaries (expert-output diversity, etc.). On
                # other micro-steps these aux contributions are zero anyway, so
                # the scale is a no-op there.
                base_model._aux_grad_accum_scale = (
                    float(grad_accum_steps) if micro_step == grad_accum_steps - 1 else 1.0
                )
                # Per-token expert-output diversity: every-N + last-micro-step.
                base_model._expert_diversity_aux_enabled = False
                base_model._expert_diversity_loss = None
                base_model._expert_diversity_token_start = 0
                _div_every = int(args.expert_diversity_every)
                if (args.expert_output_diversity_coef > 0.0
                        and micro_step == grad_accum_steps - 1
                        and _div_every > 0
                        and (next_step % _div_every == 0)):
                    base_model._expert_diversity_aux_enabled = True
                    base_model._expert_diversity_token_start = _deterministic_token_window_start(
                        int(args.seed), int(next_step), int(x.shape[1]), int(args.expert_diversity_max_tokens)
                    )
                with router_diagnostics(diag_enabled, step_tag=next_step if diag_enabled else None):
                    loss = model(x, y)

                # Lyapunov + denoising auxiliaries: hard-gated on coefs > 0.
                # The Lyapunov path directly penalizes finite expansion of
                # T_theta near the saved fixed point; it is low-cadence because
                # it costs two extra shared-block forwards when enabled.
                lyap_coef = float(base_model.lyapunov_coef)
                dn_coef = float(args.denoising_coef)
                if (lyap_coef > 0.0 or dn_coef > 0.0) and micro_step == grad_accum_steps - 1:
                    # Lyapunov / denoising share the same regularizer ramp.
                    lyap_scale = float(reg_scale)
                    z_star = getattr(base_model, '_lyapunov_z_star', None)
                    x0_lyap = getattr(base_model, '_lyapunov_x0', None)
                    # One-shot warning when the flag is on but the saved FP
                    # is missing (e.g., a future refactor moves the
                    # assignment site). CLAUDE.md "Diagnostic gates follow
                    # flags": if the flag is on, absence of effect is a
                    # silent failure.
                    if (lyap_coef > 0.0 and lyap_scale > 0.0
                            and (z_star is None or x0_lyap is None)
                            and not getattr(base_model, '_lyapunov_missing_warned', False)):
                        log0(
                            "lyapunov_warning: lyapunov_coef>0 but _lyapunov_z_star/_lyapunov_x0 "
                            "are unset; penalty becomes a no-op until populated"
                        )
                        base_model._lyapunov_missing_warned = True
                    if z_star is not None and x0_lyap is not None and lyap_scale > 0.0:
                        aux_b_bar = base_model._parcae_b_bar() if base_model.use_parcae else None
                        lyap_every = max(1, int(getattr(base_model, "lyapunov_every", 16)))
                        if lyap_coef > 0.0 and (next_step % lyap_every == 0):
                            max_tokens = max(1, int(getattr(base_model, "lyapunov_max_tokens", 64)))
                            t_lyap = min(max_tokens, int(z_star.shape[1]))
                            start_lyap = _deterministic_token_window_start(
                                int(args.seed) + 17,
                                int(next_step),
                                int(z_star.shape[1]),
                                t_lyap,
                            )
                            z_base = z_star.detach()[:, start_lyap:start_lyap + t_lyap].contiguous()
                            x0_base = x0_lyap[:, start_lyap:start_lyap + t_lyap].contiguous()
                            # Hutchinson-style probe: with `eps_unit` of unit RMS in
                            # high-D, `expansion` has expectation ‖J_M‖_F/√D, NOT the
                            # operator norm ‖J_M‖_2. So this is a Frobenius/√D
                            # proxy on whichever Jacobian `lyapunov_target` selects
                            # (T_θ, S, or F) — soft contraction pressure, not a tight
                            # Lipschitz cert. The principled FP-convergence gate is
                            # `rho_F` (spectral radius via power iteration on J_F);
                            # operator-norm probes `lip_ub_T/S/F` were removed
                            # 2026-05-15 as over-restrictive proxies.
                            # RMS form (vs unit-L2) keeps per-element
                            # magnitudes ~O(1) under bf16 finite differencing.
                            # Seeded per-step generator: deterministic across reruns
                            # at fixed (args.seed, next_step). Matches the determinism
                            # of `_deterministic_token_window_start` above.
                            lyap_gen = torch.Generator(device=z_base.device).manual_seed(
                                int(args.seed) * 7919 + int(next_step))
                            eps_dir = torch.randn(z_base.shape, dtype=z_base.dtype,
                                                  device=z_base.device, generator=lyap_gen)
                            eps_unit = eps_dir / eps_dir.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir.dtype)
                            # 1e-2 sits in the bf16 forward-difference sweet spot:
                            # large enough to clear the ~2e-3 bf16 noise floor,
                            # small enough that O(eps²) curvature error stays below
                            # the linear directional-derivative signal.
                            eps_step = 1e-2
                            # Detach B̄ on the FD probe: keeps second-order
                            # curvature out of the Parcae B̄ training signal
                            # (denoising below intentionally keeps live B̄).
                            # Ā is NOT detached in S/F branches: it appears
                            # only in outside-linear-blend coefficients
                            # (no second-order FD curvature concern), so
                            # gradients to Ā are first-order and exactly
                            # what should pressure damping when expansion>γ.
                            aux_b_bar_d = aux_b_bar.detach() if aux_b_bar is not None else None
                            # iter155 (corrected): select which Jacobian's
                            # directional derivative we penalize.
                            #   transition_T: diff = sb(z+εu) − sb(z)  (≈ J_T·u)
                            #   iteration_S:  diff = Ā·(εu) + (1−Ā)·(sb(z+εu) − sb(z))
                            #     ⇒ J_S·u where J_S = Ā·I + (1−Ā)·J_T (single
                            #     -state convex blend; advisory surrogate)
                            #   iteration_F:  joint perturbation on (y, z) with
                            #     y* = z* = z_base; diff is the directional
                            #     derivative of the two-state cycle map F.
                            #     ⇒ ‖J_F · (u_y, u_z)‖ — the gate-aligned object.
                            # Soft directional proxy for the spectral norm of
                            # the chosen J_M; the principled gate is `rho_F`
                            # (spectral radius). This FD penalty is a refuted
                            # lower-bound proxy retained only as an ablation.
                            lyap_target = str(getattr(base_model, "lyapunov_target", "transition_T"))
                            if lyap_target == "iteration_F" and base_model.use_parcae:
                                # Two-state cycle: y' = Ā·y + β·T(z),
                                #                  z' = Ā·z + β·T(y').
                                # At the saved FP, y* = z* = z_base (so the
                                # initial joint state is duplicated).  Cycle
                                # algebra delegates to `_parcae_cycle_F`.
                                # Estimator is `random_fd` only after the
                                # `power_jvp_F` branch was removed 2026-05-15
                                # along with the underlying lip_ub machinery.
                                a_bar_full = base_model._parcae_a_bar()  # NOT detached
                                a_bar_d = a_bar_full.view(*([1] * (z_base.ndim - 1)), -1).to(dtype=z_base.dtype)
                                # random_fd: independent random directions for u_y, u_z.
                                eps_dir_y = torch.randn(
                                    z_base.shape, dtype=z_base.dtype,
                                    device=z_base.device, generator=lyap_gen,
                                )
                                eps_unit_y = eps_dir_y / eps_dir_y.float().pow(2).mean().sqrt().clamp(min=1e-8).to(dtype=eps_dir_y.dtype)
                                # F at base = F(z_base, z_base); F at perturbed
                                # = F(z_base + ε·u_y, z_base + ε·u_z).  Cycle
                                # algebra delegates to `_parcae_cycle_F` (the
                                # single source of truth used by `rho_F` and
                                # `fp_residual_F` — no future drift).
                                y_new_base, z_new_base = _parcae_cycle_F(
                                    z_base, z_base, x0_base, aux_b_bar_d, a_bar_d, sb,
                                )
                                y_pert_init = z_base + eps_step * eps_unit_y
                                z_pert = z_base + eps_step * eps_unit
                                y_new_pert, z_new_pert = _parcae_cycle_F(
                                    y_pert_init, z_pert, x0_base, aux_b_bar_d, a_bar_d, sb,
                                )
                                # Directional derivative of F = concat(y', z').
                                diff_y = y_new_pert - y_new_base
                                diff_z = z_new_pert - z_new_base
                                # Joint RMS over both halves (equivalent to
                                # concatenating along last dim and RMS-reducing).
                                diff_sq_sum = diff_y.float().pow(2).sum() + diff_z.float().pow(2).sum()
                                diff_count = diff_y.numel() + diff_z.numel()
                                expansion = (diff_sq_sum / diff_count).sqrt() / float(eps_step)
                            elif lyap_target == "iteration_S" and base_model.use_parcae:
                                a_bar_full = base_model._parcae_a_bar()  # NOT detached
                                a_bar_d = a_bar_full.view(*([1] * (z_base.ndim - 1)), -1).to(dtype=z_base.dtype)
                                u_base = sb(z_base, x0_base, aux_b_bar_d)
                                u_pert = sb(z_base + eps_step * eps_unit, x0_base, aux_b_bar_d)
                                diff = a_bar_d * (eps_step * eps_unit) + (1.0 - a_bar_d) * (u_pert - u_base)
                                expansion = diff.float().pow(2).mean().sqrt() / float(eps_step)
                            else:
                                u_base = sb(z_base, x0_base, aux_b_bar_d)
                                u_pert = sb(z_base + eps_step * eps_unit, x0_base, aux_b_bar_d)
                                diff = u_pert - u_base
                                expansion = diff.float().pow(2).mean().sqrt() / float(eps_step)
                            lyap_loss = torch.relu(expansion - float(base_model.lyapunov_gamma)).pow(2)
                            # Multiply by grad_accum_steps so the per-optimizer-step
                            # gradient contribution matches the documented coef
                            # (last-only aux otherwise gets attenuated by 1/N
                            # via the global grad_scale division below).
                            loss = loss + grad_accum_steps * lyap_scale * lyap_coef * lyap_loss.to(dtype=loss.dtype)
                        if dn_coef > 0.0:
                            dn_std = float(args.denoising_noise_std)
                            eps_noise = torch.randn_like(z_star) * dn_std
                            z_noisy = z_star.detach() + eps_noise
                            f_noisy = sb(z_noisy, x0_lyap, aux_b_bar)
                            dn_loss = (f_noisy - z_star.detach()).float().pow(2).mean()
                            loss = loss + grad_accum_steps * lyap_scale * dn_coef * dn_loss

            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Effective warmup = min(absolute cap, frac × iterations). Long-run prod
        # uses the legacy 800-step cap unchanged; short comparison runs auto-scale.
        _abs_cap = int(args.muon_momentum_warmup_steps) if args.muon_momentum_warmup_steps else 0
        _adaptive = max(1, int(float(getattr(args, "muon_momentum_warmup_frac", 0.8)) * float(args.iterations)))
        _mom_steps = min(_abs_cap, _adaptive) if _abs_cap > 0 else _adaptive
        frac = min(step / max(_mom_steps, 1), 1.0)
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            _preclip_t = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        else:
            _preclip_t = None
        for opt in optimizers:
            opt.step()

        sb_post = _unwrap_compiled_module(base_model.shared_block)
        for router in _iter_unique_routers(sb_post):
            if hasattr(router, "usage_ema_update"):
                router.usage_ema_update(distributed=distributed)

        if args.router_bias_update:
            bias_lr = float(args.router_bias_lr)
            for r in _iter_unique_routers(sb):
                r.bias_update(lr=bias_lr, clip=float(args.router_bias_clip), distributed=distributed)
        zero_grad_all()

        if ema_state is not None and args.ema_update_every > 0 and (step % args.ema_update_every == 0):
            update_ema_state_(ema_state, base_model.state_dict(), decay=args.ema_decay)

        step += 1
        # Per-step latency for step_avg_w50. No sync — the 50-step window
        # averages out CUDA-launch-vs-completion noise; adding a per-step
        # synchronize() would cost ~1-2% throughput.
        _step_now = time.perf_counter()
        if _step_t_prev is not None:
            _step_dt_window.append(1000.0 * (_step_now - _step_t_prev))
        _step_t_prev = _step_now
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().float().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().float().cpu()
                swa_count += 1

        if will_log_train:
            ntp_t = getattr(base_model, '_ntp_loss_t', None)
            ctp_t = getattr(base_model, '_ctp_loss_t', None)
            ntp = float(ntp_t.detach().float().item()) if isinstance(ntp_t, torch.Tensor) else 0.0
            ctp = float(ctp_t.detach().float().item()) if isinstance(ctp_t, torch.Tensor) else 0.0
            def _log_tensor_attr(name: str) -> float:
                t = getattr(base_model, name, None)
                return float(t.detach().float().item()) if isinstance(t, torch.Tensor) else 0.0

            parcae_info = ""
            if getattr(base_model, "use_parcae", False):
                parcae_diag = base_model.parcae_diagnostics()

                def _log_parcae(name: str) -> float:
                    t = parcae_diag.get(name)
                    return float(t.detach().float().item()) if isinstance(t, torch.Tensor) else 0.0

                parcae_info = (
                    f"parcae_a_bar_min:{_log_parcae('parcae_a_bar_min'):.6f} "
                    f"parcae_a_bar_mean:{_log_parcae('parcae_a_bar_mean'):.6f} "
                    f"parcae_a_bar_max:{_log_parcae('parcae_a_bar_max'):.6f} "
                    f"parcae_a_bar_core_max:{_log_parcae('parcae_a_bar_core_max'):.6f} "
                    f"parcae_beta_mean:{_log_parcae('parcae_beta_mean'):.6f} "
                    f"parcae_beta_max:{_log_parcae('parcae_beta_max'):.6f} "
                    f"parcae_b_bar_mean:{_log_parcae('parcae_b_bar_mean'):.6f} "
                    f"parcae_b_bar_max:{_log_parcae('parcae_b_bar_max'):.6f} "
                    f"parcae_delta_mean:{_log_parcae('parcae_delta_mean'):.6f} "
                    f"parcae_delta_max:{_log_parcae('parcae_delta_max'):.6f} "
                    f"parcae_recon_amp_log10:{_log_parcae('parcae_recon_amp_log10'):.6f} "
                )
            # iter145r EMA-anchored family — log columns driven by the registry
            # so adding a future term is a one-line change in ROUTER_EMA_LOSS_TERMS.
            ema_loss_cols = " ".join(
                f"router_ema_{name}_loss:{_log_tensor_attr(f'_router_ema_{name}_loss_t'):.6f}"
                for name, _ in ROUTER_EMA_LOSS_TERMS
            )
            ema_coef_cols = " ".join(
                f"router_ema_{name}_coef_eff:{_log_tensor_attr(f'_router_ema_{name}_coef_eff_t'):.6g}"
                for name, _ in ROUTER_EMA_LOSS_TERMS
            )
            loss_info = (
                f"router_cv_loss:{_log_tensor_attr('_router_cv_loss_t'):.6f} "
                f"router_pertoken_entropy_loss:{_log_tensor_attr('_router_pertoken_entropy_loss_t'):.6f} "
                f"{ema_loss_cols} "
                f"mos_cv_loss:{_log_tensor_attr('_mos_cv_loss_t'):.6f} "
                f"expert_diversity_loss:{_log_tensor_attr('_expert_diversity_loss_t'):.6f} "
                f"mos_diversity_loss:{_log_tensor_attr('_mos_diversity_loss_t'):.6f} "
                f"router_reg_loss:{_log_tensor_attr('_router_reg_loss_t'):.6f} "
                f"router_pertoken_entropy_coef_eff:{_log_tensor_attr('_router_pertoken_entropy_coef_eff_t'):.6g} "
                f"{ema_coef_cols} "
                f"expert_diversity_coef_eff:{_log_tensor_attr('_expert_diversity_coef_eff_t'):.6g} "
                f"mos_diversity_coef_eff:{_log_tensor_attr('_mos_diversity_coef_eff_t'):.6g} "
                f"consistency_anchor_loss:{_log_tensor_attr('_consistency_anchor_loss_t'):.6f} "
            )
            base_model._deq_x0_recon_error = getattr(base_model.shared_block, "_deq_x0_recon_error_last_bwd", None)
            base_model._deq_recon_error = getattr(base_model.shared_block, "_deq_recon_error_last_bwd", None)
            base_model._deq_fp_travel = getattr(base_model.shared_block, "_deq_fp_travel_last_fwd", None)
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step, require_step_match=True) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ntp_loss:{ntp:.4f} ctp_loss:{ctp:.4f} "
                f"{loss_info}"
                f"{parcae_info}"
                f"grad_norm:{float(_preclip_t.item()) if _preclip_t is not None else 0.0:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                f"{deq_info}{expert_info}"
            )

        checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
        if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
            _save_training_checkpoint(step, approx_training_time_ms, stop_after_step)
            if distributed:
                dist.barrier()

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_vram_mb:{torch.cuda.max_memory_allocated() // 1024 // 1024}")

    # Apply SWA
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {name: (tensor / swa_count).to(dtype=current_state[name].dtype)
                     for name, tensor in swa_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    # Apply EMA
    if ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        ema_cast = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(ema_cast, strict=True)
        del ema_state, ema_cast, current_state

    # Free optimizer states + training buffers before post-training eval.
    # Muon momentum + AdamW m/v can hold 2-3× model params in VRAM; freeing
    # them prevents OOM during the 64-head roundtrip validation forward pass.
    del optimizers, optimizer_tok, optimizer_muon, optimizer_scalar
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Keep DDP alive through post-training validation so run_validation and
    # sliding_window_validation can shard the val set across both ranks.
    if distributed:
        dist.barrier()

    # Master-only: quantization, compression, artifact save, decompression,
    # and reload of dequantized weights.  These steps are inherently serial.
    base_model.train(False)
    meta_path: Path | None = None
    val_bpb_q = 0.0
    # Budget-violation flag must be visible on EVERY rank so the raise below
    # executes collectively — raising only on master would leave other ranks
    # blocked forever on a later broadcast / barrier.
    _budget_violated = torch.zeros(1, device=device)
    _budget_info = ""
    if master_process:
        code_bytes = len(code.encode('utf-8'))
        log0(f"Code size: {code_bytes} bytes")
        sd = base_model.state_dict()
        # Save full-precision (bf16) weights before quantization for diagnostic
        # re-evaluation (K-sweep at new K values, gate analysis) without retraining.
        weights_dir = Path("experiments/weights/current")
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sd, weights_dir / "model_full.pt")
        log0(f"saved full-precision weights: {weights_dir / 'model_full.pt'}")
        # Single source of truth for the submission artifact format —
        # `experiments/test_submission.py` calls the same helper so its
        # quant_categories / serialization keys / compressor settings cannot
        # drift away from what is actually scored.
        artifact = encode_scored_artifact(
            sd,
            use_grouped_artifact_compression=bool(getattr(args, "use_grouped_artifact_compression", False)),
            log_fn=log0,
        )
        compressed, qsd, meta = artifact.compressed, artifact.qsd, artifact.meta
        artifact_bytes = int(artifact.compressed_bytes)
        log0(f"artifact_bytes:{artifact_bytes} codec:{artifact.codec} compressor:{_COMPRESSOR}")

        # Parameter Golf HARD budget: total = code + compressed model ≤ 16,000,000 bytes.
        # Failing this means the artifact violates the competition constraint; we
        # fail early (before the expensive roundtrip eval) so a bad run doesn't
        # waste compute pretending to succeed.
        total_bytes = code_bytes + artifact_bytes
        log0(f"total_bytes:{total_bytes} (code:{code_bytes} + artifact:{artifact_bytes}) budget:16000000")
        if total_bytes > 16_000_000:
            _budget_violated.fill_(1.0)
            _budget_info = (
                f"Artifact budget violated: total_bytes={total_bytes} > 16,000,000 "
                f"(code={code_bytes} + artifact={artifact_bytes})"
            )
    # Broadcast violation flag from rank 0 so every rank raises collectively,
    # avoiding the deadlock where master raises but workers block on broadcasts
    # below.  dist.broadcast uses the rank-0 value on all ranks.
    if distributed:
        dist.broadcast(_budget_violated, src=0)
    if float(_budget_violated.item()) > 0.5:
        # Tear down DDP cleanly on ALL ranks before raising.
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        raise RuntimeError(_budget_info or "Artifact budget violated on rank 0 (see master log for details)")

    # Master continues: persist the int6 artifact, meta.json, and roundtrip
    # the quantized weights back into base_model before the DDP broadcast.
    # This MUST live at the master_process indent (not inside the violation
    # check) or the run silently evaluates the bf16 model.
    if master_process:
        weights_dir = Path("experiments/weights/current")
        weights_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = weights_dir / "model.int6.ptz"
        with open(artifact_path, "wb") as f:
            f.write(compressed)

        meta_path = weights_dir / "meta.json"
        with open(meta_path, "w") as f:
            # run_valid stays false until full final val_bpb is finalized at
            # the end of main(). If the run aborts between here and there,
            # stale run_valid=false keeps
            # update_results.sh --promote from picking it up.
            # Write both step/steps and commit/git_commit for schema compat.
            _git_commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            json.dump({
                "val_bpb": 0.0,
                "artifact_bytes": artifact_bytes,
                "step": step,
                "steps": step,
                "commit": _git_commit,
                "git_commit": _git_commit,
                "run_valid": False,
                "status": "artifact_written",
                "config_profile": str(args.config_profile),
                "eval_profile": str(args.eval_profile),
                "diagnostic_gate_policy": str(args.diagnostic_gate_policy),
                "artifact_codec": artifact.codec,
            }, f)

        log0("roundtrip_verification:start")
        deq_sd = load_int6_artifact(compressed, sd)
        base_model.load_state_dict(deq_sd, strict=True)

    # PROFILE_SKIP_KSWEEP=1 (set by experiments/profile_train.py) tells ALL
    # ranks to exit cleanly here, BEFORE the int6 weight broadcast +
    # roundtrip + K-sweep section. The K-sweep's OOM-prone Hutchinson/Lipschitz
    # probes (task #102) drag profile runs by 10+ min without contributing
    # any train-step ops to the trace. This check sits OUTSIDE the
    # `if master_process:` block at L4656-4693 — so all ranks reach it and
    # exit together (no NCCL hang from rank-divergent exits at the next
    # collective). Sys.exit raises SystemExit, which the
    # `experiments/profile_train.py` `try/except SystemExit: pass` catches
    # cleanly — then the profile context exits and the trace + ops table
    # are written.
    if os.environ.get("PROFILE_SKIP_KSWEEP") == "1":
        if master_process:
            log0("PROFILE_SKIP_KSWEEP=1 — all ranks exiting before roundtrip + K-sweep")
        if distributed:
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(0)

    # Disable dynamo on ALL ranks before the post-train roundtrip + K-sweep.
    # Hoisted out of the master_process block (was at L4382) because asymmetric
    # disable causes non-master ranks to recompile during the K-sweep, risking
    # NCCL desync if recompile latency varies across ranks (coderabbit Major #2).
    # T-opt 17 rationale: torch.compile after load_state_dict crashes silently
    # during inductor compilation; eager is reliable and fast enough for the
    # one-time roundtrip diagnostic with B=64 val batches.
    torch._dynamo.reset()
    torch._dynamo.config.disable = True

    # Broadcast the dequantized weights from master to all ranks so every
    # rank runs eval on the same int6-roundtripped model.  Parameters and
    # buffers iterate in deterministic registration order, so we can pair
    # broadcasts without extra synchronization.
    if distributed:
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)
        for b in base_model.buffers():
            dist.broadcast(b.data, src=0)
        dist.barrier()

    # T-opt 17: run roundtrip + K-sweep in eager mode. torch.compile after
    # load_state_dict crashes silently during inductor compilation (both
    # stale-guard reuse and compile-fresh approaches fail). Eager mode is
    # reliable and fast enough with B=64 val batches for a one-time diagnostic.

    # All ranks: roundtrip validation sharded across the val set via DDP.
    base_m_for_roundtrip = base_model
    base_m_for_roundtrip._deq_k_override = int(args.deq_k_eval)
    val_loss_q, val_bpb_q = run_validation(
        args, base_m_for_roundtrip, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        full_validation=False,  # fast eval for roundtrip sanity check (eager mode is 4× slower)
    )
    log0(f"roundtrip_verification:done val_loss:{val_loss_q:.4f} val_bpb:{val_bpb_q:.6f}")


    # DEQ fixed-point K-sweep: verify val_bpb improves (or plateaus) as K grows.
    # A valid DEQ should converge to a fixed point — more solver iterations = better
    # or equal quality, never worse.  Non-monotone behaviour indicates the model
    # is exploiting a specific iteration count rather than a true fixed point.
    # Runs DDP-parallel across ranks for a ~2x speedup on 2 GPUs.
    #
    # K-sweep reports rho_F (spectral radius, principled FP-convergence
    # gate per Hartman-Grobman) + fp_residual_F (joint two-state-cycle
    # residual) + iter_conv_rel (z-only step residual). lip_ub_T/S/F and
    # fp_bound were removed 2026-05-15 — operator-norm proxies are
    # over-restrictive for non-symmetric J_F.
    log0("k_sweep:start")
    # T-opt 15: Reset dynamo before K-sweep to prevent recompilation storm.
    # Different K values change iteration counts, triggering dynamo guards
    # that cause recompile_limit hits → stall one rank → NCCL timeout.
    torch._dynamo.reset()
    # Eval profile owns K-sweep scope.  The diagnostic profile preserves the
    # historical prime-rich matrix; submission/debug profiles reduce redundant
    # probes without changing training behavior.
    k_sweep_values = _resolve_k_sweep_values(args)
    k_sweep_results: dict[int, float] = {}
    # rho_F = |lambda_max(J_F)| via straight power iteration on J_F.
    # Necessary AND sufficient for asymptotic local FP convergence
    # (Hartman-Grobman). Architecture-agnostic: same condition applies
    # to any iteration map.
    k_sweep_rho_F: dict[int, float] = {}
    # iter168 (2026-05-16): sigma_max(J_F) operator-norm DIAGNOSTIC restored
    # alongside rho_F. NOT a gate; captures per-step contraction bound,
    # robustness to input/weight perturbations, and basin-of-attraction
    # size — properties rho_F alone does not capture.
    k_sweep_sigma_max_F: dict[int, float] = {}
    # Joint two-state-cycle residual at the saved FP, paired with rho_F
    # as a per-K convergence-quality diagnostic.
    k_sweep_fp_residual_F: dict[int, float] = {}

    # iter 100b user directive (PERMANENT 2026-04-27): emit a structured
    # per-K table of expert health + Lipschitz + sparsity + shared_gate so
    # routing trajectory across K can be plotted/compared per H-claim.
    # Header is emitted before the loop; each K appends one row.
    def _collect_eval_kdiag(base_m) -> dict[str, float]:
        """Per-K expert/sparsity/shared-gate diagnostics from the most recent
        eval forward pass. Reads from the pooled router (single source of
        truth post iter 35 consolidation): SharedBlock has `self.router` and
        per-component aliases on `self.attn.attn_router` / `self.mlp.mlp_router`
        all pointing at the same instance. Per-slice CV/min/entropy split
        the pooled usage by the (R, R) layout; ortho is per-component."""
        sb_raw = getattr(base_m, "shared_block", None)
        if sb_raw is None:
            return {}
        sb = _unwrap_compiled_module(sb_raw)
        out: dict[str, float] = {}
        if getattr(sb, "chained_stack", None) is not None:
            routers_for_conf: list[SoftDenseRouter] = []
            cvs: list[float] = []
            mins: list[float] = []
            ents: list[float] = []
            for router in _iter_unique_routers(sb):
                routers_for_conf.append(router)
                if hasattr(router, "_materialize_diag_lists"):
                    router._materialize_diag_lists()
                usage = getattr(router, "_expert_usage", None)
                if usage:
                    mins.append(min(usage))
                cv = getattr(router, "_expert_balance_cv", None)
                if cv is not None:
                    cvs.append(float(cv))
                ent = getattr(router, "_expert_entropy", None)
                if ent is not None:
                    ents.append(float(ent))
            if cvs:
                out["pool_cv"] = _safe_mean(cvs)
                out["attn_cv"] = _safe_mean(cvs)
                out["mlp_cv"] = _safe_mean(cvs)
            if mins:
                out["attn_min"] = min(mins)
                out["mlp_min"] = min(mins)
            if ents:
                out["pertoken_ent"] = _safe_mean(ents)
                out["pool_ent"] = _safe_mean(ents)
            for prefix, iterator in (("attn", sb.active_attn_modules), ("mlp", sb.active_mlp_modules)):
                vals: list[float] = []
                for _, comp in iterator():
                    v_t = getattr(comp, "_out_ortho_cos_sim_t", None)
                    if isinstance(v_t, torch.Tensor):
                        vals.append(float(v_t.float().item()))
                    else:
                        v = getattr(comp, "_out_ortho_cos_sim", None)
                        if v is not None:
                            vals.append(float(v))
                if vals:
                    out[f"{prefix}_ortho"] = _safe_mean(vals)
            sg = getattr(sb, "_shared_gate_mean", None)
            if sg is not None:
                out["shared_gate"] = float(sg) if not isinstance(sg, torch.Tensor) else float(sg.detach().float().item())
            out.update(_router_confidence_stats(routers_for_conf))
            return out
        R_total = int(getattr(sb, "num_experts", 0))
        R = R_total - int(getattr(sb, "num_shared_experts", 0))
        # Prefer the pooled router on the SharedBlock; fall back to per-comp aliases.
        router = getattr(sb, "router", None)
        if router is None:
            router = getattr(getattr(sb, "attn", None), "attn_router", None)
        routers_for_conf = [router] if router is not None else []
        if router is not None and hasattr(router, "_materialize_diag_lists"):
            router._materialize_diag_lists()
        usage = getattr(router, "_expert_usage", None) if router else None
        if usage and len(usage) >= 2 * R and R > 0:
            attn_half, mlp_half = usage[:R], usage[R:]
            for label, half in [("attn", attn_half), ("mlp", mlp_half)]:
                half_sum = _safe_sum(half)
                norm = [u / half_sum for u in half]
                mu = _safe_mean(norm)
                var = sum((x - mu) ** 2 for x in norm) / len(norm)
                out[f"{label}_cv"] = (var ** 0.5) / mu
                out[f"{label}_min"] = min(norm)
            usage_sum = _safe_sum(usage)
            pool_norm = [u / usage_sum for u in usage]
            pmu = _safe_mean(pool_norm)
            pvar = sum((x - pmu) ** 2 for x in pool_norm) / len(pool_norm)
            out["pool_cv"] = (pvar ** 0.5) / pmu
            out["pool_ent"] = -sum(x * math.log(x + 1e-8) for x in pool_norm if x > 0.0)
        pertoken = getattr(router, "_expert_entropy", None) if router else None
        if pertoken is not None:
            out["pertoken_ent"] = float(pertoken)
        for prefix in ("attn", "mlp"):
            comp = getattr(sb, prefix, None)
            if comp is None:
                continue
            v = _diag_scalar(getattr(comp, "_out_ortho_cos_sim_t", None))
            if v is None:
                v = _diag_scalar(getattr(comp, "_out_ortho_cos_sim", None))
            if v is not None:
                out[f"{prefix}_ortho"] = v
        sg = _diag_scalar(getattr(sb, "_shared_gate_mean", None))
        if sg is not None:
            out["shared_gate"] = sg
        out.update(_router_confidence_stats(routers_for_conf))
        return out

    # Tabular K-sweep header — fixed-width columns for grep + visual scanning.
    # Dirichlet diag columns are derived from ROUTER_DIRICHLET_DIAG_TERMS
    # (single source of truth — adding a 6th term means a one-line
    # registry edit, not 8 grep-and-paste sites).
    _dir_cols = [(short, 10 if short == "dir_sigma" else 9)
                 for _, _, short in ROUTER_DIRICHLET_DIAG_TERMS]
    _dir_cols.append((ROUTER_DIRICHLET_BETA_TERM[2], 9))
    _kdiag_cols = [
        ("K", 5), ("val_bpb", 9), ("attn_cv", 8), ("mlp_cv", 8), ("pool_cv", 8),
        ("attn_min", 9), ("mlp_min", 9), ("attn_ortho", 11), ("mlp_ortho", 10),
        ("pertoken_ent", 13), ("pool_ent", 9), ("shared_gate", 12),
        *_dir_cols,
        # rho_F (principled gate per Hartman-Grobman) + fp_residual_F (joint
        # two-state-cycle residual at the saved FP) + iter_conv_rel (z-only
        # step residual at the K-eval forward).
        ("rho_F", 9), ("sigma_max_F", 12), ("fp_residual_F", 14), ("iter_conv_rel", 14),
    ]
    def _fmt_kdiag(value: float | None, width: int, name: str = "") -> str:
        if value is None:
            s = "N/A"
        elif name == "K":
            s = str(int(value))
        else:
            s = f"{value:.4f}"
        return s.rjust(width)
    log0("k_sweep_table:" + " ".join(name.rjust(w) for name, w in _kdiag_cols))
    for k_eval in k_sweep_values:
        # Pass deq_k explicitly — run_validation uses it directly instead of
        # reading from args.deq_k_eval (which was the root cause of the bug
        # that made all previous K-sweeps flat: the callee clobbered the
        # caller's _deq_k_override with args.deq_k_eval=8).
        # Use fast eval (small subset) for K-sweep — full validation is too
        # expensive at high K (K=40 = 40 solver iters per batch).  The
        # roundtrip_verification already ran full validation at the default K;
        # the sweep only needs relative comparisons across K values.
        with router_diagnostics(enabled=True, step_tag=k_eval):
            _, bpb_k = run_validation(
                args, base_m_for_roundtrip, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                full_validation=False, deq_k=k_eval,
            )
        k_sweep_results[k_eval] = float(bpb_k)
        # Log val_bpb + per-iteration gate trajectories + DEQ diagnostics for this K.
        diag_parts = [f"val_bpb:{bpb_k:.6f}"]
        ag_iter = getattr(base_m_for_roundtrip, "_attn_gate_iter_last_solve", None)
        if ag_iter and len(ag_iter) > 0:
            diag_parts.append(f"attn_gate_iter:[{','.join(f'{v:.3f}' for v in ag_iter)}]")
        rg_iter = getattr(base_m_for_roundtrip, "_router_gate_iter_last_solve", None)
        if rg_iter and len(rg_iter) > 0:
            diag_parts.append(f"router_gate_iter:[{','.join(f'{v:.3f}' for v in rg_iter)}]")
        attn_rg_iter = getattr(base_m_for_roundtrip, "_attn_router_gate_iter_last_solve", None)
        if attn_rg_iter and len(attn_rg_iter) > 0:
            diag_parts.append(f"attn_rg_iter:[{','.join(f'{v:.3f}' for v in attn_rg_iter)}]")
        mlp_rg_iter = getattr(base_m_for_roundtrip, "_mlp_router_gate_iter_last_solve", None)
        if mlp_rg_iter and len(mlp_rg_iter) > 0:
            diag_parts.append(f"mlp_rg_iter:[{','.join(f'{v:.3f}' for v in mlp_rg_iter)}]")
        conv_rel_t = getattr(base_m_for_roundtrip, "_deq_iter_convergence_rel_t", None)
        if isinstance(conv_rel_t, torch.Tensor):
            diag_parts.append(f"iter_conv_rel:{float(conv_rel_t.detach().float().item()):.6f}")
        resid_t = getattr(base_m_for_roundtrip, "_deq_residual_t", None)
        if isinstance(resid_t, torch.Tensor):
            diag_parts.append(f"residual:{float(resid_t.detach().float().item()):.2f}")
        kdiag = _collect_eval_kdiag(base_m_for_roundtrip)
        _conf_names = tuple(name for name, _, _ in ROUTER_DIRICHLET_DIAG_TERMS) + (
            ROUTER_DIRICHLET_BETA_TERM[0],
        )
        for conf_name in _conf_names:
            conf_value = kdiag.get(conf_name)
            if conf_value is not None:
                diag_parts.append(f"{conf_name}:{float(conf_value):.4f}")
        # Local FP contraction metric at the saved DEQ FP (z*).  Eval profile
        # Per-K spectral probe at the saved FP. rho_F is the principled
        # gate metric (necessary AND sufficient for asymptotic FP
        # convergence per Hartman-Grobman); lip_ub_T/S/F and fp_bound
        # were removed 2026-05-15 — operator-norm proxies are
        # over-restrictive for non-symmetric J_F. fp_residual_F is the
        # joint two-state-cycle residual at the saved FP.
        rho_F_val = _rho_F_at_saved_fp(
            base_m_for_roundtrip,
            n_iters=int(getattr(args, "fp_rho_power_iters", 8)),
            log_label="ksweep_skip_reason:rho_F",
        )
        if rho_F_val is not None:
            diag_parts.append(f"rho_F:{rho_F_val:.4f}")
            k_sweep_rho_F[k_eval] = float(rho_F_val)
        # iter168 (2026-05-16): sigma_max_F restored as DIAGNOSTIC. NOT a
        # gate, NOT a penalty, NOT a prescription input. Carries the
        # robustness/basin-size info that rho_F alone does not.
        sigma_max_F_val = _sigma_max_F_at_saved_fp(
            base_m_for_roundtrip,
            n_iters=int(getattr(args, "fp_rho_power_iters", 8)),
            log_label="ksweep_skip_reason:sigma_max_F",
        )
        if sigma_max_F_val is not None:
            diag_parts.append(f"sigma_max_F:{sigma_max_F_val:.4f}")
            k_sweep_sigma_max_F[k_eval] = float(sigma_max_F_val)
        fp_residual_F_val = _joint_F_residual_at_saved_fp(
            base_m_for_roundtrip,
            log_label="ksweep_skip_reason:fp_residual_F",
        )
        if fp_residual_F_val is not None:
            diag_parts.append(f"fp_residual_F:{fp_residual_F_val:.6f}")
            k_sweep_fp_residual_F[k_eval] = float(fp_residual_F_val)
        log0(f"k_sweep:k={k_eval} {' '.join(diag_parts)}")
        # iter 100b user directive (PERMANENT): tabular per-K row.
        # iter_conv_rel (z-only step residual at K-eval forward) is the
        # empirical FP-convergence diagnostic; rho_F (spectral radius)
        # is the principled gate metric.
        conv_rel_val = float(conv_rel_t.detach().float().item()) if isinstance(conv_rel_t, torch.Tensor) else None
        kdiag_row = {
            "K": float(k_eval),
            "val_bpb": float(bpb_k),
            "attn_cv": kdiag.get("attn_cv"),
            "mlp_cv": kdiag.get("mlp_cv"),
            "pool_cv": kdiag.get("pool_cv"),
            "attn_min": kdiag.get("attn_min"),
            "mlp_min": kdiag.get("mlp_min"),
            "attn_ortho": kdiag.get("attn_ortho"),
            "mlp_ortho": kdiag.get("mlp_ortho"),
            "pertoken_ent": kdiag.get("pertoken_ent"),
            "pool_ent": kdiag.get("pool_ent"),
            "shared_gate": kdiag.get("shared_gate"),
            **{short: kdiag.get(name) for name, _, short in ROUTER_DIRICHLET_DIAG_TERMS},
            ROUTER_DIRICHLET_BETA_TERM[2]: kdiag.get(ROUTER_DIRICHLET_BETA_TERM[0]),
            "rho_F": rho_F_val,
            "sigma_max_F": sigma_max_F_val,
            "fp_residual_F": fp_residual_F_val,
            "iter_conv_rel": conv_rel_val,
        }
        log0("k_sweep_table:" + " ".join(_fmt_kdiag(kdiag_row[name], w, name) for name, w in _kdiag_cols))
    k_parts = " ".join(f"k{k}:{b:.6f}" for k, b in k_sweep_results.items())
    log0(f"k_sweep:done {k_parts}{format_k_jitter_info()}")

    # ── Final diagnostic gates on post-int6 model health ─────────────────
    # Run after the K-sweep so all diagnostics are populated from the highest
    # K eval pass.  These gates are diagnostics under the val_bpb-primary
    # promotion policy: failures emit retry prescriptions and tech-debt status,
    # while promotion authority comes from full-set final val_bpb.
    #
    # Expert usage + CV are rank-local per-router state (from the fast K-sweep
    # eval which only batched a subset), so we DDP-all-reduce them here to get
    # the global view that actually matters.
    _failures: list[str] = []

    # Deadlock-safe DDP helpers.  Every rank MUST enter all_reduce regardless
    # of whether its local value is None, so a NaN/zero sentinel + presence
    # mask is used to preserve pass-through semantics while keeping all ranks
    # in lock-step.  Returns (None, True) when the value was absent on every
    # rank (nothing to assert); otherwise the global mean over ranks that had it.
    def _ddp_mean_scalar(x: float | None) -> float | None:
        if not distributed:
            return None if x is None else float(x)
        present = torch.tensor([0.0 if x is None else 1.0], device=device)
        value = torch.tensor([0.0 if x is None else float(x)], device=device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        n = float(present.item())
        return None if n == 0.0 else float(value.item()) / n

    def _ddp_max_scalar(x: float | None) -> float | None:
        if not distributed:
            return None if x is None else float(x)
        present = torch.tensor([0.0 if x is None else 1.0], device=device)
        value = torch.tensor([-float("inf") if x is None else float(x)], device=device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return None if float(present.item()) == 0.0 else float(value.item())

    def _ddp_mean_vec(v: list[float] | None, length: int) -> list[float] | None:
        # Length-mismatch: coerce to absent and log (on master).  Raising here
        # would deadlock DDP — if only some ranks have the wrong shape, they'd
        # raise while peer ranks enter dist.all_reduce below and wait forever.
        # Soft-degrade preserves liveness; a divergent diagnostic will still
        # show up as an assertion failure (via the presence mask averaging).
        if v is not None and len(v) != length:
            if master_process:
                log0(f"[WARN] _ddp_mean_vec length mismatch: got {len(v)} expected {length} — treating as absent")
            v = None
        if not distributed:
            return None if v is None else [float(x) for x in v]
        # Pad / zero-fill based on presence so ALL ranks call all_reduce with
        # matching shapes even if some ranks never populated the diagnostic.
        present = torch.tensor([0.0 if v is None else 1.0], device=device)
        value = torch.zeros(length, device=device)
        if v is not None:
            value.copy_(torch.tensor([float(x) for x in v], device=device))
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        n = float(present.item())
        if n == 0.0:
            return None
        return (value / n).detach().cpu().tolist()

    def _ddp_mean_tensor_vec(t: Tensor | None, length: int) -> list[float] | None:
        """Like _ddp_mean_vec but takes a GPU tensor directly — avoids the
        per-rank .cpu().tolist() sync that diagnostics would need otherwise.
        Like _ddp_mean_vec, soft-degrades on length mismatch rather than raising,
        to avoid DDP deadlocks."""
        if t is not None and t.numel() != length:
            if master_process:
                log0(f"[WARN] _ddp_mean_tensor_vec length mismatch: got {t.numel()} expected {length} — treating as absent")
            t = None
        if not distributed:
            return None if t is None else t.detach().float().cpu().tolist()
        present = torch.tensor([0.0 if t is None else 1.0], device=device)
        value = torch.zeros(length, device=device)
        if t is not None:
            # .reshape() instead of .view() — if the diagnostic tensor ever
            # comes in non-contiguous (e.g., after a slice or advanced index),
            # .view() would raise but .reshape() handles it gracefully.
            value.copy_(t.detach().float().reshape(length))
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        n = float(present.item())
        if n == 0.0:
            return None
        return (value / n).detach().cpu().tolist()

    # Helpers for splitting pooled-router usage into per-type normalized halves.
    def _split_pool_usage_gpu(router, E, half):
        gpu = getattr(router, "_expert_usage_gpu", None)
        if gpu is None:
            return None
        h = gpu[:E] if half == 0 else gpu[E:]
        return h / h.sum().clamp(min=1e-8)  # normalize within type

    def _split_pool_usage_list(router, E, half):
        lst = getattr(router, "_expert_usage", None)
        if lst is None:
            return None
        h = lst[:E] if half == 0 else lst[E:]
        s = _safe_sum(h)
        return [u / s for u in h]

    # 1. Expert health per routed component (DDP-global).
    # Min-share: ≥ 0.6/E per type (normalized within each type — weakest
    # expert gets ≥60% of its fair share).  Ortho: max-pairwise |cos| ≤ 0.5
    # (stricter than near-duplicate 0.9, catches correlated experts).
    shared_block = base_m_for_roundtrip.shared_block
    mos_head = getattr(base_m_for_roundtrip, "mos_head", None)

    def _mos_usage(head: str) -> list[float] | None:
        if mos_head is None:
            return None
        return getattr(mos_head, f"_{head}_expert_usage", None)

    def _mos_usage_gpu(head: str) -> Tensor | None:
        if mos_head is None:
            return None
        return getattr(mos_head, f"_{head}_expert_usage_gpu", None)

    def _mos_num_experts(head: str) -> int:
        if mos_head is None:
            return 0
        return int(getattr(mos_head, "num_shared", 0)) + int(getattr(mos_head, "num_specialized", 0))

    def _mos_ortho(head: str) -> float | None:
        if mos_head is None or not hasattr(mos_head, "get_head_orthogonality"):
            return None
        try:
            return float(mos_head.get_head_orthogonality(head))
        except Exception:
            return None

    # Each spec: (prefix, num_experts, usage_gpu_getter, usage_list_getter, ortho_getter).
    # Dedup pooled router by id — attn_router and mlp_router are aliases.
    # For pooled router (2R routed experts), split usage into halves and check
    # each type independently with normalized per-type routed shares. Shared
    # bypass gates are separate diagnostics, not hidden routed usage.
    check_specs = []
    attn_router = getattr(shared_block.attn, "attn_router", None)
    mlp_router = getattr(shared_block.mlp, "mlp_router", None)
    E = shared_block.num_experts
    R = E - int(getattr(shared_block, "num_shared_experts", 0))
    _seen_router_ids: set[int] = set()
    if attn_router is not None and id(attn_router) not in _seen_router_ids:
        _seen_router_ids.add(id(attn_router))
        is_pooled = int(attn_router.num_experts) == 2 * R
        if is_pooled:
            # Pooled router: check per-type normalized halves
            check_specs.append((
                "attn", R,
                lambda: _split_pool_usage_gpu(attn_router, R, half=0),
                lambda: _split_pool_usage_list(attn_router, R, half=0),
                lambda: getattr(shared_block.attn, "_out_ortho_cos_sim", None),
            ))
            check_specs.append((
                "mlp", R,
                lambda: _split_pool_usage_gpu(attn_router, R, half=1),
                lambda: _split_pool_usage_list(attn_router, R, half=1),
                lambda: getattr(shared_block.mlp, "_out_ortho_cos_sim", None),
            ))
        else:
            check_specs.append((
                "attn", int(attn_router.num_experts),
                lambda: getattr(attn_router, "_expert_usage_gpu", None),
                lambda: getattr(attn_router, "_expert_usage", None),
                lambda: getattr(shared_block.attn, "_out_ortho_cos_sim", None),
            ))
    if mlp_router is not None and id(mlp_router) not in _seen_router_ids:
        _seen_router_ids.add(id(mlp_router))
        check_specs.append((
            "mlp", int(mlp_router.num_experts),
            lambda: getattr(mlp_router, "_expert_usage_gpu", None),
            lambda: getattr(mlp_router, "_expert_usage", None),
            lambda: getattr(shared_block.mlp, "_out_ortho_cos_sim", None),
        ))
    if mos_head is not None:
        # NTP head is always allocated.
        n_ntp = _mos_num_experts("ntp")
        check_specs.append((
            "mos_ntp", n_ntp,
            lambda: _mos_usage_gpu("ntp"),
            lambda: _mos_usage("ntp"),
            lambda: _mos_ortho("ntp"),
        ))
        # CTP head is opt-in (iter 94: NTP-only baseline).  When
        # `use_ctp=False` the CTP param banks are not allocated — the
        # `_mos_usage("ctp")` getters fall back to NTP/empty values, so emitting
        # a "mos_ctp" check_spec produces fake "dead expert" failures with no
        # underlying CTP capacity to fix.  Mirror the analogous guard in
        # `MoSHead.get_head_orthogonality("ctp")` (line ~1936) at the
        # diagnostic-emission site.  See EXPERIENCE.md §1
        # `diagnostic-gate-component-awareness`.
        if getattr(mos_head, "use_ctp", False):
            n_ctp = _mos_num_experts("ctp")
            check_specs.append((
                "mos_ctp", n_ctp,
                lambda: _mos_usage_gpu("ctp"),
                lambda: _mos_usage("ctp"),
                lambda: _mos_ortho("ctp"),
            ))

    for prefix, n_exp, usage_gpu_getter, usage_list_getter, ortho_getter in check_specs:
        if n_exp <= 0:
            continue
        # Prefer the GPU-tensor path (no cpu sync on non-master ranks).  Fall
        # back to the list path if that router didn't populate the GPU copy.
        usage_gpu = usage_gpu_getter()
        usage = (
            _ddp_mean_tensor_vec(usage_gpu, n_exp)
            if usage_gpu is not None
            else _ddp_mean_vec(usage_list_getter(), n_exp)
        )
        if usage is not None and len(usage) > 0:
            # Dead expert check: load balance is already optimized by the
            # balance_loss during training.  We don't re-verify balance here;
            # instead we gate on the STRUCTURAL invariant that no expert should
            # be completely dead (< 1% usage).  Sub-1% usage means that expert
            # is effectively unlearned weights being carried in the artifact —
            # wasted capacity — and routing has collapsed onto the others.
            min_share = float(min(usage))
            threshold = 0.6 / max(n_exp, 1)
            if min_share < threshold:
                _failures.append(
                    f"{prefix}_min_share={min_share:.4f} < {threshold:.4f} "
                    f"(expert below 60% of fair share 1/{n_exp})"
                )
        # Expert orthogonality: max-pairwise |cos| ≤ 0.5 — catches correlated
        # expert pairs, not just near-duplicates (0.9 was too lenient).
        ortho = _ddp_mean_scalar(ortho_getter())
        if ortho is not None and ortho > 0.5:
            _failures.append(
                f"{prefix}_ortho={ortho:.4f} > 0.5 (max pairwise |cos| — correlated experts)"
            )

    # 2. FP convergence: K-sweep degradation gate.  Threshold 0.1 catches gross
    # FP collapse (model exploits a specific K and degrades dramatically at others)
    # without firing on finite-K noise (typical K=64 vs K=32 fluctuation is
    # 0.005-0.010, which the prior 0.005 monotone gate flagged as failures —
    # the current best baseline 1.705 also fails the old 0.005 gate).
    # The per-step monotone check has been DROPPED; it was below the noise floor.
    if len(k_sweep_results) >= 2:
        ks_sorted = sorted(k_sweep_results.items())
        best_bpb = min(v for _, v in ks_sorted)
        # Gross degradation gate: any K>=16 shouldn't be 0.1+ worse than best
        worst_high_k = max(v for k, v in ks_sorted if k >= 16) if any(k >= 16 for k, _ in ks_sorted) else None
        if worst_high_k is not None and worst_high_k - best_bpb > 0.1:
            _failures.append(
                f"K-sweep degradation: best={best_bpb:.4f} worst_k>=16={worst_high_k:.4f} "
                f"(Δ={worst_high_k - best_bpb:.4f} > 0.1 — gross FP quality loss at deep K)"
            )
        # True-FP quality gate: K=64 and K=128 should be near the best
        # high-K value. Task loss along the solver trajectory is not a
        # Lyapunov function, so strict monotonic BPB is not required; a large
        # late-K regression indicates finite-depth exploitation or cycles.
        for k_check in [64, 128]:
            if k_check in k_sweep_results:
                delta = k_sweep_results[k_check] - best_bpb
                if delta > 0.02:
                    _failures.append(
                        f"K={k_check}_degradation: bpb={k_sweep_results[k_check]:.4f} "
                        f"vs best={best_bpb:.4f} (Δ={delta:.4f} > 0.02 — not a true FP)"
                    )

    conv_rel_t = getattr(base_m_for_roundtrip, "_deq_iter_convergence_rel_t", None)
    conv_rel_local = (
        float(conv_rel_t.detach().float().item())
        if isinstance(conv_rel_t, torch.Tensor) else None
    )
    conv_rel = _ddp_mean_scalar(conv_rel_local)

    # 3. Local contraction + FP convergence: rho(J_F) < 1 is the gate
    # (necessary AND sufficient for asymptotic local convergence per
    # Hartman-Grobman). lip_ub_T/S/F operator-norm proxies were removed
    # 2026-05-15 as over-restrictive.
    deepest_k = max(k_sweep_values) if k_sweep_values else None
    if deepest_k is not None:
        deepest_rho_F = _ddp_max_scalar(k_sweep_rho_F.get(deepest_k))
        deepest_fp_resid_F = _ddp_max_scalar(k_sweep_fp_residual_F.get(deepest_k))
        if deepest_rho_F is not None and deepest_rho_F >= 1.0:
            attr_str = f", fp_residual_F={deepest_fp_resid_F:.6f}" if deepest_fp_resid_F is not None else ""
            _failures.append(
                f"rho_F={deepest_rho_F:.4f} >= 1.0 at K={deepest_k}{attr_str} "
                "(spectral-radius estimate on F; necessary-and-sufficient asymptotic "
                "convergence condition violated -- formal-tier mechanism required)"
            )

    # 4. Iter convergence (empirical FP convergence signal).  Promoted to
    # gate-relevant per Tier 1 redesign: this is the operational signal
    # of asymptotic convergence (decoupled from theoretical spectral-radius
    # estimates via `rho_F`).  Threshold 0.05 chosen because iter152
    # (BPB-winning baseline) shows 0.019 at K=128; 0.05 leaves headroom
    # for legitimate optimization noise while still flagging real divergence.
    # DDP-reduce the rank-local conv_rel so the assertion sees the global
    # mean (matches the treatment of ortho/usage above).
    if conv_rel is not None and conv_rel >= 0.05:
        # Tier-graded message: the 0.05 threshold is the principled gate
        # (iter152 baseline shows 0.019 at K=128, comfortable margin); the
        # 0.1 threshold tags severe divergence specifically.  One failure
        # row regardless of severity to avoid duplicate gate signals.
        if conv_rel > 0.1:
            _failures.append(
                f"iter_conv_rel={conv_rel:.4f} > 0.1 (severe solver divergence at deepest K -- "
                "asymptotic local FP convergence empirically failed)"
            )
        else:
            _failures.append(
                f"iter_conv_rel={conv_rel:.4f} >= 0.05 (empirical FP convergence "
                "signal weak at deepest K -- the actually-required condition for "
                "asymptotic local contraction is not cleanly met)"
            )

    # 6. RevDEQ reconstruction error: end-to-end ||reverse(forward(x_0)) − x_0||,
    # decision-grade in fp32+ but saturates the bf16 reversibility budget when
    # the round-trip walks K_fwd > ~16 iters (each f_theta call carries
    # ε_bf16 ≈ 7e-3 and amplifies by 1/Ā per reverse step). The 0.1 threshold
    # is meaningful at full BPTT in fp32 or at low K_fwd in bf16; under TBPTT
    # (default) the diagnostic still walks the full K_fwd reverse via no_grad
    # continuation but at K_fwd=16-24 the bf16 budget is exhausted by design,
    # so we skip the threshold check there to avoid spurious failures on a
    # known-bounded metric. CLAUDE.md §6.1.
    bptt_k_now = int(getattr(base_m_for_roundtrip, "deq_bptt_k", 0) or 0)
    is_full_bptt = bptt_k_now == 0
    dist_local = getattr(base_m_for_roundtrip.shared_block, "_deq_x0_recon_error_last_bwd", None)
    dist_val = _ddp_mean_scalar(
        float(dist_local) if dist_local is not None else None
    )
    if is_full_bptt and dist_val is not None and dist_val > 0.1:
        _failures.append(
            f"deq_recon_err={dist_val:.3e} > 0.1 (RevDEQ reversibility degrading — "
            f"gradients getting noisy, training efficiency drops)"
        )

    # Classify each failure and prescribe a fix from the verified-hypothesis
    # troubleshooting table.  Gate failures are not metadata-invalidating by
    # themselves; they guide the NEXT iteration's config change. We do NOT
    # raise here because a clean exit preserves artifacts and log output.
    # `_prescribe_failure_fix` lives at module level for direct unit testing
    # — see experiments/test_arch.py.
    if _failures:
        # POST-INT6 diagnostics are final guardrails and retry prescriptions.
        log0("POST-INT6 DIAGNOSTIC GATE FAILURES — retry prescription follows")
        prescriptions = [_prescribe_failure_fix(f) for f in _failures]
        log0(f"  {len(_failures)} failure(s):")
        for p in prescriptions:
            log0(f"  ⚠ [{p['category']}] {p['failure']}")
            log0(f"     hypothesis: {p['hypothesis']}")
            log0(f"     fix:        {p['fix']}")
        # Aggregate config suggestions for the orchestrator / agent.
        agg: dict = {}
        for p in prescriptions:
            for k, v in p["config_change"].items():
                agg.setdefault(k, []).append(v)
        suggested_config: dict = {}
        for k, vs in agg.items():
            # For multiplicative/additive hints, prefer the max-impact adjustment.
            if k.endswith("_mult"):
                suggested_config[k] = max(vs)
            elif k.endswith("_delta"):
                suggested_config[k] = max(vs, key=abs)  # largest-magnitude delta
            else:
                suggested_config[k] = Counter(vs).most_common(1)[0][0]
        log0(f"  SUGGESTED CONFIG CHANGE: {suggested_config}")
        # Write machine-readable retry hint next to the artifacts.
        if master_process:
            retry_hint = {
                "health_valid": False,
                "failure_count": len(_failures),
                "prescriptions": prescriptions,
                "suggested_config": suggested_config,
            }
            with open(Path("experiments/weights/current") / "retry_hint.json", "w") as fh:
                json.dump(retry_hint, fh, indent=2)
            log0("  retry_hint.json written — next iter should apply suggested_config")
        # Record failure categories in meta.json for the next iter's analysis,
        # but DO NOT set run_valid=false (val_bpb-primary policy).
        if master_process and meta_path is not None:
            with open(meta_path, "r") as f:
                meta_json = json.load(f)
            meta_json["failure_categories"] = [p["category"] for p in prescriptions]
            meta_json["gate_status"] = "diagnostic_fail"
            with open(meta_path, "w") as f:
                json.dump(meta_json, f)
    _assertions_passed = not _failures
    if not _failures:
        log0("POST-INT6 HEALTH: all diagnostic gates passed (no dead experts, expert ortho, gate trend, injection, FP convergence, reversibility)")
        if master_process:
            retry_hint_path = Path("experiments/weights/current") / "retry_hint.json"
            if retry_hint_path.exists():
                retry_hint_path.unlink()  # stale hint from a previous failure

    # Restore eval K for any downstream sliding-window eval.
    base_m_for_roundtrip._deq_k_override = int(args.deq_k_eval)

    val_stride = int(getattr(args, "eval_stride", 0))
    if val_stride > 0:
        log0(f"sliding_validation:start stride={val_stride}")
        _, sliding_bpb = sliding_window_validation(
            args, base_m_for_roundtrip, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=val_stride,
        )
        log0(f"sliding_validation:done val_bpb:{sliding_bpb:.6f}")

    # Full-set validation is authoritative for promotable final metadata. Fast
    # roundtrip validation remains logged as `fast_val_bpb`, but it must not be
    # silently promoted as the final score.
    full_val_bpb: float | None = None
    if bool(getattr(args, "final_full_validation", False)):
        log0("final_full_validation:start")
        _, full_val_bpb = run_validation(
            args, base_m_for_roundtrip, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            full_validation=True,
        )
        log0(f"final_full_validation:done val_bpb:{full_val_bpb:.6f}")

    # Master-only: update meta.json with the final authoritative val_bpb and
    # run_valid flag. Writing both atomically at the very end — after sliding
    # val + all eval passes — ensures a crash between eval success and here
    # leaves run_valid=false, so update_results.sh --promote refuses it.
    if master_process and meta_path is not None:
        with open(meta_path, "r") as f:
            meta_json = json.load(f)
        # Persist both: fast_val_bpb is always the int6 roundtrip proxy; val_bpb
        # is authoritative only when full validation completed.
        diagnostic_policy = str(args.diagnostic_gate_policy).strip().lower()
        health_valid = bool(_assertions_passed)
        score_valid = full_val_bpb is not None
        meta_json["config_profile"] = str(args.config_profile)
        meta_json["eval_profile"] = str(args.eval_profile)
        meta_json["diagnostic_gate_policy"] = diagnostic_policy
        meta_json["score_valid"] = bool(score_valid)
        meta_json["health_valid"] = bool(health_valid)
        meta_json["gate_status"] = "passed" if health_valid else "diagnostic_fail"
        meta_json["fast_val_bpb"] = float(val_bpb_q)
        # Keep val_bpb float-typed even in the fast-only path; promotion gating
        # lives in `_compute_run_status` (returns validated_fast_only there).
        meta_json["val_bpb"] = float(full_val_bpb) if full_val_bpb is not None else float(val_bpb_q)
        run_valid, status, reason = _compute_run_status(
            diagnostic_policy=diagnostic_policy,
            health_valid=health_valid,
            full_val_completed=full_val_bpb is not None,
        )
        meta_json["run_valid"] = bool(run_valid)
        meta_json["status"] = status
        if reason is not None:
            meta_json["non_promotable_reason"] = reason
        with open(meta_path, "w") as f:
            json.dump(meta_json, f)
        log0(
            f"final_status: health_valid:{int(health_valid)} "
            f"gate_status:{meta_json['gate_status']} "
            f"score_valid:{int(score_valid)} run_valid:{int(run_valid)} "
            f"status:{status} codec:{meta_json.get('artifact_codec', 'int6')}"
        )

    # Tear down DDP after all eval completes.
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
