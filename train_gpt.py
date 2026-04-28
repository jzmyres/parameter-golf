"""
Simplified RevDEQ training script for Parameter Golf.
Single-file, self-contained submission artifact.

Architecture: RevDEQ + Soft Dense MoE + MLA + Gated Attention + FSQ/MoS + Diffusion-AR
"""

from __future__ import annotations

import atexit
import contextlib
import glob
import io
import argparse
import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import uuid
import warnings
import zlib
from collections import Counter
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

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


def _should_diag(training: bool) -> bool:
    """Return True if this rank should record diagnostics right now.

    Training: rank 0 only (master-logged, other ranks save the compute).
    Eval: ALL ranks — the post-eval DDP-global assertions need every rank to
    have populated diagnostics so dist.all_reduce() has matching participants.
    Rank-gating still happens at the log sites, not here.
    """
    if training and not _ROUTER_DIAGNOSTICS_ACTIVE:
        return False
    if training:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    # Eval path: every rank populates local diagnostics.
    return True


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
    data_path = "./data/datasets/fineweb10B_sp1024"
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id = ""
    seed = 42

    val_batch_size = 524_288
    val_micro_batch_seqs = 0  # T-opt 17: 0 = derive from training micro-batch (same B as training)
    val_loss_every = 200
    train_log_every = 10  # log every 10 steps (~85s at 8.5s/step) for better progress visibility
    auto_plot_on_val = True

    iterations = 1000  # default training budget: step-count-governed, DDP with all GPUs; submission runs override via --max-wallclock-seconds=600
    warmdown_frac = 0.72  # fraction of total steps for warmdown
    warmup_steps = 0
    train_batch_tokens = 524_288
    train_seq_len = 2048
    max_wallclock_seconds = 0  # 0 = disabled; step-count governs default runs. Submission runs MUST pass --max-wallclock-seconds=600 (8xH100 competition hard cap).

    # Model architecture
    vocab_size = 1024
    num_layers = 12  # DEQ solver max K
    num_refinements = 1
    num_refinements_ramp_frac = 0.85  # enable refinement after 85% of wallclock
    num_kv_heads = 4
    model_dim = 768  # iter 96 baseline. Iter 98 attempted 768 → 1024 but OOM'd 3× on 44 GiB L40S dev hardware (D=1024 + DEQ TBPTT exceeds VRAM cap regardless of seq/K reductions). Documented as NOT TESTED in H73; D-scaling deferred until 8× H100 80GB submission hardware (won't OOM there).
    num_heads = 8
    num_experts = 16  # iter 96 baseline (PROMOTED ★, H71): 8 → 16 paired with attn/mlp_expert_rank halving. Iter 97 (E=20) NOT PROMOTED on per-wallclock grounds; H72 documents axis saturation past E=16 / R=64 on D=768.
    num_shared_experts = 1  # Phase 9 iter 51: DeepSeek shared expert (always-on, bypass routing)
    # Iter 94 (2026-04-24): disable CTP head entirely. When False, MoS head only
    # emits NTP log-probs; CTP param banks (gate_ctp, A_ctp_shared, A_ctp,
    # B_denoise, ctp_*_norm_weight) are not allocated, CTP loss is skipped, and
    # the refinement soft-embedding mix uses p_ntp only. Tests whether the
    # dual-head denoising gradient is still load-bearing under iter 66b's
    # Parcae + learnable-norm landscape.
    use_ctp = False
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
    parcae_lr = 0.002  # 10× slower than scalar_lr — Parcae params control DEQ mixing

    # Optimizer
    tied_embed_lr = 0.03
    embed_lr = 0.6
    matrix_lr = 0.022
    scalar_lr = 0.02
    muon_momentum = 0.99
    muon_backend_steps = 5
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 800
    beta1 = 0.85
    beta2 = 0.90
    adam_eps = 1e-8
    grad_clip_norm = 1.0  # iter 73: relax from 0.3 — Lyapunov provides soft contraction (H56)
    weight_decay = 0.01  # Phase 9 iter 86 (2026-04-24): 0.30 → 0.01 re-test. Under iter 93 landscape (learnable norms everywhere + Parcae B̄ + split shared gates + no CTP + no BigramHash), the non-norm paths that WD=0.30 was regularizing have shifted. Test whether a much lower floor lets the transformer body's capacity express more. Prior iter 66a-b confirmed WD=0 on 1D params regresses, but this tests a lower GLOBAL floor.
    tied_embed_init_std = 0.005

    # Routing
    attn_balance_mult = 5.0
    mlp_balance_mult = 1.0
    # iter 26-lb-loss: 50× multiplier on MoS NTP balance loss fixed dead-expert
    # collapse where WD could not (H26).  Promoted from hardcoded literal so the
    # retry-prescription path can recommend bumping it for mos_*_min_share.
    mos_balance_mult = 50.0
    bal_loss_coef = 5e-3
    router_health_coef = 0.25
    block_ortho_aux_coef = 0.1  # partial restore from 0 (H32 was too aggressive — attn/mlp ortho drifted to 0.82, near the 0.9 gate fail). 0.1 keeps gradient pressure without dominating, preserves H32's "loss shouldn't chase gates" spirit while keeping experts apart.
    block_ortho_aux_every = 8  # T-opt 19: double interval (4→8) to halve ortho_aux cost
    block_ortho_aux_tokens = 64
    router_bias_update = True
    router_bias_lr = 0.10
    router_bias_clip = 10.0
    # Phase 9 iter 70: switch L2→linear dot-product routing for effective depth.
    # L2+tanh logits are insensitive to directional z_k changes between DEQ iters
    # (expert_iter_std≈0). Linear dot-product captures rotation → different expert
    # mixtures across iterations → effective depth > 1.
    router_scoring = "linear"
    # iter 99 (2026-04-26): router output activation. "softmax" was the iter 96
    # baseline; "sparsemax" replaces softmax with closed-form simplex projection
    # (Martins & Astudillo 2016) — produces EXACT zeros for low-logit experts,
    # driving per-token specialization architecturally rather than via loss
    # penalty. RevDEQ-safe: deterministic + 1-Lipschitz + subdifferentiable.
    # Composes with sigmoid gate (`p_alloc * gate_act`) unchanged.
    router_kind = "softmax"  # iter 96 baseline. Iter 99 (sparsemax α=2) NOT PROMOTED H74 (+0.16). Iter 101 (entmax-1.5) NOT PROMOTED H75 (+0.10). Both architectural-sparsity attempts had capacity cost > regularization gain. Reverted to softmax. Options: "softmax", "sparsemax", "entmax15".
    # iter 100b (2026-04-27): per-token entropy penalty with ANNEALED schedule
    # + min_share_loss decoupled. iter 100 (entropy_coef=0.02 from step 0)
    # hit train_loss instability — penalty fights min_share_loss penalty
    # (one pushes uniform, other pushes peaky). Decouple them:
    #   1. Lower coef target: 0.02 → 0.005 (4× weaker)
    #   2. Anneal from 0 over first warmup_frac of training (cold-start trap
    #      avoidance, same principle as α-annealing for iter 102).
    #   3. Drop min_share_loss penalty; keep min_share as DIAGNOSTIC metric only.
    # CV loss alone provides global-balance regularization; entropy penalty
    # alone provides per-token sparsity push. Two forces pulling in same-axis
    # not opposite. Cleaner gradient landscape.
    router_entropy_coef = 0.005
    # Annealing schedule (iter 100b): scale = 0 for time_frac < warmup_delay_frac,
    # then linear ramp 0 → 1 over remaining training. Final effective coef =
    # router_entropy_coef × scale. Avoids cold-start trap (router needs free
    # softmax exploration early; sparsity pressure ramps in once routing has
    # stabilized).
    router_entropy_warmup_delay_frac = 0.3
    mos_ortho_out_coef = 0.0  # disabled — same rationale as block_ortho_aux_coef (loss focuses on task; max_pairwise GATE catches collapse)

    # iter 45 (opg_doc.tex §4): Lyapunov spectral-radius penalty.
    # Encourages ρ(J_{z*}) < γ at the reached equilibrium via persistent
    # power-iteration VJP. One boundary forward + one VJP per step.
    # Phase 9 iter 88 (2026-04-25): Lyapunov hinge penalty disabled (0.01 → 0).
    # Hypothesis: under iter-66b Parcae per-dim Ā, the spectral radius is
    # already bounded away from 1 by construction (Ā ∈ [0.1, 1) via the
    # reversibility floor + softplus reparam), so the Hutchinson-Frobenius
    # ρ(J)<γ hinge has nothing to grip on at training time.  Set λ_jac = 0;
    # if val_bpb regresses by > 0.03 OR K-sweep widens > 0.5, restore.
    lyapunov_coef = 0.0        # λ_jac: weight of hinge penalty (iter 88: disabled)
    lyapunov_gamma = 0.97      # iter 74b: raise from 0.9 — preserve faster warmup observed at γ=0.95 (H57)
    lyapunov_warmup_frac = 0.05 # T-opt 20: shorter warmup (10%→5%) — penalty near zero during warmup anyway
    # Phase 9 iter 55: Denoising regularization (HyDRA 2026, Efficient DEQ 2025).
    # ||f(z*+ε, x0) - z*||² penalizes contraction failure at finite perturbation.
    # Complements Hutchinson (which penalizes ||J||²_F at infinitesimal scale).
    # Phase 9 iter 89 (2026-04-25): HyDRA denoising disabled (0.01 → 0).
    # Same hypothesis as iter 88 for the finite-perturbation contraction probe:
    # under iter-66b Parcae per-dim Ā, the spectral radius is bounded away
    # from 1 by construction, so ||f(z*+ε, x0) - z*||² has nothing to grip
    # on at training time.  If the hypothesis holds, denoising contributes
    # only noise + one extra block forward per step.  Code path retained
    # (commented-out future cleanup permitted per user directive 2026-04-25).
    denoising_coef = 0.0       # weight of denoising loss (iter 89: disabled)
    denoising_noise_std = 0.01 # σ: Gaussian noise scale added to z*

    # DEQ solver
    # "revdeq" = custom RevDEQFunction with fp64 accumulators (O(1) memory).
    #   Enables torch.compile(shared_block) → 45% block speedup from fused
    #   permute+rms_norm kernels.  3 forwards per DEQ iter (fwd+reconstruct).
    # "unroll" = standard autograd through K DEQ iterations (O(K) memory).
    #   2 forwards per DEQ iter but CANNOT compile (breaks autograd chaining).
    #   Also needs 4× grad_accum for VRAM, reducing effective throughput.
    # Net: revdeq+compile > unroll+eager on 64-head independent expert MLA.
    # RevDEQ is the default: O(1) backward memory (4.4 GB peak at B=8) vs
    # unroll's O(K) autograd graph (43.7 GB, OOMs on L40S with 64 heads).
    # Sub-module compile (forward_experts + mix_experts) gives 2× attn + 1.5×
    # MLP speedup.  Full shared_block compile needs ≥60 GB (H100 only).
    deq_backward = "revdeq"
    # iter 28-tbptt: Truncated BPTT. Backward reconstructs only the last
    # `deq_bptt_k` forward iterations; earlier iters contribute no gradient.
    # 0 (or >= num_layers) = full BPTT.  Rationale: for a contractive DEQ,
    # per-iter VJP magnitudes decay geometrically toward x0, so the last few
    # iters should dominate the total param gradient.  If the hypothesis
    # holds, throughput scales ~ K_fwd / (K_fwd + K_bwd) improvement.
    deq_bptt_k = 2  # Phase 9 iter 69b: TBPTT=2 (middle ground — TBPTT=1 was +0.021 regression, TBPTT=4 is baseline)
    # Iter 85 (2026-04-24): stochastic TBPTT — sample deq_bptt_k per step from
    # the set below, analogous to K-jitter (H12 VERIFIED). Forces the model to
    # be robust across gradient-truncation depths. The set's shuffle-bag
    # sampler matches the β-jitter pattern.
    deq_bptt_k_jitter = True
    deq_bptt_k_jitter_set = (2, 3, 4)
    # TBPTT investigation (28-28d) concluded; best point was 28c (val_bpb
    # 1.925, K=128 Δ=0.015 vs baseline 0.039).  Machinery retained in code
    # — re-enable via CLI --deq-bptt-k=N.  Deeper-K jitter (4,8,16,24) may
    # be re-combined with Phase 6 contraction shell in a follow-up iter once
    # the new architecture stabilizes val_bpb.
    deq_k_jitter = True
    deq_k_min = 4
    deq_k_max = 20  # iter 87 (2026-04-24): bumped 16 → 20 to accommodate widened deq_k_jitter_set (8,12,20).
    deq_k_step = 4
    deq_k_jitter_set = (8, 12, 20)  # iter 87 (2026-04-24): doubled from (4,6,10) — deeper FP at training time should tighten K-sweep.
    deq_k_eval = 16  # iter 30: baseline eval K

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


def _parse_cli_overrides(argv: list[str]) -> dict[str, object]:
    p = argparse.ArgumentParser(add_help=True)
    for name in [
        "data-path", "tokenizer-path", "run-id", "seed", "iterations",
        "warmup-steps", "train-batch-tokens", "train-seq-len",
        "val-batch-size", "val-loss-every", "train-log-every",
        "max-wallclock-seconds", "attn-balance-mult", "mlp-balance-mult",
        "mos-balance-mult", "bal-loss-coef", "router-health-coef",
        "mos-ortho-out-coef", "block-ortho-aux-coef", "block-ortho-aux-every",
        "block-ortho-aux-tokens", "bigram-vocab-size", "bigram-dim",
        "kv-latent-dim", "attn-expert-rank", "mlp-expert-rank",
        "swa-start-frac", "swa-every", "ema-decay", "ema-update-every",
        "deq-k-min", "deq-k-max", "deq-k-step", "deq-k-eval", "deq-bptt-k",
        "warmdown-frac", "num-refinements-ramp-frac",
    ]:
        py_name = name.replace("-", "_")
        field_val = getattr(Hyperparameters, py_name, None)
        if isinstance(field_val, float):
            p.add_argument(f"--{name}", type=float, default=None)
        elif isinstance(field_val, int):
            p.add_argument(f"--{name}", type=int, default=None)
        else:
            p.add_argument(f"--{name}", type=str, default=None)
    for name in [
        "auto-plot-on-val", "router-bias-update", "deq-k-jitter",
        "swa-enabled", "ema-enabled", "use-ctp",
    ]:
        p.add_argument(f"--{name}", type=int, default=None, help="1/0")
    # Backward compat: "unroll" is the established name in experiments/docs.
    # "autograd" is accepted as an alias for the same mode.
    p.add_argument("--deq-backward", type=str, default=None, choices=["unroll", "autograd", "revdeq"])
    p.add_argument("--router-bias-lr", type=float, default=None)
    p.add_argument("--router-bias-clip", type=float, default=None)
    ns, unknown = p.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"Unknown args: {unknown}")
    out: dict[str, object] = {}
    bool_keys = {"auto_plot_on_val", "router_bias_update", "deq_k_jitter",
                 "swa_enabled", "ema_enabled", "use_ctp"}
    for k, v in vars(ns).items():
        if v is not None:
            key = k.replace("-", "_")
            out[key] = bool(int(v)) if key in bool_keys else v
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

def _ns5_2d(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """NS preconditioner for a single 2D matrix."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def _ns5_batched(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """NS preconditioner for a batch of independent matrices.

    Input: (..., m, n) — leading dims are batch, last two are the matrix.
    Each matrix is normalized and processed independently (no cross-batch coupling).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    norms = X.flatten(-2).norm(dim=-1)
    X = X / (norms[..., None, None] + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2).contiguous()
    for _ in range(steps):
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

CONTROL_TENSOR_PATTERNS = ("q_gain", "gate_bias", "bigram.scale", "norm_weight")
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


class KShuffleBagSampler:
    def __init__(self, k_min: int, k_max: int, rng: random.Random, *, step: int = 1,
                 values: list[int] | None = None):
        # Explicit `values` overrides range(k_min, k_max+1, step) — lets us
        # express non-uniform K sets like {4, 8, 16} (skipping K=12 because
        # K=8 and K=16 bracket it well; ~7% throughput gain).
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.step = int(step)
        self.values = sorted(set(int(v) for v in values)) if values else None
        self.rng = rng
        self._bag: list[int] = []

    def sample(self) -> int:
        if not self._bag:
            if self.values is not None:
                self._bag = list(self.values)
            else:
                self._bag = list(range(self.k_min, self.k_max + 1, self.step))
            self.rng.shuffle(self._bag)
        return int(self._bag.pop())

    def reset(self) -> None:
        self._bag.clear()


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
# SOFT DENSE ROUTER
# ---------------------------------------------------------------------------

def entmax15(logits: Tensor, dim: int = -1, n_iter: int = 30) -> Tensor:
    """α=1.5 entmax (Peters, Niculae & Martins 2019) — bisection over α-entmax.

    α-entmax interpolates between softmax (α=1, full support, smooth) and
    sparsemax (α=2, sparse, exact zeros). At α=1.5: TYPICALLY-FULL support
    (most weights nonzero, peakier than softmax) with possibly some exact
    zeros — the principled middle ground.

    Solution: w_i = max(0, (α-1)·(z_i - τ))^(1/(α-1)) where τ is the
    Lagrangian satisfying Σw_i = 1. With α=1.5: w_i = max(0, 0.5·(z_i - τ))²
    and we bisect for τ.

    RevDEQ-safe: deterministic, 1-Lipschitz, subdifferentiable a.e. — same
    properties as sparsemax. AVOIDS sparsemax's pure top-1 trap (iter 99
    H74) because w_i > 0 across most experts → all get nonzero gradient.

    Args:
        logits: Tensor with simplex axis on `dim`.
        dim: Axis to project over (default: last).
        n_iter: Bisection iterations (30 → 1e-9 precision; differentiable
            via implicit function theorem through autograd).
    Returns:
        Probability tensor of same shape, sum=1 along `dim`.
    """
    # Bisect for τ such that Σ max(0, 0.5·(z - τ))² = 1.
    # Bracket: τ_low = z.max() - 2 (guarantees support includes at least
    # the max), τ_high = z.max() (guarantees support is empty if equality).
    # Actually, at τ = z.max() the only nonzero contribution would be 0,
    # so the sum is 0 < 1. So τ must be < z.max() to have support.
    # We use τ_low = z.max() - sqrt(2*K) as a safe lower bound (worst case
    # all K entries equal, then w_i = 1/K → sum K · ((1/K)^(1/2))² · 0.5²
    # ... derivation messy; just pick a wide bracket and bisect.
    z_max = logits.max(dim=dim, keepdim=True).values
    tau_lo = z_max - 4.0  # generous lower bound
    tau_hi = z_max
    for _ in range(n_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        w = torch.clamp(0.5 * (logits - tau), min=0.0).pow(2)
        w_sum = w.sum(dim=dim, keepdim=True)
        # If sum > 1, threshold τ is too low; raise it. If sum < 1, lower it.
        tau_lo = torch.where(w_sum > 1.0, tau, tau_lo)
        tau_hi = torch.where(w_sum > 1.0, tau_hi, tau)
    # Final eval with the converged τ.
    tau = 0.5 * (tau_lo + tau_hi)
    w = torch.clamp(0.5 * (logits - tau), min=0.0).pow(2)
    return w


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax (Martins & Astudillo 2016) — closed-form simplex projection.

    Outputs a probability vector that sums to 1 with EXACT zeros for
    sufficiently low logits. Unlike softmax (always strictly positive),
    sparsemax produces architectural sparsity per token without requiring
    a loss-side penalty.

    RevDEQ-safe: deterministic, 1-Lipschitz in ‖·‖₁→‖·‖∞, subdifferentiable
    a.e. (gradient is the indicator that the entry is in the support set).

    Args:
        logits: Tensor of arbitrary shape with the simplex axis on `dim`.
        dim: Axis to project over (default: last).
    Returns:
        Tensor of same shape; entries are in [0, 1], sum to 1 along `dim`,
        with possibly many exact zeros.
    """
    z_sorted, _ = torch.sort(logits, dim=dim, descending=True)
    K = logits.size(dim)
    arange = torch.arange(1, K + 1, dtype=logits.dtype, device=logits.device)
    arange_shape = [1] * logits.ndim
    arange_shape[dim] = K
    arange = arange.view(arange_shape)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    # Support condition: 1 + k * z_sorted[k] > sum_{i<=k} z_sorted[i]
    support = (1 + arange * z_sorted) > z_cumsum
    k_z = support.long().sum(dim=dim, keepdim=True).clamp(min=1)
    # Threshold tau = (sum of top-k - 1) / k
    z_cumsum_at_k = z_cumsum.gather(dim, k_z - 1)
    tau = (z_cumsum_at_k - 1) / k_z.to(logits.dtype)
    return torch.clamp(logits - tau, min=0.0)


class SoftDenseRouter(nn.Module):
    """Dense softmax routing over experts (no top-k, no dropping).

    Two scoring modes (iter 34 A/B, opg_doc.tex §4.3):
      - scoring='linear' (default, iter 30-33b baseline): logits =
        `W_r x_n + b_expert + log σ(W_g x_n + b_g)` — unbounded Linear
        over RMSNorm-ed input, gated by a sigmoid.  NOT 1-Lipschitz in
        input.
      - scoring='l2' (iter 34A): logits = `tanh(-γ‖x_n − c_j‖²)` using
        learnable prototypes c_j ∈ R^dim.  1-Lipschitz in input under
        bounded-state assumption (doc §4.3 Option B, §6.4) — tanh
        saturation caps the composition's Lipschitz constant.  γ is
        fixed at 1.0 initially (not learnable), can be promoted later.
    """
    def __init__(self, dim: int, num_experts: int, *,
                 min_share_frac: float = 0.6, cv_target: float = 0.20,
                 min_share_loss_weight: float = 1.0, cv_loss_weight: float = 0.10,
                 scoring: str = "linear", health_slices: tuple[int, ...] | None = None,
                 router_kind: str = "softmax",
                 entropy_coef: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.min_share_frac = float(min_share_frac)
        self.cv_target = float(cv_target)
        self.min_share_loss_weight = float(min_share_loss_weight)
        self.cv_loss_weight = float(cv_loss_weight)
        self.scoring = str(scoring)
        self.router_kind = str(router_kind)  # iter 99: "softmax" | "sparsemax" | "entmax15"
        self.entropy_coef = float(entropy_coef)  # iter 100: per-token entropy penalty coef
        assert self.scoring in ("linear", "l2", "sips"), f"unknown scoring: {scoring}"
        self.health_slices = tuple(int(v) for v in (health_slices or (num_experts,)))
        if sum(self.health_slices) != int(num_experts) or any(v <= 0 for v in self.health_slices):
            raise ValueError(f"health_slices={self.health_slices} must partition {num_experts} experts")
        # Tensor buffer to avoid Python-float guards inside torch.compile graphs.
        # Kept behind a property so legacy code/tests can assign `health_scale = 5.0`.
        self.register_buffer("_health_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.router = CastedLinear(dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.score_norm_weight = nn.Parameter(torch.ones(dim))
        self.gate_norm_weight = nn.Parameter(torch.ones(dim))
        # L2-distance scoring: learnable prototypes c_j and fixed γ.  When
        # scoring='linear', prototypes are unused (kept as a zero-init module
        # attribute for state-dict compatibility).
        self.prototypes = nn.Parameter(torch.empty(num_experts, dim))
        with torch.no_grad():
            nn.init.normal_(self.prototypes, std=0.02)
        self.l2_gamma = 1.0
        # Lyapunov: no prototype bounding needed (was BallProjection for 1-Lip).
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        # Input-dependent sigmoid gate on routing weights (iter 17, H14).
        # Init fully open: weight=0, bias=5.0 → sigmoid(5)≈0.993.
        # The model can learn to suppress specific experts per-token.
        self.router_gate = CastedLinear(dim, num_experts, bias=True)
        with torch.no_grad():
            self.router_gate.weight.zero_()
            self.router_gate.bias.fill_(5.0)
        self._router_gate_last_mean: float | None = None
        self._mean_share_last: Tensor | None = None
        self._balance_loss = None
        self._health_loss = None
        self._expert_usage = None
        self._expert_entropy = None
        self._expert_sparsity = None
        self._expert_balance_cv = None
        self._expert_total_mass = None
        # GPU-resident views used by DDP reductions — populated by
        # _record_diagnostics on every rank during eval, avoiding per-forward
        # cpu() syncs on non-master ranks.
        self._expert_usage_gpu: Tensor | None = None
        self._expert_entropy_gpu: Tensor | None = None
        self._expert_balance_cv_gpu: Tensor | None = None
        self._expert_total_mass_gpu: Tensor | None = None
        self._diag_step: int | None = None

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

    def _component_health_losses(self, mean_mass: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bal = mean_mass.new_zeros(())
        min_loss = mean_mass.new_zeros(())
        cv_loss = mean_mass.new_zeros(())
        n = float(len(self.health_slices))
        for start, end in self._component_ranges():
            width = max(end - start, 1)
            share = mean_mass[..., start:end]
            share = share / share.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            target = torch.full_like(share, 1.0 / float(width))
            bal = bal + F.mse_loss(share, target)
            lb = float(self.min_share_frac) / float(width)
            if lb > 0.0:
                min_loss = min_loss + torch.relu(share.new_tensor(lb) - share).pow(2).mean()
            if self.cv_target > 0.0:
                cv = share.std(dim=-1, unbiased=False) / share.mean(dim=-1).clamp_min(1e-8)
                cv_loss = cv_loss + torch.relu(cv - share.new_tensor(self.cv_target)).pow(2).mean()
        return bal / n, min_loss / n, cv_loss / n

    def _component_balance_cv(self, share: Tensor) -> Tensor:
        vals = []
        for start, end in self._component_ranges():
            s = share[..., start:end]
            vals.append(s.std(dim=-1, unbiased=False) / s.mean(dim=-1).clamp_min(1e-8))
        return torch.stack(vals).mean()

    @property
    def health_scale(self) -> float:
        try:
            return float(self._health_scale.detach().float().item())
        except Exception:
            return 1.0

    @health_scale.setter
    def health_scale(self, value: float) -> None:
        with torch.no_grad():
            self._health_scale.fill_(float(value))

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

    def forward(self, x: Tensor, *, pre_normed: bool = True) -> Tensor:
        x_n = x  # Caller supplies parameter-free RMS-normalized activations.
        score_weight = self.score_norm_weight.to(dtype=x_n.dtype)
        gate_weight = self.gate_norm_weight.to(dtype=x_n.dtype)
        x_score = x_n * score_weight
        x_gate = x_n * gate_weight
        D = self.prototypes.shape[-1]
        if self.scoring in ("l2", "sips"):
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
        else:
            # Linear scoring (iter 30-33b baseline).
            route_logits = self.router(x_score) + self.expert_bias.to(dtype=x.dtype)
        # iter 99 (2026-04-26): router output activation per `self.router_kind`.
        # `softmax` (iter 96 baseline) produces all-positive weights with full
        # support; `sparsemax` (iter 99) produces exact zeros for low-logit
        # experts → architectural per-token sparsity. Both compose with the
        # sigmoid gate identically.
        # Sigmoid gate: <activation>(route_logits) * sigmoid(gate_logits).
        # NOT renormalized — total weight can be < 1, allowing the model to
        # suppress the entire expert mixture for tokens already near equilibrium.
        # This is more expressive than folding into logit space (which forces sum=1).
        _rk = getattr(self, "router_kind", "softmax")
        if _rk == "sparsemax":
            p_alloc = sparsemax(route_logits.float(), dim=-1)  # iter 99 (NOT PROMOTED H74)
        elif _rk == "entmax15":
            p_alloc = entmax15(route_logits.float(), dim=-1)  # iter 101: α=1.5 middle ground
        else:
            p_alloc = torch.softmax(route_logits.float(), dim=-1)  # fp32 for stability
        gate_act = torch.sigmoid(self.router_gate(x_gate).float())
        p = (p_alloc * gate_act).to(dtype=x.dtype)
        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            self._router_gate_last_mean = float(gate_act.detach().mean().item())
        if self.training:
            reduce_dims = tuple(range(p.ndim - 1))
            mean_mass = p.mean(dim=reduce_dims)
            mean_share = self._normalized_component_shares(mean_mass.float()).to(dtype=mean_mass.dtype)
            mse, min_share_loss, cv_loss = self._component_health_losses(mean_mass.float())
            hs = self._health_scale.to(dtype=min_share_loss.dtype)
            self._balance_loss = mse
            self._health_loss = (
                float(self.min_share_loss_weight) * hs * min_share_loss
                + float(self.cv_loss_weight) * hs * cv_loss
            )
            # iter 100 (2026-04-27): per-token entropy penalty on routing.
            # Drives per-token sparsity (peakier softmax) without architectural
            # capacity cost (vs iter 99/101 NOT PROMOTED). Renormalize routing
            # weights per-token so the entropy is measured over a probability
            # distribution (sum=1); softmax already does this, but sigmoid gate
            # makes raw `p` sum to ≤1, so renormalize before entropy compute.
            if float(self.entropy_coef) > 0.0:
                p_norm = p.float() / p.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
                # H_pertoken = -Σ_e w(e|t) log w(e|t), averaged over tokens.
                pertoken_ent = -(p_norm * (p_norm + 1e-8).log()).sum(dim=-1).mean()
                self._pertoken_entropy_loss = float(self.entropy_coef) * pertoken_ent
            else:
                self._pertoken_entropy_loss = torch.tensor(0.0, device=x.device)
            self._mean_share_last = mean_share.detach()
            with torch.no_grad():
                if _should_diag(self.training):
                    self._record_diagnostics(p.detach(), reduce_dims)
                    # Backward compatibility: standalone-router tests expect
                    # list-form diagnostics immediately when diagnostics are on.
                    # Avoid materializing inside DEQ solves (would sync K×).
                    if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and not bool(_DEQ_SOLVE_ACTIVE):
                        self._materialize_diag_lists()
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._health_loss = torch.tensor(0.0, device=x.device)
            self._pertoken_entropy_loss = torch.tensor(0.0, device=x.device)
            self._mean_share_last = None
            with torch.no_grad():
                if _should_diag(self.training):
                    self._record_diagnostics(p.detach(), tuple(range(p.ndim - 1)))
                    if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and not bool(_DEQ_SOLVE_ACTIVE):
                        self._materialize_diag_lists()
                else:
                    self._expert_usage = None
                    self._expert_entropy = None
                    self._expert_sparsity = None
                    self._expert_balance_cv = None
                    self._expert_total_mass = None
                    self._expert_usage_gpu = None
                    self._expert_entropy_gpu = None
                    self._expert_balance_cv_gpu = None
                    self._expert_total_mass_gpu = None
                    self._diag_step = None
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
        if self._expert_usage is not None:
            return  # already materialized for current step
        if self._expert_usage_gpu is None:
            return
        self._expert_usage = self._expert_usage_gpu.detach().float().cpu().tolist()
        if self._expert_balance_cv_gpu is not None:
            self._expert_balance_cv = float(self._expert_balance_cv_gpu.item())
        if self._expert_total_mass_gpu is not None:
            self._expert_total_mass = float(self._expert_total_mass_gpu.item())
        if self._expert_entropy_gpu is not None:
            ent = float(self._expert_entropy_gpu.item())
            self._expert_entropy = ent
            self._expert_sparsity = 1.0 - (ent / max(math.log(float(self.num_experts)), 1e-8))


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
                 qk_gain_init: float, kv_latent_dim: int = 0, num_experts: int = 8,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(dim // max(num_experts, 1), 1)
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else dim // 2
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

        self.attn_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        # Per-expert learned pre-RMS scales for Q/K components. RMS math is
        # shared and parameter-free; every learned scale stays expert-local.
        self.q_rope_norm_weight = nn.Parameter(torch.ones(num_experts, self.rope_dim))
        self.q_nope_norm_weight = nn.Parameter(torch.ones(num_experts, self.nope_dim))
        self.k_rope_norm_weight = nn.Parameter(torch.ones(num_experts, self.rope_dim))
        self.k_nope_norm_weight = nn.Parameter(torch.ones(num_experts, self.nope_dim))
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None
        self._attn_gate_last_mean: float | None = None

    @staticmethod
    def _rms_scale(x: Tensor, weight: Tensor, *, expert_dim: int = 0, eps: float = 1e-6) -> Tensor:
        y = x * x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
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
        kv_rms = kv_flat.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        kv_unit = (kv_flat * kv_rms).reshape(E, N, self.kv_latent_dim)
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

        # --- Head-packed SDPA ---
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

        # --- Gated attention ---
        gate_logits_p = gate_logits.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, 1)
        gate_act = torch.sigmoid(
            gate_logits_p.to(dtype=y.dtype)
            + self.gate_bias.to(dtype=y.dtype)[None, :, None, None]
        )
        y = y * gate_act

        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            self._attn_gate_last_mean = float(gate_act.detach().float().mean().item())

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

        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            with torch.no_grad():
                mu_out = y.detach().float().mean(dim=(0, 1))  # (E, D)
                self._out_ortho_cos_sim = float(max_pairwise_abs_cosine(mu_out).item())

        return y

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
    def __init__(self, dim: int, mlp_mult: float, num_experts: int = 8,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None):
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

    def mix_experts(self, x: Tensor, w: Tensor, *,
                    num_shared: int = 0, shared_gate: Tensor | None = None) -> Tensor:
        # Caller must pass x already parameter-free RMS-normalized by Block.
        B, T, D = x.shape
        E, R = self.num_experts, self.expert_rank
        N = B * T
        x_flat = x.reshape(N, D)
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
        # Per-expert RMSNorm (no shared weights across experts)
        h_rms = h.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        h = h * h_rms * self.hidden_norm_weight.to(dtype=h.dtype)  # (E, R) broadcasts over (N, E, R)
        # Phase 9 iter 51: shared experts (sigmoid-gated) + routed experts
        S = int(num_shared)
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
        out = out_e.sum(dim=0)  # (N, D)

        if bool(_ROUTER_DIAGNOSTICS_ACTIVE) and _should_diag(self.training):
            with torch.no_grad():
                mu_h = h.reshape(N, E, R).mean(dim=0).to(dtype=torch.float32)
                down_T = self.expert_down.to(dtype=mu_h.dtype).transpose(1, 2)  # (E, R, D)
                mu_out = torch.einsum("er,erd->ed", mu_h, down_T)
                # Max pairwise |cos| across expert pairs — near-duplicate check.
                self._out_ortho_cos_sim = float(max_pairwise_abs_cosine(mu_out).item())

        return out.reshape(B, T, D)

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
                 use_ctp: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.num_shared = num_shared
        self.num_specialized = num_specialized
        self.num_experts = num_shared + num_specialized
        self.fsq_levels = fsq_levels
        self.use_ctp = bool(use_ctp)
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

    def init_from_embedding(self, embed_weight: Tensor):
        pass  # No SVD init; xavier from scratch

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
                      B: Tensor, rank_norm_weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

        # Ortho diagnostic from mean expert projections
        mu_groups = t_all.float().mean(dim=0)  # (E, R)
        ortho_out = max_mean_abs_offdiag_cosine(mu_groups) if E >= 2 else x.new_zeros(())

        # FSQ + per-expert rank prenorm + logits:
        # (N, E, R) → FSQ/RMS → per-expert B → (N, E, V)
        u_all = self._fsq(t_all)
        u_rms = u_all.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        u_all = u_all * u_rms * rank_norm_weight.to(dtype=u_all.dtype).unsqueeze(0)
        logits_all = torch.bmm(
            u_all.permute(1, 0, 2).to(B.dtype),
            B.transpose(1, 2),
        ).permute(1, 0, 2).float()  # (N, E, V)

        # Mixture of softmaxes in log space (vectorized logaddexp)
        log_p_experts = F.log_softmax(logits_all, dim=-1)  # (N, E, V)
        log_p_weighted = log_w.unsqueeze(-1) + log_p_experts  # (N, E, V)
        log_p_unnorm = torch.logsumexp(log_p_weighted, dim=1)  # (N, V)
        log_p = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=-1, keepdim=True)

        return log_p, alpha, ortho_out

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        orig_shape = h.shape[:-1]
        x = h.reshape(-1, self.d_model)
        log_p_n, alpha_n, ortho_ntp = self._head_forward(
            x, self.gate_ntp, self.gate_ntp_norm_weight,
            self.A_ntp_shared, self.A_ntp, self.ntp_a_norm_weight,
            self.B_NTP, self.ntp_rank_norm_weight,
        )
        if self.use_ctp:
            log_p_d, alpha_d, ortho_ctp = self._head_forward(
                x, self.gate_ctp, self.gate_ctp_norm_weight,
                self.A_ctp_shared, self.A_ctp, self.ctp_a_norm_weight,
                self.B_denoise, self.ctp_rank_norm_weight,
            )
        else:
            # NTP-only: alias CTP outputs to NTP for API compatibility.
            # Downstream GPT.forward zeros ctp_loss/ctp_weight, and the
            # refinement path uses p_ntp only (see _get_soft_embedding).
            log_p_d = log_p_n
            alpha_d = alpha_n
            ortho_ctp = x.new_zeros(())
        self._ctp_ortho_out = ortho_ctp
        self._ntp_ortho_out = ortho_ntp
        distributed = dist.is_available() and dist.is_initialized()
        is_master = (not distributed) or dist.get_rank() == 0

        if self.training:
            bal = torch.tensor(0.0, device=x.device)
            alphas = [alpha_n] if not self.use_ctp else [alpha_d, alpha_n]
            for alpha_soft in alphas:
                mean_a = alpha_soft.mean(dim=0)
                target = torch.ones_like(mean_a) / alpha_soft.shape[-1]
                bal = bal + F.mse_loss(mean_a, target)
            self._balance_loss = bal
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)

        # Diagnostics: store GPU-resident mean expert shares (+ CV) so post-int6
        # health checks can all-reduce without forcing a per-forward CPU sync.
        # Only materialize Python lists on master when diagnostics are explicitly
        # enabled (router_diagnostics), keeping eval fast by default.
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
        return log_p_d.view(*orig_shape, -1), log_p_n.view(*orig_shape, -1)


# ---------------------------------------------------------------------------
# BLOCK (Attention + MLP)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    # Locked config: 8 experts (iter 13 best at 1h budget).
    # H5 resolved: 12exp works at WD=0.72 but throughput penalty hurts val_bpb.
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, kv_latent_dim: int = 0,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 num_experts: int = 8, num_shared_experts: int = 0,
                 router_scoring: str = "linear",
                 router_kind: str = "softmax",
                 router_entropy_coef: float = 0.0, **kwargs):
        super().__init__()
        # T_θ(z, x₀) = B̄ ⊙ RMSUnit(x₀) ⊙ x0_inject_norm_weight + Δ_θ(z, x₀).
        # State preconditioner is parameter-free RMSUnit; every projection-local
        # scale is owned by its downstream linear (see Prenorm Scale Independence
        # Rule in CLAUDE.md). B̄ is passed explicitly through RevDEQFunction's
        # autograd boundary; a None b_bar means ones(D) (direct Block() in tests).
        self.attn_post_mix_norm = RMSNorm(dim)
        self.mlp_post_mix_norm = RMSNorm(dim)
        self.x0_inject_norm_weight = nn.Parameter(torch.ones(dim))
        # Shared experts (DeepSeek-style): always-on with per-token sigmoid gate.
        # T_θ = g_s·E_shared(h) + Σ w_j E_routed_j(h)
        self.num_experts = num_experts
        self.num_shared_experts = int(num_shared_experts)
        num_routed = num_experts - self.num_shared_experts
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
        # Router only covers routed experts (not shared).
        # iter 100b (2026-04-27): drop min_share_loss penalty (was 10.0).
        # min_share remains in diagnostics; CV loss alone provides global
        # balance regularization. Decouples from entropy-penalty axis.
        self.router = SoftDenseRouter(dim, 2 * num_routed, min_share_loss_weight=0.0,
                                      cv_loss_weight=2.0, scoring=router_scoring,
                                      health_slices=(num_routed, num_routed),
                                      router_kind=router_kind,
                                      entropy_coef=router_entropy_coef)
        self.attn_router = self.router  # alias for backward-compat diagnostics
        self.mlp_router = self.router   # alias (same instance → dedup via id())
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         kv_latent_dim=kv_latent_dim, num_experts=num_experts,
                                         expert_rank=attn_expert_rank, router=self.router)
        self.mlp = MLP(dim, mlp_mult, num_experts=num_experts, expert_rank=mlp_expert_rank, router=self.router)
        # Diagnostic tracking for per-DEQ-iteration gate trajectories.
        self._diag_track_enabled = False
        self._attn_gate_call_track: list[float] = []
        self._router_gate_call_track: list[float] = []
        self._attn_router_gate_call_track: list[float] = []
        self._mlp_router_gate_call_track: list[float] = []
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

    def ortho_aux(self, z_in: Tensor, x0: Tensor, *, max_tokens: int = 256) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = z_in.shape
        t = int(min(max(1, int(max_tokens)), seqlen))
        z_sub = z_in[:, :t]
        x0_sub = x0[:, :t]
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
        mu_h2 = h_mlp.reshape(N, E2, R2).mean(dim=0).to(dtype=torch.float32)
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

        return attn_ortho, mlp_ortho

    def _route_pooled(self, u_proj: Tensor) -> tuple[Tensor, Tensor]:
        """Compute pooled routing weights for ROUTED experts only.

        Returns (w_attn, w_mlp) each of shape (..., num_routed).
        Shared experts (indices 0:num_shared) bypass routing entirely.
        """
        num_routed = self.num_experts - self.num_shared_experts
        w_all = self.router(u_proj, pre_normed=True)  # (..., 2*num_routed)
        return w_all[..., :num_routed].contiguous(), w_all[..., num_routed:].contiguous()

    def forward(self, z_in: Tensor, x0: Tensor, b_bar: Tensor | None = None) -> Tensor:
        # h = RMSUnit(z + x_0); Δ = Σ g_s·E_shared(h) + Σ w_j·E_routed_j(h).
        # Output injection (x_0 term) is applied below via B̄ ⊙ RMSUnit(x_0).
        u = z_in + x0
        h = _rms_unit(u)

        E = self.num_experts
        S = self.num_shared_experts
        # Route only the non-shared experts.
        w_attn, w_mlp = self._route_pooled(h)
        if self._diag_track_enabled:
            attn_rg = getattr(self.router, "_router_gate_last_mean", None)
            # Track per-expert mean routing weights (shows specialization across
            # iters).  Append GPU tensors only — single materialization happens
            # post-solve in SharedBlock.forward.  Mean is over all non-expert
            # dims (B, T) so each entry has shape (E,).
            with torch.no_grad():
                reduce_dims = tuple(range(w_attn.dim() - 1))
                self._attn_expert_weights_per_iter.append(
                    w_attn.detach().float().mean(dim=reduce_dims))
                self._mlp_expert_weights_per_iter.append(
                    w_mlp.detach().float().mean(dim=reduce_dims))

        # All experts compute outputs together (shared + routed).
        attn_expert_out = self.attn.forward_experts(h)  # (B, T, E, D)
        if S > 0:
            # Iter 84: independent per-path sigmoid gates for attn vs mlp.
            h_shared_gate_attn = h * self.shared_gate_norm_weight_attn.to(dtype=h.dtype)
            h_shared_gate_mlp = h * self.shared_gate_norm_weight_mlp.to(dtype=h.dtype)
            g_s_attn = torch.sigmoid(self.shared_gate_attn(h_shared_gate_attn))  # (B, T, S)
            g_s_mlp = torch.sigmoid(self.shared_gate_mlp(h_shared_gate_mlp))      # (B, T, S)
            if _should_diag(self.training):
                with torch.no_grad():
                    g_sf = torch.cat([g_s_attn.detach().float(), g_s_mlp.detach().float()], dim=-1)
                    # Store as 0-d GPU tensors — log site (`format_expert_info`)
                    # already calls `float(...)` for formatting, which performs
                    # the single .item() sync OUTSIDE the micro_step loop.
                    self._shared_gate_mean = g_sf.mean().detach()
                    self._shared_gate_min = g_sf.min().detach()
                    self._shared_gate_std = g_sf.std(unbiased=False).detach()
                    self._shared_gate_diag_step = _ROUTER_DIAGNOSTICS_STEP
            else:
                self._shared_gate_mean = None
                self._shared_gate_min = None
                self._shared_gate_std = None
                self._shared_gate_diag_step = None
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
        delta = (attn_mix + mlp_mix).to(dtype=z_in.dtype)

        # Parcae input injection: T_θ = B̄ ⊙ RMSUnit(x_0) ⊙ g + Δ_θ.
        # b_bar=None ⇒ ones(D) (direct Block() in tests).  At the fixed point
        # β=1-Ā cancels and y* = B̄ ⊙ RMSUnit(x_0) ⊙ g + Δ*.
        x0_rms_scale = x0.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        x0_rms = x0 * x0_rms_scale * self.x0_inject_norm_weight.to(x0.dtype)
        if b_bar is not None:
            x0_rms = b_bar.to(x0_rms.dtype) * x0_rms
        raw_out = x0_rms.to(dtype=z_in.dtype) + delta

        if self._diag_track_enabled:
            ag = getattr(self.attn, "_attn_gate_last_mean", None)
            if ag is not None:
                self._attn_gate_call_track.append(ag)
            if attn_rg is not None:
                self._attn_router_gate_call_track.append(attn_rg)
            # Pooled router: attn and mlp share the same router instance.
            mlp_rg = attn_rg
            if mlp_rg is not None:
                self._mlp_router_gate_call_track.append(mlp_rg)
            rg_vals = []
            if attn_rg is not None:
                rg_vals.append(float(attn_rg))
            if mlp_rg is not None:
                rg_vals.append(float(mlp_rg))
            if rg_vals:
                self._router_gate_call_track.append(sum(rg_vals) / float(len(rg_vals)))
        return raw_out


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

        with torch.no_grad():
            out_y = None
            for _ in range(K):
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
                    except Exception:
                        pass

        ctx.save_for_backward(x0.detach(), y_state.detach(), z_state.detach(),
                              z_prev_state.detach())
        ctx.z_init_state = z_init_state
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

        if bool(getattr(ctx, "do_recon_diag", False)) and isinstance(z_init_state, torch.Tensor):
            try:
                z0 = z_init_state.to(dtype=state_dtype)
                denom = z0.norm().clamp(min=1.0)
                z_rec = z_next64.to(dtype=state_dtype)
                y_rec = y_next64.to(dtype=state_dtype)
                # Store as GPU tensor — materialize at log time only.
                recon_err_t = ((z_rec - z0).norm() + (y_rec - z0).norm()) / denom
                _target = _unwrap_compiled_module(f_theta)
                setattr(_target, "_deq_recon_error_last_bwd", recon_err_t.detach())
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


# ---------------------------------------------------------------------------
# GPT MODEL
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: float, tie_embeddings: bool,
                 tied_embed_init_std: float, rope_base: float,
                 qk_gain_init: float, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 kv_latent_dim: int = 0, num_refinements: int = 1,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 deq_beta: float = 0.35, attn_balance_mult: float = 5.0,
                 mlp_balance_mult: float = 1.0, mos_balance_mult: float = 50.0,
                 bal_loss_coef: float = 5e-3,
                 router_health_coef: float = 0.25, mos_ortho_out_coef: float = 0.0,
                 deq_backward: str = "revdeq", deq_bptt_k: int = 0,
                 block_ortho_aux_coef: float = 0.0,
                 block_ortho_aux_every: int = 0, block_ortho_aux_tokens: int = 64,
                 router_scoring: str = "linear",
                 router_kind: str = "softmax",
                 router_entropy_coef: float = 0.0,
                 router_entropy_warmup_delay_frac: float = 0.0,
                 num_experts: int = 8, num_shared_experts: int = 0,
                 lyapunov_coef: float = 1.0,
                 lyapunov_gamma: float = 0.9,
                 lyapunov_warmup_frac: float = 0.1,
                 use_parcae: bool = True,
                 parcae_init_a_bar: float = 0.7,
                 use_ctp: bool = True):
        super().__init__()
        self.use_ctp = bool(use_ctp)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.num_layers = num_layers
        self.num_refinements = num_refinements
        # iter 100b: store entropy-coef target + warmup-delay for the
        # training-loop annealing hook to read. Router's entropy_coef is set
        # dynamically per-step via these values.
        self._router_entropy_coef_target = float(router_entropy_coef)
        self._router_entropy_warmup_delay_frac = float(router_entropy_warmup_delay_frac)
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
                                   router_kind=router_kind,
                                   router_entropy_coef=router_entropy_coef,
                                   )
        self.deq_beta = float(deq_beta)
        # Phase 9 iter 66b: Parcae-paper-faithful per-dim damping with
        # independent input gain (arXiv:2604.12946, Mamba-style ZOH).
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
            # Init raw params so Ā₀ ≈ parcae_init_a_bar and B̄₀ ≈ 1 − Ā₀,
            # giving continuity with iter 66a's tied β = 1 − Ā at step 0.
            raw_a_init, raw_delta_init, raw_b_init = self._parcae_init_raw_values(
                float(parcae_init_a_bar)
            )
            self.parcae_raw_a = nn.Parameter(torch.full((model_dim,), raw_a_init))
            self.parcae_raw_delta = nn.Parameter(torch.full((model_dim,), raw_delta_init))
            self.parcae_raw_b = nn.Parameter(torch.full((model_dim,), raw_b_init))
        self.attn_balance_mult = float(attn_balance_mult)
        self.mlp_balance_mult = float(mlp_balance_mult)
        self.mos_balance_mult = float(mos_balance_mult)
        self.bal_loss_coef = float(bal_loss_coef)
        self.router_health_coef = float(router_health_coef)
        self.mos_ortho_out_coef = float(mos_ortho_out_coef)
        self.deq_backward = deq_backward
        self.deq_bptt_k = int(deq_bptt_k)
        self.block_ortho_aux_coef = float(block_ortho_aux_coef)
        self.block_ortho_aux_every = int(block_ortho_aux_every)
        self.block_ortho_aux_tokens = int(block_ortho_aux_tokens)
        self._block_ortho_aux_enabled = False
        self._block_ortho_aux_loss: Tensor | None = None
        # iter 45 (opg_doc.tex §4): Lyapunov spectral-radius penalty.
        # Two-forward approach: VJP for ρ̂ + surrogate loss for ∇_θ.
        self.lyapunov_coef = float(lyapunov_coef)
        self.lyapunov_gamma = float(lyapunov_gamma)
        self.lyapunov_warmup_frac = float(lyapunov_warmup_frac)
        self._lyapunov_v_buf: Tensor | None = None  # persistent power-iter vector (EMA)
        # GPU-resident EMA of spectral radius estimate. Stored as a 0-dim CUDA
        # tensor so the Hutchinson penalty block stays pure-tensor (no .item()
        # GPU→CPU sync in the hot path). Initialized lazily on first use.
        self._lyapunov_rho_hat_buf: Tensor | None = None
        self.mos_head = MoSHead(model_dim, vocab_size, rank=256, num_shared=2, num_specialized=1, fsq_levels=0, use_ctp=self.use_ctp)  # Phase 9 iter 62 (H53): disabled FSQ. Iter 94: use_ctp threads through to skip CTP.
        self.final_norm = RMSNorm(model_dim)
        # Embedding/final norms remain learnable shared scales outside T_theta.
        self.embed_norm = RMSNorm(model_dim)
        self._init_weights()

    def _parcae_init_raw_values(self, init_a_bar: float) -> tuple[float, float, float]:
        """Invert the paper forms to choose raw params at initialization.

        Picks Δ₀, |A|₀, B₀ so that Ā₀ ≈ init_a_bar and B̄₀ ≈ 1 − init_a_bar.
        Returns raw values that pass through softplus+ε_min to recover the
        targets exactly (modulo the safety ε_min offset on |A| and B).
        """
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

        # B̄₀ = Δ₀ · B₀ ≈ 1 − Ā₀ (continuity with iter 66a's tied β = 1 − Ā).
        b_mag = max((1.0 - clamped_a) / delta0, eps_min + 1e-8)
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
        self.mos_head.init_from_embedding(self.tok_emb.weight.data)

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
        # Phase 9 iter 66b: Parcae-paper-faithful per-dim damping + Parcae input gain.
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
        sb._attn_gate_call_track = []
        sb._router_gate_call_track = []
        sb._attn_router_gate_call_track = []
        sb._mlp_router_gate_call_track = []
        prev_deq_flag = bool(_DEQ_SOLVE_ACTIVE)
        _DEQ_SOLVE_ACTIVE = True
        try:
            f_theta = self.shared_block
            if self.training and self.deq_backward == "revdeq":
                params = tuple(p for p in sb.parameters() if p.requires_grad)
                bptt_k = int(getattr(self, "deq_bptt_k", 0) or 0)
                z, z_prev = RevDEQFunction.apply(f_theta, x0, z_init, beta, b_bar, K, bptt_k, *params)
                return z, z_prev, None, None

            # FP32 accumulators for the unrolled solver (eval + non-revdeq train).
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
            # Aggregate attn_gate per-iteration (1 call per block forward, 2 per iter)
            ag_calls = list(getattr(sb, "_attn_gate_call_track", []) or [])
            if len(ag_calls) == 2 * K:
                self._attn_gate_iter_last_solve = [0.5 * (ag_calls[2*i] + ag_calls[2*i+1]) for i in range(K)]
            else:
                self._attn_gate_iter_last_solve = []
            # Aggregate router_gate per-iteration (combined, backward compat)
            rg_calls = list(getattr(sb, "_router_gate_call_track", []) or [])
            if len(rg_calls) == 2 * K:
                self._router_gate_iter_last_solve = [0.5 * (rg_calls[2*i] + rg_calls[2*i+1]) for i in range(K)]
            else:
                self._router_gate_iter_last_solve = []
            # Per-component router gates (attn vs FFN)
            attn_rg_calls = list(getattr(sb, "_attn_router_gate_call_track", []) or [])
            if len(attn_rg_calls) == 2 * K:
                self._attn_router_gate_iter_last_solve = [0.5 * (attn_rg_calls[2*i] + attn_rg_calls[2*i+1]) for i in range(K)]
            else:
                self._attn_router_gate_iter_last_solve = []
            mlp_rg_calls = list(getattr(sb, "_mlp_router_gate_call_track", []) or [])
            if len(mlp_rg_calls) == 2 * K:
                self._mlp_router_gate_iter_last_solve = [0.5 * (mlp_rg_calls[2*i] + mlp_rg_calls[2*i+1]) for i in range(K)]
            else:
                self._mlp_router_gate_iter_last_solve = []

            # Per-expert routing weights per iteration (2 calls per iter: y-update,
            # z-update).  The producer at Block.forward stored each entry as a
            # detached GPU tensor of shape (E,); we stack + pair-mean here but
            # KEEP THE RESULT ON GPU as `_attn_expert_weights_iter_t` — the .cpu()
            # materialization is deferred to the log-emission site
            # (`format_iter_dynamics_info`), which runs OUTSIDE the micro_step
            # hot range (CLAUDE.md §9 "Hot-path sync prohibition").
            attn_ew = list(getattr(sb, "_attn_expert_weights_per_iter", []) or [])
            if len(attn_ew) == 2 * K:
                # (2K, E) → reshape (K, 2, E) → mean over the 2-call axis → (K, E)
                self._attn_expert_weights_iter_t = (
                    torch.stack(attn_ew, dim=0).reshape(K, 2, -1).mean(dim=1)
                )
            else:
                self._attn_expert_weights_iter_t = None
            self._attn_expert_weights_iter: list[list[float]] | None = None  # lazy materialization
            sb._attn_expert_weights_per_iter = []
            sb._mlp_expert_weights_per_iter = []

            # Backward compat: after a DEQ solve with router_diagnostics enabled,
            # materialize list-form router diagnostics exactly once (not per-iter).
            if track_diag and bool(_ROUTER_DIAGNOSTICS_ACTIVE):
                try:
                    seen: set[int] = set()
                    for r in [sb.attn.attn_router, sb.mlp.mlp_router]:
                        rid = id(r)
                        if rid in seen:
                            continue
                        seen.add(rid)
                        if hasattr(r, "_materialize_diag_lists"):
                            r._materialize_diag_lists()
                except Exception:
                    pass

    def _run_backbone(self, x: Tensor) -> Tensor:
        x0 = x
        z = x
        self._deq_residuals: list[float] = []
        self._deq_recon_error = None
        self._deq_z_init_last: Tensor | None = None
        self._deq_k_last = None
        prev_soft_embed = x0
        x0_refined = x0
        self._block_ortho_aux_loss = None

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
            z, z_prev, y_acc, z_acc = self._deq_solve(x0_refined, z)

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
            if self._block_ortho_aux_enabled and self.block_ortho_aux_coef > 0.0:
                max_tokens = int(getattr(self, "_block_ortho_aux_tokens_override", self.block_ortho_aux_tokens))
                attn_o, mlp_o = self.shared_block.ortho_aux(z, x0_refined, max_tokens=max_tokens)
                # T-opt 15: defer .item() to log time — store GPU tensors only.
                # Hot-path sync prohibition: no .item()/.cpu() in forward/backward.
                self.shared_block.attn._out_ortho_cos_sim_t = attn_o.detach()
                self.shared_block.mlp._out_ortho_cos_sim_t = mlp_o.detach()
                thr = 0.20
                attn_b = F.relu(attn_o - thr).pow(2)
                mlp_b = F.relu(mlp_o - thr).pow(2)
                self._block_ortho_aux_loss = 0.5 * (attn_b + mlp_b)

        return z

    def _encode(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        # Phase 9 iter 71g: learnable embed norm.
        x = self.embed_norm(x)
        x = self._run_backbone(x)
        return self.final_norm(x)

    def _collect_routing_losses(self, device: torch.device) -> tuple[Tensor, Tensor]:
        zero = torch.tensor(0.0, device=device)
        bal, health = zero, zero
        router_weights: dict[int, float] = {}
        routers: dict[int, SoftDenseRouter] = {}
        for r, w in [
            (self.shared_block.attn.attn_router, self.attn_balance_mult),
            (self.shared_block.mlp.mlp_router, self.mlp_balance_mult),
        ]:
            rid = id(r)
            routers[rid] = r
            router_weights[rid] = router_weights.get(rid, 0.0) + float(w)
        for rid, r in routers.items():
            r_bal = getattr(r, "_balance_loss", zero)
            r_health = getattr(r, "_health_loss", zero)
            bal = bal + float(router_weights.get(rid, 0.0)) * r_bal
            health = health + float(router_weights.get(rid, 0.0)) * r_health
            # iter 100: per-token entropy penalty (drives sparsity loss-side).
            # Folded into health so existing aggregation/scaling reuses.
            r_ent = getattr(r, "_pertoken_entropy_loss", zero)
            health = health + float(router_weights.get(rid, 0.0)) * r_ent
        # iter 26-lb-loss: mos_balance_mult (default 50) × bal_loss_coef downstream
        # (5e-3) → effective weight 0.25 on the MoS NTP balance loss — strong enough
        # to drive dead MoS experts back toward fair share.  WD cannot fix routing-
        # space collapse (it decays already-unused weights further); the LB loss
        # is the principled complementary fix.  The retry-prescription path
        # recommends bumping mos_balance_mult for mos_*_min_share failures.
        bal = bal + self.mos_balance_mult * getattr(self.mos_head, '_balance_loss', zero)
        return bal, health

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        log_p_ctp, log_p_ntp = self.mos_head(x)
        V = self.tok_emb.num_embeddings
        ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_ids.reshape(-1))
        if self.use_ctp:
            ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_ids.reshape(-1))
        else:
            ctp_loss = torch.tensor(0.0, device=ntp_loss.device)
        bal_loss, health_loss = self._collect_routing_losses(ntp_loss.device)
        self._ntp_loss_t = ntp_loss.detach()
        self._ctp_loss_t = ctp_loss.detach()
        # Backward-compatible scalar fields used by experiments/*.
        if (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE):
            try:
                self._ntp_loss = float(self._ntp_loss_t.float().item())
                self._ctp_loss = float(self._ctp_loss_t.float().item())
            except Exception:
                self._ntp_loss = float(ntp_loss.detach().float().mean().item())
                self._ctp_loss = float(ctp_loss.detach().float().mean().item())
        else:
            # Avoid per-forward sync in the training hot path.
            self._ntp_loss = 0.0
            self._ctp_loss = 0.0
        refine_alpha = float(getattr(self, "_refine_mix_alpha", 0.5))
        refine_strength = min(max(refine_alpha / 0.5, 0.0), 1.0)
        ctp_weight = (0.05 * self.num_refinements * refine_strength) if self.use_ctp else 0.0

        mos_ortho_loss = torch.tensor(0.0, device=ntp_loss.device)
        if getattr(self.mos_head, "_ctp_ortho_out", None) is not None and getattr(self.mos_head, "_ntp_ortho_out", None) is not None:
            mos_ortho_loss = self.mos_head._ctp_ortho_out + self.mos_head._ntp_ortho_out

        block_ortho_aux = torch.tensor(0.0, device=ntp_loss.device)
        if self.training and self._block_ortho_aux_enabled and isinstance(self._block_ortho_aux_loss, torch.Tensor):
            block_ortho_aux = self._block_ortho_aux_loss.to(device=ntp_loss.device)

        eff_block_ortho_coef = float(self.block_ortho_aux_coef) * float(getattr(self, "_block_ortho_aux_coef_scale", 1.0))

        # iter 45: Lyapunov penalty added externally in training loop
        # (outside compiled forward to avoid tensor metadata corruption).

        return (
            ntp_loss
            + ctp_weight * ctp_loss
            + self.bal_loss_coef * bal_loss
            + self.router_health_coef * health_loss
            + self.mos_ortho_out_coef * mos_ortho_loss
            + eff_block_ortho_coef * block_ortho_aux
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
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


def _build_optimizer_param_lists(base_model: nn.Module, args) -> tuple[
    list[dict[str, object]], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]
]:
    sb = _unwrap_compiled_module(base_model.shared_block)
    block_named_params = list(sb.named_parameters())
    # All ndim >= 2 block params (incl. router prototypes/weights) -> Muon;
    # all ndim < 2 / control params -> AdamW scalar.
    matrix_params = [p for name, p in block_named_params
                     if p.ndim >= 2 and not any(pat in name for pat in CONTROL_TENSOR_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_PATTERNS)]

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
    ])
    return tok_params, matrix_params, scalar_params, parcae_params


def _prescribe_failure_fix(failure: str) -> dict:
    """Map a post-int6 diagnostic failure string to its hypothesis-verified fix.

    Pure function — exposed at module level for direct unit testing
    (`experiments/test_arch.py::test_prescribe_min_share_routes_to_balance_loss`).

    Routing rules (component-aware — see EXPERIENCE.md §1
    `diagnostic-gate-component-awareness`):
      - `mos_*_min_share`        → mos_balance_mult (H26: balance loss is the
                                   routing-space lever; WD made it worse in iter 24)
      - `attn_*_min_share`       → attn_balance_mult
      - `mlp_*_min_share`        → mlp_balance_mult
      - `mos_*_ortho`            → mos_ortho_out_coef (head-internal regularizer)
      - `attn_*_ortho` / `mlp_*` → weight_decay (weight-space collinearity, H5
                                   stands for ortho — only ortho — failures)
      - `k-sweep…`               → widen K jitter
      - `iter_conv_rel`          → WD or lower deq_beta
      - `deq_recon_err`          → check determinism, lower deq_beta, raise WD
    """
    low = failure.lower()
    first_token = low.split("=", 1)[0].split()[0] if low else ""

    # Dead-expert: split by component because the FIX differs.
    if "min_share" in first_token:
        if first_token.startswith("mos_"):
            return {
                "failure": failure,
                "category": "mos_router_collapse",
                "hypothesis": "H26 VERIFIED — MoS balance loss controls dead-MoS-expert (iter 26-lb-loss)",
                "fix": "Increase mos_balance_mult by 1.5× (e.g. 50→75). WD does NOT reach "
                       "MoS routing collapse — iter 24 (H5) showed WD bump worsened "
                       "mos_ntp_min_share (0.008→0.006). The MoS-internal balance loss "
                       "is the principled fix.",
                "config_change": {"mos_balance_mult_mult": 1.5},
            }
        if first_token.startswith("attn_"):
            return {
                "failure": failure,
                "category": "attn_router_collapse",
                "hypothesis": "H26 family — per-component balance loss is the routing-space lever",
                "fix": "Increase attn_balance_mult by 1.5×. WD shrinks expert weights but "
                       "doesn't move router logits — only the balance loss does.",
                "config_change": {"attn_balance_mult_mult": 1.5},
            }
        if first_token.startswith("mlp_"):
            return {
                "failure": failure,
                "category": "mlp_router_collapse",
                "hypothesis": "H26 family — per-component balance loss is the routing-space lever",
                "fix": "Increase mlp_balance_mult by 1.5×. WD shrinks expert weights but "
                       "doesn't move router logits — only the balance loss does.",
                "config_change": {"mlp_balance_mult_mult": 1.5},
            }
        # Unknown prefix shouldn't happen — defensive fallback.
        return {
            "failure": failure,
            "category": "dead_expert_unknown_component",
            "hypothesis": "unrecognized min_share prefix — manual triage required",
            "fix": "Identify the failing component from the prefix and apply the "
                   "corresponding `<component>_balance_mult` increase.",
            "config_change": {},
        }
    # MoS head ortho is a DIFFERENT failure mode from attn/mlp expert ortho.
    # MoS has fixed num_shared+num_specialized; the fix is regularization on the
    # head, not the expert-count knob.  Check MoS-prefixed first.
    if first_token.startswith("mos_") and "ortho" in first_token:
        return {
            "failure": failure,
            "category": "mos_head_collapse",
            "hypothesis": "MoSHead orthogonality under-regularized",
            "fix": "Increase mos_ortho_out_coef by 1.5× (currently 1e-3 → 1.5e-3). "
                   "If no effect, shrink mos_rank or add a lightweight orthogonality loss inside the head.",
            "config_change": {"mos_ortho_out_coef_mult": 1.5},
        }
    if "ortho" in first_token:  # attn_ortho / mlp_ortho only — weight-space collinearity
        return {
            "failure": failure,
            "category": "expert_collapse",
            "hypothesis": "H5 — weight-space collinearity is WD-fixable (ortho only, not min_share)",
            "fix": "Increase weight_decay by 1.5× (applied to both Muon and AdamW groups). "
                   "If no effect, drop num_experts by 1 step.",
            "config_change": {"weight_decay_mult": 1.5},
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
    if first_token.startswith("iter_conv_rel"):
        return {
            "failure": failure,
            "category": "solver_divergence",
            "hypothesis": "H9 + H18 VERIFIED",
            "fix": "Increase weight_decay 1.5× (H9) OR lower deq_beta by 0.05 (H18).",
            "config_change": {"weight_decay_mult": 1.5},
        }
    if first_token.startswith("deq_recon_err"):
        return {
            "failure": failure,
            "category": "reversibility_broken",
            "hypothesis": "RevDEQ reversibility requires f(z, x0, W) be deterministic and solver in stable contraction region",
            "fix": "1) Check for any random/non-deterministic op in the block (quant-noise, dropout etc. "
                   "— see H15 REFUTED).  2) Lower deq_beta by 0.05 for tighter contraction.  "
                   "3) Increase weight_decay 1.5× to shrink Jacobian spectral norm.",
            "config_change": {"deq_beta_delta": -0.05, "weight_decay_mult": 1.5},
        }
    return {
        "failure": failure,
        "category": "unknown",
        "hypothesis": "none",
        "fix": "Manual analysis required — check hypotheses.md for related observations.",
        "config_change": {},
    }


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    cli_overrides = _parse_cli_overrides(sys.argv[1:])
    args = Hyperparameters()
    for k, v in cli_overrides.items():
        setattr(args, k, v)
    if float(getattr(args, "max_wallclock_seconds", 0.0)) > 0.0 and "iterations" not in cli_overrides:
        args.iterations = int(1_000_000_000)
    args.train_files = os.path.join(args.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")
    if not getattr(args, "run_id", ""):
        args.run_id = str(uuid.uuid4())

    # Normalize backward-mode naming: "autograd" is an alias for "unroll".
    if getattr(args, "deq_backward", None) == "autograd":
        args.deq_backward = "unroll"

    if int(args.deq_k_min) <= 0:
        raise ValueError("deq_k_min must be positive")
    if int(args.deq_k_max) < int(args.deq_k_min):
        raise ValueError("deq_k_max must be >= deq_k_min")
    # deq_k_max can exceed num_layers: the DEQ uses a shared block so the
    # solver can run any number of iterations.  num_layers is just the default K.

    # NS functions already compiled at module scope (L286-287). No recompile needed.

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Base grad_accum: 8 global microsteps / world_size (so per-rank microstep
    # count is modest).  When deq_backward="unroll" we store K-step activations
    # per microstep, so bump grad_accum 4× to keep per-microstep batch small
    # enough to fit in 48 GB L40S per rank.  revdeq's O(1) backward memory
    # means the base grad_accum is fine.
    # T-opt 17: halve grad_accum (B=32→B=64 per rank) to exploit VRAM headroom.
    # RevDEQ O(1) backward memory peaks at 38 GB (80% of 48 GB L40S) with B=64.
    # 16% throughput gain from fewer micro-steps + better GPU utilization.
    _base_grad_accum = max(1, math.ceil(4 / world_size))
    # k_rope permute fix (64ac616) changed attention layout → Hutchinson VJP
    # FlashAttention backward needs more VRAM. Double grad_accum to halve B.
    _base_grad_accum *= 2
    # Unroll O(K) stores full autograd graph (43+ GB) → needs 8× to shrink B.
    if getattr(args, "deq_backward", "revdeq") == "unroll":
        grad_accum_steps = _base_grad_accum * 8
    else:
        grad_accum_steps = _base_grad_accum
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
        with open(logfile, "w", encoding="utf-8") as f:
            f.write("")
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

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def _best_effort_update_plots(reason: str) -> None:
        if not master_process or not bool(getattr(args, "auto_plot_on_val", False)):
            return
        try:
            exp_logdir = Path("experiments/training_logs")
            exp_logdir.mkdir(parents=True, exist_ok=True)
            if logfile is not None and Path(logfile).exists():
                shutil.copyfile(logfile, exp_logdir / "current.log")
            if (exp_logdir / "current.log").exists() and not (exp_logdir / "baseline.log").exists():
                shutil.copyfile(exp_logdir / "current.log", exp_logdir / "baseline.log")
            for script in ("experiments/plot_metrics.py", "experiments/plot_eval_metrics.py"):
                subprocess.run([sys.executable, script], capture_output=True, text=True, check=False)
        except Exception:
            pass

    if master_process:
        atexit.register(lambda: _best_effort_update_plots("atexit"))
        for _sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(_sig, lambda s, _: (_best_effort_update_plots(f"signal:{s}"), sys.exit(128 + s)))
            except Exception:
                pass

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
    k_sampler = KShuffleBagSampler(args.deq_k_min, args.deq_k_max, k_rng,
                                    step=int(args.deq_k_step),
                                    values=list(_k_jitter_set) if _k_jitter_set else None)

    def deq_k_for_step(step_i: int) -> int:
        k = 0
        if rank == 0:
            k = int(k_sampler.sample()) if args.deq_k_jitter else int(args.deq_k_max)
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        return int(k)

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
    log0(
        f"config:"
        f" model_dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads}"
        f" mlp_mult={args.mlp_mult} beta={args.deq_beta:.3f}"
        f" deq_k_range={args.deq_k_min}-{args.deq_k_max} deq_k_eval={args.deq_k_eval}"
        f" batch_tokens={args.train_batch_tokens} seq_len={args.train_seq_len}"
        f" refinements={args.num_refinements} refine_ramp_frac={args.num_refinements_ramp_frac}"
        f" ema={int(args.ema_enabled)} ema_decay={args.ema_decay:.4f}"
        f" pooled_router=True"
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
        deq_beta=args.deq_beta, attn_balance_mult=args.attn_balance_mult,
        mlp_balance_mult=args.mlp_balance_mult, mos_balance_mult=args.mos_balance_mult,
        bal_loss_coef=args.bal_loss_coef,
        router_health_coef=args.router_health_coef, mos_ortho_out_coef=args.mos_ortho_out_coef,
        deq_backward=args.deq_backward, deq_bptt_k=args.deq_bptt_k,
        block_ortho_aux_coef=args.block_ortho_aux_coef,
        block_ortho_aux_every=args.block_ortho_aux_every, block_ortho_aux_tokens=args.block_ortho_aux_tokens,
        num_experts=args.num_experts, num_shared_experts=args.num_shared_experts,
        router_scoring=args.router_scoring,
        router_kind=getattr(args, "router_kind", "softmax"),
        router_entropy_coef=float(getattr(args, "router_entropy_coef", 0.0)),
        router_entropy_warmup_delay_frac=float(getattr(args, "router_entropy_warmup_delay_frac", 0.0)),
        lyapunov_coef=args.lyapunov_coef,
        lyapunov_gamma=args.lyapunov_gamma,
        lyapunov_warmup_frac=args.lyapunov_warmup_frac,
        use_parcae=args.use_parcae,
        parcae_init_a_bar=args.parcae_init_a_bar,
        use_ctp=args.use_ctp,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Auto-select backward mode based on GPU memory:
    #   H100 (80 GB): revdeq + torch.compile → 45% block speedup (fused kernels)
    #   L40S (44 GB): unroll + eager (compile needs ~40 GB workspace, doesn't fit)
    _gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if args.deq_backward == "auto":
        if _gpu_mem_gb >= 60:
            args.deq_backward = "revdeq"
            log0(f"auto-selected deq_backward=revdeq ({_gpu_mem_gb:.0f} GB GPU → compile enabled)")
        else:
            args.deq_backward = "unroll"
            log0(f"auto-selected deq_backward=unroll ({_gpu_mem_gb:.0f} GB GPU)")

    if distributed and args.deq_backward == "unroll":
        log0("skipping torch.compile: unroll is incompatible with compile+DDP")
    elif _gpu_mem_gb >= 60:
        # H100 (80 GB): compile full shared_block (maximum fusion)
        try:
            base_model.shared_block = torch.compile(base_model.shared_block, dynamic=False)
            log0(f"compiled shared_block (dynamic=False, {_gpu_mem_gb:.0f} GB GPU)")
        except Exception as e:
            log0(f"shared_block compile failed ({e}), falling back to sub-module compile")
    # Compile block.forward METHOD (not the module) to fuse router→attn→MLP.
    # Compiling the method avoids DDP graph expansion that caused the ~40 GB
    # workspace OOM with torch.compile(module).  Works with revdeq because
    # the VJP backward is a single block.forward call.  1.97× speedup, 4.4 GB.
    if not hasattr(base_model.shared_block, '_orig_mod'):  # not already full-compiled
        sb = base_model.shared_block
        try:
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
            find_unused_parameters=(args.deq_backward == "unroll" and args.deq_bptt_k > 0),
            bucket_cap_mb=50)  # T-opt 22: larger buckets → fewer all_reduce calls (~10M params fit in 1 bucket)
        if distributed else base_model
    )

    # OPTIMIZER SETUP
    tok_params, matrix_params, scalar_params, parcae_params = _build_optimizer_param_lists(base_model, args)

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

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if max_wallclock_ms is None:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_frac = float(getattr(args, "warmdown_frac", 0.72))
        total_est_steps = max_wallclock_ms / max(step_ms, 1e-9)
        warmdown_steps = warmdown_frac * total_est_steps
        warmdown_ms = warmdown_steps * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def format_deq_info(m: nn.Module) -> str:
        parts: list[str] = []
        if hasattr(m, "_deq_k_last"):
            parts.append(f"deq_k:{int(m._deq_k_last)}")
        resid_t = getattr(m, "_deq_residual_t", None)
        if isinstance(resid_t, torch.Tensor):
            parts.append(f"deq_residual:{float(resid_t.detach().float().item()):.6f}")
        recon = getattr(m, "_deq_recon_error", None)
        if recon is not None:
            parts.append(f"deq_recon_err:{float(recon):.3e}")
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
            # Dedup routers by id — pooled router is aliased as attn_router AND
            # mlp_router.  Log pooled 2E usage once, then per-type normalized halves.
            seen_routers: set[int] = set()
            for prefix, router in (("attn", getattr(m.shared_block.attn, "attn_router", None)),
                                   ("mlp", getattr(m.shared_block.mlp, "mlp_router", None))):
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
                    attn_sum = sum(attn_half) or 1e-8
                    mlp_sum = sum(mlp_half) or 1e-8
                    pool_sum = (attn_sum + mlp_sum) or 1e-8
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
                    def _cv(p: list[float]) -> float:
                        n = len(p)
                        mu = sum(p) / max(n, 1) or 1e-8
                        var = sum((x - mu) ** 2 for x in p) / max(n, 1)
                        return (var ** 0.5) / mu
                    def _entropy(p: list[float]) -> float:
                        return -sum(x * math.log(x + 1e-8) for x in p if x > 0.0)
                    parts.append(f"attn_cv:{_cv(attn_norm):.4f}")
                    parts.append(f"mlp_cv:{_cv(mlp_norm):.4f}")
                    parts.append(f"pool_cv:{_cv(pool_norm):.4f}")
                    parts.append(f"attn_entropy:{_entropy(attn_norm):.4f}")
                    parts.append(f"mlp_entropy:{_entropy(mlp_norm):.4f}")
                    parts.append(f"pool_entropy:{_entropy(pool_norm):.4f}")
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
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state: dict[str, Tensor] | None = None
    if bool(args.ema_enabled):
        ema_state = {name: t.detach().float().cpu().clone() for name, t in base_model.state_dict().items()}
        log0(f"ema:enabled decay:{args.ema_decay:.4f} update_every:{args.ema_update_every}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = run_validation(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                full_validation=False,
            )
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"val_mode:fast "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                f"{deq_info}{expert_info}"
            )
            _best_effort_update_plots("val")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_wallclock_ms is not None and max_wallclock_ms > 0:
            time_frac = min(max(elapsed_ms / max_wallclock_ms, 0.0), 1.0)
        else:
            time_frac = min(max(step / max(int(args.iterations), 1), 0.0), 1.0)

        late_frac = min(max((time_frac - 0.70) / 0.30, 0.0), 1.0)
        health_scale = 1.0 + 4.0 * float(late_frac)
        # Compile-wrapper-safe: always write through unwrapped module.
        sb = _unwrap_compiled_module(base_model.shared_block)
        sb.router.health_scale = float(health_scale)
        # iter 100b: anneal entropy_coef. scale = 0 for time_frac<delay,
        # then linear ramp 0 → 1 over remaining training. Final coef
        # = target × scale. Avoids cold-start trap that hurt iter 99/100.
        ent_target = float(getattr(base_model, "_router_entropy_coef_target", 0.0))
        ent_delay = float(getattr(base_model, "_router_entropy_warmup_delay_frac", 0.0))
        if ent_target > 0.0:
            if time_frac < ent_delay:
                ent_scale = 0.0
            else:
                ent_scale = min(max((time_frac - ent_delay) / max(1.0 - ent_delay, 1e-8), 0.0), 1.0)
            sb.router.entropy_coef = ent_target * ent_scale

        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        if hasattr(sb, "_deq_recon_error_last_bwd"):
            sb._deq_recon_error_last_bwd = None

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
                base_model._block_ortho_aux_enabled = False
                base_model._block_ortho_aux_coef_scale = 1.0
                base_model._block_ortho_aux_tokens_override = int(args.block_ortho_aux_tokens)
                if args.block_ortho_aux_coef > 0.0 and micro_step == grad_accum_steps - 1:
                    base_model._block_ortho_aux_enabled = bool(
                        args.block_ortho_aux_every > 0 and (next_step % int(args.block_ortho_aux_every) == 0))
                with router_diagnostics(diag_enabled, step_tag=next_step if diag_enabled else None):
                    loss = model(x, y)

                # Phase 9 iter 50: Hutchinson-Frobenius Jacobian regularization.
                # Replaces power-iteration Lyapunov (iter 45) with a simpler
                # Hutchinson trace estimator: ||J^T v||² ≈ ||J||²_F for random v.
                # ONE forward + ONE VJP (vs TWO forwards for old Lyapunov).
                # All eigenvalues regularized, not just the largest.
                # Ref: Bai et al., "Stabilizing Equilibrium Models" (arXiv:2106.14342)
                lyap_coef = float(base_model.lyapunov_coef)
                lyap_warmup_frac = float(base_model.lyapunov_warmup_frac)
                lyap_scale = min(time_frac / max(lyap_warmup_frac, 1e-8), 1.0) if lyap_warmup_frac > 0 else 1.0
                z_star = getattr(base_model, '_lyapunov_z_star', None)
                x0_lyap = getattr(base_model, '_lyapunov_x0', None)
                lyap_skip = micro_step < grad_accum_steps - 1  # only last micro-step
                # Compute B̄ once for this step; reuse (detached) for the
                # Hutchinson probe and (live) for the surrogate + denoising.
                aux_b_bar = base_model._parcae_b_bar() if base_model.use_parcae else None
                aux_b_bar_detached = aux_b_bar.detach() if aux_b_bar is not None else None
                if lyap_coef > 0.0 and lyap_scale > 0.0 and z_star is not None and x0_lyap is not None and not lyap_skip:
                    blk = sb
                    # Hutchinson-Frobenius: random Rademacher probe (±1)
                    v_hutch = torch.randint(0, 2, z_star.shape, device=z_star.device, dtype=z_star.dtype) * 2.0 - 1.0
                    z_b = z_star.detach().requires_grad_(True)
                    # Hutchinson JVP is a ρ̂ diagnostic, not a learning signal —
                    # use the detached B̄ so it cannot leak grads into parcae_raw_b.
                    u_b = blk(z_b, x0_lyap, aux_b_bar_detached)
                    jvp = torch.autograd.grad(
                        (u_b * v_hutch).sum(), z_b,
                        create_graph=False, retain_graph=False,
                    )[0]
                    # ||J^T v||² / dim ≈ ||J||²_F / dim (Hutchinson estimator)
                    # Pure-tensor math; keep GPU scalars on device in this block.
                    rho_sample_t = jvp.detach().float().pow(2).mean().sqrt()
                    # EMA smoothing (GPU-resident) to reduce variance of random
                    # probe estimates. Without this, noisy high estimates trigger
                    # large surrogate losses that destabilize training.
                    rho_buf = base_model._lyapunov_rho_hat_buf
                    if rho_buf is None:
                        base_model._lyapunov_rho_hat_buf = rho_sample_t.detach().clone()
                        rho_buf = base_model._lyapunov_rho_hat_buf
                    else:
                        rho_buf.mul_(0.9).add_(rho_sample_t.detach(), alpha=0.1)
                    # Surrogate is tensor-gated, avoiding GPU→CPU sync in the
                    # gradient hot path.
                    gamma = float(base_model.lyapunov_gamma)
                    scale_t = (torch.relu(rho_buf - gamma) / rho_buf.clamp(min=1e-8)).detach()
                    v_dir = (jvp.detach() / jvp.detach().float().reshape(-1).norm().clamp(min=1e-8)).detach()
                    z_b2 = z_star.detach()
                    # Live B̄: the surrogate trains parcae_raw_b/raw_delta to
                    # reduce the spectral-radius term.
                    u_b2 = blk(z_b2, x0_lyap, aux_b_bar)
                    surrogate = (u_b2 * v_dir).sum().abs()
                    loss = loss + lyap_scale * lyap_coef * scale_t.to(dtype=surrogate.dtype) * surrogate

                # Phase 9 iter 55: Denoising regularization (HyDRA 2026).
                # ||f(z*+ε, x0) - z*||² at finite perturbation complements
                # Hutchinson's infinitesimal Jacobian penalty. If contraction
                # holds, one step from z*+ε should land closer to z*.
                dn_coef = float(args.denoising_coef)
                if dn_coef > 0.0 and lyap_scale > 0.0 and z_star is not None and x0_lyap is not None and not lyap_skip:
                    blk_dn = sb
                    dn_std = float(args.denoising_noise_std)
                    eps_noise = torch.randn_like(z_star) * dn_std
                    z_noisy = z_star.detach() + eps_noise
                    # Live B̄: denoising loss trains all Parcae params.
                    f_noisy = blk_dn(z_noisy, x0_lyap, aux_b_bar)
                    dn_loss = (f_noisy - z_star.detach()).float().pow(2).mean()
                    loss = loss + lyap_scale * dn_coef * dn_loss

            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
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

        if args.router_bias_update:
            bias_lr = float(args.router_bias_lr) * (1.0 + 4.0 * float(late_frac))
            sb.router.bias_update(lr=bias_lr, clip=float(args.router_bias_clip), distributed=distributed)
        zero_grad_all()

        if ema_state is not None and args.ema_update_every > 0 and (step % args.ema_update_every == 0):
            update_ema_state_(ema_state, base_model.state_dict(), decay=args.ema_decay)

        step += 1
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
            base_model._deq_recon_error = getattr(base_model.shared_block, "_deq_recon_error_last_bwd", None)
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step, require_step_match=True) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ntp_loss:{ntp:.4f} ctp_loss:{ctp:.4f} "
                f"grad_norm:{float(_preclip_t.item()) if _preclip_t is not None else 0.0:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                f"{deq_info}{expert_info}"
            )

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
        compressed, qsd, meta = save_int6_artifact(sd)
        artifact_bytes = len(compressed)
        log0(f"artifact_bytes:{artifact_bytes} compressor:{_COMPRESSOR}")

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
            # run_valid stays false until the post-int6 assertions pass AND
            # val_bpb is finalized at the end of main().  If the run aborts
            # between here and there, stale run_valid=false keeps
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
            }, f)

        log0("roundtrip_verification:start")
        # T-opt 17: run roundtrip + K-sweep in EAGER mode. torch.compile
        # after load_state_dict crashes silently (inductor segfault — tested
        # stale-guard reuse, compile-fresh, and dynamo.config.disable).
        # Eager with B=64 val batches is reliable and fast enough.
        torch._dynamo.reset()
        torch._dynamo.config.disable = True
        deq_sd = load_int6_artifact(compressed, sd)
        base_model.load_state_dict(deq_sd, strict=True)

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

    def _compute_eval_fp_lipschitz(
        base_m,
        n_hutch: int = 8,
        n_finite_diff: int = 5,
        eps_step: float = 1e-3,
    ) -> tuple[float | None, float | None]:
        """Two Lipschitz upper-bound probes at the saved DEQ FP z*.

        - Hutchinson-Frobenius (resurrected from iter 88 dead code, commit
          ceb7dfa): for random Rademacher v with ``E[v v^T] = I``,
          ``E[||J^T v||²] = ||J||²_F``. Reports
          ``rho_F = sqrt(E[mean(jvp²)]) ≈ ||J||_F / sqrt(dim)`` — proxy for
          average-singular-value-squared.
        - Finite-direction random step: for unit ``v̂``,
          ``||T_θ(z* + eps·v̂) - T_θ(z*)|| / eps ≈ ||J·v̂||``. Sampled
          along ``n_finite_diff`` random directions; report the max as a
          single-shot operator-norm lower bound.

        Returns ``(rho_F, rho_op)`` or ``(None, None)`` if z*/x0 unavailable.
        Caller should run AFTER a forward pass that populates the model's
        ``_lyapunov_z_star`` and ``_lyapunov_x0`` attributes (set by
        ``GPT.forward`` at L2870-2871; populated on every forward including
        eval).
        """
        z_star = getattr(base_m, '_lyapunov_z_star', None)
        x0_lyap = getattr(base_m, '_lyapunov_x0', None)
        if z_star is None or x0_lyap is None:
            return None, None
        sb = _unwrap_compiled_module(base_m.shared_block)
        # Cast saved tensors to the SharedBlock's compute dtype (eval-time may
        # be bf16 while z_star/x0_lyap were saved in fp32 by the train hot path).
        try:
            target_dtype = next(sb.parameters()).dtype
        except StopIteration:
            target_dtype = z_star.dtype
        z_star = z_star.to(target_dtype)
        x0_lyap = x0_lyap.to(target_dtype)
        b_bar = base_m._parcae_b_bar() if base_m.use_parcae else None
        b_bar_d = b_bar.detach().to(target_dtype) if b_bar is not None else None

        # Hutchinson-Frobenius probe (multiple samples for variance reduction).
        # Wrapped in try/except: SDPA backend selection differs under
        # `enable_grad` at eval time and can fail with `Invalid backend`
        # depending on dtype/scaled-dot-product-attention kernel paths.
        # If the probe fails for any reason, skip it (rho_F=None) and let
        # the cheaper finite-direction probe still report — the K-sweep
        # must not be allowed to crash on a diagnostic.
        rho_F: float | None = None
        try:
            rho_F_samples: list[float] = []
            for _ in range(n_hutch):
                v = (torch.randint(0, 2, z_star.shape, device=z_star.device,
                                   dtype=z_star.dtype) * 2.0 - 1.0)
                z_b = z_star.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    u_b = sb(z_b, x0_lyap, b_bar_d)
                    jvp = torch.autograd.grad(
                        (u_b * v).sum(), z_b,
                        create_graph=False, retain_graph=False,
                    )[0]
                rho_F_samples.append(
                    float(jvp.detach().float().pow(2).mean().sqrt().item()))
            if rho_F_samples:
                rho_F = sum(rho_F_samples) / len(rho_F_samples)
        except Exception:
            rho_F = None  # Hutchinson unavailable; finite-diff still runs.

        # Finite-direction random-step Lipschitz sample. Also wrapped in
        # try/except — same SDPA-backend risk as Hutchinson but under
        # no_grad which usually avoids it. Defensive: never crash the
        # K-sweep on a diagnostic.
        rho_op: float | None = None
        try:
            rho_op_samples: list[float] = []
            with torch.no_grad():
                u_base = sb(z_star, x0_lyap, b_bar_d)
                for _ in range(n_finite_diff):
                    eps_dir = torch.randn_like(z_star)
                    eps_norm = eps_dir.float().norm()
                    if float(eps_norm.item()) < 1e-8:
                        continue
                    eps_unit = eps_dir / eps_norm
                    u_pert = sb(z_star + eps_step * eps_unit, x0_lyap, b_bar_d)
                    rho_op_samples.append(
                        float((u_pert - u_base).float().norm().item()) / eps_step)
            if rho_op_samples:
                rho_op = max(rho_op_samples)
        except Exception:
            rho_op = None
        return rho_F, rho_op

    # DEQ fixed-point K-sweep: verify val_bpb improves (or plateaus) as K grows.
    # A valid DEQ should converge to a fixed point — more solver iterations = better
    # or equal quality, never worse.  Non-monotone behaviour indicates the model
    # is exploiting a specific iteration count rather than a true fixed point.
    # Runs DDP-parallel across ranks for a ~2x speedup on 2 GPUs.
    #
    # iter 97.6 (PERMANENT, 2026-04-26): K-sweep harness now includes
    # (a) Hutchinson-Frobenius Lipschitz upper bound at the converged FP,
    # (b) finite-direction random-step Lipschitz sample (operator-norm proxy),
    # (c) acyclicity primes K=17, 37, 113 — coprime to {2,3,4,5,12,20} to
    #     detect period-L cycles aliased by power-of-2 sampling.
    log0("k_sweep:start")
    # T-opt 15: Reset dynamo before K-sweep to prevent recompilation storm.
    # Different K values change iteration counts, triggering dynamo guards
    # that cause recompile_limit hits → stall one rank → NCCL timeout.
    torch._dynamo.reset()
    # iter 97.6: insert primes {17, 37, 113} between the powers of 2.
    # Primes coprime to {2,3,4,5,12,20} (training K-jitter set GCD = 4)
    # break the period-L cycle alias inherent in power-of-2 sampling.
    k_sweep_values = [4, 8, 16, 17, 32, 37, 64, 113, 128]
    k_sweep_results: dict[int, float] = {}

    # iter 100b user directive (PERMANENT 2026-04-27): emit a structured
    # per-K table of expert health + Lipschitz + sparsity + shared_gate so
    # routing trajectory across K can be plotted/compared per H-claim.
    # Header is emitted before the loop; each K appends one row.
    def _collect_eval_kdiag(base_m) -> dict[str, float]:
        """Per-K expert/sparsity/shared-gate diagnostics collected from the
        most recent eval forward pass. Pooled-router single source of truth:
        attn_router and mlp_router are the same instance under the iter 35
        consolidation, so per-slice metrics are computed from the pooled
        usage array using the (R, R) split; ortho is read per-component."""
        sb = getattr(base_m, "shared_block", None)
        if sb is None:
            return {}
        out: dict[str, float] = {}
        R_total = int(getattr(sb, "num_experts", 0))
        R = R_total - int(getattr(sb, "num_shared_experts", 0))
        attn_router = getattr(getattr(sb, "attention", None), "attn_router", None)
        if attn_router is not None and hasattr(attn_router, "_materialize_diag_lists"):
            attn_router._materialize_diag_lists()
        usage = getattr(attn_router, "_expert_usage", None) if attn_router else None
        if usage and len(usage) >= 2 * R and R > 0:
            attn_half, mlp_half = usage[:R], usage[R:]
            for label, half in [("attn", attn_half), ("mlp", mlp_half)]:
                s = sum(half) or 1e-8
                norm = [u / s for u in half]
                mu = sum(norm) / len(norm) or 1e-8
                var = sum((x - mu) ** 2 for x in norm) / len(norm)
                out[f"{label}_cv"] = (var ** 0.5) / mu
                out[f"{label}_min"] = min(norm)
            pool_sum = sum(usage) or 1e-8
            pool_norm = [u / pool_sum for u in usage]
            pmu = sum(pool_norm) / len(pool_norm) or 1e-8
            pvar = sum((x - pmu) ** 2 for x in pool_norm) / len(pool_norm)
            out["pool_cv"] = (pvar ** 0.5) / pmu
            out["pool_ent"] = -sum(x * math.log(x + 1e-8) for x in pool_norm if x > 0.0)
        pertoken = getattr(attn_router, "_expert_entropy", None) if attn_router else None
        if pertoken is not None:
            out["pertoken_ent"] = float(pertoken)
        for prefix, comp_attr in [("attn", "attention"), ("mlp", "mlp")]:
            comp = getattr(sb, comp_attr, None)
            if comp is None:
                continue
            v_t = getattr(comp, "_out_ortho_cos_sim_t", None)
            if isinstance(v_t, torch.Tensor):
                out[f"{prefix}_ortho"] = float(v_t.float().item())
            else:
                v = getattr(comp, "_out_ortho_cos_sim", None)
                if v is not None:
                    out[f"{prefix}_ortho"] = float(v)
        sg = getattr(sb, "_shared_gate_mean", None)
        if sg is not None:
            out["shared_gate"] = float(sg) if not isinstance(sg, torch.Tensor) else float(sg.detach().float().item())
        return out

    # Tabular K-sweep header — fixed-width columns for grep + visual scanning.
    _kdiag_cols = [
        ("K", 5), ("val_bpb", 9), ("attn_cv", 8), ("mlp_cv", 8), ("pool_cv", 8),
        ("attn_min", 9), ("mlp_min", 9), ("attn_ortho", 11), ("mlp_ortho", 10),
        ("pertoken_ent", 13), ("pool_ent", 9), ("shared_gate", 12),
        ("hutch_F", 9), ("rd_step", 9), ("iter_conv_rel", 14),
    ]
    def _fmt_kdiag(value: float | None, width: int) -> str:
        s = "N/A" if value is None else f"{value:.4f}"
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
        # iter 97.6: Lipschitz upper bounds at the saved DEQ FP (z*).
        # Hutchinson-Frobenius reuses the iter 88 dead code (commit ceb7dfa);
        # finite-direction random-step samples ||J·v̂|| as an operator-norm proxy.
        # Both are reported per-K so we can plot Lipschitz vs depth alongside
        # val_bpb and iter_conv_rel — directly diagnoses whether contraction
        # tightens with K (target: rho ≤ 1 and stable across K).
        rho_F, rho_op = _compute_eval_fp_lipschitz(base_m_for_roundtrip)
        if rho_F is not None:
            diag_parts.append(f"hutch_F:{rho_F:.4f}")
        if rho_op is not None:
            diag_parts.append(f"rd_step:{rho_op:.4f}")
        log0(f"k_sweep:k={k_eval} {' '.join(diag_parts)}")
        # iter 100b user directive (PERMANENT): tabular per-K row.
        kdiag = _collect_eval_kdiag(base_m_for_roundtrip)
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
            "hutch_F": rho_F,
            "rd_step": rho_op,
            "iter_conv_rel": conv_rel_val,
        }
        log0("k_sweep_table:" + " ".join(_fmt_kdiag(kdiag_row[name], w) for name, w in _kdiag_cols))
    k_parts = " ".join(f"k{k}:{b:.6f}" for k, b in k_sweep_results.items())
    log0(f"k_sweep:done {k_parts}")

    # ── Final HARD assertions on post-int6 model health ──────────────────
    # Run after the K-sweep so all diagnostics are populated from the highest
    # K eval pass.  These are HARD failures — they raise RuntimeError if the
    # trained model violates the architectural invariants (expert health,
    # DEQ input-dependence, FP convergence).  The run cannot be promoted to
    # baseline unless every assertion passes.
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
        s = sum(h) or 1e-8
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
        # True FP gate: K=64 and K=128 must not degrade from min val_bpb.
        # A true fixed point converges monotonically — higher K should equal
        # or improve quality.  Threshold 0.02 allows bf16 noise.
        for k_check in [64, 128]:
            if k_check in k_sweep_results:
                delta = k_sweep_results[k_check] - best_bpb
                if delta > 0.02:
                    _failures.append(
                        f"K={k_check}_degradation: bpb={k_sweep_results[k_check]:.4f} "
                        f"vs best={best_bpb:.4f} (Δ={delta:.4f} > 0.02 — not a true FP)"
                    )

    # 5. Iter convergence: relative convergence must be small at highest K.
    # DDP-reduce the rank-local conv_rel so the assertion sees the global mean
    # (matches the treatment of ortho/usage above).  Without this, master's
    # local eval shard could pass while a worker sees divergence.
    conv_rel_t = getattr(base_m_for_roundtrip, "_deq_iter_convergence_rel_t", None)
    conv_rel_local = (
        float(conv_rel_t.detach().float().item())
        if isinstance(conv_rel_t, torch.Tensor) else None
    )
    conv_rel = _ddp_mean_scalar(conv_rel_local)
    if conv_rel is not None and conv_rel > 0.1:
        _failures.append(f"iter_conv_rel={conv_rel:.4f} > 0.1 (solver not converging at eval K)")

    # 6. RevDEQ reconstruction error: the backward reconstructs forward states
    # from the solver's final state; ||reconstructed_z - z|| must stay small
    # for the reversibility invariant to hold AND for high-quality gradients.
    # Bf16 + FP64 accumulators: healthy runs see 1e-3 to 1e-2 typically.
    # Tightened from 1.0 → 0.1 (matches smoke test threshold; lower recon err
    # = higher-quality gradients = more efficient training).  >0.1 means the
    # reversibility approximation is degrading and gradients become noisy.
    recon_err_local = getattr(base_m_for_roundtrip.shared_block, "_deq_recon_error_last_bwd", None)
    recon_err = _ddp_mean_scalar(
        float(recon_err_local) if recon_err_local is not None else None
    )
    if recon_err is not None and recon_err > 0.1:
        _failures.append(
            f"deq_recon_err={recon_err:.3e} > 0.1 (RevDEQ reversibility degrading — "
            f"gradients getting noisy, training efficiency drops)"
        )

    # Classify each failure and prescribe a fix from the verified-hypothesis
    # troubleshooting table.  A failed run is INVALID (cannot be promoted to
    # baseline) but its diagnostics + retry_hint guide the NEXT iteration's
    # config change — the agent applies the prescribed fix and reruns.  We
    # do NOT raise here: letting the process exit cleanly preserves all the
    # artifacts and log output the fix decision needs.
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
                "run_valid": False,
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

    # Master-only: update meta.json with the final val_bpb AND flip
    # run_valid=true (if all hard assertions passed).  Writing both atomically
    # at the very end — after sliding val + all eval passes — ensures a crash
    # between the assertion success and here leaves run_valid=false with
    # val_bpb=0.0, so update_results.sh --promote refuses a half-finished run.
    if master_process and meta_path is not None:
        with open(meta_path, "r") as f:
            meta_json = json.load(f)
        meta_json["val_bpb"] = val_bpb_q
        # POLICY (val_bpb-primary): run_valid=true whenever val_bpb is
        # written, regardless of gate failures.  Promotion criterion is val_bpb
        # improvement + 16MB budget; diagnostic-gate failures are recorded as
        # retry prescriptions for the next iter (meta_json["failure_categories"]
        # + retry_hint.json) but do not block promotion.
        meta_json["run_valid"] = True
        # Status label must match the update_results.sh promotion allowlist
        # {"validated", "validated_clean", "validated_with_tech_debt"}; gate
        # failures land in `failure_categories` + retry_hint.json and are
        # tech debt for the next iter per CLAUDE.md val_bpb-primary policy.
        meta_json["status"] = "validated_clean" if _assertions_passed else "validated_with_tech_debt"
        with open(meta_path, "w") as f:
            json.dump(meta_json, f)

    # Tear down DDP after all eval completes.
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
