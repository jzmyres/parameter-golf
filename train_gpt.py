"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Note: For this autoresearch workspace, `train_gpt.py` may grow beyond a small tutorial script as long as it remains a single-file, self-contained submission artifact. If you want a newcomer-friendly baseline, prefer the pinned record scripts in `records/`.
"""

from __future__ import annotations

import copy
import contextlib
import glob
import io
import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
import zlib
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
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

# Diagnostics collection is disabled by default during training for speed.
# Enable it briefly around selected forward passes (e.g. train logging steps)
# to get dense curves for expert/DEQ metrics without increasing validation cost.
_ROUTER_DIAGNOSTICS_ACTIVE = False
_ROUTER_DIAGNOSTICS_STEP: int | None = None

SIGMOID_ONE_INIT_LOGIT = 6.0  # sigmoid(6)=0.9975; "initialized to 1" without full saturation


def dynamo_disable(fn):
    """Run a function outside `torch.compile` graphs when possible."""
    try:
        import torch._dynamo as dynamo  # type: ignore

        return dynamo.disable(fn)
    except Exception:
        return fn


@contextlib.contextmanager
def router_diagnostics(enabled: bool = True, *, step_tag: int | None = None):
    global _ROUTER_DIAGNOSTICS_ACTIVE
    global _ROUTER_DIAGNOSTICS_STEP
    prev = _ROUTER_DIAGNOSTICS_ACTIVE
    prev_step = _ROUTER_DIAGNOSTICS_STEP
    _ROUTER_DIAGNOSTICS_ACTIVE = bool(enabled)
    _ROUTER_DIAGNOSTICS_STEP = step_tag if enabled else None
    try:
        yield
    finally:
        _ROUTER_DIAGNOSTICS_ACTIVE = prev
        _ROUTER_DIAGNOSTICS_STEP = prev_step


class KShuffleBagSampler:
    """Shuffle-bag sampler over an integer range [k_min, k_max] (inclusive).

    Guarantees exact coverage: each K appears exactly once per bag cycle, with random order.
    """
    def __init__(self, k_min: int, k_max: int, rng: random.Random):
        if k_min > k_max:
            raise ValueError(f"k_min must be <= k_max, got {k_min} > {k_max}")
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.rng = rng
        self._bag: list[int] = []

    def reset(self) -> None:
        self._bag.clear()

    def sample(self) -> int:
        if not self._bag:
            self._bag = list(range(self.k_min, self.k_max + 1))
            self.rng.shuffle(self._bag)
        return int(self._bag.pop())

    def set_range(self, k_min: int, k_max: int) -> None:
        k_min_i = int(k_min)
        k_max_i = int(k_max)
        if k_min_i > k_max_i:
            raise ValueError(f"k_min must be <= k_max, got {k_min_i} > {k_max_i}")
        if k_min_i != self.k_min or k_max_i != self.k_max:
            self.k_min = k_min_i
            self.k_max = k_max_i
            self.reset()

class Hyperparameters:
    # Do not override experiment configuration via environment variables.
    # (Exception: CUDA/DDP runtime env like CUDA_VISIBLE_DEVICES/RANK/WORLD_SIZE.)
    data_path = "./data/datasets/fineweb10B_sp1024"
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id = ""
    seed = 42

    # Experiment knobs: set defaults in code (do not use environment variables).
    val_batch_size = 524_288
    val_loss_every = 200  # sparse but meaningful validation curve
    train_log_every = 100

    iterations = 1000
    warmdown_iters = 1000
    warmup_steps = 20
    # Throughput-tuned default for this dev box (2×A100 DDP): maximize tokens/sec without torch.compile.
    # NOTE: larger batches can change DEQ stability; keep this conservative unless retuned.
    train_batch_tokens = 65_536
    train_seq_len = 2048
    max_wallclock_seconds = 0.0  # 0 disables wallclock early-stop
    qk_gain_init = 1.5
    deq_beta = 0.2

    vocab_size = 1024
    num_layers = 38  # DEQ solver iteration budget (max K)
    num_refinements = 2  # predict→soft_embed→re-encode cycles
    num_kv_heads = 4
    model_dim = 640
    num_heads = 8
    mlp_mult = 2.5
    tie_embeddings = True
    rope_base = 1000.0
    logit_softcap = 20.0

    embed_lr = 0.6
    head_lr = 0.008
    tied_embed_lr = 0.03
    tied_embed_init_std = 0.005
    matrix_lr = 0.02
    scalar_lr = 0.02
    muon_momentum = 0.99
    muon_backend_steps = 5
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 800
    beta1 = 0.85
    beta2 = 0.90
    adam_eps = 1e-8
    grad_clip_norm = 0.25
    weight_decay = 0.03

    # Routing regularization weights
    attn_balance_mult = 5.0
    mlp_balance_mult = 1.0
    bal_loss_coef = 1.0
    # Loss-free load balancing (bias controller). Keeps expert utilization healthy without
    # interfering gradients from strong auxiliary losses.
    router_bias_update = True
    router_bias_lr = 0.05
    router_bias_clip = 5.0
    mos_ortho_out_coef = 0.01
    attn_ortho_out_coef = 0.0
    mlp_ortho_out_coef = 0.0
    deq_backward = "revdeq"  # {autograd, revdeq}
    deq_k_jitter = True
    deq_k_min = 2
    deq_k_max = 30
    deq_k_eval = 30
    # Shuffle-bag K-jitter range ramp: maxK linearly increases from deq_k_max_start -> deq_k_max.
    # Default: no ramp (stable range from step 1).
    deq_k_max_start = deq_k_max
    deq_k_max_ramp_steps = 1
    compile_train = False  # torch.compile(shared_block) for training speed
    benchmark_mode = False  # skip post-quant eval/serialization for speed microbenchmarks

    eval_stride = 0  # 0=standard eval; set >0 for sliding window (final only)
    eval_batch_seqs = 32

    # Architecture knobs (defaults only; override via CLI, not env)
    bigram_vocab_size = 65536
    bigram_dim = 224
    kv_latent_dim = 0  # 0 = auto (dim//2)
    attn_expert_rank = 0  # 0 = auto (dim//2)
    mlp_expert_rank = 0  # 0 = auto (hidden//2)
    # Routing is dense softmax over experts (no post-softmax gating).

    # SWA knobs (defaults only; override via CLI, not env)
    swa_enabled = True
    swa_start_frac = 0.3
    swa_every = 25


def _parse_cli_overrides(argv: list[str]) -> dict[str, object]:
    """Parse optional CLI overrides for experiment knobs (no env overrides)."""
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--train-batch-tokens", type=int, default=None)
    p.add_argument("--train-seq-len", type=int, default=None)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--val-loss-every", type=int, default=None)
    p.add_argument("--train-log-every", type=int, default=None)
    p.add_argument("--max-wallclock-seconds", type=float, default=None)
    p.add_argument("--attn-balance-mult", type=float, default=None)
    p.add_argument("--mlp-balance-mult", type=float, default=None)
    p.add_argument("--bal-loss-coef", type=float, default=None)
    p.add_argument("--router-bias-update", type=int, default=None, help="1/0; loss-free expert-bias load balancing")
    p.add_argument("--router-bias-lr", type=float, default=None)
    p.add_argument("--router-bias-clip", type=float, default=None)
    p.add_argument("--mos-ortho-out-coef", type=float, default=None)
    p.add_argument("--attn-ortho-out-coef", type=float, default=None)
    p.add_argument("--mlp-ortho-out-coef", type=float, default=None)
    p.add_argument("--bigram-vocab-size", type=int, default=None)
    p.add_argument("--bigram-dim", type=int, default=None)
    p.add_argument("--kv-latent-dim", type=int, default=None)
    p.add_argument("--attn-expert-rank", type=int, default=None)
    p.add_argument("--mlp-expert-rank", type=int, default=None)
    p.add_argument("--swa-enabled", type=int, default=None, help="1/0")
    p.add_argument("--swa-start-frac", type=float, default=None)
    p.add_argument("--swa-every", type=int, default=None)
    p.add_argument("--deq-backward", type=str, default=None, choices=["autograd", "revdeq"])
    p.add_argument("--deq-k-jitter", type=int, default=None, help="1/0; sample K per optimizer step")
    p.add_argument("--deq-k-min", type=int, default=None)
    p.add_argument("--deq-k-max", type=int, default=None)
    p.add_argument("--deq-k-eval", type=int, default=None)
    p.add_argument("--deq-k-max-start", type=int, default=None, help="starting maxK for range ramp (train only)")
    p.add_argument("--deq-k-max-ramp-steps", type=int, default=None, help="steps to ramp maxK to deq_k_max")
    p.add_argument("--compile-train", type=int, default=None, help="1/0; torch.compile(shared_block) during training")
    p.add_argument("--benchmark-mode", type=int, default=None, help="1/0; skip final eval/quant/plots (speed bench)")
    ns, unknown = p.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"Unknown args: {unknown}")
    out: dict[str, object] = {}
    for k, v in vars(ns).items():
        if v is not None:
            key = k.replace("-", "_")
            if key == "swa_enabled":
                out[key] = bool(int(v))
            elif key == "deq_k_jitter":
                out[key] = bool(int(v))
            elif key == "router_bias_update":
                out[key] = bool(int(v))
            elif key == "compile_train":
                out[key] = bool(int(v))
            elif key == "benchmark_mode":
                out[key] = bool(int(v))
            else:
                out[key] = v
    return out

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

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

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
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
                    g = g.view(-1, g.shape[-1])  # flatten leading dims for NS
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    g = g.view(orig_shape)
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

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


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
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
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def _eval_seq_bounds(total_seqs: int, eval_batch_seqs: int, rank: int, world_size: int, *, full_eval: bool) -> tuple[int, int, int]:
    """Return (seq_start, seq_end, global_eval_seqs) for validation.

    - If `full_eval`, evaluate the entire validation set.
    - Otherwise, evaluate a fixed prefix of `eval_batch_seqs` sequences (for fast, comparable curves).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if total_seqs < 0:
        raise ValueError(f"total_seqs must be non-negative, got {total_seqs}")
    if full_eval or eval_batch_seqs <= 0:
        global_eval_seqs = total_seqs
    else:
        global_eval_seqs = min(total_seqs, int(eval_batch_seqs))
    seq_start = (global_eval_seqs * rank) // world_size
    seq_end = (global_eval_seqs * (rank + 1)) // world_size
    return int(seq_start), int(seq_end), int(global_eval_seqs)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    full_eval: bool,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    # Fast in-training validation: evaluate a fixed, small number of sequences for a smooth curve.
    # Full validation (entire set) is reserved for the final step.
    seq_start, seq_end, _ = _eval_seq_bounds(total_seqs, args.eval_batch_seqs, rank, world_size, full_eval=full_eval)
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    base_m = model.module if hasattr(model, "module") else model
    prev_k = getattr(base_m, "_deq_k_override", None)
    base_m._deq_k_override = int(getattr(args, "deq_k_eval", base_m.num_layers))
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _ = model(x, y)
            # Use NTP-only loss for val (exclude CTP term)
            batch_loss = torch.tensor(base_m._ntp_loss, device=device)
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
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
    if prev_k is None:
        try:
            delattr(base_m, "_deq_k_override")
        except Exception:
            base_m._deq_k_override = 0
    else:
        base_m._deq_k_override = prev_k
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION (INT8 legacy + INT6 mixed)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    # Keep only small control tensors in fp32. Avoid broad substrings like "expert_gate"
    # which can match large expert weight tensors (e.g., shared_block.mlp.expert_gate).
    "resid_mix",
    "resid_mixes",
    "gg_w",
    "gg_b",
    "q_gain",
    "skip_weight",
    "skip_weights",
    "bigram.scale",
    "gate_bias",
)
FP16_KEEP_NAME_PATTERNS = ("tok_emb",)
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if "bigram" in name:
        return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"

def quantize_intN_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Optimal scale search: try 3 candidate scales and pick best MSE
        row_max = t32.abs().amax(dim=1)
        best_q = None
        best_scale = None
        best_mse = float("inf")
        for alpha in [0.95, 1.0, 1.05]:
            rm = row_max * alpha
            s = (rm / clip_range).clamp_min(1e-12).to(torch.float16)
            s = s.clamp_min(torch.finfo(torch.float16).tiny)
            q_try = torch.clamp(torch.round(t32 / s.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
            recon = q_try.float() * s.float()[:, None]
            mse = (t32 - recon).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_q = q_try
                best_scale = s
        return best_q, best_scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / clip_range, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -(clip_range+1), clip_range).to(torch.int8)
    return q, scale

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
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for attention
            orig_shape = t.shape
            t_2d = t.view(-1, t.shape[-1]) if t.ndim > 2 else t
            q, s = quantize_intN_per_row(t_2d, clip_range=clip)
            q = q.view(orig_shape) if t.ndim > 2 else q
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{5 if cat == 'mlp' else 6}"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
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
            # Flatten to 2D for dequant if needed (3D expert params)
            q_2d = q.view(-1, q.shape[-1]) if q.ndim > 2 else q
            deq = (q_2d.float() * s.float().view(q_2d.shape[0], *([1] * (q_2d.ndim - 1))))
            out[name] = deq.view(orig_shape).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).view(orig_shape).to(orig_dtype)
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


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

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        denom = self.world_size * grad_accum_steps
        if denom <= 0:
            raise ValueError(f"world_size*grad_accum_steps must be positive, got {denom}")
        global_seqs = global_tokens // seq_len
        local_seqs = global_seqs // denom
        if local_seqs < 1:
            raise ValueError(
                "TRAIN_BATCH_TOKENS too small for the requested DDP config: "
                f"TRAIN_BATCH_TOKENS={global_tokens} TRAIN_SEQ_LEN={seq_len} "
                f"WORLD_SIZE={self.world_size} GRAD_ACCUM_STEPS={grad_accum_steps}"
            )
        local_tokens = local_seqs * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps if eps is not None else 1e-6

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.eps)


_QAT_ACTIVE = False  # Global flag for Late QAT

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        # Late QAT: add fake quantization noise during warmdown
        if _QAT_ACTIVE and self.training and w.ndim == 2 and w.numel() > 8192:
            clip = getattr(self, '_qat_clip', 31)
            w32 = w.float()
            amax = w32.abs().amax(dim=-1, keepdim=True)
            s = (amax / clip).clamp_min(1e-12)
            w_q = torch.clamp(torch.round(w32 / s), -(clip + 1), clip) * s
            w = w32 + (w_q - w32).detach()  # STE: quantized forward, identity backward
        bias = self.bias
        # If autocast is enabled, rely on it (and its weight-cast cache). Otherwise, match x dtype.
        if (x.dtype != w.dtype or (bias is not None and x.dtype != bias.dtype)) and not torch.is_autocast_enabled():
            w = w.to(dtype=x.dtype)
            if bias is not None:
                bias = bias.to(dtype=x.dtype)
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            # 1D params + control tensors → fp32 for optimizer precision
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
            # 3D+ expert params (not CastedLinear) → fp32 for Muon
            if param.ndim >= 3 and param.dtype != torch.float32:
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
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            # Important: mutating module attributes inside torch.compile can break
            # CUDA graphs (cached outputs overwritten across invocations).
            self._refresh_cache(seq_len, device)
        # PyTorch inference_mode can produce "inference tensors" that cannot be saved for backward.
        # If eval ran first and populated the cache under inference_mode, ensure any subsequent
        # non-inference computation refreshes the cache with normal tensors. This matters both for
        # standard autograd (needs to save tensors for backward) and RevDEQ (needs forward/backward
        # function evaluations to match for reversible reconstruction).
        if (not torch.is_inference_mode_enabled()) and self._cos_cached is not None and self._cos_cached.is_inference():
            self._refresh_cache(seq_len, device)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def mean_abs_offdiag_cosine(groups: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean off-diagonal |cosine similarity| for a [E, D] tensor."""
    if groups.ndim != 2:
        raise ValueError(f"groups must be rank-2 [E,D], got shape {tuple(groups.shape)}")
    e = groups.shape[0]
    if e < 2:
        return groups.new_zeros(())
    g = groups / (groups.norm(dim=-1, keepdim=True) + eps)
    cos = g @ g.T
    mask = ~torch.eye(e, dtype=torch.bool, device=cos.device)
    return cos[mask].abs().mean()


def max_abs_offdiag_cosine(groups: Tensor, eps: float = 1e-8) -> Tensor:
    """Max off-diagonal |cosine similarity| for a [E, D] tensor.

    Worst-case statistic: if this is <= thr, then *all* expert pairs satisfy |cos| <= thr.
    """
    if groups.ndim != 2:
        raise ValueError(f"groups must be rank-2 [E,D], got shape {tuple(groups.shape)}")
    e = groups.shape[0]
    if e < 2:
        return groups.new_zeros(())
    g = groups / (groups.norm(dim=-1, keepdim=True) + eps)
    cos = g @ g.T
    mask = ~torch.eye(e, dtype=torch.bool, device=cos.device)
    return cos[mask].abs().max()


def deq_maxk_ramp(step: int, *, start: int, end: int, ramp_steps: int) -> int:
    """Linear ramp for the *max K* of shuffle-bag sampling."""
    s = int(step)
    if s <= 1:
        return int(start)
    if ramp_steps <= 1:
        return int(end)
    if s >= int(ramp_steps):
        return int(end)
    frac = float(s - 1) / float(int(ramp_steps) - 1)
    k = int(round(float(start) + (float(end) - float(start)) * frac))
    return int(max(min(k, int(end)), int(start)))


class SoftDenseRouter(nn.Module):
    """Shared soft dense routing module for all MoE components.

    Per-token routing: dense softmax weights over experts (no top-k, no dropping).

    Provides balance (MSE-to-uniform) regularization and diagnostics, plus an
    optional loss-free load-balancing bias controller (expert_bias) updated once
    per optimizer step from terminal-state routing statistics.
    """
    def __init__(self, dim: int, num_experts: int, *, enable_gate: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.enable_gate = bool(enable_gate)
        self.router = CastedLinear(dim, num_experts, bias=False)
        # Small router init → near-uniform routing at start
        nn.init.normal_(self.router.weight, std=0.01)
        # Optional gating head for block-level MoE: w = softmax(logits) * sigmoid(gate_logits),
        # so sum(w) is in (0,1] and can serve as a per-token global residual gate.
        self.gate = CastedLinear(dim, num_experts, bias=True) if self.enable_gate else None
        if self.gate is not None:
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(SIGMOID_ONE_INIT_LOGIT)
        # Loss-free load-balancing bias (added to routing logits before softmax).
        # Updated outside autograd to avoid gradient interference from strong aux losses.
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        self._bias_stats_enabled = False
        self._mean_share_last: Tensor | None = None
        # Diagnostics (set during forward)
        self._balance_loss = None
        self._sparsity_loss = None
        self._expert_usage = None
        self._expert_gates = None
        self._expert_entropy = None
        self._expert_sparsity = None
        self._expert_balance_cv = None
        self._gate_mass_mean = None
        self._diag_step: int | None = None

    def set_bias_stats_enabled(self, enabled: bool) -> None:
        self._bias_stats_enabled = bool(enabled)

    @torch.no_grad()
    def bias_update(self, *, lr: float, clip: float, distributed: bool) -> None:
        """Loss-free load balancing update: expert_bias += lr * (target - mean_share)."""
        ms = self._mean_share_last
        if ms is None:
            return
        ms = ms.detach().to(dtype=torch.float32)
        if distributed and dist.is_available() and dist.is_initialized():
            ms = ms.clone()
            dist.all_reduce(ms, op=dist.ReduceOp.SUM)
            ms /= float(dist.get_world_size())
        target = torch.full_like(ms, 1.0 / float(self.num_experts))
        self.expert_bias.add_(lr * (target - ms))
        if clip > 0:
            self.expert_bias.clamp_(min=-clip, max=clip)

    def forward(self, x: Tensor) -> Tensor:
        """Returns routing weights [*, num_experts]."""
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x_n = _rms_norm(x)
        route_logits = self.router(x_n) + self.expert_bias.to(dtype=x.dtype)
        p = torch.softmax(route_logits, dim=-1)
        if self.gate is not None:
            g = torch.sigmoid(self.gate(x_n))
            route_weights = p * g
            gate_mass = route_weights.sum(dim=-1, keepdim=True)  # [*,1] in (0,1]
            share = route_weights / gate_mass.clamp_min(1e-8)    # normalized share (sum=1)
        else:
            route_weights = p
            share = p
        if self.training:
            # Balance on normalized share distribution (sum=1), even when route_weights
            # is gated and does not sum to 1.
            reduce_dims = tuple(range(share.ndim - 1))
            mean_share = share.mean(dim=reduce_dims)
            target = torch.ones_like(mean_share) / self.num_experts
            self._balance_loss = F.mse_loss(mean_share, target)
            self._sparsity_loss = torch.tensor(0.0, device=x.device)
            if self._bias_stats_enabled:
                # Store terminal-state routing stats for the loss-free bias controller update.
                self._mean_share_last = mean_share.detach()
            with torch.no_grad():
                do_diag = bool(_ROUTER_DIAGNOSTICS_ACTIVE)
                if do_diag and dist.is_available() and dist.is_initialized():
                    do_diag = dist.get_rank() == 0
                if do_diag:
                    self._record_diagnostics(share.detach(), gate_mass.detach() if self.gate is not None else None, reduce_dims)
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._sparsity_loss = torch.tensor(0.0, device=x.device)
            # Do not update bias controller from eval passes.
            self._mean_share_last = None
            with torch.no_grad():
                # Avoid host-transfer overhead on non-master ranks during DDP eval.
                # These diagnostics are only used for logging/plotting.
                do_diag = True
                if dist.is_available() and dist.is_initialized():
                    do_diag = dist.get_rank() == 0
                if do_diag:
                    self._record_diagnostics(share.detach(), gate_mass.detach() if self.gate is not None else None, tuple(range(share.ndim - 1)))
                else:
                    self._expert_usage = None
                    self._expert_gates = None
                    self._expert_entropy = None
                    self._expert_sparsity = None
                    self._expert_balance_cv = None
                    self._gate_mass_mean = None
                    self._diag_step = None
        return route_weights

    @dynamo_disable
    def _record_diagnostics(self, share_detached: Tensor, gate_mass_detached: Tensor | None, reduce_dims: tuple[int, ...]) -> None:
        # Compute diagnostics from detached tensors to avoid autograd overhead, and
        # keep all Python-side state mutation out of torch.compile graphs.
        mean_mass = share_detached.mean(dim=reduce_dims)
        self._expert_usage = mean_mass.float().cpu().tolist()  # normalized share (sum=1)
        self._expert_gates = None

        per_token_ent = -(share_detached * (share_detached + 1e-8).log()).sum(-1)
        ent = float(per_token_ent.mean().item())
        self._expert_entropy = ent
        self._expert_sparsity = 1.0 - (ent / max(math.log(float(self.num_experts)), 1e-8))
        self._expert_balance_cv = float((mean_mass.std() / mean_mass.mean().clamp_min(1e-8)).item())
        if gate_mass_detached is not None:
            self._gate_mass_mean = float(gate_mass_detached.mean().item())
        else:
            self._gate_mass_mean = None
        self._diag_step = _ROUTER_DIAGNOSTICS_STEP


class CausalSelfAttention(nn.Module):
    """MLA with Gated Attention + expert bank (Constraints #2, #3).

    - Low-rank KV compression via shared latent
    - Decoupled RoPE: half of head_dim for positional encoding
    - Query-dependent per-head sigmoid gate after SDPA (scalar gate per head)
    - Routing is performed at the Block level (shared router for the whole block)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, kv_latent_dim: int = 0, num_experts: int = 6,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(dim // max(self.num_experts, 1), 1)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # MLA: low-rank KV compression with decoupled RoPE
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else dim // 2
        self.rope_dim = self.head_dim // 2  # half for RoPE
        self.nope_dim = self.head_dim - self.rope_dim

        # Q projection outputs query + scalar per-head gate logits.
        self.c_q = CastedLinear(dim, dim + num_heads, bias=False)
        # Initialize the gate-logit slice to 0 so the attention gate starts at 0.5.
        with torch.no_grad():
            self.c_q.weight[dim:, :].zero_()
        # KV compression path
        self.c_kv_down = CastedLinear(dim, self.kv_latent_dim, bias=False)
        self.c_k_nope = CastedLinear(self.kv_latent_dim, num_kv_heads * self.nope_dim, bias=False)
        self.c_v = CastedLinear(self.kv_latent_dim, num_kv_heads * self.head_dim, bias=False)
        # Decoupled RoPE key
        self.c_k_rope = CastedLinear(dim, num_kv_heads * self.rope_dim, bias=False)
        # Full-dim low-rank experts: each expert projects dim → rank → dim (sees all dims)
        self.expert_proj = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_out = nn.Parameter(torch.empty(num_experts, dim, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_proj.data[e])
            nn.init.xavier_uniform_(self.expert_out.data[e])
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dim, base=rope_base)
        # Gated attention bias (per-head scalar); init near-1 (combined with zeroed per-token logits).
        self.gate_bias = nn.Parameter(torch.full((num_heads,), SIGMOID_ONE_INIT_LOGIT, dtype=torch.float32))
        # Soft dense routing on attention output (MoE for attention). In block-level MoE
        # mode we pass in a shared router from the parent block.
        self.attn_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        self.out_ortho_coef = 0.0
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None

    def forward_experts(self, x: Tensor) -> Tensor:
        """Return per-expert attention outputs [B, T, E, D] (no routing mix)."""
        bsz, seqlen, dim = x.shape
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x_n = _rms_norm(x)
        y = self._attn_shared_from_normed(x_n)  # [B,T,D]
        # einsum requires matching dtypes; keep compute in activation dtype under autocast.
        expert_proj = self.expert_proj.to(dtype=y.dtype)
        expert_out = self.expert_out.to(dtype=y.dtype)
        h = torch.einsum('btd,erd->bter', y, expert_proj)
        out_e = torch.einsum('bter,edr->bted', h, expert_out)
        # Output-space orthogonality. When autograd-unrolling DEQ, this can be used as a
        # true loss term (differentiable). In RevDEQFunction mode, it remains diagnostic-only.
        self._out_ortho_loss = None
        need_loss = bool(self.training and torch.is_grad_enabled() and self.out_ortho_coef > 0.0)
        do_diag = (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        if do_diag and dist.is_available() and dist.is_initialized():
            do_diag = dist.get_rank() == 0
        if need_loss:
            mu = out_e.mean(dim=(0, 1)).float()  # [E, D]
            ortho = mean_abs_offdiag_cosine(mu)
            self._out_ortho_loss = ortho
            if do_diag:
                self._out_ortho_cos_sim = float(ortho.detach().item())
        elif do_diag:
            with torch.no_grad():
                mu = out_e.mean(dim=(0, 1)).float()  # [E, D]
                self._out_ortho_cos_sim = float(mean_abs_offdiag_cosine(mu).item())
        return out_e

    def _attn_shared_from_normed(self, x_n: Tensor) -> Tensor:
        """Shared attention path: returns y [B,T,D] after SDPA and per-head gating.

        Expects x_n already pre-RMSNorm'd.
        """
        bsz, seqlen, dim = x_n.shape
        q_and_gate = self.c_q(x_n)  # [B, T, dim + num_heads]
        q_raw = q_and_gate[..., :dim].reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        gate_logits = q_and_gate[..., dim:].reshape(bsz, seqlen, self.num_heads, 1).transpose(1, 2)
        q_rope, q_nope = q_raw[..., :self.rope_dim], q_raw[..., self.rope_dim:]

        kv_latent = self.c_kv_down(x_n)
        k_nope = self.c_k_nope(kv_latent).reshape(bsz, seqlen, self.num_kv_heads, self.nope_dim).transpose(1, 2)
        v = self.c_v(kv_latent).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_rope = self.c_k_rope(x_n).reshape(bsz, seqlen, self.num_kv_heads, self.rope_dim).transpose(1, 2)

        q_rope, q_nope = _rms_norm(q_rope), _rms_norm(q_nope)
        k_rope, k_nope = _rms_norm(k_rope), _rms_norm(k_nope)

        cos, sin = self.rotary(seqlen, x_n.device, q_rope.dtype)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos, sin)

        q_full = torch.cat([q_rope, q_nope], dim=-1)
        k_full = torch.cat([k_rope, k_nope], dim=-1)
        q_full = q_full * self.q_gain.to(dtype=q_full.dtype)[None, :, None, None]

        try:
            y = F.scaled_dot_product_attention(
                q_full, k_full, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        except TypeError:
            k_use, v_use = k_full, v
            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k_use = k_full.repeat_interleave(rep, dim=1)
                v_use = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q_full, k_use, v_use, attn_mask=None, is_causal=True)

        y = y * torch.sigmoid(gate_logits.to(dtype=y.dtype) + self.gate_bias[None, :, None, None].to(y.dtype))
        return y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

    def forward_expert(self, x: Tensor, expert_idx: int) -> Tensor:
        """Return a single expert attention output [B, T, D] without materializing [B,T,E,D]."""
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x_n = _rms_norm(x)
        y = self._attn_shared_from_normed(x_n)  # [B,T,D]
        return self.project_expert_from_shared(y, expert_idx)

    def project_expert_from_shared(self, y: Tensor, expert_idx: int) -> Tensor:
        """Apply expert low-rank projections to a shared attention output y [B,T,D]."""
        e = int(expert_idx)
        proj = self.expert_proj[e].to(dtype=y.dtype)  # [R,D]
        out = self.expert_out[e].to(dtype=y.dtype)    # [D,R]
        h = torch.einsum('btd,rd->btr', y, proj)
        return torch.einsum('btr,dr->btd', h, out)

    def forward(self, x: Tensor) -> Tensor:
        raise RuntimeError("CausalSelfAttention routing is handled at the Block level; use forward_experts().")

    @property
    def attn_gate(self) -> Tensor:
        # Backwards-compat for older tests/diagnostics.
        return self.gate_bias

    def get_expert_diagnostics(self) -> dict:
        """Bridge to router diagnostics + orthogonality from expert weights."""
        diag: dict = {}
        r = self.attn_router
        if r._expert_usage is not None:
            diag["usage"] = r._expert_usage
            diag["entropy"] = r._expert_entropy
            diag["balance_cv"] = r._expert_balance_cv
        if self._out_ortho_cos_sim is not None:
            diag["ortho_cos_sim"] = self._out_ortho_cos_sim
        return diag


class MLP(nn.Module):
    """SiLU-gated MLP expert bank (routing performed at Block level)."""
    def __init__(
        self,
        dim: int,
        mlp_mult: float,
        num_experts: int = 6,
        expert_rank: int = 0,
        router: SoftDenseRouter | None = None,
    ):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(hidden // max(self.num_experts, 1), 1)
        self.expert_gate = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_fc = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_down = nn.Parameter(torch.empty(num_experts, dim, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_gate.data[e])
            nn.init.xavier_uniform_(self.expert_fc.data[e])
            nn.init.xavier_uniform_(self.expert_down.data[e])
        # Soft dense routing. In block-level MoE we pass in a shared router from the parent block.
        self.mlp_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        self.out_ortho_coef = 0.0
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None

    def forward_experts(self, x: Tensor) -> Tensor:
        """Return per-expert MLP outputs [B, T, E, D] (no routing mix).

        Supports both:
        - x: [B, T, D]   (component MoE; all experts see same x)
        - x: [B, T, E, D] (block MoE; expert e sees its own stream x[...,e,:])
        """
        if x.ndim not in (3, 4):
            raise ValueError(f"MLP expects x rank 3 or 4, got shape {tuple(x.shape)}")
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x = _rms_norm(x)
        # einsum requires matching dtypes; keep compute in activation dtype under autocast.
        expert_gate = self.expert_gate.to(dtype=x.dtype)
        expert_fc = self.expert_fc.to(dtype=x.dtype)
        expert_down = self.expert_down.to(dtype=x.dtype)
        if x.ndim == 3:
            gate_h = torch.einsum('btd,esd->btes', x, expert_gate)
            fc_h = torch.einsum('btd,esd->btes', x, expert_fc)
        else:
            gate_h = torch.einsum('bted,esd->btes', x, expert_gate)
            fc_h = torch.einsum('bted,esd->btes', x, expert_fc)
        h = F.silu(gate_h) * fc_h  # [B, T, E, expert_rank]
        out_e = torch.einsum('btes,eds->bted', h, expert_down)
        self._out_ortho_loss = None
        need_loss = bool(self.training and torch.is_grad_enabled() and self.out_ortho_coef > 0.0)
        do_diag = (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        if do_diag and dist.is_available() and dist.is_initialized():
            do_diag = dist.get_rank() == 0
        if need_loss:
            mu = out_e.mean(dim=(0, 1)).float()  # [E, D]
            ortho = mean_abs_offdiag_cosine(mu)
            self._out_ortho_loss = ortho
            if do_diag:
                self._out_ortho_cos_sim = float(ortho.detach().item())
        elif do_diag:
            with torch.no_grad():
                mu = out_e.mean(dim=(0, 1)).float()  # [E, D]
                self._out_ortho_cos_sim = float(mean_abs_offdiag_cosine(mu).item())
        return out_e

    def forward_expert(self, x: Tensor, expert_idx: int) -> Tensor:
        """Return a single expert MLP output [B, T, D] without materializing [B,T,E,D]."""
        if x.ndim != 3:
            raise ValueError(f"forward_expert expects x rank 3 [B,T,D], got shape {tuple(x.shape)}")
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x = _rms_norm(x)
        e = int(expert_idx)
        expert_gate = self.expert_gate[e].to(dtype=x.dtype)  # [R,D]
        expert_fc = self.expert_fc[e].to(dtype=x.dtype)      # [R,D]
        expert_down = self.expert_down[e].to(dtype=x.dtype)  # [D,R]
        gate_h = torch.einsum('btd,sd->bts', x, expert_gate)
        fc_h = torch.einsum('btd,sd->bts', x, expert_fc)
        h = F.silu(gate_h) * fc_h
        return torch.einsum('bts,ds->btd', h, expert_down)

    def forward(self, x: Tensor) -> Tensor:
        raise RuntimeError("MLP routing is handled at the Block level; use forward_experts().")

    @property
    def router(self) -> SoftDenseRouter:
        # Backwards-compat for older tests/diagnostics.
        return self.mlp_router

    def get_expert_diagnostics(self) -> dict:
        """Bridge to router diagnostics + orthogonality from expert weights."""
        diag: dict = {}
        r = self.mlp_router
        if r._expert_usage is not None:
            diag["usage"] = r._expert_usage
            diag["entropy"] = r._expert_entropy
            diag["balance_cv"] = r._expert_balance_cv
        if self._out_ortho_cos_sim is not None:
            diag["ortho_cos_sim"] = self._out_ortho_cos_sim
        return diag


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        # Initialize near-1: mostly current token embedding, minimal previous-token injection.
        self.gate = nn.Parameter(torch.full((dim,), SIGMOID_ONE_INIT_LOGIT, dtype=torch.float32))  # sigmoid(6)=0.9975

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1 - g) * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
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
            # Pre-RMSNorm: normalize immediately before weight multiplication.
            h = self.proj(_rms_norm(h))
        return h * self.scale.to(dtype=h.dtype)


def _fsq_ste(x: Tensor, num_levels: int, training: bool) -> Tensor:
    """FSQ with straight-through estimator: tanh → round to nearest level."""
    x_bounded = torch.tanh(x)
    step = 2.0 / (num_levels - 1)
    if training:
        x_q = torch.round(x_bounded / step) * step
        return x_bounded + (x_q - x_bounded).detach()
    return torch.round(x_bounded / step) * step


class MoSHead(nn.Module):
    """Mixture-of-Softmaxes dual head (Constraint #4+#5).

    Architecture: shared experts + specialized experts per head.
    - Shared experts: used by BOTH CTP and NTP (parameter efficient)
    - CTP-specialized expert: only for denoising head
    - NTP-specialized expert: only for next-token head
    FSQ applied in intermediate space (Constraint #4).
    """
    def __init__(self, d_model: int, vocab_size: int, rank: int = 64,
                 num_shared: int = 2, num_specialized: int = 1, fsq_levels: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.num_shared = num_shared
        self.num_specialized = num_specialized
        self.num_experts = num_shared + num_specialized  # per head
        self.fsq_levels = fsq_levels
        # Gate: routes over shared + specialized experts per head
        # CTP gate: num_shared + num_specialized
        # NTP gate: num_shared + num_specialized (different specialized)
        self.gate_ctp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
        self.gate_ntp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
        # MoS uses pure softmax routing (convex combination, sums to 1) per Mixtape paper.
        # No sigmoid gates — the softmax bottleneck is broken by the mixture itself.
        # Shared A projections: [num_shared, d_model, rank]
        self.A_shared = nn.Parameter(torch.empty(num_shared, d_model, rank))
        # Specialized A projections: 1 for CTP, 1 for NTP
        self.A_ctp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        self.A_ntp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        # Dual B matrices (shared across all experts within each head)
        self.B_denoise = nn.Parameter(torch.empty(vocab_size, rank))
        self.B_NTP = nn.Parameter(torch.empty(vocab_size, rank))
        self._diag_step: int | None = None
        self._ctp_ortho_out: Tensor | None = None
        self._ntp_ortho_out: Tensor | None = None
        self._init_params()

    def _init_params(self):
        for gate in [self.gate_ctp, self.gate_ntp]:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.zeros_(gate.bias)
        for A in [self.A_shared, self.A_ctp, self.A_ntp]:
            for e in range(A.shape[0]):
                nn.init.xavier_uniform_(A.data[e])
        nn.init.xavier_uniform_(self.B_denoise)
        nn.init.xavier_uniform_(self.B_NTP)

    def init_from_embedding(self, embed_weight: Tensor):
        """No-op: all params use xavier init from _init_params (training from scratch).

        SVD init was removed because it biases shared experts toward current-token
        prediction, creating a CTP/NTP gradient conflict.
        """
        pass

    def get_head_orthogonality(self, head: str) -> float:
        """Return latest output/latent-space orthogonality for the head (logged metric).

        Falls back to weight-space cosine only if no output-space value was computed yet.
        """
        if head not in ("ctp", "ntp"):
            raise ValueError(f"head must be 'ctp' or 'ntp', got {head!r}")
        t = self._ctp_ortho_out if head == "ctp" else self._ntp_ortho_out
        if t is not None:
            return float(t.detach().float().item())
        if self.num_shared + self.num_specialized < 2:
            return 0.0
        A_spec = self.A_ctp if head == "ctp" else self.A_ntp
        with torch.no_grad():
            groups = (
                torch.cat([self.A_shared, A_spec], dim=0)
                .float()
                .reshape(self.num_shared + self.num_specialized, -1)
            )
            return float(max_abs_offdiag_cosine(groups).item())

    def _fsq(self, x: Tensor) -> Tensor:
        return _fsq_ste(x, self.fsq_levels, self.training)

    def _head_forward(self, x: Tensor, gate: nn.Linear, A_shared: Tensor,
                      A_spec: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (log_probs, alpha_softmax, ortho_out) for one head.

        `ortho_out` is output/latent-space orthogonality: max off-diagonal |cos|
        between per-expert mean pre-FSQ latents (x @ A_e).
        """
        N = x.shape[0]
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        x = _rms_norm(x)
        alpha = F.softmax(gate(x).float(), dim=-1)  # [N, num_shared+num_spec] — convex combination
        log_w = alpha.clamp(min=1e-8).log()
        log_p_unnorm = x.new_full((N, self.vocab_size), -torch.inf, dtype=torch.float32)
        mu_groups: list[Tensor] = []
        # Shared experts
        for e in range(self.num_shared):
            t = x.to(A_shared.dtype) @ A_shared[e]  # [N, rank]
            mu_groups.append(t.float().mean(dim=0))
            u = self._fsq(t)
            logits = u.to(B.dtype) @ B.t()
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, e:e+1] + F.log_softmax(logits.float(), dim=-1))
        # Specialized experts
        for e in range(self.num_specialized):
            t = x.to(A_spec.dtype) @ A_spec[e]  # [N, rank]
            mu_groups.append(t.float().mean(dim=0))
            u = self._fsq(t)
            logits = u.to(B.dtype) @ B.t()
            idx = self.num_shared + e
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, idx:idx+1] + F.log_softmax(logits.float(), dim=-1))
        log_p = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=-1, keepdim=True)
        ortho_out = max_abs_offdiag_cosine(torch.stack(mu_groups, dim=0)) if len(mu_groups) >= 2 else x.new_zeros(())
        return log_p, alpha, ortho_out

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return (log_p_denoise, log_p_ntp), each [*, V]."""
        orig_shape = h.shape[:-1]
        x = h.reshape(-1, self.d_model)

        log_p_d, alpha_d, ortho_ctp = self._head_forward(x, self.gate_ctp, self.A_shared, self.A_ctp, self.B_denoise)
        log_p_n, alpha_n, ortho_ntp = self._head_forward(x, self.gate_ntp, self.A_shared, self.A_ntp, self.B_NTP)
        self._ctp_ortho_out = ortho_ctp
        self._ntp_ortho_out = ortho_ntp

        if self.training:
            bal = torch.tensor(0.0, device=x.device)
            for alpha_soft in [alpha_d, alpha_n]:
                mean_a = alpha_soft.mean(dim=0)
                target = torch.ones_like(mean_a) / alpha_soft.shape[-1]
                bal = bal + F.mse_loss(mean_a, target)
            self._balance_loss = bal
            # No sparsity loss for MoS: pure softmax has constant mean (1/E), no gradient signal
            self._sparsity_loss = torch.tensor(0.0, device=x.device)
            with torch.no_grad():
                do_diag = bool(_ROUTER_DIAGNOSTICS_ACTIVE)
                if do_diag and dist.is_available() and dist.is_initialized():
                    do_diag = dist.get_rank() == 0
                if do_diag:
                    for alpha_soft, usage_attr, ent_attr, cv_attr in [
                        (alpha_d, '_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        (alpha_n, '_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        a = alpha_soft.detach()
                        mean_a = a.mean(dim=0)
                        setattr(self, usage_attr, mean_a.float().cpu().tolist())
                        per_token_ent = -(a * (a + 1e-8).log()).sum(-1)
                        setattr(self, ent_attr, per_token_ent.mean().item())
                        setattr(self, cv_attr, (mean_a.std() / mean_a.mean().clamp_min(1e-8)).item())
                    self._diag_step = _ROUTER_DIAGNOSTICS_STEP
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._sparsity_loss = torch.tensor(0.0, device=x.device)
            with torch.no_grad():
                do_diag = True
                if dist.is_available() and dist.is_initialized():
                    do_diag = dist.get_rank() == 0
                if do_diag:
                    for alpha_soft, usage_attr, ent_attr, cv_attr in [
                        (alpha_d, '_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        (alpha_n, '_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        mean_a = alpha_soft.mean(dim=0)
                        setattr(self, usage_attr, mean_a.float().cpu().tolist())
                        per_token_ent = -(alpha_soft * (alpha_soft + 1e-8).log()).sum(-1)
                        setattr(self, ent_attr, per_token_ent.mean().item())
                        setattr(self, cv_attr, (mean_a.std() / mean_a.mean()).item())
                    self._diag_step = _ROUTER_DIAGNOSTICS_STEP
                else:
                    for usage_attr, ent_attr, cv_attr in [
                        ('_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        ('_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        setattr(self, usage_attr, None)
                        setattr(self, ent_attr, None)
                        setattr(self, cv_attr, None)
                    self._diag_step = None

        return log_p_d.view(*orig_shape, -1), log_p_n.view(*orig_shape, -1)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, kv_latent_dim: int = 0,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        # Block-level Soft Dense Routing: one router for the whole block (paired experts).
        # Routing weights are pure softmax over experts (sum=1). A separate per-token
        # residual gate gg is used for DEQ stability (not a post-softmax gate).
        shared_router = SoftDenseRouter(dim, 6, enable_gate=False)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            kv_latent_dim=kv_latent_dim,
            expert_rank=attn_expert_rank,
            router=shared_router,
        )
        self.mlp = MLP(dim, mlp_mult, expert_rank=mlp_expert_rank, router=shared_router)
        # Global residual gate for the DEQ iteration update: gg(x) ∈ (0,1) per token.
        self.gg_gate = CastedLinear(dim, 1, bias=True)
        with torch.no_grad():
            self.gg_gate.weight.zero_()
            # Start unsaturated (sigmoid(0)=0.5) so gg can quickly adapt to contraction needs.
            self.gg_gate.bias.zero_()
        self._gg_track_enabled = False
        self._gg_sum = 0.0
        self._gg_count = 0
        self._gg_last: float | None = None
        self._block_ortho_cos_sim: float | None = None
        # Fine-grained gg tracking: per Block.forward call (used to derive gg by DEQ iteration).
        self._gg_call_track_enabled = False
        self._gg_call_track: list[float] = []

        # Input-conditioned injection gate (hypernet).
        # Gate uses pooled pre-injection state u=mean_{b,t}(z_in) to keep the cost small and
        # stable under DEQ iteration. The gate is per-dimension:
        #   g = sigmoid(u ⊙ w_inj + b_inj) ∈ (0,1)^d
        # Injection is convex:
        #   z_inj = (1-g)⊙z_in + g⊙x0
        self.inj_w = nn.Parameter(torch.zeros((dim,), dtype=torch.float32))
        self.inj_b = nn.Parameter(torch.empty((dim,), dtype=torch.float32))
        # Diagnostics: mean gate value on the last forward call.
        self._inj_gate_last_mean: float | None = None
        with torch.no_grad():
            # Conservative init: injection almost-off at start to preserve near-identity behavior.
            self.inj_b.fill_(-6.0)

    def _inj_gate_from(self, z_in: Tensor) -> Tensor:
        """Return per-dimension injection gate g ∈ (0,1)^d computed from pooled z_in."""
        # Pre-RMSNorm: normalize immediately before weight multiplication.
        z_n = _rms_norm(z_in)
        u = z_n.float().mean(dim=(0, 1))  # [d]
        logits = u * self.inj_w + self.inj_b
        g = torch.sigmoid(logits)  # [d]
        # Avoid Python-side scalar extraction in the hot path (torch.compile friendly).
        if (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE):
            self._record_inj_diag(g.detach())
        return g

    @dynamo_disable
    def _record_inj_diag(self, g_detached: Tensor) -> None:
        self._inj_gate_last_mean = float(g_detached.mean().item())

    @dynamo_disable
    def _record_gg_diag(self, gg_tok_detached: Tensor) -> None:
        gg_val = float(gg_tok_detached.float().mean().item())
        self._gg_last = gg_val
        if self._gg_track_enabled:
            self._gg_sum += gg_val
            self._gg_count += 1
        if self._gg_call_track_enabled:
            self._gg_call_track.append(gg_val)

    @dynamo_disable
    def _record_block_ortho(self, block_mu: list[Tensor]) -> None:
        self._block_ortho_cos_sim = None
        if len(block_mu) < 2:
            return
        try:
            mu_b = torch.stack(block_mu, dim=0)
            self._block_ortho_cos_sim = float(max_abs_offdiag_cosine(mu_b).item())
        except Exception:
            self._block_ortho_cos_sim = None

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        z_in = x

        g = self._inj_gate_from(z_in).to(dtype=x.dtype)  # [d]
        x = (1.0 - g)[None, None, :] * z_in + g[None, None, :] * x0
        x_attn = self.attn_norm(x)
        x_attn_n = _rms_norm(x_attn)
        # One router for the whole block (paired expert blocks): compute weights once.
        w = self.attn.attn_router(x_attn)  # shared router instance; sum(w)=1
        gg_tok = torch.sigmoid(self.gg_gate(x_attn_n)).squeeze(-1)  # [B,T]
        if self._gg_track_enabled or self._gg_call_track_enabled:
            self._record_gg_diag(gg_tok.detach())

        # Memory-safe expert mixing: stream experts without materializing [B,T,E,D].
        # Update: z_out = (1-gg)*z_in + gg*Σ_e w_e * expert_e(z_in, x0)
        x_mix = (1.0 - gg_tok).to(dtype=x.dtype).unsqueeze(-1) * z_in
        need_loss = bool(self.training and torch.is_grad_enabled())
        do_diag = (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        if do_diag and dist.is_available() and dist.is_initialized():
            do_diag = dist.get_rank() == 0

        block_mu: list[Tensor] = []
        # Compute shared attention output once, then project per expert.
        y_shared = self.attn._attn_shared_from_normed(x_attn_n)  # [B,T,D]
        for e in range(self.attn.num_experts):
            attn_out = self.attn.project_expert_from_shared(y_shared, e)  # [B,T,D]
            z1 = x + attn_out
            mlp_out = self.mlp.forward_expert(self.mlp_norm(z1), e)  # [B,T,D]
            z2 = z1 + mlp_out
            x_mix = x_mix + (gg_tok.to(dtype=x.dtype).unsqueeze(-1) * w[..., e:e+1]) * z2
            if do_diag:
                # Track the *weighted* per-expert block contribution, since this is what
                # actually gets mixed into the DEQ update: contrib_e = (gg*w_e) * z2_e.
                contrib = (gg_tok.to(dtype=z2.dtype).unsqueeze(-1) * w[..., e:e+1]) * z2
                block_mu.append(contrib.mean(dim=(0, 1)).float())

        if do_diag:
            # Block expert output-space orthogonality (metric-only): mean |cos| across per-expert
            # mean *weighted* block outputs (gg*w_e*z2). This captures redundancy/collapse at the
            # actual mixed expert contribution level.
            self._record_block_ortho(block_mu)

        return x_mix


class RevDEQFunction(torch.autograd.Function):
    """Memory-efficient backward for RevDEQ coupled-state iteration (arxiv:2509.12917).

    Forward runs under no_grad, saves only terminal (y, z) states.
    Backward reconstructs states in reverse using fp64 accumulators and
    computes per-step VJPs. O(1) activation memory vs O(K) for standard autograd.
    """
    @staticmethod
    def _autocast_off_ctx(device_type: str) -> contextlib.AbstractContextManager[None]:
        try:
            return torch.autocast(device_type=device_type, enabled=False)
        except Exception:
            return contextlib.nullcontext()

    @staticmethod
    def _autocast_like_ctx(device_type: str, dtype: torch.dtype) -> contextlib.AbstractContextManager[None]:
        if device_type == "cpu":
            return RevDEQFunction._autocast_off_ctx(device_type)
        if dtype not in (torch.float16, torch.bfloat16):
            return RevDEQFunction._autocast_off_ctx(device_type)
        try:
            return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
        except Exception:
            return contextlib.nullcontext()

    @staticmethod
    def forward(ctx, f_theta, x0, z_init, beta, K, *params):
        acc_dtype = torch.float64
        state_dtype = torch.float32
        compute_dtype = z_init.dtype
        device_type = x0.device.type
        beta_inv = 1.0 - beta
        # Snapshot diagnostic intent: the surrounding `router_diagnostics(...)` context
        # exits before backward runs, so backward must not consult global flags.
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
                    out_z = f_theta(z_state.to(compute_dtype), x0)
                y_acc = y_acc + out_z.to(acc_dtype) * beta
                y_state = y_acc.to(state_dtype)

                z_acc = z_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_state.to(compute_dtype), x0)
                z_acc = z_acc + out_y.to(acc_dtype) * beta
                z_state = z_acc.to(state_dtype)

            # Cheap residual proxy available without extra block calls.
            # Uses the last computed f(y) from the final forward iteration.
            if out_y is not None:
                try:
                    # f_theta may be a compiled callable; only attach if attribute assignment works.
                    setattr(f_theta, "_deq_residual_proxy", float((z_state - out_y.to(state_dtype)).norm().item()))
                except Exception:
                    try:
                        setattr(f_theta, "_deq_residual_proxy", None)
                    except Exception:
                        pass

        ctx.save_for_backward(x0.detach(), y_state.detach(), z_state.detach(), z_prev_state.detach())
        ctx.z_init_state = z_init_state
        ctx.f_theta = f_theta
        ctx.beta = beta
        ctx.beta_inv = beta_inv
        ctx.K = K
        ctx.compute_dtype = compute_dtype
        ctx.device_type = device_type
        ctx.params = params
        return z_state.to(compute_dtype), z_prev_state.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_z, _grad_z_prev_ignored):
        x0, y_terminal, z_terminal, _z_prev_terminal = (t.detach() for t in ctx.saved_tensors)
        z_init_state = getattr(ctx, "z_init_state", None)
        f_theta = ctx.f_theta
        beta = ctx.beta
        beta_inv = ctx.beta_inv
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

        y_next = y_terminal
        z_next = z_terminal

        for _ in range(K):
            # VJP 1: f(y_{n+1}, x0) — used to reconstruct z_n
            y_local = y_next.detach().to(compute_dtype).requires_grad_()
            x_local = x0.detach().to(x0.dtype).requires_grad_()
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_local, x_local)

            # Reconstruct z_n: z_n = (z_{n+1} - beta*f(y_{n+1})) / (1-beta)
            z_n = (z_next.to(acc_dtype) - out_y.detach().to(acc_dtype) * beta) / beta_inv
            z_n = z_n.to(state_dtype)

            grad_seed_y = (beta * bar_z).to(out_y.dtype)
            grads_y = torch.autograd.grad(
                out_y, (y_local, x_local, *params_req),
                grad_outputs=grad_seed_y, allow_unused=True,
            )
            vjp_y = grads_y[0].to(state_dtype)
            bar_y_acc = bar_y + vjp_y

            # VJP 2: f(z_n, x0) — used to reconstruct y_n
            z_local = z_n.detach().to(compute_dtype).requires_grad_()
            x_local2 = x0.detach().to(x0.dtype).requires_grad_()
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_z = f_theta(z_local, x_local2)

            # Reconstruct y_n: y_n = (y_{n+1} - beta*f(z_n)) / (1-beta)
            y_n = (y_next.to(acc_dtype) - out_z.detach().to(acc_dtype) * beta) / beta_inv
            y_n = y_n.to(state_dtype)

            grad_seed_z = (beta * bar_y_acc).to(out_z.dtype)
            grads_z = torch.autograd.grad(
                out_z, (z_local, x_local2, *params_req),
                grad_outputs=grad_seed_z, allow_unused=True,
            )
            vjp_z = grads_z[0].to(state_dtype)

            # Adjoint updates
            bar_z = beta_inv * bar_z + vjp_z
            bar_y = beta_inv * bar_y_acc

            # Accumulate param + x grads
            for j in range(len(params_req)):
                gy = grads_y[2 + j]
                gz = grads_z[2 + j]
                if gy is None and gz is None:
                    continue
                g = (gy if gy is not None else 0.0) + (gz if gz is not None else 0.0)
                g = g.detach()
                cur_param_grads_req[j] = g if cur_param_grads_req[j] is None else cur_param_grads_req[j] + g
            if grads_y[1] is not None:
                cur_x_grad += grads_y[1].detach().float()
            if grads_z[1] is not None:
                cur_x_grad += grads_z[1].detach().float()

            y_next, z_next = y_n, z_n

        # Reconstruction diagnostic: should be near 0 when fp64 add/sub reconstruction is consistent.
        # Only meaningful for the RevDEQ backward path. We store the scalar on the module so the
        # training loop can log it without re-running extra compute.
        if bool(getattr(ctx, "do_recon_diag", False)) and isinstance(z_init_state, torch.Tensor):
            try:
                z0 = z_init_state.to(dtype=state_dtype)
                denom = max(float(z0.norm().item()), 1.0)
                recon_err = float(((z_next - z0).norm().item() + (y_next - z0).norm().item()) / denom)
                setattr(f_theta, "_deq_recon_error_last_bwd", recon_err)
            except Exception:
                pass

        # z_init gradient = bar_y + bar_z (both adjoint states at step 0)
        z_init_grad = (bar_y + bar_z).to(x0.dtype)
        # Map required grads back to full *params list for autograd/DDP.
        param_grads_out: list[torch.Tensor | None] = [None] * len(params_all)
        for j, all_idx in enumerate(req_indices):
            g = cur_param_grads_req[j]
            if g is None:
                continue
            param_grads_out[all_idx] = g.to(dtype=params_all[all_idx].dtype)

        # Returns: (f_theta, x0, z_init, beta, K, *params_all)
        return (None, cur_x_grad.to(x0.dtype), z_init_grad, None, None, *param_grads_out)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        kv_latent_dim: int = 0,
        num_refinements: int = 1,
        attn_expert_rank: int = 0,
        mlp_expert_rank: int = 0,
        deq_beta: float = 0.5,
        attn_balance_mult: float = 3.0,
        mlp_balance_mult: float = 1.0,
        bal_loss_coef: float = 0.5,
        mos_ortho_out_coef: float = 0.0,
        attn_ortho_out_coef: float = 0.0,
        mlp_ortho_out_coef: float = 0.0,
        deq_backward: str = "autograd",
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers       # DEQ solver iters per refinement
        self.num_refinements = num_refinements  # predict→soft_embed→re-encode cycles
        # Backwards-compat: older code/tests expect blocks to exist but be unused in DEQ mode.
        self.blocks = None
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        # SmearGate removed: keep the embedding path simple and avoid injecting previous-token mixing.
        self.smear = nn.Identity()
        # RevDEQ (Constraint #1): single shared block with coupled-state fixed-point iteration
        self.shared_block = Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                                  rope_base, qk_gain_init, kv_latent_dim=kv_latent_dim,
                                  attn_expert_rank=attn_expert_rank, mlp_expert_rank=mlp_expert_rank)
        self.deq_beta = float(deq_beta)
        self.attn_balance_mult = float(attn_balance_mult)
        self.mlp_balance_mult = float(mlp_balance_mult)
        self.bal_loss_coef = float(bal_loss_coef)
        self.mos_ortho_out_coef = float(mos_ortho_out_coef)
        self.attn_ortho_out_coef = float(attn_ortho_out_coef)
        self.mlp_ortho_out_coef = float(mlp_ortho_out_coef)
        if deq_backward not in ("autograd", "revdeq"):
            raise ValueError(f"deq_backward must be autograd|revdeq, got {deq_backward!r}")
        self.deq_backward = deq_backward
        # Route the output-space orthogonality coefficients into the expert modules so they
        # can materialize differentiable losses when using autograd-unrolled DEQ.
        self.shared_block.attn.out_ortho_coef = self.attn_ortho_out_coef
        self.shared_block.mlp.out_ortho_coef = self.mlp_ortho_out_coef
        # MoS output head (Constraints #4+#5): shared experts, dual B for CTP/NTP
        self.mos_head = MoSHead(model_dim, vocab_size, rank=256, num_shared=2, num_specialized=1, fsq_levels=8)
        self.final_norm = RMSNorm()
        self._init_weights()

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
        # Ensure sigmoid gate logits start at midpoint 0.5 by forcing their logit-producing
        # weights to 0 after generic init passes.
        with torch.no_grad():
            attn = self.shared_block.attn
            dim = self.tok_emb.embedding_dim
            attn.c_q.weight[dim:, :].zero_()
        # Initialize MoS head from embedding weights
        self.mos_head.init_from_embedding(self.tok_emb.weight.data)

    def _get_soft_embedding(self, z: Tensor, topk: int = 128) -> Tensor:
        """Diffusion-AR refinement: build a *full* refined embedding from CTP + shifted NTP predictions.

        For each position i, combines two signals:
        - CTP[i] from MoSHead: predicts token at position i (from context 0..i)
        - NTP[i-1] from MoSHead: predicts next token after i-1 (= token i)
        Shift NTP by one position to align predictions with the token index.
        For position 0 (no previous token), use CTP only.

        Convert log-probs to valid probability distributions, mix, take top-k,
        build sparse expected embedding, then project it into the same backbone-input
        space as x0 (RMSNorm).
        """
        def _logp_to_prob(log_p: Tensor) -> Tensor:
            # MoSHead returns normalized log-probabilities; convert to a numerically-stable
            # probability distribution (sum≈1) in fp32, then renormalize to guarantee validity.
            p = log_p.float().exp()
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.no_grad():
            h = self.final_norm(z)
            was_training = self.mos_head.training
            self.mos_head.train(False)
            log_p_ctp, log_p_ntp = self.mos_head(h)  # [B,T,V] log-probs
            self.mos_head.train(was_training)
            # Clear autocast cache to prevent stale weight caching from poisoning
            # subsequent calls with gradients enabled (PyTorch autocast bug).
            torch.clear_autocast_cache()

            log_p_ntp_shifted = torch.cat([log_p_ntp[:, :1], log_p_ntp[:, :-1]], dim=1)
            p_ctp = _logp_to_prob(log_p_ctp)
            p_ntp = _logp_to_prob(log_p_ntp_shifted)
            p_mix = 0.5 * (p_ctp + p_ntp)
            # Position 0 has no previous token; ignore NTP[0] (which predicts token 1).
            p_mix[:, 0] = p_ctp[:, 0]
            p_mix = torch.nan_to_num(p_mix, nan=0.0, posinf=0.0, neginf=0.0)
            # p_ctp and p_ntp are each normalized probability distributions (sum=1); their convex
            # combination is therefore already normalized in exact arithmetic. We intentionally
            # skip a full-vocab renormalization here for speed; the subsequent top-k renorm
            # ensures the sparse mixture is a valid distribution over its retained support.

            k = min(int(topk), int(p_mix.shape[-1]))
            topk_probs, topk_idx = p_mix.topk(k, dim=-1)  # [B,T,K]
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            W = self.tok_emb.weight.data  # [V, d]
            # Memory note: materializing `[B,T,K,d]` can be very large at `K=128`.
            # Compute the expected embedding in chunks to keep peak memory bounded.
            B, T, K = topk_idx.shape
            d = W.shape[1]
            flat_idx = topk_idx.reshape(B * T, K)
            flat_p = topk_probs.reshape(B * T, K)
            flat_out = W.new_empty((B * T, d))
            chunk = 512  # trades a small loop for much lower peak VRAM
            for s in range(0, B * T, chunk):
                e = min(s + chunk, B * T)
                idx_chunk = flat_idx[s:e]  # [N,K]
                p_chunk = flat_p[s:e]      # [N,K]
                emb = F.embedding(idx_chunk, W)  # [N,K,d]
                flat_out[s:e] = (p_chunk.unsqueeze(-1) * emb).sum(dim=1)
            soft_embed = flat_out.reshape(B, T, d)

            # Match the input embedding path as closely as possible (no bigram available for soft tokens).
            soft_embed = _rms_norm(soft_embed.to(dtype=z.dtype))
        return soft_embed

    def _deq_solve(self, x0: Tensor, z_init: Tensor):
        """Run DEQ coupled-state solver.

        - Training:
          - deq_backward=revdeq: RevDEQFunction (O(1) activation memory)
          - deq_backward=autograd: explicit unroll (stores activations; enables output-space regularizers)
        - Eval: explicit unroll for diagnostics (uses fp64 accumulators only for revdeq diagnostics)
        """
        beta = self.deq_beta
        dtype = x0.dtype
        K = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
        # Track gates only when diagnostics are enabled (keeps torch.compile graphs stable and fast).
        track_gg = (not self.training) or bool(_ROUTER_DIAGNOSTICS_ACTIVE)
        self.shared_block._gg_sum = 0.0
        self.shared_block._gg_count = 0
        self.shared_block._gg_track_enabled = bool(track_gg)
        # Also track gg per Block.forward call so we can compute gg by DEQ iteration.
        self.shared_block._gg_call_track = []
        self.shared_block._gg_call_track_enabled = bool(track_gg)
        try:
            # Optional compile path (training only): use eager block for diagnostic steps.
            f_theta = self.shared_block
            if self.training and bool(getattr(self, "compile_train", False)) and not bool(_ROUTER_DIAGNOSTICS_ACTIVE):
                f_theta = getattr(self, "_shared_block_compiled", self.shared_block)
            if self.training and self.deq_backward == "revdeq":
                params = tuple(p for p in self.shared_block.parameters() if p.requires_grad)
                z, z_prev = RevDEQFunction.apply(
                    f_theta, x0, z_init, beta, K, *params
                )
                return z, z_prev, None, None

            # Explicit coupled-state unroll (autograd-enabled in training).
            # For variable-K training (K-jitter) large K can OOM under autograd-unroll due to
            # storing 2*K block activations. Activation checkpointing trades extra compute
            # for much lower memory usage.
            use_ckpt = bool(self.training and self.deq_backward == "autograd" and K >= 5)
            def _f_theta(a: Tensor, b: Tensor) -> Tensor:
                return f_theta(a, b)
            acc_dtype = torch.float64 if (not self.training and self.deq_backward == "revdeq") else torch.float32
            y_acc = z_init.to(acc_dtype)
            z_acc = z_init.to(acc_dtype)
            z = z_init
            z_prev = z
            for _ in range(K):
                z_prev = z
                if use_ckpt:
                    f_z = checkpoint.checkpoint(_f_theta, z, x0, use_reentrant=False)
                else:
                    f_z = f_theta(z, x0)
                y_acc = (1 - beta) * y_acc + beta * f_z.to(acc_dtype)
                y = y_acc.to(dtype)
                if use_ckpt:
                    f_y = checkpoint.checkpoint(_f_theta, y, x0, use_reentrant=False)
                else:
                    f_y = f_theta(y, x0)
                z_acc = (1 - beta) * z_acc + beta * f_y.to(acc_dtype)
                z = z_acc.to(dtype)
            return z, z_prev, y_acc, z_acc
        finally:
            self.shared_block._gg_track_enabled = False
            self.shared_block._gg_call_track_enabled = False
            if self.shared_block._gg_count > 0:
                self._gg_mean_last_solve = float(self.shared_block._gg_sum / self.shared_block._gg_count)
            else:
                self._gg_mean_last_solve = None
            calls = list(getattr(self.shared_block, "_gg_call_track", []) or [])
            if len(calls) == 2 * K:
                self._gg_iter_last_solve = [0.5 * (calls[2 * i] + calls[2 * i + 1]) for i in range(K)]
            else:
                self._gg_iter_last_solve = []

    def _run_backbone(self, x: Tensor) -> Tensor:
        """Decoupled DEQ solver + Diffusion-AR refinement.

        DEQ iters (num_layers): coupled-state solver steps within one solve.
        Refinement steps (num_refinements): predict → soft_embed → re-solve cycles.
        Total block calls = (1 + num_refinements) × num_layers × 2.
        """
        x0 = x
        z = x  # warm start
        dtype = x.dtype
        self._deq_residuals: list[float] = []
        self._gg_iter: list[float] | None = None
        # Avoid leaking stale eval-only diagnostics into train-step logs.
        self._deq_recon_error = None
        self._deq_z_init_last: Tensor | None = None
        self._deq_k_last = None
        prev_soft_embed = x0
        x0_refined = x0  # track for reconstruction

        gg_iters_by_refinement: list[list[float]] = []
        for r in range(1 + self.num_refinements):
            if r > 0:
                new_soft_embed = self._get_soft_embedding(z)
                x0_refined = 0.5 * new_soft_embed + 0.5 * prev_soft_embed
                prev_soft_embed = x0_refined.detach()
                z = x0_refined  # warm start from refined input (not old fixed point)
            else:
                x0_refined = x0

            self._deq_k_last = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
            self._deq_z_init_last = z.detach()
            z, z_prev, y_acc, z_acc = self._deq_solve(x0_refined, z)
            gg_iters_by_refinement.append(list(getattr(self, "_gg_iter_last_solve", []) or []))

        # Aggregate per-DEQ-iteration gates across all refinement solves (r0 one-hot + r>0 soft-embed).
        if gg_iters_by_refinement:
            k_max = max((len(v) for v in gg_iters_by_refinement), default=0)
            if k_max > 0:
                agg: list[float] = []
                for k in range(k_max):
                    vals = [v[k] for v in gg_iters_by_refinement if len(v) > k]
                    agg.append(float(sum(vals) / max(len(vals), 1)))
                self._gg_iter = agg
            else:
                self._gg_iter = []
        else:
            self._gg_iter = None

        # Fixed-point diagnostics (desired goal, not a trained loss).
        if self.training:
            if z_prev is not None:
                abs_conv = (z - z_prev).float().norm().item()
                z_norm_diag = z.detach().float().norm().clamp_min(1.0).item()
                # Iter convergence: last solver update size.
                self._deq_iter_convergence = abs_conv  # absolute for logging
                self._deq_iter_convergence_rel = abs_conv / z_norm_diag  # relative
                # Residual proxy: use RevDEQFunction's cached proxy when available; otherwise
                # fall back to the last update magnitude (always defined in unrolled mode).
                proxy = getattr(self.shared_block, "_deq_residual_proxy", None)
                self._deq_residuals = [float(proxy)] if proxy is not None else [abs_conv]
                do_diag = bool(_ROUTER_DIAGNOSTICS_ACTIVE)
                if do_diag and dist.is_available() and dist.is_initialized():
                    do_diag = dist.get_rank() == 0
                if do_diag:
                    with torch.no_grad():
                        f_z_final = self.shared_block(z, x0_refined)
                        self._deq_residuals = [(z - f_z_final).float().norm().item()]
            if y_acc is not None:
                self._deq_yz_gap = (y_acc - z).float().norm().item()

            # Define router regularizers/diagnostics on the terminal equilibrium state.
            # Use the same router input as in Block.forward (attn_norm) to avoid component
            # mismatches when routers are tied/shared (block-level MoE).
            router = self.shared_block.attn.attn_router
            router.set_bias_stats_enabled(True)
            _ = router(self.shared_block.attn_norm(z))
            router.set_bias_stats_enabled(False)

        # Diagnostics (eval only)
        if not self.training:
            with torch.no_grad():
                f_z_final = self.shared_block(z, x0_refined)
                abs_conv = (z - z_prev).float().norm().item()
                z_norm_diag = z.float().norm().clamp_min(1.0).item()
                self._deq_residuals = [(z - f_z_final).float().norm().item()]
                self._deq_iter_convergence = abs_conv  # absolute
                self._deq_iter_convergence_rel = abs_conv / z_norm_diag  # relative
                # RevDEQ reconstruction diagnostics are produced in the RevDEQ backward path,
                # not during eval-only explicit unroll (which can spuriously diverge due to
                # backend/kernel differences and is not representative of the actual backward).

        return z

    def _encode(self, input_ids: Tensor) -> Tensor:
        """Shared embedding + backbone: input_ids → normalized hidden states."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = _rms_norm(x)
        x = self._run_backbone(x)
        return self.final_norm(x)

    def _collect_routing_losses(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """Collect balance, sparsity, and orthogonality losses from all routers."""
        zero = torch.tensor(0.0, device=device)
        bal, spar, ortho = zero, zero, zero
        # Per-component routing losses with stronger weight for attention (prevents collapse).
        # Deduplicate if routers are tied/shared (block-level MoE).
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
            r_spar = getattr(r, "_sparsity_loss", zero)
            bal = bal + float(router_weights.get(rid, 0.0)) * r_bal
            spar = spar + r_spar
        # MoS head routing
        bal = bal + getattr(self.mos_head, '_balance_loss', zero)
        spar = spar + getattr(self.mos_head, '_sparsity_loss', zero)
        return bal, spar, ortho

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        log_p_ctp, log_p_ntp = self.mos_head(x)
        V = self.tok_emb.num_embeddings
        ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_ids.reshape(-1))
        ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_ids.reshape(-1))
        bal_loss, spar_loss, _ = self._collect_routing_losses(ntp_loss.device)
        self._ntp_loss = ntp_loss.detach().item()
        self._ctp_loss = ctp_loss.detach().item()
        # CTP weight scales with refinement steps: at step 0 input is clean one-hot,
        # CTP becomes meaningful only after soft embedding refinement
        ctp_weight = 0.1 * self.num_refinements
        mos_ortho_loss = torch.tensor(0.0, device=ntp_loss.device)
        if getattr(self.mos_head, "_ctp_ortho_out", None) is not None and getattr(self.mos_head, "_ntp_ortho_out", None) is not None:
            mos_ortho_loss = self.mos_head._ctp_ortho_out + self.mos_head._ntp_ortho_out
        # Expert output-space orthogonality regularization (principled):
        # Only differentiable in autograd-unroll mode.
        zero = torch.tensor(0.0, device=ntp_loss.device)
        attn_ortho_loss = zero
        mlp_ortho_loss = zero
        if self.deq_backward == "autograd":
            a = getattr(self.shared_block.attn, "_out_ortho_loss", None)
            m = getattr(self.shared_block.mlp, "_out_ortho_loss", None)
            if isinstance(a, torch.Tensor):
                attn_ortho_loss = a.to(device=ntp_loss.device)
            if isinstance(m, torch.Tensor):
                mlp_ortho_loss = m.to(device=ntp_loss.device)

        return (
            ntp_loss
            + ctp_weight * ctp_loss
            + self.bal_loss_coef * bal_loss
            + 0.001 * spar_loss
            + self.mos_ortho_out_coef * mos_ortho_loss
            + self.attn_ortho_out_coef * attn_ortho_loss
            + self.mlp_ortho_out_coef * mlp_ortho_loss
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        _, log_p_ntp = self.mos_head(x)
        return log_p_ntp


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
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
            # MoS head returns log-probs directly → use nll_loss
            nll = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
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
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct = done / len(my_windows) * 100
                running_bpb = 0.0
                if token_count.item() > 0:
                    rl = (loss_sum / token_count).item()
                    running_bpb = rl / math.log(2.0) * (token_count.item() / byte_count.item())
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={running_bpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    cli_overrides = _parse_cli_overrides(sys.argv[1:])
    args = Hyperparameters()
    for k, v in cli_overrides.items():
        setattr(args, k, v)
    # Resolve derived paths after CLI overrides.
    args.train_files = os.path.join(args.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")
    if not getattr(args, "run_id", ""):
        args.run_id = str(uuid.uuid4())

    # Shuffle-bag K-jitter range ramp validation (maxK ramps from deq_k_max_start -> deq_k_max).
    if int(args.deq_k_min) <= 0:
        raise ValueError("deq_k_min must be positive")
    if int(args.deq_k_max_start) < int(args.deq_k_min):
        raise ValueError("deq_k_max_start must be >= deq_k_min")
    if int(args.deq_k_max) < int(args.deq_k_max_start):
        raise ValueError("deq_k_max must be >= deq_k_max_start")
    if int(args.deq_k_max) > int(args.num_layers):
        raise ValueError(f"deq_k_max must be <= num_layers={int(args.num_layers)}")
    if int(args.deq_k_max_ramp_steps) <= 0:
        raise ValueError("deq_k_max_ramp_steps must be positive")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    # Auto mode: keep per-rank microbatches reasonably small across GPU counts.
    grad_accum_steps = max(1, math.ceil(8 / world_size))
    # Ensure we can form at least 1 sequence per rank per micro-step.
    global_seqs = args.train_batch_tokens // args.train_seq_len
    while grad_accum_steps > 1 and global_seqs < world_size * grad_accum_steps:
        grad_accum_steps -= 1
    if global_seqs < world_size * grad_accum_steps:
        raise ValueError(
            "TRAIN_BATCH_TOKENS too small for DDP config: "
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} TRAIN_SEQ_LEN={args.train_seq_len} "
            f"WORLD_SIZE={world_size} GRAD_ACCUM_STEPS={grad_accum_steps}. "
            "Increase TRAIN_BATCH_TOKENS or set a smaller GRAD_ACCUM_STEPS."
        )
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    # Keep a safe fallback backend enabled so eval never fails with "Invalid backend" when
    # FlashAttention is temporarily unsupported for a given shape/dtype.
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        # Truncate run log at the start of each run (prevents multi-run concatenation).
        with open(logfile, "w", encoding="utf-8") as f:
            f.write("")
        print(logfile, flush=True)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    k_rng = random.Random(args.seed + 12345)
    k_sampler = KShuffleBagSampler(args.deq_k_min, args.deq_k_max_start, k_rng)

    def sample_deq_k() -> int:
        """Sample one DEQ iteration count K for the next optimizer step (rank0-decided, broadcast)."""
        k = k_sampler.sample() if rank == 0 else 0
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        return int(k)

    def deq_k_for_step(step_i: int) -> int:
        """Choose K for this optimizer step (rank0-decided, broadcast)."""
        k = 0
        if rank == 0:
            if args.deq_k_jitter:
                cur_max = deq_maxk_ramp(
                    step_i,
                    start=int(args.deq_k_max_start),
                    end=int(args.deq_k_max),
                    ramp_steps=int(args.deq_k_max_ramp_steps),
                )
                k_sampler.set_range(int(args.deq_k_min), int(cur_max))
                k = sample_deq_k()
            else:
                k = int(getattr(args, "deq_k_max", args.num_layers))
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        return int(k)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"run_id:{args.run_id}")
    log0(
        "config:"
        f" refinements={int(args.num_refinements)}"
        f" deq_k_jitter={int(bool(args.deq_k_jitter))}"
        f" deq_k_range={int(args.deq_k_min)}-{int(args.deq_k_max)}"
        f" deq_k_eval={int(args.deq_k_eval)}"
        f" compile_train={int(bool(getattr(args, 'compile_train', False)))}"
        " soft_topk=128"
        " moe=block"
        " router=softmax"
    )
    log0(f"router_bias_update:{int(bool(args.router_bias_update))} lr:{float(args.router_bias_lr):.4f} clip:{float(args.router_bias_clip):.2f}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # MODEL + OPTIMIZER SETUP
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim,
        num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank,
        mlp_expert_rank=args.mlp_expert_rank,
        deq_beta=args.deq_beta,
        attn_balance_mult=args.attn_balance_mult,
        mlp_balance_mult=args.mlp_balance_mult,
        bal_loss_coef=args.bal_loss_coef,
        mos_ortho_out_coef=args.mos_ortho_out_coef,
        attn_ortho_out_coef=args.attn_ortho_out_coef,
        mlp_ortho_out_coef=args.mlp_ortho_out_coef,
        deq_backward=args.deq_backward,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Optional training compile: compile only the shared block to avoid compiling the whole training step.
    if bool(getattr(args, "compile_train", False)):
        # Keep the eager module for diagnostics + attribute access; use compiled callable
        # only on non-diagnostic training steps.
        base_model._shared_block_compiled = torch.compile(base_model.shared_block, mode="reduce-overhead", fullgraph=False)
        base_model.compile_train = True
    else:
        base_model.compile_train = False
    compiled_model = base_model  # skip compile for training; use compiled forward_logits for eval
    model: nn.Module = (
        DDP(
            compiled_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if distributed
        else compiled_model
    )

    # RevDEQ: params come from shared_block
    block_named_params = list(base_model.shared_block.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim >= 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)

    # MoS head parameters: all go to Adam (A is 3D, not compatible with Muon)
    mos = base_model.mos_head
    # MoS head: all params go to Adam (3D A tensors not compatible with Muon)
    mos_params = [mos.A_shared, mos.A_ctp, mos.A_ntp, mos.B_denoise, mos.B_NTP,
                  mos.gate_ctp.weight, mos.gate_ctp.bias, mos.gate_ntp.weight, mos.gate_ntp.bias]
    scalar_params.extend(mos_params)

    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=0.04,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # DATA LOADER & MODEL WARMUP
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def compiler_step_begin() -> None:
        # When `--compile-train 1` is enabled, Inductor may use CUDA graphs.
        # Marking step boundaries prevents stale graph outputs from being overwritten
        # across invocations (common when modules keep small diagnostic tensors).
        if not bool(getattr(args, "compile_train", False)):
            return
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def format_deq_info(m: nn.Module) -> str:
        parts: list[str] = []
        if hasattr(m, "_deq_k_last"):
            parts.append(f"deq_k:{int(m._deq_k_last)}")
        if hasattr(m, "_deq_residuals") and getattr(m, "_deq_residuals"):
            parts.append(f"deq_residual:{m._deq_residuals[-1]:.6f}")
        recon = getattr(m, "_deq_recon_error", None)
        if recon is not None:
            # Reconstruction error can be extremely small; log in scientific notation to avoid
            # quantizing it to zero in logs/plots.
            parts.append(f"deq_recon_err:{float(recon):.3e}")
        if hasattr(m, "_deq_iter_convergence"):
            parts.append(f"deq_iter_conv:{m._deq_iter_convergence:.6f}")
        if hasattr(m, "_deq_iter_convergence_rel"):
            parts.append(f"deq_iter_conv_rel:{m._deq_iter_convergence_rel:.6f}")
        if hasattr(m, "shared_block") and getattr(m.shared_block, "_gg_last", None) is not None:
            parts.append(f"gg:{float(m.shared_block._gg_last):.4f}")
        gg_mean = getattr(m, "_gg_mean_last_solve", None)
        if gg_mean is not None:
            parts.append(f"gg_mean:{float(gg_mean):.4f}")
        gg_iter = getattr(m, "_gg_iter", None)
        if isinstance(gg_iter, list) and gg_iter:
            gg_str = ",".join(f"{float(v):.4f}" for v in gg_iter)
            parts.append(f"gg_iter:[{gg_str}]")
        return (" " + " ".join(parts)) if parts else ""

    def format_expert_info(
        m: nn.Module,
        *,
        include_gates: bool = False,
        step: int | None = None,
        require_step_match: bool = False,
    ) -> str:
        parts: list[str] = []

        if hasattr(m, "shared_block"):
            # Block-level router stats (shared across attention + MLP expert banks).
            router = getattr(m.shared_block.attn, "attn_router", None)
            diag_ok = (
                router is not None
                and getattr(router, "_expert_usage", None) is not None
                and (not require_step_match or getattr(router, "_diag_step", None) == step)
            )
            if diag_ok:
                usage_str = ",".join(f"{u:.3f}" for u in router._expert_usage)
                parts.append(f"expert_usage:[{usage_str}]")
                # Backward-compat with existing plotting code: expose the shared block-router
                # stats under both `mlp_*` and `attn_*` keys. Plotting will automatically
                # de-duplicate identical series for block-level MoE.
                parts.append(f"block_usage:[{usage_str}]")
                parts.append(f"mlp_usage:[{usage_str}]")
                parts.append(f"attn_usage:[{usage_str}]")
                ent = getattr(router, "_expert_entropy", None)
                if ent is not None:
                    parts.append(f"expert_entropy:{ent:.4f}")
                    parts.append(f"block_entropy:{ent:.4f}")
                    parts.append(f"mlp_entropy:{ent:.4f}")
                    parts.append(f"attn_entropy:{ent:.4f}")
                    spar = getattr(router, "_expert_sparsity", None)
                    if spar is not None:
                        parts.append(f"expert_sparsity:{float(spar):.4f}")
                cv = getattr(router, "_expert_balance_cv", None)
                if cv is not None:
                    parts.append(f"block_cv:{cv:.4f}")
                    parts.append(f"mlp_cv:{cv:.4f}")
                    parts.append(f"attn_cv:{cv:.4f}")
                if include_gates:
                    gm = getattr(router, "_gate_mass_mean", None)
                    if gm is not None:
                        parts.append(f"expert_gate_mass:{float(gm):.4f}")

            # Block-level expert output orthogonality (used by plotting code as expert_ortho).
            block_ortho = getattr(m.shared_block, "_block_ortho_cos_sim", None)
            if block_ortho is not None:
                parts.append(f"block_ortho:{float(block_ortho):.4f}")
                parts.append(f"expert_ortho:{float(block_ortho):.4f}")
        if hasattr(m, "mos_head"):
            mos = m.mos_head
            if hasattr(mos, "get_head_orthogonality"):
                parts.append(f"mos_ctp_ortho:{mos.get_head_orthogonality('ctp'):.4f}")
                parts.append(f"mos_ntp_ortho:{mos.get_head_orthogonality('ntp'):.4f}")
            mos_diag_ok = (not require_step_match) or (getattr(mos, "_diag_step", None) == step)
            for head_name in ["ctp", "ntp"]:
                usage = getattr(mos, f"_{head_name}_expert_usage", None)
                if usage is not None and mos_diag_ok:
                    usage_str = ",".join(f"{u:.3f}" for u in usage)
                    parts.append(f"mos_{head_name}_usage:[{usage_str}]")
                    ent = getattr(mos, f"_{head_name}_expert_entropy", None)
                    cv = getattr(mos, f"_{head_name}_expert_balance_cv", None)
                    if ent is not None:
                        parts.append(f"mos_{head_name}_entropy:{ent:.4f}")
                    if cv is not None:
                        parts.append(f"mos_{head_name}_cv:{cv:.4f}")

        return (" " + " ".join(parts)) if parts else ""

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            if args.deq_k_jitter:
                base_model._deq_k_override = deq_k_for_step(warmup_step + 1)
            else:
                base_model._deq_k_override = int(getattr(args, "deq_k_max", args.num_layers))
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                compiler_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            if args.router_bias_update:
                seen: set[int] = set()
                for r in [
                    base_model.shared_block.attn.attn_router,
                    base_model.shared_block.mlp.mlp_router,
                ]:
                    rid = id(r)
                    if rid in seen:
                        continue
                    seen.add(rid)
                    r.bias_update(
                        lr=float(args.router_bias_lr),
                        clip=float(args.router_bias_clip),
                        distributed=distributed,
                    )
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        # Reset the shuffle-bag after warmup so main training begins with fresh coverage cycles.
        k_sampler.reset()

    # MAIN TRAINING LOOP
    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                full_eval=False,
            )
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, include_gates=True, step=step) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"val_mode:fast "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                f"{deq_info}{expert_info}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        # Late QAT: enable fake quantization during last 15% of warmdown
        global _QAT_ACTIVE
        _QAT_ACTIVE = scale < 0.10
        zero_grad_all()
        # Clear stale RevDEQ backward reconstruction diagnostic; it is only meaningful for
        # the *current* optimizer step when computed by RevDEQFunction.backward.
        if hasattr(base_model, "shared_block") and hasattr(base_model.shared_block, "_deq_recon_error_last_bwd"):
            try:
                base_model.shared_block._deq_recon_error_last_bwd = None
            except Exception:
                pass
        train_loss = torch.zeros((), device=device)
        next_step = step + 1
        will_log_train = (
            args.train_log_every > 0
            and (next_step <= 10 or next_step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if args.deq_k_jitter:
            base_model._deq_k_override = deq_k_for_step(next_step)
        else:
            base_model._deq_k_override = int(getattr(args, "deq_k_max", args.num_layers))
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            compiler_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                diag_enabled = master_process and will_log_train and micro_step == grad_accum_steps - 1
                with router_diagnostics(diag_enabled, step_tag=next_step if diag_enabled else None):
                    loss = model(x, y)
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
            _preclip_grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm).item()
        else:
            _preclip_grad_norm = 0.0
        for opt in optimizers:
            opt.step()
        if args.router_bias_update:
            seen: set[int] = set()
            for r in [
                base_model.shared_block.attn.attn_router,
                base_model.shared_block.mlp.mlp_router,
            ]:
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                r.bias_update(
                    lr=float(args.router_bias_lr),
                    clip=float(args.router_bias_clip),
                    distributed=distributed,
                )
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA: collect checkpoints during warmdown
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            ntp = getattr(base_model, '_ntp_loss', 0.0)
            ctp = getattr(base_model, '_ctp_loss', 0.0)
            # Pull RevDEQ backward reconstruction diagnostic (if computed) onto the model so
            # the logging/plotting path stays uniform.
            base_model._deq_recon_error = getattr(base_model.shared_block, "_deq_recon_error_last_bwd", None)
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step, require_step_match=True) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ntp_loss:{ntp:.4f} ctp_loss:{ctp:.4f} "
                f"grad_norm:{_preclip_grad_norm:.4f} "
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

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Apply SWA if collected
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)

    if bool(getattr(args, "benchmark_mode", False)):
        log0("benchmark_mode:1 skipping_final_eval_and_serialization")
        if distributed:
            dist.destroy_process_group()
        return

    # SERIALIZATION + ROUNDTRIP VALIDATION
    # Weights go to experiments/weights/current/ (rotated by update_results.sh)
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "weights", "current")
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "final_model.pt")
    quant_path = os.path.join(weights_dir, "final_model.int6.ptz")

    if master_process:
        torch.save(base_model.state_dict(), model_path)
        model_bytes = os.path.getsize(model_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # INT6 mixed quantization + compression
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn", "bigram"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _COMPRESSOR == "zstd":
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, 9)
    if master_process:
        with open(quant_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(quant_path, "rb") as f:
        quant_blob_disk = f.read()
    if _COMPRESSOR == "zstd":
        decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
    else:
        decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    base_model.load_state_dict(deq_state, strict=True)

    # Sliding window eval on int6-roundtripped weights
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        log0(f"final_eval_mode:sliding_window stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.eval_batch_seqs,
        )
    else:
        log0("final_eval_mode:standard")
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            full_eval=True,
        )
    torch.cuda.synchronize()
    log0(
        f"final_int6_{_COMPRESSOR}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_{_COMPRESSOR}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Save metadata for weight tracking
    if master_process:
        import json
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
        meta = {
            "git_commit": git_hash,
            "val_bpb": round(q_val_bpb, 8),
            "val_loss": round(q_val_loss, 8),
            "artifact_bytes": quant_file_bytes + code_bytes,
            "quant_bytes": quant_file_bytes,
            "steps": step,
            "train_time_s": round(approx_training_time_ms / 1000, 1),
            "params": sum(p.numel() for p in base_model.parameters()),
        }
        meta_path = os.path.join(weights_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log0(f"Saved weight metadata to {meta_path}")

    # Auto-update comparison plots at the end of a run (rank0 only).
    # IMPORTANT: run this only after post-quant eval is logged so the plot can
    # show the scored metric (final_int6_*_roundtrip_exact).
    if master_process:
        try:
            exp_logdir = Path("experiments/training_logs")
            exp_logdir.mkdir(parents=True, exist_ok=True)
            if logfile is not None and Path(logfile).exists():
                # Copy rather than rename so `logs/<run_id>.txt` remains the canonical run log.
                shutil.copyfile(logfile, exp_logdir / "current.log")
            # Always keep plots up to date with the most recent run. If this is the first
            # run in a fresh workspace, initialize baseline from current so plots render.
            if (exp_logdir / "current.log").exists() and not (exp_logdir / "baseline.log").exists():
                shutil.copyfile(exp_logdir / "current.log", exp_logdir / "baseline.log")
            if (exp_logdir / "baseline.log").exists() and (exp_logdir / "current.log").exists():
                subprocess.run([sys.executable, "experiments/plot_metrics.py"], check=False)
                subprocess.run([sys.executable, "experiments/plot_eval_metrics.py"], check=False)
        except Exception as e:
            log0(f"auto_plot_failed:{type(e).__name__}:{e}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
# fixes applied
# tuned
