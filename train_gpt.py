"""
Simplified RevDEQ training script for Parameter Golf.
Single-file, self-contained submission artifact.

Architecture: RevDEQ + Soft Dense MoE + MLA + Gated Attention + FSQ/MoS + Diffusion-AR
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import glob
import io
import argparse
import math
import os
import random
import signal
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
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# DIAGNOSTICS CONTROL
# ---------------------------------------------------------------------------
_ROUTER_DIAGNOSTICS_ACTIVE = False
_ROUTER_DIAGNOSTICS_STEP: int | None = None


def dynamo_disable(fn):
    try:
        import torch._dynamo as dynamo
        return dynamo.disable(fn)
    except Exception:
        return fn


def _should_diag(training: bool) -> bool:
    """Return True if this rank should record diagnostics right now."""
    if training and not _ROUTER_DIAGNOSTICS_ACTIVE:
        return False
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
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
    val_loss_every = 200
    train_log_every = 100
    auto_plot_on_val = True

    iterations = 1_000_000_000  # governed by wallclock
    warmdown_frac = 0.72  # fraction of total steps for warmdown
    warmup_steps = 0
    train_batch_tokens = 524_288
    train_seq_len = 2048
    max_wallclock_seconds = 3600.0  # 2xL40S dev (1h autoresearch budget); set 600 for 8xH100

    # Model architecture
    vocab_size = 1024
    num_layers = 12  # DEQ solver max K
    num_refinements = 1
    num_refinements_ramp_frac = 0.85  # enable refinement after 85% of wallclock
    num_kv_heads = 4
    model_dim = 512
    num_heads = 8
    mlp_mult = 3.0
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0
    qk_gain_init = 5.0
    deq_beta = 0.20  # iter 0: tighter solver contraction (was 0.35, but baseline's iter_conv_rel drifted 0.002->0.5 over 3k steps)

    # Optimizer
    tied_embed_lr = 0.03
    embed_lr = 0.6
    matrix_lr = 0.022
    scalar_lr = 0.02
    router_lr = 0.005
    muon_momentum = 0.99
    muon_backend_steps = 5
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 800
    beta1 = 0.85
    beta2 = 0.90
    adam_eps = 1e-8
    grad_clip_norm = 0.3
    weight_decay = 0.06
    tied_embed_init_std = 0.005

    # Routing
    attn_balance_mult = 5.0
    mlp_balance_mult = 1.0
    bal_loss_coef = 5e-3
    router_health_coef = 0.25
    block_ortho_aux_coef = 1.0
    block_ortho_aux_every = 4
    block_ortho_aux_tokens = 64
    router_bias_update = True
    router_bias_lr = 0.10
    router_bias_clip = 10.0
    mos_ortho_out_coef = 1e-3
    tie_attn_mlp_router = True

    # DEQ solver
    deq_backward = "revdeq"
    deq_k_jitter = True
    deq_k_min = 4
    deq_k_max = 8
    deq_k_step = 4  # K in {4, 8}
    deq_k_eval = 8

    # Architecture knobs
    bigram_vocab_size = 65536
    bigram_dim = 208
    kv_latent_dim = 0  # auto: dim//2
    # iter 4: halved expert_rank to keep total rank-units constant when
    # doubling num_experts from 8 -> 16 in Block.  Test whether more-but-
    # smaller experts beats fewer-but-larger experts at constant compute.
    #   8 x 128 = 1024 attn rank-units -> 16 x 64 = 1024  (identical)
    #   8 x 192 = 1536 mlp rank-units  -> 16 x 96 = 1536  (identical)
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
    eval_batch_seqs = 32


def _parse_cli_overrides(argv: list[str]) -> dict[str, object]:
    p = argparse.ArgumentParser(add_help=True)
    for name in [
        "data-path", "tokenizer-path", "run-id", "seed", "iterations",
        "warmup-steps", "train-batch-tokens", "train-seq-len",
        "val-batch-size", "val-loss-every", "train-log-every",
        "max-wallclock-seconds", "attn-balance-mult", "mlp-balance-mult",
        "bal-loss-coef", "router-health-coef", "router-lr",
        "mos-ortho-out-coef", "block-ortho-aux-coef", "block-ortho-aux-every",
        "block-ortho-aux-tokens", "bigram-vocab-size", "bigram-dim",
        "kv-latent-dim", "attn-expert-rank", "mlp-expert-rank",
        "swa-start-frac", "swa-every", "ema-decay", "ema-update-every",
        "deq-k-min", "deq-k-max", "deq-k-step", "deq-k-eval",
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
        "tie-attn-mlp-router", "swa-enabled", "ema-enabled",
    ]:
        p.add_argument(f"--{name}", type=int, default=None, help="1/0")
    p.add_argument("--deq-backward", type=str, default=None, choices=["autograd", "revdeq"])
    p.add_argument("--router-bias-lr", type=float, default=None)
    p.add_argument("--router-bias-clip", type=float, default=None)
    ns, unknown = p.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"Unknown args: {unknown}")
    out: dict[str, object] = {}
    bool_keys = {"auto_plot_on_val", "router_bias_update", "deq_k_jitter",
                 "tie_attn_mlp_router", "swa_enabled", "ema_enabled"}
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))

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
                    g = g.view(-1, g.shape[-1])
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
                   *, full_validation: bool) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // world_size
    local_batch_seqs = local_batch_tokens // args.train_seq_len
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
    model.eval()
    base_m = model.module if hasattr(model, "module") else model
    prev_k = getattr(base_m, "_deq_k_override", None)
    base_m._deq_k_override = int(getattr(args, "deq_k_eval", base_m.num_layers))
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


# Backwards-compatible alias used by smoke_test.py and other callers
eval_val = run_validation


# ---------------------------------------------------------------------------
# QUANTIZATION (uniform INT6 + SDClip)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_PATTERNS = ("gg_w", "gg_b", "q_gain", "gate_bias", "bigram.scale")
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


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
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
        denom = self.world_size * grad_accum_steps
        global_seqs = global_tokens // seq_len
        local_seqs = global_seqs // denom
        if local_seqs < 1:
            raise ValueError(f"TRAIN_BATCH_TOKENS too small: {global_tokens}")
        local_tokens = local_seqs * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# TRANSFORMER MODULES
# ---------------------------------------------------------------------------

def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.eps)


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
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
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


class KShuffleBagSampler:
    def __init__(self, k_min: int, k_max: int, rng: random.Random, *, step: int = 1):
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.step = int(step)
        self.rng = rng
        self._bag: list[int] = []

    def sample(self) -> int:
        if not self._bag:
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
            h = self.proj(_rms_norm(h))
        return h * self.scale.to(dtype=h.dtype)


# ---------------------------------------------------------------------------
# SOFT DENSE ROUTER
# ---------------------------------------------------------------------------

class SoftDenseRouter(nn.Module):
    """Dense softmax routing over experts (no top-k, no dropping)."""
    def __init__(self, dim: int, num_experts: int, *,
                 min_share_frac: float = 0.6, cv_target: float = 0.20,
                 min_share_loss_weight: float = 1.0, cv_loss_weight: float = 0.10):
        super().__init__()
        self.num_experts = num_experts
        self.min_share_frac = float(min_share_frac)
        self.cv_target = float(cv_target)
        self.min_share_loss_weight = float(min_share_loss_weight)
        self.cv_loss_weight = float(cv_loss_weight)
        self.health_scale = 1.0
        self.router = CastedLinear(dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        self._mean_share_last: Tensor | None = None
        self._balance_loss = None
        self._health_loss = None
        self._expert_usage = None
        self._expert_entropy = None
        self._expert_sparsity = None
        self._expert_balance_cv = None
        self._diag_step: int | None = None

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
        target = torch.full_like(ms, 1.0 / float(self.num_experts))
        lb = float(self.min_share_frac) / float(self.num_experts)
        if lb > 0.0:
            boost = (lb - ms).clamp_min(0.0)
            if float(boost.sum().item()) > 0.0:
                target = (target + boost).clamp_min(1e-8)
                target = target / target.sum()
        self.expert_bias.add_(lr * (target - ms))
        if clip > 0:
            self.expert_bias.clamp_(min=-clip, max=clip)

    def forward(self, x: Tensor) -> Tensor:
        x_n = _rms_norm(x)
        route_logits = self.router(x_n) + self.expert_bias.to(dtype=x.dtype)
        p = torch.softmax(route_logits, dim=-1)
        if self.training:
            reduce_dims = tuple(range(p.ndim - 1))
            mean_share = p.mean(dim=reduce_dims)
            target = torch.ones_like(mean_share) / self.num_experts
            mse = F.mse_loss(mean_share, target)
            lb = float(self.min_share_frac) / float(self.num_experts)
            min_share_loss = torch.relu(mean_share.new_tensor(lb) - mean_share).pow(2).mean() if lb > 0.0 else mean_share.new_zeros(())
            cv = mean_share.std() / mean_share.mean().clamp_min(1e-8)
            cv_loss = torch.relu(cv - mean_share.new_tensor(self.cv_target)).pow(2) if self.cv_target > 0.0 else mean_share.new_zeros(())
            hs = float(self.health_scale)
            self._balance_loss = mse
            self._health_loss = float(self.min_share_loss_weight) * hs * min_share_loss + float(self.cv_loss_weight) * hs * cv_loss
            self._mean_share_last = mean_share.detach()
            with torch.no_grad():
                if _should_diag(self.training):
                    self._record_diagnostics(p.detach(), reduce_dims)
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._health_loss = torch.tensor(0.0, device=x.device)
            self._mean_share_last = None
            with torch.no_grad():
                if _should_diag(self.training):
                    self._record_diagnostics(p.detach(), tuple(range(p.ndim - 1)))
                else:
                    self._expert_usage = None
                    self._expert_entropy = None
                    self._expert_sparsity = None
                    self._expert_balance_cv = None
                    self._diag_step = None
        return p

    @dynamo_disable
    def _record_diagnostics(self, share: Tensor, reduce_dims: tuple[int, ...]) -> None:
        mean_mass = share.mean(dim=reduce_dims)
        self._expert_usage = mean_mass.float().cpu().tolist()
        per_token_ent = -(share * (share + 1e-8).log()).sum(-1)
        ent = float(per_token_ent.mean().item())
        self._expert_entropy = ent
        self._expert_sparsity = 1.0 - (ent / max(math.log(float(self.num_experts)), 1e-8))
        self._expert_balance_cv = float((mean_mass.std() / mean_mass.mean().clamp_min(1e-8)).item())
        self._diag_step = _ROUTER_DIAGNOSTICS_STEP


# ---------------------------------------------------------------------------
# MLA + GATED ATTENTION
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """MLA with Gated Attention + expert bank."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, kv_latent_dim: int = 0, num_experts: int = 6,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(dim // max(num_experts, 1), 1)
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else dim // 2
        self.rope_dim = self.head_dim // 2
        self.nope_dim = self.head_dim - self.rope_dim

        self.c_q = CastedLinear(dim, dim + num_heads, bias=False)
        with torch.no_grad():
            self.c_q.weight[dim:, :].zero_()
        self.c_kv_down = CastedLinear(dim, self.kv_latent_dim, bias=False)
        self.c_k_nope = CastedLinear(self.kv_latent_dim, num_kv_heads * self.nope_dim, bias=False)
        self.c_v = CastedLinear(self.kv_latent_dim, num_kv_heads * self.head_dim, bias=False)
        self.c_k_rope = CastedLinear(dim, num_kv_heads * self.rope_dim, bias=False)
        self.expert_proj = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_out = nn.Parameter(torch.empty(num_experts, dim, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_proj.data[e])
            nn.init.xavier_uniform_(self.expert_out.data[e])
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dim, base=rope_base)
        # Gate bias init at 0 (mid-point sigmoid) per EXPERIENCE.md
        self.gate_bias = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.attn_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None

    def _attn_shared_from_normed(self, x_n: Tensor) -> Tensor:
        bsz, seqlen, dim = x_n.shape
        q_and_gate = self.c_q(x_n)
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

    def mix_experts_from_shared(self, y: Tensor, w: Tensor) -> Tensor:
        B, T, D = y.shape
        E, R = self.num_experts, self.expert_rank
        y_flat = y.reshape(B * T, D)
        w_flat = w.reshape(B * T, E).to(dtype=y_flat.dtype)
        P = self.expert_proj.to(dtype=y_flat.dtype).reshape(E * R, D)
        O = self.expert_out.to(dtype=y_flat.dtype).permute(0, 2, 1).reshape(E * R, D)
        h = y_flat @ P.t()
        h = h * w_flat.repeat_interleave(R, dim=1)
        out = h @ O

        if _should_diag(self.training):
            with torch.no_grad():
                mu_h = h.reshape(B * T, E, R).mean(dim=0).to(dtype=torch.float32)
                out_T = self.expert_out.to(dtype=mu_h.dtype).permute(0, 2, 1)
                mu_out = torch.einsum("er,erd->ed", mu_h, out_T)
                self._out_ortho_cos_sim = float(mean_abs_offdiag_cosine(mu_out).item())

        return out.reshape(B, T, D)

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
# MLP with LeakyReLU(0.5)^2
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """LeakyReLU(0.5)^2-gated MLP expert bank."""
    def __init__(self, dim: int, mlp_mult: float, num_experts: int = 6,
                 expert_rank: int = 0, router: SoftDenseRouter | None = None):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.num_experts = num_experts
        self.expert_rank = expert_rank if expert_rank > 0 else max(hidden // max(num_experts, 1), 1)
        self.expert_gate = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_fc = nn.Parameter(torch.empty(num_experts, self.expert_rank, dim))
        self.expert_down = nn.Parameter(torch.empty(num_experts, dim, self.expert_rank))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_gate.data[e])
            nn.init.xavier_uniform_(self.expert_fc.data[e])
            nn.init.xavier_uniform_(self.expert_down.data[e])
        self.mlp_router = router if router is not None else SoftDenseRouter(dim, num_experts)
        self._out_ortho_cos_sim: float | None = None
        self._out_ortho_loss: Tensor | None = None

    def mix_experts(self, x: Tensor, w: Tensor) -> Tensor:
        B, T, D = x.shape
        E, R = self.num_experts, self.expert_rank
        x_n = _rms_norm(x)
        N = B * T
        x_flat = x_n.reshape(N, D)
        w_flat = w.reshape(N, E).to(dtype=x_flat.dtype)
        G = self.expert_gate.to(dtype=x_flat.dtype).reshape(E * R, D)
        Fm = self.expert_fc.to(dtype=x_flat.dtype).reshape(E * R, D)
        Dwn = self.expert_down.to(dtype=x_flat.dtype).permute(0, 2, 1).reshape(E * R, D)
        gate = x_flat @ G.t()
        fc = x_flat @ Fm.t()
        h = F.leaky_relu(gate, negative_slope=0.5).square() * fc
        h = h * w_flat.repeat_interleave(R, dim=1)
        out = h @ Dwn

        if _should_diag(self.training):
            with torch.no_grad():
                mu_h = h.reshape(N, E, R).mean(dim=0).to(dtype=torch.float32)
                down_T = self.expert_down.to(dtype=mu_h.dtype).permute(0, 2, 1)
                mu_out = torch.einsum("er,erd->ed", mu_h, down_T)
                self._out_ortho_cos_sim = float(mean_abs_offdiag_cosine(mu_out).item())

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
    """Mixture-of-Softmaxes dual head with FSQ."""
    def __init__(self, d_model: int, vocab_size: int, rank: int = 256,
                 num_shared: int = 2, num_specialized: int = 1, fsq_levels: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.num_shared = num_shared
        self.num_specialized = num_specialized
        self.num_experts = num_shared + num_specialized
        self.fsq_levels = fsq_levels
        self.gate_ctp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
        self.gate_ntp = nn.Linear(d_model, num_shared + num_specialized, bias=True)
        self.A_shared = nn.Parameter(torch.empty(num_shared, d_model, rank))
        self.A_ctp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        self.A_ntp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
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
        pass  # No SVD init; xavier from scratch

    def get_head_orthogonality(self, head: str) -> float:
        t = self._ctp_ortho_out if head == "ctp" else self._ntp_ortho_out
        if t is not None:
            return float(t.detach().float().item())
        if self.num_shared + self.num_specialized < 2:
            return 0.0
        A_spec = self.A_ctp if head == "ctp" else self.A_ntp
        with torch.no_grad():
            groups = torch.cat([self.A_shared, A_spec], dim=0).float().reshape(self.num_shared + self.num_specialized, -1)
            return float(max_mean_abs_offdiag_cosine(groups).item())

    def _fsq(self, x: Tensor) -> Tensor:
        return _fsq_ste(x, self.fsq_levels, self.training)

    def _head_forward(self, x: Tensor, gate: nn.Linear, A_shared: Tensor,
                      A_spec: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N = x.shape[0]
        x = _rms_norm(x)
        alpha = F.softmax(gate(x).float(), dim=-1)
        log_w = alpha.clamp(min=1e-8).log()
        log_p_unnorm = x.new_full((N, self.vocab_size), -torch.inf, dtype=torch.float32)
        mu_groups: list[Tensor] = []
        for e in range(self.num_shared):
            t = x.to(A_shared.dtype) @ A_shared[e]
            mu_groups.append(t.float().mean(dim=0))
            u = self._fsq(t)
            logits = u.to(B.dtype) @ B.t()
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, e:e+1] + F.log_softmax(logits.float(), dim=-1))
        for e in range(self.num_specialized):
            t = x.to(A_spec.dtype) @ A_spec[e]
            mu_groups.append(t.float().mean(dim=0))
            u = self._fsq(t)
            logits = u.to(B.dtype) @ B.t()
            idx = self.num_shared + e
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, idx:idx+1] + F.log_softmax(logits.float(), dim=-1))
        log_p = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=-1, keepdim=True)
        ortho_out = max_mean_abs_offdiag_cosine(torch.stack(mu_groups, dim=0)) if len(mu_groups) >= 2 else x.new_zeros(())
        return log_p, alpha, ortho_out

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
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
            with torch.no_grad():
                if _should_diag(self.training):
                    for alpha_soft, ua, ea, ca in [
                        (alpha_d, '_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        (alpha_n, '_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        a = alpha_soft.detach()
                        mean_a = a.mean(dim=0)
                        setattr(self, ua, mean_a.float().cpu().tolist())
                        per_token_ent = -(a * (a + 1e-8).log()).sum(-1)
                        setattr(self, ea, per_token_ent.mean().item())
                        setattr(self, ca, (mean_a.std() / mean_a.mean().clamp_min(1e-8)).item())
                    self._diag_step = _ROUTER_DIAGNOSTICS_STEP
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            with torch.no_grad():
                if _should_diag(self.training):
                    for alpha_soft, ua, ea, ca in [
                        (alpha_d, '_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        (alpha_n, '_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        mean_a = alpha_soft.mean(dim=0)
                        setattr(self, ua, mean_a.float().cpu().tolist())
                        per_token_ent = -(alpha_soft * (alpha_soft + 1e-8).log()).sum(-1)
                        setattr(self, ea, per_token_ent.mean().item())
                        setattr(self, ca, (mean_a.std() / mean_a.mean()).item())
                    self._diag_step = _ROUTER_DIAGNOSTICS_STEP
                else:
                    for ua, ea, ca in [
                        ('_ctp_expert_usage', '_ctp_expert_entropy', '_ctp_expert_balance_cv'),
                        ('_ntp_expert_usage', '_ntp_expert_entropy', '_ntp_expert_balance_cv'),
                    ]:
                        setattr(self, ua, None)
                        setattr(self, ea, None)
                        setattr(self, ca, None)
                    self._diag_step = None
        return log_p_d.view(*orig_shape, -1), log_p_n.view(*orig_shape, -1)


# ---------------------------------------------------------------------------
# BLOCK (Attention + MLP)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    # iter 4: doubled num_experts again (8 -> 16) at constant rank-units.
    # The Hyperparameters defaults halve attn_expert_rank (128 -> 64) and
    # mlp_expert_rank (192 -> 96), so total rank-units (16 x 64 = 1024;
    # 16 x 96 = 1536) and step throughput stay identical to iter 3 a3.
    # Tests whether more-but-smaller experts beats fewer-but-larger at
    # the same total compute.
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, kv_latent_dim: int = 0,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 tie_attn_mlp_router: bool = False, num_experts: int = 16):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if bool(tie_attn_mlp_router):
            shared = SoftDenseRouter(dim, num_experts, min_share_loss_weight=10.0, cv_loss_weight=2.0)
            self.attn_router = shared
            self.mlp_router = shared
        else:
            self.attn_router = SoftDenseRouter(dim, num_experts, min_share_loss_weight=10.0, cv_loss_weight=2.0)
            self.mlp_router = SoftDenseRouter(dim, num_experts, min_share_loss_weight=5.0, cv_loss_weight=1.0)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         kv_latent_dim=kv_latent_dim, num_experts=num_experts,
                                         expert_rank=attn_expert_rank, router=self.attn_router)
        self.mlp = MLP(dim, mlp_mult, num_experts=num_experts, expert_rank=mlp_expert_rank, router=self.mlp_router)
        self.gg_gate = CastedLinear(dim, 1, bias=True)
        with torch.no_grad():
            self.gg_gate.weight.zero_()
            # iter 2: initialize gg_gate.bias = 1.5 so init gg_tok = sigmoid(1.5)
            # ~ 0.82.  Combined with deq_beta=0.20, the per-iteration update at
            # init is 0.20 * 0.82 = 0.164 (~3.7x larger than iter 1's converged
            # 0.044), so K matters: K=4 reaches ~49% of the fixed point, K=8
            # reaches ~76%, K=16 reaches ~94%.  In iter 1 the model learned
            # gg_tok ~ 0.22 and the K-sweep was flat (k4=k8=k16=1.6209) because
            # the solver converged in <4 iters and the iterative depth was
            # wasted.  Starting high lets the optimizer choose whether to use
            # the depth -- if it lowers gg_tok again, we know the model is
            # actively choosing shallow; if it stays high, we get real depth.
            self.gg_gate.bias.fill_(1.5)
        self._gg_last: float | None = None
        self._gg_track_enabled = False
        self._gg_sum = 0.0
        self._gg_count = 0
        self._gg_call_track_enabled = False
        self._gg_call_track: list[float] = []

        self.inj_gate = CastedLinear(dim, 1, bias=True)
        self._inj_gate_last_mean: float | None = None
        with torch.no_grad():
            self.inj_gate.weight.zero_()
            self.inj_gate.bias.fill_(-2.1972246)  # sigmoid(-2.2) ~ 0.1

    def _inj_gate_from(self, z_in: Tensor) -> Tensor:
        z_n = _rms_norm(z_in)
        u = z_n.mean(dim=(0, 1), keepdim=True)
        g = torch.sigmoid(self.inj_gate(u)).squeeze(-1)
        if _should_diag(self.training):
            self._inj_gate_last_mean = float(g.detach().float().mean().item())
        return g

    @dynamo_disable
    def _record_gg_diag(self, gg_tok: Tensor) -> None:
        gg_val = float(gg_tok.float().mean().item())
        self._gg_last = gg_val
        if self._gg_track_enabled:
            self._gg_sum += gg_val
            self._gg_count += 1
        if self._gg_call_track_enabled:
            self._gg_call_track.append(gg_val)

    def ortho_aux(self, z_in: Tensor, x0: Tensor, *, max_tokens: int = 256) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = z_in.shape
        t = int(min(max(1, int(max_tokens)), seqlen))
        z_sub = z_in[:, :t]
        x0_sub = x0[:, :t]
        g_inj = self._inj_gate_from(z_sub).to(dtype=z_sub.dtype)
        x = z_sub + g_inj * (x0_sub - z_sub)

        x_attn = self.attn_norm(x)
        x_attn_n = _rms_norm(x_attn)
        w_attn = self.attn_router(x_attn)
        y_shared = self.attn._attn_shared_from_normed(x_attn_n)
        E, R = self.attn.num_experts, self.attn.expert_rank
        y_flat = y_shared.reshape(bsz * t, dim)
        P = self.attn.expert_proj.to(dtype=y_flat.dtype).reshape(E * R, dim)
        h = y_flat @ P.t()
        mu_h = h.reshape(bsz * t, E, R).mean(dim=0).to(dtype=torch.float32)
        out_T = self.attn.expert_out.to(dtype=mu_h.dtype).permute(0, 2, 1)
        mu_attn = torch.einsum("er,erd->ed", mu_h, out_T)
        attn_ortho = mean_abs_offdiag_cosine(mu_attn)

        attn_mix = self.attn.mix_experts_from_shared(y_shared, w_attn)
        # Parallel residuals without inner residual (matches forward()): the
        # MLP reads x, not x + attn_mix.  Ortho diagnostic stays aligned with
        # the z2 = attn + mlp form in forward().
        x_mlp = self.mlp_norm(x)
        w_mlp = self.mlp_router(x_mlp)
        x_mlp_n = _rms_norm(x_mlp)
        N = bsz * t
        x_flat = x_mlp_n.reshape(N, dim)
        E2, R2 = self.mlp.num_experts, self.mlp.expert_rank
        G = self.mlp.expert_gate.to(dtype=x_flat.dtype).reshape(E2 * R2, dim)
        Fm = self.mlp.expert_fc.to(dtype=x_flat.dtype).reshape(E2 * R2, dim)
        gate = x_flat @ G.t()
        fc = x_flat @ Fm.t()
        h_mlp = F.leaky_relu(gate, negative_slope=0.5).square() * fc
        mu_h2 = h_mlp.reshape(N, E2, R2).mean(dim=0).to(dtype=torch.float32)
        down_T = self.mlp.expert_down.to(dtype=mu_h2.dtype).permute(0, 2, 1)
        mu_mlp = torch.einsum("er,erd->ed", mu_h2, down_T)
        mlp_ortho = mean_abs_offdiag_cosine(mu_mlp)

        return attn_ortho, mlp_ortho

    def forward(self, z_in: Tensor, x0: Tensor) -> Tensor:
        g_inj = self._inj_gate_from(z_in).to(dtype=z_in.dtype)
        x = z_in + g_inj * (x0 - z_in)

        # Parallel residuals with the inner residual REMOVED.  Attention and
        # MLP both read the same pre-residual input x and their outputs sum
        # directly, without the `x +` add that the sequential and earlier
        # parallel-residual attempts used.  The residual path is supplied
        # entirely by the outer gg_gate below:
        #   out = (1 - gg_tok) * z_in + gg_tok * (attn + mlp)
        # Early training gg_tok is small so the transformation contribution
        # is automatically gate-scaled, giving the solver a wide contraction
        # margin; the optimizer can learn the gate to increase transformation
        # weight as training progresses, without the ~2x update-magnitude
        # blow-up that broke attempts 1 and 2.
        x_attn = self.attn_norm(x)
        x_attn_n = _rms_norm(x_attn)
        w_attn = self.attn_router(x_attn)
        y_shared = self.attn._attn_shared_from_normed(x_attn_n)
        attn_mix = self.attn.mix_experts_from_shared(y_shared, w_attn)

        x_mlp = self.mlp_norm(x)
        w_mlp = self.mlp_router(x_mlp)
        mlp_mix = self.mlp.mix_experts(x_mlp, w_mlp)

        z2 = attn_mix + mlp_mix

        gg_tok = torch.sigmoid(self.gg_gate(x_attn_n)).squeeze(-1)
        if self._gg_track_enabled or self._gg_call_track_enabled:
            self._record_gg_diag(gg_tok.detach())
        return (1.0 - gg_tok).to(dtype=z_in.dtype).unsqueeze(-1) * z_in + gg_tok.to(dtype=z_in.dtype).unsqueeze(-1) * z2


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
    def forward(ctx, f_theta, x0, z_init, beta, K, *params):
        acc_dtype = torch.float64
        state_dtype = torch.float32
        compute_dtype = z_init.dtype
        device_type = x0.device.type
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
                    out_z = f_theta(z_state.to(compute_dtype), x0)
                y_acc = y_acc + out_z.to(acc_dtype) * beta
                y_state = y_acc.to(state_dtype)

                z_acc = z_state.to(acc_dtype) * beta_inv
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_state.to(compute_dtype), x0)
                z_acc = z_acc + out_y.to(acc_dtype) * beta
                z_state = z_acc.to(state_dtype)

            if out_y is not None:
                try:
                    setattr(f_theta, "_deq_residual_proxy", float((z_state - out_y.to(state_dtype)).norm().item()))
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
        x0, y_terminal, z_terminal, _z_prev = (t.detach() for t in ctx.saved_tensors)
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

        y_next64 = y_terminal.to(acc_dtype)
        z_next64 = z_terminal.to(acc_dtype)

        for _ in range(K):
            y_local = y_next64.detach().to(compute_dtype).requires_grad_()
            x_local = x0.detach().to(x0.dtype).requires_grad_()
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_y = f_theta(y_local, x_local)
            z_n64 = (z_next64 - out_y.detach().to(acc_dtype) * beta) / beta_inv
            grad_seed_y = (beta * bar_z).to(out_y.dtype)
            grads_y = torch.autograd.grad(out_y, (y_local, x_local, *params_req),
                                          grad_outputs=grad_seed_y, allow_unused=True)
            vjp_y = grads_y[0].to(state_dtype)
            bar_y_acc = bar_y + vjp_y

            z_local = z_n64.detach().to(compute_dtype).requires_grad_()
            x_local2 = x0.detach().to(x0.dtype).requires_grad_()
            with torch.enable_grad():
                with RevDEQFunction._autocast_like_ctx(device_type, compute_dtype):
                    out_z = f_theta(z_local, x_local2)
            y_n64 = (y_next64 - out_z.detach().to(acc_dtype) * beta) / beta_inv
            grad_seed_z = (beta * bar_y_acc).to(out_z.dtype)
            grads_z = torch.autograd.grad(out_z, (z_local, x_local2, *params_req),
                                          grad_outputs=grad_seed_z, allow_unused=True)
            vjp_z = grads_z[0].to(state_dtype)

            bar_z = beta_inv * bar_z + vjp_z
            bar_y = beta_inv * bar_y_acc

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

            y_next64, z_next64 = y_n64, z_n64

        if bool(getattr(ctx, "do_recon_diag", False)) and isinstance(z_init_state, torch.Tensor):
            try:
                z0 = z_init_state.to(dtype=state_dtype)
                denom = max(float(z0.norm().item()), 1.0)
                z_rec = z_next64.to(dtype=state_dtype)
                y_rec = y_next64.to(dtype=state_dtype)
                recon_err = float(((z_rec - z0).norm().item() + (y_rec - z0).norm().item()) / denom)
                setattr(f_theta, "_deq_recon_error_last_bwd", recon_err)
            except Exception:
                pass

        z_init_grad = (bar_y + bar_z).to(x0.dtype)
        param_grads_out: list[torch.Tensor | None] = [None] * len(params_all)
        for j, all_idx in enumerate(req_indices):
            g = cur_param_grads_req[j]
            if g is None:
                continue
            param_grads_out[all_idx] = g.to(dtype=params_all[all_idx].dtype)
        return (None, cur_x_grad.to(x0.dtype), z_init_grad, None, None, *param_grads_out)


# ---------------------------------------------------------------------------
# GPT MODEL
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: float, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float, rope_base: float,
                 qk_gain_init: float, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 kv_latent_dim: int = 0, num_refinements: int = 1,
                 attn_expert_rank: int = 0, mlp_expert_rank: int = 0,
                 deq_beta: float = 0.35, attn_balance_mult: float = 5.0,
                 mlp_balance_mult: float = 1.0, bal_loss_coef: float = 5e-3,
                 router_health_coef: float = 0.25, mos_ortho_out_coef: float = 0.0,
                 attn_ortho_out_coef: float = 0.0, mlp_ortho_out_coef: float = 0.0,
                 deq_backward: str = "revdeq", block_ortho_aux_coef: float = 0.0,
                 block_ortho_aux_every: int = 0, block_ortho_aux_tokens: int = 64,
                 tie_attn_mlp_router: bool = True):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_refinements = num_refinements
        self.blocks = None  # Required by arch tests
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = nn.Identity()  # Required by arch tests
        self.shared_block = Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                                   rope_base, qk_gain_init, kv_latent_dim=kv_latent_dim,
                                   attn_expert_rank=attn_expert_rank, mlp_expert_rank=mlp_expert_rank,
                                   tie_attn_mlp_router=tie_attn_mlp_router)
        self.deq_beta = float(deq_beta)
        self.attn_balance_mult = float(attn_balance_mult)
        self.mlp_balance_mult = float(mlp_balance_mult)
        self.bal_loss_coef = float(bal_loss_coef)
        self.router_health_coef = float(router_health_coef)
        self.mos_ortho_out_coef = float(mos_ortho_out_coef)
        self.deq_backward = deq_backward
        self.block_ortho_aux_coef = float(block_ortho_aux_coef)
        self.block_ortho_aux_every = int(block_ortho_aux_every)
        self.block_ortho_aux_tokens = int(block_ortho_aux_tokens)
        self._block_ortho_aux_enabled = False
        self._block_ortho_aux_loss: Tensor | None = None
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
        with torch.no_grad():
            dim = self.tok_emb.embedding_dim
            self.shared_block.attn.c_q.weight[dim:, :].zero_()
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
            p_ctp = _logp_to_prob(log_p_ctp)
            p_ntp = _logp_to_prob(log_p_ntp_shifted)
            p_mix = 0.5 * (p_ctp + p_ntp)
            p_mix[:, 0] = p_ctp[:, 0]
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
            soft_embed = _rms_norm(soft_embed.to(dtype=z.dtype))
        return soft_embed

    def _deq_solve(self, x0: Tensor, z_init: Tensor):
        beta = self.deq_beta
        dtype = x0.dtype
        K = int(getattr(self, "_deq_k_override", 0) or self.num_layers)
        track_gg = _should_diag(self.training)
        self.shared_block._gg_sum = 0.0
        self.shared_block._gg_count = 0
        self.shared_block._gg_track_enabled = bool(track_gg)
        self.shared_block._gg_call_track = []
        self.shared_block._gg_call_track_enabled = bool(track_gg)
        try:
            f_theta = self.shared_block
            if self.training and self.deq_backward == "revdeq":
                params = tuple(p for p in self.shared_block.parameters() if p.requires_grad)
                z, z_prev = RevDEQFunction.apply(f_theta, x0, z_init, beta, K, *params)
                return z, z_prev, None, None

            acc_dtype = torch.float64 if (not self.training and self.deq_backward == "revdeq") else torch.float32
            y_acc = z_init.to(acc_dtype)
            z_acc = z_init.to(acc_dtype)
            z = z_init
            z_prev = z
            for _ in range(K):
                z_prev = z
                f_z = f_theta(z, x0)
                y_acc = (1 - beta) * y_acc + beta * f_z.to(acc_dtype)
                y = y_acc.to(dtype)
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
                self._gg_iter_last_solve = [0.5 * (calls[2*i] + calls[2*i+1]) for i in range(K)]
            else:
                self._gg_iter_last_solve = []

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

        if self.training:
            if z_prev is not None:
                abs_conv = (z - z_prev).float().norm().item()
                z_norm_diag = z.detach().float().norm().clamp_min(1.0).item()
                self._deq_iter_convergence = abs_conv
                self._deq_iter_convergence_rel = abs_conv / z_norm_diag
                proxy = getattr(self.shared_block, "_deq_residual_proxy", None)
                self._deq_residuals = [float(proxy)] if proxy is not None else [abs_conv]
                if _should_diag(self.training):
                    with torch.no_grad():
                        f_z_final = self.shared_block(z, x0_refined)
                        self._deq_residuals = [(z - f_z_final).float().norm().item()]

            if self._block_ortho_aux_enabled and self.block_ortho_aux_coef > 0.0:
                max_tokens = int(getattr(self, "_block_ortho_aux_tokens_override", self.block_ortho_aux_tokens))
                attn_o, mlp_o = self.shared_block.ortho_aux(z, x0_refined, max_tokens=max_tokens)
                try:
                    self.shared_block.attn._out_ortho_cos_sim = float(attn_o.detach().float().item())
                    self.shared_block.mlp._out_ortho_cos_sim = float(mlp_o.detach().float().item())
                except Exception:
                    pass
                thr = 0.20
                attn_b = F.relu(attn_o - thr).pow(2)
                mlp_b = F.relu(mlp_o - thr).pow(2)
                self._block_ortho_aux_loss = 0.5 * (attn_b + mlp_b)

        if not self.training:
            with torch.no_grad():
                f_z_final = self.shared_block(z, x0_refined)
                abs_conv = (z - z_prev).float().norm().item()
                z_norm_diag = z.float().norm().clamp_min(1.0).item()
                self._deq_residuals = [(z - f_z_final).float().norm().item()]
                self._deq_iter_convergence = abs_conv
                self._deq_iter_convergence_rel = abs_conv / z_norm_diag
                try:
                    attn_o, mlp_o = self.shared_block.ortho_aux(z, x0_refined, max_tokens=128)
                    self.shared_block.attn._out_ortho_cos_sim = float(attn_o.float().item())
                    self.shared_block.mlp._out_ortho_cos_sim = float(mlp_o.float().item())
                except Exception:
                    pass

        return z

    def _encode(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = _rms_norm(x)
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
        bal = bal + getattr(self.mos_head, '_balance_loss', zero)
        return bal, health

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        log_p_ctp, log_p_ntp = self.mos_head(x)
        V = self.tok_emb.num_embeddings
        ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_ids.reshape(-1))
        ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_ids.reshape(-1))
        bal_loss, health_loss = self._collect_routing_losses(ntp_loss.device)
        self._ntp_loss = ntp_loss.detach().item()
        self._ctp_loss = ctp_loss.detach().item()
        refine_alpha = float(getattr(self, "_refine_mix_alpha", 0.5))
        refine_strength = min(max(refine_alpha / 0.5, 0.0), 1.0)
        ctp_weight = 0.05 * self.num_refinements * refine_strength

        mos_ortho_loss = torch.tensor(0.0, device=ntp_loss.device)
        if getattr(self.mos_head, "_ctp_ortho_out", None) is not None and getattr(self.mos_head, "_ntp_ortho_out", None) is not None:
            mos_ortho_loss = self.mos_head._ctp_ortho_out + self.mos_head._ntp_ortho_out

        block_ortho_aux = torch.tensor(0.0, device=ntp_loss.device)
        if self.training and self._block_ortho_aux_enabled and isinstance(self._block_ortho_aux_loss, torch.Tensor):
            block_ortho_aux = self._block_ortho_aux_loss.to(device=ntp_loss.device)

        eff_block_ortho_coef = float(self.block_ortho_aux_coef) * float(getattr(self, "_block_ortho_aux_coef_scale", 1.0))

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

    if int(args.deq_k_min) <= 0:
        raise ValueError("deq_k_min must be positive")
    if int(args.deq_k_max) < int(args.deq_k_min):
        raise ValueError("deq_k_max must be >= deq_k_min")
    if int(args.deq_k_max) > int(args.num_layers):
        raise ValueError(f"deq_k_max must be <= num_layers={int(args.num_layers)}")

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, math.ceil(8 / world_size))
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
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
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    k_rng = random.Random(args.seed + 12345)
    k_sampler = KShuffleBagSampler(args.deq_k_min, args.deq_k_max, k_rng, step=int(args.deq_k_step))

    def deq_k_for_step(step_i: int) -> int:
        k = 0
        if rank == 0:
            k = int(k_sampler.sample()) if args.deq_k_jitter else int(args.deq_k_max)
        if distributed:
            k_t = torch.tensor([k], device=device, dtype=torch.int64)
            dist.broadcast(k_t, src=0)
            k = int(k_t.item())
        return int(k)

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
        f" tie_router={int(args.tie_attn_mlp_router)}"
    )

    # MODEL
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim, num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank, mlp_expert_rank=args.mlp_expert_rank,
        deq_beta=args.deq_beta, attn_balance_mult=args.attn_balance_mult,
        mlp_balance_mult=args.mlp_balance_mult, bal_loss_coef=args.bal_loss_coef,
        router_health_coef=args.router_health_coef, mos_ortho_out_coef=args.mos_ortho_out_coef,
        deq_backward=args.deq_backward, block_ortho_aux_coef=args.block_ortho_aux_coef,
        block_ortho_aux_every=args.block_ortho_aux_every, block_ortho_aux_tokens=args.block_ortho_aux_tokens,
        tie_attn_mlp_router=args.tie_attn_mlp_router,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    model: nn.Module = (
        DDP(base_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        if distributed else base_model
    )

    # OPTIMIZER SETUP
    block_named_params = list(base_model.shared_block.named_parameters())
    router_params = [p for name, p in block_named_params
                     if name.endswith("attn_router.router.weight") or name.endswith("mlp_router.router.weight")]
    matrix_params = [p for name, p in block_named_params
                     if p.ndim >= 2 and not any(pat in name for pat in CONTROL_TENSOR_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_PATTERNS)]
    if router_params:
        router_set = {id(p) for p in router_params}
        matrix_params = [p for p in matrix_params if id(p) not in router_set]
        scalar_params = [p for p in scalar_params if id(p) not in router_set]
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)

    mos = base_model.mos_head
    mos_params = [mos.A_shared, mos.A_ctp, mos.A_ntp, mos.B_denoise, mos.B_NTP,
                  mos.gate_ctp.weight, mos.gate_ctp.bias, mos.gate_ntp.weight, mos.gate_ntp.bias]
    scalar_params.extend(mos_params)

    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2),
                                       eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=0.04)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay, fused=True)
    optimizer_router = None
    if router_params:
        optimizer_router = torch.optim.AdamW(
            [{"params": router_params, "lr": float(args.router_lr), "base_lr": float(args.router_lr)}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.0, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_router is not None:
        optimizers.append(optimizer_router)

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
        if hasattr(m, "_deq_residuals") and getattr(m, "_deq_residuals"):
            parts.append(f"deq_residual:{m._deq_residuals[-1]:.6f}")
        recon = getattr(m, "_deq_recon_error", None)
        if recon is not None:
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
        # Per-DEQ-iteration gg trajectory: each entry is the average gg_tok
        # over the (y, z) sub-call pair at iteration index i.  A *decreasing*
        # trend across iterations means the model is making smaller updates
        # as it approaches the fixed point -- desirable.  A flat trend means
        # the iterative depth is wasted.
        gg_iter = getattr(m, "_gg_iter_last_solve", None)
        if gg_iter is not None and len(gg_iter) > 0:
            iter_str = ",".join(f"{float(v):.3f}" for v in gg_iter)
            parts.append(f"gg_iter:[{iter_str}]")
        return (" " + " ".join(parts)) if parts else ""

    def format_expert_info(m: nn.Module, *, step: int | None = None, require_step_match: bool = False) -> str:
        parts: list[str] = []
        if hasattr(m, "shared_block"):
            for prefix, router in (("attn", getattr(m.shared_block.attn, "attn_router", None)),
                                   ("mlp", getattr(m.shared_block.mlp, "mlp_router", None))):
                ok = router is not None and getattr(router, "_expert_usage", None) is not None
                if ok and require_step_match and getattr(router, "_diag_step", None) != step:
                    ok = False
                if not ok:
                    continue
                usage_str = ",".join(f"{u:.3f}" for u in router._expert_usage)
                parts.append(f"{prefix}_usage:[{usage_str}]")
                ent = getattr(router, "_expert_entropy", None)
                if ent is not None:
                    parts.append(f"{prefix}_entropy:{ent:.4f}")
                cv = getattr(router, "_expert_balance_cv", None)
                if cv is not None:
                    parts.append(f"{prefix}_cv:{cv:.4f}")
            for attr, label in [("attn", "attn_ortho"), ("mlp", "mlp_ortho")]:
                v = getattr(getattr(m.shared_block, attr, None), "_out_ortho_cos_sim", None)
                if v is not None:
                    parts.append(f"{label}:{float(v):.4f}")
        if hasattr(m, "mos_head"):
            mos = m.mos_head
            if hasattr(mos, "get_head_orthogonality"):
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
        try:
            base_model.shared_block.attn_router.health_scale = health_scale
            base_model.shared_block.mlp_router.health_scale = health_scale
        except Exception:
            pass

        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        if hasattr(base_model.shared_block, "_deq_recon_error_last_bwd"):
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
        base_model._deq_k_override = deq_k_for_step(next_step)

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
            _preclip = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm).item()
        else:
            _preclip = 0.0
        for opt in optimizers:
            opt.step()

        if args.router_bias_update:
            seen: set[int] = set()
            bias_lr = float(args.router_bias_lr) * (1.0 + 4.0 * float(late_frac))
            for r in [base_model.shared_block.attn.attn_router, base_model.shared_block.mlp.mlp_router]:
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                r.bias_update(lr=bias_lr, clip=float(args.router_bias_clip), distributed=distributed)
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
            ntp = getattr(base_model, '_ntp_loss', 0.0)
            ctp = getattr(base_model, '_ctp_loss', 0.0)
            base_model._deq_recon_error = getattr(base_model.shared_block, "_deq_recon_error_last_bwd", None)
            deq_info = format_deq_info(base_model)
            expert_info = format_expert_info(base_model, step=step, require_step_match=True) if master_process else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ntp_loss:{ntp:.4f} ctp_loss:{ctp:.4f} "
                f"grad_norm:{_preclip:.4f} "
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

    # Keep DDP alive through post-training validation so run_validation and
    # sliding_window_validation can shard the val set across both ranks.
    if distributed:
        dist.barrier()

    # Master-only: quantization, compression, artifact save, decompression,
    # and reload of dequantized weights.  These steps are inherently serial.
    base_model.train(False)
    meta_path: Path | None = None
    val_bpb_q = 0.0
    if master_process:
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")
        sd = base_model.state_dict()
        int6_cats = {"matrix", "embed", "bigram"}
        qsd, meta = mixed_quantize_int6(sd, int6_cats)
        buf = io.BytesIO()
        torch.save({"state_dict": qsd, "meta": meta}, buf)
        raw_bytes = buf.getvalue()
        if _COMPRESSOR == "zstd":
            cctx = zstandard.ZstdCompressor(level=22)
            compressed = cctx.compress(raw_bytes)
        else:
            compressed = zlib.compress(raw_bytes, 9)
        artifact_bytes = len(compressed)
        log0(f"artifact_bytes:{artifact_bytes} compressor:{_COMPRESSOR}")

        weights_dir = Path("experiments/weights/current")
        weights_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = weights_dir / "model.int6.ptz"
        with open(artifact_path, "wb") as f:
            f.write(compressed)

        import json
        meta_path = weights_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "val_bpb": 0.0,
                "artifact_bytes": artifact_bytes,
                "step": step,
                "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True, check=False).stdout.strip(),
            }, f)

        log0("roundtrip_verification:start")
        if _COMPRESSOR == "zstd":
            dctx = zstandard.ZstdDecompressor()
            decompressed = dctx.decompress(compressed)
        else:
            decompressed = zlib.decompress(compressed)
        loaded = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=True)
        deq_sd = dequantize_mixed_int6(loaded["state_dict"], loaded["meta"], sd)
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

    # All ranks: roundtrip validation sharded across the val set via DDP.
    base_m_for_roundtrip = base_model
    base_m_for_roundtrip._deq_k_override = int(args.deq_k_eval)
    val_loss_q, val_bpb_q = run_validation(
        args, base_m_for_roundtrip, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        full_validation=True,
    )
    log0(f"roundtrip_verification:done val_loss:{val_loss_q:.4f} val_bpb:{val_bpb_q:.6f}")

    # DEQ fixed-point K-sweep: verify val_bpb improves (or plateaus) as K grows.
    # A valid DEQ should converge to a fixed point — more solver iterations = better
    # or equal quality, never worse.  Non-monotone behaviour indicates the model
    # is exploiting a specific iteration count rather than a true fixed point.
    # Runs DDP-parallel across ranks for a ~2x speedup on 2 GPUs.
    log0("k_sweep:start")
    k_sweep_values = [4, 8, 16]
    k_sweep_results: dict[int, float] = {}
    for k_eval in k_sweep_values:
        base_m_for_roundtrip._deq_k_override = int(k_eval)
        _, bpb_k = run_validation(
            args, base_m_for_roundtrip, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            full_validation=True,
        )
        k_sweep_results[k_eval] = float(bpb_k)
        log0(f"k_sweep:k={k_eval} val_bpb:{bpb_k:.6f}")
    k_parts = " ".join(f"k{k}:{b:.6f}" for k, b in k_sweep_results.items())
    log0(f"k_sweep:done {k_parts}")
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

    # Master-only: update meta.json with the final val_bpb.
    if master_process and meta_path is not None:
        import json
        with open(meta_path, "r") as f:
            meta_json = json.load(f)
        meta_json["val_bpb"] = val_bpb_q
        with open(meta_path, "w") as f:
            json.dump(meta_json, f)

    # Tear down DDP after all eval completes.
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
