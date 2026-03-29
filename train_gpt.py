"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Note: For this autoresearch workspace, `train_gpt.py` may grow beyond a small tutorial script as long as it remains a single-file, self-contained submission artifact. If you want a newcomer-friendly baseline, prefer the pinned record scripts in `records/`.
"""

from __future__ import annotations

import copy
import contextlib
import glob
import io
import math
import os
import random
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 10000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 98_304))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 1))  # DEQ solver iters per refinement step
    num_refinements = int(os.environ.get("NUM_REFINEMENTS", 1))  # predict→soft_embed→re-encode cycles
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 640))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 2.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 1000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 20.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 400))
    beta1 = float(os.environ.get("BETA1", 0.85))
    beta2 = float(os.environ.get("BETA2", 0.90))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.25))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))  # 0=standard eval; set >0 for sliding window (final only)
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 65536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 224))
    kv_latent_dim = int(os.environ.get("KV_LATENT_DIM", 0))  # 0 = auto (dim//2)

    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.3))
    swa_every = int(os.environ.get("SWA_EVERY", 25))

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
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
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
            # Use NTP-only loss for val (exclude CTP and convergence terms)
            base_m = model.module if hasattr(model, 'module') else model
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
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION (INT8 legacy + INT6 mixed)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        # Keep only small control tensors in fp32. Avoid broad substrings like "expert_gate"
        # which can match large expert weight tensors (e.g., shared_block.mlp.expert_gate).
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,"
        "skip_weight,skip_weights,smear.gate,bigram.scale,diffar_scale,gate_bias,"
        "expert_gate_logits,expert_gate_ctp_logits,expert_gate_ntp_logits",
    ).split(",")
    if pattern
)
FP16_KEEP_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("FP16_KEEP_NAME_PATTERNS", "tok_emb").split(",")
    if pattern
)
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

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class SoftDenseRouter(nn.Module):
    """Shared soft dense routing module for all MoE components.

    Per-token routing: softmax weights × per-expert sigmoid gates (post-softmax),
    to break the convex-mixture constraint (allows skipping experts) while
    remaining fully differentiable.
    Provides sparsity (L1) + balance (MSE) regularization and diagnostics.
    """
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.router = CastedLinear(dim, num_experts, bias=False)
        # Small router init → near-uniform routing at start
        nn.init.normal_(self.router.weight, std=0.01)
        # Learned per-expert gate scalars (post-softmax); init open (sigmoid(1)=0.73)
        self.expert_gate_logits = nn.Parameter(torch.ones(num_experts, dtype=torch.float32))
        # Diagnostics (set during forward)
        self._balance_loss = None
        self._sparsity_loss = None
        self._expert_usage = None
        self._expert_entropy = None
        self._expert_balance_cv = None

    def forward(self, x: Tensor) -> Tensor:
        """Returns routing weights [*, num_experts]."""
        route_logits = self.router(x)
        alpha = torch.softmax(route_logits, dim=-1)
        gates = torch.sigmoid(self.expert_gate_logits.to(dtype=alpha.dtype))
        route_weights = alpha * gates
        if self.training:
            mean_alpha = alpha.mean(dim=tuple(range(alpha.ndim - 1)))
            target = torch.ones_like(mean_alpha) / self.num_experts
            self._balance_loss = F.mse_loss(mean_alpha, target)
            self._sparsity_loss = route_weights.abs().mean()
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._sparsity_loss = torch.tensor(0.0, device=x.device)
            with torch.no_grad():
                mean_alpha = alpha.mean(dim=tuple(range(alpha.ndim - 1)))
                self._expert_usage = mean_alpha.float().cpu().tolist()
                per_token_ent = -(alpha * (alpha + 1e-8).log()).sum(-1)
                self._expert_entropy = per_token_ent.mean().item()
                self._expert_balance_cv = (mean_alpha.std() / mean_alpha.mean()).item()
        return route_weights


class CausalSelfAttention(nn.Module):
    """MLA with Gated Attention + Soft Dense Routing (Constraints #2, #3).

    - Low-rank KV compression via shared latent
    - Decoupled RoPE: half of head_dim for positional encoding
    - Query-dependent per-head sigmoid gate after SDPA (scalar gate per head)
    - Soft dense routing on output projection (MoE for attention)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, kv_latent_dim: int = 0, num_experts: int = 2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_experts = num_experts
        self.expert_size = dim // num_experts
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # MLA: low-rank KV compression with decoupled RoPE
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else dim // 2
        self.rope_dim = self.head_dim // 2  # half for RoPE
        self.nope_dim = self.head_dim - self.rope_dim

        # Q projection outputs query + scalar per-head gate logits.
        self.c_q = CastedLinear(dim, dim + num_heads, bias=False)
        # KV compression path
        self.c_kv_down = CastedLinear(dim, self.kv_latent_dim, bias=False)
        self.c_k_nope = CastedLinear(self.kv_latent_dim, num_kv_heads * self.nope_dim, bias=False)
        self.c_v = CastedLinear(self.kv_latent_dim, num_kv_heads * self.head_dim, bias=False)
        # Decoupled RoPE key
        self.c_k_rope = CastedLinear(dim, num_kv_heads * self.rope_dim, bias=False)
        # Dense mixture experts: low-rank output projections per expert (dim -> r -> dim)
        # Keep name expert_proj for downstream diagnostics/tests compatibility.
        self.expert_proj = nn.Parameter(torch.empty(num_experts, self.expert_size, dim))
        self.expert_out = nn.Parameter(torch.empty(num_experts, dim, self.expert_size))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_proj.data[e])
            nn.init.xavier_uniform_(self.expert_out.data[e])
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dim, base=rope_base)
        # Gated attention bias (per-head scalar)
        self.gate_bias = nn.Parameter(torch.ones(num_heads, dtype=torch.float32))
        # Soft dense routing on attention output (MoE for attention)
        self.attn_router = SoftDenseRouter(dim, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        # Q projection outputs query vectors + per-head gate logits
        q_and_gate = self.c_q(x)  # [B, T, dim + num_heads]
        q_raw = q_and_gate[..., :dim].reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        gate_logits = q_and_gate[..., dim:].reshape(bsz, seqlen, self.num_heads, 1).transpose(1, 2)
        q_rope, q_nope = q_raw[..., :self.rope_dim], q_raw[..., self.rope_dim:]

        kv_latent = self.c_kv_down(x)
        k_nope = self.c_k_nope(kv_latent).reshape(bsz, seqlen, self.num_kv_heads, self.nope_dim).transpose(1, 2)
        v = self.c_v(kv_latent).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_rope = self.c_k_rope(x).reshape(bsz, seqlen, self.num_kv_heads, self.rope_dim).transpose(1, 2)

        q_rope, q_nope = _rms_norm(q_rope), _rms_norm(q_nope)
        k_rope, k_nope = _rms_norm(k_rope), _rms_norm(k_nope)

        cos, sin = self.rotary(seqlen, x.device, q_rope.dtype)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos, sin)

        q_full = torch.cat([q_rope, q_nope], dim=-1)
        k_full = torch.cat([k_rope, k_nope], dim=-1)
        q_full = q_full * self.q_gain.to(dtype=q_full.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        # Gated attention: query-dependent per-head gate (arxiv:2505.06708)
        y = y * torch.sigmoid(gate_logits.to(dtype=y.dtype) + self.gate_bias[None, :, None, None].to(y.dtype))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        # Soft dense routing on attention output (dense mixture-of-experts)
        route_weights = self.attn_router(x)  # [B, T, E]
        h = torch.einsum('btd,erd->bter', y, self.expert_proj)
        out_e = torch.einsum('bter,edr->bted', h, self.expert_out)
        return (out_e * route_weights.unsqueeze(-1)).sum(dim=2)

    @property
    def attn_gate(self) -> Tensor:
        # Backwards-compat for older tests/diagnostics.
        return self.gate_bias


class MLP(nn.Module):
    """SiLU-gated MLP with true expert parameters + Soft Dense Routing (Constraint #2)."""
    def __init__(self, dim: int, mlp_mult: float, num_experts: int = 2):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.num_experts = num_experts
        self.expert_size = hidden // num_experts
        # Dense mixture experts: low-rank SwiGLU blocks per expert (dim -> r -> dim)
        self.expert_gate = nn.Parameter(torch.empty(num_experts, self.expert_size, dim))
        self.expert_fc = nn.Parameter(torch.empty(num_experts, self.expert_size, dim))
        self.expert_down = nn.Parameter(torch.empty(num_experts, dim, self.expert_size))
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.expert_gate.data[e])
            nn.init.xavier_uniform_(self.expert_fc.data[e])
            nn.init.xavier_uniform_(self.expert_down.data[e])
        self.mlp_router = SoftDenseRouter(dim, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        route_weights = self.mlp_router(x)  # [B, T, E]
        gate_h = torch.einsum('btd,esd->btes', x, self.expert_gate)
        fc_h = torch.einsum('btd,esd->btes', x, self.expert_fc)
        h = F.silu(gate_h) * fc_h  # [B, T, E, expert_size]
        out_e = torch.einsum('btes,eds->bted', h, self.expert_down)
        return (out_e * route_weights.unsqueeze(-1)).sum(dim=2)

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
        # Orthogonality: pairwise cosine sim between expert weight groups
        if self.num_experts >= 2:
            with torch.no_grad():
                groups = self.expert_fc.float().view(self.num_experts, -1)
                groups = groups / (groups.norm(dim=-1, keepdim=True) + 1e-8)
                cos = groups @ groups.T
                mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=cos.device)
                diag["ortho_cos_sim"] = cos[mask].abs().mean().item()
        return diag


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        # Initialize near-identity: mostly current token, slight previous-token injection.
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))  # sigmoid(3)=0.95

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
            h = self.proj(h)
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
        # Post-softmax per-expert gates (allow skipping experts while keeping a valid mixture)
        self.expert_gate_ctp_logits = nn.Parameter(torch.ones(num_shared + num_specialized, dtype=torch.float32))
        self.expert_gate_ntp_logits = nn.Parameter(torch.ones(num_shared + num_specialized, dtype=torch.float32))
        # Shared A projections: [num_shared, d_model, rank]
        self.A_shared = nn.Parameter(torch.empty(num_shared, d_model, rank))
        # Specialized A projections: 1 for CTP, 1 for NTP
        self.A_ctp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        self.A_ntp = nn.Parameter(torch.empty(num_specialized, d_model, rank))
        # Dual B matrices (shared across all experts within each head)
        self.B_denoise = nn.Parameter(torch.empty(vocab_size, rank))
        self.B_NTP = nn.Parameter(torch.empty(vocab_size, rank))
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

    def _fsq(self, x: Tensor) -> Tensor:
        return _fsq_ste(x, self.fsq_levels, self.training)

    def _head_forward(self, x: Tensor, gate: nn.Linear, A_shared: Tensor,
                      A_spec: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (log_probs, alpha_softmax, alpha_softmax*sigmoid_gate) for one head."""
        N = x.shape[0]
        alpha = F.softmax(gate(x).float(), dim=-1)  # [N, num_shared+num_spec]
        if gate is self.gate_ctp:
            gates = torch.sigmoid(self.expert_gate_ctp_logits)[None, :]
        else:
            gates = torch.sigmoid(self.expert_gate_ntp_logits)[None, :]
        weights = alpha * gates.to(dtype=alpha.dtype)
        log_w = weights.clamp(min=1e-8).log()
        log_p_unnorm = x.new_full((N, self.vocab_size), -torch.inf, dtype=torch.float32)
        # Shared experts
        for e in range(self.num_shared):
            u = self._fsq(x.to(A_shared.dtype) @ A_shared[e])
            logits = u.to(B.dtype) @ B.t()
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, e:e+1] + F.log_softmax(logits.float(), dim=-1))
        # Specialized experts
        for e in range(self.num_specialized):
            u = self._fsq(x.to(A_spec.dtype) @ A_spec[e])
            logits = u.to(B.dtype) @ B.t()
            idx = self.num_shared + e
            log_p_unnorm = torch.logaddexp(log_p_unnorm, log_w[:, idx:idx+1] + F.log_softmax(logits.float(), dim=-1))
        log_p = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=-1, keepdim=True)
        return log_p, alpha, weights

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return (log_p_denoise, log_p_ntp), each [*, V]."""
        orig_shape = h.shape[:-1]
        x = h.reshape(-1, self.d_model)

        log_p_d, alpha_d, w_d = self._head_forward(x, self.gate_ctp, self.A_shared, self.A_ctp, self.B_denoise)
        log_p_n, alpha_n, w_n = self._head_forward(x, self.gate_ntp, self.A_shared, self.A_ntp, self.B_NTP)

        if self.training:
            bal = torch.tensor(0.0, device=x.device)
            spar = torch.tensor(0.0, device=x.device)
            for alpha_soft in [alpha_d, alpha_n]:
                mean_a = alpha_soft.mean(dim=0)
                target = torch.ones_like(mean_a) / alpha_soft.shape[-1]
                bal = bal + F.mse_loss(mean_a, target)
            for w in [w_d, w_n]:
                spar = spar + w.abs().mean()
            self._balance_loss = bal
            self._sparsity_loss = spar
        else:
            self._balance_loss = torch.tensor(0.0, device=x.device)
            self._sparsity_loss = torch.tensor(0.0, device=x.device)

        return log_p_d.view(*orig_shape, -1), log_p_n.view(*orig_shape, -1)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, kv_latent_dim: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, kv_latent_dim=kv_latent_dim)
        self.mlp = MLP(dim, mlp_mult)
        # FSQ moved to output head for param-efficient MoS
        # Small init for DEQ stability — block starts as near-identity
        self.attn_scale = nn.Parameter(torch.full((dim,), 0.01, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 0.01, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x


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

        y_state = z_init.to(state_dtype)
        z_state = z_init.to(state_dtype)
        z_prev_state = z_state

        with torch.no_grad():
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

        ctx.save_for_backward(x0.detach(), y_state.detach(), z_state.detach(), z_prev_state.detach())
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
        self.smear = SmearGate(model_dim)
        # RevDEQ (Constraint #1): single shared block with coupled-state fixed-point iteration
        self.shared_block = Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                                  rope_base, qk_gain_init, kv_latent_dim=kv_latent_dim)
        self.deq_beta = 0.5  # relaxation parameter for coupled-state iteration
        # Diffusion-AR scale: controls strength of prediction-feedback (init small for DEQ stability)
        self.diffar_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
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
        # Initialize MoS head from embedding weights
        self.mos_head.init_from_embedding(self.tok_emb.weight.data)

    def _get_soft_embedding(self, z: Tensor, topk: int = 32) -> Tensor:
        """Diffusion-AR: build soft embedding from CTP + NTP predictions.

        For each position i, combines two signals via frozen expert:
        - CTP[i] from MoSHead: predicts token at position i (from context 0..i)
        - NTP[i-1] from MoSHead: predicts next token after i-1 (= token i)
        Mix probabilities, take top-k, build sparse soft embedding.
        """
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
            p_mix = 0.5 * (log_p_ctp.float().exp() + log_p_ntp_shifted.float().exp())
            topk_probs, topk_idx = p_mix.topk(topk, dim=-1)  # [B,T,K]
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            W = self.tok_emb.weight.data  # [V, d]
            topk_embeds = F.embedding(topk_idx, W)  # [B,T,K,d]
            soft_embed = (topk_probs.unsqueeze(-1) * topk_embeds).sum(-2)  # [B,T,d]
        return self.diffar_scale.to(dtype=z.dtype) * soft_embed.to(z.dtype)

    def _deq_solve(self, x0: Tensor, z_init: Tensor):
        """Run DEQ coupled-state solver. Uses RevDEQFunction for O(1) memory in training."""
        if self.training:
            params = tuple(p for p in self.shared_block.parameters() if p.requires_grad)
            z, z_prev = RevDEQFunction.apply(
                self.shared_block, x0, z_init, self.deq_beta, self.num_layers, *params
            )
            return z, z_prev, None, None

        # Eval: explicit loop for diagnostics + reconstruction check
        beta = self.deq_beta
        dtype = x0.dtype
        acc_dtype = torch.float64
        y_acc = z_init.to(acc_dtype)
        z_acc = z_init.to(acc_dtype)
        z = z_init
        z_prev = z
        for _ in range(self.num_layers):
            z_prev = z
            f_z = self.shared_block(z, x0)
            y_acc = (1 - beta) * y_acc + beta * f_z.to(acc_dtype)
            y = y_acc.to(dtype)
            f_y = self.shared_block(y, x0)
            z_acc = (1 - beta) * z_acc + beta * f_y.to(acc_dtype)
            z = z_acc.to(dtype)
        return z, z_prev, y_acc, z_acc

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
        prev_soft_embed = None
        x0_refined = x0  # track for reconstruction

        for r in range(1 + self.num_refinements):
            if r > 0:
                new_soft_embed = self._get_soft_embedding(z)
                if prev_soft_embed is not None:
                    soft_embed = 0.5 * new_soft_embed + 0.5 * prev_soft_embed
                else:
                    soft_embed = new_soft_embed
                prev_soft_embed = soft_embed.detach()
                x0_refined = x0 + soft_embed
                z = x0_refined  # warm start from refined input (not old fixed point)
            else:
                x0_refined = x0

            z, z_prev, y_acc, z_acc = self._deq_solve(x0_refined, z)

        # Convergence loss: ||z_T - z_{T-1}||² / ||z_T||² (training)
        if self.training:
            z_norm_sq = z.detach().float().pow(2).sum().clamp_min(1.0)
            if z_prev is None:
                f_z = self.shared_block(z, x0_refined)
                self._convergence_loss = (z - f_z).float().pow(2).sum() / z_norm_sq
            else:
                self._convergence_loss = (z - z_prev).float().pow(2).sum() / z_norm_sq
                # Lightweight training diagnostics (no extra block call):
                # track the final-iter update magnitude as a proxy for solver convergence.
                self._deq_residuals = [(z - z_prev).float().norm().item()]
                self._deq_iter_convergence = self._deq_residuals[0]

        # Diagnostics (eval only)
        if not self.training:
            with torch.no_grad():
                f_z_final = self.shared_block(z, x0_refined)
                self._deq_residuals = [(z - f_z_final).float().norm().item()]
                self._deq_iter_convergence = (z - z_prev).float().norm().item()
                # fp64 backward reconstruction of last DEQ solve
                z_init_64 = (x0_refined if self.num_refinements > 0 else x0).to(torch.float64)
                yr_acc, zr_acc = y_acc.clone(), z_acc.clone()
                beta = self.deq_beta
                for _ in range(self.num_layers):
                    f_yr = self.shared_block(yr_acc.to(dtype), x0_refined)
                    zr_acc = (zr_acc - beta * f_yr.to(torch.float64)) / (1 - beta)
                    f_zr = self.shared_block(zr_acc.to(dtype), x0_refined)
                    yr_acc = (yr_acc - beta * f_zr.to(torch.float64)) / (1 - beta)
                state_norm = max(z_init_64.norm().item(), 1.0)
                recon_error = ((zr_acc - z_init_64).norm().item() + (yr_acc - z_init_64).norm().item()) / state_norm
                self._deq_recon_error = recon_error

        return z

    def _encode(self, input_ids: Tensor) -> Tensor:
        """Shared embedding + backbone: input_ids → normalized hidden states."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = _rms_norm(x)
        x = self.smear(x)
        x = self._run_backbone(x)
        return self.final_norm(x)

    def _collect_routing_losses(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """Collect balance, sparsity, and orthogonality losses from all routers."""
        zero = torch.tensor(0.0, device=device)
        bal, spar, ortho = zero, zero, zero
        # All SoftDenseRouters: attn + mlp
        routers = [self.shared_block.attn.attn_router, self.shared_block.mlp.mlp_router]
        for r in routers:
            bal = bal + getattr(r, '_balance_loss', zero)
            spar = spar + getattr(r, '_sparsity_loss', zero)
        # MoS head routing
        bal = bal + getattr(self.mos_head, '_balance_loss', zero)
        spar = spar + getattr(self.mos_head, '_sparsity_loss', zero)
        # Orthogonality: |cos_sim| between expert weight groups → 0
        for w in [
            self.shared_block.mlp.expert_fc.float(),   # [E, expert_size, dim]
            self.shared_block.mlp.expert_down.float(),  # [E, dim, expert_size]
            self.shared_block.attn.expert_proj.float(), # [E, expert_size, dim]
            self.shared_block.attn.expert_out.float(),  # [E, dim, expert_size]
            self.mos_head.A_shared.float(),             # [E, d_model, rank]
        ]:
            n_exp = w.shape[0]
            if n_exp < 2:
                continue
            groups = w.mean(dim=1)  # [E, feat]
            groups = groups / (groups.norm(dim=-1, keepdim=True) + 1e-8)
            cos = groups @ groups.T
            mask = ~torch.eye(n_exp, dtype=torch.bool, device=cos.device)
            ortho = ortho + cos[mask].abs().mean()
        return bal, spar, ortho

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids)
        log_p_ctp, log_p_ntp = self.mos_head(x)
        V = self.tok_emb.num_embeddings
        ntp_loss = F.nll_loss(log_p_ntp.reshape(-1, V), target_ids.reshape(-1))
        ctp_loss = F.nll_loss(log_p_ctp.reshape(-1, V), input_ids.reshape(-1))
        conv_loss = getattr(self, '_convergence_loss', torch.tensor(0.0, device=ntp_loss.device))
        bal_loss, spar_loss, ortho_loss = self._collect_routing_losses(ntp_loss.device)
        self._ntp_loss = ntp_loss.detach().item()
        self._ctp_loss = ctp_loss.detach().item()
        self._conv_loss = conv_loss.detach().item() if isinstance(conv_loss, torch.Tensor) else 0.0
        # CTP weight scales with refinement steps: at step 0 input is clean one-hot,
        # CTP becomes meaningful only after soft embedding refinement
        ctp_weight = 0.1 * self.num_refinements
        return ntp_loss + ctp_weight * ctp_loss + 0.001 * conv_loss + 0.1 * bal_loss + 0.001 * spar_loss + 0.01 * ortho_loss

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
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    requested_grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "0"))
    if requested_grad_accum_steps > 0:
        grad_accum_steps = requested_grad_accum_steps
    else:
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
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
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
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = base_model  # skip compile for training; use compiled forward_logits for eval
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

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
    scalar_params.append(base_model.smear.gate)
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

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
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
            )
            # Log DEQ convergence metrics
            deq_info = ""
            if hasattr(base_model, '_deq_residuals') and base_model._deq_residuals:
                deq_info = f" deq_residual:{base_model._deq_residuals[-1]:.6f}"
            if hasattr(base_model, '_deq_recon_error'):
                deq_info += f" deq_recon_err:{base_model._deq_recon_error:.6f}"
            if hasattr(base_model, '_deq_iter_convergence'):
                deq_info += f" deq_iter_conv:{base_model._deq_iter_convergence:.6f}"
            # Expert diagnostics
            expert_info = ""
            mlp = base_model.shared_block.mlp if hasattr(base_model, 'shared_block') else None
            if mlp is not None and hasattr(mlp, 'get_expert_diagnostics'):
                diag = mlp.get_expert_diagnostics()
                if 'usage' in diag:
                    usage_str = ",".join(f"{u:.3f}" for u in diag['usage'])
                    expert_info = f" expert_usage:[{usage_str}]"
                if 'entropy' in diag:
                    expert_info += f" expert_entropy:{diag['entropy']:.4f}"
                if 'balance_cv' in diag:
                    expert_info += f" expert_balance_cv:{diag['balance_cv']:.4f}"
                if 'ortho_cos_sim' in diag:
                    expert_info += f" expert_ortho:{diag['ortho_cos_sim']:.4f}"
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
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
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
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
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
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
            conv = getattr(base_model, '_conv_loss', 0.0)
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ntp_loss:{ntp:.4f} ctp_loss:{ctp:.4f} conv_loss:{conv:.6f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
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

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
# fixes applied
# tuned
