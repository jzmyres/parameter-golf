"""Tier-1 synthetic depth-hard harness for the OPG P1 design.

This script intentionally lives outside ``train_gpt.py``.  The main trainer is
the legacy small-LM path; this harness is the executable science path for
``reports/opg_doc.tex``:

* vanilla looped-transformer positive control, M0, and M_clk on S5 composition.
* P1 metrics: paired depth gain, no-degradation rate, and K sweep.
* P2 diagnostics: structural reverse reconstruction error for M0/M_clk.

It is DDP-safe and can be launched with:

    CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
      experiments/p1_synthetic.py --variant mclk --iterations 300
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_sum(x: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def build_s5_tables(device: torch.device | None = None) -> tuple[Tensor, int]:
    """Return composition table for S5 and the identity permutation index.

    ``compose[p, q]`` represents p after q: ``i -> p[q[i]]``.  A sequence is
    reduced left-to-right by ``state <- compose[token, state]``.
    """

    perms = list(itertools.permutations(range(5)))
    index = {p: i for i, p in enumerate(perms)}
    table = torch.empty((len(perms), len(perms)), dtype=torch.long)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            table[i, j] = index[tuple(p[q[k]] for k in range(5))]
    identity = index[(0, 1, 2, 3, 4)]
    if device is not None:
        table = table.to(device)
    return table, identity


def compose_s5_sequence(tokens: Tensor, compose_table: Tensor, identity_idx: int) -> Tensor:
    """Compose a batch of S5 token sequences into target permutation IDs."""

    state = torch.full(tokens.shape[:1], int(identity_idx), dtype=torch.long, device=tokens.device)
    table = compose_table.to(tokens.device)
    for t in range(tokens.shape[1]):
        state = table[tokens[:, t], state]
    return state


def sample_s5_batch(
    batch_size: int,
    seq_len: int,
    compose_table: Tensor,
    identity_idx: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    tokens = torch.randint(
        0,
        120,
        (int(batch_size), int(seq_len)),
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    target = compose_s5_sequence(tokens, compose_table, identity_idx)
    return tokens, target


def sample_task_batch(
    args: argparse.Namespace,
    compose_table: Tensor,
    identity_idx: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if args.task == "parity":
        tokens = torch.randint(
            0,
            2,
            (int(args.batch_size), int(args.seq_len)),
            device=device,
            generator=generator,
            dtype=torch.long,
        )
        target = tokens.sum(dim=1).remainder(2)
        return tokens, target
    return sample_s5_batch(args.batch_size, args.seq_len, compose_table, identity_idx, device, generator)


def sample_eval_batch(
    args: argparse.Namespace,
    compose_table: Tensor,
    identity_idx: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    batch_size = int(args.eval_batch_size)
    if args.task == "parity":
        tokens = torch.randint(0, 2, (batch_size, int(args.seq_len)), device=device, dtype=torch.long)
        target = tokens.sum(dim=1).remainder(2)
        return tokens, target
    return sample_s5_batch(batch_size, args.seq_len, compose_table, identity_idx, device)


def task_num_classes(task: str) -> int:
    if task == "parity":
        return 2
    return 120


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up = nn.Linear(dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class StandardMhaSwiGLUBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: float,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        hidden_dim = max(8, int(round(float(mlp_mult) * dim)))
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        y = x + attn_out
        h2 = self.mlp_norm(y)
        return attn_out + self.ffn(h2)


class HaltingReadout(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, states: list[Tensor]) -> Tensor:
        pooled = torch.stack([s.mean(dim=1) for s in states], dim=1)
        gates = torch.sigmoid(self.halt(pooled).squeeze(-1))
        gates = gates.clone()
        gates[:, -1] = 1.0
        survival = torch.ones_like(gates[:, :1])
        weights = []
        for k in range(gates.shape[1]):
            w = survival * gates[:, k : k + 1]
            weights.append(w)
            survival = survival * (1.0 - gates[:, k : k + 1])
        weight_t = torch.stack(weights, dim=1)
        out = (weight_t * pooled).sum(dim=1)
        return self.head(self.norm(out))


class AdditiveCouplingP1Model(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: float,
        seq_len: int,
        variant: str,
        num_classes: int,
    ):
        super().__init__()
        self.variant = str(variant)
        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.tok_emb = nn.Embedding(120, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.step_emb = nn.Embedding(512, dim) if self.variant == "mclk" else None
        self.f_block = StandardMhaSwiGLUBlock(dim, num_heads, mlp_mult)
        self.g_block = StandardMhaSwiGLUBlock(dim, num_heads, mlp_mult)
        self.norm_f = RMSNorm(dim)
        self.norm_g = RMSNorm(dim)
        self.readout = HaltingReadout(dim, int(num_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if self.step_emb is not None:
            nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)

    def _x0(self, tokens: Tensor) -> Tensor:
        return self.tok_emb(tokens) + self.pos_emb[:, : tokens.shape[1]]

    def _step(self, k: int, x0: Tensor) -> Tensor:
        if self.step_emb is None:
            return x0.new_zeros((1, 1, self.dim))
        idx = torch.full((1,), min(int(k), self.step_emb.num_embeddings - 1), device=x0.device, dtype=torch.long)
        return self.step_emb(idx).view(1, 1, self.dim)

    def recurrent_states(self, tokens: Tensor, depth: int, return_terminal: bool = False):
        x0 = self._x0(tokens)
        a = x0
        b = x0
        states: list[Tensor] = []
        for k in range(int(depth)):
            step = self._step(k, x0)
            f_in = self.norm_f(b + x0 + step)
            f_delta = self.f_block(f_in)
            a = a + f_delta
            g_in = self.norm_g(a + x0 + step)
            g_delta = self.g_block(g_in)
            b = b + g_delta
            states.append(0.5 * (a + b))
        if return_terminal:
            return states, (x0, a, b)
        return states

    def forward(self, tokens: Tensor, depth: int) -> Tensor:
        return self.readout(self.recurrent_states(tokens, depth))

    @torch.no_grad()
    def reconstruction_error(self, tokens: Tensor, depth: int) -> float:
        if self.training:
            raise RuntimeError("reconstruction_error requires eval mode")
        states, terminal = self.recurrent_states(tokens, depth, return_terminal=True)
        del states
        x0, a, b = terminal
        a0 = x0
        b0 = x0
        for k in reversed(range(int(depth))):
            step = self._step(k, x0)
            g_in = self.norm_g(a + x0 + step)
            g_delta = self.g_block(g_in)
            b = b - g_delta
            f_in = self.norm_f(b + x0 + step)
            f_delta = self.f_block(f_in)
            a = a - f_delta
        num = torch.cat([(a - a0).reshape(a.shape[0], -1), (b - b0).reshape(b.shape[0], -1)], dim=-1).norm(dim=-1)
        den = torch.cat([a0.reshape(a0.shape[0], -1), b0.reshape(b0.shape[0], -1)], dim=-1).norm(dim=-1).clamp_min(1e-8)
        return float((num / den).mean().detach().cpu())


class VanillaLoopedControl(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float, seq_len: int, num_classes: int):
        super().__init__()
        self.tok_emb = nn.Embedding(120, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.block = StandardMhaSwiGLUBlock(dim, num_heads, mlp_mult)
        self.norm = RMSNorm(dim)
        self.readout = HaltingReadout(dim, int(num_classes))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, depth: int) -> Tensor:
        x0 = self.tok_emb(tokens) + self.pos_emb[:, : tokens.shape[1]]
        z = x0
        states: list[Tensor] = []
        for _ in range(int(depth)):
            delta = self.block(self.norm(z + x0))
            z = z + delta
            states.append(z)
        return self.readout(states)


@dataclass
class EvalRow:
    K: int
    loss: float
    accuracy: float


@dataclass
class PairRow:
    K_lo: int
    K_hi: int
    G_nll: float
    G_nll_ci_low: float
    G_nll_ci_high: float
    G_acc: float
    G_acc_ci_low: float
    G_acc_ci_high: float
    NDR_epsilon: float
    n_examples: int


def _mean_ci95(total: float, total_sq: float, count: float) -> tuple[float, float, float]:
    """Streaming paired mean and normal-approx 95% CI.

    The harness keeps confidence evidence DDP-friendly by all-reducing sums and
    squared sums.  Offline reports may still bootstrap from saved per-example
    gains, but the executable gate should never report a mean without
    uncertainty.
    """

    if count <= 0:
        # Loud, not silent: an empty sample has no mean/CI. Returning 0.0 here
        # would read as a real "zero gain" measurement in the promotion gate.
        nan = float("nan")
        return nan, nan, nan
    mean = total / count
    if count <= 1:
        return mean, mean, mean
    var = max((total_sq - (total * total) / count) / (count - 1.0), 0.0)
    half_width = 1.96 * math.sqrt(var / count)
    return mean, mean - half_width, mean + half_width


def _model_for_args(args: argparse.Namespace) -> nn.Module:
    num_classes = task_num_classes(args.task)
    if args.variant == "control":
        return VanillaLoopedControl(args.model_dim, args.num_heads, args.mlp_mult, args.seq_len, num_classes)
    return AdditiveCouplingP1Model(
        args.model_dim,
        args.num_heads,
        args.mlp_mult,
        args.seq_len,
        args.variant,
        num_classes,
    )


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _sample_depth(args: argparse.Namespace, step: int) -> int:
    vals = [int(v) for v in args.train_depths]
    if not vals:
        return int(args.depth)
    return vals[step % len(vals)]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    args: argparse.Namespace,
    compose_table: Tensor,
    identity_idx: int,
    device: torch.device,
) -> tuple[list[EvalRow], list[PairRow], float | None]:
    model.eval()
    if torch.cuda.is_available():
        # evaluate() owns the eval-phase activation-memory measurement; callers
        # read torch.cuda.max_memory_allocated() after it returns.
        torch.cuda.reset_peak_memory_stats(device)
    local_rows: list[EvalRow] = []
    k_values = [int(k) for k in args.eval_depths]
    for k in k_values:
        total_loss = torch.zeros((), device=device)
        total_correct = torch.zeros((), device=device)
        total_count = torch.zeros((), device=device)
        for _ in range(int(args.eval_batches)):
            tokens, target = sample_eval_batch(args, compose_table, identity_idx, device)
            logits = model(tokens, k)
            loss_vec = F.cross_entropy(logits, target, reduction="none")
            total_loss += loss_vec.sum()
            total_correct += (logits.argmax(dim=-1) == target).float().sum()
            total_count += target.numel()
        stats = torch.stack([total_loss, total_correct, total_count])
        _all_reduce_sum(stats)
        count = float(stats[2].item())
        if count <= 0:
            raise RuntimeError(f"empty evaluation for K={k}: 0 examples (check --eval-batches/--eval-batch-size)")
        local_rows.append(EvalRow(K=k, loss=float(stats[0].item() / count), accuracy=float(stats[1].item() / count)))

    pair_rows: list[PairRow] = []
    for k_lo, k_hi in args.pairs:
        gain_sum = torch.zeros((), device=device)
        gain_sumsq = torch.zeros((), device=device)
        acc_gain_sum = torch.zeros((), device=device)
        acc_gain_sumsq = torch.zeros((), device=device)
        ndr_sum = torch.zeros((), device=device)
        count_t = torch.zeros((), device=device)
        for _ in range(int(args.eval_batches)):
            tokens, target = sample_eval_batch(args, compose_table, identity_idx, device)
            logits_lo = model(tokens, int(k_lo))
            logits_hi = model(tokens, int(k_hi))
            nll_lo = F.cross_entropy(logits_lo, target, reduction="none")
            nll_hi = F.cross_entropy(logits_hi, target, reduction="none")
            gain = nll_lo - nll_hi
            gain_sum += gain.sum()
            gain_sumsq += gain.square().sum()
            pred_lo = logits_lo.argmax(dim=-1)
            pred_hi = logits_hi.argmax(dim=-1)
            acc_gain = (pred_hi == target).float() - (pred_lo == target).float()
            acc_gain_sum += acc_gain.sum()
            acc_gain_sumsq += acc_gain.square().sum()
            ndr_sum += (gain < -float(args.ndr_epsilon)).float().sum()
            count_t += target.numel()
        stats = torch.stack([gain_sum, gain_sumsq, acc_gain_sum, acc_gain_sumsq, ndr_sum, count_t])
        _all_reduce_sum(stats)
        count = float(stats[5].item())
        if count <= 0:
            raise RuntimeError(f"empty paired evaluation for ({k_lo},{k_hi}): 0 examples (check --eval-batches/--eval-batch-size)")
        gain_mean, gain_low, gain_high = _mean_ci95(float(stats[0].item()), float(stats[1].item()), count)
        acc_mean, acc_low, acc_high = _mean_ci95(float(stats[2].item()), float(stats[3].item()), count)
        pair_rows.append(
            PairRow(
                K_lo=int(k_lo),
                K_hi=int(k_hi),
                G_nll=float(gain_mean),
                G_nll_ci_low=float(gain_low),
                G_nll_ci_high=float(gain_high),
                G_acc=float(acc_mean),
                G_acc_ci_low=float(acc_low),
                G_acc_ci_high=float(acc_high),
                NDR_epsilon=float(stats[4].item() / count),
                n_examples=int(count),
            )
        )

    rec_err: float | None = None
    base_model = _unwrap(model)
    if hasattr(base_model, "reconstruction_error"):
        rec_batch = min(args.eval_batch_size, 16)
        if args.task == "parity":
            tokens = torch.randint(0, 2, (rec_batch, int(args.seq_len)), device=device, dtype=torch.long)
        else:
            tokens, _ = sample_s5_batch(rec_batch, args.seq_len, compose_table, identity_idx, device)
        rec_err = float(base_model.reconstruction_error(tokens, max(k_values)))
    model.train()
    return local_rows, pair_rows, rec_err


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(raw).replace(",", " ").split() if x)


def _parse_pairs(raw: str) -> tuple[tuple[int, int], ...]:
    vals = _parse_int_list(raw)
    if len(vals) % 2 != 0:
        raise argparse.ArgumentTypeError("--pairs requires an even number of integers")
    pairs = tuple((vals[i], vals[i + 1]) for i in range(0, len(vals), 2))
    for lo, hi in pairs:
        if lo <= 0 or hi <= lo:
            raise argparse.ArgumentTypeError("pairs require 0 < K_lo < K_hi")
    return pairs


def _validate_budgets(args: argparse.Namespace) -> None:
    """Reject degenerate budgets shared by both the P1 and feedback runners.

    A zero/empty eval would otherwise produce a fabricated, near-threshold gate
    (``G_nll=0``, CI ``[0, 0]``, ``n_examples=1``).
    """
    for name in ("iterations", "batch_size", "eval_batch_size", "eval_batches"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1 (got {getattr(args, name)})")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["s5", "parity"], default="s5")
    p.add_argument("--variant", choices=["control", "m0", "mclk"], default="m0")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=float, default=2.0)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--train-depths", type=_parse_int_list, default=(16, 32, 64))
    p.add_argument("--eval-depths", type=_parse_int_list, default=(4, 8, 16, 32, 64))
    p.add_argument("--pairs", type=_parse_pairs, default=((16, 64), (8, 32)))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ndr-epsilon", type=float, default=0.0)
    p.add_argument("--output-json", type=Path, default=Path("experiments/training_logs/p1_synthetic_latest.json"))
    args = p.parse_args(argv)
    _validate_budgets(args)
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rank, _local_rank, world_size, device = _init_distributed()
    seed = int(args.seed) + rank * 1009
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    compose_table, identity_idx = build_s5_tables(device)
    model = _model_for_args(args).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start = time.time()
    if _rank0():
        print(
            "p1_synthetic_start:"
            f" variant={args.variant} iterations={args.iterations} world_size={world_size}"
            f" device={device.type} seq_len={args.seq_len} train_depths={tuple(args.train_depths)}",
            flush=True,
        )
    model.train()
    for step in range(1, int(args.iterations) + 1):
        depth = _sample_depth(args, step)
        tokens, target = sample_task_batch(args, compose_table, identity_idx, device)
        logits = model(tokens, depth)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite P1 logits at step {step}")
        task_loss = F.cross_entropy(logits, target)
        if not torch.isfinite(task_loss):
            raise FloatingPointError(f"non-finite P1 task loss at step {step}: {task_loss.detach()}")
        opt.zero_grad(set_to_none=True)
        task_loss.backward()
        if float(args.grad_clip) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()
        if _rank0() and (step == 1 or step % int(args.log_every) == 0 or step == int(args.iterations)):
            acc = (logits.argmax(dim=-1) == target).float().mean()
            print(
                "p1_synthetic_step:"
                f" step={step}/{args.iterations} K={depth}"
                f" loss={float(task_loss.detach().cpu()):.6f}"
                f" acc={float(acc.detach().cpu()):.4f}"
                f" elapsed_s={time.time() - start:.1f}",
                flush=True,
            )

    rows, pairs, rec_err = evaluate(model, args, compose_table, identity_idx, device)
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    params = sum(p.numel() for p in _unwrap(model).parameters())
    summary = {
        "variant": args.variant,
        "iterations": int(args.iterations),
        "task": str(args.task),
        "world_size": int(world_size),
        "seq_len": int(args.seq_len),
        "train_depths": list(map(int, args.train_depths)),
        "eval_depths": list(map(int, args.eval_depths)),
        "pairs": [asdict(p) for p in pairs],
        "k_sweep": [asdict(r) for r in rows],
        "reconstruction_error": rec_err,
        "peak_vram_mb": peak_vram,
        "parameters": int(params),
        "elapsed_s": float(time.time() - start),
    }
    if _rank0():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("p1_synthetic_summary:" + json.dumps(summary, sort_keys=True), flush=True)
        print(f"p1_synthetic_output:{args.output_json}", flush=True)
    _cleanup_distributed()


if __name__ == "__main__":
    main()
