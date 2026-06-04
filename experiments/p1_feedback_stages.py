"""Later-stage diagnostics for the OPG final-minimal feedback plan.

This file is intentionally separate from ``experiments/p1_synthetic.py``.
The P1 runner answers only the core detectability question with
``control``, ``m0``, and ``mclk``.  This runner covers the feedback stages
that are useful only after, or alongside, that core result:

* S+1: MoE/static-MoE mechanism diagnostics with plain routing.
* S+2: hidden-state cache memory proxy, explicitly not an LM KV proof.
* S+3: fake-int6 deployment sensitivity on the trained S+0 model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from experiments.p1_synthetic import (
        AdditiveCouplingP1Model,
        HaltingReadout,
        RMSNorm,
        SwiGLU,
        _all_reduce_sum,
        _cleanup_distributed,
        _init_distributed,
        _parse_int_list,
        _parse_pairs,
        _rank0,
        _sample_depth,
        _unwrap,
        _validate_budgets,
        build_s5_tables,
        evaluate,
        sample_eval_batch,
        sample_s5_batch,
        sample_task_batch,
        task_num_classes,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - script execution path
    # Only fall back for the package-resolution case; a genuinely missing
    # dependency inside p1_synthetic must surface, not be masked as "no experiments".
    if exc.name not in (None, "experiments", "experiments.p1_synthetic"):
        raise
    from p1_synthetic import (  # type: ignore
        AdditiveCouplingP1Model,
        HaltingReadout,
        RMSNorm,
        SwiGLU,
        _all_reduce_sum,
        _cleanup_distributed,
        _init_distributed,
        _parse_int_list,
        _parse_pairs,
        _rank0,
        _sample_depth,
        _unwrap,
        _validate_budgets,
        build_s5_tables,
        evaluate,
        sample_eval_batch,
        sample_s5_batch,
        sample_task_batch,
        task_num_classes,
    )


class SoftMoEFFN(nn.Module):
    """Plain dense-soft MoE feed-forward block for S+1 diagnostics.

    The router is intentionally simple: no Dirichlet-UCB, no batch assignment,
    no occupancy matching.  A static route ablation freezes the route across
    depth by replacing token logits with their current batch mean.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, static_route: bool, top_r: int = 0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.static_route = bool(static_route)
        self.top_r = int(top_r)
        self.router = nn.Linear(dim, self.num_experts)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(self.num_experts)])
        self.last_usage: Tensor | None = None
        self.last_aux_loss: Tensor | None = None

    def _route(self, x: Tensor) -> Tensor:
        logits = self.router(x)
        if self.static_route:
            logits = logits.mean(dim=(0, 1), keepdim=True).expand_as(logits)
        if 0 < self.top_r < self.num_experts:
            # NOTE: hard top-k is a NON-reversible mechanism-diagnostic knob
            # (off by default, top_r=0). It deliberately sits outside the RevDEQ
            # soft-dense-routing contract and is used only for S+1 routing analysis.
            values, index = logits.topk(self.top_r, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            logits = sparse_logits.scatter(dim=-1, index=index, src=values)
        return F.softmax(logits, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        route = self._route(x)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        out = (route.unsqueeze(-1) * expert_out).sum(dim=-2)
        usage = route.mean(dim=(0, 1))
        self.last_usage = usage.detach()
        self.last_aux_loss = float(self.num_experts) * usage.square().sum() - 1.0
        return out


class MoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: float,
        num_experts: int,
        static_route: bool,
        top_r: int = 0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        hidden_dim = max(8, int(round(float(mlp_mult) * dim)))
        self.ffn = SoftMoEFFN(dim, hidden_dim, num_experts, static_route, top_r=top_r)

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        y = x + attn_out
        return attn_out + self.ffn(self.mlp_norm(y))

    def aux_loss(self) -> Tensor:
        if self.ffn.last_aux_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.ffn.last_aux_loss

    def usage(self) -> Tensor | None:
        return self.ffn.last_usage


class MoEAdditiveCouplingModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: float,
        seq_len: int,
        num_classes: int,
        num_experts: int,
        static_route: bool,
        top_r: int = 0,
    ):
        super().__init__()
        self.variant = "static_moe" if static_route else "moe"
        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.tok_emb = nn.Embedding(120, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.f_block = MoEBlock(dim, num_heads, mlp_mult, num_experts, static_route, top_r=top_r)
        self.g_block = MoEBlock(dim, num_heads, mlp_mult, num_experts, static_route, top_r=top_r)
        self.norm_f = RMSNorm(dim)
        self.norm_g = RMSNorm(dim)
        self.readout = HaltingReadout(dim, int(num_classes))
        self.last_aux_loss: Tensor | None = None
        self.last_route_by_depth: Tensor | None = None
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _x0(self, tokens: Tensor) -> Tensor:
        return self.tok_emb(tokens) + self.pos_emb[:, : tokens.shape[1]]

    def recurrent_states(self, tokens: Tensor, depth: int, return_terminal: bool = False):
        x0 = self._x0(tokens)
        a = x0
        b = x0
        states: list[Tensor] = []
        aux_terms: list[Tensor] = []
        route_by_depth: list[Tensor] = []
        for _ in range(int(depth)):
            f_delta = self.f_block(self.norm_f(b + x0))
            a = a + f_delta
            aux_terms.append(self.f_block.aux_loss())
            g_delta = self.g_block(self.norm_g(a + x0))
            b = b + g_delta
            aux_terms.append(self.g_block.aux_loss())
            f_usage = self.f_block.usage()
            g_usage = self.g_block.usage()
            if f_usage is not None and g_usage is not None:
                route_by_depth.append(0.5 * (f_usage + g_usage))
            states.append(0.5 * (a + b))
        if aux_terms:
            self.last_aux_loss = torch.stack(aux_terms).mean()
        else:
            self.last_aux_loss = x0.new_zeros(())
        if route_by_depth:
            self.last_route_by_depth = torch.stack(route_by_depth)
        else:
            self.last_route_by_depth = None
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
        for _ in reversed(range(int(depth))):
            g_delta = self.g_block(self.norm_g(a + x0))
            b = b - g_delta
            f_delta = self.f_block(self.norm_f(b + x0))
            a = a - f_delta
        num = torch.cat([(a - a0).reshape(a.shape[0], -1), (b - b0).reshape(b.shape[0], -1)], dim=-1).norm(dim=-1)
        den = torch.cat([a0.reshape(a0.shape[0], -1), b0.reshape(b0.shape[0], -1)], dim=-1).norm(dim=-1).clamp_min(1e-8)
        return float((num / den).mean().detach().cpu())


def _model_for_args(args: argparse.Namespace) -> nn.Module:
    num_classes = task_num_classes(args.task)
    if args.variant == "m0":
        return AdditiveCouplingP1Model(
            args.model_dim,
            args.num_heads,
            args.mlp_mult,
            args.seq_len,
            "m0",
            num_classes,
        )
    return MoEAdditiveCouplingModel(
        args.model_dim,
        args.num_heads,
        args.mlp_mult,
        args.seq_len,
        num_classes,
        args.num_experts,
        static_route=args.variant == "static_moe",
        top_r=args.router_top_r,
    )


def _entropy(probs: Tensor) -> Tensor:
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum()


def _route_depth_nmi(route_by_depth: Tensor) -> float:
    joint = route_by_depth.clamp_min(0)
    total = joint.sum().clamp_min(1e-12)
    joint = joint / total
    p_k = joint.sum(dim=1, keepdim=True)
    p_e = joint.sum(dim=0, keepdim=True)
    expected = (p_k * p_e).clamp_min(1e-12)
    mi = (joint * (joint.clamp_min(1e-12) / expected).log()).sum()
    h_k = _entropy(p_k.squeeze(1))
    h_e = _entropy(p_e.squeeze(0))
    denom = torch.sqrt((h_k * h_e).clamp_min(1e-12))
    return float((mi / denom).detach().cpu())


def _effective_rank(matrix: Tensor) -> float:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if centered.numel() == 0 or centered.norm() <= 0:
        return 1.0
    singular = torch.linalg.svdvals(centered.float())
    weights = singular.square()
    total = weights.sum()
    if total <= 0:
        return 1.0
    probs = weights / total
    return float(torch.exp(_entropy(probs)).detach().cpu())


@torch.no_grad()
def _mechanism_diagnostics(
    model: nn.Module,
    args: argparse.Namespace,
    compose_table: Tensor,
    identity_idx: int,
    device: torch.device,
) -> dict[str, float | str] | None:
    base = _unwrap(model)
    if not hasattr(base, "last_route_by_depth"):
        return None
    model.eval()
    tokens, _ = sample_eval_batch(args, compose_table, identity_idx, device)
    model(tokens, max(map(int, args.eval_depths)))
    route = getattr(base, "last_route_by_depth", None)
    if route is None:
        model.train()
        return None
    route = route.detach().float()
    _all_reduce_sum(route)
    if dist.is_available() and dist.is_initialized():
        route = route / float(dist.get_world_size())
    usage = route.mean(dim=0)
    expert_util = torch.exp(_entropy(usage / usage.sum().clamp_min(1e-12))) / float(route.shape[1])
    nmi = _route_depth_nmi(route)
    aebr = _effective_rank(route)
    model.train()
    return {
        "kind": "s1_moe_mechanism_diagnostic",
        "route_depth_nmi": float(nmi),
        "AEBR": float(aebr),
        "expert_utilization": float(expert_util.detach().cpu()),
        "shuffle_baseline_note": "route-depth diagnostics are mechanism-only and not P1 gates",
    }


def _cache_policy_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    elem_bytes = 4
    one_hidden_state = int(args.eval_batch_size) * int(args.seq_len) * int(args.model_dim) * elem_bytes
    rows: list[dict[str, int]] = []
    for k in args.eval_depths:
        depth = int(k)
        rows.append(
            {
                "K": depth,
                "exact_multi_depth_hidden_bytes": one_hidden_state * depth,
                "terminal_hidden_bytes": one_hidden_state,
                "shared_hidden_bytes": one_hidden_state,
            }
        )
    return {
        "kind": "hidden_state_proxy_not_autoregressive_kv",
        "bytes_per_terminal_hidden_state": one_hidden_state,
        "alpha_exact_multi_depth_proxy": 1.0,
        "alpha_terminal_hidden_proxy": 0.0,
        "alpha_shared_hidden_proxy": 0.0,
        "quality_gap": None,
        "quality_gap_status": "not_tested_requires_autoregressive_terminal_context_attention",
        "rows": rows,
    }


def _quantize_linear_int6_(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                if not torch.isfinite(weight).all():
                    raise FloatingPointError("non-finite weight before int6 quantization (S+3 gap would be meaningless)")
                scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 31.0
                q = torch.round(weight / scale).clamp(-31, 31)
                module.weight.data.copy_(q * scale)


def _best_loss(rows: list[object]) -> float:
    return min(float(getattr(r, "loss")) for r in rows) if rows else float("nan")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["s1", "s2", "s3"], required=True)
    p.add_argument("--task", choices=["s5", "parity"], default="s5")
    p.add_argument("--variant", choices=["m0", "moe", "static_moe"], default="m0")
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
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--router-top-r", type=int, default=0)
    p.add_argument("--moe-balance-coef", type=float, default=0.01)
    p.add_argument("--output-json", type=Path, default=Path("experiments/training_logs/p1_feedback_stage_latest.json"))
    args = p.parse_args(argv)
    if args.stage in {"s2", "s3"} and args.variant != "m0":
        p.error("S+2 and S+3 diagnostics use --variant m0; MoE belongs to S+1")
    _validate_budgets(args)
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
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
            "p1_feedback_stage_start:"
            f" stage={args.stage} variant={args.variant} iterations={args.iterations}"
            f" world_size={world_size} device={device.type}",
            flush=True,
        )
    model.train()
    for step in range(1, int(args.iterations) + 1):
        depth = _sample_depth(args, step)
        tokens, target = sample_task_batch(args, compose_table, identity_idx, device)
        logits = model(tokens, depth)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite feedback-stage logits at step {step}")
        task_loss = F.cross_entropy(logits, target)
        base = _unwrap(model)
        aux_loss = getattr(base, "last_aux_loss", None)
        if aux_loss is None:
            aux_loss = task_loss.new_zeros(())
        loss = task_loss + float(args.moe_balance_coef) * aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite feedback-stage loss at step {step}: {loss.detach()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()
        if _rank0() and (step == 1 or step % int(args.log_every) == 0 or step == int(args.iterations)):
            acc = (logits.argmax(dim=-1) == target).float().mean()
            print(
                "p1_feedback_stage_step:"
                f" step={step}/{args.iterations} stage={args.stage} variant={args.variant} K={depth}"
                f" loss={float(loss.detach().cpu()):.6f} task_loss={float(task_loss.detach().cpu()):.6f}"
                f" acc={float(acc.detach().cpu()):.4f} elapsed_s={time.time() - start:.1f}",
                flush=True,
            )

    rows, pairs, rec_err = evaluate(model, args, compose_table, identity_idx, device)
    mechanism = _mechanism_diagnostics(model, args, compose_table, identity_idx, device) if args.stage == "s1" else None
    cache_diagnostics = _cache_policy_diagnostics(args) if args.stage == "s2" else None
    int6_rows = None
    int6_pairs = None
    quant_gap = None
    if args.stage == "s3":
        quantized = copy.deepcopy(_unwrap(model)).to(device)
        _quantize_linear_int6_(quantized)
        int6_rows, int6_pairs, _ = evaluate(quantized, args, compose_table, identity_idx, device)
        quant_gap = float(_best_loss(int6_rows) - _best_loss(rows))

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    params = sum(p.numel() for p in _unwrap(model).parameters())
    summary: dict[str, object] = {
        "stage": args.stage,
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
    if mechanism is not None:
        summary["mechanism_diagnostics"] = mechanism
        summary["expert_utilization"] = mechanism["expert_utilization"]
    if cache_diagnostics is not None:
        summary["cache_diagnostics"] = cache_diagnostics
    if int6_rows is not None and int6_pairs is not None:
        summary["int6_k_sweep"] = [asdict(r) for r in int6_rows]
        summary["int6_pairs"] = [asdict(p) for p in int6_pairs]
        summary["quantization_gap_best_loss"] = quant_gap

    if _rank0():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("p1_feedback_stage_summary:" + json.dumps(summary, sort_keys=True), flush=True)
        print(f"p1_feedback_stage_output:{args.output_json}", flush=True)
    _cleanup_distributed()


if __name__ == "__main__":
    main()
