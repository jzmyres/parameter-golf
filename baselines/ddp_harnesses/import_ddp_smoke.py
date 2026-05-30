#!/usr/bin/env python3
"""Tiny DDP smoke used for baseline repositories.

This is intentionally shallow: it imports the upstream checkout and then runs a
2-rank forward/backward/optimizer loop under DistributedDataParallel.  For a few
library-style baselines, it exercises a minimal model from the imported package;
otherwise it verifies that the baseline code can coexist with a DDP training
loop in the active PyTorch environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _add_paths(worktree: Path, baseline_id: str) -> None:
    candidates = [worktree]
    if baseline_id == "deq-locuslab":
        candidates.extend([worktree / "DEQ-Sequence", worktree / "lib"])
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate.resolve()))


def _import_modules(imports: str, worktree: Path) -> list[str]:
    names = [part.strip() for part in imports.split(",") if part.strip()]
    loaded: list[str] = []
    for idx, name in enumerate(names):
        if name.startswith("file:"):
            rel = name.removeprefix("file:")
            module_path = worktree / rel
            if not module_path.exists():
                raise FileNotFoundError(module_path)
            module_name = f"baseline_file_{idx}_{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load module file {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(name)
            continue
        importlib.import_module(name)
        loaded.append(name)
    return loaded


class TinyDenseModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTorchDEQModel(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        from torchdeq import get_deq

        self.core = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)
        self.deq = get_deq(
            f_solver="fixed_point_iter",
            b_solver="fixed_point_iter",
            f_max_iter=3,
            b_max_iter=3,
            f_tol=1e-3,
            b_tol=1e-3,
            grad=1,
            core="indexing",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z0 = torch.zeros_like(x)
        z_out, _info = self.deq(lambda z: torch.tanh(self.core(z) + x), z0)
        if isinstance(z_out, (list, tuple)):
            z = z_out[-1]
        else:
            z = z_out
        return self.head(z)


class TinyMoEUTModel(nn.Module):
    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        from moeut import MoEUTLM

        self.vocab = vocab
        self.model = MoEUTLM(
            vocab,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_head=8,
            ff_n_experts=2,
            att_n_experts=1,
            group_size=1,
            ff_k=1,
            att_k=1,
            ff_expert_size=16,
            dropout=0.0,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model(tokens)
        logits = out.outputs[:, :-1, :].contiguous().view(-1, self.vocab)
        target = tokens[:, 1:].contiguous().view(-1)
        return torch.nn.functional.cross_entropy(logits, target) + out.reg_loss


def _build_model(baseline_id: str) -> tuple[nn.Module, str]:
    if baseline_id == "torchdeq":
        return TinyTorchDEQModel(), "torchdeq_tiny_deq"
    if baseline_id == "moeut":
        return TinyMoEUTModel(), "moeut_tiny_lm"
    return TinyDenseModel(), "generic_import_ddp"


def _init_dist() -> tuple[int, int, torch.device, str]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank % max(torch.cuda.device_count(), 1))
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return rank, world, device, backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--imports", default="")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    worktree = Path(args.worktree)
    if not worktree.exists():
        raise SystemExit(f"missing worktree: {worktree}")

    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")

    _add_paths(worktree, args.baseline_id)
    loaded = _import_modules(args.imports, worktree)
    rank, world, device, backend = _init_dist()
    try:
        model, model_kind = _build_model(args.baseline_id)
        model.to(device)
        # Sparse/conditional baselines (MoEUT routing, DEQ branches) leave some
        # params unused on a given step; without this DDP aborts the reduction.
        ddp = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

        last_loss = None
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            if args.baseline_id == "moeut":
                x = torch.randint(0, 32, (2, 8), device=device)
                loss = ddp(x)
            else:
                x = torch.randn(4, 16 if args.baseline_id != "torchdeq" else 8, device=device)
                pred = ddp(x)
                y = torch.randn_like(pred)
                loss = torch.nn.functional.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            opt.step()
            reduced = loss.detach().clone()
            dist.all_reduce(reduced)
            reduced /= world
            last_loss = float(reduced.item())
            if rank == 0:
                print(f"step:{step}/{args.steps} loss={last_loss:.6f}", flush=True)

        dist.barrier()
        if rank == 0:
            loaded_s = ",".join(loaded) if loaded else "<none>"
            # Distinguish "the baseline's own model was exercised" (torchdeq/moeut)
            # from "the upstream package imported but a generic MLP carried the DDP
            # loop". Overstating the latter as a baseline pass is a false-positive smoke.
            exercised = args.baseline_id in {"torchdeq", "moeut"}
            token = "DDP_BASELINE_MODEL_SMOKE_PASSED" if exercised else "IMPORT_OK_GENERIC_DDP"
            scope = "baseline model exercised" if exercised else "baseline model NOT exercised (generic MLP DDP loop)"
            print(
                f"{token} baseline={args.baseline_id} model={model_kind} scope='{scope}' "
                f"backend={backend} world_size={world} imports={loaded_s} final_loss={last_loss:.6f}",
                flush=True,
            )
    finally:
        # Always tear the group down so a single-rank crash fails fast instead of
        # hanging the peer into a wasted timeout.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
