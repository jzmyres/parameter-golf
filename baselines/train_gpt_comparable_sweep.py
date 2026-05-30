#!/usr/bin/env python3
"""Run train_gpt.py-native 10-step sweeps for comparable recurrent-depth baselines.

This is intentionally separate from ``ddp_smoke_baselines.py``.  The DDP smoke
runner checks whether upstream code can be fetched/imported and whether a tiny
DDP loop works.  This runner answers a stricter comparison question: which
baseline mechanisms can be expressed inside this repository's ``train_gpt.py``
training harness, using the same data loader, optimizer, DDP path, logging, and
artifact path.

External papers that are not implemented in ``train_gpt.py`` are not run here;
the report should call those out explicitly instead of counting synthetic adapter
runs as comparable training results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from _runner_common import classify_failure, gpu_count, report_envelope, run_logged


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "baselines" / "runs" / "train_gpt_comparable"
SUMMARY_JSON = ROOT / "baselines" / "runs" / "train_gpt_comparable_10step_report.json"

TRAIN_STEP_RE = re.compile(r"^step:(\d+)/(\d+) train_loss:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.M)
VAL_STEP_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) val_bpb:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.M)
RUN_ID_RE = re.compile(r"^run_id:(\S+)", re.M)
PEAK_VRAM_RE = re.compile(r"^peak_vram_mb:(\d+)", re.M)
ARTIFACT_RE = re.compile(r"^artifact_bytes:(\d+)", re.M)
TOTAL_BYTES_RE = re.compile(r"^total_bytes:(\d+)", re.M)
CONFIG_RE = re.compile(r"^config:(.*)$", re.M)


@dataclass(frozen=True)
class Variant:
    id: str
    title: str
    category: str
    paper_ids: tuple[str, ...]
    rationale: str
    extra_args: tuple[str, ...] = ()


VARIANTS: tuple[Variant, ...] = (
    Variant(
        id="active-revdeq-parcae",
        title="Active RevDEQ + Parcae + recursive K anchors",
        category="native-current",
        paper_ids=("parcae", "deq-locuslab", "torchdeq", "revdeq", "iso-depth-looped-lm"),
        rationale=(
            "Current train_gpt.py comparator: reversible fixed-point loop, Parcae-style "
            "per-dimension damping/input injection, K-jitter, and recursive prefix-anchor consistency."
        ),
    ),
    Variant(
        id="parcae-fixed-k16",
        title="Parcae fixed-K16 loop",
        category="looped-fixed-depth",
        paper_ids=("parcae", "universal-transformer-t2t", "iso-depth-looped-lm"),
        rationale=(
            "Pins the train-time recurrence depth to K=16 while keeping Parcae damping. "
            "This is the closest train_gpt.py-native proxy for fixed-depth looped/Universal-Transformer comparisons."
        ),
        extra_args=("--deq-k-jitter=0", "--deq-k-max=16", "--deq-k-eval=16"),
    ),
    Variant(
        id="parcae-no-recursive-anchors",
        title="Parcae without recursive prefix anchors",
        category="parcae-ablation",
        paper_ids=("parcae", "iso-depth-looped-lm"),
        rationale=(
            "Keeps the Parcae loop and stochastic K curriculum but removes the local recursive multi-K consistency target. "
            "This isolates the active repo's extra contraction/anchor mechanism from the Parcae-style baseline."
        ),
        extra_args=("--deq-prefix-anchors=0", "--multi-k-consistency-anchor-coef=0"),
    ),
    Variant(
        id="scalar-revdeq-fixed-k16",
        title="Scalar-beta RevDEQ fixed-K16",
        category="deq-family",
        paper_ids=("deq-locuslab", "torchdeq", "revdeq", "universal-transformer-t2t"),
        rationale=(
            "Disables Parcae and uses the scalar DEQ beta path with a fixed K=16 solve. "
            "This is the closest local model to a classic shared-transition DEQ/loop baseline in train_gpt.py."
        ),
        extra_args=(
            "--use-parcae=0",
            "--deq-beta-jitter=0",
            "--deq-k-jitter=0",
            "--deq-k-max=16",
            "--deq-k-eval=16",
            "--deq-prefix-anchors=0",
            "--multi-k-consistency-anchor-coef=0",
        ),
    ),
    Variant(
        id="scalar-revdeq-kjitter",
        title="Scalar-beta RevDEQ with active K-jitter",
        category="deq-family",
        paper_ids=("deq-locuslab", "torchdeq", "revdeq", "iso-depth-looped-lm"),
        rationale=(
            "Disables Parcae while preserving the active stochastic K curriculum. "
            "This separates learned damping from the recurrent-depth exposure schedule."
        ),
        extra_args=(
            "--use-parcae=0",
            "--deq-prefix-anchors=0",
            "--multi-k-consistency-anchor-coef=0",
        ),
    ),
    Variant(
        id="entmax-router-parcae",
        title="Parcae loop with entmax router proxy",
        category="sparse-loop-proxy",
        paper_ids=("moeut", "moeut-training-code", "sparse-looped-moe"),
        rationale=(
            "Uses the implemented entmax routing path with linear scoring while keeping the Parcae loop. "
            "This is only a local sparse-routing proxy, not a paper-faithful MoEUT or Sparse Looped-MoE implementation."
        ),
        extra_args=("--router-scoring=linear", "--use-entmax-routing=1"),
    ),
)


def base_command(nproc: int, steps: int, run_id: str) -> list[str]:
    # Native sweep runs THIS repo's train_gpt.py, which lives in the `opg` env
    # (not the `opgbaselines` orchestration env used for external adapters).
    return [
        "conda", "run", "--no-capture-output", "-n", "opg",
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "train_gpt.py",
        f"--iterations={steps}",
        "--max-training-seconds=0",
        "--val-loss-every=0",
        "--eval-profile=debug",
        "--final-full-validation=0",
        f"--run-id={run_id}",
    ]


def parse_log(log_text: str) -> dict[str, object]:
    train_rows = [
        {"step": int(m.group(1)), "total": int(m.group(2)), "train_loss": float(m.group(3))}
        for m in TRAIN_STEP_RE.finditer(log_text)
    ]
    val_rows = [
        {"step": int(m.group(1)), "total": int(m.group(2)), "val_loss": float(m.group(3)), "val_bpb": float(m.group(4))}
        for m in VAL_STEP_RE.finditer(log_text)
    ]
    run_id = RUN_ID_RE.search(log_text)
    peak_vram = PEAK_VRAM_RE.search(log_text)
    artifact = ARTIFACT_RE.search(log_text)
    total_bytes = TOTAL_BYTES_RE.search(log_text)
    config = CONFIG_RE.search(log_text)
    return {
        "observed_train_steps": max((row["step"] for row in train_rows), default=0),
        "train_log_count": len(train_rows),
        "first_train_loss": train_rows[0]["train_loss"] if train_rows else None,
        "last_train_loss": train_rows[-1]["train_loss"] if train_rows else None,
        "final_val_bpb": val_rows[-1]["val_bpb"] if val_rows else None,
        "final_val_loss": val_rows[-1]["val_loss"] if val_rows else None,
        "run_id": run_id.group(1) if run_id else None,
        "peak_vram_mb": int(peak_vram.group(1)) if peak_vram else None,
        "artifact_bytes": int(artifact.group(1)) if artifact else None,
        "total_bytes": int(total_bytes.group(1)) if total_bytes else None,
        "config": config.group(1).strip() if config else None,
        "profile_skip_ksweep_seen": "PROFILE_SKIP_KSWEEP=1" in log_text,
    }


def run_variant(variant: Variant, *, nproc: int, steps: int, timeout_s: int, dry_run: bool) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"baseline_{variant.id}_{steps}step_{timestamp}"
    log_path = RUN_ROOT / variant.id / f"{timestamp}.log"
    command = base_command(nproc=nproc, steps=steps, run_id=run_id) + list(variant.extra_args)
    result: dict[str, object] = {
        **asdict(variant),
        "expected_steps": steps,
        "nproc": nproc,
        "status": "dry_run" if dry_run else "not_run",
        "command": command,
        "command_display": " ".join(command),
        "log_path": str(log_path),
        "duration_s": 0.0,
        "returncode": None,
        "step_requirement_met": False,
        "detail": "",
    }
    # dry-run preview must work on a GPU-less host, so it precedes the hardware gate.
    if dry_run:
        result["detail"] = "Command not executed."
        return result
    available = gpu_count()
    if available < nproc:
        result["status"] = "blocked_hardware"
        result["detail"] = f"Requires {nproc} GPUs; found {available}."
        return result

    env = os.environ.copy()
    env["PROFILE_SKIP_KSWEEP"] = "1"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    header = [
        "# train_gpt.py comparable baseline sweep",
        f"# variant: {variant.id}",
        f"# title: {variant.title}",
        f"# rationale: {variant.rationale}",
    ]
    returncode, timed_out, duration = run_logged(
        command, cwd=ROOT, log_path=log_path, timeout_s=timeout_s, env=env, header_lines=header
    )
    result["duration_s"] = round(duration, 3)
    result["returncode"] = returncode
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(log_text)
    result.update(parsed)
    result["step_requirement_met"] = int(parsed["observed_train_steps"] or 0) >= steps
    if returncode == 0 and not timed_out and result["step_requirement_met"]:
        result["status"] = "train_gpt_10step_passed"
        result["detail"] = f"train_gpt.py DDP reached {parsed['observed_train_steps']}/{steps} optimizer steps."
    elif returncode == 0 and not timed_out:
        result["status"] = "failed_step_requirement"
        result["detail"] = f"Command exited 0 but only reached {parsed['observed_train_steps']}/{steps} optimizer steps."
    else:
        result["status"] = classify_failure(log_text, timed_out)
        result["detail"] = f"Command exited {returncode}; reached {parsed['observed_train_steps']}/{steps}; see log."
    return result


def load_existing(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: prior report {path} unreadable ({type(exc).__name__}); merging into empty history", file=sys.stderr)
        return {}


def merge_results(existing: dict[str, object], new_results: list[dict[str, object]]) -> dict[str, object]:
    prior = existing.get("results", []) if isinstance(existing.get("results"), list) else []
    by_id: dict[str, dict[str, object]] = {}
    for item in prior:
        if isinstance(item, dict) and item.get("id"):
            by_id[str(item["id"])] = item
    for item in new_results:
        by_id[str(item["id"])] = item
    # Emit known variants in declared order first, then preserve any historical
    # ids that are no longer in VARIANTS so a rename/removal cannot silently drop
    # prior evidence.
    ordered_ids = [v.id for v in VARIANTS if v.id in by_id]
    ordered_ids += [i for i in by_id if i not in set(ordered_ids)]
    return report_envelope(
        runner="baselines/train_gpt_comparable_sweep.py",
        results=[by_id[i] for i in ordered_ids],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[v.id for v in VARIANTS], help="Run one train_gpt.py-compatible variant")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--json-out", default=str(SUMMARY_JSON))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true", help="Replace prior JSON instead of merging by variant id")
    args = parser.parse_args()

    selected = [v for v in VARIANTS if args.variant in (None, v.id)]
    results = [run_variant(v, nproc=args.nproc, steps=args.steps, timeout_s=args.timeout_s, dry_run=args.dry_run) for v in selected]
    out_path = Path(args.json_out)
    existing = {} if args.replace else load_existing(out_path)
    payload = merge_results(existing, results)
    payload["steps"] = args.steps
    payload["nproc"] = args.nproc
    payload["timeout_s"] = args.timeout_s
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['id']}: {result['status']} - {result['detail']}")
    # Stricter gate than ddp_smoke_baselines BY DESIGN: this runner exercises THIS
    # repo's train_gpt.py, so a dependency/OOM failure is a real regression, not an
    # expected external-code "blocked" state. `blocked_hardware` stays non-failing.
    failures = [r for r in results if str(r["status"]).startswith("failed") or str(r["status"]) in {"blocked_compute", "blocked_dependency"}]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
