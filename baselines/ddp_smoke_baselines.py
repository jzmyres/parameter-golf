#!/usr/bin/env python3
"""Launch DDP smoke attempts for baseline worktrees.

The smoke level is explicit in each manifest entry.  PyTorch projects with a
bounded upstream command can use `native_torchrun`; library-style or heavy
training-code repositories use `pytorch_import_adapter`, which imports the
checkout and runs a tiny 2-rank DDP train loop.  Paper-only and non-PyTorch
entries are reported as blocked instead of silently skipped.

Failure-gate policy: this runner exercises *external* code, so a missing
dependency is an expected `blocked_dependency` and does NOT fail the gate; only
`failed*` statuses do (see `main`). The sibling native sweep gates differently
by design — see `_runner_common`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _manifest import find_baseline, load_manifest
from _runner_common import classify_failure, gpu_count, observed_steps, report_envelope, run_logged
from fetch_baselines import is_fetchable


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
WORKTREES = ROOT / "worktrees"
RUNS = ROOT / "runs"
DEFAULT_ADAPTER_ENV = "opgbaselines"
KNOWN_RUNNABLE_MODES = {"local_script", "native_torchrun", "pytorch_import_adapter"}
BLOCKED_MODES = {
    "blocked_no_code",
    "blocked_no_training_code",
    "blocked_non_pytorch_ddp",
    "blocked_dependency",
    "blocked_compute",
    "blocked_data",
}


def command_for(item: dict[str, str], nproc: int, steps: int):
    """Return the launch command (str for shell modes, list for the adapter).

    A list return runs with shell=False (no metacharacter exposure). A str is a
    verbatim manifest `ddp_smoke_command` that genuinely needs a shell. Unknown
    modes return None so the caller can fail loudly instead of treating a typo'd
    mode as a benign "blocked" entry.
    """
    mode = item.get("ddp_mode", "")
    if mode in {"local_script", "native_torchrun"}:
        return item.get("ddp_smoke_command", "") or None
    if mode == "pytorch_import_adapter":
        env_name = (item.get("ddp_env") or DEFAULT_ADAPTER_ENV).strip()
        worktree = WORKTREES / item["id"]
        imports = (item.get("ddp_imports") or "").strip()
        cmd = [
            "conda", "run", "--no-capture-output", "-n", env_name,
            "torchrun", "--standalone", f"--nproc_per_node={nproc}",
            "baselines/ddp_harnesses/import_ddp_smoke.py",
            "--baseline-id", item["id"], "--worktree", str(worktree), "--steps", str(steps),
        ]
        if imports:
            cmd += ["--imports", imports]
        return cmd
    return None


def smoke_one(item: dict[str, str], *, nproc: int, steps: int, timeout_s: int, dry_run: bool) -> dict[str, object]:
    baseline_id = item["id"]
    mode = item.get("ddp_mode", "")
    result: dict[str, object] = {
        "id": baseline_id,
        "title": item.get("title", baseline_id),
        "mode": mode,
        "status": "not_run",
        "command": "",
        "log_path": "",
        "duration_s": 0.0,
        "expected_steps": steps,
        "observed_steps": 0,
        "step_requirement_met": False,
        "detail": "",
    }

    if mode in BLOCKED_MODES:
        result["status"] = mode
        result["detail"] = item.get("ddp_smoke_command", "blocked")
        return result

    if mode not in KNOWN_RUNNABLE_MODES:
        result["status"] = "failed"
        result["detail"] = f"Unknown ddp_mode {mode!r} in manifest (expected one of {sorted(KNOWN_RUNNABLE_MODES)})."
        return result

    # `pytorch_import_adapter` with no imports would train a generic stand-in and
    # certify nothing about the baseline — treat the empty config as a failure,
    # not a silent pass (see the harness's IMPORT_OK_GENERIC_DDP guard too).
    if mode == "pytorch_import_adapter" and not (item.get("ddp_imports") or "").strip():
        result["status"] = "failed"
        result["detail"] = "pytorch_import_adapter requires non-empty ddp_imports; refusing generic-stand-in smoke."
        return result

    command = command_for(item, nproc=nproc, steps=steps)
    if command is None:
        result["status"] = "blocked_no_training_code"
        result["detail"] = "No ddp_smoke_command configured for this mode."
        return result
    result["command"] = command if isinstance(command, str) else " ".join(command)

    # dry-run preview must work even on a GPU-less host, so it precedes the
    # hardware gate (the gate is exactly what defeated --dry-run previously).
    log_path = RUNS / baseline_id / f"ddp_smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    result["log_path"] = str(log_path)
    if dry_run:
        result["status"] = "dry_run"
        result["detail"] = "Command not executed."
        return result

    try:
        required_gpus = int(item.get("ddp_required_gpus", "2") or "2")
    except ValueError:
        result["status"] = "failed"
        result["detail"] = f"Malformed ddp_required_gpus {item.get('ddp_required_gpus')!r} in manifest."
        return result
    available = gpu_count()
    if required_gpus > 0 and available < required_gpus:
        result["status"] = "blocked_hardware"
        result["detail"] = f"Requires {required_gpus} GPUs; found {available}."
        return result

    if mode == "pytorch_import_adapter" and is_fetchable(item) and not (WORKTREES / baseline_id / ".git").exists():
        result["status"] = "failed"
        result["detail"] = "Missing fetched git worktree."
        return result

    code, timed_out, duration = run_logged(
        command, cwd=ROOT.parent, log_path=log_path, timeout_s=timeout_s, shell=isinstance(command, str)
    )
    result["duration_s"] = round(duration, 3)
    full_log_text = log_path.read_text(encoding="utf-8", errors="replace")
    obs_steps = observed_steps(full_log_text)
    result["observed_steps"] = obs_steps
    result["step_requirement_met"] = obs_steps >= steps
    if code == 0 and not timed_out and obs_steps >= steps:
        result["status"] = "ddp_smoke_passed"
        result["detail"] = f"Command exited 0 and reached step {obs_steps}/{steps}."
    elif code == 0 and not timed_out:
        result["status"] = "failed_step_requirement"
        result["detail"] = f"Command exited 0 but reached only step {obs_steps}/{steps}."
    else:
        result["status"] = classify_failure(full_log_text, timed_out)
        result["detail"] = f"Command exited {code}; reached step {obs_steps}/{steps}; see log."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Run one baseline id")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--json-out", default=str(RUNS / "ddp_smoke_report.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_manifest(MANIFEST)
    selected = [find_baseline(items, args.baseline)] if args.baseline else items
    results = [smoke_one(item, nproc=args.nproc, steps=args.steps, timeout_s=args.timeout_s, dry_run=args.dry_run) for item in selected]
    payload = report_envelope(nproc=args.nproc, steps=args.steps, timeout_s=args.timeout_s, results=results)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['id']}: {result['status']} - {result['detail']}")

    # Gate on real failures only; "blocked_*" is an expected state for external code.
    failures = [r for r in results if str(r["status"]).startswith("failed")]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
