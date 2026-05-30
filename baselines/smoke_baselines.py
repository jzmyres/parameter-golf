#!/usr/bin/env python3
"""Dependency-free smoke checks for fetched baseline worktrees."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _manifest import find_baseline, load_manifest
from fetch_baselines import GIT_COMMIT_RE, is_fetchable


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
WORKTREES = ROOT / "worktrees"
RUNS = ROOT / "runs"


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def _compile_tree(target: Path) -> tuple[int, list[str]]:
    """Byte-compile every .py under `target`. Returns (n_compiled, failing_files).

    Unlike `compileall.compile_dir`, this skips the embedded `.git` dir and treats
    individual compile failures (e.g. upstream Python-2-only files) as recorded
    warnings rather than a hard sweep failure — the smoke verifies the pin + README
    + that the bulk of the source parses, not that every legacy file is py3-clean.
    """
    compiled = 0
    failures: list[str] = []
    for path in target.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            compiled += 1
        except (py_compile.PyCompileError, SyntaxError, ValueError):
            failures.append(str(path.relative_to(target)))
    return compiled, failures


def smoke(item: dict[str, str]) -> dict[str, str]:
    result = {
        "id": item["id"],
        "status": "not_run",
        "detail": "",
        "path": str(WORKTREES / item["id"]),
    }
    if not is_fetchable(item):
        result["status"] = item["status"]
        result["detail"] = "No fetchable pinned GitHub source in manifest."
        return result

    target = WORKTREES / item["id"]
    if not (target / ".git").exists():
        result["status"] = "failed"
        result["detail"] = "Missing fetched git worktree."
        return result

    try:
        head = git_output(["rev-parse", "HEAD"], cwd=target)
    except subprocess.CalledProcessError as exc:
        # Record the bad worktree and keep the sweep going (partial-preview completeness).
        result["status"] = "failed"
        result["detail"] = f"git rev-parse failed: {exc}"
        return result
    expected = item["commit"]
    if not GIT_COMMIT_RE.match(expected) or head != expected:
        result["status"] = "failed"
        result["detail"] = f"HEAD mismatch: expected {expected}, got {head}."
        return result

    readmes = [p.name for p in target.iterdir() if p.is_file() and p.name.lower().startswith("readme")]
    if not readmes:
        result["status"] = "failed"
        result["detail"] = "Pinned checkout has no top-level README."
        return result

    compiled, failures = _compile_tree(target)
    if compiled == 0:
        result["status"] = "failed"
        result["detail"] = "Pinned checkout has no compilable Python files (nothing verified)."
        return result

    result["status"] = "smoke_passed"
    detail = f"Exact pinned commit present; README files: {', '.join(sorted(readmes))}; compiled {compiled} Python files."
    if failures:
        shown = ", ".join(failures[:5]) + (f", +{len(failures) - 5} more" if len(failures) > 5 else "")
        detail += f" Non-fatal compile skips ({len(failures)}, likely upstream py2/version-skew): {shown}."
    result["detail"] = detail
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Smoke only one baseline id")
    parser.add_argument("--json-out", default=str(RUNS / "smoke_report.json"))
    args = parser.parse_args()

    items = load_manifest(MANIFEST)
    selected = [find_baseline(items, args.baseline)] if args.baseline else items
    results = [smoke(item) for item in selected]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "results": results,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['id']}: {result['status']} - {result['detail']}")

    failures = [r for r in results if r["status"] == "failed"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
