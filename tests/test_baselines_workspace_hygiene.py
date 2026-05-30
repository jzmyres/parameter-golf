"""Witnesses for the baselines/ workspace audit rows.

Covers three EXPERIENCE.md §1 rules added 2026-05-30:
  - #vendored-workspace-hygiene        (heavy subtrees gitignored)
  - #committed-report-tracked-evidence (report numbers backed by a tracked snapshot)
  - #smoke-must-exercise-target        (no false-positive smokes; terminal-keyed classifier)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASELINES = REPO / "baselines"
SNAPSHOT = BASELINES / "runs_snapshot_2026-05-24"

sys.path.insert(0, str(BASELINES))


def _is_ignored(rel: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(REPO)).returncode == 0


# ---- #vendored-workspace-hygiene -------------------------------------------------

@pytest.mark.parametrize("rel", ["baselines/worktrees", "baselines/runs", "baselines/.envs"])
def test_heavy_subtrees_are_gitignored(rel):
    assert _is_ignored(rel + "/probe"), f"{rel} must be gitignored so third-party/volatile content cannot be committed"


def test_committable_set_excludes_worktrees_and_runs():
    staged = subprocess.check_output(["git", "add", "-An", "baselines"], cwd=str(REPO), text=True)
    for bad in ("worktrees/", "runs/", ".envs/", "__pycache__"):
        assert bad not in staged, f"git add dry-run would stage {bad!r}:\n{staged}"


# ---- #committed-report-tracked-evidence -----------------------------------------

def test_report_evidence_snapshot_is_present_and_trackable():
    report = (BASELINES / "REPLICATION_REPORT.md").read_text(encoding="utf-8")
    assert "runs_snapshot_2026-05-24" in report, "report must point at the tracked evidence snapshot"
    assert SNAPSHOT.is_dir(), "frozen evidence snapshot dir must exist"
    files = list(SNAPSHOT.glob("*.json"))
    assert files, "snapshot must contain at least one evidence JSON"
    for f in files:
        assert not _is_ignored(str(f.relative_to(REPO))), f"snapshot file {f.name} must NOT be gitignored"


# ---- #smoke-must-exercise-target ------------------------------------------------

def test_adapter_smoke_refuses_empty_imports():
    import ddp_smoke_baselines as d

    item = {"id": "x", "title": "X", "ddp_mode": "pytorch_import_adapter", "ddp_imports": "", "status": "s"}
    r = d.smoke_one(item, nproc=2, steps=2, timeout_s=10, dry_run=False)
    assert r["status"] == "failed", r


def test_unknown_ddp_mode_is_loud_failure():
    import ddp_smoke_baselines as d

    item = {"id": "y", "title": "Y", "ddp_mode": "typo_mode", "status": "s"}
    r = d.smoke_one(item, nproc=2, steps=2, timeout_s=10, dry_run=False)
    assert r["status"] == "failed", r


def test_classify_failure_keys_on_terminal_lines():
    from _runner_common import classify_failure

    # "ImportError" appears mid-log but the terminal error is unrelated -> not blocked.
    mid = "ImportError happened earlier but recovered\n" + ("noise\n" * 60) + "RuntimeError: CUDA assert\n"
    assert classify_failure(mid, timeout=False) == "failed"
    # Terminal traceback is a genuine dependency miss -> blocked_dependency.
    dep = "training...\n" + ("step\n" * 60) + "ModuleNotFoundError: No module named 'foo'\n"
    assert classify_failure(dep, timeout=False) == "blocked_dependency"
    assert classify_failure("anything", timeout=True) == "failed_timeout"


def test_generic_adapter_uses_scope_explicit_token():
    src = (BASELINES / "ddp_harnesses" / "import_ddp_smoke.py").read_text(encoding="utf-8")
    assert "IMPORT_OK_GENERIC_DDP" in src, "generic-model runs must use a scope-explicit success token"
    assert "DDP_IMPORT_SMOKE_PASSED" not in src, "the ambiguous blanket pass token must be gone"
