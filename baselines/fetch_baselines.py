#!/usr/bin/env python3
"""Fetch pinned external baseline repositories into ignored worktrees."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from _manifest import find_baseline, load_manifest


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
WORKTREES = ROOT / "worktrees"
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def is_fetchable(item: dict[str, str]) -> bool:
    return item["code_url"].startswith("https://github.com/") and GIT_COMMIT_RE.match(item["commit"]) is not None


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _fetch_commit(target: Path, commit: str, dry_run: bool) -> None:
    """Fetch a single pinned SHA, with a clear error if the server refuses it.

    Many GitHub repos disallow fetching an unadvertised SHA at depth 1; surface
    that as an actionable message instead of a downstream "pathspec did not match".
    """
    try:
        run(["git", "fetch", "--depth=1", "origin", commit], cwd=target, dry_run=dry_run)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"fetch of pinned SHA {commit} failed for {target.name}: the server may not allow "
            f"fetching an unadvertised object at depth 1. Try a full clone for this baseline."
        ) from exc


def fetch(item: dict[str, str], dry_run: bool = False) -> None:
    target = WORKTREES / item["id"]
    commit = item["commit"]
    url = item["code_url"]
    if not is_fetchable(item):
        print(f"skip {item['id']}: no fetchable pinned GitHub source")
        return

    WORKTREES.mkdir(parents=True, exist_ok=True)
    if (target / ".git").exists():
        print(f"update {item['id']} -> {commit}")
    else:
        print(f"clone {item['id']} -> {commit}")
        run(["git", "clone", "--no-tags", "--depth=1", url, str(target)], dry_run=dry_run)
    _fetch_commit(target, commit, dry_run=dry_run)
    run(["git", "checkout", "--detach", commit], cwd=target, dry_run=dry_run)

    # Verify the checkout actually landed on the pinned SHA so a wrong/stale
    # worktree never silently persists for a later tool to misread.
    if not dry_run:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(target), text=True).strip()
        if head != commit:
            raise SystemExit(f"checkout verification failed for {item['id']}: HEAD={head} != pinned {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Fetch only one baseline id")
    parser.add_argument("--dry-run", action="store_true", help="Print git commands without running them")
    parser.add_argument("--list", action="store_true", help="List fetchable baseline ids")
    args = parser.parse_args()

    items = load_manifest(MANIFEST)
    if args.list:
        for item in items:
            if is_fetchable(item):
                print(item["id"])
        return

    selected = [find_baseline(items, args.baseline)] if args.baseline else [item for item in items if is_fetchable(item)]
    for item in selected:
        fetch(item, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
