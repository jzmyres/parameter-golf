"""Executable witness for the `Enforcement-config staging` audit-checklist row.

If a project-level enforcement config (pytest.ini, pyproject.toml lint sections,
pre-commit hooks, etc.) exists on disk but is not git-tracked, the invariant it
enforces is silently disarmed on the next contributor's machine and on CI restart.
This test fails when that condition is true, converting the manual verification
recipe in CLAUDE.md / EXPERIENCE.md into an automated gate.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Add new enforcement-config paths here; absent files auto-skip.
ENFORCEMENT_CONFIGS: tuple[str, ...] = (
    "pytest.ini",
)


class TestEnforcementConfigsTracked(unittest.TestCase):
    def test_enforcement_configs_are_git_tracked(self) -> None:
        if not (REPO_ROOT / ".git").exists():
            self.skipTest("not running inside a git checkout")
        for name in ENFORCEMENT_CONFIGS:
            if not (REPO_ROOT / name).exists():
                continue
            try:
                subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", name],
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                self.fail(
                    f"{name!r} exists on disk but is not git-tracked — the "
                    f"invariant it enforces will be silently disarmed on a "
                    f"fresh clone or CI restart."
                )


if __name__ == "__main__":
    unittest.main()
