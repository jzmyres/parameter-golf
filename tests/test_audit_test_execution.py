"""Audit-test execution gate enforcement (CLAUDE.md audit row).

This test enforces the meta-rule that the audit-test suite — the set of
contract tests whose green-light is required for the corresponding CLAUDE.md
audit row to be observed — is consistently registered in three places:

1. `experiments/run_audit_tests.sh` (the executable script contributors run
   before staging a commit, per the "Audit-test execution" CLAUDE.md row).
2. The CLAUDE.md "Audit-test execution" row body (so the rule is readable).
3. Every file listed in (1) and (2) actually exists in the repo.

Without parity, the script and the rule drift: a new audit test could be
authored and added to the script but never named in CLAUDE.md (silent), or
named in CLAUDE.md without being added to the script (unenforceable).
"""
from __future__ import annotations

import os
import re
import stat
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "run_audit_tests.sh"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


_SCRIPT_PARSE_CACHE: list[str] | None = None
_CLAUDE_PARSE_CACHE: list[str] | None = None


def _parse_audit_tests_from_script() -> list[str]:
    """Extract the AUDIT_TESTS bash-array entries from run_audit_tests.sh,
    returned in source order with duplicates preserved so the dedicated
    duplicate-entry test can flag copy-paste mistakes.

    The script defines the array as:
        AUDIT_TESTS=(
            "tests/test_X.py"  # comment
            "tests/test_Y.py"
        )
    This parser is intentionally narrow: it expects double-quoted relative
    paths on their own lines between the opening `AUDIT_TESTS=(` and the
    closing `)`. Comments after the value are allowed. Memoized at module
    scope (mirrors `_TEST_SOURCES_CACHE` in
    `tests/test_optional_component_flag_contract.py:128`) so 4 test methods
    don't trigger 4 file reads + regex passes."""
    global _SCRIPT_PARSE_CACHE
    if _SCRIPT_PARSE_CACHE is not None:
        return _SCRIPT_PARSE_CACHE
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    m = re.search(r"AUDIT_TESTS=\((.*?)\)", text, flags=re.DOTALL)
    if not m:
        raise AssertionError(
            "AUDIT_TESTS array not found in experiments/run_audit_tests.sh"
        )
    body = m.group(1)
    raw: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match "tests/test_foo.py"  -> tests/test_foo.py
        match = re.match(r'^"([^"]+)"', line)
        if match:
            raw.append(match.group(1))
    _SCRIPT_PARSE_CACHE = raw
    return raw


_BEGIN_MARKER = "<!-- audit-tests-registry-begin -->"
_END_MARKER = "<!-- audit-tests-registry-end -->"


_AUDIT_ROW_HEADER = "- **Audit-test execution.**"


def _parse_audit_tests_from_claude_md() -> list[str]:
    """Extract `tests/test_*.py` filenames from the canonical registry line
    inside the CLAUDE.md "Audit-test execution" audit-checklist row.

    The registry is delimited by HTML-comment markers so the scope is
    unambiguous: matching anything inside the row body but outside the
    markers would incorrectly enrol unrelated test mentions (e.g. an
    incident write-up that references a sibling contract test). The markers
    are deliberately part of the row text — a future contributor cannot
    accidentally rewrite the row in a way that loses the registry without
    also tripping this parser.

    Scope assertion: the markers MUST appear *after* the "Audit-test
    execution" row header. Otherwise a future contributor could move the
    markers into a sibling row (e.g. "Audit-row executability") and parity
    would silently keep passing while the registry referred to the wrong
    rule. The header-before-marker invariant pins the markers to the
    correct row even if line numbers shift.

    Memoized at module scope (CLAUDE.md is ~47 KB; 4 test methods would
    otherwise read + parse it independently)."""
    global _CLAUDE_PARSE_CACHE
    if _CLAUDE_PARSE_CACHE is not None:
        return _CLAUDE_PARSE_CACHE
    text = CLAUDE_MD.read_text(encoding="utf-8")
    header = text.find(_AUDIT_ROW_HEADER)
    if header == -1:
        raise AssertionError(
            f"CLAUDE.md must contain the row header {_AUDIT_ROW_HEADER!r} "
            "(the canonical anchor for the Audit-test execution audit row)."
        )
    begin = text.find(_BEGIN_MARKER)
    end = text.find(_END_MARKER)
    if begin == -1 or end == -1 or end < begin:
        raise AssertionError(
            "CLAUDE.md must contain the registry markers "
            f"{_BEGIN_MARKER!r} and {_END_MARKER!r} (in that order) inside "
            "the Audit-test execution audit-checklist row."
        )
    if begin < header:
        raise AssertionError(
            "Registry markers appear BEFORE the Audit-test execution row "
            f"header (header at offset {header}, begin marker at offset "
            f"{begin}). The markers must be inside the row whose rule they "
            "enforce — move them after the row header."
        )
    # Also assert the markers are inside THIS row (not pushed past the next
    # audit-row delimiter). Find the next list-item delimiter after the
    # header; markers must be before it.
    next_row = re.search(r"\n- \*\*", text[header + len(_AUDIT_ROW_HEADER):])
    if next_row is not None:
        next_row_offset = header + len(_AUDIT_ROW_HEADER) + next_row.start()
        if end > next_row_offset:
            raise AssertionError(
                "Registry markers extend past the Audit-test execution row "
                f"into a sibling row at offset {next_row_offset}. The "
                "markers must be inside the row whose rule they enforce."
            )
    span = text[begin + len(_BEGIN_MARKER):end]
    found = re.findall(r"tests/test_\w+\.py", span)
    if not found:
        raise AssertionError(
            f"No `tests/test_*.py` filename inside the CLAUDE.md registry "
            f"markers — got: {span!r}"
        )
    _CLAUDE_PARSE_CACHE = sorted(set(found))
    return _CLAUDE_PARSE_CACHE


class TestAuditTestExecution(unittest.TestCase):
    def test_run_audit_tests_script_exists(self) -> None:
        self.assertTrue(
            SCRIPT_PATH.is_file(),
            f"experiments/run_audit_tests.sh must exist at {SCRIPT_PATH}",
        )

    def test_run_audit_tests_script_is_executable(self) -> None:
        # The script must have its executable bit set so `bash` is not
        # required to invoke it; this also signals to contributors that the
        # file is a runnable entry point, not a library.
        mode = SCRIPT_PATH.stat().st_mode
        self.assertTrue(
            mode & stat.S_IXUSR,
            f"{SCRIPT_PATH} must be executable (chmod +x); current mode={oct(mode)}",
        )

    def test_run_audit_tests_has_bash_shebang(self) -> None:
        first_line = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(
            first_line.startswith("#!/usr/bin/env bash") or first_line.startswith("#!/bin/bash"),
            f"{SCRIPT_PATH} must declare a bash shebang; got {first_line!r}",
        )

    def test_every_registered_audit_test_file_exists(self) -> None:
        missing: list[str] = []
        for rel in _parse_audit_tests_from_script():
            if not (REPO_ROOT / rel).is_file():
                missing.append(rel)
        self.assertFalse(
            missing,
            msg=(
                "AUDIT_TESTS in experiments/run_audit_tests.sh references "
                f"missing files: {missing}. Either add the test or remove "
                f"the entry."
            ),
        )

    def test_audit_tests_registry_matches_claude_md(self) -> None:
        """Every audit-test file named in CLAUDE.md's 'Audit-test execution'
        row body must also appear in AUDIT_TESTS, and vice versa. Adding a
        new audit test requires updating both surfaces in the same commit."""
        script_tests = set(_parse_audit_tests_from_script())
        claude_tests = set(_parse_audit_tests_from_claude_md())
        only_in_script = script_tests - claude_tests
        only_in_claude = claude_tests - script_tests
        self.assertFalse(
            only_in_script or only_in_claude,
            msg=(
                "Audit-test registry drift between CLAUDE.md and "
                "experiments/run_audit_tests.sh:\n"
                f"  - Only in script:  {sorted(only_in_script)}\n"
                f"  - Only in CLAUDE.md: {sorted(only_in_claude)}\n"
                "Both surfaces must name the same set of audit tests."
            ),
        )

    def test_audit_tests_registry_is_non_empty(self) -> None:
        # Defensive: if AUDIT_TESTS gets accidentally truncated to (), the
        # gate becomes a no-op. Pin to the current population so that a
        # deletion is forced to update this floor explicitly (the comment
        # documents WHY each entry is in the registry).
        entries = _parse_audit_tests_from_script()
        self.assertGreaterEqual(
            len(entries), 4,
            "AUDIT_TESTS appears to have been truncated or emptied — "
            f"got {entries}. The gate must enforce at least the four "
            "core contracts: optional-component-flag, removal-symmetry, "
            "enforcement-config-staged, audit-test-execution-self-check.",
        )

    def test_marker_string_constants_match_claude_md_literals(self) -> None:
        """Direct rename-drift gate: the test's `_BEGIN_MARKER` and
        `_END_MARKER` Python constants must literally appear in CLAUDE.md.
        Without this, a future rename of just the constants (or just the
        markdown text) would leave `_parse_audit_tests_from_claude_md`
        failing with a less-actionable "markers not found" message — this
        test pins the rename failure to the exact mismatch."""
        text = CLAUDE_MD.read_text(encoding="utf-8")
        for marker in (_BEGIN_MARKER, _END_MARKER):
            self.assertIn(
                marker, text,
                msg=(
                    f"The Python constant {marker!r} (used by "
                    f"`_parse_audit_tests_from_claude_md`) must appear "
                    "literally in CLAUDE.md. If you renamed one, rename "
                    "both in the same edit."
                ),
            )

    def test_audit_tests_registry_has_no_duplicates(self) -> None:
        """Script `AUDIT_TESTS` array must not contain duplicate entries —
        each entry implies a distinct CLAUDE.md row, so duplicates are a
        copy-paste mistake, not a meaningful "run-this-twice" directive.
        Documented in `_parse_audit_tests_from_script`'s docstring."""
        entries = _parse_audit_tests_from_script()
        seen: dict[str, int] = {}
        for e in entries:
            seen[e] = seen.get(e, 0) + 1
        duplicates = {k: v for k, v in seen.items() if v > 1}
        self.assertFalse(
            duplicates,
            msg=(
                "AUDIT_TESTS contains duplicate entries — each must be "
                f"unique: {duplicates}. Remove the duplicate or rename if "
                "two distinct tests share a file name."
            ),
        )


if __name__ == "__main__":
    unittest.main()
