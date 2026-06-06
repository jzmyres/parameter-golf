#!/usr/bin/env bash
# Audit-test execution gate (CLAUDE.md audit checklist row "Audit-test execution").
#
# Run this script and confirm green BEFORE staging any commit that touches:
#   - _OPTIONAL_COMPONENT_FLAGS / _OPTIONAL_COMPONENT_CAPABILITIES
#   - any `use_X` Hyperparameter
#   - an enforcement-config file (pytest.ini, pyproject.toml, .pre-commit-config.yaml,
#     experiments/run_audit_tests.sh itself)
#   - any symbol listed in tests/test_removal_symmetry.py::REMOVED_NAMES
#
# The audit tests listed below are the AST/grep-driven contract tests whose
# green-light is REQUIRED for the corresponding CLAUDE.md audit row to be
# observed in practice. Adding a new audit test requires adding it to the
# AUDIT_TESTS list below in the same commit; tests/test_audit_test_execution.py
# asserts registry parity between this list and the CLAUDE.md row.
#
# Usage:
#   bash experiments/run_audit_tests.sh          # run all audit tests
#   bash experiments/run_audit_tests.sh --list   # print the registered tests
#
# Exit code 0 on green; non-zero on any failure or missing test file.

set -euo pipefail

# Resolve repo root regardless of working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The audit-test registry. One file per CLAUDE.md audit-checklist row that
# requires execution-time verification. Keep this list and the corresponding
# CLAUDE.md "Audit-test execution" row in sync (enforced by
# tests/test_audit_test_execution.py::test_audit_tests_registry_matches_claude_md).
AUDIT_TESTS=(
    "tests/test_m0_contracts.py"                       # M0 doc-code + removal-symmetry + optimizer + int6 contracts
    "tests/test_m0_hot_path_sync.py"                   # no host-sync in the M0 micro-loop
    "tests/test_enforcement_config_staged.py"          # enforcement-config staging
    "tests/test_audit_test_execution.py"               # this rule's self-enforcement
)

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${AUDIT_TESTS[@]}"
    exit 0
fi

# Verify every registered test file actually exists. A stale entry would
# otherwise cause pytest to fail with a confusing "file not found" — fail
# fast with a clearer message.
missing=()
for t in "${AUDIT_TESTS[@]}"; do
    if [[ ! -f "$REPO_ROOT/$t" ]]; then
        missing+=("$t")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "ERROR: audit-test registry references missing files:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "Update AUDIT_TESTS in experiments/run_audit_tests.sh." >&2
    exit 2
fi

cd "$REPO_ROOT"

# Pass paths via array expansion (`"${AUDIT_TESTS[@]}"`) so each registered
# test path is forwarded to pytest as a single argument — pytest is fine
# with relative paths under the repo root and produces consistent
# reporting. Avoid `exec` so the parent shell survives if the script is
# sourced rather than invoked as a subprocess; `set -e` plus the explicit
# `exit` propagates the non-zero code in the failure path.
echo "Running audit-test suite (${#AUDIT_TESTS[@]} files)..." >&2
pytest -x "${AUDIT_TESTS[@]}"
exit $?
