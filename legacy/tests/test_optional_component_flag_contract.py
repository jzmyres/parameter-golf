"""Flag-to-effect contract gate (CLAUDE.md audit checklist).

For each entry in `_OPTIONAL_COMPONENT_FLAGS`, assert that flipping the flag
True either (a) raises SystemExit in `_validate_hyperparameters` with a
message naming the field, or (b) has at least one test in tests/ or
experiments/ whose source code references the field name AND a `=True`
construction. This pins the registry against silent no-op flags.

The witness scan is lexical (AST-light), not runtime, so it stays cheap and
runnable in the pre-commit chain.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from legacy.tests._helpers import mutate_hyperparameters as _mut  # noqa: E402
from legacy.train_gpt_rich import (  # noqa: E402
    Hyperparameters,
    _OPTIONAL_COMPONENT_CAPABILITIES,
    _OPTIONAL_COMPONENT_FLAGS,
    _validate_hyperparameters,
)


def _validator_rejects(field: str) -> bool:
    """Return True if `_validate_hyperparameters` raises SystemExit when
    `field=True` and the exception message names the field. Other validation
    failures (e.g. an unrelated invariant tripping first) do NOT count."""
    try:
        _validate_hyperparameters(_mut(**{field: True}))
    except SystemExit as e:
        return field in str(e)
    return False


def _ast_has_field_true(tree: ast.AST, field: str) -> bool:
    """Walk an AST and return True iff some node sets `field` to literal True
    in a way the runtime would observe.  Recognized shapes:

      * keyword arg in a function call:  `f(..., field=True, ...)`
      * attribute assignment:            `obj.field = True`  (e.g. Hyperparameters)
      * mapping literal entry:           `{"field": True}`
      * `setattr(obj, "field", True)`

    String literals, comments, decorators, and `field=True` references inside
    skipped tests (`@unittest.skip`) are deliberately excluded.
    """
    skip_decorators = {"skip", "skipIf", "skipUnless", "expectedFailure"}

    def _is_true(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is True

    def _is_skipped(func: ast.FunctionDef) -> bool:
        for dec in func.decorator_list:
            base = dec.func if isinstance(dec, ast.Call) else dec
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if name in skip_decorators:
                return True
        return False

    skipped_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_skipped(node):
            end = getattr(node, "end_lineno", node.lineno)
            skipped_ranges.append((node.lineno, end))

    def _under_skipped(node: ast.AST) -> bool:
        ln = getattr(node, "lineno", None)
        if ln is None:
            return False
        return any(lo <= ln <= hi for lo, hi in skipped_ranges)

    for node in ast.walk(tree):
        if _under_skipped(node):
            continue
        if isinstance(node, ast.keyword) and node.arg == field and _is_true(node.value):
            return True
        if isinstance(node, ast.Assign) and _is_true(node.value):
            for tgt in node.targets:
                # Only Attribute-target assignment (`cfg.field = True`) counts.
                # Bare-Name assignment (`field = True` as a module-level
                # constant) deliberately does NOT satisfy the contract: a
                # stray constant is not evidence the flag flows into a real
                # construction.
                if isinstance(tgt, ast.Attribute) and tgt.attr == field:
                    return True
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == field and _is_true(v):
                    return True
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "setattr" and len(node.args) == 3:
                _obj, attr, val = node.args
                if isinstance(attr, ast.Constant) and attr.value == field and _is_true(val):
                    return True
    return False


def _collect_test_sources() -> list[tuple[str, ast.AST]]:
    """Parse every `test_*.py` under tests/, experiments/, and legacy/tests/
    exactly once and return (source, ast) pairs. Cached at module scope so the
    contract test over N registry entries is O(files) rather than O(N · files).

    legacy/tests/ is included because `_OPTIONAL_COMPONENT_FLAGS` is sourced
    from the rich model (legacy/train_gpt_rich.py) and its flag-effect witness
    tests were archived to legacy/tests/ in the 2026-06-06 reorg (ADR 0003).
    The Phase-B contract port will re-point both the registry and this scan at
    the active M0 model + tests/."""
    pairs: list[tuple[str, ast.AST]] = []
    self_name = Path(__file__).name
    for d in (REPO_ROOT / "tests", REPO_ROOT / "experiments", REPO_ROOT / "legacy" / "tests"):
        if not d.exists():
            continue
        for path in d.rglob("test_*.py"):
            if path.name == self_name:
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError):
                continue
            pairs.append((source, tree))
    return pairs


_TEST_SOURCES_CACHE: list[tuple[str, ast.AST]] | None = None


def _has_effect_witness(field: str) -> bool:
    """True iff some cached test source constructs the flag with `=True`."""
    global _TEST_SOURCES_CACHE
    if _TEST_SOURCES_CACHE is None:
        _TEST_SOURCES_CACHE = _collect_test_sources()
    return any(
        field in source and _ast_has_field_true(tree, field)
        for source, tree in _TEST_SOURCES_CACHE
    )


class TestOptionalComponentFlagContract(unittest.TestCase):
    def test_registry_is_non_empty(self) -> None:
        # Guard against an accidental rename or truncation of the registry.
        self.assertGreater(len(_OPTIONAL_COMPONENT_FLAGS), 0)
        self.assertEqual(
            _OPTIONAL_COMPONENT_FLAGS,
            tuple((cap.py_name, cap.label) for cap in _OPTIONAL_COMPONENT_CAPABILITIES),
        )

    def test_capability_states_are_explicit(self) -> None:
        allowed = {"rejected", "training_effect", "eval_effect", "artifact_effect"}
        bad = [
            (cap.py_name, cap.state)
            for cap in _OPTIONAL_COMPONENT_CAPABILITIES
            if cap.state not in allowed
        ]
        self.assertFalse(bad, f"unknown capability state(s): {bad}")
        rejected = [cap.py_name for cap in _OPTIONAL_COMPONENT_CAPABILITIES if cap.state == "rejected"]
        for field in rejected:
            self.assertTrue(
                _validator_rejects(field),
                f"{field} is declared rejected but _validate_hyperparameters accepts it",
            )

    def test_each_flag_has_effect_or_explicit_reject(self) -> None:
        """For every (py_name, label) in the registry, the flag must EITHER
        be rejected loudly by the validator (scaffold) OR have an
        effect-asserting test (real, observable path)."""
        missing: list[tuple[str, str]] = []
        for py_name, _label in _OPTIONAL_COMPONENT_FLAGS:
            rejects = _validator_rejects(py_name)
            has_test = _has_effect_witness(py_name)
            if not (rejects or has_test):
                status = "no validator-reject + no effect-asserting test"
                missing.append((py_name, status))
        self.assertFalse(
            missing,
            msg=(
                "Flag-to-effect contract violated: each entry in "
                "_OPTIONAL_COMPONENT_FLAGS must EITHER raise SystemExit in "
                "_validate_hyperparameters with the field name OR appear in a "
                "test that constructs the model with field=True. Missing: "
                f"{missing}"
            ),
        )

    def test_registry_labels_are_unique(self) -> None:
        # A banner-collision would make `grep label=1 run.log` ambiguous.
        labels = [label for _py, label in _OPTIONAL_COMPONENT_FLAGS]
        self.assertEqual(
            len(labels), len(set(labels)),
            msg=f"duplicate banner labels in _OPTIONAL_COMPONENT_FLAGS: {labels}",
        )

    def test_registry_python_names_are_valid_hyperparameter_fields(self) -> None:
        # Catch typos: every registry entry must name a real Hyperparameters
        # field, otherwise the validator never sees the value.
        h = Hyperparameters()
        for py_name, _label in _OPTIONAL_COMPONENT_FLAGS:
            self.assertTrue(
                hasattr(h, py_name),
                msg=f"_OPTIONAL_COMPONENT_FLAGS lists {py_name!r}, which is not a Hyperparameters field",
            )


if __name__ == "__main__":
    unittest.main()
