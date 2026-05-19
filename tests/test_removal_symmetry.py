"""Removal-symmetry sweep audit-row enforcement.

Enforces the CLAUDE.md "Removal-symmetry sweep" rule: when a Hyperparameter,
loss term, or diagnostic is removed from `train_gpt.py`, the same commit
must drop the corresponding equation/symbol from `opg_doc.tex` (both
equation blocks AND parameter-table rows) and remove or annotate every
docstring / inline comment / prescription bullet that names the field.

The test maintains an explicit ledger of *known removed names* and the
allowed contexts in which they may still appear:

  1. Backward-compat parsers (failure-string handlers, plot-metrics
     parsers for legacy logs)
  2. Annotated historical paragraphs that explicitly note the removal
     date (e.g. "removed 2026-05-15", "were removed", "no longer active")
  3. Tests that exercise the backward-compat parsing path

Any other appearance is a removal-symmetry violation that the next
reviewer cannot tell from a missed cleanup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# Names removed from `Hyperparameters` / loss assembly / probes that the
# removal-symmetry rule guards. Each entry is the bare identifier; the test
# greps for it case-sensitively (LaTeX escapes `\_` are handled by the regex).
REMOVED_NAMES = (
    # cleanup #47 (2026-05-15): operator-norm probes + Banach bound
    "lip_ub_T",
    "lip_ub_S",
    "lip_ub_F",
    "fp_bound",
    "power_jvp_F",
    # cleanup #50 (2026-05-15): CV-as-loss multiplications
    "router_load_cv_coef",
    "mos_load_cv_coef",
)

# Phrases that mark a legitimate "removed" annotation. Any line that
# contains both a removed name AND one of these markers — or sits in a
# context window of N lines around such a marker — is exempt.
REMOVAL_ANNOTATIONS = (
    "removed 2026-05-15",
    "removed entirely 2026-05-15",
    "removed entirely",
    "were removed",
    "was removed",
    "is removed",
    "is gone",
    "are gone",
    "no longer",
    "no-longer",
    "deprecated",
    "legacy",
    "Legacy",
    "backward-compat",
    "backward compat",
    "back-compat",
    "back compat",
    "refuted",
    "Refuted",
    "previously",
    "Earlier",
    "The earlier",
    "the earlier",
    "historically",
    "historical",
    "advisory",
    # The prescription advisory branch is allowed to mention removed names
    # by design.
    "operator_norm_advisory",
    # Pedagogical use: the "Most-principled" directive in CLAUDE.md uses
    # lip_ub_F as a *negative example* contrasting with rho_F. The phrase
    # "is a proxy" or "is sufficient but over-restrictive" marks this.
    "is a proxy",
    "over-restrictive",
    "is over-restrictive",
    "sufficient but",
    "fall back",
    "are negative examples",
    "as a NEGATIVE example",
    # Educational sub-stack mention inside a docstring narrating the
    # mechanism family without claiming it's active.
    "refuted by iter155",
    "refuted by iter152",
    # The CLAUDE.md "Most-principled" line uses lip_ub_F as an example;
    # this phrase appears in that line.
    "operational tests",
)

# Number of lines of context to scan around a name match for annotation
# markers. Multi-line LaTeX paragraphs and Python comment blocks often
# put the marker one or two lines away from the name itself.
CONTEXT_WINDOW = 3


_REMOVED_PATTERN = re.compile(
    rf"\b(?P<name>{'|'.join(re.escape(n) for n in REMOVED_NAMES)})\b"
)


def _scan_file(path: Path, names: tuple[str, ...]) -> list[tuple[int, str, str]]:
    """Return [(line_no, removed_name, raw_line)] for every line that
    mentions a removed name without an exempting annotation in a
    CONTEXT_WINDOW around the match.

    LaTeX texttt rendering escapes underscores (`\\_`) so we strip those
    before matching to catch references like `\\texttt{lip\\_ub\\_F}`.

    Implementation: a single combined regex (compiled at module import)
    scans each line once for any removed name — O(N lines) instead of
    O(N * M names).
    """
    findings: list[tuple[int, str, str]] = []
    if not path.exists():
        return findings
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    n = len(raw_lines)
    for i, raw in enumerate(raw_lines, start=1):
        m = _REMOVED_PATTERN.search(raw.replace("\\_", "_"))
        if m is None:
            continue
        # Check the context window (current line ± CONTEXT_WINDOW)
        # for an annotation marker.
        lo = max(0, i - 1 - CONTEXT_WINDOW)
        hi = min(n, i + CONTEXT_WINDOW)
        context = "\n".join(raw_lines[lo:hi])
        if not any(marker in context for marker in REMOVAL_ANNOTATIONS):
            findings.append((i, m.group("name"), raw))
    return findings


@pytest.mark.parametrize(
    "rel_path",
    [
        "opg_doc.tex",
        "train_gpt.py",
        "CLAUDE.md",
    ],
)
def test_removed_names_only_in_annotated_contexts(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    findings = _scan_file(path, REMOVED_NAMES)
    if findings:
        msg = (
            f"Removal-symmetry violation in {rel_path}: removed names appear "
            f"without an annotation marker ({', '.join(REMOVAL_ANNOTATIONS[:4])}, "
            f"etc.). Each finding below must either be deleted or annotated as "
            f"historical / backward-compat:\n"
        )
        for line_no, name, line in findings[:20]:
            msg += f"  {rel_path}:{line_no}  [{name}]  {line.strip()[:200]}\n"
        if len(findings) > 20:
            msg += f"  ... ({len(findings) - 20} more findings)\n"
        pytest.fail(msg)


def test_removed_names_absent_from_active_hyperparameters() -> None:
    """The removed names must not appear as live attributes on the
    `Hyperparameters` dataclass. Inline comments are fine (the previous
    test handles annotated context), but a `removed_name = ...` line is
    a removal-symmetry P0.
    """
    src = (REPO_ROOT / "train_gpt.py").read_text(encoding="utf-8")
    # Find the Hyperparameters class body (heuristic: from `class
    # Hyperparameters` to the next top-level `class ` or `def `).
    m = re.search(r"^class Hyperparameters\b.*?$", src, re.MULTILINE)
    assert m is not None, "Hyperparameters class not found in train_gpt.py"
    start = m.end()
    end_m = re.search(r"^(class |def )", src[start:], re.MULTILINE)
    body = src[start: start + (end_m.start() if end_m else len(src) - start)]

    offenders = []
    for name in REMOVED_NAMES:
        # Skip names that were never Hyperparameters fields (e.g. `lip_ub_F`
        # was a probe-emission name, not a dataclass field).
        if name in ("router_load_cv_coef", "mos_load_cv_coef"):
            # These WERE Hyperparameters fields; assert they're gone.
            if re.search(rf"^\s*{re.escape(name)}\s*=", body, re.MULTILINE):
                offenders.append(name)
    if offenders:
        pytest.fail(
            "Removal-symmetry violation: the following names are still live "
            "Hyperparameters fields despite being removed from loss assembly: "
            + ", ".join(offenders)
            + ". Drop the field definitions in the same commit as the loss-"
            "side removal."
        )


def test_iter163_consistency_log_fields_in_parser_and_contract() -> None:
    """Companion check for the Sibling-fanout DRY gate that the iter163
    promotion commit (884b132) originally violated: log field
    `consistency_anchor_loss` must appear in `experiments/plot_metrics.py`
    (parser) and `tests/test_training_contracts.py` (required_fields). This
    failure is exactly what the Removal-symmetry sweep companion is meant
    to catch when fields are *added*, not just removed — the same
    'multi-site grep-and-paste' failure mode in reverse.

    `consistency_ext_loss` was the iter163 auxiliary extension term that was
    empirically refuted by iter163c v2 (val_bpb +22 mBPB regression for Δ=1,
    1.6× step time for Δ=K_train) and removed from the run.log emission and
    parser at commit `d7e410a` (iter163c OOM fix, 2026-05-16) — iter172
    promotion (`447367c`, 2026-05-17) did not touch this field; iter172 is
    the operative baseline only because all subsequent iters inherit
    `d7e410a`'s removal. The field is intentionally absent from current
    parser entries and contracts per the removal-symmetry-sweep audit row.
    Do NOT re-add the field name here.
    """
    plot_metrics_text = (REPO_ROOT / "experiments" / "plot_metrics.py").read_text(encoding="utf-8")
    contracts_text = (REPO_ROOT / "tests" / "test_training_contracts.py").read_text(encoding="utf-8")
    for field in ("consistency_anchor_loss",):
        assert field in plot_metrics_text, (
            f"`{field}` is emitted in run.log but `experiments/plot_metrics.py` "
            f"has no parser entry — Sibling-fanout DRY violation."
        )
        assert field in contracts_text, (
            f"`{field}` is emitted in run.log but "
            f"`tests/test_training_contracts.py` does not assert it is in "
            f"required_fields — Audit-row-self-enforcement violation."
        )
