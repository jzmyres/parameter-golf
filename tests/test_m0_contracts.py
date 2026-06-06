"""Consolidated M0 contract test (Phase-B).

A single home for the *cross-cutting* contracts of the ACTIVE M0 model
(``train_gpt.py``), distinct from the per-module unit coverage in
``tests/test_m0_modules.py`` / ``test_m0_trainer.py`` / ``test_m0_metrics.py``.

The 17 rich-model contract tests now live under ``legacy/tests/`` (they validate
``legacy/train_gpt_rich.py`` and are not run by the default ``pytest tests/``
sweep). This file ports the still-relevant *contract* shape to M0:

  1. Removal-symmetry: the ledger of names removed from the active code must not
     reappear (un-annotated) in M0 ``train_gpt.py`` or ``reports/opg_doc.tex``.
     (Ledger + allowed-context logic ported from
     ``legacy/tests/test_removal_symmetry.py``.)
  2. Optimizer param coverage: every trainable param lands in exactly one
     optimizer group (no uncovered param, none double-counted).
  3. int6 artifact roundtrip: save -> load round-trips to bytes under the 16 MB
     challenge budget, and dequantized tensors are close to the originals.
  4. Doc-code consistency: the M0 headline facts are documented in
     ``reports/opg_doc.tex`` / ``CLAUDE.md``.

All tests are plain ``def test_*`` with no return values (``pytest.ini`` errors
on ``PytestReturnNotNoneWarning``). They run FAST on CPU with a tiny model.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from train_gpt import (
    Hyperparameters,
    M0GPT,
    adamw_params,
    build_optimizers,
    load_int6_artifact,
    save_int6_artifact,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_ARTIFACT_BYTES = 16_000_000  # challenge artifact budget (CLAUDE.md invariant)


def _tiny_args() -> Hyperparameters:
    """A tiny CPU-fast M0 config mirroring ``tests/test_m0_trainer.py``."""
    return Hyperparameters(
        model_dim=64, n_heads=2, n_kv_heads=1, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=32,
        max_seq_len=32,
    )


# ---------------------------------------------------------------------------
# Contract 1 — Removal-symmetry (ported from legacy/tests/test_removal_symmetry.py,
# but scanning the ACTIVE M0 train_gpt.py + reports/opg_doc.tex).
# ---------------------------------------------------------------------------
# Names removed from the active code that the removal-symmetry rule guards.
# Ported verbatim from the legacy ledger.
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

# Phrases marking a legitimate "removed/historical" annotation. Ported verbatim
# from the legacy ledger's allowed-context list.
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
    "operator_norm_advisory",
    "is a proxy",
    "over-restrictive",
    "is over-restrictive",
    "sufficient but",
    "fall back",
    "are negative examples",
    "as a NEGATIVE example",
    "refuted by iter155",
    "refuted by iter152",
    "operational tests",
)

CONTEXT_WINDOW = 3

_REMOVED_PATTERN = re.compile(
    rf"\b(?P<name>{'|'.join(re.escape(n) for n in REMOVED_NAMES)})\b"
)


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_no, removed_name, raw_line)] for every line that mentions a
    removed name without an exempting annotation within CONTEXT_WINDOW lines.

    LaTeX texttt rendering escapes underscores (``\\_``) so we strip those before
    matching to catch references like ``\\texttt{lip\\_ub\\_F}``."""
    findings: list[tuple[int, str, str]] = []
    if not path.exists():
        return findings
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    n = len(raw_lines)
    for i, raw in enumerate(raw_lines, start=1):
        m = _REMOVED_PATTERN.search(raw.replace("\\_", "_"))
        if m is None:
            continue
        lo = max(0, i - 1 - CONTEXT_WINDOW)
        hi = min(n, i + CONTEXT_WINDOW)
        context = "\n".join(raw_lines[lo:hi])
        if not any(marker in context for marker in REMOVAL_ANNOTATIONS):
            findings.append((i, m.group("name"), raw))
    return findings


def test_removed_names_absent_from_m0_train_gpt():
    """No removed name appears (un-annotated) in the ACTIVE M0 train_gpt.py."""
    findings = _scan_file(REPO_ROOT / "train_gpt.py")
    assert not findings, (
        "Removal-symmetry violation in train_gpt.py: removed names appear "
        "without an annotation marker:\n"
        + "\n".join(
            f"  train_gpt.py:{ln}  [{name}]  {line.strip()[:160]}"
            for ln, name, line in findings[:20]
        )
    )


def test_removed_names_only_in_annotated_contexts_in_opg_doc():
    """No removed name appears (un-annotated) in the project spec opg_doc.tex."""
    findings = _scan_file(REPO_ROOT / "reports" / "opg_doc.tex")
    assert not findings, (
        "Removal-symmetry violation in reports/opg_doc.tex: removed names appear "
        "without an annotation marker:\n"
        + "\n".join(
            f"  reports/opg_doc.tex:{ln}  [{name}]  {line.strip()[:160]}"
            for ln, name, line in findings[:20]
        )
    )


# ---------------------------------------------------------------------------
# Contract 2 — Optimizer parameter coverage.
# ---------------------------------------------------------------------------
def test_optimizer_param_coverage_exact_partition():
    """Every trainable parameter lands in EXACTLY one optimizer group: none
    uncovered, none double-counted.

    ``build_optimizers`` + ``adamw_params`` together must form an exact
    partition of the trainable params (Muon group ∪ AdamW clip scope), with
    tied weights (``tok_emb.weight`` IS ``mos_head.out_embed.weight``) counted
    once by ``id``.
    """
    model = M0GPT(_tiny_args())
    optimizers = build_optimizers(model, matrix_lr=0.02, embed_lr=0.1, scalar_lr=0.02)

    seen: dict[int, bool] = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in seen, "parameter appears in more than one optimizer group"
                seen[pid] = True

    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable_ids == set(seen.keys()), (
        "optimizer groups do not exactly cover the trainable parameters: "
        f"uncovered={len(trainable_ids - set(seen))} extra={len(set(seen) - trainable_ids)}"
    )

    # ``adamw_params`` (grad-clip scope) is exactly the non-Muon trainable params,
    # so clip-scope ∪ Muon == trainable (a second, independent partition check).
    from train_gpt import Muon

    muon_ids = {
        id(p) for opt in optimizers if isinstance(opt, Muon)
        for g in opt.param_groups for p in g["params"]
    }
    clip_ids = {id(p) for p in adamw_params(model, optimizers)}
    assert not (clip_ids & muon_ids), "a param is in both Muon and the AdamW clip scope"
    assert clip_ids | muon_ids == trainable_ids, (
        "adamw_params ∪ Muon params is not an exact partition of trainable params"
    )


# ---------------------------------------------------------------------------
# Contract 3 — int6 artifact roundtrip.
# ---------------------------------------------------------------------------
def test_int6_artifact_roundtrip_bytes_budget_and_closeness():
    """save_int6_artifact -> load_int6_artifact round-trips to a ``bytes`` blob
    under the 16 MB challenge budget, and the dequantized tensors are close to
    the originals within int6 tolerance.

    Mirrors the API used in ``tests/test_m0_trainer.py`` /
    ``tests/test_m0_metrics.py``: ``save_int6_artifact(state_dict)`` returns
    ``(compressed, qsd, meta)`` and ``load_int6_artifact(compressed, template_sd)``
    returns a dequantized state_dict keyed like the template.
    """
    torch.manual_seed(0)
    model = M0GPT(_tiny_args())
    sd = model.state_dict()

    compressed, qsd, meta = save_int6_artifact(sd)
    assert isinstance(compressed, (bytes, bytearray)), "artifact must be bytes"
    assert len(compressed) > 0
    assert len(compressed) < MAX_ARTIFACT_BYTES, (
        f"artifact {len(compressed)} bytes exceeds the {MAX_ARTIFACT_BYTES} budget"
    )

    deq = load_int6_artifact(compressed, sd)
    # The dequantized state_dict carries every template key.
    assert set(deq.keys()) == set(sd.keys()), "roundtrip keys differ from the template"

    # int6 is a per-row clipped quantizer: dequantized float tensors track the
    # originals within a relative tolerance. Check the large, int6-quantized
    # tensors (the small / passthrough ones are exact-or-fp16). The tolerance is
    # the int6-grid relative error, robust to the SDCLIP clip+round.
    checked = 0
    for k, orig in sd.items():
        if not orig.is_floating_point() or orig.numel() <= 8192:
            continue  # passthrough / fp16-kept tensors are not int6-quantized
        d = deq[k].float()
        o = orig.float()
        denom = o.abs().mean().clamp_min(1e-8)
        rel_err = (d - o).abs().mean() / denom
        assert torch.isfinite(rel_err)
        assert float(rel_err) < 0.25, (
            f"int6 dequant of {k} drifted too far: rel_err={float(rel_err):.4f}"
        )
        checked += 1
    assert checked > 0, "no int6-quantized tensor was exercised by the roundtrip"


# ---------------------------------------------------------------------------
# Contract 4 — Doc-code consistency (M0).
# ---------------------------------------------------------------------------
def test_opg_doc_documents_m0_headline_metrics():
    """The project spec documents the M0 headline facts: the correctness gate
    ``recon_rel``, the expressiveness headline ``expressiveness_rho``, and the
    param-efficiency metric ``phi_isodepth`` (substring checks, robust to LaTeX
    escaping of underscores)."""
    doc = (REPO_ROOT / "reports" / "opg_doc.tex").read_text(encoding="utf-8")
    doc_plain = doc.replace("\\_", "_")
    for name in ("recon_rel", "expressiveness_rho", "phi_isodepth"):
        assert name in doc_plain, f"reports/opg_doc.tex does not mention `{name}`"


def test_claude_md_current_implementation_points_to_train_gpt():
    """CLAUDE.md's M0 section orients to the active source file."""
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Current Implementation (M0)" in claude, (
        "CLAUDE.md has no 'Current Implementation (M0)' section"
    )
    idx = claude.index("Current Implementation (M0)")
    section = claude[idx: idx + 4000]
    assert "train_gpt.py" in section, (
        "CLAUDE.md 'Current Implementation (M0)' section does not reference train_gpt.py"
    )
