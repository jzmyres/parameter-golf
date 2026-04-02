"""Eval/validation-only expert-health comparison plot.

This plot is intentionally sparse and uncluttered: it focuses on the eval-mode
metrics that define hard constraints (min usage, balance CV, orthogonality) and
the primary objective (val_bpb).

Outputs: experiments/metrics_eval_comparison.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from experiments.plot_metrics import (
    COLOR_BASELINE,
    COLOR_CURRENT,
    COMP_LINESTYLES,
    parse_log,
    usage_min_series,
    _filter_finite,
    _has_any_finite,
    _is_finite,
)


def _plot_line(ax, b, c, b_key, c_key, title, ylabel=None):
    bx, by = _filter_finite(b.get("val_steps", []), b.get(b_key, []))
    cx, cy = _filter_finite(c.get("val_steps", []), c.get(c_key, []))
    plotted = False
    if bx and by:
        ax.plot(bx, by, color=COLOR_BASELINE, alpha=0.8, label="Baseline", linewidth=2.0)
        plotted = True
    if cx and cy:
        ax.plot(cx, cy, color=COLOR_CURRENT, alpha=0.8, label="Current", linewidth=2.0)
        plotted = True
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


def _plot_components(ax, b, c, series, title, ylabel=None):
    plotted = False
    for comp_label, b_vals, c_vals in series:
        style = COMP_LINESTYLES.get(comp_label, "-")
        bx, by = _filter_finite(b.get("val_steps", []), b_vals)
        cx, cy = _filter_finite(c.get("val_steps", []), c_vals)
        if bx and by:
            ax.plot(bx, by, color=COLOR_BASELINE, linestyle=style, alpha=0.85, linewidth=2.0)
            plotted = True
        if cx and cy:
            ax.plot(cx, cy, color=COLOR_CURRENT, linestyle=style, alpha=0.85, linewidth=2.0)
            plotted = True
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    if not plotted:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_eval_comparison(baseline_log: str, current_log: str, outdir: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available", file=sys.stderr)
        return False

    b = parse_log(baseline_log)
    c = parse_log(current_log)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs Current — Eval/Val Expert Health", fontsize=15, fontweight="bold")

    # Val BPB curve
    _plot_line(axes[0, 0], b, c, "val_bpb", "val_bpb", "Val BPB", ylabel="bpb")

    # Min usage per component (val)
    usage_series = [
        ("mlp", usage_min_series(b, "mlp_usage"), usage_min_series(c, "mlp_usage")),
        ("attn", usage_min_series(b, "attn_usage"), usage_min_series(c, "attn_usage")),
        ("mos_ctp", usage_min_series(b, "mos_ctp_usage"), usage_min_series(c, "mos_ctp_usage")),
        ("mos_ntp", usage_min_series(b, "mos_ntp_usage"), usage_min_series(c, "mos_ntp_usage")),
    ]
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in usage_series):
        usage_series = [("expert", usage_min_series(b, "expert_usage"), usage_min_series(c, "expert_usage"))]
    _plot_components(axes[0, 1], b, c, usage_series, "Min Usage (val)", ylabel="min usage")

    # Balance CV per component (val)
    cv_series = [
        ("mlp", b.get("mlp_cv", []), c.get("mlp_cv", [])),
        ("attn", b.get("attn_cv", []), c.get("attn_cv", [])),
        ("mos_ctp", b.get("mos_ctp_cv", []), c.get("mos_ctp_cv", [])),
        ("mos_ntp", b.get("mos_ntp_cv", []), c.get("mos_ntp_cv", [])),
    ]
    _plot_components(axes[1, 0], b, c, cv_series, "Balance CV (val)", ylabel="CV")

    # Output-space orthogonality per component (val)
    ortho_series = [
        ("mlp", b.get("mlp_ortho", []), c.get("mlp_ortho", [])),
        ("attn", b.get("attn_ortho", []), c.get("attn_ortho", [])),
        ("mos_ctp", b.get("mos_ctp_ortho", []), c.get("mos_ctp_ortho", [])),
        ("mos_ntp", b.get("mos_ntp_ortho", []), c.get("mos_ntp_ortho", [])),
    ]
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in ortho_series):
        ortho_series = [("mlp", b.get("expert_ortho", []), c.get("expert_ortho", []))]
    _plot_components(axes[1, 1], b, c, ortho_series, "Orthogonality (val)", ylabel="mean |cos|")

    # Keep orthogonality linear and bounded.
    all_vals = []
    for _, bv, cv in ortho_series:
        all_vals.extend([v for v in (bv or []) if _is_finite(v)])
        all_vals.extend([v for v in (cv or []) if _is_finite(v)])
    hi = max(all_vals) if all_vals else 1.0
    axes[1, 1].set_ylim(0.0, max(1.0, hi * 1.05))

    plt.tight_layout()
    plt.savefig(str(Path(outdir) / "metrics_eval_comparison.png"), dpi=150)
    plt.close()
    print("Saved metrics_eval_comparison.png")
    return True


if __name__ == "__main__":
    expdir = Path(__file__).resolve().parent
    logdir = expdir / "training_logs"
    baseline = logdir / "baseline.log"
    current = logdir / "current.log"
    if not baseline.exists():
        print(f"No baseline log at {baseline}")
        sys.exit(1)
    if not current.exists():
        print("No current log — using baseline for both")
        current = baseline
    ok = plot_eval_comparison(str(baseline), str(current), str(expdir))
    if not ok:
        sys.exit(1)

