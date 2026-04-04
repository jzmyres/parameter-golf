"""Eval/validation-only comparison plot (training-style dashboard).

This plot is intentionally aligned with the main training diagnostics plot, but
restricted to eval/validation metrics (sparser cadence). It is the authoritative
view for hard expert-health constraints.

Outputs: experiments/metrics_eval_comparison.png
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

# When executed as `python experiments/plot_eval_metrics.py`, ensure repo root is on sys.path
# so `experiments.*` namespace imports work (implicit namespace package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.plot_metrics import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_CURRENT,
    COMP_LINESTYLES,
    parse_log,
    usage_min_series,
    _filter_finite,
    _has_any_finite,
    _is_finite,
)


def _plot_line(ax, b, c, key, title, ylabel=None):
    bx, by = _filter_finite(b.get("val_steps", []), b.get(key, []))
    cx, cy = _filter_finite(c.get("val_steps", []), c.get(key, []))
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
        ax.legend(fontsize=8, frameon=False)
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

def _bar_pre_post_quant(ax, b: dict, c: dict):
    ax.set_title("Val BPB (Pre vs Post-Quant)", fontsize=11)
    b_pre = b["val_bpb"][-1] if b.get("val_bpb") else None
    c_pre = c["val_bpb"][-1] if c.get("val_bpb") else None
    b_post = b.get("final_postquant_val_bpb", None)
    c_post = c.get("final_postquant_val_bpb", None)

    width = 0.35
    x = [0.0, 1.0]
    pre_vals = [float(b_pre) if b_pre is not None else math.nan, float(c_pre) if c_pre is not None else math.nan]
    post_vals = [float(b_post) if b_post is not None else math.nan, float(c_post) if c_post is not None else math.nan]

    plotted = False
    if any(_is_finite(v) for v in pre_vals):
        ax.bar([xi - width / 2 for xi in x], pre_vals, width=width, color=[COLOR_BASELINE, COLOR_CURRENT], alpha=0.45, label="Pre-quant")
        plotted = True
    if any(_is_finite(v) for v in post_vals):
        ax.bar([xi + width / 2 for xi in x], post_vals, width=width, color=[COLOR_BASELINE, COLOR_CURRENT], alpha=0.90, label="Post-quant")
        plotted = True

    if plotted:
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Current"])
        ax.set_ylabel("val_bpb")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        vals = [v for v in pre_vals + post_vals if _is_finite(v)]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        pad = max((hi - lo) * 0.5, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
        for xi, v in zip([x[0] - width / 2, x[1] - width / 2], pre_vals, strict=False):
            if _is_finite(v):
                ax.annotate(
                    f"{v:.4f}",
                    xy=(xi, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        for xi, v in zip([x[0] + width / 2, x[1] + width / 2], post_vals, strict=False):
            if _is_finite(v):
                ax.annotate(
                    f"{v:.4f}",
                    xy=(xi, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
    else:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

def _plot_gg_iter(ax, b: dict, c: dict):
    ax.set_title("GG by DEQ Iter (val)", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("gg")
    ax.grid(True, alpha=0.3)

    b_iters = b.get("gg_iter", []) or []
    c_iters = c.get("gg_iter", []) or []
    k_plot = 0
    if b_iters:
        k_plot = max(k_plot, max((len(v) for v in b_iters), default=0))
    if c_iters:
        k_plot = max(k_plot, max((len(v) for v in c_iters), default=0))
    if k_plot <= 0:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    def _k_linestyle(ki: int):
        """Line style for GG-by-DEQ-iter curves.

        ki is 0-based; k=1 should be solid, and larger k should become progressively more dashed.
        """
        if ki <= 0:
            return "-"
        dash_table = [
            (0, (10, 3)),  # k=2
            (0, (9, 3)),   # k=3
            (0, (8, 3)),   # k=4
            (0, (7, 3)),   # k=5
            (0, (6, 3)),   # k=6
            (0, (5, 2)),   # k=7
            (0, (4, 2)),   # k=8
            (0, (3, 2)),   # k=9
            (0, (2, 2)),   # k=10
            (0, (2, 1)),   # k=11
            (0, (1, 1)),   # k=12
        ]
        idx = min(ki - 1, len(dash_table) - 1)
        return dash_table[idx]

    from matplotlib.lines import Line2D
    style_handles = []
    for ki in range(k_plot):
        style = _k_linestyle(ki)
        b_vals = [v[ki] if len(v) > ki else math.nan for v in b_iters]
        c_vals = [v[ki] if len(v) > ki else math.nan for v in c_iters]
        bx, by = _filter_finite(b.get("val_steps", []), b_vals)
        cx, cy = _filter_finite(c.get("val_steps", []), c_vals)
        if bx and by:
            ax.plot(bx, by, color=COLOR_BASELINE, linestyle=style, alpha=0.75, linewidth=2.0)
        if cx and cy:
            ax.plot(cx, cy, color=COLOR_CURRENT, linestyle=style, alpha=0.75, linewidth=2.0)
        style_handles.append(Line2D([0], [0], color="#333333", lw=2.2, linestyle=style, label=f"k={ki+1}"))
    ax.set_ylim(0.0, 1.0)
    ax.legend(handles=style_handles, loc="upper right", fontsize=8, frameon=False)

def _summary_text(ax, b: dict, c: dict):
    ax.axis("off")
    def _last_finite(vals):
        for v in reversed(vals or []):
            if _is_finite(v):
                return float(v)
        return math.nan

    cur_val_bpb = _last_finite(c.get("val_bpb", []))
    base_val_bpb = _last_finite(b.get("val_bpb", []))
    cur_post = c.get("final_postquant_val_bpb", math.nan)
    base_post = b.get("final_postquant_val_bpb", math.nan)

    # Constraint checks for current run (val metrics).
    def _min_usage(d, key):
        s = usage_min_series(d, key)
        return _last_finite(s)

    lines = []
    if _is_finite(base_val_bpb) and _is_finite(cur_val_bpb):
        lines.append(f"Val BPB:    {base_val_bpb:.4f} -> {cur_val_bpb:.4f} (d={cur_val_bpb-base_val_bpb:+.4f})")
    if _is_finite(base_post) and _is_finite(cur_post):
        lines.append(f"Post-Quant: {base_post:.4f} -> {cur_post:.4f} (d={cur_post-base_post:+.4f})")

    thr_usage = 0.15
    thr_cv = 0.20
    thr_ortho = 0.20
    for comp, usage_key, cv_key, ortho_key in [
        ("MLP", "mlp_usage", "mlp_cv", "mlp_ortho"),
        ("Attn", "attn_usage", "attn_cv", "attn_ortho"),
    ]:
        mu = _min_usage(c, usage_key)
        cv = _last_finite(c.get(cv_key, []))
        ortho = _last_finite(c.get(ortho_key, []))
        ok = (_is_finite(mu) and mu >= thr_usage) and (_is_finite(cv) and cv <= thr_cv) and (_is_finite(ortho) and ortho <= thr_ortho)
        status = "PASS" if ok else "FAIL"
        lines.append(f"{comp}: min_usage={mu:.3f} cv={cv:.3f} ortho={ortho:.3f} => {status}")

    s = "\n".join(lines) if lines else "Not logged"
    ax.text(0.0, 0.5, s, fontsize=11, family="monospace", va="center", transform=ax.transAxes)

def plot_eval_comparison(baseline_log: str, current_log: str, outdir: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available", file=sys.stderr)
        return False

    b = parse_log(baseline_log)
    c = parse_log(current_log)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Baseline vs Current — Eval/Val Diagnostics", fontsize=16, fontweight="bold")

    # Global legend (colors = run, line style = component)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_BASELINE, lw=2.2, label="Baseline"),
        Line2D([0], [0], color=COLOR_CURRENT, lw=2.2, label="Current"),
    ]
    for comp in ("mlp", "attn", "mos_ctp", "mos_ntp"):
        handles.append(Line2D([0], [0], color="black", lw=2.0, linestyle=COMP_LINESTYLES.get(comp, "-"), label=comp))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.90)

    # Row 1: objective + DEQ diagnostics (val)
    _plot_line(axes[0, 0], b, c, "val_bpb", "Val BPB", ylabel="bpb")
    _plot_line(axes[0, 1], b, c, "deq_residual", "DEQ Residual (val)", ylabel="||z - f(z)||")
    _plot_line(axes[0, 2], b, c, "deq_iter_conv_rel", "DEQ Iter Conv (rel, val)", ylabel="||Δz|| / ||z||")

    # Row 2: expert-health constraints (val)
    usage_series = [
        ("mlp", usage_min_series(b, "mlp_usage"), usage_min_series(c, "mlp_usage")),
        ("attn", usage_min_series(b, "attn_usage"), usage_min_series(c, "attn_usage")),
        ("mos_ctp", usage_min_series(b, "mos_ctp_usage"), usage_min_series(c, "mos_ctp_usage")),
        ("mos_ntp", usage_min_series(b, "mos_ntp_usage"), usage_min_series(c, "mos_ntp_usage")),
    ]
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in usage_series):
        usage_series = [("expert", usage_min_series(b, "expert_usage"), usage_min_series(c, "expert_usage"))]
    _plot_components(axes[1, 0], b, c, usage_series, "Min Usage (val)", ylabel="min usage")
    axes[1, 0].axhline(0.15, color="#666666", linewidth=1.2, linestyle=":", alpha=0.8)

    cv_series = [
        ("mlp", b.get("mlp_cv", []), c.get("mlp_cv", [])),
        ("attn", b.get("attn_cv", []), c.get("attn_cv", [])),
        ("mos_ctp", b.get("mos_ctp_cv", []), c.get("mos_ctp_cv", [])),
        ("mos_ntp", b.get("mos_ntp_cv", []), c.get("mos_ntp_cv", [])),
    ]
    _plot_components(axes[1, 1], b, c, cv_series, "Balance CV (val)", ylabel="CV")
    axes[1, 1].axhline(0.20, color="#666666", linewidth=1.2, linestyle=":", alpha=0.8)

    ortho_series = [
        ("mlp", b.get("mlp_ortho", []), c.get("mlp_ortho", [])),
        ("attn", b.get("attn_ortho", []), c.get("attn_ortho", [])),
        ("mos_ctp", b.get("mos_ctp_ortho", []), c.get("mos_ctp_ortho", [])),
        ("mos_ntp", b.get("mos_ntp_ortho", []), c.get("mos_ntp_ortho", [])),
    ]
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in ortho_series):
        ortho_series = [("mlp", b.get("expert_ortho", []), c.get("expert_ortho", []))]
    _plot_components(axes[1, 2], b, c, ortho_series, "Orthogonality (val)", ylabel="mean |cos|")
    axes[1, 2].axhline(0.20, color="#666666", linewidth=1.2, linestyle=":", alpha=0.8)

    # Keep orthogonality linear and bounded.
    all_vals = []
    for _, bv, cv in ortho_series:
        all_vals.extend([v for v in (bv or []) if _is_finite(v)])
        all_vals.extend([v for v in (cv or []) if _is_finite(v)])
    hi = max(all_vals) if all_vals else 1.0
    axes[1, 2].set_ylim(0.0, max(1.0, hi * 1.05))

    # Row 3: gg_iter + scored metric + summary
    _plot_gg_iter(axes[2, 0], b, c)
    _bar_pre_post_quant(axes[2, 1], b, c)
    _summary_text(axes[2, 2], b, c)

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
