"""Plot detailed training metrics comparing baseline vs current experiment.

Shows full training curves for ALL diagnostic metrics:
- Row 1: Train Loss, Val BPB, Step Avg (ms)
- Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
- Row 3: DEQ Residual, DEQ Recon Error, DEQ Iter Convergence
- Row 4: Expert Usage (min per component), Expert Entropy, Expert Orthogonality
- Row 5: Expert Balance CV (per component), Convergence Loss, Final Post-Quant Val BPB
- Row 6: Summary text with final values comparison

All subplots use consistent colors: blue for Baseline, orange for Current.
Components (mlp/attn/mos_ctp/mos_ntp) are encoded with markers.
"""
import re
import sys
import math
from pathlib import Path

# Consistent colors: blue for Baseline, orange for Current
COLOR_BASELINE = "#1f77b4"  # matplotlib default blue
COLOR_CURRENT = "#ff7f0e"   # matplotlib default orange

# Component markers (keep consistent across subplots)
COMP_MARKERS = {
    "mlp": "o",
    "attn": "s",
    "mos_ctp": "^",
    "mos_ntp": "x",
}

_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_log(logpath: str) -> dict:
    """Parse training log for all metrics."""
    lines = Path(logpath).read_text().split("\n")
    data = {
        "train_steps": [], "train_loss": [], "ntp_loss": [], "ctp_loss": [],
        "conv_loss": [],
        "grad_norm": [],
        "step_avg_ms": [], "train_time_ms": [],
        "val_steps": [], "val_loss": [], "val_bpb": [],
        # Validation-time diagnostics (sparse unless VAL_LOSS_EVERY is small)
        "deq_residual": [], "deq_recon": [], "deq_iter_conv": [],
        # Combined expert metrics (backward compat)
        "expert_usage": [], "expert_entropy": [], "expert_ortho": [],
        # Per-component: usage (list of lists), entropy, cv
        **{f"{p}_{s}": [] for p in ("mlp", "attn", "mos_ctp", "mos_ntp")
           for s in ("usage", "entropy", "cv")},
        # Per-component orthogonality
        "mlp_ortho": [], "attn_ortho": [], "mos_ctp_ortho": [], "mos_ntp_ortho": [], "mos_ortho": [],
        # Train-time diagnostics (dense, logged alongside train_loss when enabled)
        "deq_residual_train": [], "deq_recon_train": [], "deq_iter_conv_train": [],
        **{f"{p}_{s}_train": [] for p in ("mlp", "attn", "mos_ctp", "mos_ntp")
           for s in ("usage", "entropy", "cv")},
        "mlp_ortho_train": [], "attn_ortho_train": [], "mos_ctp_ortho_train": [], "mos_ntp_ortho_train": [],
        # Final post-quant scoring metric (what the submission is scored on)
        "final_postquant_val_loss": None,
        "final_postquant_val_bpb": None,
    }

    for line in lines:
        # Training steps
        m = re.search(rf"^step:(\d+)/\d+ train_loss:{_FLOAT}.*train_time:{_FLOAT}ms step_avg:{_FLOAT}ms", line)
        if m:
            data["train_steps"].append(int(m.group(1)))
            data["train_loss"].append(float(m.group(2)))
            data["train_time_ms"].append(float(m.group(3)))
            data["step_avg_ms"].append(float(m.group(4)))
            # Parse NTP and CTP losses from training lines
            m_ntp = re.search(rf"ntp_loss:{_FLOAT}", line)
            data["ntp_loss"].append(float(m_ntp.group(1)) if m_ntp else 0.0)
            m_ctp = re.search(rf"ctp_loss:{_FLOAT}", line)
            data["ctp_loss"].append(float(m_ctp.group(1)) if m_ctp else 0.0)
            m_conv = re.search(rf"conv_loss:{_FLOAT}", line)
            data["conv_loss"].append(float(m_conv.group(1)) if m_conv else 0.0)
            # Parse pre-clip gradient norm
            m_gn = re.search(rf"grad_norm:{_FLOAT}", line)
            data["grad_norm"].append(float(m_gn.group(1)) if m_gn else 0.0)

            # Train-time DEQ + expert diagnostics (optional). Missing values become NaN.
            for key, pat in [
                ("deq_residual_train", rf"deq_residual:{_FLOAT}"),
                ("deq_recon_train", rf"deq_recon_err:{_FLOAT}"),
                ("deq_iter_conv_train", rf"deq_iter_conv:{_FLOAT}"),
                ("mlp_entropy_train", rf"mlp_entropy:{_FLOAT}"),
                ("attn_entropy_train", rf"attn_entropy:{_FLOAT}"),
                ("mos_ctp_entropy_train", rf"mos_ctp_entropy:{_FLOAT}"),
                ("mos_ntp_entropy_train", rf"mos_ntp_entropy:{_FLOAT}"),
                ("mlp_cv_train", rf"mlp_cv:{_FLOAT}"),
                ("attn_cv_train", rf"attn_cv:{_FLOAT}"),
                ("mos_ctp_cv_train", rf"mos_ctp_cv:{_FLOAT}"),
                ("mos_ntp_cv_train", rf"mos_ntp_cv:{_FLOAT}"),
                ("mlp_ortho_train", rf"mlp_ortho:{_FLOAT}"),
                ("attn_ortho_train", rf"attn_ortho:{_FLOAT}"),
                ("mos_ctp_ortho_train", rf"mos_ctp_ortho:{_FLOAT}"),
                ("mos_ntp_ortho_train", rf"mos_ntp_ortho:{_FLOAT}"),
            ]:
                m2 = re.search(pat, line)
                data[key].append(float(m2.group(1)) if m2 else math.nan)
            for prefix in ("mlp", "attn", "mos_ctp", "mos_ntp"):
                m_u = re.search(rf"{prefix}_usage:\[([\d.,]+)\]", line)
                data[f"{prefix}_usage_train"].append(
                    [float(v) for v in m_u.group(1).split(",")] if m_u else []
                )

        # Final post-quant metric (exact, if available)
        # Example:
        # final_int6_zstd_roundtrip_exact val_loss:2.53669845 val_bpb:1.50237571
        m = re.search(rf"^final_int6_\w+_roundtrip_exact val_loss:{_FLOAT} val_bpb:{_FLOAT}", line)
        if m:
            data["final_postquant_val_loss"] = float(m.group(1))
            data["final_postquant_val_bpb"] = float(m.group(2))

        # Validation steps
        m = re.search(rf"^step:(\d+)/\d+ val_loss:{_FLOAT} val_bpb:{_FLOAT}", line)
        if m:
            data["val_steps"].append(int(m.group(1)))
            data["val_loss"].append(float(m.group(2)))
            data["val_bpb"].append(float(m.group(3)))
            # Parse individual DEQ/expert metrics
            for key, pat in [
                ("deq_residual", rf"deq_residual:{_FLOAT}"),
                ("deq_recon", rf"deq_recon_err:{_FLOAT}"),
                ("deq_iter_conv", rf"deq_iter_conv:{_FLOAT}"),
                ("expert_entropy", rf"(?<!\w_)expert_entropy:{_FLOAT}"),
                ("expert_ortho", rf"expert_ortho:{_FLOAT}"),
            ]:
                m2 = re.search(pat, line)
                data[key].append(float(m2.group(1)) if m2 else math.nan)
            # Combined expert usage (backward compat)
            m2 = re.search(r"(?<!\w_)expert_usage:\[([\d.,]+)\]", line)
            if m2:
                usage = [float(v) for v in m2.group(1).split(",")]
                data["expert_usage"].append(usage)
            else:
                data["expert_usage"].append([])
            # Per-component expert usage + entropy + cv
            for prefix in ("mlp", "attn", "mos_ctp", "mos_ntp"):
                m_u = re.search(rf"{prefix}_usage:\[([\d.,]+)\]", line)
                data[f"{prefix}_usage"].append(
                    [float(v) for v in m_u.group(1).split(",")] if m_u else [])
                m_e = re.search(rf"{prefix}_entropy:{_FLOAT}", line)
                data[f"{prefix}_entropy"].append(float(m_e.group(1)) if m_e else math.nan)
                m_cv = re.search(rf"{prefix}_cv:{_FLOAT}", line)
                data[f"{prefix}_cv"].append(float(m_cv.group(1)) if m_cv else math.nan)
            # Per-component orthogonality
            for comp in ("mlp", "attn", "mos", "mos_ctp", "mos_ntp"):
                m_o = re.search(rf"{comp}_ortho:{_FLOAT}", line)
                data[f"{comp}_ortho"].append(float(m_o.group(1)) if m_o else math.nan)
            # No balance-loss keys are logged; use *_cv fields for balance diagnostics.

    return data


def usage_min_series(data: dict, key: str) -> list[float]:
    """Return min expert usage per step for a usage list-of-lists key."""
    out: list[float] = []
    for u in data.get(key, []):
        out.append(min(u) if u else math.nan)
    return out


def _is_finite(x: float) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(float(x)) and math.isfinite(float(x))


def _filter_finite(steps: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    if not steps or not values:
        return [], []
    n = min(len(steps), len(values))
    xs: list[int] = []
    ys: list[float] = []
    for i in range(n):
        v = values[i]
        if _is_finite(v):
            xs.append(steps[i])
            ys.append(float(v))
    return xs, ys


def _has_any_finite(values: list[float]) -> bool:
    return any(_is_finite(v) for v in values or [])


def _plot_line(ax, b, c, b_key, c_key, b_steps, c_steps, title, ylabel=None):
    """Plot two line series on the same axis with consistent colors."""
    plotted = False
    bx, by = _filter_finite(b.get(b_steps, []), b.get(b_key, []))
    cx, cy = _filter_finite(c.get(c_steps, []), c.get(c_key, []))
    if bx and by:
        if len(by) < 2:
            ax.scatter(bx, by, color=COLOR_BASELINE, alpha=0.85, label="Baseline", s=28)
            ax.text(bx[0], by[0], f"{by[0]:.4f}", fontsize=8, ha="left", va="bottom", color=COLOR_BASELINE)
        else:
            ax.plot(bx, by, color=COLOR_BASELINE, alpha=0.7, label="Baseline", linewidth=1.5)
        plotted = True
    if cx and cy:
        if len(cy) < 2:
            ax.scatter(cx, cy, color=COLOR_CURRENT, alpha=0.85, label="Current", s=28)
            ax.text(cx[0], cy[0], f"{cy[0]:.4f}", fontsize=8, ha="left", va="bottom", color=COLOR_CURRENT)
        else:
            ax.plot(cx, cy, color=COLOR_CURRENT, alpha=0.7, label="Current", linewidth=1.5)
        plotted = True
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


def _plot_components(ax, b, c, steps_key, component_series, title, ylabel=None):
    """Plot multiple component series with Baseline=blue and Current=orange.

    Components are encoded via markers for readability across all subplots.
    """
    any_plotted = False
    for comp_label, b_values, c_values in component_series:
        marker = COMP_MARKERS.get(comp_label, None)
        if b_values is not None and _has_any_finite(b_values):
            bx, by = _filter_finite(b.get(steps_key, []), b_values)
            if bx and by:
                if len(by) < 2:
                    ax.scatter(
                        bx,
                        by,
                        color=COLOR_BASELINE,
                        marker=marker,
                        alpha=0.85,
                        s=24,
                        label=f"Baseline {comp_label}",
                    )
                else:
                    ax.plot(
                        bx,
                        by,
                        color=COLOR_BASELINE,
                        linestyle="-",
                        marker=marker,
                        markersize=3,
                        markevery=max(len(bx) // 12, 1),
                        alpha=0.8,
                        label=f"Baseline {comp_label}",
                        linewidth=1.4,
                    )
                any_plotted = True
        if c_values is not None and _has_any_finite(c_values):
            cx, cy = _filter_finite(c.get(steps_key, []), c_values)
            if cx and cy:
                if len(cy) < 2:
                    ax.scatter(
                        cx,
                        cy,
                        color=COLOR_CURRENT,
                        marker=marker,
                        alpha=0.85,
                        s=24,
                        label=f"Current {comp_label}",
                    )
                else:
                    ax.plot(
                        cx,
                        cy,
                        color=COLOR_CURRENT,
                        linestyle="--",
                        marker=marker,
                        markersize=3,
                        markevery=max(len(cx) // 12, 1),
                        alpha=0.8,
                        label=f"Current {comp_label}",
                        linewidth=1.4,
                    )
                any_plotted = True

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    if any_plotted:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def plot_comparison(baseline_log: str, current_log: str, outdir: str) -> bool:
    """Plot baseline vs current experiment comparison with full training curves.

    Returns True if a plot was generated, False if plotting dependencies are missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available", file=sys.stderr)
        return False

    b = parse_log(baseline_log)
    c = parse_log(current_log)

    def _prefer_train(train_key: str, val_key: str) -> tuple[str, str]:
        if _has_any_finite(b.get(train_key, [])) or _has_any_finite(c.get(train_key, [])):
            return "train_steps", train_key
        return "val_steps", val_key

    def _prefer_train_usage(train_key: str, val_key: str) -> tuple[str, str]:
        if any(b.get(train_key, [])) or any(c.get(train_key, [])):
            return "train_steps", train_key
        return "val_steps", val_key

    fig, axes = plt.subplots(6, 3, figsize=(18, 26))
    fig.suptitle("Baseline vs Current Experiment — Full Diagnostics", fontsize=16, fontweight="bold")

    # Row 1: Training metrics (total loss, val bpb, step avg)
    _plot_line(axes[0, 0], b, c, "train_loss", "train_loss", "train_steps", "train_steps", "Train Loss (Total)")
    _plot_line(axes[0, 1], b, c, "val_bpb", "val_bpb", "val_steps", "val_steps", "Val BPB")
    _plot_line(axes[0, 2], b, c, "step_avg_ms", "step_avg_ms", "train_steps", "train_steps", "Step Avg (ms)")

    # Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
    _plot_line(axes[1, 0], b, c, "ntp_loss", "ntp_loss", "train_steps", "train_steps", "NTP Loss")
    _plot_line(axes[1, 1], b, c, "ctp_loss", "ctp_loss", "train_steps", "train_steps", "CTP Loss")
    _plot_line(axes[1, 2], b, c, "grad_norm", "grad_norm", "train_steps", "train_steps",
               "Pre-clip Grad Norm")

    # Row 3: DEQ diagnostics (prefer train-logged diagnostics for dense curves)
    steps_key, key = _prefer_train("deq_residual_train", "deq_residual")
    _plot_line(axes[2, 0], b, c, key, key, steps_key, steps_key, "DEQ Residual ||z - f(z)||")
    steps_key, key = _prefer_train("deq_recon_train", "deq_recon")
    _plot_line(axes[2, 1], b, c, key, key, steps_key, steps_key, "DEQ Reconstruction Error")
    steps_key, key = _prefer_train("deq_iter_conv_train", "deq_iter_conv")
    _plot_line(axes[2, 2], b, c, key, key, steps_key, steps_key, "DEQ Iter Conv ||z_T - z_{T-1}||")

    # Row 4: Expert diagnostics (usage, entropy, orthogonality). Prefer train-logged series for dense curves.
    # Usage: min per component (Attn, MLP, MoS CTP, MoS NTP)
    ax_usage = axes[3, 0]
    usage_series = []
    for comp_label, key in [
        ("mlp", "mlp_usage_train"),
        ("attn", "attn_usage_train"),
        ("mos_ctp", "mos_ctp_usage_train"),
        ("mos_ntp", "mos_ntp_usage_train"),
    ]:
        usage_series.append((comp_label, usage_min_series(b, key), usage_min_series(c, key)))
    steps_key, _ = _prefer_train_usage("mlp_usage_train", "mlp_usage")
    if steps_key == "val_steps":
        usage_series = []
        for comp_label, key in [
            ("mlp", "mlp_usage"),
            ("attn", "attn_usage"),
            ("mos_ctp", "mos_ctp_usage"),
            ("mos_ntp", "mos_ntp_usage"),
        ]:
            usage_series.append((comp_label, usage_min_series(b, key), usage_min_series(c, key)))
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in usage_series):
        # Fallback: combined expert_usage (older logs)
        usage_series = [("expert", usage_min_series(b, "expert_usage"), usage_min_series(c, "expert_usage"))]
    _plot_components(
        ax_usage,
        b,
        c,
        steps_key,
        usage_series,
        "Expert Usage (min per Component)",
        ylabel="Min usage fraction",
    )

    # Expert Entropy: per-component lines (Attn, MLP, MoS CTP, MoS NTP)
    ax_ent = axes[3, 1]
    steps_key, _ = _prefer_train("mlp_entropy_train", "mlp_entropy")
    entropy_series = []
    for comp_label, key in [
        ("mlp", "mlp_entropy_train" if steps_key == "train_steps" else "mlp_entropy"),
        ("attn", "attn_entropy_train" if steps_key == "train_steps" else "attn_entropy"),
        ("mos_ctp", "mos_ctp_entropy_train" if steps_key == "train_steps" else "mos_ctp_entropy"),
        ("mos_ntp", "mos_ntp_entropy_train" if steps_key == "train_steps" else "mos_ntp_entropy"),
    ]:
        entropy_series.append((comp_label, b.get(key, []), c.get(key, [])))
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in entropy_series):
        # Fallback: combined expert_entropy (older logs)
        entropy_series = [("expert", b.get("expert_entropy", []), c.get("expert_entropy", []))]
    _plot_components(ax_ent, b, c, steps_key, entropy_series, "Expert Entropy (per Component)", ylabel="Entropy")

    # Expert Orthogonality: per-component lines (formatted like entropy plot)
    ax_ortho = axes[3, 2]
    steps_key, _ = _prefer_train("mlp_ortho_train", "mlp_ortho")
    ortho_series = []
    for comp_label, key in [
        ("mlp", "mlp_ortho_train" if steps_key == "train_steps" else "mlp_ortho"),
        ("attn", "attn_ortho_train" if steps_key == "train_steps" else "attn_ortho"),
        ("mos_ctp", "mos_ctp_ortho_train" if steps_key == "train_steps" else "mos_ctp_ortho"),
        ("mos_ntp", "mos_ntp_ortho_train" if steps_key == "train_steps" else "mos_ntp_ortho"),
    ]:
        ortho_series.append((comp_label, b.get(key, []), c.get(key, [])))
    if not any(_has_any_finite(s[1]) or _has_any_finite(s[2]) for s in ortho_series):
        # Fallback: legacy key (MLP-only)
        ortho_series = [("mlp", b.get("expert_ortho", []), c.get("expert_ortho", []))]
    _plot_components(
        ax_ortho,
        b,
        c,
        steps_key,
        ortho_series,
        "Expert Orthogonality (per Component)",
        ylabel="Mean |cos|",
    )

    # Row 5: Balance diagnostics (CV), conv_loss, spare
    ax_bal = axes[4, 0]
    steps_key, _ = _prefer_train("mlp_cv_train", "mlp_cv")
    bal_series = []
    for comp_label, key in [
        ("mlp", "mlp_cv_train" if steps_key == "train_steps" else "mlp_cv"),
        ("attn", "attn_cv_train" if steps_key == "train_steps" else "attn_cv"),
        ("mos_ctp", "mos_ctp_cv_train" if steps_key == "train_steps" else "mos_ctp_cv"),
        ("mos_ntp", "mos_ntp_cv_train" if steps_key == "train_steps" else "mos_ntp_cv"),
    ]:
        bal_series.append((comp_label, b.get(key, []), c.get(key, [])))
    _plot_components(ax_bal, b, c, steps_key, bal_series, "Expert Balance CV (per Component)", ylabel="CV")

    # Convergence loss (from train)
    _plot_line(
        axes[4, 1],
        b,
        c,
        "conv_loss",
        "conv_loss",
        "train_steps",
        "train_steps",
        "Convergence Loss (from train)",
    )

    # Pre vs post-quant val_bpb (post-quant is the scored metric)
    ax_postq = axes[4, 2]
    ax_postq.set_title("Val BPB (Pre vs Post-Quant)", fontsize=11)
    b_pre = b["val_bpb"][-1] if b.get("val_bpb") else None
    c_pre = c["val_bpb"][-1] if c.get("val_bpb") else None
    b_post = b.get("final_postquant_val_bpb", None)
    c_post = c.get("final_postquant_val_bpb", None)

    width = 0.35
    x = [0.0, 1.0]
    pre_vals = [b_pre, c_pre]
    post_vals = [b_post, c_post]
    pre_colors = [COLOR_BASELINE, COLOR_CURRENT]
    post_colors = [COLOR_BASELINE, COLOR_CURRENT]

    plotted_any = False
    if any(v is not None for v in pre_vals):
        ax_postq.bar([xi - width / 2 for xi in x], pre_vals, width=width, color=pre_colors, alpha=0.45, label="Pre-quant")
        plotted_any = True
    if any(v is not None for v in post_vals):
        ax_postq.bar([xi + width / 2 for xi in x], post_vals, width=width, color=post_colors, alpha=0.90, label="Post-quant")
        plotted_any = True

    if plotted_any:
        for xi, v in zip([x[0] - width / 2, x[1] - width / 2], pre_vals, strict=False):
            if v is not None:
                ax_postq.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        for xi, v in zip([x[0] + width / 2, x[1] + width / 2], post_vals, strict=False):
            if v is not None:
                ax_postq.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_postq.set_xticks(x)
        ax_postq.set_xticklabels(["Baseline", "Current"])
        ax_postq.set_ylabel("val_bpb")
        ax_postq.legend(fontsize=8)
        ax_postq.grid(True, axis="y", alpha=0.3)
        vals = [v for v in pre_vals + post_vals if v is not None]
        if vals:
            lo = min(vals)
            hi = max(vals)
            pad = max((hi - lo) * 0.5, 0.002)
            ax_postq.set_ylim(lo - pad, hi + pad)
    else:
        ax_postq.text(
            0.5,
            0.5,
            "No val_bpb or post-quant metric found",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_postq.transAxes,
        )
        ax_postq.set_xticks([])
        ax_postq.set_yticks([])

    # Row 6: Summary text
    for j in range(3):
        axes[5, j].axis("off")

    summary_lines = []
    if b["val_bpb"] and c["val_bpb"]:
        delta = c["val_bpb"][-1] - b["val_bpb"][-1]
        summary_lines.append(f"Val BPB:    {b['val_bpb'][-1]:.4f} -> {c['val_bpb'][-1]:.4f} (d={delta:+.4f})")
    if b.get("final_postquant_val_bpb") is not None and c.get("final_postquant_val_bpb") is not None:
        d2 = c["final_postquant_val_bpb"] - b["final_postquant_val_bpb"]
        summary_lines.append(
            f"Post-Quant: {b['final_postquant_val_bpb']:.4f} -> {c['final_postquant_val_bpb']:.4f} (d={d2:+.4f})"
        )
    if b["train_steps"] and c["train_steps"]:
        summary_lines.append(f"Steps:      {b['train_steps'][-1]} vs {c['train_steps'][-1]}")
    if b["train_loss"] and c["train_loss"]:
        summary_lines.append(f"Train Loss: {b['train_loss'][-1]:.4f} vs {c['train_loss'][-1]:.4f}")
    if b["ntp_loss"] and c["ntp_loss"] and any(v > 0 for v in b["ntp_loss"]):
        summary_lines.append(f"NTP Loss:   {b['ntp_loss'][-1]:.4f} vs {c['ntp_loss'][-1]:.4f}")
    if b["ctp_loss"] and c["ctp_loss"] and any(v > 0 for v in b["ctp_loss"]):
        summary_lines.append(f"CTP Loss:   {b['ctp_loss'][-1]:.4f} vs {c['ctp_loss'][-1]:.4f}")
    if b["grad_norm"] and c["grad_norm"] and any(v > 0 for v in b["grad_norm"] + c["grad_norm"]):
        summary_lines.append(f"Grad Norm:  {b['grad_norm'][-1]:.4f} vs {c['grad_norm'][-1]:.4f}")
    if b["deq_residual"] and c["deq_residual"]:
        summary_lines.append(f"DEQ Res:    {b['deq_residual'][-1]:.0f} vs {c['deq_residual'][-1]:.0f}")
    if b["deq_recon"] and c["deq_recon"]:
        summary_lines.append(f"Recon Err:  {b['deq_recon'][-1]:.1f} vs {c['deq_recon'][-1]:.1f}")
    if b["deq_iter_conv"] and c["deq_iter_conv"]:
        summary_lines.append(f"Iter Conv:  {b['deq_iter_conv'][-1]:.1f} vs {c['deq_iter_conv'][-1]:.1f}")
    if b["expert_entropy"] and c["expert_entropy"]:
        summary_lines.append(f"Entropy:    {b['expert_entropy'][-1]:.4f} vs {c['expert_entropy'][-1]:.4f}")
    if b["expert_ortho"] and c["expert_ortho"]:
        summary_lines.append(f"Ortho:      {b['expert_ortho'][-1]:.4f} vs {c['expert_ortho'][-1]:.4f}")

    summary = "\n".join(summary_lines)
    axes[5, 1].text(0.0, 0.5, summary, fontsize=11, family="monospace",
                   verticalalignment="center", transform=axes[5, 1].transAxes)

    plt.tight_layout()
    plt.savefig(str(Path(outdir) / "metrics_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved metrics_comparison.png")
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

    # Main plot: baseline vs current
    ok_main = plot_comparison(str(baseline), str(current), str(expdir))

    # If previous.log exists, also generate baseline vs previous for comparison
    previous = logdir / "previous.log"
    if previous.exists():
        ok_prev = plot_comparison(str(baseline), str(previous), str(expdir))
        if ok_prev and (expdir / "metrics_comparison.png").exists():
            import shutil
            shutil.move(str(expdir / "metrics_comparison.png"), str(expdir / "metrics_previous.png"))
            print("Saved metrics_previous.png (baseline vs previous iteration)")
        # Re-generate the main plot (baseline vs current)
        ok_main = plot_comparison(str(baseline), str(current), str(expdir)) or ok_main

    if not ok_main:
        sys.exit(1)
