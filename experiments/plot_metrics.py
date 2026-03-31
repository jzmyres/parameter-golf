"""Plot detailed training metrics comparing baseline vs current experiment.

Shows full training curves for ALL diagnostic metrics:
- Row 1: Train Loss, Val BPB, Step Avg (ms)
- Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
- Row 3: DEQ Residual, DEQ Recon Error, DEQ Iter Convergence
- Row 4: Expert Usage (min per component), Expert Entropy, Expert Orthogonality
- Row 5: Expert Balance CV (per component), Conv Loss, (spare)
- Row 6: Summary text with final values comparison

All subplots use consistent colors: blue for Baseline, orange for Current.
Components (mlp/attn/mos_ctp/mos_ntp) are encoded with markers.
"""
import re
import sys
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
        "grad_norm": [],
        "step_avg_ms": [], "train_time_ms": [],
        "val_steps": [], "val_loss": [], "val_bpb": [],
        "deq_residual": [], "deq_recon": [], "deq_iter_conv": [],
        # Combined expert metrics (backward compat)
        "expert_usage": [], "expert_entropy": [], "expert_ortho": [],
        # Per-component: usage (list of lists), entropy, cv
        **{f"{p}_{s}": [] for p in ("mlp", "attn", "mos_ctp", "mos_ntp")
           for s in ("usage", "entropy", "cv")},
        # Per-component orthogonality
        "mlp_ortho": [], "attn_ortho": [], "mos_ctp_ortho": [], "mos_ntp_ortho": [], "mos_ortho": [],
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
            # Parse pre-clip gradient norm
            m_gn = re.search(rf"grad_norm:{_FLOAT}", line)
            data["grad_norm"].append(float(m_gn.group(1)) if m_gn else 0.0)

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
                data[key].append(float(m2.group(1)) if m2 else 0.0)
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
                data[f"{prefix}_entropy"].append(float(m_e.group(1)) if m_e else 0.0)
                m_cv = re.search(rf"{prefix}_cv:{_FLOAT}", line)
                data[f"{prefix}_cv"].append(float(m_cv.group(1)) if m_cv else 0.0)
            # Per-component orthogonality
            for comp in ("mlp", "attn", "mos", "mos_ctp", "mos_ntp"):
                m_o = re.search(rf"{comp}_ortho:{_FLOAT}", line)
                data[f"{comp}_ortho"].append(float(m_o.group(1)) if m_o else 0.0)
            # No balance-loss keys are logged; use *_cv fields for balance diagnostics.

    return data


def usage_min_series(data: dict, key: str) -> list[float]:
    """Return min expert usage per step for a usage list-of-lists key."""
    out: list[float] = []
    for u in data.get(key, []):
        out.append(min(u) if u else 0.0)
    return out


def _plot_line(ax, b, c, b_key, c_key, b_steps, c_steps, title, ylabel=None):
    """Plot two line series on the same axis with consistent colors."""
    if b[b_key] and c[c_key]:
        ax.plot(b[b_steps], b[b_key], color=COLOR_BASELINE, alpha=0.7, label="Baseline", linewidth=1.5)
        ax.plot(c[c_steps], c[c_key], color=COLOR_CURRENT, alpha=0.7, label="Current", linewidth=1.5)
    elif b[b_key]:
        ax.plot(b[b_steps], b[b_key], color=COLOR_BASELINE, alpha=0.7, label="Baseline", linewidth=1.5)
    elif c[c_key]:
        ax.plot(c[c_steps], c[c_key], color=COLOR_CURRENT, alpha=0.7, label="Current", linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_components(ax, b, c, steps_key, component_series, title, ylabel=None):
    """Plot multiple component series with Baseline=blue and Current=orange.

    Components are encoded via markers for readability across all subplots.
    """
    any_plotted = False
    for comp_label, b_values, c_values in component_series:
        marker = COMP_MARKERS.get(comp_label, None)
        if b_values is not None and any(v != 0 for v in b_values):
            ax.plot(
                b[steps_key],
                b_values,
                color=COLOR_BASELINE,
                linestyle="-",
                marker=marker,
                markersize=3,
                markevery=max(len(b[steps_key]) // 12, 1),
                alpha=0.8,
                label=f"Baseline {comp_label}",
                linewidth=1.4,
            )
            any_plotted = True
        if c_values is not None and any(v != 0 for v in c_values):
            ax.plot(
                c[steps_key],
                c_values,
                color=COLOR_CURRENT,
                linestyle="--",
                marker=marker,
                markersize=3,
                markevery=max(len(c[steps_key]) // 12, 1),
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
    ax.grid(True, alpha=0.3)


def plot_comparison(baseline_log: str, current_log: str, outdir: str):
    """Plot baseline vs current experiment comparison with full training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    b = parse_log(baseline_log)
    c = parse_log(current_log)

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

    # Row 3: DEQ diagnostics
    _plot_line(axes[2, 0], b, c, "deq_residual", "deq_residual", "val_steps", "val_steps",
               "DEQ Residual ||z - f(z)||")
    _plot_line(axes[2, 1], b, c, "deq_recon", "deq_recon", "val_steps", "val_steps",
               "DEQ Reconstruction Error")
    _plot_line(axes[2, 2], b, c, "deq_iter_conv", "deq_iter_conv", "val_steps", "val_steps",
               "DEQ Iter Conv ||z_T - z_{T-1}||")

    # Row 4: Expert diagnostics (usage, entropy, orthogonality)
    # Usage: min per component (Attn, MLP, MoS CTP, MoS NTP)
    ax_usage = axes[3, 0]
    usage_series = []
    for comp_label, key in [
        ("mlp", "mlp_usage"),
        ("attn", "attn_usage"),
        ("mos_ctp", "mos_ctp_usage"),
        ("mos_ntp", "mos_ntp_usage"),
    ]:
        usage_series.append((comp_label, usage_min_series(b, key), usage_min_series(c, key)))
    if not any(any(v != 0 for v in s[1] + s[2]) for s in usage_series):
        # Fallback: combined expert_usage (older logs)
        usage_series = [("expert", usage_min_series(b, "expert_usage"), usage_min_series(c, "expert_usage"))]
    _plot_components(
        ax_usage,
        b,
        c,
        "val_steps",
        usage_series,
        "Expert Usage (min per Component)",
        ylabel="Min usage fraction",
    )

    # Expert Entropy: per-component lines (Attn, MLP, MoS CTP, MoS NTP)
    ax_ent = axes[3, 1]
    entropy_series = []
    for comp_label, key in [
        ("mlp", "mlp_entropy"),
        ("attn", "attn_entropy"),
        ("mos_ctp", "mos_ctp_entropy"),
        ("mos_ntp", "mos_ntp_entropy"),
    ]:
        entropy_series.append((comp_label, b.get(key, []), c.get(key, [])))
    if not any(any(v != 0 for v in s[1] + s[2]) for s in entropy_series):
        # Fallback: combined expert_entropy (older logs)
        entropy_series = [("expert", b.get("expert_entropy", []), c.get("expert_entropy", []))]
    _plot_components(ax_ent, b, c, "val_steps", entropy_series, "Expert Entropy (per Component)", ylabel="Entropy")

    # Expert Orthogonality: per-component lines (formatted like entropy plot)
    ax_ortho = axes[3, 2]
    ortho_series = []
    for comp_label, key in [
        ("mlp", "mlp_ortho"),
        ("attn", "attn_ortho"),
        ("mos_ctp", "mos_ctp_ortho"),
        ("mos_ntp", "mos_ntp_ortho"),
    ]:
        ortho_series.append((comp_label, b.get(key, []), c.get(key, [])))
    if not any(any(v != 0 for v in s[1] + s[2]) for s in ortho_series):
        # Fallback: legacy key (MLP-only)
        ortho_series = [("mlp", b.get("expert_ortho", []), c.get("expert_ortho", []))]
    _plot_components(
        ax_ortho,
        b,
        c,
        "val_steps",
        ortho_series,
        "Expert Orthogonality (per Component)",
        ylabel="Mean |cos|",
    )

    # Row 5: Balance diagnostics (CV), conv_loss, spare
    ax_bal = axes[4, 0]
    bal_series = []
    for comp_label, key in [
        ("mlp", "mlp_cv"),
        ("attn", "attn_cv"),
        ("mos_ctp", "mos_ctp_cv"),
        ("mos_ntp", "mos_ntp_cv"),
    ]:
        bal_series.append((comp_label, b.get(key, []), c.get(key, [])))
    _plot_components(ax_bal, b, c, "val_steps", bal_series, "Expert Balance CV (per Component)", ylabel="CV")

    # Conv loss
    _plot_line(axes[4, 1], b, c, "ctp_loss", "ctp_loss", "train_steps", "train_steps",
               "Convergence Loss (from train)")
    axes[4, 2].axis("off")  # spare slot

    # Row 6: Summary text
    for j in range(3):
        axes[5, j].axis("off")

    summary_lines = []
    if b["val_bpb"] and c["val_bpb"]:
        delta = c["val_bpb"][-1] - b["val_bpb"][-1]
        summary_lines.append(f"Val BPB:    {b['val_bpb'][-1]:.4f} -> {c['val_bpb'][-1]:.4f} (d={delta:+.4f})")
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
    plot_comparison(str(baseline), str(current), str(expdir))

    # If previous.log exists, also generate baseline vs previous for comparison
    previous = logdir / "previous.log"
    if previous.exists():
        plot_comparison(str(baseline), str(previous), str(expdir))
        import shutil
        shutil.move(str(expdir / "metrics_comparison.png"), str(expdir / "metrics_previous.png"))
        # Re-generate the main plot (baseline vs current)
        plot_comparison(str(baseline), str(current), str(expdir))
        print("Saved metrics_previous.png (baseline vs previous iteration)")
