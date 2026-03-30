"""Plot detailed training metrics comparing baseline vs current experiment.

Shows full training curves for ALL diagnostic metrics:
- Row 1: Train Loss, Val BPB, Step Avg (ms)
- Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
- Row 3: DEQ Residual, DEQ Recon Error, DEQ Iter Convergence
- Row 4: Load Balance (per component per expert), Expert Entropy, Expert Orthogonality
- Row 5: Summary text with final values comparison
"""
import re
import sys
from pathlib import Path

# Consistent colors: blue for Baseline, orange for Current
COLOR_BASELINE = "#1f77b4"  # matplotlib default blue
COLOR_CURRENT = "#ff7f0e"   # matplotlib default orange


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
        "expert_usage": [],  # list of lists (variable number of experts)
        "expert_entropy": [], "expert_ortho": [],
        # Per-component expert usage: each is list of lists
        "mlp_usage": [], "attn_usage": [],
        "mlp_entropy": [], "attn_entropy": [],
        # Per-component orthogonality and balance
        "mlp_ortho": [], "attn_ortho": [], "mos_ortho": [],
        "mlp_bal": [], "attn_bal": [], "mos_bal": [],
    }

    for line in lines:
        # Training steps
        m = re.search(r"^step:(\d+)/\d+ train_loss:([\d.]+).*train_time:(\d+)ms step_avg:([\d.]+)ms", line)
        if m:
            data["train_steps"].append(int(m.group(1)))
            data["train_loss"].append(float(m.group(2)))
            data["train_time_ms"].append(float(m.group(3)))
            data["step_avg_ms"].append(float(m.group(4)))
            # Parse NTP and CTP losses from training lines
            m_ntp = re.search(r"ntp_loss:([\d.]+)", line)
            data["ntp_loss"].append(float(m_ntp.group(1)) if m_ntp else 0.0)
            m_ctp = re.search(r"ctp_loss:([\d.]+)", line)
            data["ctp_loss"].append(float(m_ctp.group(1)) if m_ctp else 0.0)
            # Parse pre-clip gradient norm
            m_gn = re.search(r"grad_norm:([\d.]+)", line)
            data["grad_norm"].append(float(m_gn.group(1)) if m_gn else 0.0)

        # Validation steps
        m = re.search(r"^step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            data["val_steps"].append(int(m.group(1)))
            data["val_loss"].append(float(m.group(2)))
            data["val_bpb"].append(float(m.group(3)))
            # Parse individual DEQ/expert metrics
            for key, pat in [
                ("deq_residual", r"deq_residual:([\d.]+)"),
                ("deq_recon", r"deq_recon_err:([\d.]+)"),
                ("deq_iter_conv", r"deq_iter_conv:([\d.]+)"),
                ("expert_entropy", r"(?<!\w_)expert_entropy:([\d.]+)"),
                ("expert_ortho", r"expert_ortho:([-\d.]+)"),
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
            # Per-component expert usage
            for comp in ("mlp", "attn"):
                m_u = re.search(rf"{comp}_usage:\[([\d.,]+)\]", line)
                if m_u:
                    data[f"{comp}_usage"].append([float(v) for v in m_u.group(1).split(",")])
                else:
                    data[f"{comp}_usage"].append([])
                m_e = re.search(rf"{comp}_entropy:([\d.]+)", line)
                data[f"{comp}_entropy"].append(float(m_e.group(1)) if m_e else 0.0)
            # Per-component orthogonality and balance
            for comp in ("mlp", "attn", "mos"):
                m_o = re.search(rf"{comp}_ortho:([\d.]+)", line)
                data[f"{comp}_ortho"].append(float(m_o.group(1)) if m_o else 0.0)
                m_b = re.search(rf"{comp}_bal:([\d.]+)", line)
                data[f"{comp}_bal"].append(float(m_b.group(1)) if m_b else 0.0)

    return data


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

    # Row 4: Expert diagnostics
    # Load Balance: per-component (MLP, Attn) per-expert lines when available,
    # falling back to combined expert_usage for older logs.
    ax_lb = axes[3, 0]
    _has_per_comp = any(len(u) > 0 for u in b.get("mlp_usage", []) + c.get("mlp_usage", []))

    if _has_per_comp:
        # Per-component expert usage: different colors per component, line styles per expert
        comp_colors = {"mlp": ("#2ca02c", "#98df8a"), "attn": ("#d62728", "#ff9896")}
        line_styles = ["-", "--", ":", "-."]
        for comp in ("mlp", "attn"):
            all_usage = b[f"{comp}_usage"] + c[f"{comp}_usage"]
            max_e = max((len(u) for u in all_usage), default=0)
            b_color, c_color = comp_colors[comp]
            for ei in range(max_e):
                ls = line_styles[ei % len(line_styles)]
                b_vals = [u[ei] if ei < len(u) else 0.0 for u in b[f"{comp}_usage"]]
                c_vals = [u[ei] if ei < len(u) else 0.0 for u in c[f"{comp}_usage"]]
                if b_vals and b["val_steps"]:
                    ax_lb.plot(b["val_steps"], b_vals, color=b_color, linestyle=ls,
                               alpha=0.7, label=f"B {comp} E{ei}", linewidth=1.5)
                if c_vals and c["val_steps"]:
                    ax_lb.plot(c["val_steps"], c_vals, color=c_color, linestyle=ls,
                               alpha=0.7, label=f"C {comp} E{ei}", linewidth=1.5)
    else:
        # Fallback: combined expert_usage (older logs)
        max_experts = max((len(u) for u in b["expert_usage"] + c["expert_usage"]), default=0)
        line_styles = ["-", "--", ":", "-."]
        for ei in range(max_experts):
            b_vals = [u[ei] if ei < len(u) else 0.0 for u in b["expert_usage"]]
            c_vals = [u[ei] if ei < len(u) else 0.0 for u in c["expert_usage"]]
            ls = line_styles[ei % len(line_styles)]
            if b_vals and b["val_steps"]:
                ax_lb.plot(b["val_steps"], b_vals, color=COLOR_BASELINE, linestyle=ls,
                           alpha=0.7, label=f"B Expert {ei}", linewidth=1.5)
            if c_vals and c["val_steps"]:
                ax_lb.plot(c["val_steps"], c_vals, color=COLOR_CURRENT, linestyle=ls,
                           alpha=0.7, label=f"C Expert {ei}", linewidth=1.5)
    ax_lb.set_title("Expert Usage (per Component)", fontsize=11)
    ax_lb.set_xlabel("Step")
    ax_lb.legend(fontsize=6, ncol=2)
    ax_lb.grid(True, alpha=0.3)

    # Expert Entropy: per-component lines
    ax_ent = axes[3, 1]
    comp_colors_ent = {"mlp": "#2ca02c", "attn": "#d62728"}
    for comp, color in comp_colors_ent.items():
        key = f"{comp}_entropy"
        if b[key] and any(v > 0 for v in b[key]):
            ax_ent.plot(b["val_steps"], b[key], color=color, linestyle="-", alpha=0.7,
                       label=f"B {comp}", linewidth=1.5)
        if c[key] and any(v > 0 for v in c[key]):
            ax_ent.plot(c["val_steps"], c[key], color=color, linestyle="--", alpha=0.7,
                       label=f"C {comp}", linewidth=1.5)
    # Fallback to combined entropy
    if not any(v > 0 for v in b.get("mlp_entropy", []) + c.get("mlp_entropy", [])):
        _plot_line(ax_ent, b, c, "expert_entropy", "expert_entropy", "val_steps", "val_steps", "")
    ax_ent.set_title("Expert Entropy (per Component)", fontsize=11)
    ax_ent.set_xlabel("Step")
    ax_ent.legend(fontsize=7)
    ax_ent.grid(True, alpha=0.3)

    # Expert Orthogonality: per-component lines
    ax_ort = axes[3, 2]
    for comp, color in {"mlp": "#2ca02c", "attn": "#d62728", "mos": "#9467bd"}.items():
        key = f"{comp}_ortho"
        if b[key] and any(v > 0 for v in b[key]):
            ax_ort.plot(b["val_steps"], b[key], color=color, linestyle="-", alpha=0.7,
                       label=f"B {comp}", linewidth=1.5)
        if c[key] and any(v > 0 for v in c[key]):
            ax_ort.plot(c["val_steps"], c[key], color=color, linestyle="--", alpha=0.7,
                       label=f"C {comp}", linewidth=1.5)
    if not any(v > 0 for v in b.get("mlp_ortho", []) + c.get("mlp_ortho", [])):
        _plot_line(ax_ort, b, c, "expert_ortho", "expert_ortho", "val_steps", "val_steps", "")
    ax_ort.set_title("Expert Orthogonality (per Component)", fontsize=11)
    ax_ort.set_xlabel("Step")
    ax_ort.legend(fontsize=7)
    ax_ort.grid(True, alpha=0.3)

    # Row 5: Regularization losses (balance, sparsity, conv_loss)
    # Balance loss per component
    ax_bal = axes[4, 0]
    for comp, color in {"mlp": "#2ca02c", "attn": "#d62728", "mos": "#9467bd"}.items():
        key = f"{comp}_bal"
        if b[key] and any(v > 0 for v in b[key]):
            ax_bal.plot(b["val_steps"], b[key], color=color, linestyle="-", alpha=0.7,
                       label=f"B {comp}", linewidth=1.5)
        if c[key] and any(v > 0 for v in c[key]):
            ax_bal.plot(c["val_steps"], c[key], color=color, linestyle="--", alpha=0.7,
                       label=f"C {comp}", linewidth=1.5)
    ax_bal.set_title("Balance Loss (per Component)", fontsize=11)
    ax_bal.set_xlabel("Step")
    ax_bal.legend(fontsize=7)
    ax_bal.grid(True, alpha=0.3)

    # Conv loss (from training lines)
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
