"""Plot detailed training metrics comparing baseline vs current experiment.

Shows full training curves for ALL diagnostic metrics:
- Row 1: Train Loss, Val BPB, Step Avg (ms)
- Row 2: DEQ Residual, DEQ Recon Error, DEQ Iter Convergence
- Row 3: Expert Usage (per expert), Expert Entropy, Expert Orthogonality
- Row 4: Summary text with final values comparison
"""
import re
import sys
from pathlib import Path

def parse_log(logpath: str) -> dict:
    """Parse training log for all metrics."""
    lines = Path(logpath).read_text().split("\n")
    data = {
        "train_steps": [], "train_loss": [], "step_avg_ms": [], "train_time_ms": [],
        "val_steps": [], "val_loss": [], "val_bpb": [],
        "deq_residual": [], "deq_recon": [], "deq_iter_conv": [],
        "expert_usage_0": [], "expert_usage_1": [], "expert_entropy": [],
        "expert_ortho": [],
    }

    for line in lines:
        # Training steps
        m = re.search(r"^step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms", line)
        if m:
            data["train_steps"].append(int(m.group(1)))
            data["train_loss"].append(float(m.group(2)))
            data["train_time_ms"].append(float(m.group(3)))
            data["step_avg_ms"].append(float(m.group(4)))

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
                ("expert_entropy", r"expert_entropy:([\d.]+)"),
                ("expert_ortho", r"expert_ortho:([-\d.]+)"),
            ]:
                m2 = re.search(pat, line)
                data[key].append(float(m2.group(1)) if m2 else 0.0)
            m2 = re.search(r"expert_usage:\[([\d.,]+)\]", line)
            if m2:
                usage = m2.group(1).split(",")
                data["expert_usage_0"].append(float(usage[0]))
                data["expert_usage_1"].append(float(usage[1]) if len(usage) > 1 else 0.0)
            else:
                data["expert_usage_0"].append(0.0)
                data["expert_usage_1"].append(0.0)

    return data


def _plot_line(ax, b, c, b_key, c_key, b_steps, c_steps, title, ylabel=None):
    """Plot two line series on the same axis."""
    if b[b_key] and c[c_key]:
        ax.plot(b[b_steps], b[b_key], "b-", alpha=0.7, label="Baseline", linewidth=1.5)
        ax.plot(c[c_steps], c[c_key], "r-", alpha=0.7, label="Current", linewidth=1.5)
    elif b[b_key]:
        ax.plot(b[b_steps], b[b_key], "b-", alpha=0.7, label="Baseline", linewidth=1.5)
    elif c[c_key]:
        ax.plot(c[c_steps], c[c_key], "r-", alpha=0.7, label="Current", linewidth=1.5)
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("Baseline vs Current Experiment — Full Diagnostics", fontsize=16, fontweight="bold")

    # Row 1: Training metrics
    _plot_line(axes[0, 0], b, c, "train_loss", "train_loss", "train_steps", "train_steps", "Train Loss")
    _plot_line(axes[0, 1], b, c, "val_bpb", "val_bpb", "val_steps", "val_steps", "Val BPB")
    _plot_line(axes[0, 2], b, c, "step_avg_ms", "step_avg_ms", "train_steps", "train_steps", "Step Avg (ms)")

    # Row 2: DEQ diagnostics
    _plot_line(axes[1, 0], b, c, "deq_residual", "deq_residual", "val_steps", "val_steps",
               "DEQ Residual ||z - f(z)||")
    _plot_line(axes[1, 1], b, c, "deq_recon", "deq_recon", "val_steps", "val_steps",
               "DEQ Reconstruction Error")
    _plot_line(axes[1, 2], b, c, "deq_iter_conv", "deq_iter_conv", "val_steps", "val_steps",
               "DEQ Iter Conv ||z_T - z_{T-1}||")

    # Row 3: Expert diagnostics
    # Expert usage per expert (line plot over val steps)
    ax_usage = axes[2, 0]
    if b["expert_usage_0"]:
        ax_usage.plot(b["val_steps"], b["expert_usage_0"], "b-", label="B-E0", linewidth=1.5)
        ax_usage.plot(b["val_steps"], b["expert_usage_1"], "b--", label="B-E1", linewidth=1.5)
    if c["expert_usage_0"]:
        ax_usage.plot(c["val_steps"], c["expert_usage_0"], "r-", label="C-E0", linewidth=1.5)
        ax_usage.plot(c["val_steps"], c["expert_usage_1"], "r--", label="C-E1", linewidth=1.5)
    ax_usage.set_title("Expert Usage", fontsize=11)
    ax_usage.set_xlabel("Step")
    ax_usage.legend(fontsize=8)
    ax_usage.grid(True, alpha=0.3)

    _plot_line(axes[2, 1], b, c, "expert_entropy", "expert_entropy", "val_steps", "val_steps",
               "Expert Entropy")
    _plot_line(axes[2, 2], b, c, "expert_ortho", "expert_ortho", "val_steps", "val_steps",
               "Expert Orthogonality (cos sim)")

    # Row 4: Summary text
    for j in range(3):
        axes[3, j].axis("off")

    summary_lines = []
    if b["val_bpb"] and c["val_bpb"]:
        delta = c["val_bpb"][-1] - b["val_bpb"][-1]
        summary_lines.append(f"Val BPB:    {b['val_bpb'][-1]:.4f} → {c['val_bpb'][-1]:.4f} (Δ={delta:+.4f})")
    if b["train_steps"] and c["train_steps"]:
        summary_lines.append(f"Steps:      {b['train_steps'][-1]} vs {c['train_steps'][-1]}")
    if b["train_loss"] and c["train_loss"]:
        summary_lines.append(f"Train Loss: {b['train_loss'][-1]:.4f} vs {c['train_loss'][-1]:.4f}")
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
    axes[3, 1].text(0.0, 0.5, summary, fontsize=11, family="monospace",
                   verticalalignment="center", transform=axes[3, 1].transAxes)

    plt.tight_layout()
    plt.savefig(str(Path(outdir) / "metrics_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved metrics_comparison.png")


if __name__ == "__main__":
    expdir = Path(__file__).resolve().parent
    baseline = expdir / "training_logs" / "baseline.log"
    current = expdir / "training_logs" / "current.log"

    if not baseline.exists():
        print(f"No baseline log at {baseline}")
        sys.exit(1)
    if not current.exists():
        print("No current log — using baseline for both")
        current = baseline

    plot_comparison(str(baseline), str(current), str(expdir))
