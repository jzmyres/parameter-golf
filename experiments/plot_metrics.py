"""Plot detailed training metrics comparing baseline vs current experiment."""
import re
import sys
from pathlib import Path

def parse_log(logpath: str) -> dict:
    """Parse training log for all metrics."""
    lines = Path(logpath).read_text().split("\n")
    data = {"steps": [], "train_loss": [], "val_loss": [], "val_bpb": [],
            "step_avg_ms": [], "deq_residual": [], "deq_recon": [],
            "deq_iter_conv": [],
            "expert_usage_0": [], "expert_usage_1": [], "expert_entropy": [],
            "expert_ortho": [], "train_time_ms": []}

    for line in lines:
        # Training steps
        m = re.search(r"^step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms", line)
        if m:
            data["steps"].append(int(m.group(1)))
            data["train_loss"].append(float(m.group(2)))
            data["train_time_ms"].append(float(m.group(3)))
            data["step_avg_ms"].append(float(m.group(4)))

        # Validation steps (with DEQ and expert metrics)
        m = re.search(r"^step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            data["val_loss"].append(float(m.group(2)))
            data["val_bpb"].append(float(m.group(3)))
            # Parse individual metrics (more robust than one giant regex)
            m2 = re.search(r"deq_residual:([\d.]+)", line)
            data["deq_residual"].append(float(m2.group(1)) if m2 else 0.0)
            m2 = re.search(r"deq_recon_err:([\d.]+)", line)
            data["deq_recon"].append(float(m2.group(1)) if m2 else 0.0)
            m2 = re.search(r"deq_iter_conv:([\d.]+)", line)
            data["deq_iter_conv"].append(float(m2.group(1)) if m2 else 0.0)
            m2 = re.search(r"expert_usage:\[([\d.,]+)\]", line)
            if m2:
                usage = m2.group(1).split(",")
                data["expert_usage_0"].append(float(usage[0]))
                data["expert_usage_1"].append(float(usage[1]) if len(usage) > 1 else 0.0)
            else:
                data["expert_usage_0"].append(0.0)
                data["expert_usage_1"].append(0.0)
            m2 = re.search(r"expert_entropy:([\d.]+)", line)
            data["expert_entropy"].append(float(m2.group(1)) if m2 else 0.0)
            m2 = re.search(r"expert_ortho:([-\d.]+)", line)
            data["expert_ortho"].append(float(m2.group(1)) if m2 else 0.0)

    return data

def plot_comparison(baseline_log: str, current_log: str, outdir: str):
    """Plot baseline vs current experiment comparison."""
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
    fig.suptitle("Baseline vs Current Experiment", fontsize=16, fontweight="bold")

    # 1. Train loss
    if b["steps"] and c["steps"]:
        axes[0, 0].plot(b["steps"], b["train_loss"], "b-", alpha=0.7, label="Baseline")
        axes[0, 0].plot(c["steps"], c["train_loss"], "r-", alpha=0.7, label="Current")
        axes[0, 0].set_title("Train Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Step avg (ms)
    if b["steps"] and c["steps"]:
        axes[0, 1].plot(b["steps"], b["step_avg_ms"], "b-", label="Baseline")
        axes[0, 1].plot(c["steps"], c["step_avg_ms"], "r-", label="Current")
        axes[0, 1].set_title("Step Avg (ms)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Val BPB
    if b["val_bpb"] and c["val_bpb"]:
        axes[0, 2].bar(["Baseline", "Current"], [b["val_bpb"][-1], c["val_bpb"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[0, 2].set_title(f"Val BPB (B={b['val_bpb'][-1]:.4f}, C={c['val_bpb'][-1]:.4f})")
        axes[0, 2].grid(True, alpha=0.3)

    # 4. DEQ Residual
    if b["deq_residual"] and c["deq_residual"]:
        axes[1, 0].bar(["Baseline", "Current"], [b["deq_residual"][-1], c["deq_residual"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[1, 0].set_title(f"DEQ Residual")
        axes[1, 0].grid(True, alpha=0.3)

    # 5. DEQ Reconstruction Error
    if b["deq_recon"] and c["deq_recon"]:
        axes[1, 1].bar(["Baseline", "Current"], [b["deq_recon"][-1], c["deq_recon"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[1, 1].set_title("DEQ Recon Error")
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Expert Usage
    if b["expert_usage_0"] and c["expert_usage_0"]:
        x = [0, 1, 3, 4]
        vals = [b["expert_usage_0"][-1], b["expert_usage_1"][-1],
                c["expert_usage_0"][-1], c["expert_usage_1"][-1]]
        colors = ["#2196F3", "#64B5F6", "#F44336", "#EF9A9A"]
        axes[1, 2].bar(x, vals, color=colors)
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(["B-E0", "B-E1", "C-E0", "C-E1"])
        axes[1, 2].set_title("Expert Usage")
        axes[1, 2].grid(True, alpha=0.3)

    # 7. Expert Entropy
    if b["expert_entropy"] and c["expert_entropy"]:
        axes[2, 0].bar(["Baseline", "Current"], [b["expert_entropy"][-1], c["expert_entropy"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[2, 0].set_title("Expert Entropy")
        axes[2, 0].grid(True, alpha=0.3)

    # 8. Expert Orthogonality
    if b["expert_ortho"] and c["expert_ortho"]:
        axes[2, 1].bar(["Baseline", "Current"], [b["expert_ortho"][-1], c["expert_ortho"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[2, 1].set_title("Expert Orthogonality")
        axes[2, 1].grid(True, alpha=0.3)

    # 9. DEQ Iter Convergence
    if b["deq_iter_conv"] and c["deq_iter_conv"]:
        axes[2, 2].bar(["Baseline", "Current"], [b["deq_iter_conv"][-1], c["deq_iter_conv"][-1]],
                      color=["#2196F3", "#F44336"])
        axes[2, 2].set_title("DEQ Iter Convergence ||z_T - z_{T-1}||")
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].axis("off")

    # 10-12. Summary text (bottom row)
    for j in range(3):
        axes[3, j].axis("off")
    summary = ""
    if b["val_bpb"] and c["val_bpb"]:
        delta = c["val_bpb"][-1] - b["val_bpb"][-1]
        summary += f"Val BPB: {b['val_bpb'][-1]:.4f} → {c['val_bpb'][-1]:.4f} (Δ={delta:+.4f})\n"
    if b["steps"] and c["steps"]:
        summary += f"Steps: {b['steps'][-1]} vs {c['steps'][-1]}\n"
    if b["deq_residual"] and c["deq_residual"]:
        summary += f"DEQ Res: {b['deq_residual'][-1]:.0f} vs {c['deq_residual'][-1]:.0f}\n"
    if b["deq_recon"] and c["deq_recon"]:
        summary += f"Recon Err: {b['deq_recon'][-1]:.4f} vs {c['deq_recon'][-1]:.4f}\n"
    if b["deq_iter_conv"] and c["deq_iter_conv"]:
        summary += f"Iter Conv: {b['deq_iter_conv'][-1]:.4f} vs {c['deq_iter_conv'][-1]:.4f}\n"
    if b["expert_entropy"] and c["expert_entropy"]:
        summary += f"Entropy: {b['expert_entropy'][-1]:.4f} vs {c['expert_entropy'][-1]:.4f}\n"
    axes[3, 1].text(0.1, 0.5, summary, fontsize=12, family="monospace",
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
