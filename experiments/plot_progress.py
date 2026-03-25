"""Generate autoresearch progress plot from results.tsv.

Generates two plots:
1. progress.png — constrained-only iterations (satisfying RevDEQ+MLA+SoftDense)
2. progress_full.png — all iterations including unconstrained exploration
"""
from pathlib import Path

# Constrained iterations start at this index (0-based) in results.tsv
CONSTRAINED_START_KEYWORD = "constraint"  # rows with this keyword in description


def plot_subset(rows, start_idx, title, outpath, annotate_best=True):
    """Plot a subset of results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    xs_keep, ys_keep = [], []
    xs_discard, ys_discard = [], []
    xs_crash, ys_crash = [], []
    best_bpb = float("inf")
    best_xs, best_ys = [], []

    for i, (bpb, status, desc) in enumerate(rows):
        x = start_idx + i + 1
        if status == "keep":
            xs_keep.append(x)
            ys_keep.append(bpb)
            if bpb < best_bpb:
                best_bpb = bpb
            best_xs.append(x)
            best_ys.append(best_bpb)
        elif status == "crash":
            xs_crash.append(x)
            ys_crash.append(max(ys_keep) if ys_keep else 2.0)
        else:
            xs_discard.append(x)
            ys_discard.append(bpb)

    if best_xs:
        ax.step(best_xs, best_ys, where="post", color="#2196F3", linewidth=2.5,
                label="Best so far", zorder=3)
    ax.scatter(xs_keep, ys_keep, c="#4CAF50", s=60, zorder=4, label="Keep (improved)")
    ax.scatter(xs_discard, ys_discard, c="#F44336", s=40, alpha=0.6, zorder=2, label="Discard (worse)")
    if xs_crash:
        ax.scatter(xs_crash, ys_crash, c="#FF9800", s=80, marker="x", zorder=5, label="Crash/OOM")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("val_bpb (post-quantization)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    if annotate_best and best_xs:
        ax.annotate(f"Best: {best_bpb:.4f}",
                   xy=(best_xs[-1], best_ys[-1]),
                   xytext=(best_xs[-1] + 1, best_ys[-1] + 0.02),
                   fontsize=10, fontweight="bold", color="#2196F3",
                   arrowprops=dict(arrowstyle="->", color="#2196F3"))

    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    return best_bpb, len(rows)


def main():
    script_dir = Path(__file__).resolve().parent
    tsv = script_dir / "results.tsv"
    if not tsv.exists():
        print("No results.tsv found")
        return

    lines = tsv.read_text().strip().split("\n")
    if len(lines) < 2:
        print("No results to plot")
        return

    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 4:
            try:
                bpb = float(parts[1])
                status = parts[3]
                desc = parts[4] if len(parts) > 4 else ""
                rows.append((bpb, status, desc))
            except ValueError:
                continue

    if not rows:
        print("No valid results")
        return

    try:
        # Find where constrained iterations start
        constrained_idx = None
        for i, (bpb, status, desc) in enumerate(rows):
            if CONSTRAINED_START_KEYWORD in desc.lower():
                constrained_idx = i
                break

        # Plot constrained-only (main plot)
        if constrained_idx is not None:
            constrained_rows = rows[constrained_idx:]
            best, n = plot_subset(
                constrained_rows, constrained_idx,
                "Parameter Golf — Constrained Architecture (RevDEQ + MLA + SoftDense)",
                script_dir / "progress.png"
            )
            print(f"Saved progress.png (constrained: {n} iters, best={best:.4f})")
        else:
            # No constrained rows found, plot all
            best, n = plot_subset(rows, 0, "Parameter Golf Autoresearch Progress",
                                  script_dir / "progress.png")
            print(f"Saved progress.png ({n} iterations, best={best:.4f})")

        # Plot full history
        best_full, n_full = plot_subset(rows, 0, "Parameter Golf — Full Experiment History",
                                        script_dir / "progress_full.png")
        print(f"Saved progress_full.png ({n_full} iterations, best={best_full:.4f})")

    except ImportError:
        print("matplotlib not available")
        best = float("inf")
        for i, (bpb, status, desc) in enumerate(rows):
            marker = " <-- NEW BEST" if status == "keep" and bpb < best else ""
            if status == "keep" and bpb < best:
                best = bpb
            print(f"{i+1:>4} {bpb:>8.4f} {status:>8} {desc[:40]}{marker}")
        print(f"\nBest val_bpb: {best:.4f}")


if __name__ == "__main__":
    main()
