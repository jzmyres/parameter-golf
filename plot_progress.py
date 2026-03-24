"""Generate autoresearch progress plot from results.tsv."""
import sys
from pathlib import Path

def main():
    tsv = Path("results.tsv")
    if not tsv.exists():
        print("No results.tsv found")
        return

    lines = tsv.read_text().strip().split("\n")
    if len(lines) < 2:
        print("No results to plot")
        return

    header = lines[0].split("\t")
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        xs_keep, ys_keep = [], []
        xs_discard, ys_discard = [], []
        xs_crash, ys_crash = [], []
        best_bpb = float("inf")
        best_xs, best_ys = [], []

        for i, (bpb, status, desc) in enumerate(rows):
            if status == "keep":
                xs_keep.append(i + 1)
                ys_keep.append(bpb)
                if bpb < best_bpb:
                    best_bpb = bpb
                best_xs.append(i + 1)
                best_ys.append(best_bpb)
            elif status == "crash":
                xs_crash.append(i + 1)
                ys_crash.append(max(ys_keep) if ys_keep else 2.0)
            else:
                xs_discard.append(i + 1)
                ys_discard.append(bpb)

        # Plot best-so-far line
        if best_xs:
            ax.step(best_xs, best_ys, where="post", color="#2196F3", linewidth=2.5,
                    label="Best so far", zorder=3)

        # Plot individual experiments
        ax.scatter(xs_keep, ys_keep, c="#4CAF50", s=60, zorder=4, label="Keep (improved)")
        ax.scatter(xs_discard, ys_discard, c="#F44336", s=40, alpha=0.6, zorder=2, label="Discard (worse)")
        if xs_crash:
            ax.scatter(xs_crash, ys_crash, c="#FF9800", s=80, marker="x", zorder=5, label="Crash/OOM")

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("val_bpb (post-quantization)", fontsize=12)
        ax.set_title("Parameter Golf Autoresearch Progress", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add annotation for best result
        if best_xs:
            ax.annotate(f"Best: {best_bpb:.4f}",
                       xy=(best_xs[-1], best_ys[-1]),
                       xytext=(best_xs[-1] + 2, best_ys[-1] + 0.02),
                       fontsize=10, fontweight="bold", color="#2196F3",
                       arrowprops=dict(arrowstyle="->", color="#2196F3"))

        plt.tight_layout()
        plt.savefig("progress.png", dpi=150)
        print(f"Saved progress.png ({len(rows)} iterations, best={best_bpb:.4f})")

    except ImportError:
        # Fallback: text-based progress display
        print("matplotlib not available, showing text summary:")
        print(f"{'Iter':>4} {'val_bpb':>8} {'Status':>8} Description")
        print("-" * 60)
        best = float("inf")
        for i, (bpb, status, desc) in enumerate(rows):
            marker = ""
            if status == "keep" and bpb < best:
                best = bpb
                marker = " <-- NEW BEST"
            print(f"{i+1:>4} {bpb:>8.4f} {status:>8} {desc[:40]}{marker}")
        print(f"\nBest val_bpb: {best:.4f}")


if __name__ == "__main__":
    main()
