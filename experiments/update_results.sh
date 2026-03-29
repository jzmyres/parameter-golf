#!/bin/bash
# Rotate training logs and regenerate experiment plots.
# Idempotent: safe to run multiple times — only rotates if run.log is newer than current.log.
#
# Usage: bash experiments/update_results.sh [run.log path]
#   If no arg given, defaults to ./run.log
#
# File layout after running:
#   experiments/training_logs/baseline.log  — best config (manually updated on new best)
#   experiments/training_logs/previous.log  — last iteration's log (auto-rotated)
#   experiments/training_logs/current.log   — this iteration's log
#   experiments/metrics_comparison.png      — baseline vs current (4x3 grid)
#   experiments/metrics_previous.png        — baseline vs previous (if previous exists)
#   experiments/progress.png                — val_bpb over iterations
#   experiments/progress_full.png           — full experiment history

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGDIR="$SCRIPT_DIR/training_logs"
RUN_LOG="${1:-$PROJECT_ROOT/run.log}"

mkdir -p "$LOGDIR"

# --- Step 1: Rotate logs (only if run.log is newer than current.log) ---
if [ ! -f "$RUN_LOG" ]; then
    echo "Warning: $RUN_LOG not found — skipping log rotation, regenerating plots only."
elif [ -f "$LOGDIR/current.log" ] && [ "$RUN_LOG" -ot "$LOGDIR/current.log" ]; then
    echo "Skipping rotation: $RUN_LOG is older than current.log (already rotated)."
elif cmp -s "$RUN_LOG" "$LOGDIR/current.log" 2>/dev/null; then
    echo "Skipping rotation: $RUN_LOG is identical to current.log (idempotent)."
else
    # Rotate: current → previous
    if [ -f "$LOGDIR/current.log" ]; then
        cp "$LOGDIR/current.log" "$LOGDIR/previous.log"
        echo "Rotated current.log → previous.log"
    fi
    cp "$RUN_LOG" "$LOGDIR/current.log"
    echo "Saved $RUN_LOG → current.log"
fi

# --- Step 2: Ensure baseline exists ---
if [ ! -f "$LOGDIR/baseline.log" ]; then
    if [ -f "$LOGDIR/current.log" ]; then
        cp "$LOGDIR/current.log" "$LOGDIR/baseline.log"
        echo "No baseline.log found — initialized from current.log"
    else
        echo "Error: No baseline.log and no current.log. Run a training first."
        exit 1
    fi
fi

# --- Step 3: Regenerate plots ---
cd "$PROJECT_ROOT"

if python experiments/plot_metrics.py 2>/dev/null; then
    echo "Updated metrics_comparison.png"
else
    echo "Warning: plot_metrics.py failed (matplotlib missing?)"
fi

if python experiments/plot_progress.py 2>/dev/null; then
    echo "Updated progress.png"
else
    echo "Warning: plot_progress.py failed"
fi

echo "Done."
