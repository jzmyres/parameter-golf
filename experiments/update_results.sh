#!/bin/bash
# Rotate training logs + model weights and regenerate experiment plots.
# Idempotent: safe to run multiple times — only rotates if run.log is newer than current.log.
#
# Usage:
#   bash experiments/update_results.sh              # rotate logs+weights, regenerate plots
#   bash experiments/update_results.sh --promote    # also promote current → baseline (new best)
#   bash experiments/update_results.sh [run.log]    # custom run.log path
#
# Directory layout:
#   experiments/weights/baseline/   — best config weights (promoted on --promote)
#   experiments/weights/previous/   — last iteration weights (auto-rotated)
#   experiments/weights/current/    — this iteration weights (written by train_gpt.py)
#   experiments/training_logs/baseline.log   — best config training log
#   experiments/training_logs/previous.log   — last iteration log
#   experiments/training_logs/current.log    — this iteration log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGDIR="$SCRIPT_DIR/training_logs"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

# Parse args
PROMOTE=false
RUN_LOG="$PROJECT_ROOT/run.log"
for arg in "$@"; do
    case "$arg" in
        --promote) PROMOTE=true ;;
        *) RUN_LOG="$arg" ;;
    esac
done

mkdir -p "$LOGDIR" "$WEIGHTS_DIR/baseline" "$WEIGHTS_DIR/previous" "$WEIGHTS_DIR/current"

# --- Step 1: Rotate logs (only if run.log is newer than current.log) ---
ROTATED=false
if [ ! -f "$RUN_LOG" ]; then
    echo "Info: $RUN_LOG not found — skipping log rotation, regenerating plots only."
elif [ -f "$LOGDIR/current.log" ] && [ "$RUN_LOG" -ot "$LOGDIR/current.log" ]; then
    echo "Skipping rotation: $RUN_LOG is older than current.log (already rotated)."
elif cmp -s "$RUN_LOG" "$LOGDIR/current.log" 2>/dev/null; then
    echo "Skipping rotation: $RUN_LOG is identical to current.log (idempotent)."
else
    # Rotate logs: current → previous
    if [ -f "$LOGDIR/current.log" ]; then
        cp "$LOGDIR/current.log" "$LOGDIR/previous.log"
        echo "Rotated logs: current.log → previous.log"
    fi
    cp "$RUN_LOG" "$LOGDIR/current.log"
    echo "Saved $RUN_LOG → current.log"

    # Rotate weights + metadata: current → previous
    if ls "$WEIGHTS_DIR/current/"* 1>/dev/null 2>&1; then
        rm -f "$WEIGHTS_DIR/previous/"*
        cp "$WEIGHTS_DIR/current/"* "$WEIGHTS_DIR/previous/" 2>/dev/null || true
        echo "Rotated weights: current/ → previous/"
    fi
    ROTATED=true
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
if ! ls "$WEIGHTS_DIR/baseline/"* 1>/dev/null 2>&1; then
    if ls "$WEIGHTS_DIR/current/"* 1>/dev/null 2>&1; then
        cp "$WEIGHTS_DIR/current/"* "$WEIGHTS_DIR/baseline/"
        echo "No baseline weights — initialized from current"
    fi
fi

# --- Step 3: Promote current → baseline (if --promote) ---
if [ "$PROMOTE" = true ]; then
    if [ -f "$LOGDIR/current.log" ]; then
        cp "$LOGDIR/current.log" "$LOGDIR/baseline.log"
        echo "Promoted: current.log → baseline.log"
    fi
    if ls "$WEIGHTS_DIR/current/"* 1>/dev/null 2>&1; then
        rm -f "$WEIGHTS_DIR/baseline/"*
        cp "$WEIGHTS_DIR/current/"* "$WEIGHTS_DIR/baseline/"
        echo "Promoted: current weights + metadata → baseline"
    fi
fi

# --- Step 4: Regenerate plots ---
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

# --- Summary ---
echo ""
echo "=== Experiment Files ==="
echo "Logs:"
for f in baseline.log previous.log current.log; do
    if [ -f "$LOGDIR/$f" ]; then
        bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$LOGDIR/$f" | tail -1)
        echo "  $f  val_bpb=${bpb:-?}"
    fi
done
echo "Weights:"
for d in baseline previous current; do
    meta="$WEIGHTS_DIR/$d/meta.json"
    if [ -f "$meta" ]; then
        # Extract key fields from meta.json
        bpb=$(python3 -c "import json; print(json.load(open('$meta'))['val_bpb'])" 2>/dev/null)
        commit=$(python3 -c "import json; print(json.load(open('$meta'))['git_commit'])" 2>/dev/null)
        size=$(python3 -c "import json; print(json.load(open('$meta'))['artifact_bytes'])" 2>/dev/null)
        steps=$(python3 -c "import json; print(json.load(open('$meta'))['steps'])" 2>/dev/null)
        echo "  $d/  val_bpb=$bpb  commit=$commit  artifact=${size}B  steps=$steps"
    else
        echo "  $d/  (no meta.json)"
    fi
done
echo ""
echo "Done."
