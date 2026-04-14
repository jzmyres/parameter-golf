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

# If RUN_LOG is the torchrun console capture, resolve the actual per-run logfile
# (train_gpt.py prints `logs/<uuid>.txt` on rank0).
if [ -f "$RUN_LOG" ] && [ "$(basename "$RUN_LOG")" = "run.log" ]; then
    hint=$(grep -oE 'logs/[0-9a-fA-F-]+\.txt' "$RUN_LOG" | tail -1 || true)
    if [ -n "${hint:-}" ] && [ -f "$PROJECT_ROOT/$hint" ]; then
        echo "Resolved run log: $RUN_LOG → $hint"
        RUN_LOG="$PROJECT_ROOT/$hint"
    fi
fi

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
    # Refuse to promote if the run was flagged INVALID by post-int6 assertions.
    # The train script writes retry_hint.json and sets run_valid=false in meta.json
    # when any hard assertion fails.  Agent must apply the prescribed fix and rerun.
    CURRENT_META="$WEIGHTS_DIR/current/meta.json"
    if [ -f "$CURRENT_META" ]; then
        run_valid=$(python3 -c "import json; d=json.load(open('$CURRENT_META')); print(d.get('run_valid', True))" 2>/dev/null || echo "True")
        if [ "$run_valid" = "False" ]; then
            echo "✗ REFUSING TO PROMOTE — run is INVALID (post-int6 assertions failed)."
            if [ -f "$WEIGHTS_DIR/current/retry_hint.json" ]; then
                echo "  See experiments/weights/current/retry_hint.json for prescribed fix."
                python3 -c "import json; d=json.load(open('$WEIGHTS_DIR/current/retry_hint.json')); print('  suggested_config:', d.get('suggested_config', {}))"
            fi
            exit 2
        fi
    fi
    # Backup existing baseline before overwriting.
    if [ -f "$LOGDIR/baseline.log" ]; then
        cp "$LOGDIR/baseline.log" "$LOGDIR/baseline_backup.log"
        echo "Backed up: baseline.log → baseline_backup.log"
    fi
    if ls "$WEIGHTS_DIR/baseline/"* 1>/dev/null 2>&1; then
        mkdir -p "$WEIGHTS_DIR/baseline_backup"
        rm -f "$WEIGHTS_DIR/baseline_backup/"*
        cp "$WEIGHTS_DIR/baseline/"* "$WEIGHTS_DIR/baseline_backup/" 2>/dev/null || true
        echo "Backed up: baseline weights + metadata → baseline_backup/"
    fi
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

# Prefer a Python with matplotlib available for plotting.
PLOT_PYTHON=(python)
if ! python -c "import matplotlib" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "deq"; then
        PLOT_PYTHON=(conda run -n deq python)
    fi
fi

if "${PLOT_PYTHON[@]}" experiments/plot_metrics.py 2>/dev/null; then
    echo "Updated metrics_comparison.png"
else
    echo "Warning: plot_metrics.py failed (matplotlib missing?)"
fi

if "${PLOT_PYTHON[@]}" experiments/plot_progress.py 2>/dev/null; then
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
        # Extract key fields from meta.json.  Use .get() with fallbacks so the
        # script never aborts under `set -e` if a key is missing or spelled
        # differently across runs (commit/git_commit, step/steps).
        bpb=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('val_bpb','?'))" 2>/dev/null || echo "?")
        commit=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('git_commit', d.get('commit','?')))" 2>/dev/null || echo "?")
        size=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('artifact_bytes','?'))" 2>/dev/null || echo "?")
        steps=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('steps', d.get('step','?')))" 2>/dev/null || echo "?")
        valid=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('run_valid', True))" 2>/dev/null || echo "?")
        echo "  $d/  val_bpb=$bpb  commit=$commit  artifact=${size}B  steps=$steps  valid=$valid"
    else
        echo "  $d/  (no meta.json)"
    fi
done
echo ""
echo "Done."
