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
    # Fail CLOSED: require both run_valid=True AND status=="validated" in meta.json.
    # Any of {missing meta.json, JSON parse error, missing keys, False, "other"}
    # refuses promotion.  The old fail-open default silently promoted half-finished
    # runs whose meta.json was truncated.
    CURRENT_META="$WEIGHTS_DIR/current/meta.json"
    if [ ! -f "$CURRENT_META" ]; then
        echo "✗ REFUSING TO PROMOTE — no meta.json at $CURRENT_META"
        exit 2
    fi
    run_valid=$(python3 - <<PY 2>/dev/null || echo "INVALID"
import json, sys
try:
    d = json.load(open("$CURRENT_META"))
    # NEW POLICY (val_bpb-primary): val_bpb_q is recorded → run_valid=true.
    # Gate failures are tracked as tech debt (status=validated_with_tech_debt)
    # but still allow promotion.  Only refuse if val_bpb wasn't written or
    # status is in_progress / artifact_written (run aborted before final eval).
    valid_statuses = {"validated", "validated_clean", "validated_with_tech_debt"}
    if d.get("run_valid") is True and d.get("status") in valid_statuses:
        print("VALID")
    else:
        print("INVALID")
except Exception:
    print("INVALID")
PY
)
    if [ "$run_valid" != "VALID" ]; then
        echo "✗ REFUSING TO PROMOTE — run is INVALID (post-int6 assertions failed or meta.json malformed)."
        if [ -f "$WEIGHTS_DIR/current/retry_hint.json" ]; then
            echo "  See experiments/weights/current/retry_hint.json for prescribed fix."
            python3 -c "import json; d=json.load(open('$WEIGHTS_DIR/current/retry_hint.json')); print('  suggested_config:', d.get('suggested_config', {}))" 2>/dev/null || true
        fi
        exit 2
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
    if command -v conda >/dev/null 2>&1; then
        if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "opg"; then
            PLOT_PYTHON=(conda run -n opg python)
        elif conda env list 2>/dev/null | awk '{print $1}' | grep -qx "deq"; then
            # Legacy fallback for older environments.
            PLOT_PYTHON=(conda run -n deq python)
        fi
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
extract_aux_terms() {
    python3 - "$1" <<'PY' 2>/dev/null || true
import math
import re
import sys

path = sys.argv[1]
try:
    with open(path, errors="ignore") as f:
        train_lines = [line for line in f if line.startswith("step:") and " train_loss:" in line]
except OSError:
    train_lines = []
if not train_lines:
    raise SystemExit(0)
line = train_lines[-1]
float_pat = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

def value(key):
    match = re.search(rf"{re.escape(key)}:{float_pat}", line)
    return float(match.group(1)) if match else math.nan

pairs = [
    ("rcv", "router_cv_loss", "router_cv_coef_eff"),
    ("rent", "router_entropy_loss", "router_entropy_coef_eff"),
    ("mcv", "mos_cv_loss", "mos_cv_coef_eff"),
    ("ediv", "expert_diversity_loss", "expert_diversity_coef_eff"),
    ("mdiv", "mos_diversity_loss", "mos_diversity_coef_eff"),
]
parts = []
for label, loss_key, coef_key in pairs:
    loss = value(loss_key)
    coef = value(coef_key)
    if math.isfinite(loss) and math.isfinite(coef):
        parts.append(f"{label}={loss * coef:.4g}")
print(" ".join(parts))
PY
}

extract_parcae_state() {
    python3 - "$1" <<'PY' 2>/dev/null || true
import math
import re
import sys

path = sys.argv[1]
try:
    with open(path, errors="ignore") as f:
        train_lines = [line for line in f if line.startswith("step:") and " train_loss:" in line]
except OSError:
    train_lines = []
if not train_lines:
    raise SystemExit(0)
line = train_lines[-1]
float_pat = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

def value(key):
    match = re.search(rf"{re.escape(key)}:{float_pat}", line)
    return float(match.group(1)) if match else math.nan

keys = [
    "parcae_a_bar_min", "parcae_a_bar_mean", "parcae_a_bar_max",
    "parcae_a_bar_core_max",
    "parcae_beta_mean", "parcae_beta_max",
    "parcae_b_bar_mean", "parcae_b_bar_max",
    "parcae_delta_mean", "parcae_delta_max",
    "parcae_recon_amp_log10",
]
vals = {key: value(key) for key in keys}
if not any(math.isfinite(v) for v in vals.values()):
    raise SystemExit(0)

def fmt(key):
    v = vals[key]
    return f"{v:.4g}" if math.isfinite(v) else "?"

parts = [
    f"abar={fmt('parcae_a_bar_min')}/{fmt('parcae_a_bar_mean')}/{fmt('parcae_a_bar_max')}",
    f"core={fmt('parcae_a_bar_core_max')}",
    f"beta={fmt('parcae_beta_mean')}/{fmt('parcae_beta_max')}",
    f"bbar={fmt('parcae_b_bar_mean')}/{fmt('parcae_b_bar_max')}",
    f"delta={fmt('parcae_delta_mean')}/{fmt('parcae_delta_max')}",
    f"amp_log10={fmt('parcae_recon_amp_log10')}",
]
print(" ".join(parts))
PY
}

echo ""
echo "=== Experiment Files ==="
echo "Logs:"
for f in baseline.log previous.log current.log; do
    if [ -f "$LOGDIR/$f" ]; then
        bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$LOGDIR/$f" | tail -1 || true)
        router_reg=$(grep -oP 'router_reg_loss:\K[-+0-9.eE]+' "$LOGDIR/$f" | tail -1 || true)
        aux_terms=$(extract_aux_terms "$LOGDIR/$f")
        parcae_state=$(extract_parcae_state "$LOGDIR/$f")
        echo "  $f  val_bpb=${bpb:-?}  router_reg=${router_reg:-?}  aux_terms=${aux_terms:-?}  parcae=${parcae_state:-?}"
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
