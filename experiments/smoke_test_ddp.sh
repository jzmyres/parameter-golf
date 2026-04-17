#!/bin/bash
# DDP+compile smoke test: run 10 training steps with torchrun to catch
# compile/DDP bugs that single-GPU smoke_test.py cannot detect.
# MUST PASS before committing to a full training run.
#
# Usage: conda activate opg && bash experiments/smoke_test_ddp.sh

set -euo pipefail

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 2 ]; then
    echo "SKIP: DDP smoke test requires ≥2 GPUs (found $NGPU)"
    exit 0
fi

echo "=== DDP+compile smoke test ($NGPU GPUs, 10 steps) ==="

# Override iterations to 10 steps, disable wallclock limit
torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py \
    --iterations=10 \
    --max-wallclock-seconds=0 \
    --val-loss-every=10 \
    2>&1 | tee /tmp/smoke_ddp.log

# Check for crashes
if grep -qE "Error|SIGSEG|SIGKILL|NaN" /tmp/smoke_ddp.log; then
    echo "SMOKE TEST DDP FAILED — see /tmp/smoke_ddp.log"
    exit 1
fi

# Check that step 10 was reached
if ! grep -q "^step:10/" /tmp/smoke_ddp.log; then
    echo "SMOKE TEST DDP FAILED — did not reach step 10"
    exit 1
fi

echo "=== DDP+compile smoke test PASSED ==="
