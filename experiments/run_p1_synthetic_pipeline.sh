#!/usr/bin/env bash
set -euo pipefail

# Tier-1 P1 pipeline for reports/opg_doc.tex.
# It intentionally runs only the necessary detectability triage:
# vanilla positive control, M0, and M_clk. MoE, cache, and deployment axes live
# in future harnesses after this gate passes.

ITERATIONS="${ITERATIONS:-300}"
NPROC="${NPROC:-2}"
TASK="${TASK:-s5}"
RUN_TAG="${RUN_TAG:-gpu67}"
OUT_DIR="${OUT_DIR:-experiments/training_logs/p1_synthetic_${RUN_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
SEQ_LEN="${SEQ_LEN:-16}"
MODEL_DIM="${MODEL_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
MLP_MULT="${MLP_MULT:-2.0}"
TRAIN_DEPTHS="${TRAIN_DEPTHS:-16,32,64}"
EVAL_DEPTHS="${EVAL_DEPTHS:-4,8,16,32,64}"
PAIRS="${PAIRS:-16,64,8,32}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-25}"
mkdir -p "${OUT_DIR}"

run_variant() {
  local variant="$1"
  local out_json="${OUT_DIR}/${variant}.json"
  echo "p1_pipeline_variant_start:${variant} task=${TASK} iterations=${ITERATIONS} seq_len=${SEQ_LEN} train_depths=${TRAIN_DEPTHS} output=${out_json}"
  torchrun --standalone --nproc_per_node="${NPROC}" experiments/p1_synthetic.py \
    --task "${TASK}" \
    --variant "${variant}" \
    --iterations "${ITERATIONS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-batches "${EVAL_BATCHES}" \
    --seq-len "${SEQ_LEN}" \
    --model-dim "${MODEL_DIM}" \
    --num-heads "${NUM_HEADS}" \
    --mlp-mult "${MLP_MULT}" \
    --train-depths "${TRAIN_DEPTHS}" \
    --eval-depths "${EVAL_DEPTHS}" \
    --pairs "${PAIRS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --grad-clip "${GRAD_CLIP}" \
    --seed "${SEED}" \
    --log-every "${LOG_EVERY}" \
    --output-json "${out_json}"
  echo "p1_pipeline_variant_done:${variant} output=${out_json}"
}

for variant in control m0 mclk; do
  run_variant "${variant}"
done

echo "p1_pipeline_done:${OUT_DIR}"
