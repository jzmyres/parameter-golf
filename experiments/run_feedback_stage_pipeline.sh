#!/usr/bin/env bash
set -euo pipefail

# Full feedback-stage runner for reports/feedback.tex and reports/opg_doc.tex.
# The core P1 runner remains minimal; this script composes S+0 with separate
# S+1/S+2/S+3 diagnostics without adding those surfaces to p1_synthetic.py.

ITERATIONS="${ITERATIONS:-300}"
NPROC="${NPROC:-2}"
TASK="${TASK:-s5}"
RUN_TAG="${RUN_TAG:-gpu67_feedback}"
OUT_DIR="${OUT_DIR:-experiments/training_logs/p1_feedback_${RUN_TAG}}"
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
NUM_EXPERTS="${NUM_EXPERTS:-4}"
ROUTER_TOP_R="${ROUTER_TOP_R:-0}"
MOE_BALANCE_COEF="${MOE_BALANCE_COEF:-0.01}"
mkdir -p "${OUT_DIR}"

run_core_variant() {
  local variant="$1"
  local out_json="${OUT_DIR}/s0_${variant}.json"
  echo "feedback_pipeline_s0_start:${variant} output=${out_json}"
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
  echo "feedback_pipeline_s0_done:${variant} output=${out_json}"
}

run_stage_variant() {
  local stage="$1"
  local variant="$2"
  local out_json="${OUT_DIR}/${stage}_${variant}.json"
  echo "feedback_pipeline_stage_start:${stage}:${variant} output=${out_json}"
  torchrun --standalone --nproc_per_node="${NPROC}" experiments/p1_feedback_stages.py \
    --stage "${stage}" \
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
    --num-experts "${NUM_EXPERTS}" \
    --router-top-r "${ROUTER_TOP_R}" \
    --moe-balance-coef "${MOE_BALANCE_COEF}" \
    --output-json "${out_json}"
  echo "feedback_pipeline_stage_done:${stage}:${variant} output=${out_json}"
}

for variant in control m0 mclk; do
  run_core_variant "${variant}"
done

run_stage_variant s1 m0
run_stage_variant s1 moe
run_stage_variant s1 static_moe
run_stage_variant s2 m0
run_stage_variant s3 m0

echo "feedback_pipeline_done:${OUT_DIR}"
