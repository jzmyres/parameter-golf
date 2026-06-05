#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# M0 control-experiment runner — the empirical gate that decides the router
# and the ranks for the final minimal P1 (reports/opg_doc.tex). It executes
# the three matched control experiments specified in the spec and aggregates
# their per-run `metrics:` / `val_bpb:` lines into one summary TSV.
#
# The two project-facing goals these experiments feed:
#   Goal 1 (quality/efficiency): val_bpb at matched config, plus R_act
#     (active expert fraction) and erank from the `metrics:` line.
#   Goal 2 (effective depth): phi(r) — the effective-depth exponent fit from
#     per-recurrence-count r validation losses (fit_phi consumes them).
#
# Experiment 1 — ReLU vs softmax router (matched config). Decides router_type
#   by comparing val_bpb / R_act under an otherwise identical model.
# Experiment 2 — Effective depth phi over recurrence counts r in {1,2,4,8}.
#   Each r is a single-element --k-set run; fit_phi turns the per-r val losses
#   into the phi exponent (added to a `metrics:` line via --R-act/--phi later).
# Experiment 3 — MLA kv_latent sweep. Ranks the KV-compression latent against
#   val_bpb and kv_bytes/token to pick the smallest latent without quality loss.
#
# GPU runs default to CUDA_VISIBLE_DEVICES=7 (overridable). Heavy training is
# intentionally NOT run by the test suite; this script is for real runs.
# ---------------------------------------------------------------------------

# --- Env knobs with defaults ------------------------------------------------
ITERATIONS="${ITERATIONS:-1000}"
SEQ_LEN="${SEQ_LEN:-512}"
MODEL_DIM="${MODEL_DIM:-768}"
N_HEADS="${N_HEADS:-8}"
N_KV_HEADS="${N_KV_HEADS:-4}"
N_EXPERTS="${N_EXPERTS:-16}"
EXPERT_RANK="${EXPERT_RANK:-32}"
KV_LATENT="${KV_LATENT:-128}"
KV_LATENT_SWEEP="${KV_LATENT_SWEEP:-32 64 128 256}"
K_SET="${K_SET:-32,64,128}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
# GPU runs default to GPU 7 per the M0 task conventions; override with
# CUDA_VISIBLE_DEVICES=... bash experiments/run_m0_control_experiments.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
RUN_TAG="${RUN_TAG:-m0_control}"
OUT_DIR="${OUT_DIR:-experiments/training_logs/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.tsv"
printf 'experiment\trun\tval_bpb\tmetrics\n' > "${SUMMARY}"

# Append one summary row by grepping the run's log. The `|| true` plus `?`
# default keeps a missing match from aborting under `set -euo pipefail`
# (per EXPERIENCE.md#explicit-boundary).
append_summary() {
  local experiment="$1"
  local run="$2"
  local log="$3"
  local bpb metrics
  bpb=$(grep -o 'val_bpb:[0-9.]*' "${log}" | tail -1 || true)
  bpb=${bpb:-?}
  metrics=$(grep '^metrics:' "${log}" | tail -1 || true)
  metrics=${metrics:-?}
  printf '%s\t%s\t%s\t%s\n' "${experiment}" "${run}" "${bpb}" "${metrics}" >> "${SUMMARY}"
}

# ---------------------------------------------------------------------------
# Experiment 1: ReLU-vs-softmax router (matched config).
# Labels print router_type=relu / router_type=softmax; the flag uses the var.
# ---------------------------------------------------------------------------
for router_type in relu softmax; do
  log="${OUT_DIR}/router_${router_type}.log"
  echo "m0_control_router_start: router_type=${router_type} (matched config) -> ${log}"
  python train_gpt_m0.py \
    --router-type "${router_type}" \
    --model-dim "${MODEL_DIM}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --n-experts "${N_EXPERTS}" \
    --expert-rank "${EXPERT_RANK}" \
    --kv-latent "${KV_LATENT}" \
    --k-set "${K_SET}" \
    --seq-len "${SEQ_LEN}" \
    --iterations "${ITERATIONS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --artifact-out "${OUT_DIR}/router_${router_type}.bin" \
    > "${log}" 2>&1
  echo "m0_control_router_done: router_type=${router_type} -> ${log}"
  append_summary "router" "router_type=${router_type}" "${log}"
done

# ---------------------------------------------------------------------------
# Experiment 2: effective-depth phi(r) over recurrence counts r in {1,2,4,8}.
# Each run uses a single-element --k-set so the model is evaluated at exactly r
# recurrences; fit_phi then consumes the per-r val losses to fit the phi
# exponent (effective-depth scaling).
# ---------------------------------------------------------------------------
for r in 1 2 4 8; do
  log="${OUT_DIR}/phi_r${r}.log"
  echo "m0_control_phi_start: r=${r} -> ${log}"
  python train_gpt_m0.py \
    --k-set "${r}" \
    --k-eval "${r}" \
    --model-dim "${MODEL_DIM}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --n-experts "${N_EXPERTS}" \
    --expert-rank "${EXPERT_RANK}" \
    --kv-latent "${KV_LATENT}" \
    --seq-len "${SEQ_LEN}" \
    --iterations "${ITERATIONS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --artifact-out "${OUT_DIR}/phi_r${r}.bin" \
    > "${log}" 2>&1
  echo "m0_control_phi_done: r=${r} -> ${log}"
  append_summary "phi" "r=${r}" "${log}"
done
# fit_phi consumes the per-r val losses logged above (phi_r{1,2,4,8}.log) to
# fit the effective-depth exponent phi; see train_gpt_m0 metrics emitters.

# ---------------------------------------------------------------------------
# Experiment 3: MLA kv_latent sweep. Ranks KV-compression latent against
# val_bpb and kv_bytes/token to pick the smallest latent without quality loss.
# ---------------------------------------------------------------------------
for kv_latent in ${KV_LATENT_SWEEP}; do
  log="${OUT_DIR}/kv_latent_${kv_latent}.log"
  echo "m0_control_kv_start: kv_latent=${kv_latent} -> ${log}"
  python train_gpt_m0.py \
    --kv-latent "${kv_latent}" \
    --model-dim "${MODEL_DIM}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --n-experts "${N_EXPERTS}" \
    --expert-rank "${EXPERT_RANK}" \
    --k-set "${K_SET}" \
    --seq-len "${SEQ_LEN}" \
    --iterations "${ITERATIONS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --artifact-out "${OUT_DIR}/kv_latent_${kv_latent}.bin" \
    > "${log}" 2>&1
  echo "m0_control_kv_done: kv_latent=${kv_latent} -> ${log}"
  append_summary "kv_latent" "kv_latent=${kv_latent}" "${log}"
done

echo "m0_control_done: summary=${SUMMARY}"
