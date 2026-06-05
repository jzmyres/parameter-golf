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
# Architecture-search axes (configurable recurrence-block structure; all
# reversibility-preserving). Each sweep varies ONE structural knob at matched
# config and writes its own `metrics:`/`val_bpb:` summary rows:
# Experiment 4 — block-order in {attn_ffn, ffn_attn, parallel}.
# Experiment 5 — attn-MoE on/off (MoEUT-style attention experts).
# Experiment 6 — DeepSeek shared always-on experts in {0, 1, 2}.
# Experiment 7 — n-experts x n-sublayers layout {16x1, 8x2, 4x4} (matched-ish).
# Experiment 8 — expert-b-init in {small, zero} (zero paired with shared base).
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
# Architecture-search axis sweeps (override via env).
BLOCK_ORDER_SWEEP="${BLOCK_ORDER_SWEEP:-attn_ffn ffn_attn parallel}"
ATTN_MOE_SWEEP="${ATTN_MOE_SWEEP:-off on}"
N_ATTN_EXPERTS="${N_ATTN_EXPERTS:-4}"
SHARED_EXPERTS_SWEEP="${SHARED_EXPERTS_SWEEP:-0 1 2}"
# n-experts x n-sublayers layouts, encoded as "experts:sublayers" pairs.
LAYOUT_SWEEP="${LAYOUT_SWEEP:-16:1 8:2 4:4}"
EXPERT_B_INIT_SWEEP="${EXPERT_B_INIT_SWEEP:-small zero}"
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
  python train_gpt.py \
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
  python train_gpt.py \
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
# fit the effective-depth exponent phi; see train_gpt.py metrics emitters.

# ---------------------------------------------------------------------------
# Experiment 3: MLA kv_latent sweep. Ranks KV-compression latent against
# val_bpb and kv_bytes/token to pick the smallest latent without quality loss.
# ---------------------------------------------------------------------------
for kv_latent in ${KV_LATENT_SWEEP}; do
  log="${OUT_DIR}/kv_latent_${kv_latent}.log"
  echo "m0_control_kv_start: kv_latent=${kv_latent} -> ${log}"
  python train_gpt.py \
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

# Shared matched-config base args for the architecture-search sweeps below.
# Each sweep appends ONLY the structural knob it varies. (DRY: one source for
# the matched config so all axes are compared under the same model size.)
base_args() {
  printf '%s ' \
    --model-dim "${MODEL_DIM}" \
    --n-heads "${N_HEADS}" \
    --n-kv-heads "${N_KV_HEADS}" \
    --expert-rank "${EXPERT_RANK}" \
    --kv-latent "${KV_LATENT}" \
    --k-set "${K_SET}" \
    --seq-len "${SEQ_LEN}" \
    --iterations "${ITERATIONS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
}

# ---------------------------------------------------------------------------
# Experiment 4: block-order in {attn_ffn, ffn_attn, parallel}. Picks how the
# attention and FFN-MoE compose inside one delta sub-block (reversibility-safe).
# ---------------------------------------------------------------------------
for block_order in ${BLOCK_ORDER_SWEEP}; do
  log="${OUT_DIR}/block_order_${block_order}.log"
  echo "m0_control_block_order_start: block_order=${block_order} -> ${log}"
  python train_gpt.py $(base_args) \
    --n-experts "${N_EXPERTS}" \
    --block-order "${block_order}" \
    --artifact-out "${OUT_DIR}/block_order_${block_order}.bin" \
    > "${log}" 2>&1
  echo "m0_control_block_order_done: block_order=${block_order} -> ${log}"
  append_summary "block_order" "block_order=${block_order}" "${log}"
done

# ---------------------------------------------------------------------------
# Experiment 5: attn-MoE on/off. Off = single shared MLA; on = MoEUT-style
# per-token MoE over N_ATTN_EXPERTS low-rank MLA experts (smooth routing).
# ---------------------------------------------------------------------------
for attn_moe in ${ATTN_MOE_SWEEP}; do
  log="${OUT_DIR}/attn_moe_${attn_moe}.log"
  echo "m0_control_attn_moe_start: attn_moe=${attn_moe} -> ${log}"
  attn_moe_flag=""
  if [ "${attn_moe}" = "on" ]; then
    attn_moe_flag="--attn-moe --n-attn-experts ${N_ATTN_EXPERTS}"
  fi
  python train_gpt.py $(base_args) \
    --n-experts "${N_EXPERTS}" \
    ${attn_moe_flag} \
    --artifact-out "${OUT_DIR}/attn_moe_${attn_moe}.bin" \
    > "${log}" 2>&1
  echo "m0_control_attn_moe_done: attn_moe=${attn_moe} -> ${log}"
  append_summary "attn_moe" "attn_moe=${attn_moe}" "${log}"
done

# ---------------------------------------------------------------------------
# Experiment 6: DeepSeek shared always-on experts in {0, 1, 2}. Adds ungated
# base experts summed into every token in addition to the routed experts.
# ---------------------------------------------------------------------------
for num_shared in ${SHARED_EXPERTS_SWEEP}; do
  log="${OUT_DIR}/num_shared_experts_${num_shared}.log"
  echo "m0_control_num_shared_start: num_shared_experts=${num_shared} -> ${log}"
  python train_gpt.py $(base_args) \
    --n-experts "${N_EXPERTS}" \
    --num-shared-experts "${num_shared}" \
    --artifact-out "${OUT_DIR}/num_shared_experts_${num_shared}.bin" \
    > "${log}" 2>&1
  echo "m0_control_num_shared_done: num_shared_experts=${num_shared} -> ${log}"
  append_summary "num_shared_experts" "num_shared_experts=${num_shared}" "${log}"
done

# ---------------------------------------------------------------------------
# Experiment 7: n-experts x n-sublayers layout {16x1, 8x2, 4x4}. Trades more
# experts-in-one-sublayer vs fewer-experts x more-unique-sublayers. Encoded as
# "experts:sublayers" pairs in LAYOUT_SWEEP.
# ---------------------------------------------------------------------------
for layout in ${LAYOUT_SWEEP}; do
  n_experts="${layout%%:*}"
  n_sublayers="${layout##*:}"
  tag="${n_experts}x${n_sublayers}"
  log="${OUT_DIR}/layout_${tag}.log"
  echo "m0_control_layout_start: layout=${tag} (n_experts=${n_experts} n_sublayers=${n_sublayers}) -> ${log}"
  python train_gpt.py $(base_args) \
    --n-experts "${n_experts}" \
    --n-sublayers "${n_sublayers}" \
    --artifact-out "${OUT_DIR}/layout_${tag}.bin" \
    > "${log}" 2>&1
  echo "m0_control_layout_done: layout=${tag} -> ${log}"
  append_summary "layout" "n_experts=${n_experts} n_sublayers=${n_sublayers}" "${log}"
done

# ---------------------------------------------------------------------------
# Experiment 8: expert-b-init in {small, zero}. zero (classic LoRA-B) is only
# sensible with a shared-expert base, so it is paired with --num-shared-experts 1.
# ---------------------------------------------------------------------------
for b_init in ${EXPERT_B_INIT_SWEEP}; do
  log="${OUT_DIR}/expert_b_init_${b_init}.log"
  echo "m0_control_b_init_start: expert_b_init=${b_init} -> ${log}"
  shared_flag=""
  if [ "${b_init}" = "zero" ]; then
    shared_flag="--num-shared-experts 1"
  fi
  python train_gpt.py $(base_args) \
    --n-experts "${N_EXPERTS}" \
    --expert-b-init "${b_init}" \
    ${shared_flag} \
    --artifact-out "${OUT_DIR}/expert_b_init_${b_init}.bin" \
    > "${log}" 2>&1
  echo "m0_control_b_init_done: expert_b_init=${b_init} -> ${log}"
  append_summary "expert_b_init" "expert_b_init=${b_init}" "${log}"
done

echo "m0_control_done: summary=${SUMMARY}"
