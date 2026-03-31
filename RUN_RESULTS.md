# Run Results

This file records the latest rerun outcomes (numbers + pointers to logs/plots). Keep `EXPERIENCE.md` for general lessons only.

## Latest Rerun

- Branch/commit under test: `03d583d`
- GPUs: `CUDA_VISIBLE_DEVICES=2,3` via `torchrun --nproc_per_node=2`
- Seed: `42`
- Steps: `2000`
- Dataset: `data/datasets/fineweb10B_sp1024` (train shards present: `1`)

**Current (rerun)**
- Log: `logs/current_03d583d_2000_rerun.txt`
- In-training final validation: `val_bpb=1.5043` (from `step:2000/2000 ... val_bpb:...`)
- Post-quant (scored): `val_bpb=1.51095852` (from `final_int6_zstd_roundtrip_exact`)
- Artifact bytes (post-quant): `15233281`

**Baseline (for comparison)**
- Baseline log used by plots: `experiments/training_logs/baseline.log`
- Baseline post-quant (scored): `val_bpb=1.50237571`

## Pending Rerun (Big Batch + Dense Diagnostics)

Recommended command (avoids port collisions via `--standalone`):

`CUDA_VISIBLE_DEVICES=2,3 ITERATIONS=2000 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=10000 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=2 SEED=42 RUN_ID=current_03d583d_2000_bigbatch torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee run.log`

After the run:
- Rotate + regenerate: `bash experiments/update_results.sh run.log`
- Extract key lines: `grep -E \"^step:2000/2000 val_bpb:|^final_int6_.*_roundtrip_exact|^Total submission size int6\" -n run.log`

## Plots

- Baseline vs Current: `experiments/metrics_comparison.png`
- Baseline vs Previous: `experiments/metrics_previous.png`
