# Run Results

This file records the latest rerun outcomes (numbers + pointers to logs/plots). Keep `EXPERIENCE.md` for general lessons only.

## Latest Status (Local Iteration Tracking)

Baseline vs current plots are sourced from:
- Baseline: `experiments/training_logs/baseline.log`
- Current: `experiments/training_logs/current.log`

### Discarded (kept for note only)
- `experiments/training_logs/discarded_resid_gate_20260403_033804.log`: post-quant `val_bpb=1.53627464` (worse)

### Baseline (kept)
- `experiments/training_logs/baseline.log`: post-quant `val_bpb=1.47009552`

## Pending Run

Next run will compare Attn+MLP MoE (baseline) vs Block-level MoE (current) via `experiments/metrics_comparison.png`.

## Plots

- Baseline vs Current: `experiments/metrics_comparison.png`
`experiments/metrics_comparison.png` is the only comparison plot.
