# Run Results

This file records the latest rerun outcomes (numbers + pointers to logs/plots). Keep `EXPERIENCE.md` for general lessons only.

## Latest Status (Local Iteration Tracking)

Baseline vs current plots are sourced from:
- Baseline: `experiments/training_logs/baseline.log`
- Current: `experiments/training_logs/current.log`

To extract the latest scored metric from a log:
```bash
grep "val_bpb:" experiments/training_logs/baseline.log | tail -1
grep "val_bpb:" experiments/training_logs/current.log | tail -1
```

## Pending Run

Next run will compare Attn+MLP MoE (baseline) vs Block-level MoE (current) via `experiments/metrics_comparison.png`.

## Plots

- Baseline vs Current: `experiments/metrics_comparison.png`
`experiments/metrics_comparison.png` is the only comparison plot.
