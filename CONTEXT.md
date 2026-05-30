# Parameter Golf Context

This file records shared project language that is broader than a single
iteration ledger entry. Operational rules remain in `CLAUDE.md` and
`EXPERIENCE.md`; experiment-specific evidence remains in
`experiments/docs/hypotheses.md`.

## Baseline Vocabulary

| Term | Meaning |
|---|---|
| Active Baseline | The current in-repo comparator, owned by `train_gpt.py::Hyperparameters` and the latest promoted/current logs. As of 2026-05-24 this is the RevDEQ + Parcae iter172-style configuration. |
| Record Baseline | Historical Parameter Golf reproductions and records, especially runs preserved under `records/`, `records_baseline/`, and their logs. These are challenge comparators, not necessarily the current architecture. |
| Comparable Baseline | External research baselines that use recurrent depth, implicit depth, looped transformers, or sparse looped layers to trade extra computation for fewer unique parameters. |

## Current Comparison Question

The active research question is whether Parameter Golf's fixed-point
RevDEQ/Parcae stack should be compared against recurrent-depth and looped-model
families rather than only against conventional fixed-depth transformers. The
baseline inventory for that comparison lives in
`experiments/docs/recurrent_depth_baselines.md`, with replication state tracked
under `baselines/`.
