# Experiment Docs

This directory is the canonical home for iteration-level research documents.

Before starting any iteration, review `experiments/docs/` comprehensively:
read this README and `hypotheses.md`, list the directory, and inspect any
relevant archive or `iterNNN_*.md` design notes. The goal is to catch active
queue decisions, superseded ideas, and implementation plans before touching
code or launching training.

- `hypotheses.md` is the active iteration ledger. Its current snapshot is the
  final-minimal P1 pipeline. The Tier-1 runner exposes only the
  `control`/`m0`/`mclk` triage. MoE, cache, and deployment diagnostics live in
  `experiments/p1_feedback_stages.py` and `experiments/run_feedback_stage_pipeline.sh`;
  they are separate later-stage diagnostics and do not widen the P1 runner.
  Dense-MoE/Parcae/RevDEQ queue material in that file is legacy evidence unless
  explicitly pulled forward by `reports/opg_doc.tex`.
- `hypotheses_archive.md` preserves historical records and long-form evidence
  that no longer belongs in the active ledger.
- `recurrent_depth_baselines.md` summarizes external recurrent-depth baselines
  and points to `baselines/` for source pins and replication status.
- `iterNNN_*.md` files are focused design or implementation plans for a single
  iteration.

Keep generated artifacts outside this directory. Checkpoints, logs, plots,
weights, and smoke outputs stay in their existing `experiments/` subfolders.

When closing or promoting an iteration, update `hypotheses.md` first. If the
change needs a reusable design note, add or update an `iterNNN_*.md` file here
and link it from the active ledger.
