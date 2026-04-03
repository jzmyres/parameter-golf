# EXPERIENCE — Lessons Learned

Short, reusable guardrails to avoid common experiment mistakes.

## Metrics
- Track and compare the *scored* metric (post-quant) separately from any in-training validation.
- Always report both the final in-training validation and the final post-quant result.
- Name metrics by what they actually measure; don’t reuse a proxy metric under a different label.
- Prefer enforcing hard constraints in the space you diagnose; remove proxy regularizers if they don’t transfer.
- Treat expert health metrics as non-negotiable guardrails; optimize everything else inside that envelope.
- Validate hard constraints in eval-mode; train-mode “healthy routing” can be misleading.
- Don’t train on a diagnostic unless it consistently improves the scored metric; keep “convergence” as a monitored signal, not a loss.

## Logging
- Treat the primary run log as the source of truth; don’t rely on wrapper tools capturing stdout/stderr.
- For progress monitoring, tail the run log directly.
- Avoid logging stale diagnostics outside their natural cadence (train-step vs val-step).
- Make ablations an explicit, logged flag so runs remain comparable and reproducible.
- Never append multiple runs into a single comparison log; truncate logs per run or make parsers select the last run.

## Validation Curves
- If you want plots with curves, make sure the run actually logs multiple validation points (set a non-zero validation interval).
- For in-training curves, validate on a fixed small subset; reserve full validation for final-only checkpoints.
- Make full-validation opt-in when iterating locally; it can dominate wall-clock and mask training issues.

## Plot Robustness
- Don’t treat missing metrics as zeros; represent “not logged” explicitly.
- Align each metric with its natural cadence (train-step vs val-step) to avoid misleading plots.
- If a metric is sparse, plot it sparsely (connect points) rather than fabricating dense values.
- Make plotting resilient to partial runs (missing finals) so early debugging doesn’t break.
- For near-zero diagnostics, log/plot with enough precision (e.g. scientific notation + log scale).
- Diagnostics should run in consistent precision; mixed-precision drift can look like “instability”.
- Prefer line styles (not point clouds) for multi-component time series, and always include a legend for the encoding.
- When two series are intentionally identical (tied components), deduplicate the plot so style overlays don’t look like mismatches.
- When adding new logged keys, update the parser and add a small unit test so plots don’t silently degrade.

## RevDEQ
- FP64 add/sub is a *reversibility* tool (reconstruction accuracy), not a default training requirement.
- Use autograd-unroll when you need output-space regularizers; use RevDEQ backward when you need constant-memory exact gradients.
- Only compute/log reconstruction error when using the RevDEQ backward path (otherwise it’s not an actionable signal).
- Prefer simple, explicit parameterizations over hidden stability clamps; diagnose fixed-point behavior directly via residual/convergence metrics.
- When you need contraction, add a single block-level gate and log it; don’t hide stability in many per-path scale knobs.

## Refinement
- When mixing token distributions, normalize each input distribution first and renormalize after mixing.

## Configuration
- Keep experiment hyperparameters in code defaults (or CLI), not hidden environment variables.
- Remove dead/unreachable configuration paths; they silently rot and confuse debugging.
- If you add gating that changes a probability simplex into sub-mass, define health metrics on the renormalized share and treat leftover mass explicitly.

## Environment Sanity Checks
- Before long runs, verify the environment can see CUDA and the dataset/tokenizer paths resolve.
- Generate plots using an environment that has the plotting dependencies installed.
