# EXPERIENCE — Lessons Learned

Short, reusable guardrails to avoid common experiment mistakes.

## Metrics
- Track and compare the *scored* metric (post-quant) separately from any in-training validation.
- Always report both the final in-training validation and the final post-quant result.
- Name metrics by what they actually measure; don’t reuse a proxy metric under a different label.
- Regularize in the same space you diagnose (e.g. output-space diversity vs weight-space proxies).
- Treat expert health metrics as non-negotiable guardrails; optimize everything else inside that envelope.

## Logging
- Treat the primary run log as the source of truth; don’t rely on wrapper tools capturing stdout/stderr.
- For progress monitoring, tail the run log directly.
- Avoid logging stale diagnostics outside their natural cadence (train-step vs val-step).

## Validation Curves
- If you want plots with curves, make sure the run actually logs multiple validation points (set a non-zero validation interval).

## Plot Robustness
- Don’t treat missing metrics as zeros; represent “not logged” explicitly.
- Align each metric with its natural cadence (train-step vs val-step) to avoid misleading plots.
- If a metric is sparse, plot it sparsely (connect points) rather than fabricating dense values.
- Make plotting resilient to partial runs (missing finals) so early debugging doesn’t break.
- For near-zero diagnostics, log/plot with enough precision (e.g. scientific notation + log scale).
- Diagnostics should run in consistent precision; mixed-precision drift can look like “instability”.

## RevDEQ
- FP64 add/sub is a *reversibility* tool (reconstruction accuracy), not a default training requirement.
- Use autograd-unroll when you need output-space regularizers; use RevDEQ backward when you need constant-memory exact gradients.
- Only compute/log reconstruction error when using the RevDEQ backward path (otherwise it’s not an actionable signal).

## Configuration
- Keep experiment hyperparameters in code defaults (or CLI), not hidden environment variables.
- Remove dead/unreachable configuration paths; they silently rot and confuse debugging.

## Environment Sanity Checks
- Before long runs, verify the environment can see CUDA and the dataset/tokenizer paths resolve.
- Generate plots using an environment that has the plotting dependencies installed.
