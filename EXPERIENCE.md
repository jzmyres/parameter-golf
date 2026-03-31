# EXPERIENCE — Lessons Learned

Short, reusable guardrails to avoid common experiment mistakes.

## Metrics
- Track and compare the *scored* metric (post-quant) separately from any in-training validation.
- Always report both the final in-training validation and the final post-quant result.

## Logging
- Treat the primary run log as the source of truth; don’t rely on wrapper tools capturing stdout/stderr.
- For progress monitoring, tail the run log directly.

## Validation Curves
- If you want plots with curves, make sure the run actually logs multiple validation points (set a non-zero validation interval).

## Plot Robustness
- Don’t treat missing metrics as zeros; represent “not logged” explicitly.
- Align each metric with its natural cadence (train-step vs val-step) to avoid misleading plots.

## Environment Sanity Checks
- Before long runs, verify the environment can see CUDA and the dataset/tokenizer paths resolve.
- Generate plots using an environment that has the plotting dependencies installed.
