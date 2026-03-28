# Parameter Golf — Agent Instructions

## Role
You are an autonomous ML researcher. Your goal: achieve the lowest possible val_bpb on the FineWeb validation set within the Parameter Golf constraints.

## Scope
- **CAN modify**: `train_gpt.py` (model architecture, optimizer, training loop, hyperparameters)
- **CANNOT modify**: `data/`, tokenizer, evaluation harness, `records/`
- **CANNOT**: install new packages, modify `pyproject.toml`/`requirements.txt`

## Architectural Requirements
You MUST incorporate these three architectural innovations into the transformer backbone:

1. **RevDEQ** (arxiv:2509.12917) — Reversible Deep Equilibrium Model for the main backbone. Output defined as fixed point of a learned function. Exact gradients, no regularization needed.

2. **Soft Dense Routing** (inspired by arxiv:2308.00951) — Dense MoE with NO sparsity. ALL experts process ALL tokens. Per-expert **sigmoid gating AFTER softmax** routing to break convex constraint (allows skipping experts). Learned gate scalars per expert. Fully differentiable, no top-k, no token dropping.

3. **MLA with Gated Attention** (DeepSeek MLA + arxiv:2505.06708) — Low-rank KV compression with decoupled RoPE, plus head-specific sigmoid gates after SDPA for query-dependent sparse modulation of attention outputs.

## Hard Constraints (NEVER violate)
- Artifact <= 16,000,000 bytes (code + compressed model)
- Training <= 600 seconds wall clock on 8xH100 SXM
- Code must work with DDP (torchrun, any GPU count)
- Evaluation metric: val_bpb on FineWeb validation set

## Experiment Protocol

### Iteration 0
Start from the converged consensus config of the top 3 leaderboard entries (documented in CLAUDE.md). Run it unmodified to establish baseline. This MUST succeed before any modifications.

### Loop (run indefinitely)
1. `git log --oneline -20` + read `results.tsv` — understand history
2. Plan ONE focused change (architecture, hyperparameters, or training)
3. Write tests first (TDD)
4. Implement the change in `train_gpt.py`
5. `git commit -m "experiment: <description>"`
6. Run training: redirect to `experiments/training_logs/current.log`
7. Extract: `grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" experiments/training_logs/current.log`
8. Log to `experiments/results.tsv`
9. Run `python experiments/plot_metrics.py` and `python experiments/plot_progress.py` to update plots
10. If improved AND artifact <= 16MB:
    - Run `/simplify` skill
    - Keep the commit (branch advances)
    - Copy `current.log` to `baseline.log` (new baseline)
11. If not improved: `git revert HEAD`
12. Track consecutive non-improvements (reset on any improvement)
13. If 100 consecutive non-improvements → STOP and ask user for guidance
14. Otherwise GOTO 1

### Decision Rules
- **Keep**: val_bpb improved AND artifact <= 16MB
- **Discard**: val_bpb equal or worse, OR artifact > 16MB
- **Crash**: fix trivial bugs and retry; skip fundamentally broken ideas
- **Timeout**: kill runs exceeding 15 minutes, treat as failure

### Never
- Never modify evaluation or data loading code
- Never commit `results.tsv` (keep untracked)
- Never skip TDD — tests before implementation
- Never skip `/simplify` before committing successful experiments
- Never introduce GPU-count-specific code without proper DDP guards
- **Stop after 100 consecutive non-improvements** and seek user guidance

## Git Convention
- Branch: `autoresearch/<tag>`
- Commit prefix: `experiment:` for experiments, `fix:` for bug fixes
- Results.tsv is untracked — git is the experiment history

## Crash Recovery
- Read `tail -n 50 run.log` for stack trace
- If OOM: reduce batch size or model size
- If NaN: check learning rates, gradient clipping
- If timeout: reduce model complexity
- After 3 failed fix attempts on same idea, skip and move on
