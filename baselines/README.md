# Baseline Replication Workspace

This directory tracks comparable recurrent-depth baselines without committing third-party source trees or heavy run artifacts.

## Layout

| Path | Purpose |
|---|---|
| `manifest.yaml` | Committed baseline registry: source URLs, pinned commits, framework, DDP mode, status, and notes. |
| `fetch_baselines.py` | Fetches pinned GitHub repositories into ignored worktrees. |
| `smoke_baselines.py` | Runs dependency-free source checks against fetched worktrees. |
| `ddp_smoke_baselines.py` | Launches upstream-code DDP smoke attempts or records explicit blocked states. |
| `train_gpt_comparable_sweep.py` | Runs native `train_gpt.py` DDP 10-step variants for strict local comparability. |
| `ddp_harnesses/` | Tiny adapter scripts used for library-style baseline DDP import/train smoke. |
| `search_ledger.md` | Search date, queries, inclusion rules, and screened false positives. |
| `REPLICATION_REPORT.md` | Human-readable replication progress report. |
| `worktrees/` | Ignored shallow clones of upstream code. |
| `runs/` | Ignored generated logs and JSON smoke reports. |

## Status Values

| Status | Meaning |
|---|---|
| `smoke_passed` | Pinned source fetched, exact HEAD verified, README present, Python syntax compilation passed. |
| `ddp_smoke_passed` | A 2-process `torchrun` smoke completed under the configured DDP mode. |
| `adapted_eval_run` | A local repo run/log has enough data to compare against the active task. |
| `blocked_no_code` | No public cloneable training-code target was found. |
| `blocked_no_training_code` | Source exists but does not expose a maintained training entrypoint for this target. |
| `blocked_non_pytorch_ddp` | The project is not PyTorch DDP, e.g. TensorFlow/T2T or JAX/Equinox. |
| `blocked_dependency` | Install/import failed in the configured environment. |
| `blocked_compute` | Code exists, but paper-faithful replication exceeds local compute. |
| `blocked_data` | Required data is unavailable or incompatible with local FineWeb. |
| `blocked_hardware` | The required GPU count is unavailable on the host (e.g. a CPU-only login node). |
| `failed_step_requirement` | The command exited 0 but did not reach the requested step count. |
| `failed` / `failed_timeout` | A command failed (or was killed on timeout) for a reason that should be debugged. |

Transient report-only values: `not_run` (initial), `dry_run` (`--dry-run` preview, command not executed).

## Environments

Two conda environments are used and both live outside the repo (`baselines/.envs/` is gitignored):

| Env | Purpose |
|---|---|
| `opg` | The project env (PyTorch + `train_gpt.py` deps). Used by the native comparable sweep and by any adapter whose `ddp_env` is `opg`. |
| `opgbaselines` | Lightweight orchestration env for fetching, source smoke checks, and import-adapter DDP smokes (default `ddp_env` for adapters). |

Create the orchestration env once (PyTorch + each baseline's import deps as needed):

```bash
conda create -y -n opgbaselines python=3.11 && conda run -n opgbaselines pip install torch
```

## Commands

List fetchable external baselines:

```bash
python baselines/fetch_baselines.py --list
```

Fetch all pinned GitHub baselines:

```bash
python baselines/fetch_baselines.py
```

Run dependency-free source smoke checks:

```bash
conda run --no-capture-output -n opgbaselines python baselines/smoke_baselines.py
```

Run upstream-code DDP smoke attempts for all baselines:

```bash
conda run --no-capture-output -n opgbaselines python baselines/ddp_smoke_baselines.py --nproc 2 --steps 10 --timeout-s 900
```

The DDP runner uses each manifest entry's `ddp_mode`. PyTorch libraries use an import-plus-DDP adapter; non-PyTorch and paper-only entries are reported as blocked instead of skipped. These adapter results are launchability checks, not direct quality comparisons with `train_gpt.py`.

Run strict `train_gpt.py`-native comparable 10-step sweeps (uses the `opg` env, since it launches this repo's `train_gpt.py`):

```bash
conda run --no-capture-output -n opg python baselines/train_gpt_comparable_sweep.py --steps 10 --nproc 2 --timeout-s 2400
```

The native comparable sweep only runs model variants implemented in this repository's `train_gpt.py` shell. External SOTA papers without a `train_gpt.py` implementation are documented in `REPLICATION_REPORT.md` as blocked or proxy-only rather than passed.
