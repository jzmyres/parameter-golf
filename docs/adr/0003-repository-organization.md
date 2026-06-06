# ADR 0003 — Repository organization: single `legacy/` archive, active-only tree, `results/` evidence

## Status

Accepted (2026-06-06). Supersedes the repository-layout aspects of the
fresh-start plan that proposed a `src/` package reorg (that plan's 1500-line
hard-stop motivation was retired by the "no line cap on the research
`train_gpt.py`" directive, 2026-06-04).

## Issue

The M0 rewrite ([0002](0002-final-minimal-p1-design.md)) completed at the
**model** layer (`train_gpt.py` = M0; the rich RevDEQ/MoE/MLA/MoS model moved to
`legacy/train_gpt_rich.py`) but left the workspace mixed:

- `experiments/components/` (20 files) and ~21 `experiments/test_*.py` existed
  only for the rich model; the active `train_gpt.py` imports none of them.
- Tests lived in **two** homes (`tests/` and `experiments/`).
- Training-run evidence had **no durable tracked home**: background-task stdout
  lands in session-scoped `/tmp` and all `*.log`/`training_logs/` paths are
  gitignored, so cited numbers resolved only to ephemeral artifacts.
- Stale build/report artifacts (`docs/superpowers/`, `reports/advisor_progress_*`,
  `RUN_RESULTS.md`, `results.tsv`, `train_gpt_mlx.py`) sat in the active tree.
- **Discovered during the reorg:** the test/contract/audit layer was *not*
  migrated to M0 — 17 of 26 `tests/` files (incl. 2 of 4 mandated audit-registry
  tests) still validate `legacy.train_gpt_rich`. M0 has ~5 unit tests and no
  contract/audit coverage of its own.

## Decision

A single invariant governs layout: **everything not actively used lives under
one `legacy/` folder; everything outside `legacy/` is actively used.**

Executed as a *targeted in-place cleanup* (no `src/` package move — keeps
`train_gpt.py` at root and all `torchrun`/test/script paths stable). Because the
test layer is entangled (shared `tests/_helpers.py`, the self-referential audit
registry), it is staged:

- **Phase A (this change):** archive everything cleanly separable —
  `experiments/components/` → `legacy/components/`; the 21 rich `experiments/`
  tests → `legacy/tests/`; consolidate the active `experiments/` tests into
  `tests/` (making `experiments/` test-free); archive `train_gpt_mlx.py`,
  `results.tsv`, `RUN_RESULTS.md`, `profile_train.py`, `hypotheses_archive.md`,
  `iter103_chained_routing_plan.md`, `docs/superpowers/`, and
  `reports/advisor_progress_2026-05-30/` into `legacy/`. Add `results/` and a
  root `conftest.py`; pin `pytest.ini` to a single test home.
- **Phase B (follow-up):** port the rich contract/audit/unit tests in `tests/`
  to target M0, then archive the rich versions and re-point the audit registry.
  Tracked in `experiments/docs/hypotheses.md`.

`records/` (official submissions) and `baselines/` (external replication) are
*not* `legacy/` — they are distinct read-only archives, kept and documented in
`CLAUDE.md`.

## Interface (the layout contract the repo conforms to)

- **`legacy/`** — the sole archive for deprecated/non-active code, tests, docs,
  and reports. May import active modules (e.g. `experiments.plotting_hook`);
  active code MUST NOT import from `legacy/`. Not collected by default pytest.
- **Outside `legacy/` = actively used.** Root: `train_gpt.py` (M0) + project
  docs. `experiments/` = active harnesses (`p1_synthetic`, `p1_feedback_stages`,
  `measure_phi`), plotting (`plot_metrics`, `plot_eval_metrics`, `plot_progress`,
  `plotting_hook`), runners (`run_*.sh`, `update_results.sh`, `smoke_test_ddp.sh`),
  and research docs (`docs/`) — **no test files**.
- **`tests/`** — the single home for all *active* unit tests. (Holds the
  unported rich contract cluster until Phase B.)
- **`results/<date>_<tag>/`** — tracked frozen run evidence (`summary.md` +
  `run_metrics.txt`); the durable home a committed citation must resolve to.
- **`records/`, `baselines/`, `data/`** — read-only references (not `legacy/`).
- Imports resolve via root `conftest.py` (repo root on `sys.path`);
  `pytest.ini` sets `testpaths = tests`.

## Evidence

Move set, import-rewrite map, and green verification (active suite 470 passed;
audit gate 19 passed; archived `legacy/tests/` collects when run explicitly):
`experiments/docs/hypotheses.md` (2026-06-06 reorg entry). First `results/`
snapshot: `results/2026-06-06_m0_halflr_extrapolation/`.

## Consequences

- The active tree now reflects what is in use; `legacy/` is the one place to look
  for retired machinery. Browsing, grep, and onboarding are cleaner.
- **M0's thin contract/audit coverage is now explicit** (was masked by the rich
  suite). Until Phase B, the mandated audit registry still validates the rich
  model; `test_optional_component_flag_contract` scans `legacy/tests/` for its
  rich-flag witnesses. This is intentional debt, not silent.
- `legacy/` tests carry now-shallow `sys.path` shims (resolve to `legacy/`); the
  root `conftest.py` makes them importable when run explicitly. Their non-import
  path computations may need fixing if/when Phase B revisits them.
