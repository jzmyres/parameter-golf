# Parameter Golf - Agent Guide

This file is the **single binding directive** for working in this repo, and a thin pointer surface: it carries the abstract principles and current-state orientation an agent needs *before* any action. Detailed bodies, incident postmortems, and verification recipes live in [`EXPERIENCE.md`](EXPERIENCE.md) and [`docs/adr/`](docs/adr/) as **non-authoritative historical context** — rationale and detail, not rules.

## Authority & orientation

`CLAUDE.md` is the **sole binding directive** (how to work). [`reports/opg_doc.tex`](reports/opg_doc.tex) is the **project spec — the research design and goal the codebase aligns to**: always refer to it before any architecture change, and align the code to it (record an intentional deviation per the [doc-code invariant](EXPERIENCE.md#doc-code-invariant) when the implementation must differ; when unsure, clarify with the user). `train_gpt.py::Hyperparameters` is the config source of truth (this file mirrors defaults but does not own them). `docs/adr/` are dated decision records / the implementation **interface** the code conforms to (and which themselves align to `opg_doc.tex`). [`EXPERIENCE.md`](EXPERIENCE.md) (incident postmortems + lessons) is **historical context, not rules** — consult the relevant incident *by topic* when it bears on your task; you need not read it in full before acting. EXPERIENCE.md / ADR entries may be **outdated** where the code they describe has since changed or been deleted: verify against current code, and on any **context-vs-code conflict, STOP and clarify with the user, then prune/supersede the stale entry** rather than enforcing it. Also load the memory files at [`#pre-action-memory`](EXPERIENCE.md#pre-action-memory) and inspect `experiments/docs/` before any iter work. New memory files: add to the memory index and the appropriate doc pointer in the same commit.

## Project Invariants

- **Challenge constraints.** Artifact ≤ 16,000,000 bytes; submission training fits the 600 s 8xH100 budget. Details: [`EXPERIENCE.md#project-constraints`](EXPERIENCE.md#project-constraints).
- **No hard line cap on the research `train_gpt.py`.** The upstream ≤1500-line note in `train_gpt_mlx.py` targets a newcomer *reference*; the OPG research model carries the mandated resource/expressiveness mechanisms (reversible recurrence, MLA, MoE, MoS) and is not line-capped. Keep it as lean as the design allows with mechanisms in clearly-bounded modules. User directive 2026-06-04.
- **Scoped edits.** Default writable surface is `train_gpt.py`, focused tests, and project docs. Do not modify `data/`, tokenizer/eval harness code, `records/`, package manifests, or dependencies without explicit user approval.
- **No new packages by default.** If approved, install with `uv pip install <pkg>` and update requirements in the same commit.
- **DDP first.** Training code works under single GPU and `torchrun` DDP; GPU-count-specific paths use `world_size`.
- **Step-matched comparisons.** Default dev comparisons are by equal step count; wall-clock is primary only for submission or throughput-focused iterations. Details: [`EXPERIENCE.md#runbook`](EXPERIENCE.md#runbook).
- **Expert health final-only.** Per-component shares are normalized for readability; total routed mass is tracked separately.
- **bf16 training default.** Training-path autocast stays bf16; no silent promotion to fp32/fp64.
- **Profile before throughput claims.** Throughput work needs trace-backed diagnosis. SDPA replacements at T=2048 are presumed suspect until profiled.

## Standing Directives

- **Align to the project spec + doc consistency.** [`reports/opg_doc.tex`](reports/opg_doc.tex) is the **project spec — the research design/goal the codebase aligns to**: always refer to it, and align the code to it (or record an intentional deviation per [#doc-code-invariant](EXPERIENCE.md#doc-code-invariant)). `CLAUDE.md` is the sole binding *process* directive; `train_gpt.py::Hyperparameters` is the config source of truth; `docs/adr/*` (immutable, numbered) record dated decisions and the implementation **interface** the code must satisfy (supersede with a new ADR — never rewrite; ADRs themselves align to `opg_doc.tex`). `EXPERIENCE.md` is **non-authoritative historical context**, not rules. Before acting on EXPERIENCE.md or an ADR, verify it against current code; on any **context-vs-code conflict, STOP and clarify with the user, then prune/supersede the stale entry** — do not enforce stale context. Any commit touching `opg_doc.tex`, `CLAUDE.md`, an ADR, `EXPERIENCE.md`, or the implementation keeps these mutually consistent in that commit (opg_doc.tex spec ↔ code; ADR interface ↔ code; `CLAUDE.md` ↔ ADR pointers). User directive 2026-06-04.
- **Three-test design discipline (Principled + Simplest + General)** applies to ANY fix, regularizer, optimization, or prescription. Closure notes must name which test failed. Details: [`EXPERIENCE.md#principled-simplest-general`](EXPERIENCE.md#principled-simplest-general). User directive 2026-05-15.
- **Final minimal P1 is the research target.** The paper-facing target is now `M0`: tied, token-injected, additive-coupling reversible recurrence with standard MHA/GQA + SwiGLU, halting readout, a Tier-1 depth-hard positive-control protocol, and a P2 target of full reversible BPTT. The current Tier-1 harness tests P1 detectability and structural reconstruction; it does not claim activation-memory scaling until the memory-saving backward path is implemented and measured. Dense MoE, Dirichlet-UCB, MLA, MoS, quantization, and cache/deployment work are staged after P1. Fixed-point diagnostics (`rho_F`, residuals, `sigma_max_F` where available) remain advisory; legacy prefix-anchor consistency and nonzero `lyapunov_coef` are rejected at launch. Details: [`reports/opg_doc.tex`](reports/opg_doc.tex) and [`docs/adr/0002-final-minimal-p1-design.md`](docs/adr/0002-final-minimal-p1-design.md).
- **Tier-1 P1 harness.** Use `experiments/p1_synthetic.py` for the executable final-minimal path. `--variant control`, `--variant m0`, and `--variant mclk` are the complete active interface. The harness intentionally rejects MoE, cache, and quantization switches. Full feedback-stage executable coverage lives separately in `experiments/p1_feedback_stages.py` and `experiments/run_feedback_stage_pipeline.sh`; do not move those surfaces back into the P1 runner.
- **Iter progress reports** include current metric, prior-healthcheck delta, and same-step baseline. Details: [`EXPERIENCE.md#iter-progress-reporting`](EXPERIENCE.md#iter-progress-reporting).
- **GPU preflight before every launch.** Preflight `pgrep` + `nvidia-smi --query-compute-apps`; stop any active training process and verify GPU memory drops to ~0 before launching. Stuck dynamo-compiles can hold 30+ GB for hours and OOM the next launch. Details: [`EXPERIENCE.md#gpu-preflight-protocol`](EXPERIENCE.md#gpu-preflight-protocol).
- Keep `experiments/docs/hypotheses.md` synchronized in real time: read before every iter, update after every iter, include required metrics.
- Before iteration work, review `experiments/docs/` comprehensively.
- Prefer architecture exploration over blind hyperparameter tuning; throughput iterations take priority when the active queue marks them throughput-bearing.
- Disable `torch.compile` for dev unless the iteration explicitly tests compile behavior; still keep code compile/DDP-safe.
- Always use all visible GPUs via DDP for normal runs; single-GPU is debug-only.
- Use `conda run --no-capture-output` when wrapping conda commands that must stream logs.
- After failures, healthcheck cadence is 5 minutes; after 3 healthy checks, widen to 20 minutes.
- When closing an iteration, record the full active config — regularizer stack, flags, and schedules.
- NTP descent rate and per-wallclock equivalent are permanent H-claim metrics.
- Cumulative averages lie early; compute per-step deltas for throughput before drawing conclusions. Details: [`EXPERIENCE.md#cumulative-metric-misread`](EXPERIENCE.md#cumulative-metric-misread).
- Load-balance is not sparsity: CV measures balance, per-token entropy measures specialization, global entropy measures utilization.
- For fast paths, dispatch on `torch.is_grad_enabled()` plus real grad needs; `Parameter.requires_grad` alone is not a runtime-mode predicate.
- Decouple antagonistic regularizers; if two objectives fight, make one a metric or run an explicit ablation.
- Anneal schedule-sensitive regularizers from zero unless a controlled test justifies a full-strength cold start.
- Treat OPG as finite-horizon recurrence for task utility. Fixed-point framing is a fallback diagnostic, not default training pressure.
- For P1, measure paired depth gain on depth-hard data with a passing positive control before interpreting internal depth metrics. Legacy `ED_update`, `ED_logit`, gate trajectories, `route_depth_nmi_mean`, `route_depth_nmi_max`, and `expert_util_mean` are diagnostics, not P1 gates.
- Constant training activations do not imply constant autoregressive KV cache. The first constant-KV candidate is terminal hidden-state cache, measured against exact multi-depth cache.
- For the final minimal P1 path, prefer standard MHA/GQA and SwiGLU over MLA/MoS/MoE feature stacks. Those richer paths are later-stage or legacy axes, not the core science path.

## Environment & Files

- Conda env: `conda activate opg`.
- Dependencies: `requirements.txt`; authorized installs use `uv`, not `pip`.
- Data is read-only: `./data/datasets/fineweb10B_sp1024/`; tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`.
- Model source: `train_gpt.py`; focused tests under `experiments/test_*.py` or `tests/`.
- Research docs: `EXPERIENCE.md` for details/rationale, `experiments/docs/hypotheses.md` for claims/results, `reports/opg_doc.tex` for paper-facing text.
- Runtime outputs untracked unless promoted: `run.log`, `results.tsv`, `experiments/training_logs/*`, `experiments/weights/*`, `experiments/checkpoints/*`.
- Optional debug plotting via `auto_plot_on_val=True` (default): refreshes `experiments/*.png` per validation via `experiments/plotting_hook.py` (rank-0 only; silent no-op if matplotlib or `experiments/plot_*.py` modules are absent).
- Historical submissions in `records/` are read-only.
- External baseline references live in the `baselines/` workspace (pinned manifest + smoke scripts) and are catalogued in `experiments/docs/recurrent_depth_baselines.md` (RevDEQ, RevFFN, MoEUT, ReMoE, Universal Transformer, Parcae, Iso-Depth/φ, etc.). The former local `/home/mzhong4/work/research/{rdeq,tsu}/...` reference paths are gone; do not cite them.
- `baselines/` is the external recurrent-depth replication workspace (pinned manifest + fetch/smoke scripts + docs). Heavy/regenerable subtrees (`worktrees/`, `runs/`, `.envs/`) are gitignored; commit only manifest+scripts+docs and verify with `git add -An`. See [`EXPERIENCE.md#vendored-workspace-hygiene`](EXPERIENCE.md#vendored-workspace-hygiene). `CONTEXT.md` holds cross-iteration baseline vocabulary; per-baseline inventory is `experiments/docs/recurrent_depth_baselines.md`.

## Current Implementation Versus Final P1 Target

Orientation mirror only. Check `train_gpt.py::Hyperparameters` before launch. The current implementation is richer than the final minimal P1 target in `reports/opg_doc.tex`; do not infer that Dirichlet-UCB, MLA, MoS, or dense MoE are required for P1.

- **Core:** 12-layer RevDEQ-style shared block, `model_dim=768`, 8 query heads, 4 KV heads, sequence length 2048, vocab 1024, tied embeddings, train batch tokens 524,288.
- **Experts:** default expert layout is `experts_per_slot=16`, `expert_slots=1`, `expert_slot_order=parallel` (`16x1`). Use `--experts-per-slot=4 --expert-slots=4 --expert-slot-order=parallel` for the matched `4x4` expressiveness/throughput ablation. Layout is not Lipschitz control. No shared/always-on expert by default; `--num-shared-experts=1` opts in per slot.
- **Solver:** Reversible finite-horizon training path with the current Parcae-style per-dim damping/injection scaffold enabled; `parcae_reversibility_floor=0.1`; `deq_bptt_k=3`; weighted finite-horizon K sampling `{32:0.20, 64:0.40, 128:0.40}` normalized at launch; scalar beta path is fallback only. Parcae is not the research goal.
- **Profiles:** `config_profile=fast_default` preserves current defaults. `score_iter152` is retained for wrapper compatibility but no longer enables rejected prefix-anchor consistency. CLI flags override profile values.
- **Finite-horizon OPG default-on:** `deq_prefix_anchors=False`, `multi_k_consistency_anchor_coef=0`, `finite_horizon_pairs=((16,64),(16,128),(32,128))`, and `finite_horizon_scale_coef>0`. The loss is final-depth task loss plus a shallow stop-gradient no-degradation hinge; router/diversity regularizers remain final-endpoint health terms.
- **KV cache policy:** no first-class autoregressive cache API is active yet. Future constant-KV work starts with terminal hidden-state cache (`z_K` or K/V derived from `z_K`) and must measure quality gap against exact multi-depth cache before claiming constant-KV inference.
- **Default-off paths:** `num_refinements=0`, `use_ctp=False`, `use_nsa_attention=False`, `lyapunov_coef=0`, `denoising_coef=0`, `mos_output_diversity_coef=0`, `logit_softcap=0`, `bigram_vocab_size=0`.
  The launch validator rejects nonzero `lyapunov_coef`: FP convergence is tracked through diagnostics, not enforced with the refuted Lyapunov pressure path.
- **Optimizer:** Muon/AdamW groups with PE-NS enabled; weight decay 0.015; grad clip 1.0; warmdown fraction 0.72. See code for exact LR groups.
- **Routing/loss stack:** Dirichlet-UCB routing; EMA-anchored balance + reverse-KL (`use_reverse_kl_balance=True`); EMA-gated alive hinge; worst-pair cosine expert-output diversity; per-token entropy; 0.07 reg warmup. Sigmoid gate off; CV is diagnostic-only. Revert flags + history: [`EXPERIENCE.md#routing-health-metrics`](EXPERIENCE.md#routing-health-metrics).
- **Quantization/eval:** int6 per-row quantization + zstd-22 compression via `encode_scored_artifact`; sliding-window eval enabled; fast eval uses `eval_batch_seqs=256` with `val_micro_batch_seqs=48`; `eval_profile=diagnostic` keeps the full K-sweep, `submission`/`debug` reduce K/lip probe scope. Promotable final metadata requires full validation; `diagnostic_gate_policy=advisory` keeps BPB-primary promotion while recording `health_valid`.

## Run Commands

Default dev run: 1000 iters, no wallclock cap, all visible GPUs.

```bash
conda activate opg
torchrun --standalone --nproc_per_node=gpu train_gpt.py
```

Use `--iterations=N` for shorter runs. If the `gpu` alias is unsupported, use an explicit count such as `--nproc_per_node=2`.

For agent-launched training runs, keep the process attached or actively monitor
the run log and check/report progress only every 10 minutes, unless the run
completes or fails sooner. This is a progress polling/reporting cadence, not a
`--max-training-seconds=600` wallclock cap.

Tier-1 P1 synthetic DDP pipeline shape:

```bash
CUDA_VISIBLE_DEVICES=6,7 ITERATIONS=300 RUN_TAG=gpu67 bash experiments/run_p1_synthetic_pipeline.sh
# Direct single-variant equivalent, useful for focused debugging:
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 experiments/p1_synthetic.py --variant m0 --iterations 300
```

This script intentionally runs only the P1 detectability triage. The runner
exposes task/budget knobs (`TASK`, `SEQ_LEN`, `ITERATIONS`, `TRAIN_DEPTHS`,
`EVAL_DEPTHS`, `PAIRS`, `MODEL_DIM`, `NPROC`) so a failed positive control can
be localized without adding model machinery. Treat a mean-only depth gain as
insufficient: the harness must emit paired confidence intervals, and promotion
needs the lower bound positive plus controlled `NDR_epsilon`.

Full feedback-stage diagnostic pipeline shape:

```bash
CUDA_VISIBLE_DEVICES=6,7 ITERATIONS=1000 RUN_TAG=gpu67_feedback_all_i1000 bash experiments/run_feedback_stage_pipeline.sh
```

This script runs the minimal S+0 triage first, then runs separate S+1 MoE/static-MoE diagnostics,
S+2 hidden-state cache proxy diagnostics, and S+3 fake-int6 sensitivity diagnostics. Treat S+1/S+2/S+3
as mechanism/deployment evidence only; they do not promote P1 while the positive-control gate fails.

Dated S+0 / feedback-stage run history (the `gpu67_*` verification results) lives in [`experiments/docs/hypotheses.md`](experiments/docs/hypotheses.md#current-feedback-stage-verification-2026-06-04), not here — CLAUDE.md stays policy-thin. Standing conclusion: P1 is not yet promoted (positive-control depth gain CI crosses zero); the next step is detectability repair of the Tier-1 task/positive-control capacity, not adding MoE, fixed-point pressure, cache, or quantization machinery.

Submission-style run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --max-training-seconds=600
```

Single-GPU `python train_gpt.py` is debug-only. Quick metric extraction:

```bash
grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log
```

Full environment, file, logging, and plotting details: [`EXPERIENCE.md#environment-and-files`](EXPERIENCE.md#environment-and-files), [`EXPERIENCE.md#experiment-loop-details`](EXPERIENCE.md#experiment-loop-details).

## Architecture Principles

- **Reversible finite recurrence.** The active model output is the finite-depth state selected by the sampled/evaluated recurrent budget. Fixed-point convergence is tracked only as terminal-cache/fallback readiness. Details: [`EXPERIENCE.md#revdeq-architecture-details`](EXPERIENCE.md#revdeq-architecture-details).
- **Reversibility is a correctness constraint.** Reconstruction uses fp64 add/sub; grad-needing tensors must cross `RevDEQFunction.apply(...)`. Details: [`EXPERIENCE.md#custom-autograd-input`](EXPERIENCE.md#custom-autograd-input).
- **DEQ input conditioning.** Only via learnable per-dim Parcae ZOH coefficient `B_bar = Delta * B`; unconditional shortcuts forbidden. Form: `T_theta(z, x0) = B_bar * RMSNorm_learn(x0) + Delta(z, x0)`.
- **Parcae floor is not a tuning knob.** Reverse-reconstruction divisor lower bounds need code + tests + smoke tolerance + paper updates together. Details: [`EXPERIENCE.md#revdeq-reversibility-floor`](EXPERIENCE.md#revdeq-reversibility-floor).
- **Backward-only floors are forbidden** for coefficients (especially `B_bar`) that do not divide reverse reconstruction; floors there bias the ZOH form.
- **Interpret recurrence diagnostics by definition.** Under TBPTT, `deq_recon_err` is forward distance, not reconstruction error. `rho_F`, `sigma_max_F`, and `iter_conv_rel` are advisory terminal-cache/fallback-readiness signals. Details: [`EXPERIENCE.md#deq-recon-err-interpretation`](EXPERIENCE.md#deq-recon-err-interpretation).
- **Do not misattribute Parcae contraction fixes.** Scalar `deq_beta` is fallback-only; the active path uses per-dim `beta = 1 - A_bar`. Contraction diagnostics do not justify global FP pressure under the active finite-horizon profile.
- **Promoted routing has explicit tech debt.** Iter145r left min-share, output-orthogonality, and operator-norm debt. The current rescue stack addresses min-share (EMA-gated alive hinge) and orthogonality (worst-pair output diversity); operator-norm "debt" is reframed as a measurement artifact.
- **Soft dense routing only.** All experts process all tokens; top-K, capacity, argmax, and threshold skips break RevDEQ reversibility unless mathematically smooth or outside the RevDEQ path. Details: [`EXPERIENCE.md#no-top-k-dispatch`](EXPERIENCE.md#no-top-k-dispatch).
- **Expert independence is hard.** Every trainable parameter inside an expert path belongs to exactly one expert; shared learned scales silently couple experts. Details: [`EXPERIENCE.md#prenorm-scale-independence`](EXPERIENCE.md#prenorm-scale-independence).
- **Router placement is semantic.** Routers route on component input. Attn/MLP routing is `simplex(allocation) * sigmoid(gate)` when the routed sigmoid gate is enabled; iter146 default disables that gate so routed mass is always used. MoS routing is pure softmax.
- **No batch-coupled expert assignment.** Expert weights come from each token + learned router parameters; no batch-level matching/occupancy/Sinkhorn. Global usage pressure goes through losses or slow bias feedback.
- **EMA anchors are detached.** EMAs are valid for diagnostics, alive-loss weighting, slow bias, and straight-through EMA-anchored balance; a plain detached `KL(EMA || uniform)` does not backprop into the current token router.
- **Full-D LoRA-style experts are the standard.** Expert rank constrains parameter count, not activation width. Bottleneck bodies closed as a scaling direction. Details: [`EXPERIENCE.md#bottleneck-experts-closed`](EXPERIENCE.md#bottleneck-experts-closed).
- **Per-expert MLA is the attention standard.** Each expert owns Q, KV compression/decompression, K-rope, Wo, gated-attention parameters, and learned norms.
- **FSQ and CTP are retained as ablation paths.** Baseline promotion runs keep them disabled unless explicitly testing.
- **Routing metrics have separate axes.** Global entropy = utilization; per-token entropy = specialization; CV = balance; router mass ≠ normalized shares. Details: [`EXPERIENCE.md#routing-health-metrics`](EXPERIENCE.md#routing-health-metrics).
- **Routing regularizers operate on routed mass** (combined gate × allocation when the gate is on; pure softmax allocation otherwise). MoS softmax is the exception. Details: [`EXPERIENCE.md#routing-reg-input-invariant`](EXPERIENCE.md#routing-reg-input-invariant).
- **Regularizers should be decomposed.** Balance ≠ specialization ≠ orthogonality; bundled penalties are allowed only as explicit ablations. Details: [`EXPERIENCE.md#routing-health-metrics`](EXPERIENCE.md#routing-health-metrics).
- **Disabled techniques are documented outside CLAUDE.md.** Details: [`EXPERIENCE.md#disabled-techniques`](EXPERIENCE.md#disabled-techniques).

## Research Protocol

- **Setup discipline.** Read branch state, recent commits, `results.tsv`, `run.log`, and relevant hypothesis entries before changing code.
- **One focused change per iteration.** A clean hypothesis needs a single controlled variable unless explicitly testing a bundle.
- **TDD for architecture.** Add or update focused tests first when feasible — shape, gradients through FP, quantization roundtrip, DDP, artifact size.
- **Smoke before long train.** Smoke must show loss decreasing, finite grads, no NaN/Inf, healthy routing. Reconstruction-near-precision only expected in full-BPTT smoke.
- **Hypothesis log is mandatory.** Every iteration entry records claim, evidence, status, confounds, int6 result, K-sweep table, trajectory, acyclicity-prime check, NTP descent-rate windows. Details: [`EXPERIENCE.md#hypothesis-log-detail`](EXPERIENCE.md#hypothesis-log-detail).
- **Update plots/log rotation after runs.** `experiments/update_results.sh` is part of the iteration loop.
- **Architecture beats knob churn.**
- **Simplicity criterion.** All else equal, simpler wins; code removal with equal results is a keep.
- **DRY and orthogonal functions.** One helper per shared setup/format/formula. Details: [`EXPERIENCE.md#dry-function-orthogonality`](EXPERIENCE.md#dry-function-orthogonality).
- **Diagnostic metric contract.** Required metrics need compute, freshness tagging, log emission, parser support, and focused tests. Details: [`EXPERIENCE.md#diagnostics`](EXPERIENCE.md#diagnostics).
- **Pre-commit review chain.** `/simplify`, `coderabbit:review`, `pr-review-toolkit:review-pr`, `superpowers:requesting-code-review`, run audit checklist, apply simple first-principled fixes.
- **New audit rules.** Add to `EXPERIENCE.md` Section 1 with an index row; cite concisely from this file.

## Audit Checklist

Run before commits that touch `train_gpt.py`, model tests, `CLAUDE.md`, `EXPERIENCE.md`, or `experiments/docs/hypotheses.md`. Each row is a 1-line pointer; the rule body, root cause, and verification recipe live at the cited EXPERIENCE.md anchor (consult the relevant anchor when a row bears on your change — EXPERIENCE.md is historical context, so verify the cited incident still matches current code).

> **Meta-principle.** Read metric definitions before values; compute from raw when in doubt; user pushback is a signal to re-derive, not defend. See [`EXPERIENCE.md#reading-derived-metrics`](EXPERIENCE.md#reading-derived-metrics).

- **Single source of truth.** Tunable knobs live in `Hyperparameters`; consumers read `args.<field>`. See [#config-drift](EXPERIENCE.md#config-drift), [#hyperparameter-fanout](EXPERIENCE.md#hyperparameter-fanout).
- **Four-touch new knobs.** `Hyperparameters` field + CLI parser (if user-facing) + consumer reads from `args` + docs/paper mirror.
- **Permutation consistency.** Related tensor layout transforms use consistent index tuples; manual flatten/reshape needs an einsum-equivalence test. See [#permutation-consistency](EXPERIENCE.md#permutation-consistency).
- **Optimizer coverage.** Every trainable parameter lands in exactly one optimizer group.
- **Dead-code removal.** Removing a feature removes code paths, docs, tests, diagnostics, and constant tracking in the same change. See [#dead-code-tracking](EXPERIENCE.md#dead-code-tracking).
- **Module aliases** — see [#router-alias](EXPERIENCE.md#router-alias).
- **No hot-path syncs.** Training hot path doesn't branch on CUDA scalars or use `.item()` / `.cpu()` outside log sites. See [#hot-path-sync](EXPERIENCE.md#hot-path-sync).
- **Explicit boundaries.** Wrapper, device, and optional-telemetry boundaries need explicit unwraps, tensor-safe control flow, graceful shell fallbacks, contract tests. See [#explicit-boundary](EXPERIENCE.md#explicit-boundary).
- **Custom autograd inputs** — see [#custom-autograd-input](EXPERIENCE.md#custom-autograd-input).
- **Predicate migrations** — see [#routing-predicate-migration](EXPERIENCE.md#routing-predicate-migration).
- **Identifier uniqueness** — see [#identifier-uniqueness](EXPERIENCE.md#identifier-uniqueness).
- **Learned norm-scale independence.** Each learned scale conditions exactly one linear weight. See [#prenorm-scale-independence](EXPERIENCE.md#prenorm-scale-independence).
- **Doc-code invariant.** If `reports/opg_doc.tex` describes an algorithm and the implementation intentionally differs, document the deviation. See [#doc-code-invariant](EXPERIENCE.md#doc-code-invariant).
- **Diagnostic gates follow flags** — see [#diagnostic-gate-component-awareness](EXPERIENCE.md#diagnostic-gate-component-awareness).
- **Keep CLAUDE concise.** Principles + short rationale only; long examples and iter history go to `EXPERIENCE.md` or `hypotheses.md`. See [#claude-md-size-budget](EXPERIENCE.md#claude-md-size-budget).
- **Cumulative metrics are not instantaneous.** Before s50, compute per-step deltas from raw `train_time`. See [#cumulative-metric-misread](EXPERIENCE.md#cumulative-metric-misread).
- **Routing-reg input invariant.** Router regs operate on combined routed mass, not renormalized shares. See [#routing-reg-input-invariant](EXPERIENCE.md#routing-reg-input-invariant).
- **No discrete dispatch in RevDEQ.** Top-K, capacity, argmax, hard-threshold dispatch require explicit proof or must remain off. See [#no-top-k-dispatch](EXPERIENCE.md#no-top-k-dispatch).
- **Grad-enabled checks.** Fast dispatch uses `torch.is_grad_enabled()` and actual grad requirements; `Parameter.requires_grad` alone is not a runtime-mode predicate. See [#grad-enabled-vs-requires-grad](EXPERIENCE.md#grad-enabled-vs-requires-grad).
- **Partial previews are complete.** Short runs still report every mandated component they can compute, with Caveats for missing artifacts. See [#partial-preview-completeness](EXPERIENCE.md#partial-preview-completeness).
- **Move means tracked.** Relocating files: destination MUST be `git add`-ed in the same commit as the source deletion. See [#move-tracked-invariant](EXPERIENCE.md#move-tracked-invariant).
- **Enforcement-config staging.** Any new project-level config that *enforces* an invariant (`pytest.ini`, `pyproject.toml` lint sections, `.pre-commit-config.yaml`, ruff/mypy rule files, coverage thresholds) MUST be staged in the same commit as the code it enforces. Enforced by `tests/test_enforcement_config_staged.py`. See [#enforcement-config-staging](EXPERIENCE.md#enforcement-config-staging).
- **Audit-row executability.** Every new audit row ships with at least one executable witness (unit test, pre-commit hook, or CI gate) in the same commit. Shell snippets are not enforcement. See [#audit-row-executability](EXPERIENCE.md#audit-row-executability).
- **Audit-test execution.** Every commit touching `_OPTIONAL_COMPONENT_FLAGS`, `_OPTIONAL_COMPONENT_CAPABILITIES`, any `use_X` Hyperparameter, an enforcement-config file, or a symbol listed in `tests/test_removal_symmetry.py` MUST run `bash experiments/run_audit_tests.sh` and confirm green BEFORE staging. The canonical registry of audit-test files is the line below, kept inside HTML-comment markers so the parser at `tests/test_audit_test_execution.py` scopes precisely (do not edit the markers or the line between them by hand without also updating `experiments/run_audit_tests.sh::AUDIT_TESTS`): <!-- audit-tests-registry-begin --> `pytest tests/test_optional_component_flag_contract.py tests/test_removal_symmetry.py tests/test_enforcement_config_staged.py tests/test_audit_test_execution.py` <!-- audit-tests-registry-end --> . Enforced by `tests/test_audit_test_execution.py`. See [#audit-test-execution](EXPERIENCE.md#audit-test-execution).
- **Audit-row self-enforcement.** Any commit that adds an audit row, contract paragraph, or policy section MUST sweep every existing sibling site in the same commit and ship a witness that iterates over the whole scope. See [#audit-row-self-enforcement](EXPERIENCE.md#audit-row-self-enforcement).
- **Loss-form changes are triple-touch.** Implementation + assembly docstring/Hyperparameters formula comment + `reports/opg_doc.tex` equation/parameter table. See [#loss-form-triple-touch](EXPERIENCE.md#loss-form-triple-touch).
- **Untested-path executability.** New control-flow branches need at least one focused test or a smoke that visits them before commit. See [#untested-path-executability](EXPERIENCE.md#untested-path-executability).
- **Sibling-fanout DRY gate.** ≥3 parallel siblings × ≥3 code sites ⇒ a single registry/tuple/dict drives the fanout. Plot/log/parser layers read from the same registry as the trainer. See [#sibling-fanout-dry-gate](EXPERIENCE.md#sibling-fanout-dry-gate).
- **Promotion propagation.** Promoting an iteration updates `Hyperparameters` defaults + every mirroring `__init__` signature + hard-coded test values + CLAUDE.md "Current Architecture" + `reports/opg_doc.tex` + `hypotheses.md` in the same commit. See [#promotion-propagation](EXPERIENCE.md#promotion-propagation).
- **Loss-quantity / gate-quantity alignment** — penalty, gate, formula comment, and `reports/opg_doc.tex` must name the same mathematical object. See [#loss-gate-quantity-alignment](EXPERIENCE.md#loss-gate-quantity-alignment).
- **Root-cause fix preference.** Prefer architecture/parameterization → shared controllers → invariant-level regularizers; metric-specific losses only as temporary ablations with an exit plan. Mirrored in `reports/opg_doc.tex`.
- **Scalar-semantic shift triple-touch** — meaning changes (units, ownership, inclusion, nullability) update implementation + every caller + every doc surface, with a numeric test. See [#scalar-semantic-shift](EXPERIENCE.md#scalar-semantic-shift).
- **Flag-to-effect contract.** Every new `use_X` Hyperparameter MUST have an observable, asserted effect on the training-path tensor flow OR scored artifact at the project's default `train_seq_len`, exercised by a flip-True-vs-baseline test. Enforced by `tests/test_optional_component_flag_contract.py`. See [#flag-to-effect-contract](EXPERIENCE.md#flag-to-effect-contract).
- **Removal-symmetry sweep** — removing a Hyperparameter / loss / diagnostic is a same-commit sweep across code, `reports/opg_doc.tex`, docstrings, prescriptions, and tests. Enforced by `tests/test_removal_symmetry.py`. See [#removal-symmetry-sweep](EXPERIENCE.md#removal-symmetry-sweep).
- **Vendored-workspace hygiene.** Replication/vendoring workspaces gitignore third-party source trees and volatile run artifacts (`worktrees/`/`runs/`/`.envs/`) at the directory level; commit only manifest+scripts+docs; verify with `git add -An` before commit. Enforced by `tests/test_baselines_workspace_hygiene.py`. See [#vendored-workspace-hygiene](EXPERIENCE.md#vendored-workspace-hygiene).
- **Committed-report tracked evidence.** A committed report's numbers must resolve to tracked evidence (a frozen snapshot) or carry a dated provenance caveat — never cite gitignored, regenerable artifacts as standing fact. Enforced by `tests/test_baselines_workspace_hygiene.py`. See [#committed-report-tracked-evidence](EXPERIENCE.md#committed-report-tracked-evidence).
- **Smoke must exercise target.** A "passed" smoke verifies the named target (not a generic stand-in / empty import); failure classifiers key off the terminal traceback, and heuristic `blocked_*` must not silence the gate. Enforced by `tests/test_baselines_workspace_hygiene.py`. See [#smoke-must-exercise-target](EXPERIENCE.md#smoke-must-exercise-target).

## Promotion Rules

- **Primary gate:** promote if full-validation post-quant `val_bpb` improves and artifact is under 16 MB. Fast-only finals are non-promotable; diagnostic failures are next-iter prescriptions, not blockers.
- **Procedure on improvement:** run the review chain and audit, promote with `bash experiments/update_results.sh --promote`, then review/simplify before committing.
- **Strict generalization:** if a change's functional class exactly contains the previous baseline, promote unconditionally and tune the optimization path instead of reverting. See [#strict-generalization](EXPERIENCE.md#strict-generalization).
- **Non-improvements:** revert to the previous good state while preserving logs and hypothesis evidence.

## Submission Process

1. Run 3 seeds on 8xH100 (e.g. 42, 1337, 2024).
2. Report mean and standard deviation of `val_bpb`.
3. Create `records/track_10min_16mb/YYYY-MM-DD_<name>/`.
4. Include `README.md`, `submission.json`, `train_gpt.py`, and training logs.
5. Submit PR to `main`.

## Where Details Go

- `EXPERIENCE.md`: incidents, lessons, runbook, metric definitions, architecture rationale, verification recipes.
- `experiments/docs/hypotheses.md`: active queue, per-iteration evidence, verdicts, confounds, research archive.
- `train_gpt.py::Hyperparameters`: current config defaults.
- `reports/opg_doc.tex`: paper-facing algorithm description; update when implementation intentionally diverges.
