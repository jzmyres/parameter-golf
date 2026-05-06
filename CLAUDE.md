# Parameter Golf - Agent Guide

This file is the enforcement surface: it states the principle, the short rationale, and where the detailed explanation lives. Keep it concise. Put incident history, examples, verification recipes, and long runbook detail in `EXPERIENCE.md`.

## Read First

- **Mandatory memory pass.** Before any code change, doc edit, training launch, or commit, read the memory files listed in the project memory index. Individual memory files are not assumed loaded. Details and categories: [`EXPERIENCE.md#pre-action-memory`](EXPERIENCE.md#pre-action-memory).
- **Incident rules first.** Before code or experiment work, read `EXPERIENCE.md` Section 1 index and any incident linked from the relevant audit row.
- **Research ledger first.** Before choosing, launching, closing, or documenting an iteration, read `experiments/hypotheses.md`.
- **Config source of truth.** `train_gpt.py::Hyperparameters` is authoritative. This file may summarize defaults, but it must never become the config source.
- **New memory files.** Add new memory files to the memory index and to the appropriate documentation pointer in the same commit that creates them, or future sessions will miss the directive.

## Project Invariants

- **Challenge constraints.** The artifact must stay under 16,000,000 bytes and submission training must fit the 600 s 8xH100 budget. These constraints define viability, not just packaging. Details: [`EXPERIENCE.md#project-constraints`](EXPERIENCE.md#project-constraints).
- **Scoped edits.** Default writable surface is `train_gpt.py`, focused tests, and project docs. Do not modify `data/`, tokenizer/eval harness code, `records/`, package manifests, or dependencies without explicit user approval. The constraint prevents accidental benchmark drift.
- **No new packages by default.** If the user explicitly approves a dependency, install with `uv pip install <pkg>` and update requirements in the same commit. Reproducibility beats local convenience.
- **DDP first.** Training code must work under single GPU and `torchrun` DDP. Any GPU-count-specific path must use `world_size` correctly because dev and submission hardware differ.
- **Step-matched comparisons.** Default dev comparisons are by equal step count, not wall-clock. Wall-clock becomes primary only for submission or throughput-focused iterations. Details: [`EXPERIENCE.md#runbook`](EXPERIENCE.md#runbook).
- **Expert health final-only.** Expert-health diagnostics normalize per-component shares for readability; total routed mass is tracked separately and must not be conflated with balance.
- **bf16 training default.** Training-path autocast stays bf16; do not silently promote training compute to fp32/fp64 except for explicit reversibility accumulators or approved diagnostics.
- **Profile before throughput claims.** Throughput work needs trace-backed diagnosis, not log fragments. SDPA replacements at T=2048 are presumed suspect until profiled.

## Standing Directives

- Keep `experiments/hypotheses.md` synchronized in real time: read before every iter, update after every iter, and include required metrics.
- Prefer architecture exploration over blind hyperparameter tuning; throughput iterations take priority when the active queue marks them throughput-bearing.
- Disable `torch.compile` for dev unless the iteration explicitly tests compile behavior; still keep code compile/DDP-safe.
- Always use all visible GPUs via DDP for normal runs; use single-GPU only for debug.
- Before long launches, check GPU/process state (`pgrep`, `nvidia-smi`) and verify data/tokenizer paths.
- Use `conda run --no-capture-output` when wrapping conda commands that must stream logs.
- After failures, healthcheck cadence is 5 minutes; after 3 healthy checks, widen to 20 minutes.
- When closing an iteration, record the full active config, especially regularizer stack and flags.
- NTP descent rate and per-wallclock equivalent are permanent H-claim metrics.
- Cumulative averages lie early; compute per-step deltas for throughput before drawing conclusions.
- Load-balance is not sparsity: CV measures balance, per-token entropy measures specialization, and global entropy measures utilization.
- For fast paths, dispatch on `torch.is_grad_enabled()` plus real grad needs; `Parameter.requires_grad` alone is not a runtime-mode predicate.
- Decouple antagonistic regularizers; if two objectives fight, make one a metric or run an explicit ablation.
- Anneal sparsity coefficients from zero unless a controlled test justifies full-strength cold start.
- Treat DEQ as a fixed-point model, not a depth stack. Refinement is a separate forward pass.
- Prefer MLA over MHA/GQA, full-dim low-rank experts over bottlenecks, and pure softmax MoS routing.

## Environment & Files

- Conda env: `conda activate opg`.
- Dependencies: `requirements.txt`; authorized installs use `uv`, not `pip`.
- Data is read-only: `./data/datasets/fineweb10B_sp1024/`; tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`.
- Model source: `train_gpt.py`; focused tests live under `experiments/test_*.py` or `tests/` as appropriate.
- Research docs: `EXPERIENCE.md` for details/rationale, `experiments/hypotheses.md` for claims/results, `opg_doc.tex` for paper-facing algorithm text.
- Runtime outputs are untracked unless explicitly promoted: `run.log`, `results.tsv`, `experiments/training_logs/*`, `experiments/weights/*`.
- Optional debug plotting: `auto_plot_on_val=True` (default) refreshes `experiments/*.png` on each validation via `experiments/plotting_hook.py` (rank-0 only; silent no-op if matplotlib or `experiments/plot_*.py` modules are absent).
- Historical submissions in `records/` are read-only.
- Reference implementations are read-only: RevDEQ at `/home/mzhong4/work/research/rdeq/WIP-ARWDEQ/code/arwdeq/qwen3_utmoe_revdeq.py`; TSU/CTP/NTP/MoS at `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py`.

## Current Architecture

This is an orientation mirror only. Check `train_gpt.py::Hyperparameters` before launch.

- **Core:** 12-layer RevDEQ-style shared block, `model_dim=768`, 8 query heads, 4 KV heads, sequence length 2048, vocab 1024, tied embeddings, train batch tokens 524,288.
- **Experts:** 16 total experts with 1 always-on shared expert; routed experts are full-D LoRA-style with attention rank 64 and MLP rank 96.
- **Solver:** Parcae-style per-dim damping/injection enabled; `parcae_reversibility_floor=0.1`; `deq_bptt_k=3`; K-jitter default `(16, 24)`; scalar beta path is fallback only.
- **Default-off paths:** `num_refinements=0`, `use_ctp=False`, `use_nsa_attention=False`, `lyapunov_coef=0`, `denoising_coef=0`, `mos_output_diversity_coef=0`, `logit_softcap=0`, `bigram_vocab_size=0`.
- **Optimizer:** Muon/AdamW groups with PE-NS enabled by default; weight decay 0.01; grad clip 1.0; warmdown fraction 0.72. See code for exact LR groups.
- **Routing/loss stack:** router CV, MoS CV, per-token entropy, and expert diversity are decomposed direct-coefficient terms; CV uses combined routed mass.
- **Quantization/eval:** int6 per-row quantization plus zstd-22 compression is the scored artifact path; sliding-window eval remains enabled.

## Run Commands

Default dev run: 1000 iters, no wallclock cap, all visible GPUs.

```bash
conda activate opg
torchrun --standalone --nproc_per_node=gpu train_gpt.py
```

Use `--iterations=N` for shorter runs. If the `gpu` alias is unsupported, use an explicit count such as `--nproc_per_node=2`.

Submission-style run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --max-training-seconds=600
```

Single-GPU `python train_gpt.py` is debug-only. Quick metric extraction:

```bash
grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log
```

Full environment, file, logging, and plotting details live in [`EXPERIENCE.md#environment-and-files`](EXPERIENCE.md#environment-and-files) and [`EXPERIENCE.md#experiment-loop-details`](EXPERIENCE.md#experiment-loop-details).

## Architecture Principles

- **RevDEQ fixed point.** The model output is a fixed point of a learned map, with the DEQ solver loop separate from optional refinement loops. This separation protects the model class from confusing "more solver depth" with "another prediction pass." Details: [`EXPERIENCE.md#revdeq-architecture-details`](EXPERIENCE.md#revdeq-architecture-details).
- **Reversibility is a correctness constraint.** RevDEQ reconstruction uses fp64 add/sub and explicit custom-autograd inputs. Any tensor that affects the reverse pass and needs gradients must cross the `RevDEQFunction.apply(...)` boundary. Details: [`EXPERIENCE.md#custom-autograd-input`](EXPERIENCE.md#custom-autograd-input).
- **DEQ input conditioning.** The DEQ block may add output-side `x0` only through the learnable per-dim Parcae ZOH coefficient `B_bar = Delta * B`; unconditional shortcuts like `T = x0 + Delta` are prohibited. Current form: `T_theta(z, x0) = B_bar * RMSNorm_learn(x0) + Delta(z, x0)`. Any equation change updates `train_gpt.py`, targeted tests, and `opg_doc.tex` together.
- **Parcae floor is not a tuning knob.** Coefficients that divide reverse reconstruction need a lower bound; changing the floor requires code, tests, smoke tolerance, and paper-doc updates together. Details: [`EXPERIENCE.md#revdeq-reversibility-floor`](EXPERIENCE.md#revdeq-reversibility-floor).
- **Backward-only floors are forbidden.** Coefficients that do not divide reverse reconstruction, especially `B_bar`, must not gain artificial floors; doing so biases the ZOH form.
- **Interpret DEQ diagnostics by definition.** Under TBPTT, `deq_recon_err` measures forward distance travelled, not true reconstruction error. Use `deq_iter_conv_rel`, residuals, K-sweep, and full-BPTT smoke when true reconstruction is needed. Details: [`EXPERIENCE.md#deq-recon-err-interpretation`](EXPERIENCE.md#deq-recon-err-interpretation).
- **Soft dense routing only.** All experts process all tokens. Top-K gather, capacity drop, argmax dispatch, and threshold skips break RevDEQ reversibility unless explicitly outside the RevDEQ path or mathematically smooth. Details: [`EXPERIENCE.md#no-top-k-dispatch`](EXPERIENCE.md#no-top-k-dispatch).
- **Expert independence is hard.** Every trainable parameter inside an expert computation path belongs to exactly one expert. Shared learned scales silently couple experts and projections; non-learned ops and routers may be shared. Details: [`EXPERIENCE.md#prenorm-scale-independence`](EXPERIENCE.md#prenorm-scale-independence).
- **Router placement is semantic.** Routers route on component input before expert computation. Attn/MLP routing is `softmax_or_entmax(allocation) * sigmoid(gate)` and is not renormalized; MoS routing is pure softmax.
- **Full-D LoRA-style experts are the standard.** Expert rank constrains parameter count, not activation width: attention and MLP computation stay in full `model_dim`. Bottleneck expert bodies are closed as a scaling direction. Details: [`EXPERIENCE.md#bottleneck-experts-closed`](EXPERIENCE.md#bottleneck-experts-closed).
- **Per-expert MLA is the attention standard.** Each expert owns Q, KV compression/decompression, K-rope, Wo, gated-attention parameters, and learned norms. Head-packed SDPA preserves the GQA ratio.
- **FSQ and CTP are retained as ablation paths.** FSQ in MoS and CTP/refinement machinery remain code paths, but baseline promotion runs keep them disabled unless explicitly testing them.
- **Routing metrics have separate axes.** Global entropy means utilization; per-token entropy means specialization; CV means balance; router mass is separate from normalized shares. Do not collapse these into one "health" reading. Details: [`EXPERIENCE.md#routing-health-metrics`](EXPERIENCE.md#routing-health-metrics).
- **Routing regularizers operate on routed mass.** Router regularizers use combined `p = softmax/entmax(allocation) * sigmoid(gate)` so load balance sees gate effects. MoS softmax is the exception. Details: [`EXPERIENCE.md#routing-reg-input-invariant`](EXPERIENCE.md#routing-reg-input-invariant).
- **Regularizers should be decomposed.** CV handles usage, per-token entropy handles specialization, diversity handles output direction, and the optimizer handles norms. Bundled penalties are allowed only as explicit ablations because they mix objectives. Details: [`EXPERIENCE.md#routing-health-metrics`](EXPERIENCE.md#routing-health-metrics).
- **Disabled techniques stay documented outside CLAUDE.** SWA, BigramHash, FSQ, Lyapunov, HyDRA, CTP, and variance-reg status live in [`EXPERIENCE.md#disabled-techniques`](EXPERIENCE.md#disabled-techniques).

## Research Protocol

- **Setup discipline.** Start from a clean understanding of branch state, recent commits, `results.tsv`, `run.log`, and the relevant hypothesis entries before changing code.
- **One focused change per iteration.** A clean hypothesis needs a single controlled variable unless the entry explicitly states it is testing a bundle. Confounded results can be useful, but they cannot verify individual causes.
- **TDD for architecture.** For architectural changes, add or update focused tests first when feasible. Minimum categories: shape, gradients through fixed point, quantization roundtrip, DDP correctness, and artifact size when touched.
- **Smoke before long train.** Smoke must show loss decreasing, finite grads, no NaN/Inf, convergence not exploding, and healthy routing. Reconstruction-near-precision is only expected in full-BPTT smoke.
- **Hypothesis log is mandatory.** Every iteration entry records the claim, evidence, status, confounds, int6 result, K-sweep table, trajectory, acyclicity-prime check, and NTP descent-rate windows. Partial previews must emit computable components and Caveat missing ones. Details: [`EXPERIENCE.md#hypothesis-log-detail`](EXPERIENCE.md#hypothesis-log-detail).
- **Update plots/log rotation after runs.** `experiments/update_results.sh` is part of the iteration loop, not optional housekeeping.
- **Architecture beats knob churn.** Prefer changes that alter model capacity, routing, solver behavior, or evaluation capability over blind hyperparameter sweeps.
- **Simplicity criterion.** All else equal, simpler wins. Small score gains do not justify brittle code; code removal with equal results is a keep.
- **Pre-commit review chain.** Before every commit: `/simplify`, `coderabbit:review`, `pr-review-toolkit:review-pr`, `superpowers:requesting-code-review`, run this audit checklist, then apply simple first-principled fixes.
- **New audit rules.** Add incident-driven rules to `EXPERIENCE.md` Section 1 with an index row and cite them here concisely; do not grow long rule bodies in `CLAUDE.md`.

## Audit Checklist

Run before commits that touch `train_gpt.py`, model tests, `CLAUDE.md`, `EXPERIENCE.md`, or `experiments/hypotheses.md`.

> **Meta-principle:** read metric definitions before values; compute from raw when in doubt; user pushback is a signal to re-derive, not defend. Details: [`EXPERIENCE.md#reading-derived-metrics`](EXPERIENCE.md#reading-derived-metrics).

- **Single source of truth.** Tunable knobs live in `Hyperparameters`, are CLI-plumbed when needed, and consumers read `args.<field>`. Details: [`EXPERIENCE.md#config-drift`](EXPERIENCE.md#config-drift), [`EXPERIENCE.md#hyperparameter-fanout`](EXPERIENCE.md#hyperparameter-fanout).
- **Four-touch new knobs.** New knobs require: `Hyperparameters` field, CLI parser if user-facing, consumer reads from `args`, and docs/paper mirror when the knob belongs there. Effective magnitude must match documented magnitude or be explicitly explained.
- **Permutation consistency.** Related tensor layout transforms must use consistent index tuples; manual flatten/reshape needs an einsum-equivalence test. Details: [`EXPERIENCE.md#permutation-consistency`](EXPERIENCE.md#permutation-consistency).
- **Optimizer coverage.** Every trainable parameter must land in exactly one optimizer group; constructor assertions enforce this, but new parameter families still need review.
- **Dead-code removal.** Removing a feature means removing code paths, docs, tests, diagnostics, and constant-valued tracking in the same change. Details: [`EXPERIENCE.md#dead-code-tracking`](EXPERIENCE.md#dead-code-tracking).
- **Module aliases.** If modules are merged or aliased, optimizer filters and diagnostics must use canonical parameter names or `id()` dedup. Details: [`EXPERIENCE.md#router-alias`](EXPERIENCE.md#router-alias).
- **No hot-path syncs.** Training hot path must not branch on CUDA scalar values or use accidental `.item()` / `.cpu()` syncs outside log sites. Details: [`EXPERIENCE.md#hot-path-sync`](EXPERIENCE.md#hot-path-sync).
- **Explicit boundaries.** Wrapper, device, and optional-telemetry boundaries need explicit unwraps, tensor-safe control flow, graceful shell fallbacks, and contract tests. Details: [`EXPERIENCE.md#explicit-boundary`](EXPERIENCE.md#explicit-boundary).
- **Custom autograd inputs.** Grad-needing non-parameter tensors cross `apply(...)` explicitly, optional tensors live on `ctx`, and re-instantiated leaves use clone semantics. Details: [`EXPERIENCE.md#custom-autograd-input`](EXPERIENCE.md#custom-autograd-input).
- **Predicate migrations.** Optimizer groups, quantization tiers, tensor-pattern lists, EMA keys, hooks, and diagnostic buckets need before/after enumeration and tests. Details: [`EXPERIENCE.md#routing-predicate-migration`](EXPERIENCE.md#routing-predicate-migration).
- **Identifier uniqueness.** A name must not mean both method and cached attribute across wrapper-adjacent classes. Details: [`EXPERIENCE.md#identifier-uniqueness`](EXPERIENCE.md#identifier-uniqueness).
- **Learned norm-scale independence.** Each learned scale conditions exactly one linear weight; shape follows ownership. Details: [`EXPERIENCE.md#prenorm-scale-independence`](EXPERIENCE.md#prenorm-scale-independence).
- **Doc-code invariant.** If `opg_doc.tex` describes an algorithm and implementation intentionally differs, document the practical deviation. Details: [`EXPERIENCE.md#doc-code-invariant`](EXPERIENCE.md#doc-code-invariant).
- **Diagnostic gates follow flags.** Disabled components do not emit live diagnostics or stale prescriptions; prescriptions target the component-specific lever. Details: [`EXPERIENCE.md#diagnostic-gate-component-awareness`](EXPERIENCE.md#diagnostic-gate-component-awareness).
- **Keep CLAUDE concise.** This file keeps principles and brief rationale only; long examples and iter history go to `EXPERIENCE.md` or `experiments/hypotheses.md`. Details: [`EXPERIENCE.md#claude-md-size-budget`](EXPERIENCE.md#claude-md-size-budget).
- **Cumulative metrics are not instantaneous.** Before s50, compute per-step deltas from raw `train_time`; do not make throughput decisions from cold-start-contaminated cumulative `step_avg`. Details: [`EXPERIENCE.md#cumulative-metric-misread`](EXPERIENCE.md#cumulative-metric-misread).
- **Routing-reg input invariant.** Router regularizers operate on combined routed mass, not renormalized shares that hide gate effects. Details: [`EXPERIENCE.md#routing-reg-input-invariant`](EXPERIENCE.md#routing-reg-input-invariant).
- **No discrete dispatch in RevDEQ.** Top-K, capacity, argmax, and hard threshold dispatch require explicit proof or must remain off under RevDEQ. Details: [`EXPERIENCE.md#no-top-k-dispatch`](EXPERIENCE.md#no-top-k-dispatch).
- **Grad-enabled checks.** Fast dispatch uses `torch.is_grad_enabled()` and actual grad requirements; `Parameter.requires_grad` alone is not a runtime-mode predicate. Details: [`EXPERIENCE.md#grad-enabled-vs-requires-grad`](EXPERIENCE.md#grad-enabled-vs-requires-grad).
- **Partial previews are complete.** Short runs still report every mandated component they can compute, with Caveats for missing artifacts. Details: [`EXPERIENCE.md#partial-preview-completeness`](EXPERIENCE.md#partial-preview-completeness).
- **Move means tracked.** When relocating files (`active/` → `archive/`, package reshuffles, etc.), the destination MUST be `git add`-ed in the same commit as the source deletion. Verify with `git status --short | grep "^??"` before staging — any `??` paths inside the moved tree are silent dataloss risks. Details: [`EXPERIENCE.md#move-tracked-invariant`](EXPERIENCE.md#move-tracked-invariant).

## Promotion Rules

- **Primary gate:** promote if post-quant `val_bpb` improves and artifact is under 16 MB. Diagnostic failures become next-iteration prescriptions, not promotion blockers.
- **Procedure on improvement:** run the review chain and audit, promote with `bash experiments/update_results.sh --promote`, then review/simplify before committing the improvement.
- **Strict generalization:** if a change's functional class exactly contains the previous baseline, promote unconditionally and tune the optimization path instead of reverting. Include the recovery setting, representability argument, and optimizer-access argument in the commit message. Details: [`EXPERIENCE.md#strict-generalization`](EXPERIENCE.md#strict-generalization).
- **Non-improvements:** if the change is not a strict generalization and scored quality is equal/worse, revert to the previous good state while preserving logs and hypothesis evidence.

## Submission Process

1. Run 3 seeds on 8xH100, for example 42, 1337, and 2024.
2. Report mean and standard deviation of `val_bpb`.
3. Create `records/track_10min_16mb/YYYY-MM-DD_<name>/`.
4. Include `README.md`, `submission.json`, `train_gpt.py`, and training logs.
5. Submit PR to `main`.

## Where Details Go

- `EXPERIENCE.md`: incidents, lessons, detailed runbook, metric definitions, architecture rationale, verification recipes.
- `experiments/hypotheses.md`: active queue, per-iteration evidence, verdicts, confounds, and research archive.
- `train_gpt.py::Hyperparameters`: current config defaults.
- `opg_doc.tex`: paper-facing algorithm description; update when implementation intentionally diverges.
