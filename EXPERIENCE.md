# EXPERIENCE — Incident Archive & Lessons Learned

This file has two roles, in this order:

1. **Incident-Driven Rules Archive (§1)** — dated postmortems, ported from former `CLAUDE.md` development-practice text. Each section documents one bug class that shipped, the root cause, and the verification recipe. `CLAUDE.md` cites these from its audit checklist.
2. **Lessons Learned (§2)** — generic guardrails from research-process experience, not tied to specific code paths. Background reading; not enforcement.

`CLAUDE.md` is the **enforcement surface** (principle + concise rationale + reference). This file is the **historical record and detail store** (why the rule exists, examples, verification recipes, and runbook detail). New rules from pre-commit reviews are added to §1, then cited from the `CLAUDE.md` audit checklist — they MUST NOT accrete in `CLAUDE.md` itself.

---

## §1. Incident-Driven Rules Archive

### Index

| Date       | Anchor                                                       | One-line summary                                                          |
|------------|--------------------------------------------------------------|---------------------------------------------------------------------------|
| 2026-04-15 | [#config-drift](#config-drift)                               | Five drift defects shipped together — config duplicated across files       |
| 2026-04-17 | [#router-alias](#router-alias)                               | Module aliases broke optimizer param filter; router silently @ 4.4× LR     |
| 2026-04-18 | [#dead-code-tracking](#dead-code-tracking)                   | ~100 lines of always-1.0 tracking survived feature removal                 |
| 2026-04-18 | [#hot-path-sync](#hot-path-sync)                             | Lyapunov `.item()` syncs after a "no .item()" comment                      |
| 2026-04-23 | [#explicit-boundary](#explicit-boundary)                     | Compile-wrapper write + Lyapunov branch + shell `set -euo pipefail`        |
| 2026-04-23 | [#custom-autograd-input](#custom-autograd-input)             | B̄ stored on Block attribute; no grad edge through `RevDEQFunction`         |
| 2026-04-23 | [#strict-generalization](#strict-generalization)             | iter 66b vs 74b — capacity question vs landscape question                  |
| 2026-04-24 | [#prenorm-scale-independence](#prenorm-scale-independence)   | One `state_norm.weight` shared across 8+ distinct linears                  |
| 2026-04-24 | [#identifier-uniqueness](#identifier-uniqueness)             | `_parcae_b_bar` as method on GPT and attribute on Block                    |
| 2026-04-26 | [#diagnostic-gate-component-awareness](#diagnostic-gate-component-awareness) | After iter 94 (use_ctp=False), `mos_ctp` diagnostic + WD prescription stale |
| (meta)     | [#routing-predicate-migration](#routing-predicate-migration) | Cross-incident pattern (2026-04-15 / -17 / -18 / -23): silent predicate migration |
| (pre-arch) | [#permutation-consistency](#permutation-consistency)         | k_rope outlier permute index produced silent transposition (commit ec1048b) |
| (pre-arch) | [#doc-code-invariant](#doc-code-invariant)                   | Pseudocode in `opg_doc.tex` diverged from implementation; required co-update |
| 2026-04-28 | [#hyperparameter-fanout](#hyperparameter-fanout)             | Five "tunable" knobs documented in CLAUDE.md §5 were hardcoded inside constructors |
| 2026-04-30 | [#claude-md-size-budget](#claude-md-size-budget)             | CLAUDE.md hit 51 887 chars (>40k perf warning) from accreted iter-history annotations |
| 2026-04-30 | [#variance-reg-ns-cascade](#variance-reg-ns-cascade)         | Iter 117 NaN cascade attributed to PE-NS was actually variance-reg gradients on entmax exact-zeros |
| 2026-05-02 | [#cumulative-metric-misread](#cumulative-metric-misread)     | Iter 117b-3 erroneously killed at s10 because cumulative `step_avg` was misread as instantaneous step time |
| 2026-05-05 | [#partial-preview-completeness](#partial-preview-completeness) | iter-142-refactor 100-step preview silently skipped the descent-rate component because the run was "partial"  |
| 2026-05-06 | [#move-tracked-invariant](#move-tracked-invariant)           | Components-archive move staged 9 deletions but left destinations untracked; archive-not-gitignored copies would have vanished |
| 2026-05-06 | [#loss-form-triple-touch](#loss-form-triple-touch)           | iter 142b cv² promotion shipped with stale `cv_hinge` docstring + stale `opg_doc.tex` parameter table & loss equation; magnitude-only audit didn't catch the structural change |
| 2026-05-08 | [#untested-path-executability](#untested-path-executability) | iter145r resume `step=0` reset + iter103 chained `Block` alias landed structurally-correct but never end-to-end exercised |
| 2026-05-08 | [#sibling-fanout-dry-gate](#sibling-fanout-dry-gate)         | iter145 EMA-loss triplet hand-rolled at 9 sites; future term would be a 9-place grep-and-paste |
| 2026-05-08 | [#promotion-propagation](#promotion-propagation)             | iter145r promotion changed `Hyperparameters` defaults but signature defaults / regression test / `opg_doc.tex` lagged silently |
| 2026-05-09 | [#loss-gate-quantity-alignment](#loss-gate-quantity-alignment) | iter146 Lyapunov penalty trains on a Frobenius/√D Hutchinson proxy while the post-final `lip_ub` gate measures operator norm; both share the colloquial name "Lyapunov / contraction" |
| 2026-05-09 | [#scalar-semantic-shift](#scalar-semantic-shift)              | `max_training_seconds` semantic flipped from process-total to training-only without updating callers; canonical `--max-training-seconds=600` would silently overrun the 600 s 8×H100 invariant by 120 s |
| 2026-05-11 | [#enforcement-config-staging](#enforcement-config-staging)   | iter146-151 commit window introduced `pytest.ini` to convert legacy `return failures` test patterns into hard CI failures via `PytestReturnNotNoneWarning`; the file was untracked while the test rewrites were staged — would have silently disarmed the gate on the next contributor's machine |
| 2026-05-11 | [#audit-row-executability](#audit-row-executability)         | three-reviewer audit found that the Sibling-fanout DRY gate row added in `5ecf324` was first violated one commit later (EMA log fields with no parser entries); same review found the Enforcement-config staging row shipped only a manual recipe — both symptoms of audit rules without executable witnesses |
| 2026-05-12 | [#flag-to-effect-contract](#flag-to-effect-contract)         | iter152-batch un-archived 7 components and wired 8 new `use_*` flags; pre-commit review found `use_rr_attention=1` was a no-op at the project default `train_seq_len=2048` (silent fallback to dense SDPA above 512 tokens) and `use_gptq`/`use_lqer`/`use_ttt_eval`/`use_caseops` ran only synthetic-tensor or hardcoded-string smokes while the startup banner advertised them as "1" — a research-log-integrity hazard |
| 2026-05-12 | [#audit-row-self-enforcement](#audit-row-self-enforcement)   | iter152 profile-system + capability-registry diff added the "Root-cause fix preference" audit row, yet the same diff's `_prescribe_failure_fix.mos_ortho` branch returned a per-symptom loss bump with no exit-ablation framing — exactly the failure mode the new policy forbids. A 5-agent parallel review found that the witness for the new rule covered only 2 of 8 sibling branches |
| 2026-05-15 | [#removal-symmetry-sweep](#removal-symmetry-sweep)           | cleanup #47 + cleanup #50 (commits 7410cb9 + 56b2f29) removed `lip_ub_T/S/F` + `fp_bound` + `power_jvp_F` + `router_load_cv_coef` + `mos_load_cv_coef` from `Hyperparameters` and loss assembly, but left ~14 stale references across `opg_doc.tex` (parameter table rows + equation blocks still showing the removed CV terms, feature table still claiming `deq_prefix_anchors=false`, Local Stability section still describing `lip_ub_F` as an "auxiliary K-sweep estimator") and ~10 fossil docstrings/comments in `train_gpt.py`. iter163 promotion commit (884b132) simultaneously violated the Sibling-fanout DRY gate by emitting `consistency_anchor_loss`/`consistency_ext_loss` log fields with no `plot_metrics.py` parser entries, no `test_training_contracts.py` required_fields update, and no `test_plot_metrics_parse.py` fixture. Cleanup #51 added the "Removal-symmetry sweep" audit row, the companion `tests/test_removal_symmetry.py` enforcement test, and swept all sibling sites in the same commit |

### Section template

```
### <slug>

**Date:** YYYY-MM-DD review of iter NN
**Rule in CLAUDE.md:** audit checklist row · <section name>

**What happened.** <Concrete description: code paths, symptom, how it was caught.>

**Root cause.** <One paragraph.>

**The rule.** <Verbatim copy of the rule body so this file is self-contained.>

**Verification recipe.** <Grep commands, contract test names, what to look for.>

**Cross-references.** Related rules · affected commits.
```

---

### config-drift

**Date:** 2026-04-15 review
**Rule in CLAUDE.md:** §9 row "Single-source-of-truth (Hyperparameters)"

**What happened.** Five drift defects shipped in one PR: a stale "27b" comment lived inside iter 27d code; CLAUDE.md's config table was two phases behind the real code; `num_experts` was hard-coded in three different classes; a test was silently relaxed from `== 6` to `>= 2`; and `opg_doc.tex` described a removed `gg_gate` that no longer existed.

**Root cause.** Configuration was duplicated across files with no single source of truth, so each editor only updated the file in front of them. Partial updates compounded into systemic drift.

**The rule.** Every tunable architectural knob (`num_layers`, `num_heads`, `num_experts`, `model_dim`, `mlp_mult`, ranks, etc.) MUST appear exactly once — in `train_gpt.py`'s `Hyperparameters` class — and be plumbed from there to every constructor. Defaults inside sub-module `__init__` signatures are allowed only as a fallback; the authoritative value for any run is `Hyperparameters.<field>`.

1. **Add to Hyperparameters first**, then thread through `GPT.__init__` → `Block.__init__` → leaf modules. Never introduce a new knob whose only home is a constructor default.
2. **Tests assert against the config, not a literal**. `assert model.num_experts == args.num_experts` is allowed; `assert model.num_experts == 6` (or `>= 2`) is forbidden — the first catches drift, the second hides it.
3. **The documentation mirror is mirror-only**. Edits to `Hyperparameters` and the current orientation mirror (now [`#current-architecture-reference`](#current-architecture-reference), formerly `CLAUDE.md` §5) MUST land together when the mirror covers the changed knob.
4. **`opg_doc.tex` §2.1 and the "Current SOTA"/"working baseline" lines are dated artifacts**. A PR that mutates `Block.forward`, the DEQ equation, or the promoted baseline MUST update these in the same commit, or open a `TODO(paper)` ticket noting the divergence.

**Verification recipe.** Before committing any change to architectural knobs:
- `grep -n 'num_experts\|num_layers\|model_dim' train_gpt.py` — confirm constructors read from `args.<field>`, not literals.
- Tests: `assert model.<field> == args.<field>`; never compare to a hard-coded number.
- Documentation mirror updated in the same commit as `Hyperparameters` when the changed knob appears in that mirror.

**Cross-references.** Sibling family with [#router-alias](#router-alias) and [#routing-predicate-migration](#routing-predicate-migration) — all share the partial-migration root cause.

---

### router-alias

**Date:** 2026-04-17 review of iter 35
**Rule in CLAUDE.md:** §9 row "Module-alias audit"

**What happened.** Iter 35's pooled-router merge created module aliases (`attn_router` = `mlp_router` = `router` — three attribute names pointing to the same `nn.Module` instance). The optimizer param filter still looked for `attn_router.router.weight`, a name PyTorch never generates for an alias because `named_parameters()` deduplicates by tensor identity and emits the canonical (first-registered) name. The router silently trained with Muon at 4.4× the intended LR and with weight decay applied, making `router_lr` a no-op.

**Root cause.** PyTorch deduplicates `named_parameters()` by tensor identity, so the *canonical* name an alias produces is the first-registered name, not the alias name. Predicates that filter on the alias name silently miss every aliased parameter. Same root cause as [#config-drift](#config-drift): partial migration left stale references.

**The rule.** When replacing N independent modules with a single shared instance (e.g., `attn_router` + `mlp_router` → single `router`):
1. Verify all `named_parameters()` filters still match the canonical name (PyTorch deduplicates by identity).
2. Verify all per-instance loops (bias_update, health_scale, diagnostics) use `id()` dedup.
3. Remove duplicate set/get operations on the same instance.

**Verification recipe.**
- `grep -n 'attn_router\|mlp_router' train_gpt.py` — zero hits in optimizer group construction once a merge is done.
- Contract test: assert `id(self.attn_router) == id(self.mlp_router)` (when aliasing) and that the canonical name appears in `dict(model.named_parameters())`.

**Cross-references.** Same family as [#config-drift](#config-drift) and [#routing-predicate-migration](#routing-predicate-migration).

---

### dead-code-tracking

**Date:** 2026-04-18 review
**Rule in CLAUDE.md:** §9 row "Dead-code audit"

**What happened.** `gg_gate` and `inj_lin` were removed from the model, but their tracking infrastructure survived (~100 lines): fields, methods, aggregation, and prescription paths kept recording constant `1.0` values with zero diagnostic signal. Tautological validation checks on always-1.0 values could never fire.

**Root cause.** Feature removal stopped at the forward-path code paths. Diagnostic/tracking infrastructure that consumed the removed values was missed. Constant-valued tracking is dead code too.

**The rule.** When a feature is removed (e.g., `gg_gate`, `SmearGate`, `tie_attn_mlp_router`), the SAME commit must:
1. Remove ALL code paths that reference it (CLI args, constructor params, `CONTROL_TENSOR_PATTERNS`, diagnostic tracking).
2. Remove ALL doc references (CLAUDE.md tables, comments saying "removed").
3. Update ALL tests that assert on the removed feature.
4. `grep -rn` for the removed name across the entire codebase — if any match remains, it's incomplete.
5. Remove ALL tracking infrastructure (fields, methods, aggregation, prescriptions) that recorded values for the removed feature — constant-valued tracking is dead code too.

**Verification recipe.** `grep -rn '<removed_name>' .` in the same commit returns zero matches across `train_gpt.py`, tests, docs, and shell scripts.

**Cross-references.** Adjacent to [#routing-predicate-migration](#routing-predicate-migration) — both are about removing/changing things without leaving stragglers.

---

### hot-path-sync

**Date:** 2026-04-18 review
**Rule in CLAUDE.md:** §9 row "Hot-path sync prohibition"

**What happened.** Lyapunov penalty code at `train_gpt.py:3037` carried a comment "Pure-tensor math — no .item() GPU-CPU syncs" immediately followed by `if scale_t.item() > 0:`. Three more `.item()` syncs were found in the same block. The comment lied; the code didn't.

**Root cause.** Comments are not enforcement. Reviewer attention drops on text-after-the-comma even when the comma promises "no syncs". Automated grep is the only source of truth for "no sync" claims.

**The rule.** The training loop (gradient accumulation + optimizer step) MUST NOT contain any `.item()`, `.cpu()`, or Python-scalar branching on GPU tensors. All control flow must use pure-tensor math (e.g., `torch.relu`, `torch.where`). GPU→CPU syncs are only permitted at log sites (guarded by `will_log_train`).

**Verification recipe.** Before committing any training-loop code (especially Lyapunov/regularization):
```
grep -n '\.item()\|\.cpu()' train_gpt.py
```
Verify ZERO hits between the `for micro_step` loop and `train_loss /= grad_accum_steps`. Comments claiming "no .item()" are not sufficient — automated grep is the source of truth.

**Cross-references.** [#explicit-boundary](#explicit-boundary) — both involve hot-path correctness across implicit boundaries.

---

### explicit-boundary

**Date:** 2026-04-23 review
**Rule in CLAUDE.md:** §9 row "Explicit boundary" / "Compile-wrapper writes"

**What happened.** Three boundary-crossing bugs shipped together in iter 66a:
1. Validation wrote `_deq_k_override` to a wrapper module instead of the underlying `GPT`, so the override silently never took effect on the training graph.
2. A Lyapunov CUDA scalar branch (`if scale_t.item() > 0`) risked hot-path synchronization on every step.
3. `update_results.sh` could abort on an incomplete log before printing the intended `val_bpb=?` fallback (because `set -euo pipefail` killed the pipeline at the missing-grep step).

**Root cause.** Each bug crossed an *implicit* boundary — wrapper, device, optional-telemetry — without making the boundary explicit in code or tests. The intent ("write to GPT", "stay on GPU", "preserve fallback") lived in comments or local context, not in enforcement.

**The rule.** Any code that crosses a wrapper, device, or optional-telemetry boundary MUST make that boundary explicit:
1. Mutate model state only on the semantic owner after unwrapping compile/DDP/DataParallel wrappers with `_unwrap_compiled_module()`.
2. Keep GPU tensors on GPU in the training hot path; use tensor math (`torch.relu`, `torch.where`, `clamp`) instead of Python branches on CUDA scalar tensors.
3. Treat optional summary metrics as optional under `set -euo pipefail`; shell summaries must preserve `?` fallbacks instead of aborting when a grep has no matches.
4. Add or update a contract test for the boundary. Comments and local intent are not enforcement.

**Compile-wrapper write pattern.** All attribute writes to `base_model.shared_block` (or any potentially-compiled module) MUST go through `_unwrap_compiled_module()`. The pattern is: `sb = _unwrap_compiled_module(base_model.shared_block)` once per training-loop scope, then use `sb` for all reads/writes.

**Verification recipe.**
- `grep -n 'shared_block\.' train_gpt.py` — every assignment site uses an unwrapped reference.
- `grep -n '\.item()\|\.cpu()' train_gpt.py` — zero in hot path (see [#hot-path-sync](#hot-path-sync)).
- Shell summaries in `experiments/update_results.sh` retain the `?` fallback under `set -euo pipefail`.
- Contract test for each boundary (e.g., `_deq_k_override` actually applies to the underlying `GPT`).

**Cross-references.** [#hot-path-sync](#hot-path-sync), [#custom-autograd-input](#custom-autograd-input).

---

### custom-autograd-input

**Date:** 2026-04-23 review (with 2026-04-24 follow-up)
**Rule in CLAUDE.md:** §9 row "Custom autograd inputs" · §10 RevDEQ Specifics

**What happened.** `B̄ = Δ·B` was computed in `GPT._deq_solve()` and stored on a `Block` attribute, but `RevDEQFunction.apply(...)` only accepted `beta` and block parameters. The reverse pass re-ran `Block.forward`, so outputs numerically depended on `B̄`, but `backward(...)` never returned a `grad_b_bar`; task-loss gradients to `parcae_raw_b` were therefore silently absent. Iter 66b learned `B̄` only through indirect couplings (the shared `Δ` step size), not through the direct task-loss path the math assumed.

**Root cause.** `torch.autograd.Function` only chains gradients through tensors that cross the `apply(...)` boundary. Reading a trainable tensor from module state inside `forward`/`backward` produces a *numerical* dependency without a *graph* dependency. Autograd has no edge to chain through unless the tensor is an explicit input.

A 2026-04-24 follow-up review caught a related sub-bug: re-instantiating `b_bar` as a leaf inside the truncated-BPTT `for _ in range(K)` loop in `backward()` via `.detach().requires_grad_(...)` made successive iterations and the y-leg/z-leg pair share underlying storage. The current backward is correct because autograd tracks by tensor identity, but any future in-place edit of `b_bar` inside `f_theta` would silently corrupt the other leg.

**The rule.** Every tensor that should receive gradients through a custom `torch.autograd.Function` MUST be an explicit `apply(...)` input and have a corresponding gradient slot in `backward(...)`.

When a saved tensor is re-instantiated as a leaf inside a `for _ in range(K)` loop in `backward()` (for truncated-BPTT or K-step unrolled reverse solves), use `.clone().requires_grad_(flag)` — NOT `.detach().requires_grad_(flag)` — so that successive iterations and multiple legs (y-leg, z-leg) within one iteration do not share underlying storage. `.detach()` returns a view; a future in-place edit to the leaf inside `f_theta` then silently aliases across legs. Cost is one small-tensor allocation per step; the robustness is permanent.

Optional tensor inputs (e.g. `b_bar` when `use_parcae=False`) MUST be stored on `ctx` as a plain attribute (`ctx.b_bar_saved = tensor_or_None`), not embedded in `save_for_backward` via an empty-tensor sentinel. `save_for_backward` is semantically a list of *real* tensors; iterating it elsewhere and assuming non-empty shapes must remain safe.

**Verification recipe.**
- For each custom autograd Function, contract test compares `apply()` gradients against an explicit unroll for *every* non-parameter tensor input (`beta`, `b_bar`, etc.).
- `grep -n '\.detach()\.requires_grad_' train_gpt.py` — zero hits inside any custom `backward()`.
- For optional inputs: `grep -n 'ctx\.<name>_saved' train_gpt.py` — assigned in `forward`, read in `backward`.

**Cross-references.** [#explicit-boundary](#explicit-boundary), [#identifier-uniqueness](#identifier-uniqueness) (the Block-attribute version of `B̄` was also a name collision with a `GPT` method).

---

### strict-generalization

**Date:** 2026-04-23, iter 66b
**Rule in CLAUDE.md:** §11 Promotion Rules

**What happened.** Iter 66b's `T_θ = B̄ ⊙ RMSNorm_learn(x₀) + Δ` was a strict generalization of iter 74b's `T_θ = Δ` — setting `parcae_raw_b → −∞` recovers the baseline forward map exactly. A short-run val_bpb regression on dev hardware made the iter look worse, prompting reflexive revert pressure. Reverting would have lost expressive capacity the baseline could not reach.

**Root cause.** Conflating *capacity* questions ("can the new form represent the old?") with *training-dynamics* questions ("did the new form's optimizer reach a good point in 1000 steps?"). Reverting a strict generalization is categorically the wrong move — any regression on a more-general form is by construction an optimization-landscape artifact (init, LR, gradient topology), not a capacity loss.

**The rule.** When an iteration's functional class **strictly subsumes** the baseline — i.e. there exists a setting of the new learnable parameters under which the iteration's forward map is *exactly* identical to the baseline's (not approximately, not "close enough") — the iteration MUST be promoted unconditionally, regardless of any val_bpb delta measured on a short dev-hardware run.

**Examples of strict generalization** (this repo):
- Iter 66b's `T_θ = B̄ ⊙ RMSNorm_learn(x₀) + Δ` strictly generalizes iter 74b's `T_θ = Δ`: setting `parcae_raw_b → −∞` (or `x0_inject_norm_weight → 0`) recovers `T_θ = Δ` exactly (up to an ε_min·Δ injection term that the optimizer absorbs).
- Changing a tied parameter to independent (e.g. iter 66a's tied `B̄ = 1 − Ā` → iter 66b's independent `B̄ = Δ·B`) generalizes as long as the tied setting is representable in the new parametrization.
- Adding a learnable gate initialized open + residual bypass: `y = (1−g)·x + g·f(x)` with `g = σ(·)` init to `0` gives `y = x` at init, recovering identity.

**Examples that are NOT strict generalization** (look-alikes to reject):
- A re-parametrization with a different output range even in the limit (e.g. adding `clamp(·, 0, c)` with `c < ∞` to a previously-unbounded output — the new form cannot represent outputs outside `[0, c]`).
- Adding a new loss term with strictly positive coefficient (the baseline is only recovered by setting the coefficient to exactly zero — not representable if the coefficient is a positive softplus).
- Replacing a full-rank projection with a low-rank factorization (low-rank cannot represent all full-rank maps).

**Required in the commit message for a strict-generalization iter:**
1. The exact setting of the new parameters that recovers the baseline's forward map.
2. A 1–2 line argument that this setting is representable in the new parametrization (e.g. "`softplus(raw_b)` can get arbitrarily close to 0; `Δ·ε_min ≈ 1e-3` is below training-noise scale").
3. How the optimizer's access to this baseline-equivalent point is preserved (LR, weight decay, init range).

**What to do if val_bpb regresses after a strict-generalization promote.**
- Do NOT revert. The new iter cannot be worse in capacity than the baseline — any regression is an optimization-landscape artifact.
- Tune in decreasing order of suspicion: (i) LR of the new params, (ii) initialization (try matching baseline at step 0 exactly), (iii) gradient flow paths if the new params sit behind a chain of reparametrizations.
- Record the diagnostic in `experiments/docs/hypotheses.md` but continue running the next queued iter on top of the new baseline.

**Verification recipe.** No grep — this is a promotion-policy invariant. The commit message for any strict-generalization iter must include the three items above, and `CLAUDE.md` Promotion Rules enforce "do not revert".

**Cross-references.** Independent of the audit-rule family; this is a workflow rule, not a code-shape rule.

---

### prenorm-scale-independence

**Date:** 2026-04-24 review (pre-iter-66b state)
**Rule in CLAUDE.md:** §9 row "Prenorm scale independence (HARD)"

**What happened.** Pre-iter-66b, a single `state_norm.weight` `nn.Parameter` was multiplied before the router *and* every attention projection (Q/KV/Kr) *and* every MLP input *and* the shared gate *and* every MoS A-bank — eight+ distinct linear maps tied to one learned vector. Knowing `state_norm.weight` told the optimizer something about all eight downstream linear weights simultaneously, violating the expert/projection independence invariant.

**Root cause.** A learned multiplicative scale and the parameter-free RMS statistic were treated as one inseparable block, so reuse of "the prenorm" looked free. In fact only the `RMSUnit(x) = x / sqrt(mean(x²) + ε)` statistic is parameter-free; the trailing scale is a *parameter* and conditions a specific linear weight. Sharing it across linears couples them.

**The rule (HARD CONSTRAINT).** All learnable prenorm scales are independent — no two distinct linear inputs inside the training graph may share the same learned scale `Parameter`. The RMS statistic (`RMSUnit(x)`) is parameter-free and may be reused freely; the learned multiplicative scale that follows it is owned exclusively by the linear input it conditions.

Formally: for every linear map `y = W·(RMSUnit(x) ⊙ g) + b` inside `T_theta` or its auxiliary heads, `g` is an `nn.Parameter` that conditions exactly one linear weight (`W`). The same `(W, g)` pair may appear in multiple forward paths (e.g. the main expert mixer and the orthogonality diagnostic), but `g` must never condition a second, distinct weight tensor.

**Shape follows the linear, not the rule.** The rule is about *independence*, not shape:
- A per-expert linear (per-expert Q/K/V/O banks, per-expert MLP gate/fc, per-expert MoS A-banks) owns a scale of shape `(E, D)` or `(E, rank)` — leading dim `E` matches the linear's per-expert dimension.
- A shared linear that the expert path routes *through* (router score, router gate, shared-expert gate, MoS routing gates) owns a scale of shape `(D,)` — there is no `E` dimension because the linear itself is not per-expert. This is still "independent": the router's score-scale and gate-scale are two distinct Parameters, even though both are shape `(D,)`.

The fix from one shared `state_norm.weight` was: one `Parameter` per linear input — `router.score_norm_weight`, `router.gate_norm_weight`, `attn.q_down_norm_weight`, `attn.kv_a_norm_weight`, `attn.kr_a_norm_weight`, `attn.k_nope_in_norm_weight`, `attn.v_in_norm_weight`, `mlp.gate_in_norm_weight`, `mlp.fc_in_norm_weight`, `shared_gate_norm_weight`, `mos_head.gate_ctp_norm_weight`, `mos_head.gate_ntp_norm_weight`, `mos_head.ctp_a_norm_weight`, `mos_head.ntp_a_norm_weight`, and any per-rank post-projection scales on the expert path.

**Verification recipe.** Before committing any code that adds a learned prenorm scale:
```
grep -n '_norm_weight' train_gpt.py
```
For each scale, verify every usage multiplies the SAME linear weight — i.e. the scale conditions one `W`. A scale that appears against two distinct `W` Parameters in forward is a violation.

Contract tests `test_expert_path_parameters_are_expert_independent` and `TestOptimizerCoverage.test_all_trainable_parameters_are_grouped_once` together catch missing scales and check per-expert shapes. The grep audit is the first line of defense against cross-linear sharing.

**Cross-references.** Adjacent to the soft-dense-routing "expert independence" hard constraint in `CLAUDE.md` Architecture Principles — both enforce that no shared trainable parameter couples expert/projection paths.

---

### identifier-uniqueness

**Date:** 2026-04-24 (iter 66b pre-commit review)
**Rule in CLAUDE.md:** §9 row "Identifier uniqueness across wrappers"

**What happened.** `_parcae_b_bar` was simultaneously a `GPT` method (computes `Δ·B` on the fly) and a `Block` attribute (threaded by `_deq_solve`). The assignment-and-clear in `_deq_solve` became dead code once all readers were migrated to take `b_bar` as an explicit argument, but the shared name made dead writes look like live state. One reviewer mis-escalated to a correctness CRITICAL because they could not distinguish the two semantics. The `/simplify` pass dropped the `Block` attribute entirely; the method on `GPT` is now the sole holder of that name.

**Root cause.** Two distinct semantics (a method that computes; an attribute that caches) bound to one identifier. Even though Python dispatch resolved correctly today, a future proxy, wrapper, or `_unwrap_compiled_module` call could promote or demote the lookup and silently flip semantics.

**The rule.** An identifier that appears on more than one class in the same call graph (e.g. `GPT`/`Block`, `GPT`/`MoSHead`, `Block`/`MLP`) MUST NOT be a method on one class and an attribute on another. Two distinct semantics for one name — even when Python dispatch resolves them today — is a latent collision.

**Checklist for every new attribute or method on a class used inside the training graph:**
1. `grep -n '\.<name>\b' train_gpt.py tests/ experiments/` — confirm no pre-existing method/attribute with the same base name on a sibling class.
2. If the name must be reused (e.g. a cached value mirroring a computed property), suffix the cache or delete the duplicate if it is dead state.
3. When adding a method whose name conflicts with a pre-existing attribute on a sibling class, rename (or delete) the attribute in the same commit and update every reader.

**Verification recipe.**
```
grep -n '\.<new_name>\b' train_gpt.py tests/ experiments/
```
Every match should resolve to the same class (or be unambiguously distinct, e.g. cache-suffixed).

**Cross-references.** [#custom-autograd-input](#custom-autograd-input) — the dead-attribute version of `_parcae_b_bar` was also entangled with the missing-grad-edge bug.

---

### diagnostic-gate-component-awareness

**Date:** 2026-04-26 review of iter 94 follow-up
**Rule in CLAUDE.md:** §9 row "Diagnostic-gate component awareness"

**What happened.** Iter 94 promoted the **NTP-only baseline** by setting `Hyperparameters.use_ctp = False`, which gates `MoSHead` so the CTP param banks (`gate_ctp`, `A_ctp_shared`, `B_denoise`, etc.) are never allocated. But two adjacent diagnostic paths kept treating CTP as a live component:

1. **Stale `mos_ctp` diagnostic (`train_gpt.py:4339`).** The post-int6 gate appended both `mos_ctp` and `mos_ntp` check_specs unconditionally. With CTP banks unallocated, `_mos_usage("ctp")` returned aliased / empty values, producing fake `mos_ctp_min_share` "dead expert" failures on a head that was never trained. `MoSHead.get_head_orthogonality("ctp")` already had the right guard (returns `0.0` if `head == "ctp" and not self.use_ctp`); the spec-emission site forgot to mirror it.

2. **Wrong retry prescription (`train_gpt.py:4471`).** Every `min_share` failure mapped to `weight_decay_mult: 1.5`. Iter 24 (H5) controlled-tested exactly this: WD bump 0.72→1.08 *worsened* `mos_ntp_min_share` (0.008→0.006). Iter 26 (H26-lb-loss) fixed dead expert by 50× MoS balance loss. The gate's prescription contradicted both verified hypotheses — and the comment string "H5 RESOLVED — routing collapse is WD-addressable" was actively wrong.

The root cause is the same in both: a feature-flag flip (CTP off) and a hypothesis-controlled refute (H5 for routing) were incorporated into the architecture but not into the diagnostic emitters that pattern-match on stale assumptions.

**Root cause.** Diagnostic gates and their downstream prescriptions accumulate string-prefix matchers and global-knob recommendations that age out of date faster than the architecture they describe. When a feature flag disables a code path, the diagnostic emitters keep firing at the dead site. When a hypothesis flips a fix's verdict (H5 verified for ortho but refuted for routing collapse), the prescription tables keep recommending the disproved fix.

**The rule.** When a feature flag disables a code path:
1. Audit every diagnostic / metric / log emitter that references the gated component. Each must either be removed or guarded by the same flag.
2. Audit every retry prescription / config-suggestion path that recommends fixes for that component. Stale fixes must be removed or rerouted to the actually-effective lever.

When a hypothesis flips a verdict:
1. The prescription dispatch must split by the same axis the hypothesis split (component prefix, in our case).
2. The hypothesis tag in the prescription string must reflect the *current* verdict; "H5 RESOLVED" was misleading because H5 stands for the ortho/weight-space axis only — H26 covers routing-space collapse.

Component-aware retry prescriptions for our gate:
- `mos_*_min_share`  → `mos_balance_mult` (H26 lever)
- `attn_*_min_share` → `attn_balance_mult` (H26 family)
- `mlp_*_min_share`  → `mlp_balance_mult`  (H26 family)
- `mos_*_ortho`      → `mos_ortho_out_coef` (head-internal regularizer)
- `attn_*_ortho`, `mlp_*_ortho` → `weight_decay` (H5 stands for ortho only — weight-space collinearity)

**Verification recipe.**
- `grep -n 'mos_ctp\|use_ctp' train_gpt.py` — every spec referencing a CTP-only attribute must live behind a `mos_head.use_ctp` guard.
- `experiments/test_arch.py::test_post_int6_gate_skips_mos_ctp_when_disabled` — instantiates `use_ctp=False`, asserts no `mos_ctp` prefix in the gate's spec list.
- `experiments/test_arch.py::test_prescribe_min_share_routes_to_balance_loss` — synthetic failure strings drive `_prescribe_failure_fix` and assert each `*_min_share` prefix maps to the matching `*_balance_mult` knob (NOT `weight_decay`).
- Fast smoke run: `torchrun … --iterations=20 2>&1 | grep "mos_ctp\|min_share\|category"`. Expect zero `mos_ctp` rows; any `min_share` prescription names a `*_balance_mult` knob.

**Cross-references.** Sibling family with [#config-drift](#config-drift) — both stem from architecture state advancing faster than auxiliary state machines (config tables, diagnostic dispatchers). Sub-task delivered alongside this incident: hardcoded `50.0` literal (`train_gpt.py:3015`) promoted to `Hyperparameters.mos_balance_mult` so the prescription path can recommend bumping it without a separate refactor.

---

### routing-predicate-migration

**Date:** Cross-incident pattern (2026-04-15 / -17 / -18 / -23, plus iter 66a pre-commit review)
**Rule in CLAUDE.md:** §9 row "Routing-predicate migration"

**What happened.** Every prior silent-migration bug in this repo traces to a classification predicate that changed in one place while pre-existing objects quietly flipped regimes elsewhere. Concrete instances:
- The router-alias bug ([#router-alias](#router-alias)) silently disabled `router_lr` because the optimizer-group filter used the alias name.
- The `shared_block.named_parameters()` scope silently froze four RMSNorm scales (`bigram.proj_norm.weight`, `mos_head.input_norm.weight`, `final_norm.weight`, `embed_norm.weight`) for an entire phase of training.
- Adding `"norm_weight"` to `CONTROL_TENSOR_PATTERNS` silently migrated `kv_norm_weight` and `hidden_norm_weight` from Muon/int6 to AdamW/fp32.

**Root cause.** Partial migrations compound. A predicate that classifies parameters/modules into regimes (optimizer groups, quantization tiers, compile targets, `CONTROL_TENSOR_PATTERNS`, `FP16_KEEP_PATTERNS`, EMA keys, gradient hooks, diagnostic buckets) changes meaning when its inputs change. Without explicit enumeration of before/after, items silently flip regimes.

**The rule.** When adding or changing any predicate that classifies parameters, modules, or tensors into regimes, the same commit MUST:
1. Enumerate every pre-existing item the new predicate matches, and record the before/after regime for each in the commit message.
2. Add (or extend) a contract test that asserts at least one representative pre-existing item lands in the intended regime — not the legacy one.
3. If the new predicate claims *full coverage* (e.g. "every trainable param"), add an assertion at construction time that raises if any required item is missing or duplicated (as `_assert_optimizer_param_coverage` does).

**Verification recipe.**
- Commit message includes a before/after enumeration table.
- Contract test at the predicate's call site (e.g. `TestOptimizerCoverage` for optimizer groups, `TestQuantizationTierAssignment` for quantization classification).
- For full-coverage predicates: a construction-time assertion (`_assert_*_coverage`) that raises on missing/duplicated items.

**Cross-references.** [#config-drift](#config-drift), [#router-alias](#router-alias), [#dead-code-tracking](#dead-code-tracking), [#explicit-boundary](#explicit-boundary). This rule is the meta-pattern; the others are specific instances.

---

### permutation-consistency

**Date:** Pre-archive (commit ec1048b, k_rope incident)
**Rule in CLAUDE.md:** §9 row "Permutation consistency"

**What happened.** When multiple tensors of identical shape (`(E,B,T,H,d)`) were permuted-and-reshaped to the same target layout (`(B,E·H,T,d)`), one tensor's `permute(...)` index tuple differed from its siblings. The result was a silent transposition: shapes matched downstream but semantics did not, so values were stored where the consumer expected different values.

**Root cause.** Reshape correctness is verified by shape, not by index ordering. A single outlier index tuple in a group of related permutes is invisible to shape-only assertions.

**The rule.** When multiple tensors of identical shape undergo `permute()+reshape()` to the same target layout, ALL must use identical permutation indices. Before committing any attention/expert tensor reshaping code:
```
grep -n 'permute(' train_gpt.py | grep -v '#'
```
Verify all groups of related permutes use the same index tuple. A single outlier is almost certainly a bug.

**Verification recipe.**
- The grep above; manually inspect groups of related permutes for shared index tuple.
- Add a unit test that asserts the post-permute layout matches an `einsum`-derived reference tensor (not just shape).

**Cross-references.** Adjacent to the audit-checklist tensor-layout guidance (manual flatten/reshape needs einsum equivalence) — both stem from "shape is not semantics".

---

### doc-code-invariant

**Date:** Pre-archive (cited in 2026-04-15 config-drift review as one of the five drift defects)
**Rule in CLAUDE.md:** §9 row "Doc-Code Invariant"

**What happened.** `opg_doc.tex` described an algorithm variant that `train_gpt.py` no longer implemented (a removed `gg_gate`); the doc and the code drifted out of sync because nothing required them to update together. Readers using the doc as ground-truth were misled.

**Root cause.** Code is the implementation; the paper draft is a *theoretical* description. Without an explicit invariant, the two diverge silently as code evolves.

**The rule.** When `opg_doc.tex` describes an algorithm and `train_gpt.py` implements a different (better) variant, the doc MUST note the deviation in a "Practical implementation" paragraph. The pseudocode represents the theoretical formulation; the implementation note is the source of truth for code.

**Verification recipe.**
- Any PR that mutates `Block.forward`, the DEQ equation, or the promoted baseline must update `opg_doc.tex` in the same commit, OR open a `TODO(paper)` ticket noting the divergence.
- Reviewers grep `opg_doc.tex` for stale references to removed names (e.g. `gg_gate`, `SmearGate`) when those features are dropped.

**Cross-references.** [#config-drift](#config-drift) (this was one of the five defects).

---

### hyperparameter-fanout

**Date:** 2026-04-28 review of Phase 9 cleanup
**Rule in CLAUDE.md:** audit row "Hyperparameter fan-out" · source-of-truth principle

**What happened.** A multi-agent pre-commit review of the Phase 9 throughput chain found five knobs documented in CLAUDE.md §5 (Routing & Expert Ranks table) as `Hyperparameters` fields that were ACTUALLY hardcoded as constructor literals deep inside `SoftDenseRouter` / `Block` / `MoSHead` / `_parcae_init_raw_values`:

- `min_share_loss_weight = 1.0` (in `SoftDenseRouter.__init__`) and a SECOND override `0.0` in `Block.__init__` — three sources of truth, none of them `Hyperparameters`.
- `cv_loss_weight = 0.10` (in `SoftDenseRouter.__init__`) and `2.0` in `Block.__init__`.
- `router_entropy_coef` was a `Hyperparameters` field but reached the consumer through several `getattr(args, ...)` calls — every layer added a hardcoded fallback default.
- `router_entropy_warmup_delay_frac` — same pattern.
- `parcae_init_b_bar` — not a `Hyperparameters` field at all, only derived as `1 − parcae_init_a_bar` inside `_parcae_init_raw_values`.

A user attempting to override these via CLI saw no effect: the override either never reached the consumer or was shadowed by a function-default literal. The doc said "tunable", but the code was not.

In the same review, the per-token entropy loss term was found to be silently scaled by `(attn_balance_mult + mlp_balance_mult)` because the pooled router is summed across attn+mlp slices via `id()` dedup in `_collect_routing_losses`. The documented `router_entropy_coef = 0.005` was acting as `~0.03` in effect — the doc and the running model disagreed on the loss landscape.

**Root cause.** When a knob lives in `Hyperparameters` and is also a constructor argument with a default, future hands edit the constructor default and forget the dataclass — or vice versa. The doc says one thing, the code does another, neither catches it.

**The rule (Hyperparameter fan-out invariant).** Any tunable knob with a documented effect on a metric (val_bpb, throughput, an audit invariant) MUST live in `Hyperparameters` and be reachable through `_parse_cli_overrides`. The four-touch checklist for adding a knob:

1. Field in `Hyperparameters` with the documented default.
2. `args.<field>` read at the consumer site — NO constructor literal default that shadows it.
3. Documentation mirror row or note when the knob belongs in [`#current-architecture-reference`](#current-architecture-reference).
4. Row in `opg_doc.tex` parameter table (or a "Practical implementation" deviation note per the doc-code-invariant rule).

When effective magnitude differs from documented magnitude (as with the entropy loss × balance-mult dedup), document the *effective* value in the config mirror OR fix the multiplication so documented = effective. Do not silently leave readers with the wrong mental model.

**Verification recipe.**
- Pre-commit grep: every documented config-mirror row name must match `args.<row>` somewhere in `train_gpt.py`.
- `experiments/test_cli_parser.py::test_default_parity` iterates the documented routing knobs and asserts each is reachable through `_parse_cli_overrides` with the documented default.
- Reviewers reading a PR that adds a knob should grep `Hyperparameters`, `_parse_cli_overrides`, and the config mirror for the new field name BEFORE approving.

**Cross-references.** [#config-drift](#config-drift) (parent pattern), [#doc-code-invariant](#doc-code-invariant) (the doc-side enforcement).

---

### claude-md-size-budget

**Date:** 2026-04-30 review
**Rule in CLAUDE.md:** §9 audit checklist row "CLAUDE.md size budget"

**What happened.** CLAUDE.md grew to 51 887 chars and triggered Claude Code's "Large CLAUDE.md will impact performance (>40 000 chars)" warning. Almost every knob in §5 had accreted a multi-line iter-history annotation ("iter 96 baseline; iter 97 attempt NOT PROMOTED on per-wallclock grounds, see H72…"); §6.3 carried a paragraph-long postmortem of the bottleneck-experts approach; §7 metrics-table prose duplicated definitions already living in `experiments/docs/hypotheses.md`.

**Root cause.** Promotion etiquette put a "why this knob has its current value" annotation on the knob itself. Each annotation was reasonable in isolation; together they coupled a stable rule file (CLAUDE.md) to an unstable narrative file (`hypotheses.md`). Stories about *past* iterations don't compose with rules about the *current* state — they only accumulate.

**The rule.**

> `wc -c CLAUDE.md` < 40 000. Iter-history prose ("iter X NOT PROMOTED because Y") is **content rot in CLAUDE.md** — it belongs in `experiments/docs/hypotheses.md` (per-iter narrative) or `EXPERIENCE.md` §2 (durable lessons). Before adding to CLAUDE.md, ask: "is this an *invariant* (current state) or a *story* (history)?". Invariants stay; stories go elsewhere with a one-line pointer left behind.

**Verification recipe.**
- Pre-commit: `wc -c CLAUDE.md` returns < 40 000.
- Anchor resolution: `grep -oE 'EXPERIENCE.md#[a-z-]+' CLAUDE.md | sort -u`; each anchor has a matching `### <slug>` heading in this file.
- Knob-row size cap (informal): any config-mirror annotation should fit in one line of prose; longer annotations route to `experiments/docs/hypotheses.md` H## with a short pointer left in the mirror.

**Cross-references.** [#config-drift](#config-drift), [#hyperparameter-fanout](#hyperparameter-fanout) — both share the "single source of truth" theme; this rule applies it to *narrative* drift, not numeric drift.

---

### variance-reg-ns-cascade

**Date:** 2026-04-30 review (post-iter 121b)
**Rule in CLAUDE.md:** §5 Optimizer (`muon_backend_steps` row), `feedback_diagnosis_context.md`

**What happened.** Iter 117 v2 hit a NaN cascade and an early diagnostic blamed the **Polar-Express Newton-Schulz** (PE-NS) orthogonalizer at `muon_backend_steps=10` (later 7), with the proximate symptom "entmax + entropy" instability. PE-NS was about to be reverted from the default. Re-investigation showed the actual driver was the **variance regularizer** introduced in iter 111 H83 (`routing_variance_coef = -λ · Σ_e Var_token(w(e|t))`): once entmax-1.5 produced exact-zero routing weights, the variance gradient amplified those zeros into a Newton-Schulz blow-up. Iter 117 v3 *removed* the variance reg and ran cleanly with PE-NS @ 7; iter 117 v5 promoted on that combination.

**Root cause.** A surface symptom (NS amplification of bad gradients) was treated as the cause without isolating the gradient *source*. The triggering condition (entmax + variance reg) was a regression of an unrelated commit; the orthogonalizer was a passive amplifier.

**The rule.** When closing an iter due to instability, document the **full active config** at the moment of failure — every regularizer coef, every flag, every annealing schedule — so re-opening is automatic when one of the triggering conditions is removed. Don't write "iter 121 closed: PE-NS instability" if the actual story is "iter 121 closed under (variance_coef=λ, use_entmax=True, PE-NS @ 7) — closure is conditional on the active regularization stack". See `feedback_diagnosis_context.md`.

**Verification recipe.**
- Iter-closure note must list the active regularization stack, not just the suspected component.
- Re-open the closed knob whenever a triggering condition is removed; do NOT treat closure as permanent.

**Cross-references.** [#dead-code-tracking](#dead-code-tracking) (the variance reg was eventually removed, becoming dead code that needed full purging).

### cumulative-metric-misread

**Date:** 2026-05-02 review of iter 117b-3 (sparse MoE dispatch C=8) erroneous kill.
**Rule in CLAUDE.md:** §9 audit checklist row "Cumulative-vs-instantaneous metric distinction (HARD)" + the §9 META-PRINCIPLE above the checklist.

**What happened.** At iter 117b-3 healthcheck #1 (s10 of 1000-step run, ~11 min after launch), the agent read `step_avg = 33.2s` and concluded sparse dispatch was 42% slower than dense iter 95 baseline (23.5s). Killed iter 117b-3 mid-run. Wrote a "throughput-economics" incident report claiming sparse dispatch was structurally throughput-negative, pulled iter 117b-3b out of the queue under the same theory, and pivoted to the TBPTT scaling sweep instead.

The user immediately questioned: *"Would the high step time be due to compile not yet amortized?"* They were right. Looking at per-step deltas from train_time:
- s1 → s2: 22s
- s2 → s3: 28s (K=24 sample)
- s5 → s6: 21s
- s9 → s10: 28s

Per-step deltas were 21-28s (mean ~24s) — only ~5% slower than baseline. The 33.2s `step_avg` reading was almost entirely the 111s s1 compile-init amortized over 10 steps. Over 1000 steps, the compile cost amortizes to <0.1s/step — negligible.

The claimed "throughput-economics" math was also wrong: at C=8 with E=15, sparse dispatch handles ~`C × N` tokens (= 8N for balanced routing), not `C × E × N`. That's FEWER tokens than dense's `15N` — sparse should be faster, not slower, at steady state.

iter 117b-3 was relaunched with the corrected reading; the NOT-PROMOTED documentation was reverted to KILLED-PREMATURELY-RELAUNCH-PENDING.

**Root cause.** Two layered errors:
1. **Surface error**: confused two different metrics (`step_avg = total_train_time / step` cumulative vs. `Δ_t = train_time[t] − train_time[t-1]` instantaneous). The former is sample-mean-with-outliers, the latter is per-step rate. They only agree when `t » outlier_cost / asymptotic_rate`.
2. **Deeper error**: didn't read the metric definition before drawing conclusions. `step_avg` is computed and emitted by `train_gpt.py` as cumulative (`train_time / step`). Treating its value as if it were the instantaneous step time skips the "what does this number mean" check that should precede every conclusion.

The META-failure is the deeper one. The cumulative-vs-instantaneous distinction is just one common instance.

**The rule.**

**META-PRINCIPLE: Read the definition before reading the value.** Every derived metric (running average, windowed smooth, normalized score, post-softmax probability, log-loss, etc.) is a *function* of raw signals. Before concluding anything from its value, explicitly write down:
- (a) which raw signal(s) it derives from
- (b) what transformation is applied
- (c) when the derived value is within ε of the underlying truth you actually care about

For step_avg specifically: source = `train_time` cumulative + step counter; transform = `total / N`; convergence to asymptotic rate = `t » outlier_cost / asymptotic_step ≈ 200 steps for our typical compile init`.

**Decision rule for healthchecks before s50**: ALWAYS compute and report per-step delta `Δ_t = train_time[t] − train_time[t-1]` over the last 5-10 steps. The cumulative `step_avg` is only the right metric past `t > 200` OR when there's no compile/warmup/recompile activity.

**Bayesian prior on user feedback**: when the user questions a conclusion, the prior should be that they spotted a real issue. Verify by re-deriving from raw signals BEFORE defending the original reading.

**Verification recipe.**

```bash
# Per-step throughput delta (correct):
grep -E "^step:[0-9]+/" run.log | grep -oE "step:[0-9]+/|train_time:[0-9.]+" | paste -d',' - - | awk -F'[,:]' 'NR==1{prev=$NF; next} {print $2, ($NF-prev); prev=$NF}'

# Cumulative step_avg (only valid past t>200):
grep -E "^step:[0-9]+/" run.log | grep -oE "step:[0-9]+/|step_avg:[0-9.]+" | paste -d',' - -
```

The first command gives instantaneous per-step latency. The second gives cumulative — only trust it once enough samples (≥200) have washed out the warmup outliers.

**Cross-references.** This incident generalizes [#hot-path-sync](#hot-path-sync) (also a "look at the actual cost, not the documented intent" failure) and [#diagnostic-gate-component-awareness](#diagnostic-gate-component-awareness) (also a "stale assumption applied to current state" pattern). The unifying theme: **don't trust a derived value without re-checking how it's computed**.

---

### partial-preview-completeness

**Date:** 2026-05-05 review of iter 142-refactor
**Rule in CLAUDE.md:** §9 audit checklist row · §7 step 12 (Hypothesis Log)

**What happened.** The iter-142-refactor entry in `experiments/docs/hypotheses.md` (commit `3e62655`) reported a 100-step preview verdict with components (a) roundtrip int6, (b) k_sweep_table, (c) trajectory, (d) acyclicity-prime check — but silently omitted (e) `ntp_loss` descent rate, the permanent-metric component mandated by user directive 2026-05-02. The author implicitly granted themselves an exemption because "100 steps is a partial signal" (per the entry's Caveats) and a longer run was pending. Caught during pre-commit review (this file).

**Root cause.** The five-component requirement was framed assuming a 1000-step run with the canonical descent windows (s30-s100, s100-s200, …). When an iter publishes a verdict from fewer steps, no rule said how to behave — the author silently skipped rather than emitting partial windows. Same shape will recur on every short-budget preview, K-sweep skipped on OOM, log-rotation race, etc.

**The rule.** Partial-step previews are NOT exempt from the 5-component report. Emit the partial windows that are computable (e.g. one s30–s60 row instead of the full s30→s1000 table). If a component is genuinely uncomputable from the run.log (logs rotated, OOM during K-sweep, etc.), state it explicitly under **Caveats** with a recovery plan ("iter-Xa will re-emit"). Silently omitting shifts a tracked debt into an untracked one. **Generalizes**: graceful-degradation > silent-skip for any mandated artifact (k_sweep_table on partial K-coverage, log_summary on missing fields, etc.).

**Verification recipe.** `grep -E "^### iter " experiments/docs/hypotheses.md | tail -5` then inspect the most recent entries. Each must show all 5 §7-step-12 components or an explicit Caveats note for the missing one. Pre-commit reviewers should specifically check that "longer run pending" is paired with whatever partial data IS available, not used as a wholesale exemption.

**Cross-references.** Related: [#cumulative-metric-misread](#cumulative-metric-misread) (also a "convenient simplification eats a required signal" pattern), [#diagnostic-gate-component-awareness](#diagnostic-gate-component-awareness) (also a "stale exemption survived a regime change" pattern). The unifying theme: **mandated artifacts degrade gracefully; they do not silently skip.**

---

### move-tracked-invariant

**Date:** 2026-05-06 review of components-archive move
**Rule in CLAUDE.md:** Audit Checklist row · "Move means tracked"

**What happened.** Pre-commit review of branch `autoresearch/phase2-optimization` found 9 untracked files at `experiments/components/archive/` paired with 9 staged deletions at `experiments/components/`. The archive copies were created via `cp + rm` (or `mv` outside git), leaving the destinations as untracked while the sources appeared as staged-deleted. `.gitignore` listed `experiments/archive/` — *not* `experiments/components/archive/` — so the new files were not ignored, just unstaged. A routine `git add -u` followed by commit would have shipped only the deletions, dropping the 9 archived scaffolds from history. The archive directory had previously contained only `orthogonal_expansion_routing.py` (commit `3e74393`), so there was no precedent of these files surviving a deletion-only commit.

**Root cause.** "Archive, don't delete" is a documentation pattern that depends on git tracking the destination. The pattern fails silently when the move is performed outside git: the source's deletion is staged automatically by `git add -u`, but the destination requires an explicit `git add` that no rule was forcing. The named iters in the active hypothesis queue (H91 TTT, H94 GPTQ+LQER, iter 120 RRAttention, etc.) all reference scaffolds that were about to be lost.

**The rule.** A file move from path A to path B requires `git add B` in the same commit as the deletion of A. `cp + delete` and `mv` outside git both leave B as untracked while A appears as staged-deleted; a routine `git add -u` then commits the deletion alone, dropping B from history.

**Verification recipe.**
1. After any rename / move / archive operation, run `git status --short | grep "^??"`.
2. Any untracked path inside or below the moved tree must be explicitly `git add`-ed (not relying on `git add -u` or `git add .` which can drag in unrelated cruft).
3. Prefer `git mv` for renames so both sides are staged atomically. For bulk moves, `git mv` each file then verify with `git status --short`.
4. If a directory is intended to be ignored (true scratch), add it to `.gitignore` in the same commit and document the intent.

**Cross-references.** Related: [#dead-code-tracking](#dead-code-tracking) (companion rule: removals must purge all references in the same commit), [#config-drift](#config-drift) (sibling rule: source-of-truth integrity), [#enforcement-config-staging](#enforcement-config-staging) (generalizes the same staging-omission failure mode to enforcement-side configs). The unifying theme: **partial automation is worse than none — `git add -u` is convenient until silent omissions accrue cost.**

---

### enforcement-config-staging

**Date:** 2026-05-11 review of iter146-151 rescue commit window
**Rule in CLAUDE.md:** Audit Checklist row · "Enforcement-config staging"

**What happened.** Pre-commit review of the iter146-151 rescue commit found `pytest.ini` as `??` in `git status` while a 7-test rewrite (`return failures` → `assert failures == 0`) was already staged. The new `pytest.ini` was the *reason* for the rewrite: `filterwarnings = error::pytest.PytestReturnNotNoneWarning` flips pytest's previously-warning return-non-None behavior into a hard test failure. With the test rewrite staged but `pytest.ini` unstaged, the gate would have been silently inactive on the next contributor's machine and on CI restart — and any future contributor writing `return failures` in a new test function would not see the failure. The rewrite would have looked successful but the enforcement would not be active.

**Root cause.** The existing *Move means tracked* rule covers source-file *relocations* — `mv outside git`, `cp + rm`, archive-style moves — where the destination needs an explicit `git add`. It does NOT cover *new* enforcement-config files (`pytest.ini`, `pyproject.toml [tool.X]` sections, `.pre-commit-config.yaml`, ruff/mypy rule files, `tool.coverage` thresholds) that are created alongside the code they gate. The failure mode is identical (staged code change relies on an unstaged sibling file to behave correctly) but the trigger is *creation* rather than *relocation*, so the move-tracked verification recipe (`git status --short | grep "^??"` inside the moved tree) didn't surface it because there was no "moved tree" to scope the grep to.

**The rule.** Any new project-level config that *enforces* an invariant — `pytest.ini`, `pyproject.toml` lint/format/test sections, `.pre-commit-config.yaml`, ruff/mypy/pyright rule files, `tool.coverage` thresholds, `.editorconfig`, dependency-pinning manifests — MUST be staged in the same commit as the code change it enforces. If the config file appears in `git status` as `??` while the code it gates is staged, the invariant is silently inactive on the next contributor's machine and on CI restart.

**Verification recipe.**
1. Before staging, run `git status --short | grep "^??"`. Inspect every untracked path.
2. For any untracked path that is a config file (filename matches `*.ini`, `*.toml`, `*.cfg`, `*.yaml` at project root or inside `.config/` / `.github/`, or matches `pyproject.toml`, `setup.cfg`, `.pre-commit-config.yaml`, `pytest.ini`, `tox.ini`, `ruff.toml`, `mypy.ini`, `pyrightconfig.json`, `.editorconfig`, `requirements*.txt`, `uv.lock`), ask: *does the staged code change depend on this config to behave correctly?* If yes → `git add` it in the same commit.
3. For an enforcement-config that is intentionally machine-local (developer-only, e.g. a private `.editorconfig` extension), add it to `.gitignore` *in the same commit* and document why.
4. The verification step is one shell command; no rule should be allowed to add review burden without an automated check.

**Cross-references.** Generalizes [#move-tracked-invariant](#move-tracked-invariant) (same failure class: staged change depends on unstaged sibling) and [#dead-code-tracking](#dead-code-tracking) (companion: removals must purge references). Companion to [#untested-path-executability](#untested-path-executability): a gate with no executable enforcement is dead code; an enforcement-config that isn't staged is the same failure expressed at the config-staging layer. Generalized further by [#audit-row-executability](#audit-row-executability), which requires every new audit row to ship with an executable witness rather than a manual recipe.

---

### audit-row-executability

**Date:** 2026-05-11 three-reviewer audit of iter146-151 cleanup commit
**Rule in CLAUDE.md:** Audit Checklist row · "Audit-row executability"

**What happened.** A three-agent pre-commit review (coderabbit + pr-review-toolkit + superpowers) of the iter146-151 cleanup commit found two related failures, both downstream of the same meta-pattern:

1. The **Sibling-fanout DRY gate** audit row added in `5ecf324` (one commit earlier) was first violated by the *next* commit: six new EMA share-log fields (`attn_ema_min`, `mlp_ema_min`, `pool_ema_min`, `attn_ema_cv`, `mlp_ema_cv`, `pool_ema_cv`) were emitted from `train_gpt.py` but had no parser entries in `experiments/plot_metrics.py`. The row's own definition (3 siblings × 3 sites) was satisfied — trainer + log + plot — and yet the new siblings shipped without the plot-parser leg. The row had no executable witness: just CLAUDE.md prose plus an `EXPERIENCE.md` recipe asking the reviewer to grep.

2. The **Enforcement-config staging** audit row added in the *same* iter146-151 cleanup commit codified itself as a manual recipe (`git status --short | grep "^??"`). Its own closing clause said "no rule should be allowed to add review burden without an automated check," yet the rule shipped exactly that review burden. The same review window flagged both — making the pattern unmistakable.

**Root cause.** Audit rules in `CLAUDE.md` are durable; verification recipes expressed as shell snippets or prose checklists are reviewer-time procedures. Recipes degrade in two predictable ways: (a) the next contributor doesn't run them, (b) the next reviewer doesn't know to look for them. A test, hook, or CI gate is the only enforcement that survives staffing turnover. The pattern is general: every new manually-verified rule is one commit away from its first silent violation, often by the diff that creates it.

**The rule.** Every new audit-checklist row added to `CLAUDE.md` MUST ship with at least one executable witness in the same commit:

1. **A unit test** (`tests/test_*.py` or `experiments/test_*.py`) that asserts the invariant the rule names. Example: `tests/test_enforcement_config_staged.py` asserts `pytest.ini` is git-tracked when it exists on disk.
2. **A pre-commit hook entry** in `.pre-commit-config.yaml` if the rule is about staging hygiene rather than code semantics.
3. **A CI gate** if the invariant can only be observed at build time (build size, artifact size, smoke `val_bpb`).

A manual "verification recipe" is acceptable *only as supporting documentation* of how the executable witness works, not as a substitute for one.

**Verification recipe (for adding the next rule).**
1. Identify the invariant the new rule asserts.
2. Search `tests/`, `experiments/test_*.py`, and hook configs: does any existing test assert it? If yes, link from the rule body ("Enforced by …").
3. If no, write one and stage it in the same commit. A 15-line test that runs in <1 s is cheap and stays paid back forever.
4. Cross-check: is the new rule itself enforced by something? If the answer is "the reviewer reads CLAUDE.md" — that is by definition manual; promote it to executable form.

**Cross-references.** Companion to [#untested-path-executability](#untested-path-executability) (same principle at the code-branch layer), [#enforcement-config-staging](#enforcement-config-staging) (same principle at the config-staging layer), and [#move-tracked-invariant](#move-tracked-invariant) (same principle at the file-relocation layer). The four rules together express one underlying invariant: **every claim of correctness must have a runnable artifact that asserts it; prose-only rules accrue silent violations.**

---

### audit-row-self-enforcement

**Date:** 2026-05-12 five-reviewer pre-commit audit of the iter152 profile-system + capability-registry diff
**Rule in CLAUDE.md:** Audit Checklist row · "Audit-row self-enforcement"

**What happened.** The diff that added the new "Root-cause fix preference" audit row also added a new `mos_ortho` branch to `_prescribe_failure_fix` (`train_gpt.py:6383-6392`). The new row's policy is explicit: "Prefer architecture/parameterization, then shared controllers, then invariant regularizers; use metric-specific losses only as temporary ablations with an exit plan." The non-MoS `ortho` branch in the same function correctly emitted `{"needs_expert_bank_geometry_constraint": True, ...}` and framed `expert_output_diversity_coef_mult` as "only as a temporary ablation." The MoS branch — sibling code, same function, same diff — returned `{"mos_output_diversity_coef": 0.05}` and the `fix` text contained no exit-ablation framing. The motivating example for the new rule was honored; the sibling instance the rule also applied to was not.

A five-agent parallel review (coderabbit + pr-review-toolkit + feature-dev high-bar + silent-failure-hunter + pr-test-analyzer) converged on this finding independently. The Audit-row-executability witness for "Root-cause fix preference" covered only two of the eight `_prescribe_failure_fix` branches (`min_share` and `lip_ub`), so neither CI nor a pre-commit hook would have caught the omission.

**Root cause.** [`#audit-row-executability`](#audit-row-executability) requires a new rule to ship with a witness, but does not require the witness to *iterate over the rule's full scope*. A single-instance witness is satisfied by the motivating case alone. Sibling instances — whose existence is what makes the rule worth writing — silently slip through. Three previous incidents share this signature:

- iter146 EMA log fields landed without parser entries (witness covered only the motivating field).
- The `5ecf324` Sibling-fanout DRY gate row landed without iterating over every plot/log/parser triple.
- The current diff: a per-symptom loss bump landed inside the very commit that codified the policy against per-symptom losses.

The general pattern: rules name *classes* of sites (every loss branch, every `use_X` flag, every `setattr` loop), and the witness must enumerate the class, not pick a representative.

**The rule.** Every new audit-checklist row, contract paragraph, or policy section MUST be accompanied in the same commit by:

1. **A scope-enumerating witness.** The executable test iterates over the *full set* of sites the rule applies to and asserts the invariant on each. Examples: `experiments/test_arch.py::test_prescriptions_route_to_invariant_mechanisms_not_per_symptom_losses` iterates every `_prescribe_failure_fix` failure-class; `tests/test_optional_component_flag_contract.py::test_each_flag_has_effect_or_explicit_reject` iterates every `_OPTIONAL_COMPONENT_FLAGS` entry; `tests/test_hyperparameter_validation.py::test_hyperparameter_fields_covers_score_iter152_profile` iterates every `_CONFIG_PROFILES` profile.
2. **A same-commit compliance sweep.** Before the commit lands, every existing sibling that the rule applies to must already be compliant. The witness catches future violations; the manual sweep catches the violations the rule itself introduces.

A "representative case" witness is acceptable only if the rule's scope is provably singleton (e.g. a rule about a unique class).

**Verification recipe (for adding the next rule).**
1. State the rule's scope explicitly in the row body: "applies to every `_prescribe_failure_fix` branch", "applies to every `use_X` Hyperparameter", "applies to every `setattr` on `args`", etc.
2. Grep the codebase for the scope predicate (e.g. `grep -n "return {" train_gpt.py | grep "_prescribe_"`).
3. Confirm every grep hit is compliant; bring any non-compliant site into compliance in the same commit.
4. Add or extend a test that iterates over the same scope predicate and asserts the invariant. The test must fail before the compliance sweep and pass after it.

**Cross-references.** Companion to [#audit-row-executability](#audit-row-executability) (that rule says "ship a witness"; this one says "the witness must cover every sibling"). Same underlying invariant as [#sibling-fanout-dry-gate](#sibling-fanout-dry-gate) and [#untested-path-executability](#untested-path-executability): **every claim of policy compliance must enumerate the policy's scope; representative-case enforcement decays to motivating-case enforcement the moment a sibling is added.**

---

### loss-form-triple-touch

**Date:** 2026-05-06 review of iter 142b cv² promotion
**Rule in CLAUDE.md:** Audit Checklist row · "Loss-form changes are triple-touch"

**What happened.** Iter 142b replaced the routing-balance loss form `relu(cv − cv_target)²` with continuous `cv²` (and removed the `cv_target` / `mos_cv_target` knobs entirely). The implementation in `train_gpt.py::SoftDenseRouter._update_health` and `MoSHead._head_forward` was correct, but a pre-commit review found that several adjacent surfaces still described the old form: the `_collect_routing_losses` docstring still said `Σ_r cv_hinge(r)` and `mos_cv_hinge`; `opg_doc.tex` parameter table still listed `cv_target = 0.20`, `mos_cv_target = 0.20` and the old coef magnitudes; and `opg_doc.tex` loss equation still wrote `λ_rcv·Σ[CV(m_r) − τ_cv]_+²`. Stale diagnostic prescriptions in `_prescribe_failure_fix` also cited old default values (`0.5→0.75`, `0.25→0.375`, `0.1→0.15`), which would have been emitted into `run.log` and persisted into `meta_json["failure_categories"]` — surfacing a wrong recovery recipe to anyone reading a future failure.

**Root cause.** The "Four-touch new knobs" hyperparameter rule covers magnitude tuning (`Hyperparameters` field, CLI parser, consumer reads `args`, doc mirror). It does NOT cover *structural* changes to a loss term — replacing a hinge with a quadratic, swapping a sum for a mean, dropping a target threshold. Structural changes leave the field name unchanged so the magnitude-touch checklist passes; meanwhile the formula-bearing surfaces (loss-assembly docstring, paper-facing equation, prescription strings that name old defaults) silently drift.

**The rule.** When the *functional form* of a loss term changes (not just its coefficient — e.g., hinge → quadratic, sum → mean, L2 → L1, removing or adding a target threshold), the change must touch in the same commit:
1. Implementation in `train_gpt.py`.
2. The assembly docstring describing the loss (`_collect_routing_losses` doc comment, the `Hyperparameters` formula comment block) and any `_prescribe_failure_fix` prescription strings that cite default magnitudes.
3. `opg_doc.tex` paper-facing equation **and** parameter table.

Magnitude-only edits keep using the existing four-touch hyperparameter rule. Structural edits use this triple-touch rule.

**Verification recipe.**
1. After any structural loss-form change, list the old-form tokens (e.g., `cv_target`, `cv_hinge`, `relu(cv`, the dropped knob name) and grep them across the three surfaces:
   ```bash
   grep -nE "cv_target|cv_hinge|relu\(cv" train_gpt.py opg_doc.tex CLAUDE.md
   ```
2. Zero matches outside an explicit deviation note (e.g., a row in `experiments/docs/hypotheses.md` documenting why the equation surface intentionally lags) is the post-condition.
3. If `_prescribe_failure_fix` cites default magnitudes inline, refactor to read defaults dynamically from `Hyperparameters` (drift-proof: `f"{Hyperparameters.foo:g}→{Hyperparameters.foo * mult:g}"`).
4. Add or rename a focused test that exercises the new form's distinguishing property (e.g., `test_cv_squared_has_gradient_below_old_target` — the old hinge was silent in this regime; the new form is not).

**Cross-references.** Related: [#hyperparameter-fanout](#hyperparameter-fanout) (sibling rule for magnitude-only changes), [#doc-code-invariant](#doc-code-invariant) (parent principle: paper-facing pseudocode must track implementation), [#diagnostic-gate-component-awareness](#diagnostic-gate-component-awareness) (companion: prescriptions must reflect current defaults, not historical ones), [#strict-generalization](#strict-generalization) (form changes are usually NOT strict generalizations — promotion gating must use the standard `val_bpb` rule, not the auto-promote shortcut, and the deviation must be recorded in `experiments/docs/hypotheses.md`).

### untested-path-executability

**Date:** 2026-05-08 pre-commit review of iter145r promotion + iter103 chained scaffold
**Rule in CLAUDE.md:** Audit Checklist row · "Untested-path executability"

**What happened.** Two new branches landed on `autoresearch/phase2-optimization` that were structurally correct under static reading but had never been exercised end-to-end:

1. The new resume path (`--resume-from`/`--resume-latest`) restored `step` from `ckpt.get("step", step)`, then a flat `step = 0` further down the function unconditionally erased the resumed value. Anyone calling resume would silently restart the loop counter from 0, double-count gradient updates, replay the LR/regularizer warmup, and contaminate `training_time_ms`. A second bug in the same block — `step` was used as a fallback name before any prior assignment — would have raised `NameError` if the checkpoint key was missing.
2. The chained-routing `Block` constructor set stage-0 backward-compat aliases `self.attn = chained_stack.attns[0]`, `self.mlp = ...`, `self.router = ...`. Today's hot-path consumers all early-return on `chained_stack is not None`, but the alias pattern was a future-proof landmine: a new diagnostic that forgot the guard would silently see only stage-0 and report on it.

**Root cause.** Static review of an "obvious" branch is a different signal than a runtime witness. A branch that compiles and reads correctly is not the same thing as a branch that runs. The resume reset survived precisely because nobody had run a `--resume-latest` smoke after the rest of the resume infrastructure landed.

**The rule.** Every new top-level control-flow branch — resume path, preset, scoring mode, optimizer group, alias surface — needs at least one of (a) a focused unit test that visits the branch, (b) a smoke run whose log shows the branch fired, or (c) an explicit "manually verified at <commit-sha>" note in the PR description. A branch with no executable witness is dead code: either delete it or add the test that proves it works. If the branch can only be witnessed under DDP / multi-GPU / hardware-specific conditions, document the manual recipe in the PR.

**Verification recipe.**
1. Diff for new `if`/`elif`/`match` branches and new attribute aliases.
2. For each, find the call site in tests or smoke logs. If there is no call site, write the smallest possible test that reaches it (CPU-only is fine for routing; DDP-only paths get a documented manual recipe).
3. The resume path specifically: `--checkpoint-every=10 --iterations=25` then `--resume-latest=1 --iterations=25` and assert the second run reports the resumed step in the log.

**Cross-references.** Related: [#dead-code-tracking](#dead-code-tracking) (companion principle: branches without consumers are dead), [#hypothesis-log-detail](#hypothesis-log-detail) (failed branches should be recorded as Caveats, not silent skips).

### sibling-fanout-dry-gate

**Date:** 2026-05-08 pre-commit review of iter145 EMA-anchored loss family
**Rule in CLAUDE.md:** Audit Checklist row · "Sibling-fanout DRY gate"

**What happened.** iter145 introduced three sibling EMA-anchored routing-loss terms (`alive`, `balance`, `specialization`) added to the existing CV/entropy/MoS-CV stack. The implementation hand-rolled the triplet at nine separate sites: `Hyperparameters` defaults; `SoftDenseRouter.__init__` zero-init; `SoftDenseRouter.forward` else-branch zero-fill; `GPT.__init__` coef field + target store + `_loss_t` cache + `_coef_eff_t` cache; `_collect_routing_losses` accumulators; the annealer; the per-step log f-string; and `experiments/plot_metrics.py` parser/spec list. Adding a future strict-alive-hinge fourth term would require nine independent edits without the registry, each of which is a place to silently drift in sign or magnitude. The Research-Protocol "DRY and orthogonal functions" bullet had been advisory, not a hard audit gate, and an advisory rule does not survive a multi-site fanout.

**Root cause.** Mechanical parallelism between siblings looks "explicit" line-by-line and is easy to write, but every site is a separate place to forget. The `_collect_routing_losses` site additionally separated the sign of the `specialization` term (`-` operator at the call line) from the formula (`KL(token || ema)` at the definition line), so a future reviewer could not tell from the call site whether the sign was a typo or intentional.

**The rule.** When a feature introduces three or more parallel siblings AND each sibling repeats across three or more code sites, the implementation MUST replace the boilerplate with a single registry constant (tuple/dataclass list) plus iteration. The (N+1)th sibling addition has to be a one-line registry change, not a multi-site grep-and-paste. Plot/log/parser layers must read from the same registry as the trainer; if cross-process import is impractical (analysis env without GPU/torch), the secondary site declares a mirror list with a `NOTE: must match <registry>` comment so drift is loud. If the siblings genuinely have different shapes that defeat iteration, justify the exception in the commit message.

**Verification recipe.**
1. Identify the parallel triplet (or larger) in the diff.
2. Count call sites per sibling. If ≥3 sites × ≥3 siblings, refactor to registry + loop.
3. Move sign / direction / magnitude data into the registry tuple. The call site reads from the registry, never hard-codes a sign.
4. After refactor, adding a fake fourth member should compile, run tests, and the trainer should emit a fourth log column without further edits to the trainer code.

**Cross-references.** Companion: [#dry-function-orthogonality](#dry-function-orthogonality) (parent principle), [#hyperparameter-fanout](#hyperparameter-fanout) (similar rule for hyperparameter knobs), [#loss-form-triple-touch](#loss-form-triple-touch) (related: cross-surface drift after structural loss changes).

### promotion-propagation

**Date:** 2026-05-08 review of iter145r promotion (Dirichlet-UCB + EMA-anchored + warmup + sigmoid-gate-off)
**Rule in CLAUDE.md:** Audit Checklist row · "Promotion propagation"

**What happened.** When iter145r was promoted into `train_gpt.py::Hyperparameters` defaults (router_load_cv_coef 1.0→0.0, router_ema_balance_coef 0.0→0.15, router_ema_specialization_coef 0.0→0.1, mos_load_cv_coef 1.0→0.15, expert_output_diversity_coef 1.0→0.15, regularizer_warmup_frac 0.0→0.07, use_router_sigmoid_gate True→False, weight_decay 0.01→0.015, router_scoring linear→dirichlet_ucb, router_dirichlet_ucb_beta 0.0→0.5), the promotion was incomplete in three places:

1. `experiments/test_arch.py` had a regression test that hard-coded the prior values; it would have failed on first run.
2. `GPT.__init__`, `Block.__init__`, and `SoftDenseRouter.__init__` signature defaults still mirrored the prior iter142b values. In production this is hidden because `args.<field>` is always passed; in tests, `_make_model(**defaults)` constructs `GPT()` without overriding routing fields, so every architectural test was silently running on iter142b values, NOT iter145r values. The promotion was effectively untested.
3. `opg_doc.tex` defaults table and §router/§loss subsections still described the linear-scorer + softmax-times-sigmoid + CV-as-balance world. The paper, as published from this branch, would misrepresent the model.

**Root cause.** The promotion changed `Hyperparameters` and CLAUDE.md "Current Architecture" — the two surfaces the user thinks of as authoritative — but the promotion's effective surface is wider: every place that mirrors a default. CLAUDE.md's existing "Single source of truth" rule names `Hyperparameters` as authoritative, but a `__init__` signature default that drifts from `Hyperparameters` is silent because no test used to compare them.

**The rule.** A promotion commit must touch every surface that mirrors a `Hyperparameters` value:
1. `Hyperparameters` defaults — primary.
2. Every `__init__` signature default in `train_gpt.py` (`GPT`, `Block`, `SharedBlock`, `SoftDenseRouter`, `MoSHead`, ...) for any field that the production CLI passes through. If a sub-module doesn't take a field, no change; if it does, the default must match `Hyperparameters`.
3. Every test that hard-codes the prior value, especially regression-guard asserts. Renaming the test is appropriate when the rationale changes.
4. CLAUDE.md "Current Architecture" + any §-Architecture-Principles bullet that names a default magnitude.
5. `opg_doc.tex` defaults table; if the loss form or router form changes, also the relevant subsection — or an explicit "Implementation deviates from §X — paper update queued for iter<N+1>" deviation note in the same subsection.
6. `experiments/docs/hypotheses.md` queue header + the iter row's verdict (PROMOTED / PROMOTED_WITH_TECH_DEBT) + the "active config" recipe block.

If any of (2)-(5) is intentionally deferred, the deferral must be explicit in the commit message AND the paper must carry a deviation note (not a silent stale section).

**Verification recipe.**
1. After updating `Hyperparameters`, grep each new value against signature defaults: `grep -nE "field_name: (float|bool|str|int) = " train_gpt.py` and confirm every match equals the new `Hyperparameters` value.
2. Run the full test suite. Any test that fails on the new defaults either (a) is a regression-guard that needs renaming + new asserts, or (b) is a real signature-default-drift bug found by the test.
3. Grep `opg_doc.tex` for the field name and old value; the value must match (or carry a deviation note).
4. Grep `CLAUDE.md` for the field name; "Current Architecture" must reflect the new value.
5. The promotion test in `experiments/test_arch.py::test_routing_regularizer_coefficients_match_promoted_defaults` asserts `Hyperparameters` AND a constructed-model attribute, so signature drift (item 2) cannot recur silently.

**Cross-references.** Companion: [#hyperparameter-fanout](#hyperparameter-fanout) (single-source-of-truth principle), [#loss-form-triple-touch](#loss-form-triple-touch) (paper-side rule when loss form changes during promotion), [#untested-path-executability](#untested-path-executability) (a promoted but-untested path is the same failure mode at the architecture layer).

---

### loss-gate-quantity-alignment

**Date:** 2026-05-09 pre-commit review of iter146 rescue stack (Lyapunov penalty + `lip_ub` gate)
**Rule in CLAUDE.md:** Audit Checklist row · "Loss-quantity / gate-quantity alignment"

**What happened.** iter146 introduced a "finite-perturbation Lyapunov penalty" using a unit-RMS random direction `u` and `expansion = ‖T(z + ε·u) − T(z)‖_RMS / ε`. In high-D this is a Hutchinson-style estimator with expectation `‖J‖_F / √D`, NOT the operator norm `‖J‖_2` that the post-final `lip_ub` gate measures via power iteration. With `lyapunov_gamma=0.97` the penalty fires only when `‖J‖_F > 0.97 · √D ≈ 26.9`, which permits operator norms vastly above 1 — i.e. the penalty does not directly enforce contraction even though both surfaces shared the colloquial name "Lyapunov / finite-expansion / contraction". The first-pass code reviewer accepted the iter146 diff as "READY TO COMMIT" because penalty-name and gate-name matched. A second-pass deep review caught the mismatch by deriving the Hutchinson expectation from the normalization choice. (Cosmetic L2-vs-RMS swap does NOT fix this — both have the same `‖J‖_F/√D` expectation; only power iteration or many-direction max would estimate the operator norm.)

**Root cause.** Penalty and gate were named after the same physical quantity ("contraction of T_theta") but implemented different mathematical estimators. The "Loss-form triple-touch" rule guards a *single-side* form change; this incident is the cross-side analogue: when *two separately implemented* surfaces both claim to constrain quantity X, the implementations themselves must agree on what X is, formula by formula. Naming alone is insufficient; reviewer intuition trained on penalty-form changes did not transfer to penalty/gate-form mismatches.

**The rule.** When a training-time penalty and a promotion-gate diagnostic both claim to constrain the same physical quantity (spectral radius, operator norm, expert orthogonality, expert min-share, output-cosine pair statistic, etc.), the penalty implementation, the gate implementation, the `Hyperparameters` formula comment, and `opg_doc.tex` must name the *same mathematical object* by formula (norm choice, reduction choice, normalization choice, gating choice). Each side carries a forward cross-reference to the other side's anchor in code comments. A change to one side proposes a corresponding change to the other; an intentional asymmetry (e.g. soft Frobenius proxy as cheap penalty vs strict operator-norm gate) must be called out explicitly in both comments AND in `opg_doc.tex`.

**Verification recipe.**
1. For each promotion-gate diagnostic, grep its name in `train_gpt.py` to find the matching training penalty (or confirm none exists).
2. For each found pair, confirm the penalty docstring and the gate docstring use the same formula notation (e.g. both `‖J‖_2` or both `‖J‖_F/√D`, NOT one of each).
3. If the formulas intentionally differ, both docstrings must say "soft proxy for X (see `<other_anchor>` for tight cert)" or symmetric language — never just X on both sides.
4. Add a contract test that grep-asserts the cross-references survive future edits: penalty's docstring must mention the gate's identifier, and vice versa.

**Companion meta-lesson.** A first-pass code review that says "READY TO COMMIT" on a 793-line cross-cutting diff without finding any IMPORTANT issues is suspect; default to a second-pass deep review when the diff (a) exceeds ~500 lines, (b) changes loss form and diagnostic surface together, or (c) touches both a penalty and the gate it claims to satisfy.

**Cross-references.** Companion: [#loss-form-triple-touch](#loss-form-triple-touch) (single-side form-change rule), [#diagnostics](#diagnostics) (diagnostic-metric contract), [#diagnostic-gate-component-awareness](#diagnostic-gate-component-awareness) (gate emission rules).

### scalar-semantic-shift

**Date:** 2026-05-09 pre-commit review of iter146 rescue stack
**Rule in CLAUDE.md:** Audit Checklist row · "Scalar-semantic shift triple-touch"

**What happened.** The iter146 rescue stack refactored the budget timer: a new helper `_compute_training_budget_ms(max_training_seconds)` took only one argument and dropped the `eval_reservation_seconds` subtraction that the previous inline computation performed. The Hyperparameters comment was simultaneously rewritten to declare *"Process wallclock = max_training_seconds + eval_reservation_seconds"* — flipping the meaning of `max_training_seconds` from a process-total budget to a training-only budget. **Every existing caller, including the canonical submission command in CLAUDE.md "Run Commands" (`--max-training-seconds=600`), continued to pass the same numeric value.** Under the new semantic that command would run for `600 s training + 120 s post-loop = 720 s` total wallclock — a silent 20% violation of the project invariant *"submission training must fit the 600 s 8×H100 budget"*. Compounded by `final_full_validation=True` becoming default-on, the post-loop work expanded beyond the historical 120 s carve-out, so even reverting the bare meaning would not have been enough — the eval reservation needed re-profiling.

The mistake passed every existing audit gate: the new helper had a (trivial) test (`assertEqual(_compute_training_budget_ms(600), 600_000.0)`); promotion propagation was satisfied because no *default value* changed; type signatures, units in their colloquial names ("seconds"), and CLI plumbing were all consistent. What changed silently was the *meaning* of the scalar — a class of drift the existing rules did not name.

**Root cause.** The promotion-propagation rule covers default-value drift, the loss-form triple-touch rule covers single-side form changes, and the loss/gate-quantity alignment rule covers cross-side mathematical mismatches. None of them covers the case where the implementation, the Hyperparameters comment, and the test all agree on a *new* meaning while every external caller silently keeps the *old* meaning. The cost is invisible until a 600-second hardware constraint is breached on the actual submission run.

**The rule.** When the *meaning* of an existing scalar contract changes — units (seconds vs ms vs steps), ownership (training-only vs process-total), inclusion or exclusion of a previously-bundled term, nullability of a metadata field, or any unit-level invariant — the same commit must:

1. **Implementation site.** Update the function body and signature; if the new meaning is sufficiently different, prefer renaming the symbol (`max_training_seconds → max_process_seconds`) over silently re-meaning it.
2. **Every caller of the scalar.** Sweep `CLAUDE.md` "Run Commands" snippets, `records/` submission scripts, `update_results.sh` consumers, `tests/test_training_contracts.py` schema asserts, and any historical command captured in iteration entries. If a caller's intent shifts, change the caller's value.
3. **Every doc surface that names the unit.** `Hyperparameters` field comment, `opg_doc.tex` parameter table, EXPERIENCE.md runbook references, and CLAUDE.md "Project Invariants" if the constraint is project-level.
4. **A focused numeric test pinning the new semantic against the project invariant.** For the budget case: assert `_compute_training_budget_ms(600, eval_reservation=120) ≤ (600 − 120) · 1000`. Renaming the field to `max_process_seconds` would make the test self-evidently correct without the need for the assert.
5. **If a rename is too disruptive,** the field comment must declare the superseded semantic *adversarially*, e.g. *"NOTE: prior semantic was X; current semantic is Y; existing scripts that passed `--name=N` now produce Z and must be updated to N′."* — phrased as a warning to the reader, not a quiet rationale.

**Verification recipe.**
1. `grep -rn '<scalar-name>' .` — verify every match either was edited in this commit or its surrounding intent matches the new semantic.
2. Confirm the Hyperparameters comment, `opg_doc.tex` table, and CLAUDE.md "Project Invariants" all describe the same meaning.
3. Run the focused numeric test that pins the new semantic against the project invariant.
4. If a default-on safety knob (`final_full_validation`) compounds the shift, run a smoke that *measures* total process wallclock against the project invariant on the dev profile — not just total training-loop seconds.

**Cross-references.** Companion: [#promotion-propagation](#promotion-propagation) (default-value drift), [#loss-form-triple-touch](#loss-form-triple-touch) (single-side form change), [#loss-gate-quantity-alignment](#loss-gate-quantity-alignment) (cross-side mathematical mismatch), [#config-drift](#config-drift) (single-source-of-truth for tunables), [#cumulative-metric-misread](#cumulative-metric-misread) (semantic vs. instantaneous interpretation of the same scalar).

---

### flag-to-effect-contract

**Date:** 2026-05-12 pre-commit review of iter152 promotion + remaining-queue activation
**Rule in CLAUDE.md:** Audit Checklist row · "Flag-to-effect contract"

**What happened.** The iter152 commit un-archived 7 default-off components and wired 8 new `use_*` flags (`use_smear_gate`, `use_sparse_attn_head_gate`, `use_rr_attention`, `use_ttt_eval`, `use_gptq`, `use_lqer`, `use_grouped_artifact_compression`, `use_caseops`) plus `deq_prefix_anchors`. Pre-commit review using two parallel reviewer subagents found that four of these flags were silent no-ops at the project default `train_seq_len=2048`:

- `use_rr_attention=True` silently fell through to `F.scaled_dot_product_attention` because `rr_attention.py` gates the masked path on `T <= _RR_MAX_TOKEN_MASK_TOKENS=512` and the project default `train_seq_len=2048` exceeds it. No warning, no metric, but the startup banner printed `rr_attention=1`.
- `use_gptq=True` / `use_lqer=True` only invoked `run_gptq_lqer_component_smoke` on a synthetic 16×24 matrix. The scored int6 artifact path (`save_int6_artifact → mixed_quantize_int6`) was unchanged, but the banner advertised `gptq=1 lqer=1`.
- `use_ttt_eval=True` had zero consumers in `train_gpt.py` (`grep is_ttt_eval_enabled` returned no hits outside imports).
- `use_caseops=True` ran a hardcoded `caseops_smoke_text` fixture print and never touched the pre-tokenized FineWeb shards.

Existing tests partially covered each component module (helper-level smokes), but no test asserted that flipping `use_X=True` produced an *observable* difference in the training-path forward output relative to the default-off baseline. The earlier audit-row-executability rule had been satisfied at the per-module level while silently failing at the integration level.

**Root cause.** Optional-component flags occupy a special place in the research workflow: their value is precisely the ability to compare `use_X=True` vs `use_X=False` runs in the hypothesis log. A flag that prints `X=1` in the startup banner but produces baseline behaviour corrupts the *evidence base*, not just runtime behaviour. The prior audit gates (`promotion-propagation`, `loss-form-triple-touch`, `scalar-semantic-shift`) covered scalars whose *values* shifted silently; they did not cover bool flags whose *effect* was silently absent. This is the bool-flag analogue of the `scalar-semantic-shift` rule.

**The rule.** Every new `use_X` Hyperparameter MUST have at least one **observable, asserted effect on the training-path tensor flow OR on the scored artifact** at the project's default `train_seq_len`, exercised by a test that flips the flag True and asserts the output **differs** from the disabled baseline (within a fixed tolerance). Smoke-only fixtures, hardcoded-string codec prints, synthetic-tensor quantization rehearsals, and silent dense fallbacks at the default T do **not** satisfy this contract. If the implementation is scaffolding pending a future iter, the validator MUST `SystemExit` with a message that names the scaffold scope explicitly — research-log entries must never claim to test a technique that was silently disabled.

**Verification recipe.**
1. Maintain a single registry `_OPTIONAL_COMPONENT_FLAGS` in `train_gpt.py` that names every default-off `use_X` field. The CLI bool list, the `bool_keys` set, and the startup-banner emission read from it.
2. Provide a parameterised contract test `tests/test_optional_component_flag_contract.py::test_each_flag_has_effect_or_explicit_reject` that, for each entry in `_OPTIONAL_COMPONENT_FLAGS`, asserts **either** (a) `_validate_hyperparameters(_mut(use_X=True))` raises `SystemExit` with a message naming the field, **or** (b) a test exists in `tests/` or `experiments/` whose name contains the field and which exercises a `use_X=True` forward path that asserts an output-differs invariant. Discover (b) by lexical AST scan, not runtime — keep the contract test cheap.
3. On a new optional component, add the registry entry plus *either* a `SystemExit` branch in `_validate_hyperparameters` (with a matching `test_use_X_rejected_*` case) *or* an effect-asserting test in `experiments/test_remaining_components.py`. The contract test fails-loudly otherwise.

**Cross-references.** Companion: [#scalar-semantic-shift](#scalar-semantic-shift) (the scalar analogue), [#untested-path-executability](#untested-path-executability) (the structural prior), [#sibling-fanout-dry-gate](#sibling-fanout-dry-gate) (paired DRY enforcement for the registry), [#audit-row-executability](#audit-row-executability) (the contract test is the executable witness for this rule).

---

### removal-symmetry-sweep

**Date:** 2026-05-15 pre-production review of iter163 promotion + cleanup #47 + cleanup #50 (commit window 884b132 → 7410cb9)
**Rule in CLAUDE.md:** Audit Checklist row · "Removal-symmetry sweep"

**What happened.** Three commits landed in tight succession:

1. **iter163 promotion (884b132)** — added multi-K consistency loss (`multi_k_consistency_anchor_coef=0.1`, `multi_k_consistency_extension_coef=0.1`); emitted `consistency_anchor_loss:` and `consistency_ext_loss:` to `run.log`.
2. **cleanup #50 (56b2f29)** — removed `router_load_cv_coef` and `mos_load_cv_coef` loss multiplications. CV remains as diagnostic-only.
3. **cleanup #47 (7410cb9)** — removed `lip_ub_T/S/F` operator-norm probes, `fp_bound` Banach error bound, and the `power_jvp_F` estimator branch.

Pre-production review with two parallel reviewer subagents (coderabbit + pr-review-toolkit) plus four exploration agents and 720 lines of `train_gpt.py` cleanup found that the removals + addition left ~14 stale references the same-commit cleanup missed:

- `opg_doc.tex` parameter table (lines 195, 197) still listed `router_load_cv_coef=0.0` and `mos_load_cv_coef=0.15` *with the pre-removal value*, i.e. arithmetically wrong.
- `opg_doc.tex` loss equation block (line 676-680) still contained `λ_rcv·Σ CV²(m_r)` and `λ_mcv·CV(m_MoS)²` terms, with prose two paragraphs later (line 735-737) saying these are "gone" — internally inconsistent.
- `opg_doc.tex` feature table (line 243) still claimed `deq_prefix_anchors=false`, conflicting with the CLAUDE.md default-on directive (2026-05-13).
- `opg_doc.tex` Local Stability Guarantee section (lines 1068-1077, 1170-1180, 1218-1228) still described `lip_ub_F` as an "auxiliary K-sweep estimator" and named it as the gate-relevant signal.
- `train_gpt.py` had ~10 stale docstrings/comments naming `lip_ub_F`/`fp_bound`/`power_jvp_F` (`_collect_routing_losses`, `_prescribe_failure_fix`, `_parcae_cycle_F`, `_joint_F_residual_at_saved_fp`, the Hyperparameters Lyapunov section, the Lyapunov FD branch comments).
- `train_gpt.py:8581-8582` emitted `consistency_anchor_loss:`/`consistency_ext_loss:` to `run.log`, but `experiments/plot_metrics.py::parse_log` had no parser entry, `tests/test_training_contracts.py::test_training_log_emits_auxiliary_loss_components` had no `required_fields` entry, and `experiments/test_plot_metrics_parse.py` had no fixture — a Sibling-fanout DRY violation in the *addition* direction.
- `_consistency_extend_no_grad` (the iter163 extension term implementation) silently ran the iteration in fp32 because `_parcae_a_bar()` returns fp32 — a CLAUDE.md "bf16 training default" invariant violation that biased the consistency target relative to the actual training-time DEQ trajectory.

**Root cause.** Two converging pressures: (i) the existing `loss-form-triple-touch` rule names the surfaces for *additions* and "mirror" updates but says nothing about *removals*; (ii) the `promotion-propagation` rule covers default-value drift but not equation-block / probe-name removals. Cleanups #47 and #50 were correctly applied to `Hyperparameters` and the loss assembly, but the same diff did not sweep `opg_doc.tex` paragraph text, docstring bullets, or comment-block narratives that named the removed objects. A removal that leaves the equation in the doc but says "removed" in the prose is the worst failure mode because a future reader cannot tell which version is authoritative.

**The rule.** When removing a Hyperparameter, loss term, or diagnostic, the same commit MUST: (a) delete the field from `Hyperparameters` and CLI parser; (b) delete the corresponding equation/symbol from `opg_doc.tex` (both equation blocks AND parameter-table rows — not just one); (c) remove or annotate every `train_gpt.py` docstring, inline comment, and `_prescribe_failure_fix` bullet that names the field or its formula; (d) leave intact any intentional backward-compat parsing (failure-string handlers, plot-metrics parsers for legacy logs) and co-locate a comment explaining the back-compat scope; (e) sweep `experiments/test_*.py` and `tests/test_*.py` to drop tests that asserted the removed mechanism is active.

**Companion to** `loss-form-triple-touch` (functional-form *changes*, this rule covers *removals*), `promotion-propagation` (this rule covers the removal mirror image of the same surfaces), `sibling-fanout-dry-gate` (the parser-coverage violation in this incident is the addition-direction analogue).

**Verification recipe.** Enforced by `tests/test_removal_symmetry.py`, which maintains an explicit `REMOVED_NAMES` ledger and asserts each appearance in `opg_doc.tex`, `train_gpt.py`, and `CLAUDE.md` is either: (1) inside a backward-compat parser whitelist; (2) on a line that contains an explicit removal annotation (`removed 2026-05-15`, `legacy`, `backward-compat`, `historical`, etc.); or (3) inside a `_prescribe_failure_fix` advisory branch designed to re-parse legacy log strings.

**Cross-references.** Companion: [#loss-form-triple-touch](#loss-form-triple-touch) (functional-form change, this rule's positive-direction counterpart), [#promotion-propagation](#promotion-propagation) (default-value drift, this rule's mirror image for removals), [#sibling-fanout-dry-gate](#sibling-fanout-dry-gate) (paired DRY enforcement for new-field addition — the iter163 log-field parser gap is the addition-direction analogue caught in this same review), [#dead-code-tracking](#dead-code-tracking) (the precursor incident — always-1.0 tracking that survived feature removal at the code level; this rule extends that principle to docstrings and paper-facing equations).

---

## §2. Lessons Learned

Generic guardrails distilled from research-process experience. Not tied to specific code paths or dated incidents — background principles, not enforcement.

- Learned parameters inside an expert path, including norm scales and output heads, must be per-expert.

### Reading derived metrics

Every metric you read in a healthcheck or postmortem is a *function* of raw signals: a running average, a windowed smooth, a per-batch normalization, a post-softmax probability, a cumulative count divided by step number, a log-loss in some unit. Treating the displayed value as if it were the underlying signal is the most common analysis-error class on this project (see [#cumulative-metric-misread](#cumulative-metric-misread) for the canonical 2026-05-02 incident).

**The principle** distills to three sentences:

1. **Read the definition before the value.** Before concluding anything from a derived metric, write down (a) which raw signal it derives from, (b) what transformation is applied, (c) when the derived value is within ε of the underlying truth you actually care about. Skip step (c) and you will mistake convergence-time artifacts for real signals.

2. **When in doubt, compute from raw.** Every derived metric has a raw counterpart you can reconstruct from log lines (e.g. `Δ_t = train_time[t] − train_time[t−1]` instead of `step_avg[t] = train_time[t]/t`). The raw signal is always interpretable. If two readings disagree about whether the system is healthy, the raw signal is right and the derived one is missing context.

3. **User pushback is a Bayesian prior, not a debate.** When the user questions a conclusion, the correct first action is to verify by re-deriving from raw, not to defend the original reading. The user typically has domain context (compile init costs, K-jitter expectations, optimizer warmup) that closes the gap between the derived metric's value and the underlying truth.

**Concrete examples of the trap on this codebase:**

- `step_avg = train_time/step`: cumulative average; converges to per-step rate only past `t » outlier_cost / asymptotic_step`. For our compile-init ~100s and asymptotic ~25s, that's `t ≥ 200`. Healthchecks before s50 must compute per-step delta. ([#cumulative-metric-misread](#cumulative-metric-misread))
- `step_avg_w50`: windowed over last 50 steps; still cold-start-contaminated when `step < 50` (the window includes warmup samples).
- `deq_recon_err` under TBPTT: not a true reconstruction error when `deq_bptt_k < num_layers` — see [#deq-recon-err-interpretation](#deq-recon-err-interpretation).
- `attn_ortho` / `mlp_ortho` train-time vs val-time: train uses smaller batch averaging, val uses full-batch — the val signal is more stable and is the diagnostic-grade reading.
- `pertoken_entropy` vs `pool_entropy`: per-token concentration vs global utilization, completely different axes despite both being entropy-of-routing-weights.
- Bits vs nats vs bpb: easy to read a "loss = 2.5" without checking units; if it's ntp_loss in nats, that's bpb ≈ 1.45; if bits, bpb ≈ 0.43.

**Recovery protocol when an analysis error is suspected:**

1. Identify the metric that drove the conclusion.
2. Re-derive from raw signals (`grep` the underlying values from `run.log`).
3. If raw and derived disagree, raw wins.
4. Update the conclusion AND document the misread in the relevant EXPERIENCE.md section so the same trap is closed for future sessions.

### Metrics
- Track and compare the *scored* metric (post-quant) separately from any in-training validation.
- Always report both the final in-training validation and the final post-quant result.
- Name metrics by what they actually measure; don't reuse a proxy metric under a different label.
- For hard constraints, use a worst-case aggregation that matches the guarantee you want (e.g. max over per-expert means).
- Prefer enforcing hard constraints in the space you diagnose; remove proxy regularizers if they don't transfer.
- Treat expert health metrics as non-negotiable guardrails; optimize everything else inside that envelope.
- When a metric is a hard end constraint, add a direct soft barrier loss for it instead of hoping a proxy will transfer.
- Validate hard constraints in eval-mode; train-mode "healthy routing" can be misleading.
- Don't train on a diagnostic unless it consistently improves the scored metric; keep "convergence" as a monitored signal, not a loss.

### Logging
- Treat the primary run log as the source of truth; don't rely on wrapper tools capturing stdout/stderr.
- For progress monitoring, tail the run log directly.
- Avoid logging stale diagnostics outside their natural cadence (train-step vs val-step).
- Make ablations an explicit, logged flag so runs remain comparable and reproducible.
- Never append multiple runs into a single comparison log; truncate logs per run or make parsers select the last run.
- Avoid multiple processes writing to the same log file concurrently; keep a single canonical log and derive copies from it.

### Validation Curves
- If you want plots with curves, make sure the run actually logs multiple validation points (set a non-zero validation interval).
- For in-training curves, validate on a fixed small subset; reserve full validation for final-only checkpoints.
- Make full-validation opt-in when iterating locally; it can dominate wall-clock and mask training issues.

### Plot Robustness
- Don't treat missing metrics as zeros; represent "not logged" explicitly.
- Align each metric with its natural cadence (train-step vs val-step) to avoid misleading plots.
- If a metric is sparse, plot it sparsely (connect points) rather than fabricating dense values.
- Make plotting resilient to partial runs (missing finals) so early debugging doesn't break.
- For near-zero diagnostics, log/plot with enough precision (e.g. scientific notation + log scale).
- Diagnostics should run in consistent precision; mixed-precision drift can look like "instability".
- Prefer line styles (not point clouds) for multi-component time series, and always include a legend for the encoding.
- When two series are intentionally identical (tied components), deduplicate the plot so style overlays don't look like mismatches.
- When adding new logged keys, update the parser and add a small unit test so plots don't silently degrade.
- When a metric is renamed for clarity, keep a compatibility alias until all plots/tests have been updated.

### RevDEQ
- FP64 add/sub is a *reversibility* tool (reconstruction accuracy), not a default training requirement.
- Use autograd-unroll when you need output-space regularizers; use RevDEQ backward when you need constant-memory exact gradients.
- Only compute/log reconstruction error when using the RevDEQ backward path (otherwise it's not an actionable signal).
- Reconstruction error is bounded by state rounding; interpret it as a trend/guardrail, not a "should be zero" assertion.
- Avoid caching inference-mode tensors into modules that are reused for training; refresh caches when switching modes.
- Prefer simple, explicit parameterizations over hidden stability clamps; diagnose fixed-point behavior directly via residual/convergence metrics.
- When you need contraction, add a single block-level gate and log it; don't hide stability in many per-path scale knobs.
- Intermediate DEQ supervision shapes early iterates, but it's compute-heavy; keep it sparse and aligned with the scored output.
- When enforcing a hard constraint, prefer a barrier loss (penalize violations only) over always-on regularization.
- For hard constraints under expensive solvers, use frequent small-prefix barriers rather than rare full-prefix penalties.

### Refinement
- When mixing token distributions, normalize each input distribution first and renormalize after mixing.

### Configuration
- Keep experiment hyperparameters in code defaults (or CLI), not hidden environment variables.
- Remove dead/unreachable configuration paths; they silently rot and confuse debugging.
- For gated routers, define expert health on renormalized expert shares; track total routed mass as a separate metric or loss.
- Prove flattened tensor-bank layouts against an index-explicit reference such as `einsum`; shape checks do not prove semantic correctness.
- Assert optimizer coverage for every trainable parameter; manual groups silently miss newly added norms and heads.
- In iterative solvers, prefer pooled convex gates with conservative initialization to preserve a stable identity path.
- When an ablation is complete, delete the deprecated mode so logging, plots, and constraints can't silently drift.
- Throughput tuning must be validated against stability signals; "faster" configs that break DEQ behavior are not viable defaults.
- Avoid saturated sigmoid gate inits; mid-point initialization keeps gradients alive and improves solver stability.
- Routers are control modules; give them a gentle optimizer/separate LR so they don't destabilize late training.
- Don't bury hard-constraint barriers behind tiny global coefficients; separate "health" losses so violations remain enforceable.
- Treat norm placement as an ablation knob; revert quickly if it worsens stability or the scored metric.
- Initialize gates to meaningful mid-range values; extreme gate biases can hide capacity or destabilize dynamics.
- Keep evaluation batching independent from training gradient-accumulation; eval should reflect the true throughput target.

### Environment Sanity Checks
- Before long runs, verify the environment can see CUDA and the dataset/tokenizer paths resolve.
- Generate plots using an environment that has the plotting dependencies installed.

### Randomness
- When you need randomness *and* coverage, use a shuffle-bag sampler instead of i.i.d. draws.

### Diagnostics
- Match plotting scale and summaries to the metric's dynamic range and sampling scheme so you don't mistake artifacts for behavior.
- When experiments have hard constraints, bias toward changes that can be bounded and verified early (avoid hour-long runs that only fail at the end).
- Metric contract rule: every required diagnostic needs a compute site, a freshness tag when it is tied to a forward pass, human log emission, parser support, and a focused test. Independent metrics must not be gated by unrelated diagnostic availability.

### Efficiency
- In DEQ-style models, cost scales with batch × iterations × refinements; tune these jointly to avoid runaway wall time.
- Treat `torch.compile` as an experiment: guard it behind a flag and verify peak VRAM, since recompiles/capture can unexpectedly OOM.

### Distributed
- Any rank-conditional control flow around collectives is a correctness bug; all ranks must execute collectives in the same order.

### DRY Function Orthogonality
- Shared tensor preparation, context selection, DDP reduction, metric formulas, and log formatting should each have one implementation. If fast validation and K-sweep need the same fixed-point probe, both call the same setup helper.
- Keep functions orthogonal: a probe returns raw measurements, a metric helper transforms raw measurements into a gate value, and a logger formats fields. Do not hide policy decisions inside measurement routines.
- When adding a metric, name the mathematical object precisely. If a value is a numerical estimate or conservative metric rather than a formal proof, the function name, log label, and docs must say so.

---

### Routing Health Metrics

`CLAUDE.md` keeps the concise metric principle; this section holds the definitions, targets, and decompositions (moved out of `CLAUDE.md` to keep it concise — see [#claude-md-size-budget](#claude-md-size-budget)).

- **`pertoken_entropy`** — `H_pertoken = mean_token(−Σ_e w(e|t) log w(e|t))`. LOW means each token concentrates on few experts → specialization. Single pool-level value (per-token entropy is a pool-level quantity by construction — each token has ONE distribution).
- **`*_entropy`** (global utilization) — `H_global = −Σ_e p̄_e log p̄_e` over batch-averaged shares `p̄_e`. HIGH ≈ log(N_routed) means uniform usage across batch — no dead experts. Reported per-slice (attn / mlp, renormalized within-slice) AND pool (full unrenormalized 2R distribution).
- **You can have HIGH global *and* LOW per-token simultaneously** — that's the target regime. Every expert gets used somewhere in the batch; each individual token uses only a few experts strongly.
- **`*_min_share`** — `min_e p̄_e`. Sentinel for dead experts; report per-slice because the per-component shares differ.
- **`*_cv`** — coefficient of variation. Per-slice CV uses the renormalized within-slice distribution; pool CV uses the full 2R unrenormalized distribution. Diagnostic: large gap between attn_cv and mlp_cv = role-asymmetric routing (e.g. iter 100b s120 attn_cv≈1.07 / mlp_cv≈0.18: attn winner-take-all, MLP uniform). Large pool_cv with small per-slice CVs = cross-slice dominance.
- **`*_ortho`** — `max|cos_sim|` between expert OUTPUT means. Reported per-slice because attn experts and MLP experts produce DIFFERENT outputs even with the shared (pooled) router.
- **`router_mass`** — mean `sigmoid(gate)`. Drops when the model gates the mixture down.
- **`lip_ub`** — the single logged numerical local-contraction metric at the saved DEQ FP `z*`. Internally it is a power-iteration estimate of `||∂T_θ/∂z||_2` with `fp_lip_ub_safety` / `fp_lip_ub_margin` applied. `lip_ub < 1` is the operational sufficient local contraction check, but it is still a numerical metric, not a formal interval/linear-relaxation certificate.
- **`fp_residual_rel`** — direct relative fixed-point residual proxy from the saved solve, currently `deq_iter_conv_rel` / `iter_conv_rel` under the existing solver diagnostics.
- **`fp_bound`** — a posteriori relative fixed-point distance proxy `fp_residual_rel / (1 - lip_ub)` when `lip_ub < 1`. This is the compact convergence certificate: small residual plus a contraction margin bounds distance to the local fixed point. `N/A` means `lip_ub` was missing or not below 1.

**Parcae and contraction attribution.** In the active Parcae path, scalar
`deq_beta` is not the solver blend; Parcae computes per-dim
`beta = 1 - A_bar`. More importantly, `lip_ub` is a probe of the
transition map `T_theta(z, x0)`, not probes of the blended solver update
`(1-beta)z + beta T_theta(z, x0)`. Therefore lowering scalar `deq_beta` cannot
be a principled fix for a failed `lip_ub` under Parcae. A real fix must shrink
or regularize the transition Jacobian itself, for example through weight decay,
an output-scale constraint, or a dedicated transition-Jacobian penalty.

**iter145r promotion tech debt and future fix.** iter145r promoted
Dirichlet-UCB + EMA-balanced routing for BPB, but its post-int diagnostics did
not certify strict health: attention min share stayed just under the hard floor,
expert-output cosine gates failed, and `lip_ub` stayed far above 1 while
older auxiliary probes gave conflicting trend signals. Treat this as a reason
to report one contraction metric: `lip_ub`. The next run is intentionally split
for attribution: iter146 tests doubled ideal-target regularizers and weighted
high-K jitter, while iter147 separately tests direct Lyapunov contraction.

- Iter146 doubles `router_ema_balance_coef`, `router_ema_specialization_coef`,
  and `expert_output_diversity_coef` without enabling an EMA alive hinge. This
  tests whether stronger pressure toward balance, specialization, and output
  orthogonality fixes the health gates without changing the loss form.
- Iter146 also emits Dirichlet confidence diagnostics: strength `S`, uncertainty
  mass `E/S`, marginal sigma, evidence, normalized `H(mu)`, and current UCB beta.
  Expected learning signal is rising strength/evidence, falling `E/S`/sigma,
  and lower `H(mu)` if routing specializes.
- Iter147, if needed, adds a low-cadence Lyapunov expansion penalty on
  `T_theta`: `relu(||T(z*+eps v,x)-T(z*,x)||/eps - gamma)^2`. The clean
  certificate is still `lip_ub < 1`; do not re-enable the old stochastic
  Lyapunov surrogate unchanged.
- Keep routing token-local. Do not introduce Sinkhorn, capacity matching, or
  batch-coupled assignment to force expert usage.

**Evidential router / EMA-anchor principle.** A Dirichlet router can be tested
without changing the RevDEQ contract: each token maps `h -> Softplus(W h + b) ->
alpha=e+1 -> mu=alpha/sum(alpha)`, then an annealed local UCB bonus from the
marginal Beta variance can be added before a smooth positive simplex projection.
This is token-local and linear in expert count; it is not a batch-level
assignment. Persistent EMA usage is also valid, but it is detached state.
Therefore a plain detached `KL(EMA || uniform)` scalar is only a diagnostic. If
it is used as a trainable balance objective, use an explicit straight-through
EMA anchor whose forward value is `KL(EMA || uniform)` but whose gradient flows
through the current token-local routing share. Pair that global balance term
with bounded `-KL(P_token || stopgrad(EMA))` specialization if sharp assignments
are desired. This preserves the RevDEQ map: routing decisions depend on the
token and parameters only; historical usage influences only auxiliary gradients
or slow state updates.

**Orthogonality principle.** Router-row orthogonality is only a weak conditioning
prior. It does not guarantee that experts are used or that they compute different
functions. The more direct specialization signal is the normalized expert-output
Gram/cosine penalty: zero off-diagonal output Gram on active tokens means active
experts produce decorrelated transformations on the data manifold. Pair this with
EMA usage/liveness; output Gram alone cannot rule out unused experts.

**Prefix convention** (iter 100b). The SoftDenseRouter is a SINGLE pooled router shared across attn and mlp components. Routing-distribution metrics decompose into THREE values: `attn_*` (per-slice renormalized), `mlp_*` (per-slice renormalized), and `pool_*` (full 2R distribution). Metrics derived from **expert outputs** (usage, ortho, min_share per slice) keep `attn_*`/`mlp_*` only — there is no pool variant.

**K-sweep tabular emission** (PERMANENT iter 100b; simplified 2026-05-08). The eval K-sweep emits a `k_sweep_table:` row per K with fixed-width columns: `K val_bpb attn_cv mlp_cv pool_cv attn_min mlp_min attn_ortho mlp_ortho pertoken_ent pool_ent shared_gate dir_S dir_U dir_sigma dir_evid dir_Hmu ucb_beta lip_ub fp_bound iter_conv_rel`. A header row precedes data rows. `N/A` indicates an unavailable field. The legacy `k_sweep:k=N val_bpb:... attn_gate_iter:[…] router_gate_iter:[…] iter_conv_rel:… residual:…` line is preserved for `experiments/plot_metrics.py` back-compat and now also emits full `router_dir_*` confidence fields plus `lip_ub` for parser-friendly grep. Use `k_sweep_table:` for cross-K and cross-iter routing-health comparisons; use `k_sweep:` for per-iter gate trajectories. Fast validation logs the same fixed-point certificate fields when `fp_lip_fast_val_every > 0`.

---

## Operational Reference

Detailed operational material lives here so `CLAUDE.md` can stay limited to principles and short rationale. If this section disagrees with code, `train_gpt.py::Hyperparameters` and the runnable scripts win; update this section as documentation debt.

### pre-action-memory

Memory directory: `/home/mzhong4/.claude/projects/-project-ylin-mzhong4-research-opg-parameter-golf/memory/`.

Before changing code, docs, launches, or commits, read the memory index and then the relevant files. The durable categories are:

- User profile: `user_profile.md`.
- Feedback rules: hypothesis sync, pre-commit review chain, simplify-before-commit, dry fixes, architecture-over-hparams, compile-disabled-for-dev, DDP default, wakeup cadence, Lipschitz/K-sweep, per-wallclock override, sparsity value props, `uv` installs, GPU preflight, `conda run --no-capture-output`, throughput priority, SDPA replacement caution at T=2048, diagnosis context, NTP descent-rate metric, cumulative-vs-instantaneous metrics, routing metric axes, grad-enabled checks, bf16 training default, profile-before-throughput, decoupled regularizers, annealed sparsity coefficients, DEQ fixed-point framing, full-dim experts, MLA preference, MoS softmax routing, and decoupled refinement.
- Project state: autoresearch protocol, Group-F lessons, experiment results, phase learnings, throughput-first notes, DEQ depth insight, architecture ideas, and deferred tech debt.

When adding a memory file, add it to the memory index and to the appropriate category documentation in the same commit so future sessions discover it.

### project-constraints

OpenAI Parameter Golf target: train the best small LM that fits in a 16 MB artifact, trains in no more than 10 minutes on 8xH100 SXM, and scores by FineWeb validation `val_bpb`. Lower is better.

Hard constraints:

- Artifact: code plus compressed model must be no more than 16,000,000 bytes.
- Submission training: no more than 600 seconds on 8xH100 SXM. `--max-training-seconds=600` is the explicit submission opt-in; evaluation reservation time is part of the loop budget.
- Data/tokenizer: FineWeb validation with SentencePiece BPE vocab 1024.
- Baseline pointer: the active promoted baseline is recorded in `experiments/docs/hypotheses.md`; update that entry when promoting.

### environment-and-files

- Conda env: `conda activate opg` before training or tests.
- Dependencies: `requirements.txt`; do not add packages unless explicitly authorized. Authorized installs use `uv pip install <pkg>` and update requirements in the same commit.
- Data: `./data/datasets/fineweb10B_sp1024/` is read-only. Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`.
- Main implementation: `train_gpt.py`.
- Research log: `experiments/docs/hypotheses.md`.
- Incident/rationale archive: `EXPERIENCE.md`.
- Plot/log rotation: `experiments/update_results.sh`.
- Untracked runtime files: `results.tsv`, `run.log`, `experiments/training_logs/*`, and `experiments/weights/*`.
- Historical submissions: `records/` is read-only.
- Paper-facing algorithm doc: `opg_doc.tex`; update it when implementation intentionally diverges.
- Reference implementations: RevDEQ at `/home/mzhong4/work/research/rdeq/WIP-ARWDEQ/code/arwdeq/qwen3_utmoe_revdeq.py`; TSU/CTP/NTP/MoS at `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py`.

### runbook

Default dev run, all visible GPUs, step-governed:

```bash
conda activate opg
torchrun --standalone --nproc_per_node=gpu train_gpt.py
```

Shorter dev run: add `--iterations=N`. Explicit GPU count: `torchrun --standalone --nproc_per_node=2 train_gpt.py`. Single-GPU `python train_gpt.py` is debug-only.

Submission-style run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --max-training-seconds=600
```

Evaluate a run log quickly:

```bash
grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log
```

Fair comparison default: compare equal step count. Use wall-clock comparisons only for submission or explicit throughput iterations, and compute instantaneous per-step deltas before trusting early `step_avg`.

### experiment-loop-details

Canonical iteration loop:

1. Review `experiments/docs/` comprehensively: read `README.md` and `hypotheses.md`, list the directory, and inspect any relevant archive or `iterNNN_*.md` design notes.
2. Read git state, recent logs, `results.tsv`, and relevant hypotheses.
3. State the hypothesis and make one focused change unless explicitly testing a bundle.
4. Add or update focused tests before implementation when behavior changes.
5. Run smoke before long training.
6. Run training with output captured to `run.log`.
7. Inspect scored metrics, K-sweep, artifact bytes, peak VRAM, and failure diagnostics.
8. Run `bash experiments/update_results.sh` after every iteration to rotate logs/weights and regenerate plots.
9. Apply promotion rules.
10. Update `experiments/docs/hypotheses.md` immediately with evidence, status, and confounds.
11. Stop after 100 consecutive non-improvements and ask for direction.

Logging/plot artifacts:

- Training logs: `experiments/training_logs/{baseline,previous,current}.log`.
- Weights: `experiments/weights/{baseline,previous,current}/`.
- Metric plots: `experiments/metrics_comparison.png`, `experiments/progress.png`, `experiments/progress_full.png`.

### hypothesis-log-detail

Each iteration entry must include the complete evidence packet when available:

- roundtrip int6 `val_bpb` and validation loss;
- the full `k_sweep_table:` matrix from `run.log`, including acyclicity primes 17, 37, and 113 when emitted;
- validation trajectory by checkpoint with deltas;
- acyclicity-prime check against nearest power-of-two K;
- NTP descent rate as per-step `Δntp / 10 steps` and per-wallclock equivalent over computable windows.

Partial previews are not exempt. Emit the windows and components that exist; put missing components under `Caveats` with a recovery plan. This rule prevents useful short-run evidence from silently dropping mandatory diagnostics.

### current-architecture-reference

The authoritative defaults live in `train_gpt.py::Hyperparameters`. This mirror is for orientation only and should be refreshed when a promoted config changes.

Current high-level shape:

- 12-layer RevDEQ-style shared block, `model_dim=768`, sequence length 2048, vocab 1024, tied embeddings.
- Dense soft MoE with 16 experts plus one always-on shared expert, full-D LoRA-style expert internals, MLA attention, and NTP MoS output by default.
- Parcae-style per-dimension damping/injection enabled; scalar beta path is fallback only.
- Default training uses K-jitter and TBPTT; default comparison is 1000 iterations unless overridden.
- int6 per-row quantization plus zstd-22 compression is the scored artifact path.

Current default families to check in code before launch:

- Core dimensions: layers, heads/KV heads, expert count, ranks, sequence length, batch tokens, refinement count, CTP flag, NSA flag.
- Solver: Parcae init/floor, `deq_bptt_k`, weighted K-jitter set, beta fallback/jitter, Lyapunov/denoising disabled state.
- Optimizer: Muon/AdamW grouping, LRs, PE-NS backend, momentum warmup, weight decay, gradient clipping, warmdown.
- Routing/loss stack: iter146 Dirichlet-UCB router, doubled EMA balance/specialization, no routed sigmoid gate, router CV off by default, MoS CV, per-token entropy, doubled expert diversity, MoS diversity default-off, entmax blend, logit softcap, routing mass and Dirichlet confidence diagnostics.
- Quantization/eval: int6 roundtrip, sliding-window eval, artifact byte accounting.

### revdeq-architecture-details

RevDEQ model class:

- The DEQ solver loop updates coupled states and should be treated as a fixed-point solve, not as a stack of independent transformer layers.
- Optional refinement is a separate predict -> soft-embed -> re-solve loop. `num_refinements=0` keeps it off by default; preserving the path enables future diffusion/AR experiments without changing the solver definition.
- Warm start uses token embedding `x0`; refinement warm start uses the refined input.
- Add/sub reconstruction uses fp64 because reversibility is a numerical correctness requirement.
- Smoke must check loss descent, finite gradients, non-exploding convergence, and routing health. True precision-level reconstruction is only expected under full-BPTT smoke.

Full-D MLA standard:

- Every expert owns its Q/KV/K-rope/Wo, MLP, norms, and MoS A-bank parameters.
- Low-rank factors constrain parameter count, but activations and SDPA stay at full model dimension with tensor-core-friendly head dims.
- Expert index is packed into the head dimension for one SDPA call where possible.
- Decoupled RoPE and per-expert gated attention are part of the standard attention path.
- NSA/sparse attention variants remain default-off unless explicitly ablated; replacing optimized SDPA at T=2048 requires profiling evidence.

MoS/refinement defaults:

- MoS routing is pure softmax.
- CTP is preserved behind a flag but disabled by default; NTP-only is the promoted baseline behavior.
- FSQ in the MoS intermediate projection is disabled by default; low-rank projection alone is sufficient under current evidence.

### revdeq-reversibility-floor

Any coefficient used as a divisor in RevDEQ backward reconstruction needs a lower bound. For Parcae-style damping, the relevant term is `A_bar = 1 - beta`; the floor controls worst-case amplification across reverse steps.

Current invariant:

```text
A_bar = eps_rev + (1 - eps_rev) * exp(delta * A)
eps_rev = parcae_reversibility_floor = 0.1
```

`eps_rev` is a correctness constant, not a tuning knob. Changing it requires a same-commit update to the Hyperparameter/default, reconstruction-floor test, smoke tolerance, and `opg_doc.tex` remark. Coefficients that do not divide the reverse reconstruction, such as `B_bar` inside `T_theta`, must not receive artificial floors.

### routing-reg-input-invariant

Router regularizers must see the combined routed mass:

```text
p = simplex(allocation_scores) * sigmoid(gate_logits)
```

The principle is simple: if the sigmoid gate suppresses an expert path, load-balance and sparsity losses must see that suppression. Renormalizing shares before the loss hides gate effects and optimizes a different distribution. MoS is exempt because it is intentionally a pure softmax convex combination. If a run explicitly disables the routed sigmoid gate, this reduces to `p = simplex(allocation_scores)` and forces routed expert output to participate; treat that as a controlled ablation, not a silent default change.

Useful audit:

```bash
grep -nE 'share / share\.sum\(' train_gpt.py
```

Hits in routing-regularization paths require review.

### no-top-k-dispatch

Hard discrete routing decisions are not RevDEQ-safe in the learned fixed-point map. Top-K gather, argmax routing, capacity drop, and hard threshold skips make the forward map piecewise/discontinuous and can make reverse reconstruction depend on a different branch than the forward pass.

Permitted categories:

- dense soft routing;
- smooth per-token relaxations such as softmax/entmax or Gumbel-softmax when used differentiably;
- epsilon skips only when the truncation is below bf16 numerical floor and proven not to change reconstruction decisions;
- discrete logic outside the RevDEQ path or guarded off under RevDEQ.

Sinkhorn/OT-style routers are not permitted for expert assignment in the learned
map, even when differentiable. They make a token's expert weights depend on the
other tokens in the batch and on global expert occupancy/capacity. Project
routing semantics require local token routing: each token's expert weights are a
function of that token representation and learned router parameters only. Global
usage pressure belongs in auxiliary losses or slow router-bias feedback, not in
the forward assignment solver.

Useful audit:

```bash
grep -nE 'topk\(.*expert|capacity_factor.*ceil|argmax.*router' train_gpt.py
```

Every match must be either outside RevDEQ, default-off, or justified by a smooth/reconstruction-safe argument.

### grad-enabled-vs-requires-grad

`nn.Parameter.requires_grad` is a static property, not a runtime-mode signal. A fast path chosen only from `requires_grad` can incorrectly disable inference/no-grad kernels because parameters still have `requires_grad=True` during `torch.no_grad()` fixed-point evaluation.

Dispatch rules:

- use `torch.is_grad_enabled()` for runtime autograd mode;
- combine it with actual tensor/parameter grad requirements when deciding whether a custom kernel must preserve gradients;
- test both training and no-grad/validation paths.

### Bottleneck Experts (closed)

**Decision.** Do NOT re-introduce bottleneck-style experts (`BottleneckIn` `D→proj_rank→r` + `ExpertBody` at small `r` + `BottleneckOut` `r→proj_rank→D`) as a scaling axis. Tested as Group D (iter 90, 91+92) and NOT PROMOTED.

**Why.** Two compounding penalties:
1. **Per-param efficiency**: bpb/param 1.49 vs full-D LoRA's 1.17 — ~27 % worse (H70).
2. **SDPA throughput**: `proj_rank ≤ 192` with `H_in ≥ 4` forces `d_head ≤ 48`, off the FlashAttention tensorcore sweet spot (64+).

Both penalties compound when scaling `N_expert`. The iter 96 PROMOTED axis — full-D LoRA with rank-halving / E-doubling at iso-cost on linears (H71) — supersedes it.

**Archival.** Bottleneck infrastructure preserved at git tag `iter-91+92-bottleneck-NOT-PROMOTED` (commit `3e35655`) and side branch `autoresearch/bottleneck-rescue` (`proj_rank=48/64` rescue workspace). Routing semantics it would feed into are unchanged — see `CLAUDE.md` Architecture Principles.

---

### deq-recon-err-interpretation

Whenever `deq_bptt_k < num_layers` (the default since iter 28-tbptt) the RevDEQ reverse loop stops at iteration `K_fwd − K_bwd`, NOT at `z_0`. The logged metric

```
deq_recon_err = (‖z_rec − z_0‖ + ‖y_rec − z_0‖) / ‖z_0‖
```

then measures how far the forward FP *travelled* in the un-reconstructed iterations — a "distance travelled" gauge, NOT a numerical reconstruction error. Expect values O(1) once the FP is non-trivial; do NOT gate divergence / promotion on its absolute magnitude. To measure true RevDEQ reconstruction error (target near fp64 precision, ~1e-12), set `deq_bptt_k = 0` (full BPTT) and re-run; only that regime makes `recon_err` comparable to the fp64 floor.

**Smoke-test caveat.** The smoke test asserts "recon near precision" — that assertion is only valid when the smoke runs **full BPTT**. Under TBPTT-default the smoke must use a different stability check (loss decreasing · `deq_iter_conv_rel` not exploding · no NaN/Inf · expert routing healthy).

---

### Disabled Techniques

Maintained here so removed/disabled techniques don't accrete annotations in `CLAUDE.md` or the config mirror.

- **SWA (Sliding-Window Attention)** — disabled iter 1: dragged gates toward identity at the 1 h budget. Sliding-window EVAL (stride = 64) is unrelated and stays enabled.
- **BigramHash** — `bigram_vocab_size = 0` (iter 93 / H64). Code retained behind the flag.
- **FSQ in MoS head** — `fsq_levels = 0` (iter 62 / H53). Low-rank MoS projection alone is sufficient; FSQ code retained for re-enabling.
- **Lyapunov regularizer** — `lyapunov_coef = 0.0` (iter 88). Parcae per-dim Ā already bounds spectral radius.
- **HyDRA denoising** — `denoising_coef = 0.0` (iter 89). Same Parcae-redundancy logic as iter 88.
- **CTP head** — `use_ctp = False` (iter 94 / H60). NTP-only; CTP param banks not allocated.
- **Variance regularizer** — removed entirely (iter 117 v3). Was the underlying driver of the iter 121 PE-NS NaN cascade — see [#variance-reg-ns-cascade](#variance-reg-ns-cascade).
