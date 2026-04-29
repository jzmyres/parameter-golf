# EXPERIENCE — Incident Archive & Lessons Learned

This file has two roles, in this order:

1. **Incident-Driven Rules Archive (§1)** — dated postmortems, ported from former `CLAUDE.md` §Development Practices. Each section documents one bug class that shipped, the root cause, and the verification recipe. `CLAUDE.md` §9 audit checklist cites these by anchor.
2. **Lessons Learned (§2)** — generic guardrails from research-process experience, not tied to specific code paths. Background reading; not enforcement.

`CLAUDE.md` is the **enforcement surface** (terse rule + grep command). This file is the **historical record** (why the rule exists). New rules from pre-commit reviews are added to §1, then cited from `CLAUDE.md` §9 — they MUST NOT accrete in `CLAUDE.md` itself.

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

### Section template

```
### <slug>

**Date:** YYYY-MM-DD review of iter NN
**Rule in CLAUDE.md:** §9 audit checklist row · §<N> <Section name>

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
3. **CLAUDE.md "Current Architecture" table is mirror-only**. Edits to `Hyperparameters` and edits to that table MUST land in the same commit.
4. **`opg_doc.tex` §2.1 and the "Current SOTA"/"working baseline" lines are dated artifacts**. A PR that mutates `Block.forward`, the DEQ equation, or the promoted baseline MUST update these in the same commit, or open a `TODO(paper)` ticket noting the divergence.

**Verification recipe.** Before committing any change to architectural knobs:
- `grep -n 'num_experts\|num_layers\|model_dim' train_gpt.py` — confirm constructors read from `args.<field>`, not literals.
- Tests: `assert model.<field> == args.<field>`; never compare to a hard-coded number.
- CLAUDE.md §5 table updated in the same commit as `Hyperparameters`.

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
- Record the diagnostic in `experiments/hypotheses.md` but continue running the next queued iter on top of the new baseline.

**Verification recipe.** No grep — this is a promotion-policy invariant. The commit message for any strict-generalization iter must include the three items above, and CLAUDE.md §11 enforces "do not revert".

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

**Cross-references.** Adjacent to the soft-dense-routing "expert independence" hard constraint in CLAUDE.md §6 — both enforce that no shared trainable parameter couples expert/projection paths.

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

**Cross-references.** Adjacent to the "Tensor Layout" guidance in §9 (manual flatten/reshape needs einsum equivalence) — both stem from "shape is not semantics".

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
**Rule in CLAUDE.md:** §9 row "Hyperparameter fan-out" · §5 Single-source-of-truth

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
3. Row in CLAUDE.md §5 mirroring the dataclass default.
4. Row in `opg_doc.tex` parameter table (or a "Practical implementation" deviation note per the doc-code-invariant rule).

When effective magnitude differs from documented magnitude (as with the entropy loss × balance-mult dedup), document the *effective* value in §5 OR fix the multiplication so documented = effective. Do not silently leave readers with the wrong mental model.

**Verification recipe.**
- Pre-commit grep: every CLAUDE.md §5 row name must match `args.<row>` somewhere in `train_gpt.py`.
- `experiments/test_cli_parser.py::test_default_parity` iterates the documented routing knobs and asserts each is reachable through `_parse_cli_overrides` with the documented default.
- Reviewers reading a PR that adds a knob should grep CLAUDE.md §5 + `_parse_cli_overrides` for the new field name BEFORE approving.

**Cross-references.** [#config-drift](#config-drift) (parent pattern), [#doc-code-invariant](#doc-code-invariant) (the doc-side enforcement).

---

## §2. Lessons Learned

Generic guardrails distilled from research-process experience. Not tied to specific code paths or dated incidents — background principles, not enforcement.

- Learned parameters inside an expert path, including norm scales and output heads, must be per-expert.

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

### Efficiency
- In DEQ-style models, cost scales with batch × iterations × refinements; tune these jointly to avoid runaway wall time.
- Treat `torch.compile` as an experiment: guard it behind a flag and verify peak VRAM, since recompiles/capture can unexpectedly OOM.

### Distributed
- Any rank-conditional control flow around collectives is a correctness bug; all ranks must execute collectives in the same order.
