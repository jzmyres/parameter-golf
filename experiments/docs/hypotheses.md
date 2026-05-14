# Experiment Hypothesis Log

Concise working ledger for active research decisions. Detailed historical
entries and raw evidence are preserved in
[`experiments/docs/hypotheses_archive.md`](./hypotheses_archive.md). Operational
protocol and failure-mode details live in [`EXPERIENCE.md`](../../EXPERIENCE.md).

Use this file for:
- what was tested,
- what changed,
- what the takeaway was,
- what remains in the queue,
- whether the queued change is principled and efficient to test.

## Status Vocabulary

| Status | Meaning |
|---|---|
| `PROMOTED` | Controlled result improved or preserved the promotion gates and should become the default or default-ready path. |
| `NOT PROMOTED` | Controlled result regressed, cost too much, or failed required health gates. |
| `VERIFIED` | The specific hypothesis was directly supported by a controlled test. |
| `REFUTED` | The specific hypothesis was directly contradicted by a controlled test. |
| `SUPERSEDED` | The idea was absorbed by a newer mechanism or no longer matches the active code. |
| `PROPOSED` | Principled but not yet tested under the current baseline. |
| `DEFERRED` | Plausible, but not efficient to test before nearer-term blockers are resolved. |

Confounded bundles may only claim that the bundle worked or failed. A single
mechanism is verified only after a one-variable comparison against the active
baseline.

## Current Snapshot

| Area | Current state |
|---|---|
| Code source of truth | `train_gpt.py::Hyperparameters`; docs must track active defaults, not old records. |
| Backbone | RevDEQ + Parcae, `use_parcae=True`, weighted `deq_k_jitter_set=(16,24,32,64)` with weights `(0.50,0.40,0.07,0.03)`, `deq_bptt_k=3`. |
| Routing loss stack | Root-cause rescue stack: Dirichlet-UCB routing, router CV off, EMA-anchored balance/specialization (`router_ema_balance_coef=0.30`, `router_ema_specialization_coef=0.20`), small alive hinge (`router_ema_alive_coef=0.02`), worst-pair cosine expert-output diversity (`expert_output_diversity_coef=0.30`), MoS-CV/per-token entropy unchanged, and 0.07 warmup. Legacy nested routing-gram stack is superseded. |
| Main active question | Health-first rescue after user directive: promotion hard-gates on validation BPB only. iter155 (corrected after user feedback 2026-05-13) redirects the contraction work from probing $T_\theta$ to probing the **actual two-state Parcae cycle map** $F$ where $y'=\bar A y+\beta T(z),\;z'=\bar A z+\beta T(y')$, $\beta=1-\bar A$. The earlier single-state surrogate $S=\bar A z+(1-\bar A)T_\theta$ was demoted to advisory because the solver does not iterate $S$. `lip_ub_F<1` is the gate-aligned **empirical local-contraction metric** (safety-adjusted power-iteration estimate, not a formal certificate); `lip_ub_S` and `lip_ub_T` retained as decomposition diagnostics. Sparse routing, orthogonality, EMA load balance, liveness, and `lip_ub_F<1` are soft diagnostic goals. |
| Near queue | iter153 (reverse-KL balance) is default-on (commit `ea236bc`); iter155 (Lyapunov-on-$F$ + `lip_ub_F` diagnostic on the actual two-state Parcae cycle) is the next contraction-targeted test. iter154 (inverse-popularity per-expert NTP) and iter156 (hard $\bar A$ projection governor) are escalation-only — run only if the corresponding iter153/iter155 measurement leaves the relevant invariant open. Rescue coefficient escalation remains stopped after iter150. Iter152 conditional prefix-K anchors are the BPB-winning reference on the iter149 coefficient base: final full `1.471820` vs iter149 `1.478698`. Do not retry pure IFT; if gradient-quality is revisited, redesign it as a hybrid finite-K/IFT estimator with a fresh hypothesis. |
| Heavy queue | After the routing/contraction queue clears, the active heavy items are H94 GPTQ+LQER (direct quant-tax attack, `--use-gptq=1`, `--use-lqer=1`), H96 grouped artifact compression (frees budget; `--use-grouped-artifact-compression=1`), H91 phased test-time training (`--use-ttt-eval=1`). H95 SP8192+CaseOps depends on H96 budget relief. H99 SmearGate (`--use-smear-gate=1`), iter118b sparse attention head gate (`--use-sparse-attn-head-gate=1`), and iter120 RRAttention (`--use-rr-attention=1`) are held until a measurement-driven trigger condition fires (no current motivation). Each component is removable by deleting its module plus the narrow `train_gpt.py` hook if its isolated iteration fails. |

## Tested Iterations

| Iteration | Change tested | Result | Takeaway | Follow-up |
|---|---|---|---|---|
| H12 | K-jitter for fixed-point quality. | `VERIFIED` | Train across solve depths; fixed K encourages finite-K overfitting. | Keep K-jitter permanent. |
| H35/H36 | Beta jitter vs fixed high beta. | `VERIFIED` / fixed high beta unsafe | Solver robustness needs damping diversity; fixed beta 0.7 breaks RevDEQ reconstruction. | Parcae supersedes scalar beta in default path; keep beta jitter only for non-Parcae fallback. |
| H67 | Disable Lyapunov hinge under Parcae. | `PROMOTED` | Explicit Jacobian hinge was redundant/noisy under Parcae damping. | Keep `lyapunov_coef=0.0`; use probes as diagnostics, not default loss. |
| H68 | Disable HyDRA finite-perturbation denoising. | `PROMOTED` | Extra denoising forward did not justify cost under the current damping regime. | Keep `denoising_coef=0.0`. |
| H60 | Disable CTP head. | `PROMOTED` | NTP-only path is the active default; CTP remains an explicit ablation path. | Keep `use_ctp=False`, `ctp_weight=0.0`. |
| H64 | Disable BigramHash. | `PROMOTED` | Capacity is better spent in the RevDEQ/expert body than a hash table under the current architecture. | Keep `bigram_vocab_size=0`. |
| H71 | More smaller full-D experts at iso-cost. | `PROMOTED` | Full-D LoRA-style expert scaling is preferred over bottleneck expert scaling. | Keep E=16 with current ranks; do not revive bottlenecks without a new reason. |
| H69/H70 | Bottleneck experts. | `NOT PROMOTED` | Bottlenecks are capacity-bound and inefficient at the tested scale. | Closed as a forward scaling path. |
| H74/H75 | Sparsemax / alpha=1.5 entmax routing. | `NOT PROMOTED` | Harder architectural sparsity carried a quality cost in this dense-soft MoE. | Only revisit through smooth/default-off sparsity mechanisms. |
| H87 / iter 117 | Adaptive entmax alpha + padded compute-skip. | `PROMOTED` | Strict-gen routing-function infrastructure can work when default-off and guarded. | Keep as optional path; do not make it the default without a fresh win. |
| H87b / iter 117b-1 | 10x router entropy bump. | `NOT PROMOTED` | Entropy magnitude alone does not overcome CV redistribution in soft-dense MoE. | Prefer joint/diversity mechanisms over raw entropy increases. |
| H84+H93 / iter 112+122 | Gram expansion routing + logit softcap. | `PROMOTED` historically | Softcap remains useful; gram-routing conclusions are now superseded by flattened loss-stack evidence. | Keep softcap; route new diversity tests through iter 142b/142c. |
| H63 / iter 95 | `deq_bptt_k: 2 -> 3`. | `PROMOTED` | One more backward step improved gradient quality within the reversibility budget. | Current default is `deq_bptt_k=3`; iter 143 tests `4`. |
| iter 131 | Deep K-jitter `(32,48)` under dense routing. | `NOT PROMOTED` | Deeper forward solve is too expensive when routing remains dense. | Retry only after effective experts are sparse enough to pay for K. |
| iter 133 | Promoted routing baseline before the iter-142 refactor. | `PROMOTED`, but dense | Good BPB anchor, but per-token entropy was near maximum, so sparse-kernel ROI was absent. | Use as historical anchor, not as the active loss design. |
| iter 138/138a | Unified legacy routing regs at coef 0.1 and drop-gram ablation. | `SUPERSEDED` | The old coefficient stack and gram analysis were confounded/stale after loss flattening. | Fold lessons into 142b/142c; do not run the old matrix. |
| iter 140/141 | Output-space/expert-output gram direction. | `SUPERSEDED` as standalone | Expert-output diversity remains the useful part, but legacy routing-gram plumbing was removed. | Test diversity through cosine vs Frobenius in 142b/142c. |
| iter 142-fixes | Seven review-item fixes: hot-path syncs, wallclock naming, optimizer arity, validation invariants, logging labels. | `OBSERVED` behavior-neutral | Fixes passed smoke/focused tests and form a cleaner baseline surface. | Keep; compare behavior through iter-142-refactor. |
| iter 142-refactor | Flatten routing/MoS/diversity loss stack; direct coefficients; cosine default; remove active routing gram. | `VERIFIED` at 100 steps | Refactor improved 100-step int6 BPB by about 0.075 and made the objective legible. | Active baseline; needs longer-run confirmation and 142b/142c counterfactual. |
| iter 142b | Decomposed canonical stack at 1000 steps: cosine `expert_output_diversity_coef=1.0` + default router CV/entropy + MoS CV; `regularizer_warmup_frac=0`. | `PROMOTED` (user override 2026-05-06) | Result: val_bpb 1.5009 full / 1.5112 post-int6; Δ +0.0079 vs c94899a-era baseline 1.4930 (full), +0.0125 at K=128.<br>• BPB regression is intentional: the prior baseline ran on the pre-iter-142-refactor stack with the legacy nested gram penalty (`attn_ortho 0.141` vs 142b's 0.026) — not a comparable optimization path.<br>• Diagnostic gate flagged `router_collapse` (attn_min_share 0.0135 vs fair 0.0667); attn_cv peaked 2.68 (s100), finished 0.72 (vs baseline 0.35); mlp_cv 0.18 (vs 0.25). Shared-expert sigmoid gate 0.52 (vs 0.22).<br>• K-sweep: K=24 best at 1.5063; Δ(K=8→K=64)=−0.110, Δ(K=16→K=64)=−0.004 (well-converged at K≥16). Acyclicity primes 17/37/113 match neighbours within ±0.001.<br>• Step time 23.46 s (+3.2 % vs baseline 22.73 s). Artifact 7.65 MB. | Promoted as default. See Tested-Iterations notes above for full evidence; the verdict-side summary:<br>• **Defaults updated** (all five): `router_load_cv_coef 0.5 → 1.0`, `mos_load_cv_coef 0.25 → 1.0`, `router_pertoken_entropy_coef 0.00125 → 1.0` (renamed from `router_entropy_coef`), `expert_output_diversity_coef 0.1 → 1.0`, `regularizer_warmup_frac 0.10 → 0.0`.<br>• **Loss-form change**: `relu(cv − cv_target)²` → continuous `cv²`; `cv_target` and `mos_cv_target` knobs removed.<br>• **Strict-generalization deviation**: `cv²` is NOT a strict generalization of `relu(cv − τ)²` — promotion forced under user-override, not the auto-promote rule.<br>• **Cold-start deviation**: `regularizer_warmup_frac=0` with `router_pertoken_entropy_coef=1.0` deviates from the CLAUDE.md anneal-from-zero directive; kept because no s≤100 spike was observed.<br>• **Re-litigation gate**: revert to hinge + warmup if any iter at K≥16 shows ≥ +0.005 BPB regression attributable to entropy specialization beyond what diversity catches. |
| iter 142c | Frobenius-only counterfactual at 1000 steps: `expert_output_diversity_coef=1.0` Frobenius + `router_load_cv_coef=0`, `router_pertoken_entropy_coef=0`, `mos_load_cv_coef=0`; `regularizer_warmup_frac=0`. | `NOT PROMOTED` | Full val_bpb 1.5143, post-int6 1.5245 (+0.0213 vs c94899a-era baseline; +0.0134 worse than 142b). Diagnostic gate flagged `router_collapse + expert_collapse ×2`. attn_cv ≤ 1.70 throughout (peaked at s100, settled at 0.65); MLP side worse than 142b (mlp_cv 0.20 vs 0.18). K-sweep: K=24 best at 1.5233; Δ(K=8→K=64)=−0.067, Δ(K=16→K=64)=+0.000 (well-converged at K≥16). Acyclicity prime K=113 shows +0.004 bump from K=64 (small acyclicity artifact, not seen in 142b). Step time 23.45 s (+3.2 % vs baseline). Artifact 7.61 MB. | Confirms decomposed cosine (142b) > Frobenius-only (142c) by +0.013 BPB at 1000 steps. Frobenius-alone path closed; remove from default consideration. |
| iter 142d | Doubled-coef variant queued in pre-override defaults: `expert_output_diversity_coef=2.0`, `router_load_cv_coef=1.0`, `router_pertoken_entropy_coef=0.0025`, `mos_load_cv_coef=0.5`, `regularizer_warmup_frac=0`. | `SUPERSEDED` (2026-05-06) | Never run — 2026-05-06 user override set defaults to uniform 1.0, which is the same end-state 142d aimed at exploring. iter 143 (`deq_bptt_k=4`) runs on the uniform-1.0 baseline and supersedes the standalone 142d sweep. | Closed; do not run as a standalone iter. |

## Remaining Queue

| Priority | Item | Change | Principled? | Efficient to test? | Merge/test decision |
|---|---|---|---|---|---|
| Done | iter 142b | Cosine expert-output diversity + router CV + router entropy + MoS CV. | Yes. | 1000 steps, 6.51 h, 23.46 s/step. | **PROMOTED** (user override 2026-05-06). Defaults updated to match. |
| Done | iter 142c | Frobenius expert-output diversity alone; CV/entropy/MoS-CV all zero. | Yes: counterfactual to 142b. | 1000 steps, 6.51 h, 23.45 s/step. | `NOT PROMOTED`. Frobenius-alone path closed. |
| Deferred | iter 103a | Chained expert routing, `split_2stage`: split the current expert-module budget across two mixed attn+MLP stages, with each stage owning 7 routed + 1 shared attention expert and 7 routed + 1 shared MLP expert; stage 2 consumes the residual-updated stage-1 state. | Yes: tests sequential routing composition while preserving the active expert-module budget and ensuring every stage has a shared expert path. | 1000 steps, 8.33 h, 30.00 s/step. | Best chained variant so far, but deferred vs non-chained iter142b. Half per-token entropy passed: final fast `val_bpb=1.5489`, roundtrip `1.5880`, K-sweep best `K=17 val_bpb=1.5875`, peak VRAM `30758 MB`, artifact `8.01 MB`; final validation `chain_ema_expert_min_share=0.1230`, `chain_router_batch_min=0.1168`, entropy `0.0013`; post-int6 health passed. Non-chained iter142b remains much better (`1.5112` post-int6 K16, best K24 `1.5063`) and faster (23.46 s/step vs 30.00 s/step). Rerun chained only after higher-ROI architecture tests. |
| Done | iter 103b | Chained expert routing, `attn_first_2stage`: first stage owns attention experts only, second stage owns MLP experts only, with `--router-pertoken-entropy-coef=0.5`. | Yes: standard transformer order inside the DEQ transition. | 1000 steps, 7.57 h, 27.25 s/step. | `NOT PROMOTED` vs 103a: final `val_bpb=1.5532` vs 103a `1.5489`; roundtrip `1.5936`, K-sweep best `K=17 val_bpb=1.5929`, peak VRAM `35040 MB`, artifact `7.94 MB`. Early BPB lead disappeared by step 800. Health weaker: final `chain_ema_expert_min_share=0.0366`, `chain_router_batch_min=0.0288`, `chain_router_cv=0.1437`; post-int6 gate failed `mlp_min_share=0.0383 < 0.0400` and `mlp_ortho=0.5117 > 0.5`. |
| Deferred | iter 103c | Chained expert routing, `mlp_first_2stage`: first stage owns MLP experts only, second stage owns attention experts only, with `--router-pertoken-entropy-coef=0.5`. | Yes: order counterfactual to attn-first. | OOM at step 810 on 44 GB dev GPU. | `NOT PROMOTED` for now and deferred: step-800 fast `val_bpb=1.6308`, worse than 103a `1.5965` and 103b `1.6021` at the same checkpoint; liveness alive but weaker than 103a (`chain_ema_expert_min_share=0.0610`, batch min `0.0469`, CV `0.1119`). OOM occurred shortly after step-800 validation/Hutchinson with rank0 allocator fragmentation and retained saved FP tensors; code now clears validation FP tensors and writes resumable checkpoints every 200 steps. Rerun later only if chained routing becomes relevant after other architecture gains. |
| Deferred | iter 103e | Four-stage alternating typed chain: `attn -> mlp -> attn -> mlp`, with each stage owning 7 routed + 1 shared expert of that stage's type. | Yes: tests whether repeated transformer-order composition gives useful intermediate states while preserving the 32 expert-module budget and reducing per-stage peak VRAM. | Requires preset implementation; run after higher-ROI architecture tests. | Defer. Must keep at least one shared expert in every stage and persistent per-stage EMA liveness. Compare against 103a mixed split and non-chained iter142b, not only against 103b/103c. |
| Deferred | iter 103f | Four-stage alternating typed chain: `mlp -> attn -> mlp -> attn`, with each stage owning 7 routed + 1 shared expert of that stage's type. | Yes: reverse-order counterfactual to 103e; tests whether MLP-first transformations make later attention stages more effective in a deeper chain. | Requires preset implementation; run after higher-ROI architecture tests. | Defer. Same health gates as 103e; only worth continuing if it improves over 103a or materially improves VRAM/step-time without BPB damage. |
| Deferred | iter 103d | Chained-router health fix: per-stage/per-health-slice alive hinge `sum relu(tau - normalized_usage)^2`, delayed router sparsity/entropy ramp, and per-stage minima diagnostics. | Yes: zero alive loss directly implies no dead expert in every staged router/slice; sparsity remains token-local and input-dependent. | Not efficient now. | Defer behind stronger architecture work: no chained preset beat the non-chained baseline, so health-fixing chained routing is lower ROI than `iter145`. |
| Done | iter 145 | Evidential Dirichlet-UCB router + EMA-anchored GJSD-style routing objective: `router_scoring=dirichlet_ucb`, `router_dirichlet_ucb_beta=0.5`, `router_load_cv_coef=0`, `router_ema_balance_coef=0.1`, `router_ema_specialization_coef=0.1`, other active reg targets `0.1`, `regularizer_warmup_frac=0.07`, and `use_router_sigmoid_gate=0`. | Yes with caveat: routing remains token-local; persistent EMA is detached, so the balance loss uses a straight-through EMA anchor to give current routing a gradient while tracking historical usage. Normalized expert-output Gram/cosine remains the principled orthogonality pressure. | 1000 steps, 6.50 h train, 23.40 s/step, peak VRAM 34.86 GB. | `VALIDATED_WITH_TECH_DEBT`, not promoted as-is. BPB improved over iter142b: fast `1.4642` vs `1.4715`, final full `1.4840` vs `1.5009`, K-sweep best `K=24 1.4918` vs `1.5063`. Router/MoS health improved versus 142b and no dead expert was observed during training, but post-int gate failed `attn_min_share=0.0319 < 0.0400`, `mlp_ortho=0.5312 > 0.5`, and `lip_ub=65.3672 >= 1`. Root cause: router/liveness fix works for BPB but does not control the RevDEQ transition Jacobian; later contraction work must target `T_theta` directly. |
| Done | iter 145r | Promoted Dirichlet-UCB + EMA-balance routing defaults: `router_scoring=dirichlet_ucb`, `router_dirichlet_ucb_beta=0.5`, `router_load_cv_coef=0`, `router_ema_balance_coef=0.15`, `router_ema_specialization_coef=0.1`, `router_pertoken_entropy_coef=0.1`, `mos_load_cv_coef=0.15`, `expert_output_diversity_coef=0.15`, `regularizer_warmup_frac=0.07`, `use_router_sigmoid_gate=0`, `weight_decay=0.015`. | Yes with documented tech debt: routing remains token-local; EMA balance directly improves historical usage and BPB, but KL-to-uniform alone does not guarantee a hard minimum share. Scalar `deq_beta` is not a Parcae contraction fix because `lip_ub` probes `T_theta`, not the solver blend. | 1000 steps, 6.50 h train, 23.41 s/step, peak VRAM 34.86 GB. | **PROMOTED_WITH_TECH_DEBT** by user directive 2026-05-08. BPB improved: final full `1.4818` vs iter145 `1.4840` and iter142b `1.5009`; fast step1000 `1.4567`; roundtrip/K16 `1.4909`; K-sweep best `K=24 1.4899`. Open issues: strict post-int `attn_min_share=0.0371 < 0.0400`; output collinearity `attn_ortho=0.5098`, `mlp_ortho=0.5703`; contraction failure `lip_ub=45.7943` at K128. Promotion reason: task BPB win and no true dead expert during training. Tech-debt fix is queued separately. |
| Done | iter146 rescue | Behavior-changing rescue on top of promoted `iter145r`: deterministic weighted K bag `{16,24,32,64}` with weights `(0.50,0.40,0.07,0.03)`, EMA alive hinge, stronger EMA balance/specialization, worst-pair cosine expert-output diversity, full-final-validation metadata, and router-confidence diagnostics. | Yes, with attribution caveat: this is a targeted failure-mode bundle, not an attribution-clean mechanism test. | 1000 steps, 6.78 h train, 24.41 s/step. | `NOT PROMOTED` for strict health, though BPB is competitive. Fast step1000 `val_bpb=1.4575`; final full `1.479157`; K-sweep best `K=24 1.484500`, `K=128 1.485336`; sampled K counts `[16:500,24:400,32:70,64:30]`. Post-int gate failed `router_collapse` (`attn_min_share=0.0341 < 0.0375`) and `local_contraction_failed` (`lip_ub=21.5896 >= 1`). Takeaway: K/liveness/orthogonality rescue preserved BPB and improved contraction vs iter145r, but not enough for the strict contraction gate. Run the one-variable Lyapunov contraction ablation next; do not combine the router liveness floor with iter147 unless contraction is first isolated. |
| Done | iter147 contraction ablation | Same iter146 baseline plus only the low-cadence Lyapunov transition expansion loss: `lyapunov_coef=0.005`, `lyapunov_gamma=0.97`, `lyapunov_every=16`, `lyapunov_max_tokens=64`. | Yes: one-variable test of direct transition-map contraction pressure. | 1000 steps, 6.79 h train, 24.45 s/step. | `NOT PROMOTED` for strict health, but the hypothesis is partially supported. Fast step1000 `val_bpb=1.4582`; final full `1.480286`; K-sweep best `K=24 1.486178`, `K=128 1.487220`; post-int gate failed `router_collapse` (`attn_min_share=0.0320 < 0.0375`) and `local_contraction_failed` (`lip_ub=12.5324 >= 1`). Compared with iter146, K128 `lip_ub` improved `21.5896 -> 12.5324` while final full BPB regressed only `+0.001129`, so iter148's coefficient-only escalation condition is met. |
| Done | iter148 contraction rescue | Same as iter147, but only `lyapunov_coef` increased `0.005 -> 0.0075`; routing, K-jitter, weight decay, and all other knobs fixed. | Yes: one-axis coefficient response after the first direct transition penalty moved `lip_ub` in the right direction. | 1000 steps, 6.78 h train, 24.42 s/step. | `NOT PROMOTED`; escalation refuted. Fast step1000 `val_bpb=1.4590`; final full `1.481556`; K-sweep best `K=24 1.488734`, `K=128 1.490097`; K counts `[16:500,24:400,32:70,64:30]`. Post-int gate failed `router_collapse` (`attn_min_share=0.0366 < 0.0375`) and `local_contraction_failed` (`lip_ub=23.9952 >= 1`). Compared with iter147, BPB regressed and K128 `lip_ub` worsened `12.5324 -> 23.9952`; Lyapunov escalation is closed. |
| Deferred | iter 143 | `deq_bptt_k=4` counterfactual. Original queue target was iter142b uniform-1.0 defaults; if revived, explicitly choose whether to run on the promoted iter145r stack or the historical iter142b stack. | Yes: one extra Neumann/VJP term after K=3 promoted. | Full 1000 steps. | Deferred by user directive. |
| Done | iter 144 | Pure IFT adjoint gradient for the truncated input/embedding signal. | Refuted for the current stack: the pure equilibrium estimator is mismatched to the noncontractive finite-K transition map, and the implementation removes `z_init` and Parcae-beta task-gradient paths that TBPTT preserves. | 1000 steps, 5.59 h train, 20.12 s/step, peak VRAM 30.28 GB. | `NOT PROMOTED`; pure IFT-2 refuted. Fast step1000 `val_bpb=2.0268`; final full `2.033712`; K-sweep best `K=8 2.028207`, `K=128 2.088864`; K counts `[16:500,24:400,32:70,64:30]`. Post-int failed K64/K128 degradation vs best and `local_contraction_failed` (`lip_ub=32.7620 >= 1`). Throughput improved about 18% vs TBPTT, but BPB regressed by over 0.55. Pure IFT code and CLI support were removed; any revisit needs a new hybrid design. |
| Done | iter149 combined health double | Cumulative health-first rescue from iter146: double all direct observed-problem coefficients together: `router_ema_balance_coef 0.30 -> 0.60`, `router_ema_specialization_coef 0.20 -> 0.40`, `router_pertoken_entropy_coef 0.10 -> 0.20`, `expert_output_diversity_coef 0.30 -> 0.60`; keep `lyapunov_coef=0`. `router_ema_alive_coef 0.02 -> 0.04` is a guard for the strict min-share failure, not evidence of true dead experts. | Yes: each active coefficient maps to an observed failure axis without adding a direct `lip_ub` penalty; alive is tracked separately as a persistent-underuse guard because iter146's alive raw loss was zero. | 1000 steps, 6.78 h train, 24.41 s/step, peak VRAM 34.86 GB. | `CURRENT_BEST_BY_BPB`; continue rescue. Final full `val_bpb=1.478698`, a small win over iter146 `1.479157`; fast step1000 `1.4569`; K-sweep best `K=24 1.484021`, `K=128 1.485084`; K counts `[16:500,24:400,32:70,64:30]`. Soft diagnostic debt remains: post-int reported `attn_min_share=0.0307 < 0.0375` and `lip_ub=35.3658`; step1000 soft metrics `attn_cv=0.3438`, `mlp_cv=0.0848`, `pool_cv=0.2504`, `pertoken_entropy=3.2304`, `attn_ortho=0.0283`, `mlp_ortho=0.2051`. Do not reject the BPB win on these; use them to run iter150 with the active problem-loss coefficients doubled again. |
| Done | iter150 combined problem-loss double x2 + K32-200 | Cumulative on iter149: doubled the observed-problem coefficients again: balance `1.20`, specialization `0.80`, per-token entropy `0.40`, expert diversity `1.20`; kept `router_ema_alive_coef=0.04`, `router_load_cv_coef=0`, and `lyapunov_coef=0`. Changed the K-jitter mix to weights `(0.43,0.34,0.20,0.03)`, targeting counts `[16:430,24:340,32:200,64:30]`, so K=32 received 200/1000 training steps with lower average K than the aborted K32-300 plan. | Yes, but confounded: it jointly tested coefficient escalation and a targeted finite-depth distribution change. | 1000 steps, 7.11 h train, 25.59 s/step, peak VRAM 34.87 GB. | `NOT PROMOTED` by the validation-BPB hard gate. Final full `val_bpb=1.501195`, worse than iter149 `1.478698`; fast step1000 `1.4765`; roundtrip/K16 `1.510464`; K-sweep best remained `K=24 1.504070`, with `K=32 1.504192`, `K=64 1.504609`, `K=128 1.504683`; K counts `[16:430,24:340,32:200,64:30]`. K32-200 is not the better training distribution. Soft diagnostics were mixed: entropy and orthogonality improved (`pertoken_entropy=3.1821`, `attn_ortho=0.0219`, `mlp_ortho=0.2002`) and K128 `lip_ub=18.6603` improved vs iter149, but EMA/current load worsened (`attn_ema_min=0.0534`, `attn_ema_cv=0.2309`, `pool_ema_min=0.0276`, `pool_ema_cv=0.1749`, `attn_cv=0.3779`, `pool_cv=0.2758`). Revert the K mix for iter151 and continue the coefficient-dose stress test once more before selecting the BPB winner. |
| Aborted | iter151 combined problem-loss double x3 + K rollback | Planned coefficient stress test: revert K weights to `(0.50,0.40,0.07,0.03)` and double active observed-problem coefficients to balance `2.40`, specialization `1.60`, per-token entropy `0.80`, expert diversity `2.40`. | Principled as a stress test, but lower priority after iter150 showed diagnostic-first over-regularization and user pivoted to queue clearing. | Aborted at step 10; not comparable and not counted. | Do not use as evidence. If rescue is revisited later, require a new BPB-first reason rather than continuing automatic coefficient doubling. |
| Done | iter152 conditional prefix-K multi-anchor | Use the iter149 BPB-winning coefficients and replace single-endpoint jitter supervision with conditional prefix anchors: sample the usual `K` from `{16,24,32,64}` with weights `(0.50,0.40,0.07,0.03)`, store endpoints only for jitter depths `<= sampled K`, and apply normalized losses/TBPTT tails to those endpoints. Expected anchors per step are `0.50*1 + 0.40*2 + 0.07*3 + 0.03*4 = 1.63`, while expected forward K remains `21.76`. | Yes: same finite-K objective family, but lower-variance multi-depth supervision for all prefix endpoints actually traversed by the sampled solve. It is not a forced `K=64` pass and does not change the K sampling distribution. | 1000 steps, 8.26 h train, 29.74 s/step, peak VRAM 36.67 GB. | `PROMOTED_WITH_TECH_DEBT` by the validation-BPB hard gate. Fast step1000 `val_bpb=1.4489`; final full `1.471820` vs iter149 `1.478698`; roundtrip/K16 `1.480791`; K-sweep best `K=24 1.479684`, `K=128 1.481126`; sampled K counts `[16:500,24:400,32:70,64:30]`. Late small pre-clip `grad_norm` (`~0.026-0.070`) was a stability sign, not a failure: task loss and validation kept improving. Soft diagnostics were mixed: post-int gate failed `router_collapse` (`attn_min_share=0.0315 < 0.0375`) and `local_contraction_failed` (`lip_ub=22.2777 >= 1`); final step metrics `attn_ema_min=0.0342`, `mlp_ema_min=0.0619`, `pool_ema_min=0.0177`, `pertoken_entropy=3.1635`, `attn_ortho=0.0376`, `mlp_ortho=0.1934`. Treat prefix anchors as the BPB base; if the next iteration targets diagnostics, start from the generated `retry_hint.json` rather than continuing coefficient doubling. |
| Done | iter158 reverse-KL on corrected baseline (2026-05-14) | Bundled retest replacing iter153/iter155/iter157: drop Lyapunov entirely (refuted by Tier 1+2 redesign as targeting wrong condition), inherit corrected defaults (`deq_prefix_anchors=True`, `reverse_kl_balance=True`, `lyapunov_coef=0`), test reverse-KL as one-variable change vs iter152 on the now-correct baseline. Extended K-jitter set to `(16,24,32,64,96,128)` per user directive (low-frequency tail at K=96 and K=128). First-ever empirical `rho_F` measurements (Tier 2 spectral-radius probe). | Yes: clean test of the iter153 hypothesis. Eliminates the prefix-anchors confound that handicapped iter153/iter155. K-sweep emits new gate-relevant `rho_F` (necessary AND sufficient for asymptotic FP convergence by Hartman-Grobman). | 1000 steps, ~9.0 h train (~32 s/step due to extended K-jitter compile cost), peak VRAM 36.50 GB. K-sweep adds ~8 JVPs per K for rho_F probe. | `NOT_PROMOTED` by validation-BPB hard gate. **K=16 final_full_validation val_bpb=1.475130** (Δ +0.0033 vs iter152 1.471820). Per user directive Option-C re-eval: K=24 final_full_validation **val_bpb=1.474375** (Δ +0.0026 vs iter152 — still NOT-PROMOTE; K=24 only Δ -0.0008 better than K=16). K-sweep best K=24=1.4814 (essentially TIED with iter152 K=24 1.4815 on subset, but full val gap remains). K=128 lip_ub_F=15.07; K=128 **rho_F=0.86 (PASS gate < 1)**. **Routing health: only ONE post-int failure** (router_collapse: attn_min_share=0.0327 < 0.0375); expert_collapse + local_contraction_failed both PASS (vs iter155's 3 failures). Reverse-KL hypothesis CONFIRMED principled but BPB-neutral on the corrected baseline. Empirical landmark: first ρ(J_F) < 1 ever measured in project (lip_ub_F=15 vs rho_F=0.86 = 17× σ-vs-ρ gap, validating Tier 1+2 framework). Routing balance (attn_min_share) is the actually-remaining bottleneck → iter162 (doubled coefs, simple) or iter160 (per-expert NTP, more code) is the principled next step. Logs at `experiments/training_logs/iter158_revKL_cleanbaseline_20260513.log` + `iter158_kbest24_fullval_20260514.log`. Checkpoints at `experiments/checkpoints/iter158_revKL_cleanbaseline_20260513_1602/{step_000900,step_001000,latest}.pt`. |
| Done | iter155 Lyapunov-on-F + lip_ub_F diagnostic (corrected 2026-05-13) | Architecture-agnostic empirical-tier contraction work: redirect Lyapunov FD probe from `T_θ` (iter147) to the actual two-state Parcae cycle map `F`. New flag `lyapunov_target=iteration_F` un-detaches Ā in the FD branch so gradients flow to Parcae damping AND transition parameters jointly. K-sweep emits 3-way decomposition: `lip_ub_T` (transition diagnostic), `lip_ub_S` (single-state advisory surrogate), `lip_ub_F` (gate-aligned operator-norm probe). New post-int gate triggers on `lip_ub_F >= 1`. Joint cycle residual `fp_residual_F` paired with `lip_ub_F` in `fp_bound`. Mid-run code patch (per user directive 2026-05-13): `lip_ub_F` and `fp_residual_F` made always-on at every FP eval (fast-val + every K-sweep row), no longer gated by `fp_lip_fast_val_every` or `EvalProfile.lip_probe_set` — patch landed AFTER iter155 completed so iter153/iter154 onward gain it. | Yes: principled fix for the iter145r/149/152 contraction debt. Penalty pressures `σ_max(J_F)` directly via cross-block coupling `D_β·J_T`. **Confirmed at the gradient level**: per-dim Ā migrated 0.700 → 0.481 mean / 0.333 min over 1000 steps (β = 1−Ā: 0.300 → 0.519 mean = 1.7× more damping). | 1000 steps, 6.81 h train, 24.43 s/step, peak VRAM 34.92 GB. **18% faster than iter152** (which carried `deq_prefix_anchors=1` overhead). | `NOT_PROMOTED` by validation-BPB gate. **final_full_validation val_bpb=1.481344 vs iter152 1.471820 (Δ +0.0095, regression)**. K-sweep: K=4=1.8633, K=8=1.5756, K=16=1.4886, K=17=1.4874, K=24=**1.4861 (best)**, K=32=1.4867, K=37=1.4869, K=64=1.4873, K=113=1.4875, K=128=1.4875. iter152 K=24 best=1.4797 → iter155 +0.0064 worse. Post-int gate failures (3): `router_collapse` (attn_min_share=0.0362 < 0.0375 — slightly better than iter152's 0.0315 but still below gate), `expert_collapse` (mlp_ortho=0.5098 > 0.5 — markedly worse than iter152's 0.1934), `local_contraction_failed` (`lip_ub_F=17.08, lip_ub_S=16.07, lip_ub_T=25.84` at K=128 — gate-aligned object never fell below 15 across the K-sweep, oscillating 15-20). Train_loss had a consistent **−0.27 lead vs iter152** across the entire trajectory (s80→s940), but did NOT translate to val_bpb: iter155 val_bpb trajectory s200→s400→s600→s800: `1.9791 → 1.6405 → 1.5375 → 1.4969`; iter152 `1.9942 → 1.6298 → 1.5346 → 1.4942` (gap was −0.015 at s200, flipped to +0.011 at s400, settled at +0.003 by s600). Lyapunov-on-F at coef=0.005 with random_fd estimator successfully trained Ā but: (a) the increased damping moved the model into a different optimization regime that hurt val_bpb generalization, (b) random_fd directional probe is not a tight bound on `σ_max(J_F)` — the K-sweep operator-norm power-iteration estimate kept reading 15-20 throughout. **Aug-Lagrangian escalation BLOCKED** by BPB regression > +0.005 stop condition. Soft-pressure tier exhausted. iter156 (hard governor 156a or learned `c·Δ` gain 156b — formal-tier mechanism) is the next required step. Artifact 8.08 MB (well under 16 MB). Log: `experiments/training_logs/iter155_1000step_20260513.log`. Checkpoints: `experiments/checkpoints/iter155_1000step_20260513_0135/{step_000900,step_001000,latest}.pt`. |
| Done | iter155 Lyapunov-on-F + lip_ub_F diagnostic | See Tested Iterations row 92. **NOT PROMOTED**: val_bpb 1.481344 vs iter152 1.471820 (Δ +0.0095); 3 health gates fail (router_collapse 0.0362, expert_collapse 0.5098 mlp_ortho, local_contraction_failed lip_ub_F=17.08). Soft-pressure tier exhausted; iter156 escalation required. **CONFOUND DISCOVERED 2026-05-13**: iter155 inadvertently ran with `deq_prefix_anchors=False` (Hyperparameters default never propagated after iter152's promotion) — so iter155 vs iter152 actually mixed `+Lyapunov-on-F` (target intervention) with `−prefix-anchors` (silent regression of iter152's promoted feature). Estimated train_loss penalty from −prefix-anchors alone: ~−0.27 (consistent across iter153 and iter155 trajectories vs iter152). The +0.0095 BPB regression is mixed between "Lyapunov-on-F hurt" and "lost prefix-anchors benefit". Default flipped to `True` 2026-05-13; iter157 onward will inherit the fix. |
| 1.5 | **iter152 K-sweep measurement on saved checkpoint** *(NEW 2026-05-13, scheduled between iter153 and iter157)* | Resume from `experiments/checkpoints/iter152_prefix_1000/step_001000.pt` and run 1 step + full K-sweep with the iter155-corrected diagnostic suite (lip_ub_T/S/F, fp_residual_F always-on per commit `e7e02c6`). iter152's training log only has the legacy `lip_ub:` field (= `lip_ub_T`); the gate-aligned `lip_ub_F` was never measured for the iter152 baseline. This run gives the missing reference point. | Yes: makes the iter155 → iter152 contraction comparison decisive (currently we have iter155 lip_ub_F=17.08 but no iter152 baseline number to compare against). Resolves the speculative "iter152's lip_ub_F is probably 12-17 by extrapolation" with an actual measurement. | ~10-15 min on 2× L40S with 1 train step + K-sweep. Resume from saved checkpoint, set `--iterations=1001` (resume at step 1000, do 1 more step to populate FP, then K-sweep fires). | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1001 --resume-from=experiments/checkpoints/iter152_prefix_1000/step_001000.pt --eval-profile=diagnostic --val-loss-every=1000000 --run-id=iter152_kresweep_20260513`. Output: K-sweep table with `lip_ub_F` per K. No promotion implied — pure measurement run. |
| **1** | **iter153 reverse-KL balance** *(running, launched 09:28)* | Swap the EMA balance loss form from forward `KL(EMA‖U)` to reverse `KL(U‖EMA)`. Forward KL weights each per-expert term by `EMA_i`, so a dead expert's contribution is `EMA_i · log(EMA_i/U_i) → 0` even when the log ratio is large — the gradient on the tail vanishes. Reverse KL weights by `U_i`, so each expert contributes equally regardless of its mass, and the dead-expert gradient is `−U_i/EMA_i`, diverging as `EMA_i → 0`. One-line code change at `train_gpt.py:2856` gated by `use_reverse_kl_balance` (Hyperparameters default True, iter153 default-on). Provisional default-on for testing; reverts to forward KL if iter153 regresses BPB. | Yes: principled fix for the asymmetric tail weighting diagnosed in the iter152 post-int min-share failure (`attn_min_share=0.0315 < 0.0375`). For the diagnosed numbers (E=16, one dead expert at half its fair share), reverse KL gives ~3.4× stronger gradient than forward KL; the gap widens as EMA → 0. No new coefficient; existing `router_ema_balance_coef=0.30` carries the same magnitude budget. | 1000 steps, ~6.78 h train, ~24.4 s/step, peak VRAM ~34.86 GB (iter146 baseline; reverse KL adds <1% compute — one extra reciprocal per expert per step). | Run with no extra flags; `--use-reverse-kl-balance=0` recovers the forward-KL baseline for direct A/B. Promotion condition: full val_bpb ≤ iter152's `1.471820` AND post-int `attn_min_share ≥ 0.0375`. If BPB regresses, revert `Hyperparameters.use_reverse_kl_balance` to False and re-queue. If BPB ties and min-share improves, promote and document the policy match. **Promoted to Pri 1 on 2026-05-13** (was Pri 2): iter155's contraction A/B closed as NOT-PROMOTED, so iter153 (the routing-balance fix) becomes the next BPB-promotion candidate. iter156 (formal-tier hard governor) is the principled escalation for the contraction debt iter155 confirmed. |
| Closed | iter157 (Lyapunov-on-F with power_jvp_F estimator) | **Closed 2026-05-13**: superseded by iter158 bundled rerun, then closed entirely after iter152 K-sweep measurement empirically refuted soft Lyapunov as principled. The principled framework (Tier 1+2 redesign) targets ρ(J_F) < 1, NOT σ_max < 1 — making the Lyapunov-on-F penalty over-restrictive regardless of estimator quality. No iter157 needed. |
| **2** | **iter158 reverse-KL on corrected baseline** *(HIGH PRIORITY 2026-05-13, replaces iter157 + iter153-rerun)* | Bundled retest of iter153's reverse-KL hypothesis on the corrected baseline (deq_prefix_anchors=True default after promotion-propagation fix). NO Lyapunov penalty — refuted by iter155 evidence + Tier 1+2 theoretical analysis (σ_max is over-restrictive proxy; ρ is the necessary-AND-sufficient condition). Inherits all defaults: deq_prefix_anchors=True, use_reverse_kl_balance=True, lyapunov_coef=0. Tests reverse-KL as a one-variable change vs iter152. Tier 1+2 K-sweep emits new gate-relevant metric `rho_F` (spectral radius via straight power iteration on J_F) alongside diagnostic `lip_ub_F`. | Yes: clean test of the iter153 hypothesis on the actually-correct baseline. Eliminates the prefix-anchors confound that handicapped both iter153 and iter155. The Tier 0 decision (drop Lyapunov) is principled per Hartman-Grobman: ρ(J_M) < 1 is necessary AND sufficient for asymptotic local convergence; σ_max < 1 is sufficient but over-restrictive; iter152 baseline empirically demonstrates this (σ_max ≈ 17 with iter_conv_rel ≈ 0.02, clearly converging). Architecture-agnostic: same condition applies to any iteration mechanism. | 1000 steps, ~6.8 h train, ~24.4 s/step, peak VRAM ~34.86 GB. No extra Lyapunov compute. K-sweep adds ~8 JVPs per K for the new rho_F probe (8-iter power iteration on J_F) — small marginal cost. | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1000 --checkpoint-every=100 --run-id=iter158_revKL_cleanbaseline_<DATE>`. Promotion condition: full `val_bpb ≤ 1.471820` (iter152) AND `rho_F < 1` at K=128 (theoretical) OR `iter_conv_rel < 0.05` at K=128 (empirical). Branch outcomes: (a) BPB win + convergence signals OK → ship reverse-KL on corrected baseline as new defaults; (b) BPB tie + convergence OK → A/B inconclusive, defer; (c) BPB regress → reverse-KL doesn't help even on corrected baseline; (d) convergence signals fail → escalate to formal-tier mechanism (spectral norm on T_θ component layers, NOT soft penalty). |
| 3 | **iter160 = iter154 reframed: inverse-popularity expert NTP on iter158 base (conditional)** | Per-expert NTP loss weighted by `1/(s̄_k + ε)`: under-utilized experts get more independent task gradient than they earn through the routing weight `w_k`. Direct form of "under-utilized experts should learn the task". | Plausible: BPB-additive, targets the diagnosed routing-collapse mechanism. Risk: per-expert independent forward = O(E×) compute. | Implementation: medium (new loss term, per-expert head reuse, popularity-EMA lookup); compute: 1.2–2× iter158 step time. | Contingent on iter158. Run only if iter158 closes BPB but not attn_min_share gate. If iter158 closes both, iter160 held. |
| **4** | **iter162 routing sparsity push (entropy coef 0.10 → 0.30)** *(NEW 2026-05-13, user directive cap=0.3)* | Current `router_pertoken_entropy_coef=0.10` leaves routing essentially DENSE: pertoken_entropy ~3.23 nats vs max ln(32)=3.47 → ~25 of 32 experts effectively active per token. User-desired state: pertoken_entropy ≤ ln(16)=2.77 nats (≤8 experts per component). Run macro: `--router-pertoken-entropy-coef=0.30`, all other defaults inherited. Caps at 0.30 per user directive — higher would over-restrict. | Yes: simplest knob that targets the diagnosed dense-routing failure mode. No new code, no architectural change. | 1000 steps, ~7h, similar VRAM. | Promotion: full val_bpb ≤ baseline AND attn_min_share ≥ 0.0375 AND pertoken_entropy ≤ 2.77 (strict-sparsity goal). Branch: (a) BPB neutral + sparser → ship as new default; (b) BPB regress → coef too aggressive, try 0.20; (c) both bad → entropy is wrong lever, route to entmax (iter135/136 unhold). |
| **5** | **iter163 multi-K consistency loss (Final-K-as-GT)** *(NEW 2026-05-13, principled FP convergence learning)* | User-directed Option-A++ design for natural FP convergence. Algorithm: (i) Forward to FINAL K (no-grad) to produce `z_GT` (ground-truth FP); (ii) Forward K_train WITH gradient (TBPTT), save last `K_bptt` intermediates `z_k`; (iii) Add consistency loss `λ · Σ_k ||z_k − z_GT.detach()||²`. Each TBPTT pass anchors to the converged-FP target → model must learn to converge BY K_train. New Hyperparameters: `multi_k_consistency_coef` (default 0, enable 0.1), `multi_k_consistency_target_K` (default 64), `multi_k_consistency_anchor_steps` (default 3 = K_bptt). | **Architecture-agnostic ✓** (no weight reparameterization). **Implicitly enforces ρ(J_F) < 1** without direct rho_F penalty: consistency loss can only minimize if iteration map IS contractive. **Avoids iter144 IFT failure** (no IFT gradient — pure self-distillation across iterations). **Avoids iter155 Lyapunov failure** (no operator-norm penalty). | ~3× total step time (forward to K_final=64 vs K_train=16 dominates) → ~22h vs iter158's 7h. New code: ~200 lines. | Run macro: `--multi-k-consistency-coef=0.1 --multi-k-consistency-target-K=64`. Promotion: full val_bpb ≤ baseline AND **K-sweep monotone** (val_bpb(K=128) ≤ val_bpb(K_train) — deeper K cannot hurt; the principled convergence-learned property). Contingent: implement after iter158/iter160 close to know if natural FP convergence is needed. |
| 6 | **iter164 anytime-K routing (per-K output heads)** *(NEW 2026-05-13, future iter)* | Anytime-K routing parameterizes layers to give DIFFERENT outputs at different K-values, then inference picks best K per token. Mechanism: per-K layer norms or per-K output heads; learned per-token confidence selects K-output at inference. Different from iter163 multi-K consistency (which forces all K to be the same): anytime-K explicitly says K=16 may differ from K=64 and uses both. | Connection to literature: "anytime DEQ" / cascaded inference. Architecture change (parameter count grows). | Substantial: ~3-5× model size if naive per-K layer norms; could share most params with K-modulated scales. | Far-future direction; only run if iter160/iter161/iter162/iter163 all close without further BPB gains. |
| Closed | iter156 hard contraction governor | **Closed 2026-05-13 (Tier 1+2 redesign)**: the entire iter156 design was based on the false premise that $\sigma_{\max}(J_F)<1$ is required for FP convergence. The principled framework (Hartman--Grobman) is $\rho(J_F)<1$ (spectral radius, necessary AND sufficient); $\sigma_{\max}<1$ is sufficient but over-restrictive. iter152 baseline empirically converges fine with $\sigma_{\max}\approx 17$ because $\rho \ll \sigma_{\max}$ for non-symmetric $J_F$. The "contraction debt" the iter156 was meant to address was a measurement artifact (operator-norm gate was mis-calibrated). Re-open ONLY if iter158 K-sweep shows `rho_F >= 1` AND `iter_conv_rel >= 0.05` at K=128 (i.e., asymptotic convergence is genuinely failing) — currently no such evidence. If re-opened, the principled mechanism is spectral normalization on $T_\theta$ component layers (bounds $\sigma_{\max}(J_T)$ per-layer, gives certified $\sigma_{\max}(J_F)$ via composition), NOT soft penalty (refuted) and NOT Ā governor (cross-block coupling makes it non-monotone). |
| 5 | H94 GPTQ + LQER int4-rank4 | Direct attack on the int6 quantization tax (~0.01 BPB at iter152). | Yes: attacks quantization tax directly. | Multi-day calibration and quantizer rewrite. | Highest BPB ROI among heavy items; run after the contraction/routing queue stabilizes. |
| 6 | H96 per-group artifact compression | Per-group codec stack. | Engineering-principled: frees artifact budget, not BPB by itself. | Medium-heavy. | Frees artifact-size headroom for H94/H95 follow-ups. Run before H95. |
| 7 | H91 phased test-time training | Per-document adaptation. | Yes: high BPB ROI. | Multi-day eval/training loop change. | Defer behind quantization because TTT adds eval cost. Heavy one-session feature. |
| Removed | IFT hybrid redesign | Do not keep an IFT implementation in the active codebase. Any future IFT-like work must start from a new design that preserves the Parcae beta task signal and either the finite-K TBPTT tail or an explicitly bounded implicit correction. | Plausible only as a new mechanism: pure IFT assumes a useful equilibrium, uses an ill-conditioned adjoint when $\rho(\nabla T_\theta)\gg 1$, and discards solver-relaxation task gradient, which failed here. | Not runnable. | Write a design doc before code; no more pure-IFT points. Only allow a smoke test after a concrete design explains why it avoids iter144's failure mode. |
| Held | iter 135/136 entmax tweaks | Entmax blend init/LR tweaks. | Conditional: useful only if sparsity remains the active bottleneck. | Yes: CLI-only. | Trigger: per-token entropy plateaus near max with BPB stuck. |
| Closed | iter 142 deep K `(32,48)` | **Closed 2026-05-13**: subsumed by the K-jitter extension (16, 24, 32, 64, 96, 128) that landed in commit `d0bcade`. iter142 wanted deep-K coverage at (32, 48); the new K-jitter spans much wider (up to 128) at low frequency. No separate iter needed. |
| Held | H99 SmearGate (`--use-smear-gate=1`) | Default-off position-mixing local memory channel. | Plausible: BOS-masked local memory channel. | Medium: code exists; needs controlled run. | Trigger: a specific BPB failure mode emerges that local memory could address; no current motivation. |
| Held | iter118b sparse attention head gate (`--use-sparse-attn-head-gate=1`) | RevDEQ-safe smooth sparsity primitive. | Yes if smooth/differentiable; hard Top-K is disallowed. | No: 6-10h feature. | Trigger: attention throughput is measured bottleneck (chrome trace required, not log fragments). |
| Held | iter120 RRAttention (`--use-rr-attention=1`) | Dynamic block sparse attention. | Plausible but kernel-sensitive at T=2048. | No: flex/Triton rewrite. | Trigger: same as iter118b; needs flex_attention/Triton path first. |
| Held | H95 SP8192 + CaseOps (`--use-caseops=1`) | Tokenizer expansion + CaseOps fixture. | Plausible: closes vocab inefficiency. | No: tokenizer retrain and artifact budget pressure. | Trigger: H96 lands first to free artifact budget. |
| Deferred | iter 143 (`deq_bptt_k=4`) | One extra Neumann/VJP term after K=3 promoted. | Yes: clean one-axis test. | Full 1000 steps. | User-deferred 2026-05-08; revisit only on explicit request. |
| Closed | iter 103a/b/c/e/f/d chained routing | All chained-routing presets (`split_2stage`, `attn_first_2stage`, `mlp_first_2stage`, four-stage typed chains, chained-router health fix). | Best chained variant (103a) is strictly worse than non-chained iter142b on BPB AND step time; subsequent chains regressed further. No principled path to recovery without first improving non-chained baseline beyond what chains can match. | Would waste runs vs higher-ROI architecture work. | Closed in iter155 cleanup (was Deferred). Archived to `hypotheses_archive.md`. Re-open only if a fundamentally new chaining mechanism emerges. |
| Closed | iter 138/140/141 legacy gram queue | Old routing-gram ablations. | Stale under current flat objective. | Would waste runs. | Closed; archived. |
| Closed | H97/H98/H105 | Attn-gate quantization / hard sparse head gate / weak stale proposals. | Weak or mismatched to active architecture. | Not worth current queue slots. | Closed; archived. |

## Run Macros

These are experiment shapes, not mandatory commands. Iteration comparisons use
the full 1000-step budget by default. Keep all logs under
`experiments/training_logs/` and archive the result row back into this file.

```bash
# iter 142b: decomposed canonical stack
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --expert-diversity-kind=cosine --expert-output-diversity-coef=1.0 \
  --regularizer-warmup-frac=0 \
  --val-loss-every=1000000

# iter 142c: Frobenius-only counterfactual
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --expert-diversity-kind=frobenius --expert-output-diversity-coef=1.0 \
  --router-load-cv-coef=0 --router-entropy-coef=0 --mos-load-cv-coef=0 \
  --regularizer-warmup-frac=0 \
  --val-loss-every=1000000

# iter 103a: chained mixed split
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --chained-stages-preset=split_2stage
# iter 103a health comparison: keep balance alive, reduce entropy sparsity pressure
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --chained-stages-preset=split_2stage \
  --router-pertoken-entropy-coef=0.5

# iter 103b/103c: typed-chain order counterfactuals
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --chained-stages-preset=attn_first_2stage \
  --router-pertoken-entropy-coef=0.5
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --chained-stages-preset=mlp_first_2stage \
  --router-pertoken-entropy-coef=0.5

# iter 145: evidential router + EMA-GJSD CV replacement
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --router-scoring=dirichlet_ucb \
  --router-dirichlet-ucb-beta=0.5 \
  --router-load-cv-coef=0 \
  --router-ema-balance-coef=0.1 \
  --router-ema-specialization-coef=0.1 \
  --router-pertoken-entropy-coef=0.1 \
  --mos-load-cv-coef=0.1 \
  --expert-output-diversity-coef=0.1 \
  --regularizer-warmup-frac=0.07 \
  --use-router-sigmoid-gate=0 \
  --checkpoint-every=100

# iter146 rescue: bundled liveness/orthogonality/K-coverage rescue
# These flags are now the Hyperparameters defaults; keep explicit flags only
# for reproduction or A/B runs against older checkpoints.
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --deq-k-jitter-set=16,24,32,64 \
  --deq-k-jitter-weights=0.50,0.40,0.07,0.03 \
  --router-scoring=dirichlet_ucb \
  --router-dirichlet-ucb-beta=0.5 \
  --router-load-cv-coef=0 \
  --router-ema-alive-coef=0.02 \
  --router-ema-balance-coef=0.30 \
  --router-ema-specialization-coef=0.20 \
  --router-pertoken-entropy-coef=0.1 \
  --mos-load-cv-coef=0.15 \
  --expert-output-diversity-coef=0.30 \
  --regularizer-warmup-frac=0.07 \
  --use-router-sigmoid-gate=0 \
  --weight-decay=0.015 \
  --checkpoint-every=100

# iter147 contraction ablation: run only if iter146 leaves lip_ub >= 1
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --lyapunov-coef=0.005 \
  --lyapunov-gamma=0.97 \
  --lyapunov-every=16 \
  --lyapunov-max-tokens=64 \
  --checkpoint-every=100

# iter148 contraction rescue: run only if iter147 improves lip_ub but does not solve it
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --lyapunov-coef=0.0075 \
  --lyapunov-gamma=0.97 \
  --lyapunov-every=16 \
  --lyapunov-max-tokens=64 \
  --checkpoint-every=100

# iter149 health-first rescue: double all direct health losses, no Lyapunov penalty
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --router-ema-alive-coef=0.04 \
  --router-ema-balance-coef=0.60 \
  --router-ema-specialization-coef=0.40 \
  --router-pertoken-entropy-coef=0.20 \
  --expert-output-diversity-coef=0.60 \
  --lyapunov-coef=0 \
  --checkpoint-every=100

# iter150 health-first rescue: repeat combined observed-problem doubling + K32-200
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --deq-k-jitter-set=16,24,32,64 \
  --deq-k-jitter-weights=0.43,0.34,0.20,0.03 \
  --router-ema-alive-coef=0.04 \
  --router-ema-balance-coef=1.20 \
  --router-ema-specialization-coef=0.80 \
  --router-pertoken-entropy-coef=0.40 \
  --expert-output-diversity-coef=1.20 \
  --lyapunov-coef=0 \
  --checkpoint-every=100

# iter151 health-first rescue: third combined observed-problem doubling + K rollback
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --deq-k-jitter-set=16,24,32,64 \
  --deq-k-jitter-weights=0.50,0.40,0.07,0.03 \
  --router-ema-alive-coef=0.04 \
  --router-ema-balance-coef=2.40 \
  --router-ema-specialization-coef=1.60 \
  --router-pertoken-entropy-coef=0.80 \
  --expert-output-diversity-coef=2.40 \
  --lyapunov-coef=0 \
  --checkpoint-every=100

# iter152 conditional prefix-K anchors on iter149 coefficients
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --deq-prefix-anchors=1 \
  --deq-k-jitter-set=16,24,32,64 \
  --deq-k-jitter-weights=0.50,0.40,0.07,0.03 \
  --router-ema-alive-coef=0.04 \
  --router-ema-balance-coef=0.60 \
  --router-ema-specialization-coef=0.40 \
  --router-pertoken-entropy-coef=0.20 \
  --expert-output-diversity-coef=0.60 \
  --lyapunov-coef=0 \
  --checkpoint-every=100

# iter155 corrected (2026-05-13): Lyapunov-on-F (two-state Parcae cycle)
# + lip_ub_F gate diagnostic, on iter152 base.  The single-state
# lyapunov_target=iteration_S surrogate from the earlier draft is kept
# as an advisory option but is no longer the recommended target — the
# actual solver iterates F, and damping cannot rescue an expansive T
# via S either (σ_max bound argument; see opg_doc.tex Remark on
# "Choosing the contraction object").
# Result: NOT PROMOTED (val_bpb 1.481344 vs iter152 1.471820, Δ +0.0095).
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --use-reverse-kl-balance=0 \
  --lyapunov-coef=0.005 \
  --lyapunov-target=iteration_F \
  --lyapunov-gamma=0.97 \
  --lyapunov-every=16 \
  --lyapunov-max-tokens=64 \
  --checkpoint-every=100

# iter158 (2026-05-13): bundled retest replacing iter157 + iter153-rerun.
# Tier 0+1+2 redesign: drop Lyapunov-on-F entirely (refuted as not
# principled for asymptotic convergence by Hartman-Grobman + iter152
# K-sweep evidence: sigma_max=17 with iter_conv_rel=0.02 clearly
# converges, so sigma_max<1 is over-restrictive and Lyapunov-on-F
# targeted the wrong condition).  Tests reverse-KL as one-variable
# change vs iter152 on the corrected baseline (prefix-anchors=True).
# Inherits all defaults: deq_prefix_anchors=True, reverse_kl_balance=True,
# lyapunov_coef=0.  K-sweep emits new rho_F (spectral radius) as the
# gate-relevant convergence signal alongside lip_ub_F (advisory).
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --checkpoint-every=100 \
  --run-id=iter158_revKL_cleanbaseline_<DATE>

# iter162 (2026-05-13): routing sparsity push.  User directive caps
# router_pertoken_entropy_coef at 0.30.  Tests whether higher entropy
# pressure achieves the desired "less than half experts active per
# token" state (target pertoken_entropy <= ln(16)=2.77 nats vs current
# 3.23 nats which means ~25 of 32 experts effectively active).
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --router-pertoken-entropy-coef=0.30 \
  --checkpoint-every=100 \
  --run-id=iter162_sparsity_push_<DATE>

# iter163 (2026-05-13): multi-K consistency loss with Final-K-as-GT.
# Principled architecture-agnostic mechanism for natural FP convergence
# learning.  Forward to K_final (no-grad) for the FP target; TBPTT to
# K_train; consistency loss anchors each TBPTT iteration to the FP target.
# Implicitly enforces rho_F < 1 (necessary for the consistency loss to
# minimize) without any direct rho_F penalty.  Avoids iter144 IFT failure
# and iter155 Lyapunov failure.  ~3x step time vs baseline due to
# forward-to-K_final overhead.  Implementation pending (~200 lines).
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=1000 \
  --multi-k-consistency-coef=0.1 \
  --multi-k-consistency-target-K=64 \
  --multi-k-consistency-anchor-steps=3 \
  --checkpoint-every=100 \
  --run-id=iter163_finalK_GT_<DATE>

# iter164 (2026-05-13, far-future): anytime-K routing.  Per-K layer
# norms / per-K output heads.  Inference picks best K per token via
# learned confidence.  Substantial implementation (~3-5x param count if
# naive per-K layer norms).  Only run if iter160/iter161/iter162/iter163
# all close without further BPB gains.  Run macro TBD pending design doc.
```

## Durable Rules

- Routing penalties operate on active routed mass. With the default
  `use_router_sigmoid_gate=False` (iter146), mass is `simplex(allocation_scores)`;
  when the optional sigmoid gate is enabled, mass is the combined
  `simplex(allocation_scores) * sigmoid(gate)`. Either way, never on
  renormalized slices.
- Hard Top-K / magnitude-skip dispatch is not RevDEQ-safe. Any sparsity path
  must be smooth, default-off, and exact/no-op at the strict-generalization
  point.
- Loss metrics and hard gates can differ when their roles differ: losses need
  dense gradients; gates need clean failure detection.
- All gates must be token-local and input-dependent; no batch/sequence
  reductions may influence a token's fixed-point map.
- Do not use Sinkhorn/OT/capacity matching for expert assignment; global usage
  pressure belongs in auxiliary losses or slow router-bias feedback.
- Persistent EMA usage is detached state. It can anchor diagnostics, alive-loss
  weights, slow bias feedback, and straight-through EMA-balance losses. Detached
  `KL(EMA||uniform)` alone has no gradient to the current router.
- Promoted iter145r defaults improve BPB but do not close strict expert-health
  or contraction gates. Iter146 is a bundled 1000-step rescue run, not an
  attribution-clean mechanism test: keep the EMA alive hinge in because it
  directly targets the observed min-share failure, but keep Lyapunov/Jacobian
  contraction losses out so contraction remains isolated in iter147.
- The iter149-151 health-first rescue sequence intentionally kept
  `lyapunov_coef=0`: `lip_ub` (now `lip_ub_T`) was a reported diagnostic with
  desired goal `< 1`, and the sequence prioritized sparse, live, orthogonal
  experts before adding any direct transition-map penalty. **Superseded by
  iter155 (corrected 2026-05-13)**: the gate-aligned contraction object is
  `lip_ub_F` on the actual two-state Parcae cycle map $F$ (see Durable
  Rules below for the cycle definition), and the Lyapunov target is
  selectable via `lyapunov_target` ∈ {`transition_T`, `iteration_S`,
  `iteration_F`}. `lip_ub_T` and `lip_ub_S` are retained as advisory
  decomposition diagnostics; `iteration_S` is preserved as a backward-
  compatible code path but is NOT the iterated map (the user-feedback
  correction).
- For iter149-151, every observed problematic health metric maps to a direct
  training loss and all active problem losses are doubled together per
  iteration: long-run load balance uses `router_ema_balance_loss` and should be
  judged primarily by EMA usage/min-share diagnostics, sparse specialization
  uses `router_pertoken_entropy_loss` plus EMA specialization, and output
  orthogonality uses `expert_diversity_loss`. Current-batch `attn_cv`,
  `mlp_cv`, and `pool_cv` are secondary smoke diagnostics only; they can rise
  on specialist batches and should not be the promotion criterion while
  `router_load_cv_coef=0`. `router_ema_alive_loss` is a conditional guard for
  persistent under-floor usage; iter146 had `router_ema_alive_loss=0`, so do
  not treat alive escalation as an active rescue lever unless the hinge becomes
  nonzero or a true dead/under-floor EMA failure appears.
- Promotion hard-gates on validation BPB only. EMA balance/min-share, sparsity,
  orthogonality, current-batch CV, `lip_ub_S`, and `lip_ub_T` are soft
  diagnostics: report them, then map each undesirable metric to its
  corresponding principled rescue regularizer for the next iteration rather
  than rejecting an otherwise better validation-BPB point.
- iter149 is BPB-winner-but-not-promoted-to-defaults. Final full
  `val_bpb=1.478698` is a small win over iter146 `1.479157` and iter149 is the
  comparison base for iter152+. However, `Hyperparameters` defaults remain at
  the iter146 values (`router_ema_balance_coef=0.30`,
  `router_ema_specialization_coef=0.20`, `router_pertoken_entropy_coef=0.10`,
  `expert_output_diversity_coef=0.30`, `router_ema_alive_coef=0.02`) because
  iter150 regressed the doubled coefficients and iter151 was aborted before
  any promotion-propagation commit. Iter149's coefficients must be passed
  explicitly via CLI (`--router-ema-balance-coef=0.60 ...`) until a follow-up
  cleanly wins on BPB and triggers a full defaults-propagation commit.
- Weighted K jitter over `{16,24,32,64}` is a finite-depth robustness add-on,
  not the contraction fix. Low-probability high-K sampling is represented by
  the explicit `deq_k_jitter_weights` field; the sampler normalizes weights,
  samples exact weighted bags, restores checkpoint state, and logs sampled-K
  counts for audit.
- Pure IFT adjoint with no TBPTT/Parcae-beta task signal is not viable under
  the current noncontractive transition map: iter144 was faster but
  catastrophically worse. The code and CLI knob were removed; any revisit must
  be a hybrid redesign, not another pure IFT coefficient point.
- Router confidence tracking is diagnostic-only in iter146. Expected trajectory:
  Dirichlet strength/evidence rise, `E/S` and sigma fall, normalized `H(mu)`
  falls if routing specializes, and UCB beta anneals toward zero.
- **Lyapunov contraction escalation policy (aug-Lagrangian style).**
  `lyapunov_coef=0` by default. For iter155-style contraction testing,
  start with a small coefficient (e.g. 0.005, the iter147 BPB-cheap point)
  and treat it as **soft pressure, not a guarantee**. Escalate ONLY when
  the gate-aligned `lip_ub_F` (safety-adjusted empirical metric, not a formal certificate) stays
  $\ge \rho_{\rm target} = 0.95$ after the prior coefficient round; cap
  the coefficient (no automatic doubling beyond 2-3 escalations); and
  STOP escalation if validation BPB regresses by more than $+0.005$ vs
  the iter152 baseline. The aug-Lagrangian intuition: iterate
  `coef ← coef × growth` only while the constraint is violated AND the
  task loss tolerates it. Do not treat any soft-penalty schedule as
  proof of contraction — only iter156's hard governor or a model-class
  change (spectral norm / bounded Lipschitz / learned $c\cdot\Delta$ gain)
  provides a formal certificate. The `lyapunov_estimator=power_jvp_F`
  opt-in (worst-direction probe, ~3× compute) is the next escalation
  rung between random_fd soft pressure and iter156's hard governor.
- iter155 (corrected 2026-05-13) cuts the contraction work over from probing
  $T_\theta$ to probing the **actual two-state Parcae cycle** $F$.  The
  earlier single-state $S=\bar A z+(1-\bar A)T_\theta$ surrogate was
  rejected because the solver does not iterate $S$, and $\sigma_{\max}(J_S)
  \le \bar A+(1-\bar A)\sigma_{\max}(J_T)>1$ for any $\bar A<1$ when
  $\sigma_{\max}(J_T)>1$ — damping cannot rescue an expansive T via S
  either.  `lip_ub_F` is the gate-aligned operator-norm probe on $J_F$;
  `lip_ub_S` and `lip_ub_T` retained as advisory decomposition diagnostics.
  Lyapunov-on-F (and Lyapunov-on-S) does NOT detach $\bar A$, so gradients
  flow to Parcae damping AND transition parameters jointly — the right
  mechanism for shrinking the cross-block coupling $\beta\cdot J_T$ in
  $J_F$.  If iter155 leaves `lip_ub_F>=1` after the Lyapunov-on-F pressure,
  escalate to iter156 (hard $\bar A$ projection governor 156a or new
  $c\cdot\Delta$ gain 156b) — not another Lyapunov-on-T coefficient sweep.
- For orthogonality/specialization, prefer normalized expert-output Gram/cosine
  over router-row orthogonality; the latter is only a weak conditioning prior.
- **Asymptotic local FP convergence is the architecture-agnostic
  ultimate goal** (Tier 1+2 redesign 2026-05-13). The principled
  framework: $\rho(J_M)<1$ (spectral radius of the iteration map's
  Jacobian) is the necessary AND sufficient condition (Hartman--Grobman).
  $\sigma_{\max}(J_M)<1$ is sufficient but over-restrictive --- for
  non-symmetric $J_M$ the gap can be huge (iter152 baseline: $\sigma_{\max}
  \approx 17$, $\text{iter\_conv\_rel} \approx 0.02$ at K=128, clearly
  converging). Promotion gate is on `rho_F < 1` (theoretical, by
  power-iteration probe on $J_F$ directly) OR `iter_conv_rel < 0.05`
  at deepest K (empirical). Diagnostics retained as advisory only:
  `lip_ub_F` (operator norm $\sigma_{\max}(J_F)$, over-restrictive
  proxy), `lip_ub_S` and `lip_ub_T` (decomposition), `fp_residual_F`,
  and `fp_bound = fp_residual_F/(1-lip_ub_F)` (Banach bound, only
  meaningful if $\sigma_{\max}<1$ which is rarely the case). The
  framework applies to ANY iteration mechanism: alternative DEQ
  solvers, refinement loops, recurrent layers --- all subject to the
  same `rho < 1` constraint regardless of model class.
- **Lyapunov/operator-norm soft penalties refuted as not principled
  for asymptotic convergence** (2026-05-13, iter155 evidence). The
  penalty (a) targets the wrong condition ($\sigma_{\max}$ over the
  necessary $\rho$), (b) cannot reliably constrain spectral properties
  via soft pressure even with a tighter probe, (c) was empirically
  measured to make `lip_ub_F` *worse* on average across K than the
  untreated iter152 baseline (18.05 vs 17.73 mean) while paying
  +0.0095 BPB. Default `lyapunov_coef=0` is the principled choice;
  do not enable for new iters. If `rho_F` or `iter_conv_rel` actually
  fail at promotion, escalate to formal-tier mechanisms --- but
  architecturally, NOT via soft penalties on operator-norm proxies.
- **Future cleanup: remove lip_ub_F/lip_ub_S, keep rho_F as sole
  contraction gate metric** (queued as task #47, deferred until
  iter158 + iter162 verification stabilizes defaults).  Rationale:
  iter158 K-sweep at K=24 measured `rho_F=0.83` (gate-relevant, well
  under 1) vs `lip_ub_F=14.63` (over-restrictive proxy that would
  have falsely flagged failure under the iter155-era operator-norm
  gate) — a 17× gap empirically confirming `lip_ub_F<1` is essentially
  unachievable for non-symmetric J_F without crippling expressivity,
  while `rho_F<1` is achievable AND the necessary-AND-sufficient
  Hartman--Grobman condition.  Continuing to track lip_ub_F adds
  false-failure noise (iter155 NOT-PROMOTED was driven partly by
  lip_ub_F=17.08, but iter152's untreated baseline has rho_F~0.83
  with the same lip_ub_F~17 — measurement artifact, not model defect).
  Cleanup removes ~250 lines (`_run_spectral_norm_power`,
  `_lip_ub_at_saved_fp`, K-sweep emissions, parser entries),
  updates `fp_bound = fp_residual_F / (1−rho_F)` (tighter Banach
  bound), keeps `lip_ub_T` as opt-in decomposition diagnostic for
  potential iter156 re-opening (only useful if asymptotic
  convergence ever genuinely fails).
- **Principled mechanism for natural FP-convergence learning:
  Final-K-as-GT consistency loss (iter163 design)**, NOT direct
  rho_F penalty.  Algorithm: forward to FINAL K (no-grad) → ground-
  truth FP target $z_{GT}$; forward to K_train with TBPTT → save last
  K_bptt intermediates; add consistency loss
  $\lambda \sum_k \|z_k - z_{GT}.\text{detach}()\|^2$ anchoring each
  TBPTT pass to the converged-FP target.  **Architecture-agnostic** ✓
  (no weight reparameterization).  **Implicitly enforces $\rho(J_F)<1$**:
  consistency loss can only minimize if iteration map IS contractive.
  Avoids iter144 IFT failure (no IFT gradient — pure self-distillation
  across iterations) and iter155 Lyapunov failure (no operator-norm
  penalty).  Cost: ~3× step time (forward to K_final dominates).
  Spectral normalization on $T_\theta$ component layers is rejected
  as a principled mechanism because it constrains a specific model
  architecture (transformer-block weights) and violates the
  arch-agnostic constraint --- the FP-convergence framework must apply
  to any iteration mechanism, not specifically to weight-parameterized
  transition maps.  Pure IFT (Option C) is also rejected: iter144
  empirically refuted with +0.55 BPB regression.  Net: iter163's
  Final-K-as-GT consistency is the principled path; iter156 (operator-
  norm-targeted hard governor) stays closed.
- **`lip_ub_F` (and the paired `fp_residual_F`) is ALWAYS reported on
  every FP eval — train-time fast-val AND every K-sweep row — never
  gated by `fp_lip_fast_val_every` or `EvalProfile.lip_probe_set`.**
  Decomposition diagnostics `lip_ub_T` / `lip_ub_S` remain on the
  cadence because they cost extra JVPs without changing the gate
  decision (they're only useful when `lip_ub_F` has crossed the
  prescription threshold and the next iter needs to know which
  component — J_T magnitude vs Ā damping — is the lever). Silently
  dropping the gate-aligned object on intermediate evals is the
  "decision based on metric we haven't measured" failure mode.
  Enforced by `experiments/test_lip_ub_fix.py::
  test_lip_ub_F_is_always_probed_on_FP_eval_unconditional_of_cadence_knob`
  which static-asserts both site structures (fast-val + K-sweep)
  in `train_gpt.py`.  User directive 2026-05-13.
- **Iter-progress reports always pair current-step value with same-step
  baseline.** When summarizing a running iter to the user, every
  reported metric needs three numbers: (a) current-step value, (b)
  delta-since-prior-check (per-step or absolute), and (c) baseline at
  the same step from the prior promoted iter's run.log (parse via
  grep `^step:N ` against `experiments/training_logs/<baseline>.log`).
  Cumulative-from-launch comparisons hide regimes where the new iter
  trails early then catches up (or vice versa) — only same-step
  comparisons are decision-relevant.  If the baseline log is missing
  for a given step, say so explicitly so the user can produce one
  before drawing conclusions.  User directive 2026-05-13.
- Follow DRY and function-orthogonality by default: shared probe setup, metric
  formulas, reduction helpers, and formatting each live in one place; raw
  measurement functions must stay independent from policy/gating functions.
- Required diagnostics must satisfy the metric contract: compute site,
  forward-freshness tag when needed, train/val/K-sweep log emission, parser
  support, and a focused test. Do not make one metric disappear because an
  unrelated diagnostic was unavailable.
- Promote only with a controlled comparison against the active baseline plus
  post-int6 K-sweep and health diagnostics. For short probes, state the budget
  limit explicitly.
- Keep implementation details and long evidence in `EXPERIENCE.md` or the
  archive; keep this file as the decision ledger.
