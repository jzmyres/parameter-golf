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
| Main active question | Health-first rescue after user directive: promotion hard-gates on validation BPB only. Do not explicitly penalize `lip_ub` with Lyapunov because it hurt BPB. Treat sparse routing, orthogonality, EMA load balance, liveness, and `lip_ub < 1` as soft diagnostic goals that select the next principled rescue loss/coef rather than blocking promotion by themselves. |
| Near queue | iter153 (reverse-KL balance) is the next test: provisional default-on `use_reverse_kl_balance=True` swaps the EMA balance form from forward `KL(EMA‖U)` to reverse `KL(U‖EMA)` so the dead-expert gradient diverges as `−U_i/EMA_i` instead of vanishing under forward KL's mass-weighted sum. Targets the iter152 promotion's open `attn_min_share=0.0315 < 0.0375` failure directly without adding a new coefficient. iter154 (inverse-popularity per-expert NTP) is queued as the follow-up mechanism if iter153 closes min-share but BPB stalls. Rescue coefficient escalation remains stopped after iter150. Iter152 conditional prefix-K anchors are the BPB-winning reference on the iter149 coefficient base: final full `1.471820` vs iter149 `1.478698`. Do not retry pure IFT; if gradient-quality is revisited, redesign it as a hybrid finite-K/IFT estimator with a fresh hypothesis. `iter143` and chained routing remain deferred. |
| Heavy queue | Remaining non-deferred/held items are now active default-off components: H91 TTT (`--use-ttt-eval=1`), H94 GPTQ+LQER (`--use-gptq=1`, `--use-lqer=1`), H96 grouped artifact compression (`--use-grouped-artifact-compression=1`), H95 CaseOps fixture (`--use-caseops=1`), iter118b sparse attention head gate (`--use-sparse-attn-head-gate=1`), iter120 RRAttention (`--use-rr-attention=1`), and H99 SmearGate (`--use-smear-gate=1`). Each component is removable by deleting its module plus the narrow `train_gpt.py` hook if its isolated iteration fails. |

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
| 1 | iter153 reverse-KL balance | Swap the EMA balance loss form from forward `KL(EMA‖U)` to reverse `KL(U‖EMA)`. Forward KL weights each per-expert term by `EMA_i`, so a dead expert's contribution is `EMA_i · log(EMA_i/U_i) → 0` even when the log ratio is large — the gradient on the tail vanishes. Reverse KL weights by `U_i`, so each expert contributes equally regardless of its mass, and the dead-expert gradient is `−U_i/EMA_i`, diverging as `EMA_i → 0`. One-line code change at `train_gpt.py:2856` gated by `use_reverse_kl_balance` (Hyperparameters default True, iter153 default-on). Provisional default-on for testing; reverts to forward KL if iter153 regresses BPB. | Yes: principled fix for the asymmetric tail weighting diagnosed in the iter152 post-int min-share failure (`attn_min_share=0.0315 < 0.0375`). For the diagnosed numbers (E=16, one dead expert at half its fair share), reverse KL gives ~3.4× stronger gradient than forward KL; the gap widens as EMA → 0. No new coefficient; existing `router_ema_balance_coef=0.30` carries the same magnitude budget. | 1000 steps, ~6.78 h train, ~24.4 s/step, peak VRAM ~34.86 GB (iter146 baseline; reverse KL adds <1% compute — one extra reciprocal per expert per step). | Run with no extra flags; `--use-reverse-kl-balance=0` recovers the forward-KL baseline for direct A/B. Promotion condition: full val_bpb ≤ iter152's `1.471820` AND post-int `attn_min_share ≥ 0.0375`. If BPB regresses, revert `Hyperparameters.use_reverse_kl_balance` to False and re-queue. If BPB ties and min-share improves, promote and document the policy match. |
| 2 | iter154 inverse-popularity expert NTP | Per-expert NTP loss weighted by `1/(s̄_k + ε)`: under-utilized experts get more independent task gradient than they earn through the routing weight `w_k`. Direct form of "under-utilized experts should learn the task" — the principled alternative to the multiplicative coupling `(L_bal + 1) · L_ntp` (rejected because the global multiplier doesn't target the dead experts: their `w_k · multiplier ≈ tiny`). | Plausible: BPB-additive (sum of weighted CE terms, units preserved), targets the diagnosed mechanism (per-expert task signal vs. global gradient amplification), and does not couple loss objectives multiplicatively. Risk: needs per-expert forward to compute `E_k(x)` independently — either O(E×) compute or a low-cost approximation (e.g., pseudo-target from `w_k · E_k`). | Implementation: medium (new loss term, per-expert head reuse, popularity-EMA lookup); compute: 1.2–2× iter153 step time depending on approximation. | Queue behind iter153. If iter153 closes min-share but BPB stalls, iter154 is the next mechanism candidate. If iter153 closes both, iter154 is held. |
| Removed | IFT hybrid redesign | Do not keep an IFT implementation in the active codebase. Any future IFT-like work must start from a new design that preserves the Parcae beta task signal and either the finite-K TBPTT tail or an explicitly bounded implicit correction. | Plausible only as a new mechanism: pure IFT assumes a useful equilibrium, uses an ill-conditioned adjoint when `lip_ub >> 1`, and discards solver-relaxation task gradient, which failed here. | Not runnable. | Write a design doc before code; no more pure-IFT points. Only allow a smoke test after a concrete design explains why it avoids iter144's failure mode. |
| Held | iter 135/136 | Entmax blend init/LR tweaks. | Conditional: useful only if sparsity remains the active bottleneck. | Yes: CLI-only. | Hold until smooth diversity/rescue work shows sparsity is still the bottleneck. |
| Held | iter 142 deep K | `deq_k_jitter_set=(32,48)`. | Conditional: deeper solve only pays off after routing sparsifies. | Yes: CLI-only, but costly. | Run only if effective experts are roughly `<= 7` and step cost has sparse-kernel headroom. |
| 5 | H99 SmearGate | Default-off position-mixing memory channel. | Plausible: BOS-masked local memory channel. | Medium: code exists; needs controlled run. | Keep default-off; run only after the current routing/TBPTT queue. |
| 6 | iter 118b | RevDEQ-safe smooth sparsity primitive. | Yes if smooth/differentiable; hard Top-K is disallowed. | No: 6-10h feature. | Defer until 142b/142c and cheap entmax knobs fail to produce useful sparsity. |
| 7 | iter 120 | RRAttention / dynamic block sparse attention. | Plausible but kernel-sensitive at T=2048. | No: flex/Triton rewrite. | Defer until attention scaling is the measured bottleneck. |
| 8 | H91 | Phased test-time training. | Yes: high likely BPB ROI from per-document adaptation. | No: new eval/training loop. | Heavy one-session feature after baseline stabilizes. |
| 8 | H94 | GPTQ + LQER int4-rank4. | Yes: attacks quantization tax directly. | No: calibration and quantizer rewrite. | Run after model-side baseline stabilizes. |
| 8 | H96 | Per-group compression stack. | Engineering-principled: frees artifact budget, not BPB by itself. | Medium-heavy. | Separate artifact-efficiency track; useful before tokenizer expansion. |
| 8 | H95 | SP8192 + CaseOps. | Plausible: closes vocab inefficiency. | No: tokenizer retrain and artifact budget pressure. | Depends on H96 or equivalent budget relief. |
| Closed | iter 138/140/141 legacy gram queue | Old routing-gram ablations. | Partly, but stale under current flat objective. | Would waste runs. | Mark superseded; do not run. |
| Closed | H97/H98/H105 | Attn-gate quantization / hard sparse head gate / weak stale proposals. | Weak or mismatched to active architecture. | Not worth current queue slots. | Close or keep archived only. |

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
- The iter149-151 health-first rescue sequence intentionally keeps
  `lyapunov_coef=0`: `lip_ub` remains a reported diagnostic with desired goal
  `< 1`, but the sequence prioritizes sparse, live, orthogonal experts before
  adding any direct transition-map penalty.
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
  orthogonality, current-batch CV, and `lip_ub` are soft diagnostics: report
  them, then map each undesirable metric to its corresponding principled rescue
  regularizer for the next iteration rather than rejecting an otherwise better
  validation-BPB point.
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
- If the root-cause rescue does not materially reduce `lip_ub`, enable the
  separate contraction ablation: a Lyapunov loss on measured expansion of
  `T_theta`, not the old stochastic surrogate re-enabled unchanged.
- For orthogonality/specialization, prefer normalized expert-output Gram/cosine
  over router-row orthogonality; the latter is only a weak conditioning prior.
- For DEQ local-contraction evidence, use one reported Lipschitz metric:
  `lip_ub`. The sufficient local condition is `lip_ub < 1`; fast validation
  and K-sweep also report the separate residual-side
  `fp_bound = fp_residual_rel/(1-lip_ub)` when that bound is defined.
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
