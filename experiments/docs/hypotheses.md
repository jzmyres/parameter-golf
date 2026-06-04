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
| Code source of truth | `reports/opg_doc.tex` and ADR 0002 define the active final-minimal P1 pipeline. `experiments/p1_synthetic.py` is the executable Tier-1 harness; `train_gpt.py` is legacy small-LM/deployment infrastructure until Tier 1 passes. |
| Active goal | P1 first: show positive paired depth utility on a depth-hard task with a passing positive control. P2 activation-memory evidence is separate. P3 MoE-basis recovery is staged only after P1 holds. |
| Active models | P1 promotion path: `control`, `m0`, and `mclk`. The active harness intentionally has no MoE, cache, or int6 switches. Separate feedback-stage diagnostics now live in `experiments/p1_feedback_stages.py` for S+1/S+2/S+3 coverage without widening P1. |
| Active metrics | Paired `G_nll`/`G_acc` with confidence intervals, `NDR_epsilon`, K sweep loss/accuracy, peak VRAM, and structural reconstruction error for `m0`/`mclk`. Mean-only gains are not promotion evidence. |
| Latest run | `RUN_TAG=gpu67_feedback_all_i1000`, GPUs 6/7 DDP, 1000 iterations, seq_len 16. The minimal S+0 runner plus separate S+1/S+2/S+3 diagnostics completed. P1 did not promote: positive-control `G_nll(16,64)=-0.000477` with CI `[-0.000951,-0.000002]` and `NDR_epsilon=0.5122`. `M0`/`M_clk` reconstruction stayed near `1e-5`. |
| Next active question | Repair Tier-1 detectability before model changes. Sweeping `SEQ_LEN` and 1000 iterations was not enough, so the next move is to verify the task/target and positive-control capacity: use a simpler provable positive control, a shorter/easier compositional curriculum, or a target representation with a stronger training signal. |
| Closed for active P1 | Nonzero `lyapunov_coef`, Lipschitz-band pressure, prefix-anchor consistency, Dirichlet-UCB, MLA, MoS, distillation, UFID, anytime/adaptive diagnostics, plain MoE, static MoE, cache proxy, and int6 eval. MoE/static-MoE, hidden-state cache proxy, and fake-int6 are available only through the separate feedback-stage diagnostic runner and do not promote P1. |

## Active Run Macros

```bash
# Default S+0 DDP smoke on the requested GPU pair.
CUDA_VISIBLE_DEVICES=6,7 ITERATIONS=300 RUN_TAG=gpu67 bash experiments/run_p1_synthetic_pipeline.sh

# Positive-control detectability sweep examples. Change only task/budget knobs.
CUDA_VISIBLE_DEVICES=6,7 SEQ_LEN=8 ITERATIONS=1000 RUN_TAG=gpu67_seq8_i1000 bash experiments/run_p1_synthetic_pipeline.sh
CUDA_VISIBLE_DEVICES=6,7 SEQ_LEN=16 ITERATIONS=1000 RUN_TAG=gpu67_seq16_i1000 bash experiments/run_p1_synthetic_pipeline.sh

# Full feedback-stage executable coverage; S+1/S+2/S+3 are diagnostics only.
CUDA_VISIBLE_DEVICES=6,7 ITERATIONS=1000 RUN_TAG=gpu67_feedback_all_i1000 bash experiments/run_feedback_stage_pipeline.sh
```

## Current Feedback-Stage Verification (2026-06-04)

`RUN_TAG=gpu67_feedback_all_i1000` completed on GPUs 6/7 with DDP and 1000 iterations per stage row.
It verifies executable coverage of the feedback document without widening the P1 runner.
It supersedes, but does not erase, the pruned S+0 verification
`RUN_TAG=gpu67_s0_pruned_i1000`, which already showed the same positive-control blocker.

| Row | Best K/loss | `G_nll(16,64)` 95% CI | NDR | Notes |
|---|---:|---:|---:|---|
| `s0_control` | 16 / 4.793021 | `-0.000477 [-0.000951, -0.000002]` | 0.5122 | Positive control fails P1. |
| `s0_m0` | 16 / 4.793416 | `0.000071 [-0.000270, 0.000412]` | 0.4968 | Structural reconstruction `9.57e-6`. |
| `s0_mclk` | 16 / 4.793167 | `-0.000073 [-0.000351, 0.000205]` | 0.5146 | Structural reconstruction `1.12e-5`. |
| `s1_moe` | 16 / 4.794307 | `-0.000133 [-0.000396, 0.000130]` | 0.5159 | Route-depth NMI `8.02e-6`, AEBR `1.385`, utilization `0.9992`. |
| `s1_static_moe` | 16 / 4.794298 | `-0.000134 [-0.000393, 0.000125]` | 0.5110 | Route-depth NMI `4.93e-6`, AEBR `1.512`, utilization `0.9995`. |
| `s2_m0` | 16 / 4.793416 | `0.000071 [-0.000270, 0.000412]` | 0.4968 | Hidden-state proxy only: exact slope 1, terminal/shared slope 0; KV quality gap untested. |
| `s3_m0` | 16 / 4.793416 | `0.000071 [-0.000270, 0.000412]` | 0.4968 | Fake-int6 best-loss gap `+0.000907`; int6 `G_nll(16,64)=0.000044 [-0.000295,0.000383]`. |

Verdict: implementation coverage is now complete for S+0 through S+3, but the scientific blocker is
unchanged. Do not escalate mechanisms. Repair the Tier-1 detectability target or positive-control
capacity first.

## Legacy Dense-MoE Snapshot (Superseded)

Everything in this section and below is historical unless explicitly referenced
by the final-minimal P1 protocol. It is kept as evidence, not as the active
research queue. In particular, pure IFT remains removed, and nonzero
`lyapunov_coef` remains rejected for active P1.

| Area | Current state |
|---|---|
| Code source of truth | `train_gpt.py::Hyperparameters`; docs must track active defaults, not old records. |
| Backbone | RevDEQ + Parcae, `use_parcae=True`, weighted `deq_k_jitter_set=(16,24,32,64,96,128)` with weights `(0.50,0.40,0.07,0.03,0.015,0.0075)` normalized at launch, `deq_bptt_k=3`. |
| Routing loss stack | Root-cause rescue stack: Dirichlet-UCB routing, router CV off, EMA-anchored balance/specialization (`router_ema_balance_coef=0.30`, `router_ema_specialization_coef=0.20`), small alive hinge (`router_ema_alive_coef=0.02`), worst-pair cosine expert-output diversity (`expert_output_diversity_coef=0.30`), MoS-CV/per-token entropy unchanged, and 0.07 warmup. Legacy nested routing-gram stack is superseded. |
| Main active question | Health-first rescue after user directive: promotion hard-gates on validation BPB only. **Tier 1+2 redesign (corrected 2026-05-13)** establishes ρ(J_F) < 1 as the necessary AND sufficient gate (Hartman--Grobman); σ_max(J_F) < 1 was over-restrictive (iter152 has σ_max ≈ 17 yet iter_conv_rel ≈ 0.02). Promotion gate: `rho_F < 1` (theoretical) OR `iter_conv_rel < 0.05` at deepest K (empirical). `lip_ub_F` retained as decomposition diagnostic only. **Routing-balance push class closed 2026-05-15** (3-iter empirical refutation iter158/162/165). **iter163 PROMOTED 2026-05-15 as new champion** (val_bpb=1.471598, Δ −0.22 mBPB vs iter152): the Final-K-as-GT consistency loss DID induce natural FP convergence (rho_F 1.30 → 0.92) AND tightened K-extrapolation 4× (gap +0.0004 vs iter158's +0.0018). Next active question: can iter163's principle be sharpened further (higher λ, deeper Δ, anchor-on-final-K) to widen the K-extrapolation lead? Or is the next BPB headroom in quantization (iter161 H94+H96)? Sparse routing, orthogonality, EMA load balance, liveness retained as soft diagnostic goals (informational, not BPB-actionable). |
| Near queue | **iter172 PROMOTED 2026-05-17 as new champion** at full_full_validation val_bpb=**1.462898** (Δ −0.008700 vs iter163's 1.471598 = **−8.7 mBPB, 39× iter163's 0.22 mBPB margin over iter152**). K-sweep dominates uniformly K=16→K=128 by 8.5-9.5 mBPB; rho_F at K=128=0.77 vs iter163's 0.92 (principled FP-convergence dramatically improved). Routing-quality side-effects positive: attn_cv −0.028, mlp_ortho −0.020. Artifact 7.40 MB. Promotion-propagation: `Hyperparameters.deq_prefix_anchor_set` default flipped `()` → `(8, 16, 24, 32, 64, 128)` (K=8 shallow anchor preserves iter163's healthy gap=8 regime). **Queue re-optimized 2026-05-17 by post-iter172 principleness audit + iter173 resume-anneal result**: (1) **iter161-QAT-late on iter172 base** — Pri 1 NEXT, deferred to explicit launch. Code SHIPPED 2026-05-16; direct attack on measured 26 mBPB fast→full gap (highest measurable ROI in queue). (2) **iter175 E=64 + hard-orthogonal expert W_o via Cayley map** — tests user-stated "expert basis" theory (multi-day arch change). (3) **iter178 expand expert rank** — principled-simplest-first test of "more per-expert capacity"; gates iter176. (4a) **iter174 iter173-anneal + K=256** — conditional on iter173 deferred. (iter176 per-dim sigmoid gate CLOSED 2026-05-19 as architecturally redundant with SwiGLU+Down; see the iter176 row.) **DEFERRED**: **iter173 K-jitter weight annealing curriculum** (full 800-step from scratch). Quick test via resume-from-step-900 (2026-05-17, anneal window compressed to [900, 1000]) was INCONCLUSIVE for the design hypothesis because LR warmdown (warmdown_frac=0.72) reduced effective LR to ~18% mean during the 100-step anneal window — model could SEE deeper-K samples but couldn't LEARN from them. Results showed K=4 −17 mBPB improvement (consistency-loss effect, not learning) but K=8-128 uniform +2 mBPB regression (perturbation effect). Net K=16 gate: +0.0021 mBPB worse. Full 800-step iter173 from scratch (anneal window [200, 1000] spans 520 high-LR steps) is the only test that can validate the actual hypothesis but at ~16h cost is deferred behind higher-ROI items. **CLOSED by principleness audit**: iter163c, iter168, iter166 (subsumed into iter172); iter167 (no base — extension term refuted); iter169 (subsumed into iter174); iter164, iter170 spectral-norm, iter171 H96 (previously closed for arch-specificity/budget); **iter176 per-dim sigmoid gate (architecturally redundant with SwiGLU+Down composition; user critique 2026-05-19 — see iter176 row for the principled Simplest argument)**. If iter173 or iter161-QAT-late ties or regresses, accept that asymptotic ceiling rather than escalate to non-agnostic mechanisms. |
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

## Legacy Remaining Queue (Superseded)

| Priority | Item | Change | Principled? | Efficient to test? | Merge/test decision |
|---|---|---|---|---|---|
| **Deferred** | **iter173 full 800-step K-jitter weight annealing curriculum** *(designed 2026-05-17, runs on iter172 base, DEFERRED 2026-05-17 per user)* — resume-from-step-900 quick test (2026-05-17) **CONFIRMED NOT_PROMOTED for that variant**: `final_full_validation val_bpb=1.464400` vs iter172 baseline `1.462898` = **+1.502 mBPB regression**. Root cause per user-stated insight: starting anneal at step 900 was too late — LR warmdown (warmdown_frac=0.72) reduced mean LR to ~18% of peak during the 100-step anneal window, preventing the model from LEARNING deeper representations. The model saw deeper-K samples but couldn't move weights enough to incorporate them. K-sweep showed K=4 −17 mBPB improvement (consistency-loss effect, not learning) but K=8-192 uniform +0.5 to +2.1 mBPB regression (K=16-128 = +2.1, K=192 = +0.5). rho_F improved across all K (0.79-0.84 vs iter172's 0.77-0.92) — proving the curriculum CAN tighten contraction in 100 steps but at the cost of K=16 gate val_bpb. **Full 800-step run from scratch** (~16h, anneal spans 520 high-LR steps) is the only test that can validate the actual design hypothesis but DEFERRED behind iter161-QAT-late (higher measurable ROI). Re-prioritize if other queue items close without moving val_bpb. | Anneal `deq_k_jitter_weights` from iter172 default `(0.50, 0.40, 0.07, 0.03, 0.015, 0.0075)` (E[K]≈22.9, shallow-biased) to a deeper-biased final distribution (default `(0.125, 0.125, 0.125, 0.125, 0.25, 0.25)`, E[K]≈51) linearly over **the final 80% of training**, i.e. window `[anneal_start_frac, anneal_end_frac]·N_steps = [0.2, 1.0]·N_steps`. 3 new Hyperparameters: `deq_k_jitter_weights_final` (empty=disabled), `deq_k_jitter_anneal_start_frac=0.2`, `deq_k_jitter_anneal_end_frac=1.0`. The first 20% of training locks in iter172's baseline FP convergence at shallow K before pushing depth; the final 80% smoothly shifts weights so the model reaches the deep distribution exactly when training stops (no settling buffer needed because final eval IS the target regime). Implementation via new `KShuffleBagSampler.set_weights()` method called inside `deq_k_for_step` only when the discretized (permille-rounded) weights change — ~1000 distinct stages over the window, no per-step bag rebuild churn. | Yes: principled — iter172 achieved rho_F=0.77 at K=128 (vs iter163's 0.92), proving the model is "ready" for deeper K. Shifting K-jitter probability mass toward larger K over training teaches the model to use the deep regime that K-sweep eval actually uses, without paying deep-K compute for the full training run. Simplest mechanism (single linear interpolation per-step + lazy bag rebuild). Arch-agnostic — applies to any iteration-depth scheduling for solvers/DEQs/refinement loops. Doesn't compete with iter172 anchor design (orthogonal axis: anchor SET vs sampling WEIGHTS). | 1000 steps, ~16 h train (~57 s/step ≈ 1.42× iter172 baseline due to deep-K cost: E[step_cost] increases over the anneal window as K_mean grows from 22.9 to 51 = ~2.2× K_mean shift, but actual step-time multiplier is sub-linear due to fixed forward overheads). Peak VRAM ~36.5 GB (unchanged — same model, just sampling deeper depths more often). Test coverage: 9 focused unit tests in `experiments/test_iter173_k_jitter_anneal.py` (annealing helper edge cases + sampler.set_weights + Hyperparameter defaults). | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1000 --deq-k-jitter-weights-final "0.125,0.125,0.125,0.125,0.25,0.25" --run-id=iter173_anneal_<DATE>` (anneal window defaults to [0.2, 1.0] = final 80%; override via `--deq-k-jitter-anneal-start-frac` / `--deq-k-jitter-anneal-end-frac`). **Promotion gate**: full val_bpb ≤ 1.462898 (iter172). Branch outcomes: (a) BPB wins + K-extrapolation (K=128 − K=16) tightens → ship as new defaults; (b) BPB ties + rho_F at K=128 improves → keep iter172 default, log iter173 as "principled but BPB-neutral"; (c) BPB regresses → revert; try alternate final distribution (e.g. uniform `(1/6, 1/6, 1/6, 1/6, 1/6, 1/6)`, E[K]≈60); (d) deep-K compute kills throughput AND BPB → close annealing as a scaling direction at this batch size, revisit only at larger T. **Companion measurement: deq_k_counts trajectory** — sample at s100, s300, s500, s700, s900 to confirm the annealing IS shifting the empirical K distribution as designed (not just by intent). |
| **NOT_PROMOTED** | **iter161-QAT-late on iter172 base** *(launched 2026-05-18 per "clear the queue" directive)* — `final_full_validation val_bpb=1.480545` vs iter172's 1.462898 = **+17.6 mBPB regression**. The pre-QAT RNG drift (val_bpb +17 mBPB at s800 from cudnn non-determinism) was the dominant signal; QAT-late could NOT compensate. Worse, fast→full gap was UNCHANGED at +26 mBPB (iter172: 1.4369→1.4629 = +26; iter161-QAT-late: 1.4541→1.4805 = +26) — proving QAT-late did NOT improve int6 robustness as designed. K-sweep uniformly +18 mBPB across K=16-192 vs iter172. Possible causes: (1) 200-step QAT window too short to retrain weights for quant; (2) STE forward may not exactly match deployment quantizer; (3) the pre-QAT drift dominated. For a clean test of the QAT-late HYPOTHESIS (vs RNG drift), need deterministic launch (torch.use_deterministic_algorithms + cudnn benchmark off) OR resume from iter172 step 800 checkpoint and apply QAT-late for remaining 200 steps. CLOSED for now; revisit only if a deterministic-baseline test shows QAT-late mechanism works. |
| **Pri 1 (NEXT, deferred to explicit launch)** | **iter175: E=64 + hard-orthogonal expert output projections via Cayley map** *(was Pri 2, bumped to Pri 1 after iter161-QAT-late NOT_PROMOTED 2026-05-18)* | Two coupled changes: (a) increase `num_experts` from 16 → 64 (4× more experts, same per-expert rank 64 attn / 96 mlp); (b) parameterize each expert's output projection `W_o^i` as orthogonal-by-construction via Cayley map: `W_o^i = (I − A^i)(I + A^i)^{-1} · W_o_base` where `A^i` is skew-symmetric (`A = -A^T`, parameterized as upper-triangular minus its transpose). The stacked matrix `[W_o^1 ‖ … ‖ W_o^64]` is orthogonal by construction across experts AND across output dimensions. Tests the user-stated "expert basis" theory directly: the per-token achievable polytope dimension expands from 15-D (current E=16) to 63-D (E=64); orthogonality maximizes the polytope volume; combined with full-rank routing the soft-MoE can in principle approach dense-MLP expressiveness when E ≥ D (E=768 would be required to fully span R^D, but E=64 at 63-D is still 4× the current ceiling). | Yes per most-principled-simplest-general directive. **Principled**: replaces refutable soft penalty (iter165 +13 mBPB) with hard orthogonal parameterization. Optimizer CANNOT ignore the constraint. Directly tests the basis-rank hypothesis. **Simplest**: single Cayley map + matrix-inverse Newton step (or `torch.linalg.solve`); standard differentiable orthogonal parameterization. Doesn't add a new loss term — removes the existing `expert_output_diversity_coef` cosine penalty (no need; orthogonality is structural). **General**: any linear-output expert MoE. **RevDEQ compatibility**: Cayley map is differentiable + invertible → reverse reconstruction works. Caveat: per-step int6 quantization roundtrip may slightly degrade orthogonality at deployment; needs verification. | ~Multi-day implementation (Cayley parameterization + 4× expert count + parameter-budget audit for the 16MB artifact cap). Parameter budget: each expert at full-D rank 64 attn ≈ 100K params; 64 experts × 100K = 6.4M for attn alone (vs iter172 13.9M total). Likely needs RANK reduction (e.g., rank 16-32 per expert) to stay under 16MB int6-quantized. Step time: ~2-3× iter172 due to Cayley matrix-inverse per forward + 4× expert compute. Likely needs `--max-training-seconds=600` adjustment OR reduced rank to fit. | Multi-day design + implementation. Promotion gate: full val_bpb ≤ 1.462898 AND artifact ≤ 16 MB. Branch outcomes: (a) BPB wins → "expert basis theory" empirically validated; ship as new architecture; (b) BPB ties + better K-extrapolation → architecture is principled but not BPB-optimal; document as "expressiveness-vs-NTP trade-off"; (c) BPB regresses → close the "more experts + orthogonal" direction; the iter165 lesson generalizes (current E=16 redundancy is functional, not wasteful); (d) artifact too large → reduce expert rank, re-test. Companion measurement: per-token polytope-volume estimate (matrix-determinant of expert output Gram), `mlp_ortho` near 0 by construction, `pertoken_entropy` behavior under hard-orthogonal experts. (iter177 bundle option dropped 2026-05-19 — iter176 CLOSED as architecturally redundant with SwiGLU+Down.) |
| **Pri 3 (after iter175)** | **iter178: expand expert rank (simpler capacity-scaling baseline)** *(NEW 2026-05-17, principled-simplest-first test of "more per-expert capacity")* | Single-knob change: raise `attn_expert_rank` from 64 → 96 and `mlp_expert_rank` from 96 → 128 (or whatever combination fits the 16 MB int6-quantized artifact cap). Per-expert parameter count grows ~50 %; total model size grows ~30 % (vs iter172 13.9M). Tests the "more per-expert capacity" hypothesis at the simplest mechanism — pure rank increase, no new architectural primitive. Principled-first capacity-scaling baseline; complements iter175's basis-expansion direction (iter175 = more orthogonal directions; iter178 = more capacity per direction). | Yes per most-principled-simplest-general. **Principled**: tests "expert capacity is the bottleneck" via the most direct knob. **Simplest**: one config-line change. **General**: applies to any expert architecture. | ~+10-15 % step time (more compute per expert); +30 % artifact size if rank goes 64→96, 96→128 — need to verify int6-quantized artifact still ≤ 16 MB. May need rank tuning. | Run macro (TBD after rank-vs-artifact budget verification): `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1000 --attn-expert-rank=96 --mlp-expert-rank=128 --run-id=iter178_rank_expand_<DATE>`. Promotion gate: full val_bpb ≤ 1.462898 AND artifact ≤ 16 MB. Branch outcomes: (a) BPB wins → capacity scaling is the right axis; (b) BPB ties → current rank already saturates per-expert capacity; (c) BPB regresses → expert-count vs expert-rank trade-off favors current point. (Note: iter176 per-dim sigmoid gate, which was previously gated on iter178, was CLOSED 2026-05-19 as architecturally redundant with SwiGLU+Down — see the iter176 row.) |
| **CLOSED (architecturally redundant with SwiGLU)** | **iter176: per-dim sigmoid gate per expert (low-rank R=16)** *(designed 2026-05-17, implemented 2026-05-18 a0873fe, init-fix iter176b 2026-05-19 6bb59e0, REVERTED 2026-05-19 after architectural critique)* | Original proposal: add `e_i^gated(x) = e_i(x) ⊙ σ(V_e (U_e x) + b_e)` where `V_e U_e` is a low-rank (R=16) projection `R^D → R^D`. Intended to give O(E·D) bits of conditional information per token vs O(E) bits for a per-expert scalar gate. iter176 ran with a broken init (xavier U+V, bias 0 → σ ≈ 0.5 → halved expert outputs at init, violating strict-gen). iter176b fixed the init (LoRA-zero-V, bias +6.0 → σ ≈ 1.0 at init) and was 510 steps into a 1000-step run when closure landed. | **REFUTED 2026-05-19 by architectural-redundancy critique** (user directive). The most-principled-simplest-general gate: **Fails Simplest** — the SwiGLU expert body `(SiLU(x@G^T) * (x@Fc^T)) @ Down_e` already provides per-token per-dim gating capability. SiLU gates the hidden R-dim axis; `Down_e: R^R → R^D` is a learnable D×R matrix that realizes per-input per-output-dim zeroing whenever the SiLU-gated hidden state satisfies `h(x) ⊥ Down_e[d, :]` for that input. Adding `σ(V(Ux)+b)` multiplicative on `out_e` adds 25,344 per-expert params at R_g=16 (U: 16·768 + V: 768·16 + b: 768), which is +11.4% over the full per-expert SwiGLU body of 222,816 params (`3·R·D + R + 2·D = 3·96·768 + 96 + 1536` at iter172 defaults R=96, D=768). This new pathway does not reach expressivity unreachable by the existing parameters — a redundant overhead, not a new mechanism. The empirical iter176 NOT_PROMOTED (+25.8 mBPB) was already consistent with this redundancy (`attn_cv +0.13`, `rho_F +0.10` — gate added training-dynamics noise without expressivity gain). iter176b's 1000-step empirical answer was unnecessary: the architectural answer dominates. **Future lesson**: a new "conditional gating" mechanism for soft-dense MoE experts is only principled if the existing SwiGLU+Down composition cannot realize the conditional mapping. Test BEFORE coding — for full-D LoRA-style experts with `Down_e ∈ R^{D×R}` and SiLU on hidden, the composition is already universal in the per-token-output-dim sense; stacking another sigmoid is over-engineering. Principled "expert-basis" iterations must operate OUTSIDE the SwiGLU+Down span: constrain parameter space (iter175 Cayley orthogonal W_o), expand basis count (iter175 E=16→64), or change the body activation entirely — not stack post-output gates. Reverted commits: a0873fe (implementation), 6bb59e0 (iter176b init fix), 950d098 (hypotheses entry). |
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
| **NOT_PROMOTED** | iter173-resume K-jitter weight anneal (compressed [900, 1000], 2026-05-17, resume from iter172 step 900) | Resume from iter172 step 900 (only available checkpoint earlier than step 1000) with K-jitter weight annealing compressed to anneal window `[0.9, 1.0]·1000_steps = [900, 1000]`. Final K-jitter weights `(0.125, 0.125, 0.125, 0.125, 0.25, 0.25)` interpolated from iter172 default `(0.50, 0.40, 0.07, 0.03, 0.015, 0.0075)` linearly over the 100 resume steps. Tests whether the K-jitter annealing curriculum hypothesis (push model to use deeper K) can move val_bpb in a quick test. | Diagnostic-quality test; not principled-as-designed because the LR warmdown (warmdown_frac=0.72) reduces effective LR to ~18% mean over the anneal window → model sees deeper-K samples but cannot LEARN deeper representations from them. The 100-step compressed anneal is structurally different from the original 800-step gradual anneal design. | 100 train steps (~70s/step due to deeper-K cost in anneal window), ~117 min train + ~70 min full val = ~3.1h total. ~2.6× more deep-K samples (K=64/96/128 cumulative 81 vs iter172's 50 at step 1000) — annealing fired correctly at the K-distribution level. | `NOT_PROMOTED` by validation-BPB hard gate. **final_full_validation val_bpb=1.464400** vs iter172 1.462898 = **+0.001502 = +1.5 mBPB regression**. K-sweep pattern reveals the trade-off: K=4=1.6062 (−17 mBPB vs iter172, dramatic improvement from consistency-loss effect), K=8=1.4761 (+1.6), K=16=1.4737 (+2.1), K=24-128 uniform ≈ +2.1 mBPB, K=192=1.4743 (+0.5). rho_F across K=4-192 = 0.79-1.06 vs iter172 0.78-2.70 — curriculum genuinely tightened contraction (especially at K=4 where iter172's rho was 2.70 → iter173-resume 0.89). But contraction came at the cost of K=16 gate expressiveness (+2.1 mBPB). Confirms user-stated insight: "annealing starting at step 900 is too late to let the model learn deeper representation" — LR warmdown prevented learning, leaving only perturbation effects. The full 800-step iter173 from scratch (anneal window spans 520 high-LR steps) remains the only valid test of the design hypothesis; DEFERRED behind iter161-QAT-late per user 2026-05-17. Log: `logs/iter173_resume_anneal_from_step900_20260517.txt`. Run-id: `iter173_resume_anneal_from_step900_20260517`. |
| **PROMOTED** | iter172 K=8 shallow anchor + recursive consistency (2026-05-17, supersedes iter163) | Same recursive anchor formula as iter163 (`λ · Σ_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²`) on iter152 z_stack, but with explicit prefix-anchor set `(8, 16, 24, 32, 64, 128)` — the K=8 shallow anchor (gap=8 to K=16) gives consistency loss a non-trivial pair at K_sampled=16 (~88% of training steps) without collapsing rho_F like iter170/171's K=4 (gap=12) did. NO extension term (iter163's `‖z_K − z_{K+Δ}.detach()‖²` removed; refuted by iter163c v2 at +22 mBPB regression for Δ=1 form, 1.6× step time for Δ=K_train). Inherits iter163's default K-jitter `(16,24,32,64,96,128)` weights `(0.50,0.40,0.07,0.03,0.015,0.0075)`. | Yes: simplest-general per CLAUDE.md most-principled-simplest-general directive. Reuses iter163's PROMOTED recursive formula; only the anchor SET changes from `()` (fallback to jitter set, jitter-min=16) to `(8,16,24,32,64,128)` (explicit K=8 shallow). The K=8 choice is principled: gap=8 to next-deeper K=16 matches iter163's healthy regime (jitter-mean K ≈ 23.6, so anchor pairs operate at the typical training depth, preserving optimization landscape that iter170/171's K=4 gap=12 broke). Architecture-agnostic. | 1000 steps, 11.1h train (40.0s/step ≈ 1.00× iter163 baseline modulo K-sweep diagnostic overhead), peak VRAM 36.5 GB. fast_val_k_sweep_set diagnostic (CLI knob: `--fast-val-k-sweep-set "8,128"`) confirmed healthy fast convergence at every val checkpoint (k8=k16=k128 all identical, no degeneracy). | **PROMOTED at full_full_validation val_bpb=1.462898 (vs iter163's 1.471598, Δ −0.008700 = −8.7 mBPB, 39× iter163's 0.22 mBPB margin over iter152)**. **K-sweep dominates uniformly**: K=4=1.6235 (vs iter163 1.8204, −197 mBPB), K=8=1.4745 (vs 1.5478, −73 mBPB), K=16=1.471556 (vs 1.481036, −9.5 mBPB), K=24=1.4720 (vs 1.4805, −8.5 mBPB), K=64=1.4722 (vs 1.4814, −9.2 mBPB), K=128=1.4722 (vs 1.4814, −9.3 mBPB). **rho_F at K=128=0.7709 vs iter163's 0.9194** (principled FP-convergence dramatically improved; consistency loss progressively contracted spectral radius over training). iter_conv_rel K=128=0.0115 (PASS empirical gate < 0.05). Val trajectory: s200 Δ −32 mBPB, s400 Δ −13, s600 Δ −14, s800 Δ −16 (lead GREW), s1000-fast=1.4369 (NEW project fast-val record, was iter163's 1.4492). Consistency_anchor_loss 0.025 at s1000 (vs iter163's saturated 0.0008) — still actively pushing FP-convergence at training end. **Routing-quality side-effects**: attn_cv K=128=0.315 vs iter163's 0.344 (BETTER), mlp_ortho K=128=0.231 vs iter163's 0.250 (BETTER). Post-int diagnostic: 2 refuted advisories (`router_collapse_advisory: attn_min_share=0.0370 < 0.0375` and `mos_router_collapse_advisory: mos_ntp_min_share=0.150 < 0.200`) — informational only, both prescription classes empirically refuted (3-iter closure iter158/162/165 for router; CV-as-loss removal 2026-05-15 for MoS). parcae_a_min asymptote=0.170 (vs iter163's 0.229, monotonically lower throughout training — the consistency loss drove deeper damping basin). parcae_recon_amp_log10=18.44 at s1000 (vs iter163's 15.4, +3 orders) — still 20 orders below bf16 overflow limit, but the trajectory rate suggests submission-run (4500-step) might approach bf16 limits. Future submission validation should monitor. Artifact 7.40 MB (smallest of recent iters). Promotion-propagation completed in same commit: `Hyperparameters.deq_prefix_anchor_set` default flipped `()→(8,16,24,32,64,128)`, `GPT.__init__` signature default updated, `experiments/test_iter163_consistency.py::test_hyperparameter_defaults_match_promoted_iter172` updated to assert iter172 default (renamed from `_iter163c`), CLAUDE.md "Current Architecture" + Standing Directive updated. Log: `experiments/training_logs/baseline.log` (promoted via `experiments/update_results.sh --promote`). Run-id: `iter172_full_1000step_20260516`. |
| **PROMOTED** (superseded by iter172 2026-05-17) | iter163 multi-K consistency loss for natural FP-convergence learning (2026-05-15, hybrid CM-paradigm) | Two principled loss terms: (a) recursive anchor `λ_a · Σ_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²` reusing iter152 z_stack; (b) extension `λ_e · ‖z_K − z_{K+Δ}.detach()‖²` extending from z by Δ no-grad Parcae two-state iterations. λ_a=λ_e=0.1, Δ=K_train. Targets ρ(J_F)<1 by *learning* FP convergence via the FP equation T(z*)=z* directly, not via refuted soft penalty (iter155 Lyapunov). Architecture-agnostic per CLAUDE.md most-principled-simplest-general directive. | Yes: principled per Hartman-Grobman + arch-agnostic + reuses existing prefix-anchor infrastructure. Avoids iter144 IFT refutation. Avoids iter155 Lyapunov refutation. Distinct from iter146-147 prefix-anchor NTP. | 1000 steps, 14.0h train (50.5s/step = 1.6× iter152 baseline due to extension forward), peak VRAM 36.5 GB. | **PROMOTED at full val_bpb=1.471598 (vs iter152 1.471820, Δ −0.000222 = −0.22 mBPB)**. Promoted by smallest possible margin but real improvement. **K-sweep best K=24=1.4805 (vs iter158 1.4814, −0.0009)**, K=128=1.4814 (vs iter158 1.4833, −0.0019). **K-extrapolation gap (K=128 − K=16) = +0.0004 — 4× TIGHTER than iter158's +0.0018** (the consistency mechanism's principled goal achieved). val_bpb trajectory: s200=2.0027 (deficit), s400=1.6374, s600=1.5326, s800=1.4956, s1000-fast=1.4492 (lead held). rho_F: trained from 1.30 (init) → 0.92 (s1000) — model LEARNED contraction. consistency_anchor 1.05 → 0.0008, consistency_ext 1.73 → 0.0008 (both saturated by step 80). Diagnostic: ONE failure (`router_collapse_advisory: attn_min_share=0.0339 < 0.0375` — informational only after the 2026-05-15 prescription refutation). Artifact 7.48 MB (smallest of recent iters). FIRST project mechanism that *trains* FP convergence rather than enforcing or measuring it. Promotion-propagation completed in same commit: Hyperparameters defaults to 0.1/0.1/0, GPT.__init__ defaults updated, focused test renamed to `test_hyperparameter_defaults_match_promoted_iter163`, CLAUDE.md "Current Architecture" updated. Log: `experiments/training_logs/iter163_full_1000step_20260515.log`. Run-id: `iter163_full_1000step_20260515`. |
| Done | iter165 routing-balance bundle (doubled coefs + router_bias_update, 2026-05-15) | Bundled retest of the routing-balance prescription class on the iter158 base: `--router-pertoken-entropy-coef=0.20` (doubled from 0.10 default), `--expert-output-diversity-coef=0.60` (doubled from 0.30), `--router-bias-update=1` (slow generic usage-prior controller — the post-int prescription's recommended fix). NO redundant `--router-load-cv-coef` floor (CV is subsumed by EMA-anchored balance loss; user-identified as redundant). Tests whether stronger routing-quality pressure can move BPB; companion to iter158 (reverse-KL alone) and iter162 (entropy-only push, killed before completion in favor of this bundle). | Yes: targets the post-int prescription directly via the recommended router_bias_update mechanism. Removes prior iter162's confound by adding bundled regularizers in one A/B vs iter158/152. | 1000 steps, 8.82 h train, 31.75 s/step, peak VRAM 36.50 GB. Final val took ~50 min (full-validation across all val tokens at K=16 + sliding window). | `NOT_PROMOTED` by validation-BPB hard gate. **final_full_validation val_bpb=1.484948** (Δ +0.0131 vs iter152 1.471820 — strong regression). K-sweep: K=4=1.8724, K=8=1.5878, K=16=1.4924, K=17=1.4909, K=24=**1.4889 (best)**, K=32=1.4893, K=37=1.4895, K=64=1.4899, K=113=1.4900, K=128=1.4900. iter152 K=24 best=1.4797 → iter165 +0.0092 worse. Trajectory: s200=1.9655 (best of all iters at s200, Δ −0.029 vs iter152), s400=1.6466 (+0.017, regressed), s600=1.5417 (+0.007), s800=1.4977 (+0.0035, gap narrowing), s1000-fast=1.4577. Routing health: mlp_ortho=0.2100 (vs iter152's 0.1934, slightly worse), attn_ema_min=0.0342 (= iter152). Post-int gate: ONE failure (`router_collapse: attn_min_share=0.0317 < 0.0375` — actually slightly worse than iter158's 0.0327, despite the doubled push). **rho_F at K=128=0.867 < 1 (PASS, FP convergence gate)**, iter_conv_rel=0.011<<0.05 (PASS). **Notable rho_F anomaly: K=16=1.027 (>1!), bracketed by K=8=0.886 and K=17=0.852** — only K=16 spike, likely either (a) power-iteration estimator noise on near-marginal Jacobian or (b) the most-trained K (490/1000 steps) specialized into a flatter contraction zone. Asymptotic FP convergence remains healthy (K=128 < 1). **High recon_amp_log10=12.88 (vs iter155's 7.6)**: doubled regularizers indirectly drove low Ā regime (more β·T contribution → higher reverse-reconstruction amplification). Within precision tolerance for bf16 (~0.01 rel), but quantization tax wider. **Routing-balance push class CLOSED**: iter158 (reverse-KL alone, +0.003), iter162 (entropy-only, killed mid-run), iter165 (bundled, +0.013) — three-iter empirical refutation that the diagnosed `attn_min_share` gate is the BPB bottleneck at iter152's operating point. Pushing harder on routing-balance metrics ACTIVELY HURTS BPB. The post-int gate at 0.0375 is an arbitrary threshold, not a correctable defect. Log: `run.log` (will archive as `experiments/training_logs/iter165_bundle_clean_20260514.log`). Run-id: `iter165_bundle_clean_20260514_1713`. |
| Done | iter155 Lyapunov-on-F + lip_ub_F diagnostic (corrected 2026-05-13) | Architecture-agnostic empirical-tier contraction work: redirect Lyapunov FD probe from `T_θ` (iter147) to the actual two-state Parcae cycle map `F`. New flag `lyapunov_target=iteration_F` un-detaches Ā in the FD branch so gradients flow to Parcae damping AND transition parameters jointly. K-sweep emits 3-way decomposition: `lip_ub_T` (transition diagnostic), `lip_ub_S` (single-state advisory surrogate), `lip_ub_F` (gate-aligned operator-norm probe). New post-int gate triggers on `lip_ub_F >= 1`. Joint cycle residual `fp_residual_F` paired with `lip_ub_F` in `fp_bound`. Mid-run code patch (per user directive 2026-05-13): `lip_ub_F` and `fp_residual_F` made always-on at every FP eval (fast-val + every K-sweep row), no longer gated by `fp_lip_fast_val_every` or `EvalProfile.lip_probe_set` — patch landed AFTER iter155 completed so iter153/iter154 onward gain it. | Yes: principled fix for the iter145r/149/152 contraction debt. Penalty pressures `σ_max(J_F)` directly via cross-block coupling `D_β·J_T`. **Confirmed at the gradient level**: per-dim Ā migrated 0.700 → 0.481 mean / 0.333 min over 1000 steps (β = 1−Ā: 0.300 → 0.519 mean = 1.7× more damping). | 1000 steps, 6.81 h train, 24.43 s/step, peak VRAM 34.92 GB. **18% faster than iter152** (which carried `deq_prefix_anchors=1` overhead). | `NOT_PROMOTED` by validation-BPB gate. **final_full_validation val_bpb=1.481344 vs iter152 1.471820 (Δ +0.0095, regression)**. K-sweep: K=4=1.8633, K=8=1.5756, K=16=1.4886, K=17=1.4874, K=24=**1.4861 (best)**, K=32=1.4867, K=37=1.4869, K=64=1.4873, K=113=1.4875, K=128=1.4875. iter152 K=24 best=1.4797 → iter155 +0.0064 worse. Post-int gate failures (3): `router_collapse` (attn_min_share=0.0362 < 0.0375 — slightly better than iter152's 0.0315 but still below gate), `expert_collapse` (mlp_ortho=0.5098 > 0.5 — markedly worse than iter152's 0.1934), `local_contraction_failed` (`lip_ub_F=17.08, lip_ub_S=16.07, lip_ub_T=25.84` at K=128 — gate-aligned object never fell below 15 across the K-sweep, oscillating 15-20). Train_loss had a consistent **−0.27 lead vs iter152** across the entire trajectory (s80→s940), but did NOT translate to val_bpb: iter155 val_bpb trajectory s200→s400→s600→s800: `1.9791 → 1.6405 → 1.5375 → 1.4969`; iter152 `1.9942 → 1.6298 → 1.5346 → 1.4942` (gap was −0.015 at s200, flipped to +0.011 at s400, settled at +0.003 by s600). Lyapunov-on-F at coef=0.005 with random_fd estimator successfully trained Ā but: (a) the increased damping moved the model into a different optimization regime that hurt val_bpb generalization, (b) random_fd directional probe is not a tight bound on `σ_max(J_F)` — the K-sweep operator-norm power-iteration estimate kept reading 15-20 throughout. **Aug-Lagrangian escalation BLOCKED** by BPB regression > +0.005 stop condition. Soft-pressure tier exhausted. iter156 (hard governor 156a or learned `c·Δ` gain 156b — formal-tier mechanism) is the next required step. Artifact 8.08 MB (well under 16 MB). Log: `experiments/training_logs/iter155_1000step_20260513.log`. Checkpoints: `experiments/checkpoints/iter155_1000step_20260513_0135/{step_000900,step_001000,latest}.pt`. |
| Done | iter155 Lyapunov-on-F + lip_ub_F diagnostic | See Tested Iterations row 92. **NOT PROMOTED**: val_bpb 1.481344 vs iter152 1.471820 (Δ +0.0095); 3 health gates fail (router_collapse 0.0362, expert_collapse 0.5098 mlp_ortho, local_contraction_failed lip_ub_F=17.08). Soft-pressure tier exhausted; iter156 escalation required. **CONFOUND DISCOVERED 2026-05-13**: iter155 inadvertently ran with `deq_prefix_anchors=False` (Hyperparameters default never propagated after iter152's promotion) — so iter155 vs iter152 actually mixed `+Lyapunov-on-F` (target intervention) with `−prefix-anchors` (silent regression of iter152's promoted feature). Estimated train_loss penalty from −prefix-anchors alone: ~−0.27 (consistent across iter153 and iter155 trajectories vs iter152). The +0.0095 BPB regression is mixed between "Lyapunov-on-F hurt" and "lost prefix-anchors benefit". Default flipped to `True` 2026-05-13; iter157 onward will inherit the fix. |
| 1.5 | **iter152 K-sweep measurement on saved checkpoint** *(NEW 2026-05-13, scheduled between iter153 and iter157)* | Resume from `experiments/checkpoints/iter152_prefix_1000/step_001000.pt` and run 1 step + full K-sweep with the iter155-corrected diagnostic suite (lip_ub_T/S/F, fp_residual_F always-on per commit `e7e02c6`). iter152's training log only has the legacy `lip_ub:` field (= `lip_ub_T`); the gate-aligned `lip_ub_F` was never measured for the iter152 baseline. This run gives the missing reference point. | Yes: makes the iter155 → iter152 contraction comparison decisive (currently we have iter155 lip_ub_F=17.08 but no iter152 baseline number to compare against). Resolves the speculative "iter152's lip_ub_F is probably 12-17 by extrapolation" with an actual measurement. | ~10-15 min on 2× L40S with 1 train step + K-sweep. Resume from saved checkpoint, set `--iterations=1001` (resume at step 1000, do 1 more step to populate FP, then K-sweep fires). | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1001 --resume-from=experiments/checkpoints/iter152_prefix_1000/step_001000.pt --eval-profile=diagnostic --val-loss-every=1000000 --run-id=iter152_kresweep_20260513`. Output: K-sweep table with `lip_ub_F` per K. No promotion implied — pure measurement run. |
| Done | **iter153 reverse-KL balance** *(default-on 2026-05-13)* | Swap the EMA balance loss form from forward `KL(EMA‖U)` to reverse `KL(U‖EMA)`. Forward KL weights each per-expert term by `EMA_i`, so a dead expert's contribution is `EMA_i · log(EMA_i/U_i) → 0` even when the log ratio is large — the gradient on the tail vanishes. Reverse KL weights by `U_i`, so each expert contributes equally regardless of its mass, and the dead-expert gradient is `−U_i/EMA_i`, diverging as `EMA_i → 0`. One-line code change at `train_gpt.py:2856` gated by `use_reverse_kl_balance` (Hyperparameters default True, iter153 default-on). Provisional default-on for testing; reverts to forward KL if iter153 regresses BPB. | Yes: principled fix for the asymmetric tail weighting diagnosed in the iter152 post-int min-share failure (`attn_min_share=0.0315 < 0.0375`). For the diagnosed numbers (E=16, one dead expert at half its fair share), reverse KL gives ~3.4× stronger gradient than forward KL; the gap widens as EMA → 0. No new coefficient; existing `router_ema_balance_coef=0.30` carries the same magnitude budget. | 1000 steps, ~6.78 h train, ~24.4 s/step, peak VRAM ~34.86 GB (iter146 baseline; reverse KL adds <1% compute — one extra reciprocal per expert per step). | Run with no extra flags; `--use-reverse-kl-balance=0` recovers the forward-KL baseline for direct A/B. Promotion condition: full val_bpb ≤ iter152's `1.471820` AND post-int `attn_min_share ≥ 0.0375`. If BPB regresses, revert `Hyperparameters.use_reverse_kl_balance` to False and re-queue. If BPB ties and min-share improves, promote and document the policy match. **Promoted to Pri 1 on 2026-05-13** (was Pri 2): iter155's contraction A/B closed as NOT-PROMOTED, so iter153 (the routing-balance fix) becomes the next BPB-promotion candidate. iter156 (formal-tier hard governor) is the principled escalation for the contraction debt iter155 confirmed. |
| Closed | iter157 (Lyapunov-on-F with power_jvp_F estimator) | **Closed 2026-05-13**: superseded by iter158 bundled rerun, then closed entirely after iter152 K-sweep measurement empirically refuted soft Lyapunov as principled. The principled framework (Tier 1+2 redesign) targets ρ(J_F) < 1, NOT σ_max < 1 — making the Lyapunov-on-F penalty over-restrictive regardless of estimator quality. No iter157 needed. |
| Done | **iter158 reverse-KL on corrected baseline** *(NOT_PROMOTED 2026-05-13, val_bpb=1.475 vs iter152 1.472)* | Bundled retest of iter153's reverse-KL hypothesis on the corrected baseline (deq_prefix_anchors=True default after promotion-propagation fix). NO Lyapunov penalty — refuted by iter155 evidence + Tier 1+2 theoretical analysis (σ_max is over-restrictive proxy; ρ is the necessary-AND-sufficient condition). Inherits all defaults: deq_prefix_anchors=True, use_reverse_kl_balance=True, lyapunov_coef=0. Tests reverse-KL as a one-variable change vs iter152. Tier 1+2 K-sweep emits new gate-relevant metric `rho_F` (spectral radius via straight power iteration on J_F) alongside diagnostic `lip_ub_F`. | Yes: clean test of the iter153 hypothesis on the actually-correct baseline. Eliminates the prefix-anchors confound that handicapped both iter153 and iter155. The Tier 0 decision (drop Lyapunov) is principled per Hartman-Grobman: ρ(J_M) < 1 is necessary AND sufficient for asymptotic local convergence; σ_max < 1 is sufficient but over-restrictive; iter152 baseline empirically demonstrates this (σ_max ≈ 17 with iter_conv_rel ≈ 0.02, clearly converging). Architecture-agnostic: same condition applies to any iteration mechanism. | 1000 steps, ~6.8 h train, ~24.4 s/step, peak VRAM ~34.86 GB. No extra Lyapunov compute. K-sweep adds ~8 JVPs per K for the new rho_F probe (8-iter power iteration on J_F) — small marginal cost. | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1000 --checkpoint-every=100 --run-id=iter158_revKL_cleanbaseline_<DATE>`. Promotion condition: full `val_bpb ≤ 1.471820` (iter152) AND `rho_F < 1` at K=128 (theoretical) OR `iter_conv_rel < 0.05` at K=128 (empirical). Branch outcomes: (a) BPB win + convergence signals OK → ship reverse-KL on corrected baseline as new defaults; (b) BPB tie + convergence OK → A/B inconclusive, defer; (c) BPB regress → reverse-KL doesn't help even on corrected baseline; (d) convergence signals fail → escalate to formal-tier mechanism (spectral norm on T_θ component layers, NOT soft penalty). |
| Closed | iter160 / iter162 / iter165 routing-balance push class | **Closed 2026-05-15 by 3-iter empirical refutation**: iter158 (reverse-KL alone, Δ +0.003), iter162 (entropy-only push, killed mid-run after seeing same regression pattern), iter165 (bundled doubled coefs + router_bias_update, Δ +0.013). All three confirm: pushing on the post-int `attn_min_share` prescription (the iter152-promoted "tech debt") ACTIVELY HURTS BPB. **FAILED Principled test** per the most-principled-simplest-general directive (CLAUDE.md Standing Directives 2026-05-15): `attn_min_share > τ` is a symptom correlated with collapse, not a root-cause invariant for BPB. The 0.0375 fair-share threshold is an arbitrary diagnostic, not a correctable BPB defect at iter152's operating point. iter160 (per-expert NTP) is closed unrun because iter165 already pushed the same direction harder via router_bias_update + doubled regularizers and only regressed further. Re-open ONLY if a fundamentally new routing-balance mechanism emerges (e.g., explicit hard-routing dispatch gated outside RevDEQ, or dynamic expert pruning with re-spawn). The post-int prescription `_prescribe_failure_fix.router_collapse` was renamed to `router_collapse_advisory` 2026-05-15 with the empirical-refutation note in-code (commit `4af904f`). |
| Done | **iter163 multi-K consistency loss (Final-K-as-GT, hybrid recursive + extension)** *(PROMOTED 2026-05-15, val_bpb=1.471598 = current champion)* | User-directed Option-A++ design for natural FP convergence. Algorithm: (i) Forward to FINAL K (no-grad) to produce `z_GT` (ground-truth FP); (ii) Forward K_train WITH gradient (TBPTT), save last `K_bptt` intermediates `z_k`; (iii) Add consistency loss `λ · Σ_k ||z_k − z_GT.detach()||²`. Each TBPTT pass anchors to the converged-FP target → model must learn to converge BY K_train. New Hyperparameters: `multi_k_consistency_coef` (default 0, enable 0.1), `multi_k_consistency_target_K` (default 64), `multi_k_consistency_anchor_steps` (default 3 = K_bptt). | **Architecture-agnostic ✓** (no weight reparameterization). **Implicitly enforces ρ(J_F) < 1** without direct rho_F penalty: consistency loss can only minimize if iteration map IS contractive. **Avoids iter144 IFT failure** (no IFT gradient — pure self-distillation across iterations). **Avoids iter155 Lyapunov failure** (no operator-norm penalty). | ~3× total step time (forward to K_final=64 vs K_train=16 dominates) → ~22h vs iter158's 7h. New code: ~200 lines. | Run macro: `--multi-k-consistency-coef=0.1 --multi-k-consistency-target-K=64`. Promotion: full val_bpb ≤ baseline AND **K-sweep monotone** (val_bpb(K=128) ≤ val_bpb(K_train) — deeper K cannot hurt; the principled convergence-learned property). Contingent: implement after iter158/iter160 close to know if natural FP convergence is needed. |
| **1** | **iter163c: anchor-on-deepest consistency loss (coarse-only after OOM fix)** *(SHIPPED 2026-05-16, supersedes iter163 extension AND subsumes the planned iter166)* | Compute z_{K+1} = F(z_K, x0) via ONE no-grad Parcae iter (~+1 % step time). Loss: `L = anchor_coef · mean_i ‖z_i − z_{K+1}.detach()‖²` for every z_i in the iter152 coarse prefix-anchor z_stack. Anchor-on-deepest target z_{K+1} is strictly stronger than pair-with-next (z_{i+1}): signal magnitude is "distance from FP" not just one-step residual, AND no trivial-zero collapse risk because target is input-driven (z_{K+1} = F(z_K, x0), not constant). **Note (2026-05-16)**: a first attempt augmented anchors with TBPTT-window iterations {K-bptt_k+1, ..., K-1} for per-iter pairs at every K_sampled, but OOM'd at step ~13 on dev L40S (~6 anchors × 3 bptt × T=2048 SDPA backward activations > 44 GB VRAM). Reverted to coarse-only design — the user's original question only asked for the anchor TARGET change, not the TBPTT augmentation; the augmentation was an overcomplication. | Yes per most-principled-simplest-general directive. **Principled**: anchor-on-deepest pulls every gradient-carrying z toward the model's best FP estimate; the FP equation z = F(z) is the universal asymptotic condition; no trivial-FP risk because target is input-driven. **Simplest**: one inline no-grad iter (~10 lines); delete the entire `_consistency_extend_no_grad` machinery (~30 LOC removal); NO TBPTT augmentation. **General**: any iterative model has this structure (z_{K+1} target works for any DEQ solver, recurrent layer, refinement loop). Trade-off: |z_stack| pair contributions per step at coarse depths (e.g. {16,24,32,64} → 4 anchors at K_sampled=64) vs iter163's 1 boundary pair at Δ=K_train. | ~1.01-1.02× iter152 step time. Memory: marginal — same as iter152 baseline (no anchor augmentation; one no-grad forward iter for z_{K+1}). | Run macro: `--multi-k-consistency-anchor-coef=0.1` (default; extension knobs removed). Promotion: full `val_bpb ≤ 1.471598` (iter163 champion) AND K-extrapolation gap (K=128 − K=16) `≤ +0.0004`. Branch outcomes: (a) BPB tie + K-gap held → ship as default; (b) BPB regresses by > +0.001 → revert to iter163 hybrid; (c) K-gap widens beyond +0.001 → tune anchor_coef higher (0.2) before reverting. **Smoke v2 confirmed step 0 healthy with σ_max_F=4.22 emission active (iter168 diagnostic restoration working)**; awaiting full smoke completion before 1000-step launch. |
| **2** | **iter168: rho_F multi-seed + restore σ_max(J_F) diagnostic** *(SHIPPED 2026-05-16)* | Two deliverables in one commit: (a) **multi-seed power iteration** — `_rho_F_at_saved_fp` now runs `n_seeds=4` independent power-iteration trials and returns the median, closing the K=24/32 estimator-artifact issue (iter163 K-sweep showed rho_F=1.04 / 1.15 at K=24/32 with K=128=0.92 — single-seed power iteration on non-symmetric J was undersampled); (b) **restore `sigma_max_F = σ_max(J_F)` as DIAGNOSTIC** via new `_sigma_max_F_at_saved_fp` (NOT a gate, NOT a penalty, NOT a prescription input) — emitted in fast-val log and K-sweep table alongside rho_F. Restores robustness/basin-size visibility lost in 2026-05-15 removal; prerequisite for iter161-QAT-late which depends on weight-perturbation tolerance bounded by σ_max. Path-spectral (rho_F at multiple K depths) was deferred to a follow-up iter — the saved-FP probe is a single-point measurement; per-K path requires re-running forward at each K which doubles the K-sweep cost. Skipped for principled simplicity. | Yes per most-principled-simplest-general directive. **Principled**: no single scalar is necessary AND sufficient for "model is operationally good" — rho_F characterizes asymptotic local convergence only; σ_max characterizes per-step contraction, basin size, robustness to weight/input perturbation. These are independent operational properties. Multi-seed averaging is principled estimator hardening (the dominant eigenvalue is invariant; per-seed convergence speed depends on init projection). **Simplest**: probe-only changes (~150 LOC), no training-time code touched, no loss term added; path-spectral correctly deferred to keep this iter focused. **General**: applies to any iterative model. Crucially: the 2026-05-15 σ_max removal correctly eliminated it as a GATE/PENALTY/PRESCRIPTION input, but wrongly eliminated it as a DIAGNOSTIC. This iter restores only the diagnostic role. | <1 % step time (probes are fast-val + K-sweep only); multi-seed adds ~3× rho_F probe cost which is still negligible in absolute terms (one extra power iteration per fast-val emission). | Test-only iter; no model change. Output: hardened rho_F + restored sigma_max_F diagnostic in K-sweep tables and fast-val log lines. Promotion: 7/7 focused tests pass (rho_F multi-seed, sigma_max finite + ≥ rho_F bound, fast-val emission, K-sweep emission, negative assertion that sigma_max is NOT in `_prescribe_failure_fix`). 106/106 broader regression tests pass. **Prerequisite for iter161-QAT-late**. |
| **3** | **iter161-QAT-late: deterministic STE int6-SDCLIP fake-quant on last 20 % of training** *(SHIPPED 2026-05-16)* | For the last `(iterations - qat_late_start_step)` training steps, route each `CastedLinear.weight` with `numel > 8192` through a straight-through-estimator (STE) wrapper. Forward applies the EXACT same per-row int6 SDCLIP quantization as `encode_scored_artifact.quantize_int6_sdclip` (sd-clip factor K=12.85, clamp to int6 range [-32, 31], scale per row); backward passes gradient through identity. Module-global `_QAT_LATE_STATE.set_step(step)` is called each training iteration; `_QAT_LATE_STATE.active()` returns True iff `start_step >= 0 AND current_step >= start_step`. Default `qat_late_start_step=-1` (OFF). Targets the **~22 mBPB fast→full gap** directly by training the model to BE quant-robust against the EXACT deployment quantization, not patching frozen weights post-hoc with PTQ. | Yes per most-principled-simplest-general directive. **Principled**: root-cause attack. PTQ optimizes `argmin_W̃ ‖W·X − W̃·X‖²` (per-layer reconstruction) with W fixed; QAT optimizes `argmin_W E[L(quant(W), X, y)]` (joint over the quantization-constrained weight space). The model trains against the EXACT artifact codec quantizer (per-row SDCLIP, not max-clip). **Simplest**: one custom STE Function + one global state class (~80 LOC); no SVD machinery, no per-layer calibration walks, no two-mechanism bundle. **General**: applies to any nn.Linear-shaped weight; bit-width and gating-step are knobs. **RevDEQ-safe**: forward depends only on weight values (no RNG, no random noise) → reverse reconstruction works (distinct from iter20's stochastic noise injection refuted 2026-04). Bonus: σ_max(J_F) diagnostic (iter168) bounds the weight-perturbation amplification a model can absorb, so QAT sensitivity is now observable. | ~1.02-1.05× iter163c step time during the QAT window (STE adds one `round/scale` per matrix forward; STE is identity in backward so no graph growth). Memory: zero overhead. | Run macro: `--qat-late-start-step=800` (last 200/1000 steps = 20 %). Promotion: full `val_bpb ≤ 1.471598` AND artifact ≤ 16 MB. Implementation: 15 focused tests pass + 121 broader regression. Branch outcomes after iter163c base is known: (a) val_bpb ≤ baseline → ship as default; (b) val_bpb regresses > +0.005 → keep iter163c, document trade-off; (c) further gains possible → try int4 bits in a follow-up. |
| Subsumed | iter166: anchor-on-deepest-sampled-K | **ABSORBED INTO iter163c 2026-05-15**: the anchor-on-deepest design principle (every gradient-carrying z anchored to the most-converged available target) was incorporated directly into iter163c after user-raised principled question about "z_{i+1} weak signal vs z_{K_max} strong signal" (also addresses trivial-zero collapse concern). iter166 standalone is no longer needed because iter163c IS the anchor-on-deepest mechanism — target is z_{K+1} (one no-grad iter beyond deepest sampled K). If iter163c saturates, the next iter is iter167 (Δ-curriculum on the anchor target's depth). |
| Closed | iter167: constant Δ=8 anchor-target depth | **CLOSED 2026-05-17 by post-iter172 principleness audit**: iter167's mechanism (set `multi_k_consistency_extension_delta=8` on the no-grad extension forward) requires iter163c's extension term to exist. iter172 PROMOTED 2026-05-17 has NO extension term — the extension class was refuted by iter163c v2 (+22 mBPB regression at Δ=1, 1.6× step time at Δ=K_train) and entirely removed. iter167 therefore has no base to run on. Re-open only if a new principled mechanism reintroduces an extension target. |
| **Pri 4b (conditional on iter173 deferred-test)** | **iter174: iter173 anneal + K=256 K-jitter extension** *(NEW 2026-05-17, subsumes iter169 per principleness audit; effectively also DEFERRED while iter173 is)* | Compose iter173's K-jitter weight annealing with the principled K=256 jitter-set extension into a single 1000-step run. Final K-jitter set = `(16, 24, 32, 64, 96, 128, 256)`; final weights e.g. `(0.10, 0.10, 0.10, 0.10, 0.15, 0.225, 0.225)` → E[K] ≈ 70 (vs iter173 final E[K] ≈ 51). Conditional: runs ONLY if iter173 promotes (if iter173 regresses or ties, the K-jitter annealing class is closed and K=256 alone is too weak per the principled audit — K=128 already passes the rho_F gate). | Yes per most-principled-simplest-general. **Principled**: composes two changes targeting the same axis (deeper K-jitter sampling). K=256 standalone (the original iter169) was MARGINAL — per the iter172 K-sweep, K=128 already passes the rho_F gate (0.77); K=256 alone would fire too rarely (3-4×/1000 steps) to move the needle. Folded into the iter173 annealing schedule, it becomes a meaningful share (~22% of late-training samples). **Simplest**: one combined launch macro instead of two iters. **General**: any K-jittered iterative model. | ~1.5× iter173 step time due to K=256 samples firing more often after annealing concentrates mass there. | Run macro: `torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=1000 --deq-k-jitter-set "16,24,32,64,96,128,256" --deq-k-jitter-weights "0.50,0.40,0.07,0.03,0.015,0.0075,0.00375" --deq-k-jitter-weights-final "0.10,0.10,0.10,0.10,0.15,0.225,0.225" --run-id=iter174_anneal_k256_<DATE>`. Promotion gate: full val_bpb ≤ iter173_promoted_baseline AND K-gap ≤ iter173_K_gap. Branch outcomes: (a) BPB wins → deepest-K target = z_{257} is the model's strongest FP estimate; (b) BPB ties → close deeper-K direction; (c) BPB regresses → revert to iter173 alone. **SUBSUMES iter169** (the standalone K=256 entry, now closed). |
| Closed | iter169: extend K-jitter set with K=256 at half-freq | **SUBSUMED into iter174 2026-05-17 by post-iter172 principleness audit**: standalone K=256 was MARGINAL per the original principled audit (K=128 already passes rho_F gate at iter172; K=256 fires too rarely). Folded into iter174 where it becomes a meaningful share of late-training samples. Same launch, two changes targeting the same axis — one-variable discipline preserved by the conditional gating (iter174 runs only if iter173 promotes). |
| 6 | iter164 anytime-K routing | **CLOSED 2026-05-15 (principle conflict)**: directly contradicts iter163's promoted principle. iter163 forces all-K outputs to converge to the same FP (rho(J_F)<1 via consistency loss); anytime-K explicitly says K=16 should give DIFFERENT outputs than K=64 and uses both. Cannot coexist as an active candidate without invalidating the current champion. **Re-open requires a new design doc explaining why anytime supersedes consistency (e.g., per-token K selection with confidence estimation, hierarchical refinement) AND a refutation argument against iter163's per-step FP-condition framework.** Three-test failure: FAILED Principled (conflicts with promoted principle); FAILED Simplest (3-5× model size); FAILED General (architecture-specific to K-axis models). |
| Closed | iter156 hard contraction governor | **Closed 2026-05-13 (Tier 1+2 redesign)**: the entire iter156 design was based on the false premise that $\sigma_{\max}(J_F)<1$ is required for FP convergence. The principled framework (Hartman--Grobman) is $\rho(J_F)<1$ (spectral radius, necessary AND sufficient); $\sigma_{\max}<1$ is sufficient but over-restrictive. iter152 baseline empirically converges fine with $\sigma_{\max}\approx 17$ because $\rho \ll \sigma_{\max}$ for non-symmetric $J_F$. The "contraction debt" the iter156 was meant to address was a measurement artifact (operator-norm gate was mis-calibrated). Re-open ONLY if iter158 K-sweep shows `rho_F >= 1` AND `iter_conv_rel >= 0.05` at K=128 (i.e., asymptotic convergence is genuinely failing) — currently no such evidence. If re-opened, the principled mechanism is spectral normalization on $T_\theta$ component layers (bounds $\sigma_{\max}(J_T)$ per-layer, gives certified $\sigma_{\max}(J_F)$ via composition), NOT soft penalty (refuted) and NOT Ā governor (cross-block coupling makes it non-monotone). |
| Superseded | H94 GPTQ + LQER int4-rank4 | **SUPERSEDED 2026-05-15 by iter161-QAT-late.** PTQ (GPTQ alone) and PTQ+LQER (low-rank SVD error compensation) are both symptom-patching: they fix-at-the-end with frozen weights. iter161-QAT-late instead trains the model to BE quant-robust (root-cause attack), with a single mechanism (deterministic STE) instead of the GPTQ+LQER bundle. RevDEQ-safe because the STE forward is deterministic (unlike iter20 random-noise injection, refuted 2026-04). May return as iter161b fallback if QAT-late regresses BPB. | — | — | — |
| Closed | iter170: spectral normalization on T_θ component linears | **CLOSED 2026-05-15 (FAILED General test)**: spectral normalization on T_θ component linears is architecture-specific to weight-parameterized transition maps. Does not generalize to alternative DEQ solvers, refinement loops, recurrent layers, or other iteration mechanisms — violates CLAUDE.md most-principled-simplest-general directive AND the existing standing directive "Spectral normalization on T_θ component layers is rejected as a principled FP-convergence mechanism because it's specific to weight-parameterized transition maps (violates arch-agnostic)." If soft-tier consistency (iter163c/166/167) saturates with rho_F stuck > 0.9, the principled response is to accept the asymptotic limitation rather than escalate to a non-agnostic mechanism. |
| Closed | iter171: H96 per-group artifact compression | **CLOSED 2026-05-15 (no triggering condition reachable)**: champion iter163 artifact = 7.48 MB / 16 MB cap = 47 % utilization. The active queue (iter163c, iter168, iter161-QAT-late, iter166, iter167) only DECREASES artifact size: iter163c removes code; iter168 is probe-only; iter161-QAT-late reduces weights from bf16→int4; iter166/167 are scheduling tweaks. The only Held items that could push artifact > 13 MB are H95 SP8192+CaseOps (tokenizer expansion). Re-queue iter171 only if H95 unblocks. |
| Held | H91 phased test-time training | Per-document adaptation. | Yes: principled (Sun et al. TTT). | Multi-day eval-loop rework. | Defer behind quantization (TTT adds eval cost). Re-evaluate after iter161-QAT-late closes. |
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

## Legacy Run Macros (Superseded)

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
  --deq-k-jitter-set=16,24,32,64,96,128 \
  --deq-k-jitter-weights=0.50,0.40,0.07,0.03,0.015,0.0075 \
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
