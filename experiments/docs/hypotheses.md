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
| Main active question | Test whether direct root-cause pressure closes the promoted tech debt: alive hinge for min-share, worst-pair diversity for post-int orthogonality, deterministic weighted K bags for finite-depth coverage, and default-off finite-expansion Lyapunov control for `T_theta`. |
| Near queue | Current code is the next smoke candidate. `lyapunov_coef` remains default-off and should be enabled only for the contraction ablation if `lip_ub` stays high. `iter144` remains the next nominal gradient-quality item but should not be launched automatically. `iter143` remains deferred. Chained routing rerun/fix path is deferred; when resumed, include 4-stage alternating typed chains `attn -> mlp -> attn -> mlp` and `mlp -> attn -> mlp -> attn`. |
| Heavy queue | H91 TTT (scaffold archived: `experiments/components/archive/phased_ttt.py`), H94 GPTQ+LQER (`archive/gptq_lqer.py`), H96 compression, H95 tokenizer/CaseOps, iter 118b smooth sparsity (`archive/sparse_attn_head_gate.py`, `archive/sparse_attention_dispatch.py`), iter 120 RRAttention (`archive/rr_attention.py`). Restore archived scaffolds into `experiments/components/` per `experiments/components/README.md` before reactivating. |

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
| Done | iter 145 | Evidential Dirichlet-UCB router + EMA-anchored GJSD-style routing objective: `router_scoring=dirichlet_ucb`, `router_dirichlet_ucb_beta=0.5`, `router_load_cv_coef=0`, `router_ema_balance_coef=0.1`, `router_ema_specialization_coef=0.1`, other active reg targets `0.1`, `regularizer_warmup_frac=0.07`, and `use_router_sigmoid_gate=0`. | Yes with caveat: routing remains token-local; persistent EMA is detached, so the balance loss uses a straight-through EMA anchor to give current routing a gradient while tracking historical usage. Normalized expert-output Gram/cosine remains the principled orthogonality pressure. | 1000 steps, 6.50 h train, 23.40 s/step, peak VRAM 34.86 GB. | `VALIDATED_WITH_TECH_DEBT`, not promoted as-is. BPB improved over iter142b: fast `1.4642` vs `1.4715`, final full `1.4840` vs `1.5009`, K-sweep best `K=24 1.4918` vs `1.5063`. Router/MoS health improved versus 142b and no dead expert was observed during training, but post-int gate failed `attn_min_share=0.0319 < 0.0400`, `mlp_ortho=0.5312 > 0.5`, and `lip_ub=65.3672 >= 1`. Root cause: router/liveness fix works for BPB but does not control the RevDEQ transition Jacobian; rescue before `iter144`. |
| Done | iter 145r | Promoted Dirichlet-UCB + EMA-balance routing defaults: `router_scoring=dirichlet_ucb`, `router_dirichlet_ucb_beta=0.5`, `router_load_cv_coef=0`, `router_ema_balance_coef=0.15`, `router_ema_specialization_coef=0.1`, `router_pertoken_entropy_coef=0.1`, `mos_load_cv_coef=0.15`, `expert_output_diversity_coef=0.15`, `regularizer_warmup_frac=0.07`, `use_router_sigmoid_gate=0`, `weight_decay=0.015`. | Yes with documented tech debt: routing remains token-local; EMA balance directly improves historical usage and BPB, but KL-to-uniform alone does not guarantee a hard minimum share. Scalar `deq_beta` is not a Parcae contraction fix because `lip_ub` probes `T_theta`, not the solver blend. | 1000 steps, 6.50 h train, 23.41 s/step, peak VRAM 34.86 GB. | **PROMOTED_WITH_TECH_DEBT** by user directive 2026-05-08. BPB improved: final full `1.4818` vs iter145 `1.4840` and iter142b `1.5009`; fast step1000 `1.4567`; roundtrip/K16 `1.4909`; K-sweep best `K=24 1.4899`. Open issues: strict post-int `attn_min_share=0.0371 < 0.0400`; output collinearity `attn_ortho=0.5098`, `mlp_ortho=0.5703`; contraction failure `lip_ub=45.7943` at K128. Promotion reason: task BPB win and no true dead expert during training. Tech-debt fix is queued separately. |
| 1 | root-cause rescue | Direct fixes on top of promoted `iter145r`: deterministic weighted K bag over `{16,24,32,64}`, EMA alive hinge, worst-pair cosine expert-output diversity, full-final-validation metadata, and router-confidence diagnostics. | Yes: each change targets an observed failure mode without batch-coupled routing or hard sparse dispatch. | Implemented; requires smoke before full run. | Success criteria: preserve iter145r BPB within `+0.003`, raise post-int `attn_min_share >= 0.040`, keep `mlp_min_share` healthy, bring `attn_ortho/mlp_ortho <= 0.5`, and keep final metadata full-validation promotable. |
| 2 | contraction ablation | Enable the implemented low-cadence Lyapunov loss on measured RMS expansion of `T_theta`, `relu(rms(T(z*+eps v,x)-T(z*,x))/eps - gamma)^2`, with RMS-normalized `v`, `eps=1e-2`, and `gamma=0.97`. | Yes: Lyapunov stability is principled when it directly constrains the transition map whose `lip_ub` gate is failing. | Code path exists behind `lyapunov_coef`; requires controlled run. | Run only if the root-cause rescue leaves `lip_ub` too high. Success: materially reduce `lip_ub`; clean contraction requires `lip_ub < 1`; BPB regression no worse than `+0.003` unless followed by a rescue. |
| Deferred | iter 143 | `deq_bptt_k=4` counterfactual. Original queue target was iter142b uniform-1.0 defaults; if revived, explicitly choose whether to run on the promoted iter145r stack or the historical iter142b stack. | Yes: one extra Neumann/VJP term after K=3 promoted. | Full 1000 steps. | Deferred by user directive. |
| 3 | iter 144 | IFT adjoint gradient for the truncated input/embedding signal. Design: [`iter144_ift_adjoint_plan.md`](./iter144_ift_adjoint_plan.md). | Yes: standard DEQ implicit-gradient correction and complementary to routing changes. | No: code change and likely about 2x backward cost. | Implement/test after user chooses whether to prioritize contraction follow-up (`iter147`) or this gradient-quality item. Land default-off only and compare equal-step plus wallclock. |
| 4 | iter 135/136 | Entmax blend init/LR tweaks. | Conditional: useful only if sparsity remains the active bottleneck. | Yes: CLI-only. | Hold until 142b/142c show whether smooth diversity is insufficient. |
| 4 | iter 142 deep K | `deq_k_jitter_set=(32,48)`. | Conditional: deeper solve only pays off after routing sparsifies. | Yes: CLI-only, but costly. | Run only if effective experts are roughly `<= 7` and step cost has sparse-kernel headroom. |
| 5 | H99 SmearGate | Default-off position-mixing memory channel. | Plausible: BOS-masked local memory channel. | Medium: code exists; needs controlled run. | Keep default-off; run only after the current routing/TBPTT queue. |
| 6 | iter 118b | RevDEQ-safe smooth sparsity primitive. | Yes if smooth/differentiable; hard Top-K is disallowed. | No: 6-10h feature. | Defer until 142b/142c and cheap entmax knobs fail to produce useful sparsity. |
| 7 | iter 120 | RRAttention / dynamic block sparse attention. | Plausible but kernel-sensitive at T=2048. | No: flex/Triton rewrite. | Defer until attention scaling is the measured bottleneck. |
| 8 | H91 | Phased test-time training. | Yes: high likely BPB ROI from per-document adaptation. | No: new eval/training loop. | Heavy one-session feature after baseline stabilizes. |
| 8 | H94 | GPTQ + LQER int4-rank4. | Yes: attacks quantization tax directly. | No: calibration and quantizer rewrite. | Run after model-side baseline stabilizes. |
| 8 | H96 | Per-group compression stack. | Engineering-principled: frees artifact budget, not BPB by itself. | Medium-heavy. | Separate artifact-efficiency track; useful before tokenizer expansion. |
| 8 | H95 | SP8192 + CaseOps. | Plausible: closes vocab inefficiency. | No: tokenizer retrain and artifact budget pressure. | Depends on H96 or equivalent budget relief. |
| Closed | iter 138/140/141 legacy gram queue | Old routing-gram ablations. | Partly, but stale under current flat objective. | Would waste runs. | Mark superseded; do not run. |
| Closed | H97/H98/H105 | Attn-gate quantization / hard sparse head gate / weak stale proposals. | Weak or mismatched to active architecture. | Not worth current queue slots. | Close or keep archived only. |

## Fast Test Macros

These are experiment shapes, not mandatory commands. Keep all logs under
`experiments/training_logs/` and archive the result row back into this file.

```bash
# iter 142b: decomposed canonical stack
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=50 \
  --expert-diversity-kind=cosine --expert-output-diversity-coef=1.0 \
  --regularizer-warmup-frac=0 \
  --val-loss-every=1000000

# iter 142c: Frobenius-only counterfactual
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=50 \
  --expert-diversity-kind=frobenius --expert-output-diversity-coef=1.0 \
  --router-load-cv-coef=0 --router-entropy-coef=0 --mos-load-cv-coef=0 \
  --regularizer-warmup-frac=0 \
  --val-loss-every=1000000

# iter 103a: chained mixed split
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
  --chained-stages-preset=split_2stage
# iter 103a health probe: keep balance alive, reduce entropy sparsity pressure
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
  --chained-stages-preset=split_2stage \
  --router-pertoken-entropy-coef=0.5

# iter 103b/103c: typed-chain order counterfactuals
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
  --chained-stages-preset=attn_first_2stage \
  --router-pertoken-entropy-coef=0.5
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
  --chained-stages-preset=mlp_first_2stage \
  --router-pertoken-entropy-coef=0.5

# iter 145: evidential router + EMA-GJSD CV replacement
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
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

# iter 146: coefficient-first rescue + weighted K jitter + confidence diagnostics
# These flags are now the Hyperparameters defaults; keep explicit flags only
# for reproduction or A/B runs against older checkpoints.
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
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
```

## Durable Rules

- Routing penalties operate on combined routed mass
  (`simplex(allocation_scores) * sigmoid(gate)`), not on renormalized slices.
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
  or contraction gates. Iter146 deliberately tests doubled ideal-target
  regularizers first; keep EMA alive hinges and Lyapunov/Jacobian contraction
  losses out of that run so attribution stays clean.
- Weighted K jitter over `{16,24,32,64}` is a finite-depth robustness add-on,
  not the contraction fix. Low-probability high-K sampling is represented by
  the explicit `deq_k_jitter_weights` field; the sampler normalizes weights,
  samples exact weighted bags, restores checkpoint state, and logs sampled-K
  counts for audit.
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
