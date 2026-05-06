# Experiment Hypothesis Log

Concise working ledger for active research decisions. Detailed historical
entries and raw evidence are preserved in
[`experiments/hypotheses_archive.md`](./hypotheses_archive.md). Operational
protocol and failure-mode details live in [`EXPERIENCE.md`](../EXPERIENCE.md).

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
| Backbone | RevDEQ + Parcae, `use_parcae=True`, `deq_k_jitter_set=(16,24)`, `deq_bptt_k=3`. |
| Routing loss stack | Flat direct coefficients: router CV, router entropy, MoS CV, expert-output diversity. Legacy nested routing-gram stack is superseded. |
| Main active question | Resolved: cosine + decomposed stack (142b) PROMOTED as canonical loss design under uniform-1.0 reg coefs (user override 2026-05-06). Frobenius-only (142c) NOT-PROMOTED. iter 142d SUPERSEDED by the uniform-1.0 baseline; iter 143 (deq_bptt_k=4) currently runs on top of it. |
| Near queue | iter 143 (running, deq_bptt_k=4 on uniform-1.0 baseline); iter 144 only after iter 143 confirms TBPTT-depth headroom. Each iteration runs the full 1000-step budget per user directive 2026-05-06. |
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
| 1 | iter 143 | `deq_bptt_k=4` at iter-142b promoted defaults (uniform-1.0 reg stack). | Yes: one extra Neumann/VJP term after K=3 promoted. | Running at full 1000 steps. | Promote if recon remains near floor and val_bpb improves at acceptable cost. |
| 3 | iter 144 | IFT adjoint gradient for the truncated input/embedding signal. | Yes: standard DEQ implicit-gradient correction. | No: code change and likely about 2x backward cost. | Defer until iter 143 says the TBPTT-depth path still has headroom. Land default-off only. |
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

# iter 143: TBPTT depth bump
torchrun --standalone --nproc_per_node=gpu train_gpt.py \
  --iterations=100 \
  --deq-bptt-k=4
```

## Durable Rules

- Routing penalties operate on combined routed mass
  (`softmax_or_entmax(scores) * sigmoid(gate)`), not on renormalized slices.
- Hard Top-K / magnitude-skip dispatch is not RevDEQ-safe. Any sparsity path
  must be smooth, default-off, and exact/no-op at the strict-generalization
  point.
- Loss metrics and hard gates can differ when their roles differ: losses need
  dense gradients; gates need clean failure detection.
- All gates must be token-local and input-dependent; no batch/sequence
  reductions may influence a token's fixed-point map.
- Promote only with a controlled comparison against the active baseline plus
  post-int6 K-sweep and health diagnostics. For short probes, state the budget
  limit explicitly.
- Keep implementation details and long evidence in `EXPERIENCE.md` or the
  archive; keep this file as the decision ledger.
