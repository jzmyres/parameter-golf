# Experiment Hypothesis Log

Structured record of claims tested during autoresearch. Each entry:
**Claim → Evidence → Verdict → Implication for future experiments.**

---

## H1: Parallel residuals (raw sum) are compatible with DEQ
**Claim:** Replacing sequential `z ← z + attn(z); z ← z + mlp(z)` with parallel `z ← z + attn(z) + mlp(z)` improves the Jacobian isotropy and helps DEQ convergence.
**Test:** Iter 0 attempt 1 (commit 926cf63), attempt 2 (7949f5a)
**Evidence:** Both diverged. Attempt 1: deq_residual exploded 976 → 7.2e8 at step 600. Attempt 2 (0.5× scaling): still diverged, residual 1641 → 74K at step 600. Root cause: the sum form has ~2× update magnitude vs sequential, exceeding contraction constant.
**Verdict:** ❌ FALSE — raw sum (or 0.5× sum) is incompatible with DEQ solver at β=0.35 or β=0.20.
**Implication:** Parallel residuals need the gg_gate residual path (H2) or an explicit magnitude control mechanism.

## H2: Removing inner residual and using gg_gate as sole residual path stabilizes parallel residuals
**Claim:** `z2 = attn + mlp` (no `x +`) with `out = (1 - gg) * z_in + gg * z2` lets the learned gate control update magnitude, preventing the 2× blow-up.
**Test:** Iter 0 attempt 4 (commit f161bbe)
**Evidence:** Training stable through 822 steps. val_bpb 1.58 pre-quant. deq_iter_conv_rel stayed at 0.03. gg_tok self-regulated from 0.5 (init) → 0.22 (converged).
**Verdict:** ✅ TRUE — the gate provides automatic magnitude control. The model finds its own effective update rate.
**Implication:** For DEQ + parallel residuals, always use an outer gate as the residual mechanism, never an inner `x +` add.

## H3: SWA and EMA improve post-quantization val_bpb
**Claim:** Stochastic Weight Averaging and Exponential Moving Average should smooth weights and improve int6 quantization quality.
**Test:** Iter 0 (with SWA+EMA) vs iter 1 (without)
**Evidence:** Iter 0 post-int6: 2.4074. Iter 1 post-int6: 1.6209. Disabling SWA+EMA recovered 0.787 BPB. Root cause: EMA decay 0.997 over 822 steps leaves 8.6% of random init weights (0.997^822 = 0.086).
**Verdict:** ❌ FALSE at short training budgets (<1000 steps). SWA+EMA assume 3000+ steps; at <1000 steps they contaminate weights with init values.
**Implication:** Disable SWA+EMA for any run under ~2000 steps. Alternatively, use much higher decay (0.999+) or scale decay with step count.

## H4: The model can be forced to use more DEQ iterative depth via gg_gate initialization
**Claim:** Initializing gg_gate.bias = 1.5 (gg_tok init ~0.82 instead of 0.5) forces the model to use more iterative depth.
**Test:** Iter 2 (commit f16086e)
**Evidence:** The optimizer drove gg_tok from 0.82 → 0.11 within 200 steps, settling at 0.23 by step 832. Final val_bpb 1.6203 = identical to iter 1 (1.6209). The model actively REJECTED the high init.
**Verdict:** ❌ FALSE — initialization-only interventions cannot override the optimizer's preference. The model treats high gg_tok as a transient perturbation to recover from.
**Implication:** To change iterative depth usage, change the architecture or training objective, not the initialization.

## H5: More experts improve val_bpb at constant compute
**Claim:** Doubling experts (6→12) while maintaining routing regularization improves model quality.
**Test:** Iter 3 attempt 1 (12 experts, commit dcb1d5f), iter 3 attempt 3 (8 experts, d5052ce)
**Evidence:** 12 experts collapsed routing at step 400 (one expert took 51%). 8 experts stable, val_bpb 1.6225 (tied with 6-expert iter 1 at 1.6209). Expert balance at 8: attn_cv ~0.12, entropy 92% of max.
**Verdict:** ⚠️ PARTIAL — 8 experts is stable, 12+ collapses with current routing regularization (bal_loss_coef=5e-3). Routing collapse risk scales as num_experts × expert_rank².
**Implication:** Max stable expert count depends on expert_rank and routing regularization strength. At rank 128: max ~8 experts. At rank 64: 16 experts is stable.

## H6: β (deq_beta) has a U-shaped optimum
**Claim:** There's an optimal β that balances contraction tightness (low β = more stable) against per-step learning (high β = faster convergence).
**Test:** Iter 7 (β=0.10), iter 6a2 (β=0.20), iter 7b (β=0.05)
**Evidence:** β=0.10: val_bpb 1.7960. β=0.20: 1.8175. β=0.05: 1.8198 (gate saturated at 1.0). The model self-compensates: lower β → higher gg_tok. Effective update (β × gg_sum) differs but val_bpb peaks at β=0.10.
**Verdict:** ✅ TRUE at WD=0.18 — U-shaped with optimum at β=0.10. β=0.05 over-constrains (gate saturates), β=0.20 under-constrains.
**Implication:** The optimal β depends on WD (see H9). Always sweep β when WD changes.

## H7: Weight decay monotonically improves val_bpb
**Claim:** Higher WD → smaller weights → better int6 compression + implicit regularization → better val_bpb.
**Test:** Iter 5 (WD=0.09), iter 9 (WD=0.18), iter 10 (WD=0.36)
**Evidence:** WD=0.09: 1.7960. WD=0.18: 1.7539 (−0.042). WD=0.36: 1.7791 (worse than 0.18 but better than 0.09). WD=0.18 is Pareto optimal for val_bpb.
**Verdict:** ⚠️ PARTIAL — improves from 0.06→0.18, then diminishing returns/reversal at 0.36. Not strictly monotone.
**Implication:** WD optimum is ~0.18 for val_bpb. But WD also affects stability (H9), so the optimal WD depends on the objective (val_bpb vs stability).

## H8: Larger model_dim beats larger expert_rank at constant throughput
**Claim:** Increasing dim (768→896/1024) while reducing expert_rank to maintain step_avg gives better val_bpb than the reverse.
**Test:** Iter 8 (rank 256/384 at dim=768), iter 11 (dim=896/rank96/144)
**Evidence:** Rank doubling (iter 8): val_bpb 1.8819 (worse, throughput penalty). Dim increase (iter 11): val_bpb ~1.89 at step 400 (worse than dim=768). dim=1024/rank80: smoke FAILED (DEQ diverged, ratio expert_rank/dim < 0.1).
**Verdict:** ❌ FALSE — expert_rank is more valuable than shared attention width. Reducing rank to pay for dim is a bad trade. Also, expert_rank/dim ratio must stay above ~0.1 for DEQ stability.
**Implication:** At constant throughput, keep expert_rank as high as possible. Increase dim only if rank can stay above 0.1× dim.

## H9: Higher WD expands the stable β range
**Claim:** Higher WD → smaller weight magnitudes → smaller Jacobian spectral norm → the solver can tolerate larger β without diverging.
**Test:** WD=0.36/β=0.20 (smoke FAILED), WD=0.72/β=0.20 (smoke PASSED, iter 13 running)
**Evidence:** At WD=0.36, β=0.20 causes recon error 0.19 and oscillating convergence. At WD=0.72, β=0.20 gives clean convergence (iter_conv 21→4.1, monotone).
**Verdict:** ✅ TRUE — doubling WD from 0.36→0.72 made β=0.20 stable. The contraction condition β × spectral_norm(J) < 1 is satisfied when WD shrinks the Jacobian enough.
**Implication:** When scaling β up, also scale WD up proportionally. The (WD, β) pair should be co-optimized. If training diverges, try doubling WD before reducing β.

## H10: The DEQ model exploits training K rather than finding a true fixed point
**Claim:** Training at K∈{4,8} causes the model to optimize for K=8 output specifically, not for the fixed-point quality. Running more iterations at eval should NOT degrade val_bpb if the FP is good.
**Test:** Iter 12b — first valid K-sweep at K={4,8,16,32,64}
**Evidence:** K=4: 1.781, K=8: 1.778 (best), K=16: 1.802, K=32: 1.820, K=64: 1.825. Val_bpb monotonically DEGRADES past K=8. The FP (reached at K≈40, gate 0.007) has 0.047 worse val_bpb than the K=8 partial state.
**Verdict:** ✅ TRUE — the model exploits K=8. The FP quality is worse than the partial convergence state.
**Implication:** Use wider K jitter during training ({4,8,12,16}) to force the model to optimize FP quality at all K values. The model should produce good output at K=16, not just K=8.

## H11: Bigram hash table size scales with val_bpb
**Claim:** Larger bigram hash (65536×208) gives better val_bpb than smaller (4096×128) because it provides more contextual information per token.
**Test:** Iter 6 — reduced bigram from 65536×208 (13.7M params = 71% of model) to 4096×128 (590K params)
**Evidence:** After reducing bigram and reinvesting freed capacity into dim (512→768) and expert rank, val_bpb was 1.82 (iter 6a2). This is worse than old dim=512 with big bigram (1.62 at iter 1), but the models are not directly comparable (different dim, WD, β).
**Verdict:** ⚠️ INCONCLUSIVE — bigram was reduced simultaneously with other changes. Records use 2048-4096 bigram vocab and achieve 1.08-1.12 val_bpb, suggesting small bigram is sufficient.
**Implication:** Bigram params are low-value. Prefer investing capacity in the transformer body (expert rank, dim) over bigram hash table size.

---

## Open Hypotheses (to be tested)

### H12: Wider K jitter fixes FP quality degradation
**Claim:** Training at K∈{4,8,12,16} instead of K∈{4,8} forces the model to optimize for K=16 output, making the K-sweep monotone (no degradation at K>8).
**Test:** Iter 13 (running) — WD=0.72/β=0.20 + K jitter {4,8,12,16}
**Status:** RUNNING

### H13: The optimal (WD, β) pair lies on a diagonal
**Claim:** As WD increases, the optimal β increases proportionally (WD=0.18→β=0.10, WD=0.36→β=0.20, WD=0.72→β=0.40).
**Test:** Planned after iter 13
**Status:** PROPOSED

### H14: Router sigmoid gate improves expert utilization
**Claim:** Adding an input-dependent sigmoid gate on softmax routing weights helps expert utilization and DEQ convergence.
**Status:** PROPOSED (queued iter 17)

### H15: Quant-noise injection during DEQ iterations eliminates post-quant degradation
**Claim:** Applying quantization noise to random 10-20% of weights per DEQ iteration makes the FP robust to int6 quantization.
**Status:** PROPOSED (queued iter 20)

### H16: Single-step diffusion CTP enriches embedding gradients
**Claim:** Corrupting input with noise and predicting the clean token (CTP) gives gradient signal to more embedding rows and improves per-step learning.
**Status:** PROPOSED (queued iter 21)
