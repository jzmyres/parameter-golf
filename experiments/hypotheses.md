# Experiment Hypothesis Log

Structured record of claims. Each entry:
**Claim → Evidence → Status → Implication.**

Status levels:
- **VERIFIED**: controlled experiment directly tested the claim with all else equal
- **OBSERVED**: evidence exists from experiments but confounded by other changes
- **PROPOSED**: untested, queued for future experiment
- **REFUTED**: controlled experiment directly disproved the claim

A hypothesis is only VERIFIED (or REFUTED) when a future iteration is explicitly
designed to test it with a single controlled variable change.

**Confound-aware logging:** When multiple variables change simultaneously:
- **CAN claim:** "the combination (A + B) produces outcome X"
- **CANNOT claim:** "A alone causes X" or "B alone causes X"
- **To isolate:** test A alone (hold B constant) and B alone (hold A constant)
- Example: if (12exp + WD=0.72) works but (12exp + WD=0.06) collapsed, we
  cannot say "12 experts is inherently unstable" — only that it collapsed
  UNDER the tested WD. The collapse might be addressable by increasing WD.

---

## VERIFIED

### H9: Higher WD expands the stable β range
**Claim:** Higher WD → smaller weight magnitudes → smaller Jacobian spectral norm → solver tolerates larger β.
**Test:** WD=0.36/β=0.20 (smoke FAILED: recon 0.19, oscillating) → WD=0.72/β=0.20 (smoke PASSED: clean convergence 21→4.1). Single variable change: WD doubled, everything else identical.
**Status:** ✅ VERIFIED
**Implication:** When DEQ training diverges at a given β, double WD before reducing β. The (WD, β) pair must be co-optimized.

### H12: Wider K jitter fixes FP quality degradation — VERIFIED
**Claim:** Training at K∈{4,8,12,16} forces the model to optimize FP quality at all K, making K-sweep monotone.
**Test:** Iter 13 (WD=0.72/β=0.20 + K jitter {4,8,12,16}) vs iter 12b (fixed K=8).
**Evidence:** K=8→K=16 Δ reduced from +0.025 (iter 12b) to +0.0002 (125× improvement). K=8→K=64 Δ reduced from +0.047 to +0.009 (5× improvement). Confirmed across 5+ subsequent iters — all show monotone or near-monotone K-sweep.
**Status:** ✅ VERIFIED
**Implication:** K jitter is a permanent training requirement. Small residual degradation at K>32 remains (+0.011 at K=64 in best config) but is 5× smaller than without jitter.

### H15: Quant-noise injection in DEQ iterations — REFUTED (fundamental incompatibility)
**Claim:** Quantization noise during DEQ iterations makes FP robust to int6.
**Test:** Iter 20 — quant-noise at rate=0.10 (recon 10.9, smoke FAILED) and rate=0.01 (recon 3.55, smoke FAILED).
**Root cause (fundamental, not implementation):** RevDEQ achieves O(1) memory backward by *reconstructing* forward states during backward, which requires `f(z, x0, W)` to be deterministic. Fresh-sampled per-call noise `W_noisy = W + ε_n` gives different outputs in forward vs. backward reconstruction because `ε_n^forward` is not saved. Noise magnitude tracks reconstruction error linearly (10% → recon 10.9; 1% → recon 3.55).
**Why fixes don't help:** Saving every ε_n defeats O(1) memory; deterministic noise from (iter_index, seed) removes the stochasticity benefit; a single ε across all K iterations reduces to "train a perturbed model" (not noise).
**Status:** ❌ REFUTED (incompatibility is fundamental to RevDEQ, not a bug)
**Implication:** Quant-noise is incompatible with RevDEQ's reversibility. The quant gap (0.002-0.008 BPB) is already tiny and not worth addressing this way.

### H18: β controls DEQ convergence speed — VERIFIED (PARTIAL)
**Claim:** Higher β → faster convergence → better FP quality.
**Test:** Iter 13 (β=0.20) vs iter 14a (β=0.30), both at WD=0.72. Single variable change.
**Evidence:** β=0.30 residuals 40-50% lower (converges ~2× faster per iteration) ✓. BUT K=8→K=64 Δ=+0.020 vs β=0.20's +0.009 (2× worse FP quality) ✗.
**Status:** ✅ VERIFIED (PARTIAL) — β controls convergence SPEED (confirmed) but NOT FP quality. Higher β converges faster TO A DIFFERENT (WORSE) fixed point.
**Implication:** Use the LOWEST β that converges within the K budget. β=0.20 at WD=0.72 is optimal.

---

## OBSERVED (need controlled verification)

### H1: Raw parallel residuals are incompatible with DEQ
**Observation:** `z2 = x + attn + mlp` diverged (iter 0a1, 0a2). `z2 = attn + mlp` with gg_gate stable (iter 0a4).
**Confounds:** β also changed between attempts. Need: test raw sum vs gg_gate at identical (WD, β, K).

### H2: gg_gate as sole residual path stabilizes parallel residuals
**Observation:** Removing inner residual (`z2 = attn + mlp`, gate handles residual) was stable for 822+ steps.
**Confounds:** Tested simultaneously with β=0.20→0.10 change. Need: test with/without inner residual at identical β.

### H3: SWA/EMA are net-negative at short training budgets
**Observation:** Disabling SWA+EMA recovered 0.787 BPB (iter 0 vs iter 1). EMA decay 0.997^822 = 8.6% init contamination.
**Confounds:** Could be specific to this model size/config. Need: test at longer budgets or different decay values.

### H4: Initialization cannot force iterative depth usage
**Observation:** gg_gate.bias=1.5 (init gg_tok=0.82) was driven back to 0.11 by step 200 (iter 2).
**Confounds:** Only tested one init value at one config. Need: test with constrained/frozen gate.

### H5: Routing collapse is caused by expert count alone — RESOLVED
**Original observation:**
- (12exp, rank128, WD=0.06, β=0.35) → collapsed at step 400
- (8exp, rank128, WD=0.06, β=0.35) → stable 743 steps
- (16exp, rank128, WD=0.18, β=0.10) → collapsed at step 400
**Iter 15 test:** (12exp, rank128, WD=0.72, β=0.20) → **STABLE 300 steps, attn_cv 0.017→0.225→0.182 (DECREASING), all 12 experts active.**
**Verdict:** ❌ REFUTED — routing collapse is NOT caused by expert count alone. It is caused by insufficient WD relative to expert count. WD=0.72 stabilizes 12 experts where WD=0.06 and WD=0.18 collapsed. Consistent with verified H9 (higher WD → more stability).
**CAN claim:** (12exp, rank128, WD=0.72) is stable.
**CANNOT claim:** 12exp is better than 8exp for val_bpb (12exp gets fewer steps from throughput penalty: 300 vs 356 steps at 1h budget).
**Implication:** When scaling expert count, increase WD proportionally. The collapse threshold is a (num_experts, WD) function, not a fixed expert limit.

### H6: β has a U-shaped optimum (at WD=0.18)
**Observation:** β=0.05 (1.820), β=0.10 (1.796), β=0.20 (1.818). Best at 0.10.
**Confounds:** Different β values may interact with K jitter range. Need: re-test at optimal WD with wider K jitter.

### H7: WD=0.18 is optimal for val_bpb
**Observation:** WD=0.09 (1.796), WD=0.18 (1.754), WD=0.36 (1.779). Best at 0.18.
**Confounds:** β was held at 0.10; optimal β may differ at each WD (per H9). Need: 2D sweep.

### H8: Expert rank is more valuable than model dim at constant throughput
**Observation:** dim=896/rank96 worse than dim=768/rank128. dim=1024/rank80 smoke FAILED.
**Confounds:** Only tested 2 dim values. Need: systematic grid at constant compute.

### H10: Model exploits training K rather than finding a good fixed point — RESOLVED by H12
**Observation:** First valid K-sweep (iter 12b) showed K=8 val_bpb 1.778 but K=64 val_bpb 1.825 (FP worse).
**Resolution:** K jitter (H12, now VERIFIED) reduced this gap from +0.047 to +0.009 (5×). Residual +0.011 at K=64 in best config suggests the model still slightly prefers its training K range, but the gap is small enough to be acceptable.

### H11: Bigram hash table size is low-value relative to transformer body capacity
**Observation:** Records use 2048-4096 bigram vocab vs our 65536. Reducing freed 71% of params.
**Confounds:** Reduced simultaneously with dim increase. Need: controlled test of bigram size at fixed everything else.

### H20: Post-norm unlocks flat gg regime (every DEQ iteration does real work)
**Observation:** Iter 19 — adding RMSNorm on Block output changed gg_iter from dome [1.0→0.08] to flat [1.0→0.67]. Biggest single arch improvement: -0.061 val_bpb. Without post-norm, model reduces gates to prevent magnitude blow-up through DEQ iterations. Post-norm bounds hidden state magnitude, letting gates stay high.
**Confounds:** Tested on top of additive injection (iter 18) and router sigmoid gate (iter 17). Need: test post-norm alone.
**Evidence strength:** STRONG — the gg regime change is dramatic and the val_bpb delta is the largest from any single arch change.
**Implication:** Post-norm is load-bearing for deep DEQ utilization. Do not remove.

### H20a: Per-component post-norm (attn + FFN separately) may improve on Block-level
**Claim:** The Block-level post-norm (`rms_norm(raw_out)` on `(1-gg)*z_in + gg*(attn+mlp)`) bounds the TOTAL magnitude but not the individual branch magnitudes. If attn and FFN operate at very different scales, one branch's magnitude dominates `attn_mix + mlp_mix` while the other is effectively suppressed. Per-component post-norm (`rms_norm(attn_mix) + rms_norm(mlp_mix)`) lets each branch find its own operating point.
**Mechanism:** Decouples magnitude control between attn and FFN; each branch can independently learn how much to contribute per iteration.
**Prediction:** -0.005 to -0.015 bpb if branches had imbalanced magnitudes at the Block-output-norm baseline.
**Risk:** Doubles the bounded magnitude in z2 (each branch now norm-1 instead of their sum being norm-1). May require compensating gg_gate adjustment.
**Test:** Phase 4.5 iter 22a.

### H20b: Per-expert post-norm may improve on per-component
**Claim:** Within a component (attn or FFN), experts with different magnitudes get weighted by the router. If expert i has magnitude 10× expert j, router weights effectively give expert i 10× more influence regardless of routing decision. Per-expert post-norm (before weight-mixing) removes this inter-expert magnitude conflict.
**Mechanism:** Each expert output is normalized to unit RMS before the softmax-routing weighted mix, so the router's weight fully determines expert contribution (not magnitude × weight).
**Prediction:** -0.005 to -0.015 bpb if expert magnitudes are imbalanced under current init/training.
**Risk:** Adds num_experts RMSNorms per component (16 per block at 8 experts × 2 components). Cost is small but non-zero. Also changes expert dynamics fundamentally — the routing+mixing regime is no longer just softmax-weighted sum of raw outputs, it's softmax-weighted sum of normalized outputs.
**Test:** Phase 4.5 iter 22b (only if 22a confirms per-component path is productive).

### H21: Correct per-expert Muon preconditioning changes optimization landscape
**Observation:** Batched NS fix (per-expert instead of flattened) + torch.compile → val_bpb 1.705 (from 1.897). The NS fix alone changed which optimization trajectory the model follows. Previously, all expert gradients were flattened into one (E*R, D) matrix for NS, corrupting per-expert preconditioning.
**Confounds:** torch.compile also changed (more steps due to throughput gain). But NS fix is the primary driver — it changes gradient direction, not just speed.
**Implication:** Muon NS must always operate at the correct tensor granularity. Any new parameter groups must verify NS input shape.

### H22: Router sigmoid gate improves FP quality (not val_bpb)
**Observation:** Iter 17 — router sigmoid gate gave -0.005 val_bpb but more importantly improved FP quality: K=8→K=64 Δ from +0.009 to +0.006 (33% better). Router gate learned INCREASING per-iter trajectory [0.30→0.76] — the only gate that opens over iterations.
**Confounds:** Combined with prior changes. The FP quality improvement is consistent but small.

---

## PROPOSED (untested)

### H17: WD controls training stability (prevents collapse) — PRIORITY
**Claim:** Higher WD → smaller weight magnitudes → smaller Jacobian → prevents training degeneracy (routing collapse, gradient explosion, loss divergence). WD is the lever for STABILITY, independent of convergence speed.
**Mechanism:** WD shrinks ||W|| → shrinks spectral_norm(∂f/∂z) → contraction condition β×||∂f/∂z|| < 1 is easier to satisfy → training stays in the stable basin.
**Prediction:** A config that collapses at WD=X should become stable at WD=2X without changing β.
**Partial evidence:** H9 verified (WD=0.36/β=0.20 FAILED → WD=0.72/β=0.20 PASSED). But H9 only tested one β value.
**To verify:** Test a config that collapsed (e.g., 12exp rank128) at progressively higher WD {0.36, 0.72, 1.44} with β held constant. If it stabilizes, WD→stability is confirmed.
**Isolation:** β must be held constant. If β also changes, the effect is confounded.

**Relationship between H17 and H18:**
- WD and β address DIFFERENT failure modes: WD→stability, β→convergence speed
- They are NOT fully orthogonal — see H19 (gate defense mechanism)
- Recipe: first set WD high enough for stability (H17), then tune β for convergence speed (H18)
- BUT: increasing β may require proportionally more WD (H19)

### H19: Higher β requires proportionally higher WD to maintain dome-shaped gate — PRIORITY
**Claim:** The model has a "defense mechanism": when β is too high relative to WD, the gate flattens to ~0.22 (shallow DEQ regime) to prevent instability. Only when WD is high enough does the model feel "safe" to open the gate in a dome-shaped pattern (genuine iterative depth usage).
**Mechanism:** At β×spectral_norm(J) ≈ 1 (near instability boundary), the gate flattens to reduce effective per-iter update. At β×spectral_norm(J) << 1 (well within stability), the gate opens to dome shape because larger updates are safe.
**Observed evidence:**
- β=0.20 at WD=0.36 → FAILED (unstable, smoke crashed)
- β=0.20 at WD=0.72 → dome gate [0.88→0.08] ✓ (safe enough to open gate)
- β=0.30 at WD=0.72 → **flat gate [0.30→0.19]** (defensive — WD=0.72 not enough for β=0.30)
**Prediction:** β=0.30 at WD=1.44 → dome gate (WD high enough for β=0.30 to feel safe)
**Emerging scaling law for min WD:**

| β | Min WD for dome-shaped gate | Evidence |
|---|---|---|
| 0.10 | ~0.18 | iter 7 |
| 0.20 | ~0.72 | iter 13 (dome), iter 12b WD=0.36 failed |
| 0.30 | ~1.44? | iter 14a flat gate at WD=0.72 → needs more WD |

Suggests WD_min ∝ β² (or some power law). Each β increment needs proportionally MORE WD.
**Update from iter 14a:** The flat gate at step 200 was TRANSIENT — by step 356, the dome shape formed even at WD=0.72/β=0.30. The "defense mechanism" is a training-phase phenomenon, not a permanent state. H19 may not need WD=1.44; the dome just forms slower at higher β.
**Status:** OBSERVED — initial "flat gate = defense" interpretation was premature (dome formed by step 356 at both WD levels).
**Iter 14b result:** WD=1.44/β=0.30 produced dome at step 356 (same timing as WD=0.72). But WD=1.44 had WORSE convergence (gg[15]=0.104 vs 0.067) and WORSE FP quality (K=8→K=64 Δ=+0.027 vs +0.020). Higher WD weakens per-iteration effectiveness without accelerating dome formation.
**Conclusion:** WD has a U-shaped optimum per β. WD=0.72 is near-optimal for β=0.20-0.30. WD=1.44 is too high.

### H13: The optimal (WD, β) pair lies on a diagonal
**Claim:** As WD increases, optimal β increases proportionally.
**Test:** Planned — need 2D sweep data
**Partial evidence:** H19 suggests WD_min ∝ β². The diagonal relationship exists for STABILITY (min WD per β), but may differ for PERFORMANCE (optimal WD per β for best val_bpb).

### H14: Router sigmoid gate improves expert utilization — OBSERVED (H22)
**Claim:** Input-dependent sigmoid gate on softmax routing weights helps DEQ convergence.
**Test:** Iter 17 — small val_bpb win (-0.005) + 33% FP quality improvement. See H22 in OBSERVED.
**Status:** Moved to OBSERVED as H22.

### H16: Single-step diffusion CTP enriches embedding gradients
**Claim:** Noisy soft-embed input + CTP denoising gives gradient to more embedding rows.
**Test:** Queued (advanced training block)

### H23: Vanishing injection violates DEQ input-dependence requirement — PRIORITY
**Claim:** The current injection gate decays to ~0.002 by DEQ iter 5, making `f(z, x0) ≈ f(z)` at later iterations. This means different inputs converge toward the SAME fixed point at high K, violating the DEQ requirement that `z* = f(z*, x0)` must depend on `x0`. This explains the K=64 val_bpb degradation (+0.011): the model loses input-specificity at deep K.
**Evidence:** inj_iter trajectory [0.42→0.10→0.02→0.005→0.002→...→0.000]. At K≥8, the solver is essentially input-blind. K-sweep shows val_bpb worsens at K>16 — consistent with input collapse at deep K.
**Nuance:** `z_in` at late iterations already encodes `x0` from early iterations (when injection was active). The concern is not that `x0` is completely lost, but that the solver cannot CORRECT based on `x0` at late iterations. Without ongoing `x0` signal, errors accumulate and the fixed point drifts.
**Implication:** Injection must remain non-trivial throughout ALL DEQ iterations. The gate should have a floor, or the mechanism should ensure input-dependence persists at any K.

### H24: Per-iteration injection schedule improves FP quality
**Claim:** Replace the single learned injection gate with K separate injection strengths (one per DEQ iteration). The current gate must serve two conflicting goals: high injection at iter 0 (bootstrap) and zero injection at iter 5+ (converge). An explicit schedule decouples these.
**Mechanism:** Use a small learnable vector of length K_max (e.g., 16) as sigmoid-gated injection strengths. At iter i, use `inj_strength[i]`. This lets the model learn "how much x0 to inject at each iteration" independently.
**Expected:** Better FP quality at high K (input-dependence maintained). The schedule should show a decaying but non-zero tail.
**Risk:** More parameters (K_max scalars). May need to freeze the schedule during early training to prevent collapse.

### H25: Residual injection (inject x0 - z error signal) improves convergence
**Claim:** Instead of injecting raw `x0`, inject the error signal `x0 - z_in`. This gives the solver corrective feedback proportional to deviation from input — it naturally decays as the solver converges (z→x0-like representation), so the gate doesn't need to learn decay.
**Mechanism:** `x = z_in + g_inj * (x0 - z_in)` — this is actually the ORIGINAL lerp form, but the key difference from iter 18's test is that now we understand the theoretical motivation: the error signal ensures input-dependence persists (z* = f(z*, x0) requires ongoing x0 contribution).
**Expected:** Better FP quality at high K. The gate should stay higher throughout because the signal is self-regulating.
**Note:** Iter 18 tested lerp→additive and found a wash (-0.003). But that test was at WD=0.72/β=0.20 + post-norm, which already had good FP quality. Re-testing with explicit focus on K=64/128 quality and the theoretical framing may yield different conclusions.

### H26: Multi-scale injection (different x0 projections per iteration)
**Claim:** Inject different aspects of x0 at different DEQ iterations. Early iterations get coarse (low-rank) x0 features for bootstrapping; late iterations get fine (full-rank) x0 features for input-specific correction.
**Mechanism:** Per-iteration projection matrices: `inj_i = W_i @ x0` where `W_i` varies in rank across iterations. Or simpler: use the same `x0` but with per-iteration learned scaling vectors (dim-wise gating).
**Expected:** Better FP quality + possibly better val_bpb from richer injection signal.
**Risk:** Parameter cost (K × projection matrices). May conflict with RevDEQ's weight-sharing requirement (same f across iterations). Could be implemented as a small number of "injection modes" (2-3) rather than K separate projections.

### H28: FSQ-based weight QAT closes the post-int6 quantization gap — RevDEQ-compatible
**Claim:** Replacing post-hoc int6 quantization with FSQ-based weight quantization during training (QAT via STE) closes the 0.002-0.008 BPB quant gap because the model learns representations robust to lattice-projection quantization noise.
**Key insight:** Unlike random per-call quant-noise (H15 REFUTED, breaks RevDEQ reversibility), FSQ is **deterministic** — same weights → same quantized output. RevDEQ backward reconstruction re-applies FSQ to the current weights and reproduces the forward values exactly. **No reversibility break.**
**Mechanism:** Apply FSQ-STE to each weight matrix during forward. Forward uses lattice-projected weights; backward passes the gradient through as identity (STE). The model's gradients now "see" the quantization, so it learns to compensate.
**Prediction:** Closes 0.002-0.008 BPB quant gap + possibly enables higher compression (FSQ index storage vs int6+zstd could save bytes).
**Risk:** Implementation cost. The current codebase has FSQ only for MoS intermediate projections, not weights. Extending to weights requires wrapping all matmul weights with `_fsq_ste(w, fsq_levels)`.
**Compatibility:** Fully compatible with RevDEQ (see "Key insight"). Also compatible with Muon optimizer (operates on gradients, unaffected by forward-time lattice projection).
**Test:** Queued for late-stage exploration (after Phase 5 injection work + Phase 4.5/4.6). Could potentially replace int6+zstd entirely.

### H27: Injection from refinement soft-embed during DEQ solve
**Claim:** Currently `x0_refined` (soft embedding from prior refinement step) only initializes `z0`. Injecting it during the DEQ solve (as a second input signal alongside raw `x0`) gives the solver access to denoised context throughout.
**Mechanism:** `x = z_in + g_inj * x0 + g_ref * x0_refined` with a separate gate for the refinement signal. At refinement step 0 (no prior prediction), `x0_refined = x0` so it reduces to current behavior.
**Expected:** Better refinement utilization. Currently the Diffusion-AR refinement only helps at z0 init; this makes it help throughout.
**Risk:** Refinement signal quality depends on prior-step prediction accuracy. If prediction is poor, injecting it throughout could hurt.

---

## Completed Iterations

| Iter | Config (changes from prev) | val_bpb | Status | Hypotheses tested |
|---|---|---|---|---|
| 0a1 | parallel resid raw sum, β=0.35 | — | crash (step 600) | H1 |
| 0a2 | parallel resid 0.5× | — | crash (step 600) | H1 |
| 0a4 | no inner resid + gg_gate + β=0.20 | 2.407 | discard (SWA/EMA killed) | H2, H3 |
| 1 | + SWA off + EMA off | 1.621 | discard | H3 |
| 2 | + gg_gate bias 1.5 | 1.620 | discard (no effect) | H4 |
| 3a1 | 12exp rank128 | — | crash (routing collapse) | H5 |
| 3a3 | 8exp rank128 | 1.623 | discard | H5 |
| 4 | 16exp rank64 (1 GPU) | 2.135 | discard (structural test) | H5 |
| 5 | WD 0.06→0.09 (1 GPU) | 1.884 | discard | H7 |
| 6a1 | dim768 + bigram↓ + 16exp rank128 | — | crash (routing collapse) | H5 |
| 6a2 | dim768 + bigram↓ + 8exp rank128 | 1.818 | discard | H8, H11 |
| 7 | β=0.10 | 1.796 | discard | H6 |
| 7b | β=0.05 | 1.820 | discard (gate saturated) | H6 |
| 8 | rank 256/384 | 1.882 | discard (throughput penalty) | H8 |
| **9** | **WD=0.18** | **1.754** | **KEEP (baseline)** | H7 |
| 10 | WD=0.36 | 1.779 | discard | H7 |
| 11 | dim896 rank96/144 | ~1.89 | killed (worse) | H8 |
| 12b | WD=0.36 (fixed K-sweep) | 1.778 | discard | H10 (first valid K-sweep) |
| 13 | WD=0.72, β=0.20, K jitter {4,8,12,16} | 1.976 | discard | H9, H12, H17 |
| 14a | β=0.30 at WD=0.72 | 2.046 | discard | H18 (speed yes, FP quality no) |
| 14b | WD=1.44, β=0.30 | 2.081 | discard | H19 (WD=1.44 too high) |
| 15 | 12exp rank128 at WD=0.72 | 2.033 | discard | **H5 RESOLVED** (collapse = WD-fixable) |
| 16 | Gate stats infra (gg/inj/attn_gate per-iter) | 1.966 | discard | Observability — 3-gate discovery |
| 17 | Router sigmoid gate (H14) | 1.961 | discard | H22 — FP quality +33%, val -0.005 |
| 18 | Additive injection (was lerp) | 1.958 | discard | Injection mechanism ≈ wash |
| **19** | **Post-norm (RMSNorm on Block output)** | **1.897** | **KEEP** | **H20 — biggest arch change, -0.061** |
| 20 | Quant-noise injection | — | crash | **H15 REFUTED** — incompatible with RevDEQ |
| **best** | **Muon batched NS + compile + all fixes** | **1.705** | **KEEP (baseline)** | **H21 — correct per-expert preconditioning** |

**Current best: val_bpb 1.705 post-int6, artifact 4.90 MB, K-sweep: k4=1.731 k8=1.711 k16=1.711 k32=1.715 k64=1.722**
**Record baseline: val_bpb 1.213 → gap = 0.49 BPB**

## Iteration Schedule

### Phases 1-3: COMPLETE ✅
- **Phase 1** (iters 13-14b): Verified H9, H12, H18. Locked (WD=0.72, β=0.20).
- **Phase 2** (iters 15-16): H5 RESOLVED (collapse = WD-fixable). Gate stats infra built.
- **Phase 3** (iters 17-20): Router sigmoid gate, additive injection, post-norm (KEEP), quant-noise (REFUTED).
- **Muon fix**: Batched NS + compile → NEW BEST 1.705.

### Phase 4: Throughput baseline + untied routers (RUNNING)

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **21** | Untied attn/mlp routers + per-component router gate tracking + all throughput fixes (memmap, FP32 eval, K=128) | Separate router learning + throughput baseline | — |

### Phase 4.5: Post-norm granularity (PRIORITY — front of queue)

Post-norm on Block output (iter 19) gave −0.061 bpb — biggest arch improvement.
Test whether more granular normalization helps: per-component (before gg gate
integration) or even per-expert (before expert-weight mixing).  Both iters run
independently (22b does not wait on 22a) — they test different hypotheses and
give complementary data even if one loses.

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **22a** | RMSNorm on attn output AND RMSNorm on FFN output separately (before z2 = attn_mix + mlp_mix) | H20a: per-component normalization may give independent control over each branch's magnitude, letting attn and FFN each find their own operating point | iter 21 |
| **22b** | RMSNorm on each expert output BEFORE weight-mixing (one norm per expert, not per-component) | H20b: if per-component helps, the finer granularity may help more — each expert can stabilize its magnitude independently, reducing inter-expert magnitude conflict | iter 21 (independent from 22a) |

**Outcomes are not exclusive:**
- Both win → keep the stricter one (22b), combined with Block-output norm
- 22a wins, 22b loses → per-component is the right granularity
- 22a loses, 22b wins → per-expert is the right granularity (more surprising)
- Both lose → Block-output norm granularity is optimal (negative-result signal useful for scaling law)

### Phase 4.6: Throughput — switch to unrolled backward

`deq_backward="revdeq"` does 3× forward FLOPs per backward (forward + reconstruction + gradient).  Unrolled does 2× (standard autograd).  ~33% faster per step at the cost of O(K) activation memory.  On 48GB L40S with K=16/dim=768/seq=2048: ~3 GB activations — fits comfortably.

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **22c** | `deq_backward="unroll"` (standard autograd through unrolled K DEQ iterations) | Eliminates reconstruction compute; should give ~1.3× training throughput with unchanged or better val_bpb (FP32 accumulators instead of FP64) | iter 21 |

### Phase 5: Injection mechanism rework (H23-H27) — PRIORITY

The injection gate decays to ~0.002 by iter 5, potentially violating the DEQ requirement that z* depends on x0. These experiments test whether improving injection fixes the K=64 degradation (+0.011).

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **23** | Per-iteration injection schedule (K learnable scalars) | H24 — decouple bootstrap vs late-iter injection | Phase 4.5 |
| **24** | Residual injection: inject (x0 - z_in) error signal | H25 — self-regulating input-dependence | Phase 4.5 |
| **25** | Injection floor: clamp inj_gate ≥ 0.05 | H23 — minimal fix to ensure input-dependence persists | Phase 4.5 |
| **26** | Multi-scale injection (per-iter dim-wise gating on x0) | H26 — coarse early, fine late | iter 23-25 best |
| **27** | Refinement soft-embed injection during DEQ solve | H27 — dual x0 + x0_refined injection | iter 23-25 best |

### Phase 6: Training objectives + activation

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **27** | Single-step diffusion CTP (noisy soft-embed + denoise) | H16 — enriches embedding gradients | Phase 5 best |
| **28** | LeakyReLU(0.5)² in MLP experts | Records evidence — consistent wins | Phase 5 best |

### Phase 7: Advanced techniques (if gap to record > 0.3 BPB)

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| 29 | OrthoInit on all large weight matrices | Records evidence — better gradient propagation through DEQ | Phase 6 |
| 30 | Skip gates between DEQ iterations (U-Net style) | Adapted from records 2026-04-09 | Phase 6 |
| 31 | Gated attention gate position (before SDPA vs after) | Records + paper arXiv 2505.06708 | Phase 6 |
| 32 | FSQ-STE weight QAT (replace post-hoc int6 with trained-in FSQ) | H28 — closes quant gap, RevDEQ-compatible (deterministic) | Phase 6 |

### Phase 8: Scaling law grid (FINAL — locked config)

Locked config: WD=0.72, β=0.20, K jitter {4,8,12,16}, 8exp, dim=768, post-norm, untied router sigmoid gates, best injection mechanism from Phase 5, batched Muon NS, compiled shared_block, best techniques from Phase 7.

**FSQ on MoS is already enabled by default (`fsq_levels=8`) via `_fsq_ste` in MoSHead forward.** The hyperparams `fsq_levels` and `mos_rank` are currently hard-coded in `GPT.__init__` (L1688) and should be plumbed through to `args` for the scaling law sweep.

| Iter | Config change | Variable | Depends on |
|---|---|---|---|
| 33 | dim=512, rank=128/192, 8exp | dim↓ (baseline comparison) | Phase 7 |
| 34 | dim=1024, rank=96/144, 8exp | dim↑ (test if post-norm enables larger dim) | Phase 7 |
| 35 | dim=768, rank=192/288, 4exp | fewer experts, higher rank | Phase 7 |
| 36 | dim=768, rank=96/144, 12exp | more experts, lower rank (H5 confirmed stable at WD=0.72) | Phase 7 |
| 37 | dim=768, rank=128/192, 8exp, mlp_mult=4 | wider MLP | Phase 7 |
| 38 | FSQ levels sweep: fsq_levels ∈ {4, 6, 8, 12, 16} | MoS output-head lattice granularity; higher = smoother logits but lossier quant | plumb `fsq_levels` as arg first |
| 39 | MoS rank sweep: mos_rank ∈ {128, 192, 256, 320} | Output head capacity vs artifact size | plumb `mos_rank` as arg first |
| 40 | Joint (fsq_levels, mos_rank) at best-dim from 33-34 | Combined output-head sweep | iter 38 + 39 |

### Troubleshooting table — hypothesis-verified fixes for assertion failures

When a post-int6 hard assertion fails, the training script writes `retry_hint.json`
with the prescribed fix from the table below.  The next iteration should apply
the fix (not discard the run) and re-test.  Each prescription traces to a
VERIFIED or RESOLVED hypothesis from this document.

| Failure category | Prescribed fix | Hypothesis |
|---|---|---|
| routing_imbalance (min_share, balance_cv) | `muon_weight_decay × 1.5` (cap 1.44) | H9 VERIFIED, H5 RESOLVED |
| expert_collapse (ortho) | `muon_weight_decay × 1.5`, else drop num_experts by 1 step | H5 RESOLVED |
| injection_collapse (inj_max, inj_mean) | Apply Phase 5 iter 24 (injection floor) or iter 22 (per-iter schedule) | H23 PROPOSED |
| gate_collapsed (gg_max<0.3) | Verify post_norm on; else `deq_beta - 0.05` | H20 VERIFIED |
| gate_saturated (gg_min>0.95) | `deq_beta + 0.05` (smaller per-iter update) | H18 VERIFIED |
| fp_quality_loss (K-sweep non-monotone) | Widen K jitter (`deq_k_max + 4`); fix injection first if also flagged | H12 VERIFIED, H23 PROPOSED |
| solver_divergence (iter_conv_rel>0.1) | `muon_weight_decay × 1.5` (H9) or `deq_beta - 0.05` (H18) | H9 + H18 |

**Retry protocol:**
1. If `run_valid=false` in `experiments/weights/current/meta.json`, the run is INVALID
2. Read `retry_hint.json` for the prescribed config change
3. Apply the change to `train_gpt.py` defaults (or pass as CLI override)
4. Re-run training from scratch (same iter number + "retry N" suffix in commit)
5. If the retry also fails with a different category, apply that fix next
6. Give up after 3 retries — the config may not be reachable from the current basin

### Permanent protocol for all iterations
- K jitter: {4, 8, 12, 16} (train at varying K to force good FP — H12 VERIFIED)
- K-sweep: {4, 8, 16, 32, 64, 128} with fast eval (256 seqs) + per-K diagnostics
- Pre-commit: /simplify → coderabbit → pr-review-toolkit → superpowers review
- Save full-precision weights (model_full.pt) before quantization
- **Update this hypothesis log BEFORE each iteration** (review relevant hypotheses, state what's being tested) **AND AFTER** (record results, update hypothesis statuses)
- Stability over task performance
- Locked (WD, β) = (0.72, 0.20) unless explicitly testing a WD/β hypothesis
- Post-norm on Block output is load-bearing — do not remove (H20)
- Muon NS must operate at correct tensor granularity — verify shape for any new param groups (H21)
- **Post-int6 HARD assertions** (run fails if any violated, DDP-global aggregation):
  - Per-component min_share ≥ 0.6/E (weakest expert ≥60% of fair share)
  - Per-component balance_cv ≤ 0.20 (routing balance)
  - Per-component ortho ≤ 0.20 (max-mean |cos|, not mean-|cos|)
  - gg_max ≥ 0.3 AND gg_min ≤ 0.95 (gate active, not collapsed/saturated)
  - inj_max ≥ 0.05 AND inj_mean ≥ 0.01 (x0 injection non-zero — H23 DEQ input-dependence)
  - K-sweep monotone from K=8 (Δ ≤ 0.005), worst K≥16 within 0.03 of best
  - iter_conv_rel ≤ 0.1 at highest K
- **Artifact budget HARD cap**: code + compressed model ≤ 16,000,000 bytes (fails early)
- Untied attn/mlp routers (separate sigmoid gates and routing weights per component)
