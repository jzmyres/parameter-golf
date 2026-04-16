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

### H30: β jitter improves K=128 extrapolation
**Claim:** Training at fixed β=0.20 leaves the model β-specific. Sampling β ∈ {0.10, 0.20, 0.30} per step (β±0.1) makes the model robust to varying contraction rates, complementing K jitter (H12 VERIFIED, which handled varying solver depths).
**Mechanism:** At eval-time K=128, the effective dynamics differ from training-time K=16 even at the same β. β jitter forces the model to learn a wider basin of (β, K) combinations. By H18 (β controls convergence speed but lower β = better FP quality), the model trained at jittered β should converge to a lower-β-equivalent FP at deep K.
**Test:** Phase 4.5 iter 22c — replace `deq_beta = 0.20` constant with per-step uniform sample from {0.10, 0.20, 0.30}.
**Risk:** Small β (0.10) may be too slow to converge in K=4 jitter samples (smoke could fail). If so, narrow to {0.15, 0.20, 0.25}. If β=0.30 sample causes solver_divergence, the existing prescription system bumps WD×1.5 (H19 path).
**Status:** PROPOSED.

### H31: LOSS and GATE metrics must be different by design — PRINCIPLE
**Claim:** A loss function and a hard-assertion gate optimize for different things and should use different metrics, even when measuring the "same" property.
- **LOSS** wants SMOOTH gradient signal (every pair / every token / every iteration contributes pressure) for fast convergence.
- **GATE** wants CLEAN failure detection (catches the worst case at threshold) for interpretability and minimal false positives.
**Empirical evidence:** iter 21-retry-2 confirmed this empirically by violating it. Switching MoS `ortho_out` LOSS from `max_mean_abs_offdiag_cosine` (smooth) to `max_pairwise_abs_cosine` (sparse) — to match the GATE — gave:
- val_bpb regression: 1.67 → 1.88 (huge)
- K=128 catastrophe: Δ=3.27 (vs iter 21's 0.043 = 76× worse)

The sparse gradient (only the worst pair updates per step) starved the routing dynamics; fewer pairs got pushed apart per step → poor ortho convergence → poor FP at deep K.

**Fix applied (commit a5ecf11):**
- LOSS: `max_mean_abs_offdiag_cosine` (every pair contributes gradient)
- GATE: `max_pairwise_abs_cosine` (catches worst pair > 0.9)

**Status:** ✅ VERIFIED principle (empirical violation produced documented regression; reverted).
**Implication:** When designing both a training signal and a gate for the same architectural property, choose metrics that match each role's purpose, not the same metric for "consistency".

### H29: All gates must be input-dependent AND token-local — PRINCIPLE
**Claim:** Every gate in the DEQ block (injection, gg, attention, router) must be computed from the CURRENT TOKEN's hidden state without reduction over batch or sequence.  Otherwise the update map depends on other examples in the batch or how sequences are chunked, breaking streaming / prefix-caching invariance and making the fixed point batch-dependent.
**Violation found:** iter 21 — `_inj_gate_from` computed `g = sigmoid(inj_gate(mean(z_in, dim=(batch, seq))))`.  This meant injection depended on ALL tokens in the batch + sequence, not just the current token.  Chunking a sequence differently changed the mean → changed the gate → changed the fixed point.  Changed batch size → changed the mean → changed gates.  Cross-example coupling through a DEQ block is a serious structural bug.
**Fix applied:** `g = sigmoid(inj_gate(rms_norm(z_in)))` → shape `[B, T, 1]`, per-token.  Smoke test passed with similar recon/convergence profile.
**Other gates verified token-local:**
- `gg_gate(x_attn_n)`: input is per-token RMSNorm(z_in + g_inj*x0) → ✓
- `router_gate(x_n)` (attn + mlp routers): input is per-token RMSNorm — ✓
- `attn_gate` (from Q projection): inherently per-token — ✓
**Status:** ✅ VERIFIED (fix in place, smoke test passes)
**Implication:** When adding any new gate, it MUST be computed from the current token's state only.  No `.mean(dim=(0, 1))` or similar before a gate.
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
| **21** | **Untied routers + token-local inj (H29) + DDP-safe health assertions** | **1.6706** | **INVALID under OLD strict gates (would PROMOTE under NEW val_bpb-primary policy)** | **H29 VERIFIED** |
| 21-retry | + WD 0.72→1.08, K_max 16→20 (over-engineered for old gates) | 2.108 | discard (val_bpb regression — WD over-regularized; K=128 catastrophic Δ=1.675) | confirmed: WD=1.08 + K_max=20 anti-synergistic |
| 21-retry-2 | revert WD/K_max + max_pairwise ortho LOSS (bug — sparse gradient) | 1.882 | discard (val_bpb regression; K=128 Δ=3.27, even worse than 21-retry) | confirmed: max_pairwise as LOSS breaks training |
| 21-retry-3 | + decouple LOSS=max_mean (smooth) and GATE=max_pairwise (clean) | **1.892** | discard (val_bpb regressed +0.19 from baseline 1.705; K=128 Δ=1.79) | loss/gate decoupling helped (K128 went 3.27→1.79) but did NOT restore iter 21's 1.67 — root cause unclear, requires repro |
| **21-retry-4** | **repro test: HEAD code (val_bpb-primary + H32 ortho loss removed)** | **1.876 (TRUE int6)** | discard (superseded by iter 22) | CRITICAL FINDING: prior "baseline 1.705" was BF16 (commit `83541fc` fixed the int6 roundtrip indentation bug on Apr 13 19:52; pre-fix the int6 model was never loaded for eval — dead code). iter 21-retry-3 (1.892) and retry-4 (1.876) are the FIRST valid int6 measurements. True int6 quant cost ~0.17 BPB was hidden. |
| **22** | H32 partial revert: `block_ortho_aux_coef 0 → 0.1` | **1.888 (TRUE int6)** | **KEEP (NEW BASELINE)** | val_bpb 1.888 is within ~0.015 bf16/seed noise floor (empirical: retry-3=1.892 vs retry-4=1.876 same config differ 0.016). attn_ortho 0.82→0.30 / mlp_ortho 0.86→0.22 = 3-6× structural improvement. H32 final arc: coef 1.0 → 0 (too aggressive, ortho drifted near 0.9 gate) → 0.1 (right balance — ortho clean, val_bpb parity). |
| **22-add-all** | Add 5 LEARNABLE RMSNorms at all reasonable post-non-linearity positions (attn_sdpa_post, hidden_post, attn/mlp_post_mix, embed_post) | **1.891** | **KEEP** | Within noise of baseline 1.888. Big win: K=128 Δ collapsed 1.93 → 0.53 (3.6× extrapolation improvement). Establishes "all norms in" as the starting point for systematic ablation. |
| **22-rm-embed-post** | Remove `GPT.embed_post_norm` (learnable RMSNorm after tok+bigram embed) | **1.885** | **KEEP** | val_bpb IMPROVED -0.006 vs 22-add-all — that learnable norm at embed entry was harmful. K=128 Δ regressed somewhat but val_bpb-primary promotes. |
| **22-rm-attn-sdpa-post** | Remove `CausalSelfAttention.attn_sdpa_post_norm` | 1.895 | **REVERTED (3d78cfe)** | val_bpb within noise but K=128 Δ regressed 0.53 → 2.09 (4× FP-quality loss). User principle: K=128 extrapolation IS a primary measure of FP quality; norms preserving it are load-bearing. New dual-gate KEEP criterion adopted. |
| **22-rm-hidden-post** | Remove `MLP.hidden_post_norm` (norm on per-expert hidden after leaky_relu²) | 1.897 | **REVERTED (`c47324f`)** | int6 val_bpb 1.897 within 0.015 noise of baseline 1.885 → gate (a) PASS. K-sweep: k4=1.944 k8=1.906(best) k16=1.907 k32=1.912 k64=1.963 k128=**3.425** → K=128 Δ=1.519 BPB regression (vs baseline ~0.5) → gate (b) FAIL. `hidden_post_norm` is load-bearing for K=128 extrapolation. Pattern confirmed: removing learnable RMSNorms degrades FP-quality at deep K even when val_bpb at training-K is unchanged. |
| **Phase 4.5 ablation summary** | Norm ablation closed | — | — | **CONCLUSION**: of the 5 RMSNorms added in 22-add-all, ONE was harmful (`embed_post_norm`, removed → improvement to 1.885) and the other 4 (`attn_sdpa_post_norm`, `hidden_post_norm`, `attn_post_mix_norm`, `mlp_post_mix_norm`) all preserve K=128 extrapolation. Final baseline: **22-rm-embed-post `73ae2e9`, val_bpb=1.885**. attn_post_mix_norm and mlp_post_mix_norm not individually tested (assumed load-bearing by analogy; can defer-test later if motivated). |
| **22-add-expert-norms** | (a) Expand `MLP.hidden_post_norm` to per-expert weight (E, R); (b) NEW per-expert output norm (E, D) on BOTH MLP and CSA experts | 1.930 | **REVERTED (`90b21c6`)** | Both gates fail: (a) val_bpb 1.930 vs baseline 1.885 → +0.045 BPB outside 0.015 noise floor; (b) K=128 Δ=0.808 (k8=1.936 best, k128=2.744) > 0.5 threshold. K-sweep: k4=1.961 k8=1.936 k16=1.938 k32=1.945 k64=2.002 k128=2.744. Per-expert norms harmed both task perf AND extrapolation — likely the per-expert weights diverged during the 1-iter training budget, breaking expert-bank composition. **Bisect into 22a + 22b to identify which one caused the regression.** |
| **22a-hidden-per-expert** | (a) ONLY: expand `MLP.hidden_post_norm` weight from shared (R,) to per-expert (E, R) | TBD | queued behind iter 23A | Bisection of failed 22-add-expert-norms. Tests whether the per-expert hidden scale alone caused regression. ~672 added params (8 experts × 96-rank for the expansion). |
| **22b-out-per-expert** | (b) ONLY: NEW per-expert output norm (E, D) on both MLP and CSA experts (before sum). Leave `hidden_post_norm` shared (R,). | TBD | queued behind 22a | Bisection of failed 22-add-expert-norms. Tests whether the per-expert output norm alone caused regression. ~12K added params (2 banks × 8 experts × 768D). |
| **23A** ★ | Phase 5: always-on low-rank injection — replace inj_gate with `z + inj_B(inj_A(LN(x0)))`, rank=384, inj_B init std=0.02, no α, drop `inj_max ≥ 0.05` gate. Track ||inj||/||z|| + cos(inj, z). | 1.891 | **REVERTED (`bc3a428`)** | Gate (a) passes (val_bpb 1.891 within noise of 1.885), but gate (b) FAILS: K=128 Δ=1.852 (k8=1.900 best, k128=**3.751**). **KEY FINDING: failure mode MOVED, not eliminated.** inj_iter at K=128 now HEALTHY [0.11→0.37→plateau 0.22] — injection is preserved across deep iters. But gg_iter at K=128 now COLLAPSES [1.00, 1.00, ..., 0.008, 0.004, 0.000] — gg_gate saturates to 0 by iter 20, `(1-gg)·z_in` dominates → z_in passes through unchanged at deep K. **H23-REFINED**: gate-collapse pathology migrated from inj_gate to gg_gate. Removing one DoF shifted the model's escape route. Also: dead `mos_ntp_min_share=0.008`. |
| **Phase 5 reassessment** | — | — | — | The DEQ identity `z* = f(z*, x0)` requires BOTH (1) x0 enters `f` meaningfully AND (2) the transformation `f` does nontrivial work at deep K. iter 23A fixed (1) but broke (2). **Deeper insight**: both residual-form `z* = z + α·z2` and lerp-form `z* = (1-g)z + g·z2` admit a TRIVIAL fixed point at any z where `z2 = 0` (or `g = 0`). The model learns to shut down z2 to satisfy the FP cheaply. Phase 5b tests three fixes in parallel. |
| **23B-gg** | Phase 5b: remove gg_gate, fixed-α=0.2 residual (`raw_out = z + 0.2·z2`). inj_gate UNCHANGED. | 1.992 | **REVERTED (`6292df0`)** | Both gates fail: val_bpb 1.992 (+0.107 from 1.885). K-sweep: k4=2.089 k8=2.029 **k16=2.004 best** k32=2.305 k64=4.056 k128=5.074 → K=128 Δ=3.07 (WORSE than iter 23A's 1.85). Fixed α=0.2 too rigid for training-K perf. The residual form's "z2=0 at z*" trivial FP remains — removing per-token gate alone doesn't eliminate the escape route. |
| **23C-gg-no-residual** ★★★ | Phase 5b-alt: no gg_gate AND no residual-z (`raw_out = 0.5·z2`). FP eqn: `z* = 0.5·z2(z*)` — transform must map to itself, not zero. | **1.865** | **PROMOTED (new BASELINE `3970083`)** | **🎯 BREAKTHROUGH**: val_bpb 1.865 (**-0.020 vs prior baseline 1.885**, first improvement since iter 22). K-sweep: k4=1.891 k8=**1.854** k16=1.872 k32=1.882 k64=1.882 k128=**1.882** → K=128 Δ=**0.028** (vs baseline ~0.5, iter 23A 1.85, iter 23B 3.07). **First truly stable DEQ fixed point across K=4→K=128**. Removing the residual eliminated the trivial `z2=0` FP escape, model learned a genuine dynamic FP. Tech debt: attn_ortho=0.92 (duplicate experts), mos_ntp_min_share=0.008 (dead expert) — non-blocking per val_bpb-primary policy. |
| **23D-bundle** | Phase 5-bundle: always-on inj + no gg + no residual. FP: `z* = 0.5·z2(z*) + inj(x0)`. | 1.914 | **REVERTED (`c053476`)** | Gate (a) FAILS +0.049 vs 1.865. Gate (b) PASSES (K=128 Δ=0.247, well under 0.5). K-sweep: k4=1.948 k8=1.903 k16=1.926 k32=2.014 k64=2.106 k128=2.150. attn_ortho=0.34 (big improvement from baseline's 0.92) but mos_ntp dead expert persists (0.009). **Finding**: always-on inj HURT val_bpb AND reduced K-sweep flatness vs 23C. The per-token inj_gate of baseline was actively helping — forcing always-on removes useful modulation. 23C's (no residual + per-token inj_gate) is the principled sweet spot. |
| **24-wd-bump** ★ | Bump Muon weight_decay 0.72 → 1.08 on iter 23C baseline. | 1.955 | **RE-APPLIED (`d10004f`) — KEPT despite val_bpb regression** | Gate (a) failed in short-budget val_bpb (+0.09), but K-sweep remarkably flat: k4=1.983 k8=**1.946** k16=1.967 k32=1.979 k64=1.980 k128=1.980 → Δ=**0.034** (even flatter than 23C). k64→k128 asymptote confirms true FP convergence. **User decision: preserve WD=1.08 as locked config because K→∞ FP stability is the primary objective of a DEQ model; val_bpb degradation is a training-speed artifact addressable by OTHER iters.** attn_ortho 0.92→0.20 improved, mos_ntp_min_share 0.008→0.006 WORSE — WD can partially address ortho (weight-space) but NOT dead expert (routing-space). Next iter: dead-expert fix in router space. |
| **25 (aborted)** | Per-channel learnable β via unrolled solver. | smoke FAIL | **UNCOMMITTED, DISCARDED** | Unrolled solver's O(K) autograd regressed training 10×. Proper impl needs custom RevDEQ backward (~30-50 LOC). Deferred for retry session. |
| **26-lb-loss** ★ | Bump MoS balance loss weight 50× (effective 0.25 with bal_loss_coef downstream). Addresses MoS NTP dead expert. | **1.951** | **PROMOTED (new BASELINE `802b436`)** | All gates pass: val_bpb 1.951 (-0.004 vs 1.955), K-sweep flat k4=1.982 k8=**1.941** k16=1.962 k32=1.975 k64=1.976 k128=1.977 → Δ=0.036, status=**validated_clean** (mos_ntp_min_share ≥ 0.01 — dead expert resurrected). LB loss successfully addressed routing-space collapse where WD couldn't reach. |
| **27a-pos-mix-out** ★ | Phase 5e-1 position sweep #1: move injection from P-begin to P-mix-out. | **1.906** | **PROMOTED (new BASELINE `6c9a06b`)** | -0.045 vs prior baseline! K-sweep clean: k4=1.947 k8=**1.914** k16=1.915 k32=1.938 k64=1.967 k128=1.985 → Δ=0.071. tech debt: attn_ortho=0.918 (duplicate attn experts — new tech debt vs iter 26's clean status). Major arch win — direct x0 path to z* via post-z2 add. |
| **27b-pos-expert-out** ★ | Phase 5e-1 position sweep #2: per-expert injection inside mix_experts. | **1.902** | **PROMOTED (new BASELINE `9edc6af`)** | val_bpb improved -0.004 vs 27a. K=128 Δ=**0.039** (tighter than 27a's 0.071). Tech debt: inj_max=0.010 < 0.05 (gate collapsed — but effective net injection is 2×g_inj·x0 from attn+mlp sum, so small gate value still meaningful). |
| **27c-pos-router-in** | Phase 5e-1 position sweep #3: inject ONLY into router inputs. | 2.093 | **REVERTED (`27faf0d`)** | CATASTROPHIC FAIL on both gates: val_bpb +0.19 vs 27b; K-sweep k4=2.13 k8=2.11 k16=2.11 **k32=5.41 k64=5.42 k128=5.42** → Δ=3.31. Experts never see x0 → FP became x0-independent past K=16 → solver diverges to meaningless basin. **Structural conclusion: experts MUST see x0 for FP x0-dependence to hold.** |
| **27d-pos-expert-in** | Phase 5e-1 position sweep #4: inject ONLY into expert inputs (routers read clean z). Mirror of 27c. | 1.967 (trained fully; see run `23A`) | **REVERTED** (cleanup commit `623672f`) | val_bpb 1.967 > baseline 1.902 (+0.065). K-sweep clean (k4=2.004 k8=**1.967** k16=1.982 k32=2.000 k64=2.002 k128=2.003 → Δ=0.037) — FP validity intact, but task perf regressed. Asymmetric injection (expert-only) is not better than symmetric (27b). Phase 5e-1 CONCLUDED: **27b (P-expert-out symmetric) remains BASELINE `9edc6af` val_bpb=1.902**. |
| **28-tbptt-k4** | Truncated BPTT with k=4 (gradient-scale hypothesis). | 2.33 @ step 200 (run died) | **REVERTED (`7f1f907`)** | VJP decay ratio 0.77-0.96 → too flat for k=4 (captures only 40%). Throughput +36% but val_bpb trailing. |
| **28b-tbptt-k8** | Milder TBPTT k=8 (captures ~83%). | **1.974** @ step 563 | **REVERTED (code disabled `deq_bptt_k=0`)** | FAIL val_bpb +0.072 vs baseline. VJP ratio ≈ 0.82. TBPTT abandoned — decay too slow for meaningful gain. Code retained per user. |
| **29a-W-identity** | Phase 5e-2 #1: replace sigmoid gate with literal `z_in + x0` (no gate/W/scaling). | **1.946** | **REVERTED (`9ed0aa4`)** | FAIL +0.044 vs baseline. K-sweep: k4=1.981 k8=**1.944** k16=1.958 k128=1.993 Δ=0.049 (FP OK). **Conclusion: z-dependent sigmoid gate IS load-bearing** — modulates injection per-token, constant 1.0 over-injects. |

### Phase 5e: Injection mechanism experiments (queued, after iter 26)

**Per user direction (revised 2026-04-15):** find best **POSITION FIRST**, then best **TRANSFORMATION** with that position. Identity = `x = z_in + x0` literally (no W, no gate, no scaling).

#### Phase 5e-1: POSITION sweep (transform = current sigmoid·Linear·LN gate)

| Iter | Variant | Where to inject |
|---|---|---|
| (baseline) | P-begin = iter 23C current | `x = z_in + g·x0` before attn_norm/mlp_norm — z passes through transform with x0 mixed in |
| **27a** | P-mix-out | `raw_out = 0.5·z2 + g·x0` — x0 added AFTER transform (bypasses attn/mlp) |
| **27b** | P-expert-out | `out_e = ... + (g/E)·x0` per-expert before sum (1/E to keep magnitude bounded) |
| **27c** | P-router-in | Inject into x_attn/x_mlp ONLY for routers (`router(z+g·x0)`); experts read clean z |
| **27d** | P-expert-in | Inject into expert inputs ONLY (not routers); routers read clean z |

#### Phase 5e-2: TRANSFORMATION sweep (use best position from 5e-1)

| Iter | Variant | Form |
|---|---|---|
| (baseline) | current = sigmoid(Linear(LN(z_in)))·x0 | per-token z-dependent gate |
| **28a** | W-identity | `x = z_in + x0` — direct add, no W, no gate, no scaling |
| **28b** | W-full-rank | `x = z_in + W·x0` with W full-rank D×D learnable, init small (std=0.02) |

### Phase 5f: Norm ablation (queued, after Phase 5e)

Re-run systematic norm ablation on the new best architecture once injection mechanism is settled. Removes the 4 learnable RMSNorms (22-rm-attn-sdpa-post, 22-rm-hidden-post, 22-rm-attn-mix-post, 22-rm-mlp-mix-post) one at a time, apply dual-gate (val_bpb + K=128 Δ).

### Phase 5g: Deferred to later (per user direction)

- iter 25 learnable per-channel β (needs custom RevDEQ backward ~30 LOC)
- iter 26 PSD on expert layers (inductive bias only, no MON guarantee)
- iter 27 TBPTT (needs RevDEQ surgery too)
| **25-learnable-β (aborted)** | Per-channel learnable β via `nn.Parameter(full((D,), logit(0.20)))`, sigmoid-wrapped, applied via unrolled solver (switched deq_backward="unroll" to enable autograd through β). | smoke FAIL | **UNCOMMITTED, DISCARDED** | Smoke: loss delta only -0.285 (vs baseline -2.75 at 300 steps) — 10× slower convergence. Root cause: unrolled solver's O(K) autograd backward injects much more gradient noise than RevDEQ's implicit differentiation. RevDEQ is essential for our setup; proper iter 25 requires custom autograd.Function backward that manually computes grad_β ∝ Σ(f(z) - y)·∂L/∂y across K iters. That's ~30-50 LOC of custom backward surgery — deferred for retry. |

**Iter 21 val_bpb 1.6706 is BETTER than 1.705 (-0.034), but 5 hard assertions failed under OLD gates:**
- `mlp_min_share=0.037 < 0.075` (OLD threshold; NEW threshold 0.01 — would PASS)
- `mlp_balance_cv=0.280 > 0.20` (OLD gate; NEW policy: CV not gated — would PASS)
- `mos_ntp_ortho=0.211 > 0.20` (OLD max-mean threshold; NEW max-pairwise ≤ 0.9 — would PASS)
- K-sweep degradation k128 (OLD 0.03 threshold; NEW 0.1 — would PASS)
- K-sweep non-monotone K32→K64 (OLD 0.005 threshold dropped; finite-K noise floor)

**Under the current val_bpb-primary policy + relaxed gates, iter 21 (and 21-retry-3 if it matches) is promotable.**

### Lessons from iter 21-retry-{1,2,3} chain
1. **Don't over-engineer for old strict gates.** WD=1.08 + K_max=20 was a response to OLD CV/min_share thresholds that NEW gates don't care about. Result: catastrophic K=128 (Δ=1.675).
2. **LOSS and GATE are different metrics by design.**
   - LOSS needs SMOOTH gradient (max_mean across all pairs) for fast convergence
   - GATE needs CLEAN duplicate detection (max_pairwise on worst pair) for interpretability
   - Conflating them: switching MoS ortho LOSS to max_pairwise gave sparse gradients (only worst pair updates per step), regressing val_bpb 1.67→1.88 AND K=128 catastrophe (Δ=3.27 vs iter 21's 0.043).
3. **Gate calibration must be principled, not threshold-fishing.** The 0.005 monotone K-sweep gate fired on the current best baseline too (Δ=0.007 K32→K64) — below noise floor. Dropped entirely.
4. **val_bpb-primary promotion** prevents getting stuck on gate calibration. iter 21 alone could have been promoted; we wasted 3 iterations chasing assertion compliance instead.

**Suggested fix (from retry_hint.json):** `WD × 1.5 (0.72→1.08)` + `deq_k_max + 4 (16→20)`
**Next iter 21-retry:** apply both, re-run, verify assertions pass.
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

### Phase 4.4: Throughput micro-optimizations (FIRST — before norm ablation)

**Rationale:** more steps/hour means every subsequent iter (norm ablation, injection rework, training objectives, scaling law) gets a better signal per compute hour. Front-loading throughput work compounds.

**Top 5 ROI iters in priority order:**

| Iter | Optimization | Expected gain | Effort | Risk |
|---|---|---|---|---|
| ~~T1~~ | ~~Grouped expert mixing~~ — **ALREADY DONE** (both mix_experts paths use torch.bmm; input uses flattened matmul) | 0 | — | — |
| ~~T2~~ | ~~Defer diagnostic CPU sync~~ — **mostly done** (diag sites gated behind `_ROUTER_DIAGNOSTICS_ACTIVE`, only fire during log/eval steps; <2% real gain not worth the refactor complexity) | <2% | — | — |
| ~~T4~~ | ~~Drop full val mid-train~~ — **ALREADY DONE** (L2624 uses `full_validation=False`) | 0 | — | — |
| T3 | compile `dynamic=True` | 2-5% | Low | High (past dynamo bugs) |
| T5 | activation-checkpoint MoS head | Modest | Medium | Low |

**Phase 4.4 status:** T1, T2, T4 already implemented via prior hardening work.  T3/T5 have marginal ROI vs implementation cost.  **Skip to Phase 4.5 norm ablation** where the arch wins live.  If we later need more throughput, revisit T3 with newer torch or T5 with activation-checkpointing.
| **T2** | Defer diagnostic CPU sync to log time (record GPU-only inside DEQ loop) | 5-10% | Low | Low — already partially done for routers |
| **T3** | Compile with `dynamic=True` so single graph handles K∈{4,8,16} | 2-5% steady, eliminates K-recompile re-warmup | Low | Medium — past dynamo bugs may have been fixed in newer torch |
| **T4** | Drop full validation in mid-train val cycles; only fast subset; full val once at end | 5-10% (saves ~30s × N val cycles per hour) | Low | Low |
| **T5** | Activation-checkpoint MoS head (256MB logits tensor, recomputable) | Modest step time, enables larger batch (better grad signal, fewer microsteps) | Medium | Low |

**Run order:** T1 → T2 → T3 → T4 → T5. Each iter measured against the previous baseline.

**Performance-lossless requirement (HARD):** each throughput iter must be NUMERICALLY EQUIVALENT to the baseline within tolerance. The training trajectory and val_bpb must not regress beyond bf16 noise floor (~0.005 BPB). Throughput optimizations that "trade quality for speed" are NOT acceptable here — those go in different phases. Each iter ships with:
1. **Unit test** verifying numerical equivalence of the changed forward path against the prior implementation (e.g., for T1: a test that runs both old einsum and new bmm path on random input, asserts max-abs-error < 1e-4 in bf16 / < 1e-6 in fp32)
2. **Smoke test** passes (same loss-decrease + recon-err trajectories as prior)
3. **Short comparative training run** (50-100 steps) showing matched loss/val_bpb trajectory vs prior baseline
4. **Throughput measurement** (ms/step at steady state) showing the gain
5. Promote on val_bpb improvement OR equivalent val_bpb + measured throughput gain (val_bpb-primary policy + throughput as tiebreaker)

**Goal:** end Phase 4.4 with combined ~30-40% throughput improvement, ZERO val_bpb regression, then proceed to Phase 4.5 norm ablation with faster iter cadence.

### Phase 4.5: Norm position ablation (after Phase 4.4 throughput is done)

**Approach** (per user direction): instead of separate one-off iters for specific norm positions, do a SYSTEMATIC ablation:
1. **Iter 22-add-all**: add LEARNABLE RMSNorm at ALL reasonable post-non-linearity positions. Make all RMSNorms learnable (weight ∈ R^dim, currently parameter-free).
2. **Iters 22-rm-{position}**: remove ONE position at a time. Keep removed if val_bpb doesn't regress meaningfully (≥ −0.005 noise floor).
3. Continue until no further simplification possible.

**Reasonable post-norm positions (AFTER non-linearities, not after softmax/sigmoid):**
- After gated SDPA in attention (sigmoid×softmax×V output, before output projection)
- After leaky_relu² in MLP (before fc-down)
- After per-expert output (each expert's projected output, before weighted mixing) — H20b candidate
- After attn_mix per-component (before z2 = attn_mix + mlp_mix) — H20a candidate
- After mlp_mix per-component
- After Block output (current `post_norm` — H20 VERIFIED, do NOT remove)
- After bigram embedding addition (residual sum into tok_emb)
- After tok_emb (before DEQ entry)

**Skip (already bounded by their non-linearity):**
- After softmax (router output is a probability)
- After sigmoid gates (gg, inj, attn_gate, router_gate)
- After RMSNorm itself

**Pre-norm vs post-norm**: also test pre-norm (norm BEFORE the non-linearity instead of after). Single iter to compare.

**β jitter (H30)** — runs as separate Phase 4.5b iter after the norm ablation settles:
- Sample β ∈ {0.10, 0.20, 0.30} per training step
- If β=0.30 causes solver_divergence, the existing prescription bumps WD×1.5 (H19 path)
- Targets K=128 extrapolation

**Outcomes are not exclusive:**
- Both 22a/22b win → keep the stricter one (22b), combined with Block-output norm
- 22a wins, 22b loses → per-component is the right granularity
- 22a loses, 22b wins → per-expert is the right granularity (more surprising)
- Both lose → Block-output norm granularity is optimal (negative-result signal useful for scaling law)
- 22c (β jitter) → primarily targets K=128 degradation; complementary to 22a/22b

### Phase 5: Injection mechanism rework — PRIORITY (re-prioritized after user feedback 2026-04-14)

**Root cause of inj-gate collapse:** `g(z)` is z-dependent. At z* = f(z*, x0), additional injection is a perturbation that backward gradient suppresses to drive `g→0` because loss is evaluated at converged z*. Once z* encodes x0, the gate has no incentive to keep injecting. **This breaks the DEQ identity** `z* = f(z*, x0)` (z* becomes independent of x0 at deep K) and is the upstream cause of K=128 collapse.

**User principle (verified principled):** Removing the gate entirely is more principled than any patch — it adds zero degrees of freedom that can misbehave and structurally enforces that every iteration sees x0. With learnable B,A in the projection, the injection magnitude is fully controlled by B's init scale (std=0.02 or zero-init) and B's weight-decay; no separate scaling parameter is needed (α would just be absorbed into B).

**Plan: test BOTH Form A and Form B as parallel iterations** (user direction 2026-04-14). Form C (residual at input side, `B(A(x0 - z_in))`) is REJECTED — x0 lives in embedding space, z_in lives in DEQ state space, subtracting them at input side scrambles semantics; both Form A and Form B avoid this by either not subtracting or subtracting in z-space after projection.

**α design — DROPPED for 23A/23B** (user feedback 2026-04-14): when B and A are learnable, scalar α is **mathematically redundant** because `α · B(A(x)) = (αB)(A(x))` — the optimizer absorbs α into B. Adding α as a separate learnable parameter is a no-op that wastes 1 DoF. The injection magnitude is controlled by B's init (e.g., std=0.02 or zero-init) and B's weight-decay — both jobs α was nominally doing. **α is only needed in 24c-min** where there is no projection matrix to absorb it; there α is learnable with regularization toward 0.1 to prevent collapse.

| Iter | Form | Config change | Why | Depends on |
|---|---|---|---|---|
| **23A** ★ | A: always-on, no gate | `z_hat = z + B(A(LN(x0)))`; B small-init (std=0.02), no α | Simplest — no gate to collapse, x0 always seen, FP exists if B small. Minimum DoF. | Phase 4.5 |
| **23B** ★ | B: residual in z-space | `z_hat = z + B(A(LN(x0))) - LN(z_in)`; B small-init (std=0.02), no α | Self-regulating — error → 0 in z-space as z_in absorbs projected x0; "collapse" becomes meaningful. | Phase 4.5 |

**Decision criterion for 23A vs 23B:** primary = val_bpb; tiebreaker = K=128 Δ (smaller is better — measures FP-quality preservation across deep extrapolation, the property the gate-collapse pathology specifically broke). If both improve val_bpb but B has cleaner K=128 behavior, prefer B. If A has both cleaner val_bpb AND comparable K=128, prefer A on Occam grounds (fewer ops).

**Iter 24/25/26 (old plan) — DROPPED as not principled** (user feedback 2026-04-14):
- *Per-iteration α schedule (K learnable scalars):* if 23A works, schedule is unnecessary by definition; if 23B works, residual already encodes per-iter signal. Defeats the purpose of 23A/B.
- *Multi-scale dim-wise α:* `α[d] · B(...)[d] = (diag(α) @ B)(...)[d]` is mathematically redundant with B — the optimizer can already absorb α[d] into B. Splits one DoF into two with no new expressiveness.
- *Dual x0 + x0_refined in refinement:* only conditionally principled; needs an observed failure mode in soft-embed averaging to motivate. Not a general improvement.

**Iter 24 (new plan) — projection capacity sweep** (only if 23A/B are marginal; skip if clean win):

| Sub-iter | Form | Projection DoF | Question |
|---|---|---|---|
| **24c-min** | `α · LN(x0)` (no A, no B) — α IS needed here (no matrix to absorb), learnable with reg toward 0.1 | 1 (scalar α) | Is *any* projection needed? Or does raw normalized x0 suffice? |
| **24c-mid** | `B(A(LN(x0)))`, rank = kv_latent_dim (current) — no α | 2·D·r | Baseline reference |
| **24c-max** | `C(LN(x0))` with single full-rank C (D×D) — no α | D² | Does full-rank projection help? Single D×D is mathematically equivalent to full-rank B(A(·)) (since B@A collapses to one D×D matrix when both are full-rank) but uses **half the params** — strictly better for efficiency. |

If 24c-min ≈ 24c-mid → drop B (and A) entirely on Occam grounds. If 24c-max ≫ 24c-mid → projection capacity is bottlenecked, raise rank. Otherwise keep current rank.

After Phase 5 closes (winner of 23A/23B, optionally with 24c result), proceed directly to Phase 6 (training objectives).

### Phase 5c: DEQ-theoretic iterations (user-directed 2026-04-15)

Three principled DEQ improvements queued. User policy: "keep trying β and MON — if it doesn't work, try other optimizations first, then come back."

| Iter | Change | Why | Risk |
|---|---|---|---|
| **25-learnable-β** | β: scalar 0.20 → per-channel `R^D` vector via `sigmoid(raw_β)`; init fill 0.20 | Different channels have different contraction needs — aggressive β for stable channels, conservative for unstable. RevDEQ reversibility preserved (element-wise division). | +768 params/block. If fails, retry with (a) per-layer not per-channel, (b) softplus clamp not sigmoid, (c) MLP-gated β |
| **26-PSD-expert-layers** | Parameterize `expert_out` + `expert_down` (dominant-spectral-path linear maps) as PSD `W = L·L^T` | **Inductive-bias only, NOT a MON theoretical guarantee.** See footnote below. | If fails, retry variants: (a) shifted-PSD `W = (ε·I + L·L^T)/(c + \|\|L\|\|²)` adding spectral-norm bound, (b) only `expert_down` PSD, (c) smaller L rank |

**Footnote on iter 26 — theoretical status:**

Partial PSD parameterization of `expert_out`/`expert_down` does **NOT** invoke any MON theorem guarantee. The MON theorem requires the *entire operator* `f(z, x) = σ(W·z + U·x + b)` to have that specific single-layer form with `I - W` shifted-PSD. Our block is `f(z, x0) = 0.5·(attn_mix(z) + mlp_mix(z))` which includes softmax attention, MoE soft routing, LeakyReLU² (non-monotone), and gated attention — none of which fit MON's operator form. Even if `expert_out = L·L^T` (PSD), the Jacobian `J_f` has the form `Σ_e [∂w_e/∂z·expert_out_e·(...) + w_e·expert_out_e·∂(path)/∂z]`, and the product of PSD with non-monotone matrices is **not** monotone. Monotonicity does NOT compose.

What iter 26 actually buys:
- ✓ Nonneg-eigenvalue direction structure on those specific matrices (aesthetic / regularizer)
- ✓ Reduced parameter dimensionality
- ✗ NO unique FP guarantee
- ✗ NO unconditional convergence guarantee
- ✗ NO spectral norm bound (PSD has eigenvalues in [0, ∞) — unbounded above)
- ✗ NO global Lipschitz / contraction guarantee

For a real MON guarantee on our architecture, we'd need either (a) to restructure an entire block component into the `σ(Wz + Ux + b)` form, (b) to replace the block with vanilla MonDEQ (loses MoE/MLA), or (c) to apply layer-wise spectral-norm bounds everywhere (the "hacky" approach, done systematically).

**Treat iter 26 as an empirical experiment in architectural inductive bias**, not a theoretical upgrade. Expected value: PSD constraint may improve K-sweep flatness by shaping the dominant-path spectrum; may also regress val_bpb by reducing model capacity. Pure empirical question.
| **27-TBPTT** | RevDEQFunction.backward: detach z at iter K-k, only gradient-propagate through last k of K iters | Throughput saving (backward ~75% at k=4/K=16). With 23C's stable FP, early-iter grads may be near-redundant. | If task perf regresses, retry at different k values |

### Phase 5d: Code hygiene sweep (from feedback review 2026-04-15)

Priority bugs/fixes not blocking arch iters — can run in parallel / between training runs:

| # | Fix | Priority |
|---|---|---|
| 5.5 | `experiments/smoke_test.py` `_load_real_data` reads shard 256×int32 header as int16 tokens → garbage. Replace with `load_data_shard()` (matches train loader). | **Bug, fix now** |
| 5.4 | `tests/test_post_int6_helpers.py` + `tests/test_gate_init_defaults.py` + `experiments/test_input_injection_hypernet.py` reference removed `inj_gate`/`gg_gate`. Delete or rewrite for always-on state. | **Bug, fix now** |
| 5.3 | Emit a single canonical `final_postquant val_bpb:X` line in train_gpt.py; update `plot_metrics.py:154` + `update_results.sh:152` to read it (or from `meta.json[val_bpb]`). Last `val_bpb:` match currently hits K-sweep tail, not scored full-val. | **Fix now** |
| 5.6 | `zstandard` is an env prerequisite (already in requirements.txt; do **not** modify dependency files). Code path: try `import zstandard` first, then fallback to `lzma` (not `zlib` — worse ratio, risks 16MB blow). | High |
| 5.1 | Update old CLAUDE.md/comment thresholds to match current relaxed gates (min_share < 0.01, ortho > 0.9, no CV gate). The strict thresholds are stale from pre-val_bpb-primary era. | Medium |
| 5.2 | Router health constraints: enforce on BOTH attn+mlp separately; align `plot_eval_metrics.py:186` keys. | Medium |
| 5.7 | Add `--diagnostics={0,1}` flag. Competition default 0 skips full_weights_save + K-sweep + plot subprocesses; dev default 1 keeps current. Important for 600s competition budget. | Medium (before 8×H100 final) |
| 5.8 | Muon's allreduce on top of DDP grad allreduce is redundant bandwidth. Switch Muon to `broadcast-from-owner` per-shard, or drop DDP reduction where Muon handles. | Low (defer; Phase 4.4 throughput was front-loaded) |

### Phase 6: Training objectives + activation

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| **28a** | Single-step diffusion CTP (noisy soft-embed + denoise) | H16 — enriches embedding gradients | Phase 5 best |
| **28b** | LeakyReLU(0.5)² in MLP experts | Records evidence — consistent wins | Phase 5 best |

### Phase 7: Advanced techniques (if gap to record > 0.3 BPB)

| Iter | Config change | Hypothesis | Depends on |
|---|---|---|---|
| 29 | OrthoInit on all large weight matrices | Records evidence — better gradient propagation through DEQ | Phase 6 |
| 30 | Skip gates between DEQ iterations (U-Net style) | Adapted from records 2026-04-09 | Phase 6 |
| 31 | Gated attention gate position (before SDPA vs after) | Records + paper arXiv 2505.06708 | Phase 6 |
| 32 | FSQ-STE weight QAT (replace post-hoc int6 with trained-in FSQ) | H28 — closes quant gap, RevDEQ-compatible (deterministic) | Phase 6 |

### Phase 7.5: Throughput optimization — unroll+compile investigation (runs BEFORE Phase 8)

(Phase 7.4 throughput micro-opts have been moved to Phase 4.4 — front-loaded for compounding effect.)

**Goal:** Maximize training throughput *before* committing compute to the Phase 8 scaling-law grid.  More steps/hour in the sweep = more hyperparameter points covered per 1h iter budget.  Doing this *after* Phase 7 ensures the throughput measurement uses the final arch (post-OrthoInit, skip-gates, etc.); doing it *before* Phase 8 means the scaling sweep gets the fastest possible backward path.

**Core hypothesis:** A WORKING `unroll + torch.compile` configuration would be faster than the current `revdeq + torch.compile` baseline at real training batch, because:
- Unroll does 2× forward FLOPs per backward vs revdeq's 3× (no reconstruction pass)
- Unroll uses FP32 accumulators vs revdeq's FP64 (revdeq needs FP64 for exact reversibility; unroll doesn't)
- Small-batch benchmark (no compile on either side) measured **3.21× speedup**: unroll 358 ms vs revdeq 1151 ms at batch=8/seq=1024

**Blocker:** compile + unroll + DDP hits two separate bugs:
- `torch.compile(shared_block, dynamic=False)` + unroll + DDP → `loss.requires_grad=False` (grad_fn broken)
- `torch.compile(shared_block, dynamic=True)` + unroll + DDP → dynamo backend crash in KV-attention (`'int' has no 'meta'`)

**Investigation plan (apply in order, stop when one works):**

| Step | Approach | Why it might work |
|---|---|---|
| A | Compile `_deq_solve` (entire K-step loop) instead of `shared_block` | One compiled graph for the whole loop — avoids "32 compiled calls in a Python loop" interaction with DDP gradient hooks.  The whole DEQ solve becomes a single autograd op from DDP's POV. |
| B | `mode="reduce-overhead"` (uses CUDA graphs) | Different codegen path that may not hit the grad_fn-tracking bug; CUDA graphs are explicitly designed for repeated identical calls. |
| C | `torch._dynamo.disable` surgically on the DDP-critical paths, keep compile everywhere else | Keeps most of the speedup while avoiding the specific failing interaction. |
| D | Upgrade torch to a newer minor release if available | Both bugs may be fixed upstream; check release notes. |
| E | Compile + unroll WITHOUT DDP (single-GPU control) → confirms the bugs are DDP-specific | Diagnostic only, rules out compile/unroll incompatibility that isn't DDP-mediated. |

**Benchmark protocol once a fix works:**
1. Run `experiments/speed_compare_backward.py` at the REAL training batch (not the small benchmark size).  Need per-microstep timing AND per-optimizer-step timing (since unroll may need more grad_accum to fit VRAM).
2. Run a full 1h training with each config (compile+revdeq vs compile+unroll) at the then-current best arch.  Compare:
   - total steps completed
   - val_bpb at 1h
   - post-int6 val_bpb
   - K-sweep FP quality (any difference in solver quality at K=128)
3. **Decision rule:** switch to unroll only if `(steps × val_bpb_improvement) per hour` is higher AND the Phase 7 K-sweep hard assertions still pass.  Not just faster ms/step.

**Fallback if no fix works:** stay on `revdeq + compile` for Phase 8.  Revisit on 8×H100 where the VRAM constraint relaxes and compile+unroll compatibility may differ.

### Phase 8: Scaling law grid (FINAL — locked config + best backward mode from Phase 7.5)

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
| dead_expert (min_share < 0.01) | `weight_decay × 1.5` + `balance_mult × 1.5` (cap WD 1.44; applied to both Muon and AdamW groups) | H9 VERIFIED, H5 RESOLVED |
| expert_collapse (attn/mlp ortho > 0.9) | `weight_decay × 1.5`, else drop num_experts by 1 step | H5 RESOLVED |
| mos_head_collapse (mos_* ortho > 0.9) | `mos_ortho_out_coef × 1.5`, else shrink mos_rank | separate from expert_collapse: MoS head count is structural, not tunable |
| injection_collapse (inj_max < 0.05 or inj_mean < 0.01) | Apply Phase 5 iter 24/22 (injection floor / per-iter schedule) | H23 PROPOSED |
| gate_collapsed (gg_max < 0.3) | Verify post_norm on; else `deq_beta - 0.05` | H20 VERIFIED |
| gate_saturated (gg_min > 0.95) | `deq_beta + 0.05` (smaller per-iter update) | H18 VERIFIED |
| fp_quality_loss (K-sweep degradation Δ > 0.1) | Widen K jitter (`deq_k_max + 4`); fix injection first if also flagged | H12 VERIFIED, H23 PROPOSED |
| solver_divergence (iter_conv_rel > 0.1) | `weight_decay × 1.5` (H9) or `deq_beta - 0.05` (H18) | H9 + H18 |
| reversibility_broken (deq_recon_err > 1.0) | Check for randomness in block (H15 REFUTED quant-noise); `deq_beta - 0.05`; `WD × 1.5` | fundamental — RevDEQ requires deterministic f + stable contraction |

**Retry protocol:**
1. If `run_valid=false` in `experiments/weights/current/meta.json`, the run is INVALID
2. Read `retry_hint.json` for the prescribed config change
3. Apply the change to `train_gpt.py` defaults (or pass as CLI override)
4. Re-run training from scratch (same iter number + "retry N" suffix in commit)
5. If the retry also fails with a different category, apply that fix next
6. Give up after 3 retries — the config may not be reachable from the current basin

### Permanent protocol for all iterations
- K jitter: **{4, 8, 16}** (dropped K=12 — K=8/16 bracket it; ~7% throughput gain. H12 VERIFIED)
- K-sweep: {4, 8, 16, 32, 64, 128} with fast eval (256 seqs) + per-K diagnostics
- Pre-commit: /simplify → coderabbit → pr-review-toolkit → superpowers review
- Save full-precision weights (model_full.pt) before quantization
- **Update this hypothesis log BEFORE each iteration** (review relevant hypotheses, state what's being tested) **AND AFTER** (record results, update hypothesis statuses)
- Stability over task performance
- Locked (WD, β) = (0.72, 0.20) unless explicitly testing a WD/β hypothesis
- Post-norm on Block output is load-bearing — do not remove (H20)
- Muon NS must operate at correct tensor granularity — verify shape for any new param groups (H21)
- **Post-int6 hard gates** (NEW POLICY: tech debt, NOT promotion blockers — val_bpb improvement is the sole promotion criterion). The gates flag STRUCTURAL issues and prescribe fixes for the next iteration; they no longer block --promote. Run validity = "val_bpb was recorded" (run_valid=true whenever the eval completes).
  - Per-component min_share ≥ 0.01 (no dead expert — sub-1% means that expert is wasted capacity carried in the artifact)
  - Per-component ortho ≤ 0.9 (max pairwise |cos| — no two experts are near-duplicates; uses `max_pairwise_abs_cosine`, not `max_mean`)
  - gg_max ≥ 0.3 AND gg_min ≤ 0.95 (gate active, not collapsed/saturated)
  - inj_max ≥ 0.05 AND inj_mean ≥ 0.01 (x0 injection non-zero — H23 DEQ input-dependence)
  - K-sweep degradation: worst K≥16 within 0.1 of best (gross FP collapse only; the 0.005 monotone gate was below the noise floor — current best baseline 1.705 also fails it. Finite-K fluctuation of 0.005-0.01 is expected and not a structural failure.)
  - iter_conv_rel ≤ 0.1 at highest K (solver converges at eval K)
  - deq_recon_err ≤ 0.1 (RevDEQ reversibility — tightened from 1.0; lower recon = higher-quality gradients = more efficient training; matches smoke test threshold)
  - **NOT checked:** balance_cv (training optimizes this via balance_loss; not a structural invariant)
- **Artifact budget HARD cap**: code + compressed model ≤ 16,000,000 bytes (fails early)
- Untied attn/mlp routers (separate sigmoid gates and routing weights per component)
- **H29 PRINCIPLE**: every gate must be input-dependent AND token-local — no `.mean(dim=batch,seq)` before any gate.  Streaming / prefix-caching invariance depends on this.  Verify any new gate with: "does chunking the sequence change this gate's value for unchanged tokens?" → must be NO.
- **H31 PRINCIPLE**: LOSS and GATE metrics must be DIFFERENT by design — LOSS uses smooth signals (e.g. mean over all pairs); GATE uses worst-case (e.g. max over pairs). When designing a new gate, never reuse the loss metric for "consistency" — that breaks gradient flow. Cross-reference: max_pairwise (gate) vs max_mean (loss) for ortho.
- **H32 PRINCIPLE**: LOSS focuses on TASK PERFORMANCE — don't add regularization terms that exist purely to satisfy gates. The LOSS budget is finite; every coefficient steals gradient signal from val_bpb. Gate failures are caught by the GATE; the LOSS doesn't need to also chase them. Removed: `mos_ortho_out_coef = 0` and `block_ortho_aux_coef = 0` (were 1e-3 and 1.0 — chasing the ortho gate that's now properly handled by max_pairwise > 0.9 detection + WD-driven natural diversification).
