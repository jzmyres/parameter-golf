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

### H32: Contraction-preserving proposals only — PRINCIPLE (2026-04-16)
**Claim:** Every architectural change proposal must either (a) preserve strict contraction of $T_x$ under the chosen norm (so Banach still applies), or (b) arrive with a new convergence proof. A change that improves val_bpb while breaking the contraction property is a hidden regression at deep K (K=128+) — it exploits a specific iteration count rather than finding a unique fixed point.
**Supporting evidence (from prior iters):**
- iter 27c (router-only injection) — val_bpb +0.19, K=128=5.42 (Δ=3.31): solver diverges past K=16 because FP became x0-independent. Broke contraction in a specific direction.
- sigmoid-gate collapse (baseline → iter 29) — gate learned to ≈0.01, making injection negligible; FP equation approximately x0-independent, K=128 Δ=0.039 (tax for breaking dependence).
- iter 29a identity injection — preserved x0-dependence but broke contraction balance, val_bpb +0.044.
**Status:** ✅ VERIFIED PRINCIPLE (via counterexamples).
**Implication:** Before any smoke test for a new architectural proposal, audit **Lip_z(T_x) < 1** by tracing the Lipschitz constant through each module. Every proposal must trace to a sufficient condition in opg_doc.tex §6 (linear: `‖W‖_2 ≤ 1`; softmax: ≤ 1/2; RoPE: orthogonal; activation: 1-Lip; dense mixture: use `Π_R` projection or bound `L_w`).

### H33: Iter 30 contraction-shell is STRUCTURALLY aligned, not CERTIFIED — AUDIT (2026-04-16)
**Claim:** The current iter 30 implements the doc's *form* `T_x = (1-τ)b(x_0) + τ G_θ(z,x_0)` but does not satisfy the *premise* Lip_z(G_θ) ≤ 1.
**Component audit (current iter 30 code):**
| Component | 1-Lip in z? | Doc § | Fix needed |
|---|---|---|---|
| `b(x_0)` identity + `U·rms_norm(x_0)` | ✓ constant in z | §4.2 | — |
| `U` (inj_lin) | ✓ **CERTIFIED 1-Lip** via spectral_norm (iter 31, commit `e41dee9`) | §6.1 | done |
| `attn_norm`/`mlp_norm` (state path) | ✓ **CERTIFIED 1-Lip** via Π_R ball projection R=2√d (iter 32, commit `5cff932`) | §4.1 | done |
| `attn.expert_out`, `mlp.expert_down` (output expert matrices) | ✓ **CERTIFIED per-expert σ_max ≤ 1** via `PerExpertSpectralNormCap` (iter 33, commit `a14ac6e`) | §6.1 | done |
| `attn.expert_proj`, `mlp.expert_gate`, `mlp.expert_fc` (input-side expert matrices) | ✓ **CERTIFIED per-expert σ_max ≤ 1** via `PerExpertSpectralNormCap` (iter 33b, commit `a8c0ffb`) — **all 5 expert banks now certified** | §6.1 | done |
| router scoring (attn_router, mlp_router) | ✓ **CERTIFIED Lipschitz-bounded** via `s_j = tanh(-γ·‖x_n − c_j‖²)` + softmax (iter 34A, commit `bef50df`) — tanh is 1-Lip, softmax is ½-Lip per doc §6.4 | §4.3 B, §6.4 | done |
| `u = z + b(x_0)` | ✓ Lip=1 | §4.2 | — |
| `attn_norm`/`mlp_norm` (learnable RMSNorm) | ✗ | §4.1 | Replace with `Π_R` |
| `attn_router` (softmax of `Linear+logσ`) | ✗ (unbounded logits) | §4.3 | Use L2-distance or SIPS |
| `attn.mix_experts` (MLA + SDPA) | ✗ | §6.6 | Replace with L2 attention |
| `mlp.mix_experts` (unconstrained `‖W‖_2`) | ✗ | §6.1-6.2 | Spectral norm on expert weights |
| `attn_post_mix_norm`, `mlp_post_mix_norm` | ✗ | §4.1 | Replace with `Π_R` |
| `0.5·(Δ_a + Δ_m)` scale | ≤ 1/2 (helps) | §4.4 | — |
| `post_norm` (RMSNorm) | ✗ → **REMOVED** (2026-04-16 user direction) | §app:current_code | ✓ done; Π_R replacement queued for iter 32 |
| τ-shell `(1-τ)b + τG` with τ≤0.9 | scales Lip(G) by τ | §4.5 | — |
**Status:** Structurally aligned; Lipschitz certification deferred to iter 31-37 (systematic, one component per iter).
**Implication:** iter 30 is still worth running as a "structural baseline" for the contraction shell: tests whether the exogenous injection + τ-form alone improves val_bpb / K-sweep, without certification. Subsequent iters add 1-Lip constraints one at a time.

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

### H34: Minimal principled norms — Π_R inside T_x, RMSNorm outside
**Claim:** The clean separation is: inside the DEQ fixed-point map T_x(z), use ONLY non-expansive Π_R clamps (1-Lip by construction, no analysis needed).  Outside the DEQ solve (embeddings, final head), keep norms for optimization without affecting contraction.
**Insight from iter 37b:** Replacing per-component RMSNorm with per-component Π_R caused +0.069 val_bpb regression BUT K=128 Δ=0.003 (tightest ever).  The design had 4 separate Π_R projections — too many clamps, too restrictive.
**Revised design (iter 39):** Just TWO Π_R projections inside Block.forward:
1. One shared pre: `u = Π_R(z + b(x0))` — replaces attn_norm + mlp_norm
2. One post on combined update: `Δ = Π_R(0.5*(Δ_attn + Δ_mlp))` — replaces all post-norms
All other norms inside T_x REMOVED (Q/K norms, hidden_post_norm, sdpa_post_norm).
**Prediction:** Simpler + fewer clamps = less capacity loss than iter 37b's 4× Π_R.  K-sweep should stay tight (1-Lip chain + Π_R post).  Val_bpb should be within carry-forward threshold because the model has ONE shared input (no separate norm paths) and ONE output clamp (vs 4 in iter 37b).
**Test:** iter 39-minimal-principled-norms (after iter 35).
**Status:** PROPOSED — queued.

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

### H35: β jitter improves K=128 extrapolation — VERIFIED (2026-04-19)
**Claim:** Training at varying β per step forces the model to learn solver-agnostic convergence.
**Test:** Iter 49 — sample β from {0.3, 0.5, 0.7} per step (like K-jitter H12).
**Evidence:** K=128 Δ collapsed from 0.008 (fixed β=0.5) to **0.000** (β jitter). val_bpb unchanged (2.1149 both). Near-perfect FP convergence across all K.
**Status:** ✅ VERIFIED — β jitter is now permanent (like K-jitter).
**Implication:** H30 confirmed. β jitter and K-jitter are complementary regularizers — both should remain active.

### H36: Fixed β=0.7 breaks RevDEQ reconstruction — VERIFIED (2026-04-19)
**Claim:** Higher β amplifies RevDEQ reconstruction error via 1/(1-β) factor.
**Test:** Iter 48 — fixed β=0.7 (was 0.5).
**Evidence:** deq_recon_err=1.205 (vs 0.886 at β=0.5). Workers crashed silently after step 100. Reconstruction divides by (1-β)=0.3, amplifying errors 3.33× vs 2× at β=0.5.
**Status:** ✅ VERIFIED — fixed high β is incompatible with RevDEQ. β jitter (H35) is the safe approach.

### H37: DeepSeek shared expert improves expert specialization — PROPOSED
**Claim:** Dedicating 1-2 of 8 experts as always-on (g=1, bypass routing) offloads universal patterns, freeing routed experts for specialization.
**Mechanism:** T_θ = x0 + E_shared(h) + Σ w_j E_routed_j(h). Shared expert handles common transformations every iteration; routed experts specialize.
**Expected:** Lower attn_ortho (better expert diversity), potentially better val_bpb from more efficient capacity allocation.
**Reference:** DeepSeek-V3 (arXiv:2401.06066)
**Risk:** Low — routing-only change, zero param increase, easily reversible.

### H38: KV latent subspace orthogonalization — PROPOSED
**Claim:** Penalizing off-diagonal Frobenius norms of expert KV down-projection interaction matrix guarantees distinct latent subspaces.
**Mechanism:** G_ij = ||W_i^T W_j||_F for i≠j; penalty = ||G_offdiag||²_F. Forces expert KV compressions to be orthogonal in weight space.
**Expected:** Structural guarantee of expert diversity (vs current output-level ortho which depends on input distribution).
**Risk:** Low — training-time penalty only, no inference cost.

### H39: Causal conv1d pre-conditioning improves z0 quality — PROPOSED
**Claim:** A lightweight causal conv1d (kernel=4) after embedding provides local temporal context, placing z0 closer to the fixed point and reducing effective K needed.
**Mechanism:** x0 = RMSNorm(conv1d(tok_emb + bigram)). Runs OUTSIDE DEQ loop, one-time per forward.
**Expected:** Faster DEQ convergence (fewer iterations to reach ε tolerance). Could enable reducing K_max.
**Reference:** Simplified from SSM/Mamba-2 proposal; ELM (ICLR 2026) identity-init finding.
**Risk:** Medium — adds parameters (compete with 16MB budget), may not help at short training.

### H40: Removing K=4 from jitter set improves deep-K quality — PROPOSED
**Claim:** K=4 in the jitter set {4,6,10} trains the model to produce useful output in only 4 DEQ iterations. This "shallow convergence" pressure conflicts with deep-K quality (K=64/128), where the model should keep refining. Removing K=4 → {6,10} forces a minimum of 6 iterations, training deeper fixed-point structure.
**Mechanism:** At K=4 the model learns to "settle" early — gates close, updates vanish by iter 4. This learned early-convergence behavior persists at eval-time K=16/64/128, limiting how much the model benefits from additional iterations. By guaranteeing K≥6, the model must learn to do useful work through at least 6 steps.
**Evidence:** K=2 curriculum (iter 53) created permanent quality deficit (+0.058-0.122 train_loss gap). If K=2 is catastrophic, K=4 may be mildly harmful. K jitter H12 was VERIFIED with {4,8,12,16}; the benefit came from *breadth* of jitter, but the *floor* was never tested.
**Test:** Iter 59 — change deq_k_jitter_set from (4,6,10) to (6,10). Compare val_bpb and K-sweep quality.
**Risk:** Low — K=6 is still shallow enough for fast steps. Throughput may improve slightly (K=4 steps are cheapest but also least useful for training signal).
**Status:** PROPOSED

### H41: K=32 in jitter set with TBPTT trains true FP — PROPOSED
**Claim:** The current K_max=10 means the model never trains beyond 10 iterations. At eval K=16+ it extrapolates beyond training distribution. Including K=32 in the jitter set trains the model to optimize the true fixed point, not partial convergence states.
**Mechanism:** TBPTT (deq_bptt_k=4) decouples backward cost from forward K: backward always unrolls only the last 4 iterations regardless of K. So K=32 forward costs 32×2=64 block forwards (~1280ms), but backward costs the same 4 VJPs as K=6. Sampled 1/3 of the time in jitter set {6,10,32}, amortized overhead is ~3% per step — negligible.
**Key insight:** At K=32 the model is close to true z*. The TBPTT gradient flows through the last 4 of those 32 iterations, giving the optimizer direct signal about what the Jacobian looks like NEAR the fixed point. Without K=32, the optimizer only sees Jacobian behavior at iterations 1-10 — far from z*.
**Evidence:** H12 showed wider K jitter improves K-sweep (125× improvement). Iter 55 K-sweep shows K8→K128 Δ=+0.0004 — a residual training-distribution bias. K=32 in the jitter set should eliminate this by making deep-K states part of the training distribution.
**Test:** Iter 59 — change deq_k_jitter_set from (6,10) to (6,10,32). Compare K-sweep quality at K=64/128.
**Risk:** Low — TBPTT keeps backward cost constant. Only forward cost increases for K=32 samples.
**Status:** PROPOSED

### H42: Full-rank MLA pre-conditioning (DeepSeek-style non-MoE layer) — PROPOSED
**Claim:** A full-rank MLA block run ONCE before the DEQ loop gives z0 full-sequence
attention context, placing it closer to z*. Like DeepSeek-V3's non-MoE layers at the
bottom of the stack — one standard attention pass before the iterative MoE layers.
**Mechanism:** x0 = MLA_precond(tok_emb + bigram). Full attention (not routed) → z0
has global context before entering the DEQ solver.
**Risk:** Adds ~2-5M params and ~20ms per forward. RevDEQ-safe (outside solver).
**Status:** PROPOSED

### H43: Low-dim expert computation (D→r per expert, mix in D-space) — PROPOSED
**Claim:** Instead of experts operating at full D=768 with internal low-rank projections,
each expert should project to its OWN low-dimensional subspace (D→r), do ALL computation
(attention, MLP) at dim r, then project back (r→D) before router-weighted mixing.
**Mechanism:**
```
for expert_i:
    z_i = down_i(x)           # D → r  (independent projection per expert)
    q_i, k_i, v_i = attn(z_i) # all at dim r (smaller/fewer heads)
    a_i = sdpa(q_i, k_i, v_i) # attention at dim r
    m_i = mlp(a_i)             # MLP at dim r
    y_i = up_i(m_i)            # r → D  (project back to full space)
output = Σ w_i · y_i           # mix in full D-space
```
**Why this helps:**
- Throughput: attention at dim r costs O(r²) vs O(D²) — r=192 → 16× cheaper per expert
- VRAM: intermediate tensors at r, not D. Less activation memory.
- Params: expert weights scale as O(r²), freeing budget for more experts
- Expressiveness: E experts at rank r → aggregate rank E×r. With E=8, r=192: 1536 > D=768
- Enables scaling to 16-32 experts at same compute budget
**Head-packed SDPA compatibility:** All experts use same r → packable as E×H_r heads.
If r=192 and H_r=4 heads per expert, that's 32 query heads total — single FlashAttention call.
**Difference from current low-rank:** Current design uses low-rank WITHIN each linear layer
(D→rank→D), but attention and MLP intermediate tensors are still D-dimensional. This
proposal puts the entire expert computation in a lower dimension.
**Unlocks model_dim scaling:** With expert compute at dim r (independent of D), increasing
model_dim from 768→1024 costs only +33% on down/up projections (linear matmuls) while
SDPA and MLP costs stay constant. Net: ~33% more mixing expressiveness at negligible
throughput degradation. Binding constraint becomes 16MB artifact budget, not compute.
**Pre-conditioning residual:** All pre-conditioning variants (conv1d, MLA, low-rank experts)
MUST preserve residual to raw token embedding: x0 = precond(x) + x. The raw embedding
is the identity signal that x0 falls back to when pre-conditioning is unhelpful.
**Risk:** Medium — significant architectural change. Head-packed SDPA needs validation at
smaller head dim. Per-expert expressiveness decreases (compensated by aggregate rank).
**Status:** PROPOSED

### H44: Full-rank expert internals at dim r (remove low-rank factorization) — PROPOSED
**Claim:** With H43 moving expert computation to dim r, low-rank factorization inside
experts becomes unnecessary. Expert linear layers should be full-rank at dim r instead
of factored (r→rank→output). This is simultaneously simpler, more expressive, and faster.
**Current design (low-rank at D=768):**
```
Q: x(768) → A(768×128) → B(128×out)   # two matmuls, rank=128, params=768×128+128×out
```
**Proposed (full-rank at r=192):**
```
x(768) → down(768×192) → z(192) → W(192×out)   # one matmul, full rank=192, params=192×out
```
**Three simultaneous wins:**
1. **More expressive**: full rank 192 > artificially limited rank 128. No information
   bottleneck within the expert.
2. **Fewer FLOPs**: one matmul instead of two per layer. Fewer kernel launches, better
   GPU utilization. With torch.compile, fusing one large GEMM > two small GEMMs.
3. **Fewer params**: 192×192=37K vs 768×128+128×192=123K per expert Q. The small
   dimension r naturally controls params — the factorization was redundant.
**Simplifies hyperparameters:** `attn_expert_rank` and `mlp_expert_rank` become unnecessary.
Expert capacity controlled by single knob: r (the expert subspace dimension).
**Depends on:** H43 (low-dim expert computation). Without H43, full-rank at D=768 would
explode params (768²=590K per expert per layer).
**Status:** PROPOSED

### H45: Independent-weight MLA pre-conditioning block — PROPOSED
**Claim:** The pre-conditioning block for z0 must have INDEPENDENT weights from the DEQ
shared block. Reusing shared_block weights makes the pre-conditioning pass equivalent to
K+1 DEQ iterations — not a fundamentally different computation. An independent block
provides qualitatively different z0 context (like DeepSeek-V3's non-MoE layers).
**Mechanism:** A separate Block instance (DeepSeek-style MLA attention block) with its own
parameters and standard residual connection: `x0 = precond_block(emb) + emb`. The output
of pre-conditioning becomes the x0 input to the DEQ solver. The residual ensures the raw
embedding signal is always preserved.
**Core insight:** The pre-conditioning creates a **dynamic, context-aware embedding** rather
than a static one. Currently x0 = tok_emb + bigram — each token's x0 depends only on its
identity. The DEQ solver must discover ALL inter-token relationships through iterations.
With MLA pre-conditioning, x0 already encodes full-sequence attention context. The DEQ
solver iterates on a richer input, focusing on deeper structure rather than basic context.
**Architecture:** Same structure as a standard transformer attention block (not necessarily
MoE). Could be a single-expert MLA block (no routing) or a smaller MoE block. Independent
weights because it solves a different problem (context building) than T_θ (FP refinement).
**Cost:** ~2-5M params depending on rank/expert config. ~10-20ms compiled per forward
(amortized over K DEQ iterations). At 11M current params, budget is tight — may need
rank reduction or fewer experts in the DEQ block to compensate.
**Risk:** High param cost. May need to reduce DEQ block rank to fit 16MB.
**Status:** PROPOSED

### H46: Exponential-distribution K sampling for DEQ jitter — PROPOSED
**Claim:** Instead of uniform sampling from a fixed set {4,6,10}, sample K from an
exponential distribution with low mean (e.g., λ=1/8, mean K=8). This gives:
- **Most steps at low K** (cheap, maintains step count throughput)
- **Occasional high K** (K=20-40, provides deep FP training signal)
- **Rare very deep K** (K=60+, trains near-true FP, but only ~2% of steps)
**Why better than fixed set {4,6,10,32}:** Iter 59 showed K=32 at 25% sampling rate
cost -24% steps. With exponential sampling, K>20 might only be sampled ~10% of the time,
and K>32 only ~5%. The deep-K training signal is still present but amortized over many
more cheap steps. Continuous distribution also avoids the "mode" artifacts of a discrete set.
**Implementation:** `K = max(4, min(64, int(np.random.exponential(scale=8))))` per step.
Round to even for RevDEQ compatibility. TBPTT=4 keeps backward cost constant.
**Expected throughput:** mean K≈8 (same as current avg of {4,6,10}), but with a heavy tail
providing occasional deep-K signal. Step count should match baseline (~747 steps).
**Risk:** Low — no structural change, just K sampling strategy. Easy to tune scale parameter.
**Status:** PROPOSED

### H47: Scale up number of experts — PROPOSED
**Claim:** More experts (16, 32) at same or reduced rank improves routing diversity and
model capacity. Currently 8 experts (7 routed + 1 shared). Scaling to 16-32 experts gives
better coverage of the input space.
**With H43 (low-dim experts):** Each expert computes at dim r, so adding experts costs
only the down/up projections (2×D×r per expert) plus small r-dim internal weights. Going
from 8→16 experts at r=128 adds ~2×768×128×8 = 1.6M params (15% of model). Throughput
stays similar if head-packed SDPA can handle 16×H heads.
**Without H43:** At current full-dim, 16 experts at rank 128/192 was tested in iter 15
(stable, no collapse at WD=0.72) but throughput penalty dominated (11.6s/step). Low-dim
experts (H43) would make this affordable.
**Evidence:** Iter 15 confirmed 12 experts stable at WD=0.72. Iter 3a3 confirmed 8 experts
balanced. The scaling law is: more experts = better IF per-expert compute is cheap enough.
**Risk:** Head-packed SDPA with 16×8=128 query heads may hit FlashAttention limits.
Routing balance harder with more experts (higher balance loss needed).
**Status:** PROPOSED

### H48: Parcae negative diagonal injection (guaranteed contraction) — PROPOSED
**Source:** "Parcae: Scaling Laws For Stable Looped Language Models" (arXiv:2604.12946, April 2026)
**Claim:** Replace scalar β with per-dimension learned `A = Diag(-exp(a))` and step size dt.
Update: `z_{n+1} = exp(dt·A)·z_n + (I-exp(dt·A))·f(z_n, x0)`. Since -exp(a) < 0 always,
exp(dt·A) has all eigenvalues in (0,1) → spectral radius < 1 by construction.
**Why high ROI:**
- Eliminates need for Hutchinson AND denoising regularization (guaranteed stability)
- Net code REDUCTION (~60 lines removed, ~30 added)
- Per-dimension learned damping → more expressive than scalar β
- Parcae 770M matches 1.3B standard transformer
**RevDEQ compatibility:** Need to verify reversibility with diagonal parameterization.
The reverse step z_prev = (z - (I-exp(dt·A))·f(y)) / exp(dt·A) is still algebraically
exact since exp(dt·A) is diagonal and invertible (all entries > 0).
**Risk:** Medium — changes the core solver dynamics. Needs careful integration with β jitter.
**Status:** PROPOSED

### H49: Per-iteration LoRA adapters — PROPOSED
**Source:** "Relaxed Recursive Transformers" (ICLR 2025, arXiv:2410.20672)
**Claim:** Add tiny rank-4 LoRA offsets (B_i·A_i) per DEQ iteration to shared block's key
projections. Each iteration gets slightly different behavior while maintaining weight sharing.
**Params:** 12 iterations × rank-4 × 768 × 2 matrices = 73,728 params (~0.7% of model).
**Why high ROI:**
- Current DEQ: all iterations use IDENTICAL weights → limited depth utilization
- Per-iter LoRA: each iteration can specialize (early: coarse features, late: fine details)
- Tiny parameter cost, negligible throughput overhead
- Recursive Gemma 1B with LoRA outperforms TinyLlama 1.1B
**Risk:** May interfere with fixed-point convergence if LoRA offsets too large.
Init with small scale (1e-3) to start near identity.
**Status:** PROPOSED

### H50: DeltaDEQ dimension skipping — PROPOSED
**Source:** "DeltaDEQ: Exploiting Heterogeneous Convergence" (NeurIPS 2024)
**Code:** github.com/ZuowenWang0000/Delta-Deep-Equilibrium-Models
**Claim:** Track per-dimension convergence δ_d = |z_new[d] - z_old[d]|. Skip recomputation
for dimensions where δ_d < ε in later iterations. Many dimensions converge by iter 4-5.
**Why high ROI:**
- 20-40% wall-clock speedup with <0.01 bpb cost
- With MoE, could skip entire expert evaluations when routing weights stabilize
- Only affects forward pass (backward uses TBPTT on last 4 iters, all dims)
**Risk:** Sparse ops may not play well with torch.compile. RevDEQ backward needs all dims
for reconstruction. May only be applicable to forward, not backward.
**Status:** PROPOSED

### H53: Disable FSQ quantization, keep low-rank MoS projection — PROPOSED
**Claim:** FSQ (Finite Scalar Quantization) applies level discretization via STE in the
MoS head's intermediate projection. The STE gradient approximation (round in forward,
pass-through in backward) introduces a gradient mismatch that may hurt training quality.
The low-rank projection alone (without quantization) provides sufficient parameter
compression for the 16MB artifact budget.
**Mechanism:** Set `fsq_levels=0` (or bypass the FSQ quantize step) in MoSHead while
keeping the low-rank projection. The projection still maps through a bottleneck rank
for compression, but values are continuous (not discretized to finite levels).
**Why it might help:**
- STE gradient bias: round(x) has zero gradient almost everywhere, STE uses identity
  gradient as approximation. This mismatch accumulates over training.
- The low-rank projection already compresses the weight representation. FSQ on top of
  low-rank may be over-constraining.
- At int6 quantization for the artifact, the final weights are already discretized.
  FSQ during training adds a SECOND quantization step that's redundant.
**Risk:** Low — if FSQ helps quantization robustness, val_bpb may slightly increase.
But the quant gap (pre-quant vs post-quant) has been tiny (0.002-0.008) in recent iters,
suggesting FSQ's quantization-awareness isn't needed.
**Status:** PROPOSED

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
| **28-tbptt-k4** | Truncated BPTT k=4 (gradient-scale hypothesis). | 2.33 @ step 200 (died) | **REVERTED (`7f1f907`)** | VJP decay ratio 0.77-0.96 (flat vs expected 0.5). Tail-4 captures only 40% → 60% gradient discarded. Throughput +36% but val_bpb trailing. |
| **28b-tbptt-k8** | Milder k=8 (~83% capture). | **1.974** | **REVERTED** | FAIL by +0.072. Too much bias without compensation. |
| **28c-tbptt-deeper-K** ★ | k=8 + K jitter (4,8,16,24), eval K=24. Use TBPTT savings for deeper K training. | **1.925** | **KEPT as gen-scaffold** (not promoted — val_bpb regression) | **K=128 Δ=0.015 vs baseline 0.039 — 2.6× TIGHTER**. Deeper-K hypothesis CONFIRMED. Val_bpb cost is the embed gradient truncation tax. K-sweep: k4=1.978 k8=**1.926** k16=1.932 k32=1.938 k64=1.940 k128=1.941. |
| **28d-tbptt-k12** | Reduce truncation (k=8 → k=12, ~92% capture) to recover val_bpb gap. | **1.963** | **REGRESSED vs 28c** | FAIL by +0.061. K=128 Δ=0.012 (marginal vs 28c's 0.015). **Regressed on val_bpb** because +9% per-step cost (k=12 backward) gave only 425 steps vs 28c's 489. **TBPTT Pareto optimum is 28c (k=8 + K=24)**. |

**TBPTT investigation closed (2026-04-16).**  Best point = 28c val_bpb 1.925
(+0.023 vs baseline 1.902) trading task perf for **2.6× tighter K-sweep**.
No config reached val_bpb gate.  Machinery retained (`deq_bptt_k=0` default,
opt-in via CLI `--deq-bptt-k=N`).  The K-sweep tightening is a real
generalization win and should be combined with Phase 6 contraction-shell
architecture for a possible compounding improvement.

### Phase 5e/5f/5g: CLOSED (2026-04-16)

Removed from active queue per user direction.  Summary of concluded work:

**Phase 5e (injection mechanism)** — Position sweep 27a/b/c/d + transformation
sweep 29a/b ran to completion.  Position 27b (P-expert-out) was briefly
baseline (val_bpb 1.902), but both identity (29a: 1.946) and full-rank W
(29b: 1.953) **regressed** vs the sigmoid gate.  **Root-cause finding
(user):** the sigmoid gate enables convergence to a trivial FP (x0-
independent) because it can collapse to ~0, AND WD=1.08 forces any
learnable injection parameter toward 0.  Neither identity nor full-rank W
gave a Pareto improvement.  **Superseded by Phase 6 contraction shell**
(exogenous injection `b(x_0) = x_0 + U·x_0` with identity path PLUS
contraction wrapper `T_x = (1-τ)b + τG` — see new queue below).

**Phase 5f (norm ablation)** — Dropped.  The proposed 1-Lipschitz
projection `Π_R` from opg_doc.tex §4.1 supersedes ad-hoc RMSNorm
removal.  Norm redesign will happen as part of Phase 6 Lipschitz
certification, not as independent ablation.

**Phase 5g (deferred: PSD, learnable-β, TBPTT)** —
- TBPTT: tested (28-28d); conclusion below.
- Learnable-β: still deferred (needs RevDEQ backward surgery).
- PSD on expert layers: **DROPPED** — superseded by spectral-norm
  constraint on U (`‖U‖_2 ≤ 1`) and 1-Lipschitz FFN experts per
  opg_doc.tex §6 (same goal: controlled Lipschitz; cleaner
  formulation).
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

### Phases 1-5: COMPLETE (summary only; see § "Completed Iterations" above for results)
- **Phase 1** (iters 13-14b): H9, H12, H18 VERIFIED. Locked (WD=0.72, β=0.20, K jitter).
- **Phase 2** (iters 15-16): H5 RESOLVED (collapse = WD-fixable). Gate stats infra.
- **Phase 3** (iters 17-20): Router sigmoid gate, additive injection, post-norm, quant-noise REFUTED.
- **Phase 4** (iters 21-22): Throughput + untied routers + systematic norm ablation.
- **Phase 4.5 result:** 22-add-all + targeted norm removals landed val_bpb 1.891-1.955.
- **Phase 5a/b** (iters 23A-23C): injection rework (z_hat = z + B(A(LN(x0))) or residual). 23C-gg-no-residual baseline val_bpb ≈ 1.94-1.95.
- **Phase 5c** (iters 24-27, 28-28d): WD bump (24), injection position/transformation sweep (27-29b), TBPTT (28-28d). **All closed.** See iteration table at line ~316.
- **Phase 5d** code hygiene: 5.3/5.4/5.5/5.6 deferred as tech debt; not blocking.
- **Phase 5e/5f/5g (norm ablation + injection mechanism + PSD-expert-layers): CLOSED** (2026-04-16 user direction). Superseded by Phase 6 below.

**Current baseline:** 27b `9edc6af`, val_bpb=1.9020, K=128 Δ=0.039.

### Phase 6: Certified contraction shell (opg_doc.tex §3-6, 2026-04-16 — ACTIVE)

**Guiding principle (opg_doc.tex, H32):** every proposal must preserve strict
contraction of $T_x$ under Frobenius norm (so Banach applies), be **simple**
(one change at a time), and **principled** (traceable to a sufficient condition
in doc §6).  Before the smoke test for any new iter, audit Lip_z(T_x) < 1
tracing through each module; update H33 table with the outcome.

**Certified design target (doc §4-5):**
```
b(x_0) = x_0 + U · Π_R(x_0)                # exogenous, identity-preserving
u(z, x_0) = Π_R(z + b(x_0))                # shared expert/router input
G_θ(z, x_0) = 0.5 · (Δ_attn + Δ_mlp)       # parallel mix of 1-Lip experts
T_x(z) = (1-τ) · b(x_0) + τ · G_θ          # strict contraction (τ_max<1)
```

**Sequential queue** (iter runs only after predecessor promotes).  Each row
tests ONE doc-aligned change.  Fill result columns on completion; revert on
failure.

| # | Iter name | Change (one component) | Doc § | Status | val_bpb | K=128 Δ |
|---|---|---|---|---|---|---|
| 30 | **contraction-shell** | exogenous `b(x_0)=x_0+U·rms_norm(x_0)` + τ-shell + shared `u=z+b(x_0)` + `post_norm` removed | §4.2, §4.4, §4.5 | **PROMOTED ★ (new Phase 6 baseline, commit `64387e3`)** | **1.9223** (+0.020 vs 27b=1.902) | **0.018** (2.2× tighter than baseline 0.039) |
| 31 | spectral-U | `‖U‖_2≤1` via `nn.utils.parametrizations.spectral_norm` (1 power iter/fwd) | §6.1 | **PROMOTED ★ (commit `e41dee9`)** | **1.9294** (+0.007 vs iter30) | **0.019** (≈ iter30's 0.018) |
| 32 | pi_R-state | replace `attn_norm`, `mlp_norm` on state path with Π_R ball projection (R=2√d≈55.4). Post-mix norms deferred to later iter. | §4.1 | **PROMOTED ★ (commit `5cff932`)** | **1.9197** (-0.010 vs iter31) | **0.012** (tightest so far) |
| 33 | spectral-experts | per-expert σ_max ≤ 1 via `PerExpertSpectralNormCap` on `attn.expert_out` and `mlp.expert_down` (output-side matrices) | §6.1, §6.6 | **PROMOTED ★ (commit `a14ac6e`)** | **1.9266** (+0.007 vs iter32) | **0.020** (≈ iter32's 0.012) |
| 33b | spectral-experts-full | extend per-expert σ_max ≤ 1 to `attn.expert_proj`, `mlp.expert_gate`, `mlp.expert_fc` (input-side matrices); total 5 banks certified | §6.1 | **PROMOTED ★ (commit `a8c0ffb`)** | **1.9174** (-0.009 improvement!) | **0.016** (tighter) |
| 34A | router-L2 | L2-distance router `s_j=tanh(-γ‖q-c_j‖²)` with γ=1.0 + learnable prototypes | §4.3 Option B | **PROMOTED ★ (commit `bef50df`)** | **1.9217** (+0.004 vs iter33b) | **0.037** (looser than iter33b's 0.016, still ≤ 0.5) |
| 34B | router-SIPS | cosine-similarity `s_j=γ·cos(q, k_j)` (reuses iter 34A prototypes) | §4.3 Option A | **A/B LOSER** (commit `a405e9e` not promoted) | 1.9267 (+0.005 vs 34A) | **0.030** (better than 34A's 0.037 but loses on val_bpb) | SIPS has tighter K-sweep but +17% per-step cost kills wallclock val_bpb. L2+tanh remains default. |
| 35 | **single-router (MANDATORY, doc §4.3)** | Single pooled `SoftDenseRouter(2E=16)`, one softmax across combined pool, `w_attn = w[..., :E]`, `w_mlp = w[..., E:]`. Forces per-token attn-vs-MLP budget allocation. Required @dynamo_disable on _route_pooled for torch.compile compat. | §4.3 | **PROMOTED ★ (commit `2f60108`)** | **1.9197** (+0.009 vs 37=1.9111) | **0.016** (slightly looser than 37's 0.011) |
| 36 | **L2-attention + Phase 6a fixes** | MLA+SDPA → L2 attention via SDPA fast path (RMS-norm cancellation trick, γ=0.051) + Phase 6a: deterministic spectral-norm, matmul router, fp32 softmax, bounded prototypes, SDPA fail-loud | §6.6 | **PROMOTED ★ (commit `9146d74`)** | **1.9167** (-0.005 vs 34A=1.9217) | **0.030** (tighter than baseline 0.037) |
| 37 | **lipschitz-mlp** | Drop `.square()` from MLP activation: `leaky_relu(0.5)*fc` (1-Lip). 3% throughput gain (459 vs 447 steps). K=128 Δ collapsed 0.030→0.011 (2.7× tighter FP) | §6.2 | **PROMOTED ★ (commit `e3fa2eb`)** | **1.9111** (-0.006 vs 36=1.9167) | **0.011** (2.7× tighter than iter 36's 0.030) |
| 37b | post-mix-norm-Π_R | Replace `attn_post_mix_norm` and `mlp_post_mix_norm` from RMSNorm → BallProjection(R=2√d). K=128 Δ spectacularly tight (0.003) but val_bpb regressed +0.069 uniformly. **RMSNorm's learnable scale provides capacity Π_R cannot.** | §4.1 | **REVERTED** | 1.9801 (+0.069 vs 37=1.9111) | 0.003 (best FP ever, but val_bpb fails gate) |
| 39 | minimal-Π_R | Remove ALL RMSNorm inside T_x, 2×Π_R only (pre+post). Remove ½ factor. | §4.4 | **REVERTED** | 2.0295 (+0.115) | 0.001 |
| 39b | **certified-contraction** | Homogeneous coord L2-attn (no Q/K rms_norm), exogenous τ(x_0)=τ_max·σ(f(b(x_0))), analytic τ_max=c/L_G, remove ½, R=√d_head. **PERFECT FP: K=128 Δ=0.0000.** But τ≈0.016 too small for val_bpb. P0 fixes (b3ef90e) make τ even smaller (0.0027). | §4.4-4.5 | **REVERTED** (val_bpb +0.22) | 2.1421 (+0.22 vs 35=1.9197) | **0.0000** (perfect FP — Banach validated) |
| 40 | **orthogonal-experts** | Replace all `PerExpertSpectralNormCap` (σ_max ≤ 1) with `OrthogonalParametrization` (all σ = 1, exact isometry) via Newton-Schulz iteration (20 iters). Applied to 5 expert banks + inj_lin. Stateless, RevDEQ-safe. step_avg=9.0s (+15% vs baseline 7.8s). **All-σ=1 isometry too restrictive** — kills singular value diversity needed for expressiveness. Same regression magnitude as iter 39b. Validates Phase 7.8 L1 (relax to σ_max ≤ c). | §6.1 | **REVERTED** (val_bpb +0.226) | 2.1456 (+0.226 vs 35=1.9197) | TBD (training killed early) |
| 41 | **lyapunov-arch** | **Banach→Lyapunov paradigm shift.** Remove ALL hard constraints: Π_R→RMSNorm, remove spectral-norm caps, remove τ-shell/inj_lin. T_θ(z,x₀) = x₀ + Δ_θ(z,x₀) with shared `state_norm=RMSNorm(z+x₀)`. Remove 0.5 scale. Keep post-mix RMSNorm, L2-attn, leaky_relu. step_avg=7647ms (vs baseline 8088ms, **5% faster**). 600K fewer params (9.14M vs 9.73M). Smoke: iter_conv 8.7→3.9 (solver learning stable FPs naturally). -176 net lines of constraint code removed. | §3 (Lyapunov) | **PROMOTED ★ (commit `3c588ce`)** | 2.1585 (fast@200, ≈baseline 2.1668) | TBD |
| 45 | **lyapunov-full** | Full doc alignment: SwiGLU + standard SDPA + NormedLinear (kv_pre_norm, expert_h_pre_norm) + Lyapunov boundary VJP penalty (surrogate loss, EMA μ=0.9, outside compiled graph, donated_buffer=False). Q/K rms_norm restored (removing crashes AOTAutograd). **Best val_bpb yet.** | §3-4 (Lyapunov) | **PROMOTED ★ (commit `2d71a79`)** | **2.1376** (fast@200, -0.03 vs baseline) | TBD |
| 46 | **independent-expert-MLA** | Replace shared SDPA + expert output banks with full per-expert MLA pipeline: per-expert low-rank Q (dim→rank→H*d+H), per-expert low-rank KV compression (dim→kv_rank→kv_latent), per-expert KV decompression (kv_latent→K_nope,V — independent dicts), shared K_rope. Head-packed SDPA with E×H=64 Q heads, E×H_kv=32 KV heads, single FlashAttention call. Net -440K params. **Massive val_bpb improvement.** OOM during roundtrip verification (43GB/44.4GB VRAM) — tech debt. | §3.4 | **PROMOTED ★ (commit `cb6fd63`)** | **1.7806** (-0.357 vs iter45=2.1376, -16.7%) | TBD |
| 47 | **fully-independent-experts** | Zero shared trainable params in expert path: per-expert K_rope (low-rank kr_rank=32), per-expert KV/MLP norm weights, per-expert Wo output proj (low-rank wo_rank=64). RevDEQ default + block.forward compile (2× speedup). Diagnostics fixed: pooled router dedup, per-type min_share=0.6/E, ortho gate=0.5. K-sweep true FP gate (K=64,128 non-degradation). +1.04M params. Step-matched at 709 steps (2h wallclock). Peak VRAM 22.3 GB (vs 42.9 GB with unroll). Expert independence is HARD CONSTRAINT. | §2,§3 | **PROMOTED ★ (commit `b1565a7`)** | **1.8243** (709 steps, +0.044 vs iter46@722 steps) | TBD |

### Phase 9: DEQ Architecture Improvements (2026-04-19 — ACTIVE)

**Sources:** DEQ literature survey (Bai 2021, Efficient DEQ 2025, HyDRA 2026, DeltaDEQ NeurIPS 2024, ELM ICLR 2026) + user proposals (DeepSeek shared expert, SSM pre-conditioning, KV latent ortho).

**Training budget:** 1.5hr wallclock (5400s) for convergence. Post-warmup step_avg ~8,500ms.

| # | Iter | Change | Hypothesis | Status | val_bpb | K=128 Δ |
|---|---|---|---|---|---|---|
| 48 | β=0.7 fixed | Higher β for faster convergence | H36 | **FAILED** (high recon_err, worker crash) | — | — |
| 49 | β jitter {0.3,0.5,0.7} | Per-step β sampling (K-jitter analog) | H35 ✅ | **KEPT** (K128 Δ=0.000, 8× tighter) | 2.1149 | **0.000** |
| 50 | Hutchinson Jacobian reg | Replace Lyapunov power-iter with random VJP | Bai 2021 | **KEPT** (smoother contraction) | 1.8673 | 0.000 |
| 51 | DeepSeek shared expert | 1 shared + 7 routed (always-on universal expert) | H37 | **KEPT** (-0.009 val_bpb) | 1.8651 | 0.000 |
| 52 | KV latent subspace ortho | Penalize off-diag cosine on KV down-proj | H38 | **PROMOTED ★** (baseline) | 1.8651 | 0.000 |
| 53 | K curriculum | Shallow K early → deep K late | DEQ practices | **REVERTED** (K=2 too aggressive, permanent deficit) | 1.875 | — |
| 54 | Avg FP warm start | Init z0 from previous batch z* | Efficient DEQ 2025 | **REVERTED** (neutral, +0.002) | 1.867 | — |
| 55 | Denoising regularization | ||f(z*+ε,x0) - z*||² post-convergence | HyDRA 2026 | **PROMOTED ★** (val_bpb -0.042, near-perfect FP) | 1.8236 | +0.0004 |
| 56 | Causal conv1d pre-conditioning | Conv1d(k=4) before DEQ for temporal z0 | H39 | **REVERTED** (+0.008, conv1d didn't help) | 1.8311 | -0.006 |
| 57 | Anderson accel (eval only) | 2-8× eval speedup, K-sweep quality | arXiv:2410.19460 | **DEFERRED** (K-sweep already near-perfect Δ=0.0004) | — | — |
| 58 | K jitter: drop K=4 | Remove K=4 from {4,6,10} → {6,10} | H40 | **REVERTED** (+0.009, fewer steps outweighed tighter FP) | 1.8329 | +0.0003 |
| 59 | K jitter: add K=32 | {4,6,10,32} with TBPTT=4. Deep K without removing cheap K | H41 | **REVERTED** (+0.050, -24% steps dominated. K32 optimal in sweep though!) | 1.8733 | +0.0002 |
| 60 | Independent MLA pre-cond | Separate Block (1 expert, rank 128/192) for dynamic x0 | H45 | **REVERTED** (+0.036, DEQ attention already builds context) | 1.8593 | +0.003 |
| 61 | Exponential K sampling | K ~ Exp(mean=6)+4, clamped [4,48]. Heavy tail for rare deep K | H46 | **REVERTED** (+0.020, -13% steps from higher avg K) | 1.8437 | +0.0006 |
| 62 | Disable FSQ quantization | Keep low-rank MoS projection but remove FSQ level discretization | H53 | **PROMOTED ★** (val_bpb -0.005, zero overhead) | 1.8183 | +0.0008 |
| 63 | Low-dim expert computation | Each expert: D→r, compute at r, r→D, mix in D-space | H43 | Queued | — | — |
| 64 | Full-rank expert internals | Remove low-rank factorization inside experts (full rank at dim r) | H44 | Queued (after 63) | — | — |
| 65 | Scale up experts (16-32) | More experts at same/reduced rank for routing diversity | H47 | Queued | — | — |
| 66 | Parcae negative diagonal | Replace β with per-dim learned A=Diag(-exp(a)), guaranteed ρ<1 | H48 | Queued | — | — |
| 67 | Per-iteration LoRA | Rank-4 LoRA per DEQ iter (98K params). Each iter specializes | H49 | **REVERTED** (not principled, doesn't generalize to K>16) | 1.8149 | +0.001 |
| 68 | DeltaDEQ dim skipping | Track per-dim convergence, skip converged dims in later iters | H50 | Queued | — | — |
| 69 | Reduce TBPTT 4→1 | Phantom gradient: 1-step backward, 43% faster, 76% more steps | H51 | **REVERTED** (+0.021 post-quant, fast only +0.013) | 1.8396 | -0.002 (K128 best!) |
| 69b | Reduce TBPTT 4→2 | 2-step backward: 29% faster, 40% more steps (1049 vs 747) | H51 | **KEPT** (val_bpb -0.001, 30% throughput gain) | 1.8169 | +0.002 |
| 70 | L2→softmax routing | Replace L2+tanh logits with linear dot-product (standard MoE) | H54 | **PROMOTED ★** (val_bpb -0.024, expert_iter_range 20× higher) | 1.7934 | +0.001 |
| 70 | model_dim 768→1024 | Scale D with low-dim experts (cheap: only down/up grow) | H43 | Queued (after 64) | — | — |
| 71 | Reduce weight_decay 1.08→0.3 | WD=1.08 was biggest expressiveness killer | H54 | **PROMOTED ★** (val_bpb -0.263!!! Largest single improvement ever) | 1.5302 | +0.007 |
| 71b | Reduce weight_decay 0.30→0.10 | Further WD reduction — better fast but worse post-quant | H54 | **REVERTED** (+0.051 post-quant, quant gap 0.047 vs 0.029. WD=0.30 optimal) | 1.5810 | +0.014 |
| 71c | ~~Reduce weight_decay 0.10→0.01~~ | ~~Cancelled: WD=0.10 already regresses post-quant~~ | H54 | CANCELLED | — | — |
| 71d | Drop β=0.7 from jitter | {0.3,0.5,0.7}→{0.3,0.5} | H58 | **REVERTED** (+0.044, less jitter diversity hurt more than recon fix helped) | 1.5744 | +0.017 |
| 71e | Re-enable FSQ, no bounding | FSQ with round+STE but NO tanh bounding (tanh causes saturation) | H59 | Queued (after 71d) | — | — |
| 71f | FSQ with clamp(-1,1) | If no-bound fails, test hard clamp instead of tanh (no saturation) | H59 | Queued (after 71e) | — | — |
| 72 | Remove post-mix RMSNorm | Replace attn/mlp_post_mix_norm with learned scalar scale | H55 | **REVERTED** (+0.024 val_bpb, K128 Δ=0.001 tightest ever but val regressed) | 1.5495 | +0.001 |
| 73 | Relax grad_clip 0.3→1.0 | Aggressive clip slows learning. Lyapunov provides soft contraction | H56 | **PROMOTED ★** (val_bpb -0.003, every K improved, zero-cost change) | 1.5225 | +0.009 |
| 74 | Raise Lyapunov γ 0.9→0.95 | Allow ρ(J) closer to 1 for more expressive state changes | H57 | **REVERTED** (wash: +0.0005, γ=0.9→0.95 has no measurable effect. Penalty too small at λ=0.01) | 1.5230 | +0.010 |
| 71g | Learnable RMSNorm everywhere | Q/K norms + embed + MoS + bigram (removed soft_embed_norm: DDP unused param) | project constraint | **PROMOTED ★** (val_bpb -0.005, 10% faster, K128 Δ=0.009) | 1.5254 | +0.009 |
| 74b | Lyapunov γ 0.9→0.97 | Push warmup advantage further (γ=0.95 was -0.012@200) | H57 | **PROMOTED ★** (val_bpb -0.008, every K improved -0.006 to -0.008, K8 breaks 1.50) | 1.5150 | +0.008 |
| 74c | WD 0.30→0.01 | Test floor — quant gap may shrink with learnable norms | H54 | Queued | — | — |
| 74d | Remove β=0.7 from jitter | {0.3,0.5,0.7}→{0.3,0.5} for reversibility (3.3× recon amp) | H58 | Queued | — | — |
| 74e | Restore squared gate leaky_relu(0.5)² | Phase 6 remnant: original activation was more expressive | L9 | Queued | — | — |
| 74f | Independent shared gates (attn vs mlp) | Fix 1-dim shared gate → 2-dim for independent control | arch | Queued | — | — |
| 74g | Remove x0 residual: T_θ = Δ(z,x0) | More expressive FP equation (x0 still enters via state_norm) | arch | Queued | — | — |
| 74h | Remove attn_post_mix_norm only | Bisect iter 72 | H55 | **REVERTED** (+0.034, attn norm IS load-bearing) | 1.5488 | — |
| 74i | Remove mlp_post_mix_norm only | Bisect iter 72 | H55 | **REVERTED** (+0.029, mlp norm ALSO load-bearing. Both essential) | 1.5436 | — |
| 74j | Remove state_norm | Pre-expert RMSNorm on z+x0 | arch | **REVERTED** (+0.199@400, catastrophic. state_norm IS load-bearing) | 1.8878@400 | — |
| 74k | Remove bigram proj_norm | Bigram pre-projection RMSNorm | arch | **REVERTED** (+0.174@400, bigram norm also load-bearing. ALL norms essential) | 1.8636@400 | — |
| 74l | Remove shared expert gate (always=1) | DeepSeek-V3 style unconditional shared expert | arch | **REVERTED** (+0.020, shared gate provides useful per-token modulation) | 1.5346 | — |
| 75 | ELM identity init | Expert weights init near identity | ICLR 2026 | Queued | — | — |
| 77 | ~~Residual injection (x0-z)~~ | ~~SKIP: iter 18 tested lerp→additive = wash. Math shows degenerate (z+(x0-z)=x0)~~ | H25 | SKIPPED | — | — |
| 78 | ~~FSQ weight QAT~~ | ~~SKIP: STE noise harmful (H15 REFUTED), quant gap only 0.029~~ | H28 | SKIPPED | — | — |
| 79 | ~~Per-iter injection~~ | ~~SKIP: violates RevDEQ weight sharing (f must be identical across iters)~~ | H24 | SKIPPED | — | — |
| 80 | Refinement soft-embed injection during DEQ | Inject x0_refined as second signal alongside x0 | H27 | Queued | — | — |
| 76 | Self-refinement (num_refinements=1) | Enable refinement: predict→soft_embed→re-solve (moved to end) | H16 | Queued | — | — |

**Throughput baseline (T-opt 12-22 complete):** step_avg=8,494ms (-16.3% from iter 47 baseline). block.forward=20ms compiled (hardware-limited). 86% compute-bound, 14% DDP overhead.

**Promotion rule — carry-forward on no-significant-degradation (user direction 2026-04-16):**

The Phase 6 queue is building a *certified* contraction arch.  Each individual
Lipschitz constraint may slightly reduce capacity (small val_bpb hit) — but
the CUMULATIVE architectural direction is the goal.  We therefore do NOT
revert at each step on marginal regression; instead we carry forward unless
the change is broken.

- **Accept (carry to next iter)** if the change does NOT *significantly*
  degrade gate metrics:
  - val_bpb regression ≤ **0.03** (small arch-reshape cost, noise-ish).
  - K=128 Δ ≤ 0.5 (FP still converges at deep K).
  - No catastrophic failure on other gates (no NaN, no routing collapse,
    mos_ortho ≤ 0.9, expert_min_share ≥ 0.005, etc.).
  → `update_results.sh --promote`, update H33 audit row for the newly-
     certified component, launch next iter in the queue.

- **Fix-and-retry (stay on current iter)** if the change IS significant:
  - val_bpb regression > 0.03 OR K=128 Δ > 0.5 OR any gate catastrophes.
  → identify the **simplest, most principled fix** for the specific
     failure mode, rerun the SAME iter with the fix.  Iterate (30.1 →
     30.2 → ...) until the change is good enough to carry forward.
  → Do NOT move to the next iter; do NOT abandon the constraint.
     The architectural alignment direction is committed.

- **Never revert past iter 30 baseline.**  The structural contraction shell
  is the established foundation.  All subsequent Lipschitz constraints are
  additive refinements — regressions are fixes-to-apply, not signals-to-abort.

- **Principled-fix catalog per component (starting points; use the simplest
  that works):**

  | Iter | Failure mode | Simplest principled fix |
  |---|---|---|
  | 31 spectral-U | val_bpb ↑↑ | raise bound `‖U‖_2 ≤ c` with `c∈(1,2]`, ramp down over training; or warm up spectral norm from 2.0 to 1.0 over first N steps |
  | 32 Π_R-state | val_bpb ↑↑ | tune radius R upward (e.g. `2·√d`→`4·√d`); apply Π_R only at Block boundary first, not per-component |
  | 33 spectral-experts | val_bpb ↑↑ | relax `‖W‖_2 ≤ c` with c>1 globally; or exclude the LOWEST-rank expert weight (e.g., only expert_out, expert_down constrained) |
  | 34A L2-router | routing collapse | tune γ upward to sharpen distances; widen prototype init range; try `ρ=identity` first before `tanh` |
  | 34B SIPS | same | tune γ, bound φ and ψ ranges, normalize q to unit-sphere before cosine |
  | 35 single-router | capacity loss | raise expert pool size E=E_attn+E_mlp+extra |
  | 36 L2-attention | perf drop | tune γ for attention sharpness; keep causal mask; try hybrid L2 + softmax convex combo |
  | 37 lipschitz-MLP | perf drop | relax ‖W^(1)‖·‖W^(2)‖ ≤ c with c>1; try GroupSort instead of LeakyReLU; widen hidden dim to recover capacity |

- **Final-state check (after 35 lands):** run K-sweep to K=256+ and verify
  monotone convergence to the Banach fixed point.  If achieved, the
  certified contraction arch is in place as the new permanent baseline.

### Phase 6 post-cert ablation (IMMEDIATELY after iter 35 promotes):
- **39-minimal-principled-norms** (revised user direction 2026-04-16):
  Clean separation: **inside T_x(z) only non-expansive Π_R clamps**;
  **outside the DEQ solve keep norms for optimization**.

  **INSIDE Block.forward (the DEQ fixed-point map T_x(z)):**
  Only TWO Π_R projections, no RMSNorm/LN:

  1. **ONE shared pre-projection:**
     `u = Π_R(z + b(x0))`
     Feed same `u` to router AND all experts (collapse separate
     `attn_norm`/`mlp_norm` into one `Π_R`).

  2. **ONE post-projection on the combined update:**
     `Δ = Π_R(0.5 * (Δ_attn(u) + Δ_mlp(u)))`
     Then: `T_x(z) = (1-τ) b(x0) + τ Δ`

  **REMOVE all other norms inside Block.forward:**
  - `attn_norm = BallProjection` → replaced by shared u = Π_R(...)
  - `mlp_norm = BallProjection` → replaced by shared u
  - `attn_post_mix_norm = RMSNorm` → REMOVED (replaced by post Π_R on Δ)
  - `mlp_post_mix_norm = RMSNorm` → REMOVED (replaced by post Π_R on Δ)
  - `attn_sdpa_post_norm = RMSNorm` → REMOVED
  - `hidden_post_norm = RMSNorm` → REMOVED
  - `_rms_norm(q_rope/q_nope)` → REMOVED (Q,K bounded by Π_R input)
  - `_rms_norm(k_rope/k_nope)` → REMOVED
  - `_rms_norm(x)` in MLP.mix_experts → use pre_normed=True from u
  - `_rms_norm(x)` in Router.forward → use pre_normed=True from u

  **Note:** removing Q/K norms changes the effective attention temperature.
  May need to recalibrate `l2_attn_gamma` (currently 0.051 = 1/(2√d_head)
  calibrated for RMS-normed Q,K).

  **OUTSIDE the DEQ solve (keep for optimization, not contraction):**
  - `GPT._encode: _rms_norm(x)` — keep (embedding normalization)
  - `GPT.final_norm = RMSNorm(model_dim)` — keep (pre-norm for MoS head)
  - `GPT._get_soft_embedding: _rms_norm(soft_embed)` — keep
  - `BigramHash: _rms_norm(h)` — keep
  - MoSHead norms — keep (output head, not inside T_x)

  **Why this is simpler + stronger than learnable-pre-norm approach:**
  - Π_R is 1-Lip by construction (no analysis needed)
  - RMSNorm inside T_x is NOT globally 1-Lip (complicates contraction)
  - TWO Π_R projections vs 10+ learnable RMSNorm = much simpler
  - τ < 1 + Π_R post = strict contraction "by construction"
  - Without ANY projection, bf16 saturation/blow-ups aren't prevented
    unless you add strict σ_max everywhere (more complex than Π_R)

  **Contraction proof sketch:**
  - Pre: ‖u‖ ≤ R (Π_R is 1-Lip)
  - Experts: 1-Lip (spectral norm ≤ 1, 1-Lip activation)
  - Dense mixture: convex combination of bounded expert outputs → ≤ R
  - 0.5 scale + Π_R post: ‖Δ‖ ≤ R (safety net, rarely clips)
  - T_x = (1-τ)b + τΔ: Lip_z(T_x) = τ·Lip_z(Δ) ≤ τ < 1 ✓

  Update opg_doc.tex §4.1/§4.4 to match.

### Phase 7.5 — deeper training + self-refinement (after Phase 7 throughput):
- **30d-deeper-K**: add `deq_k_jitter_set=(4,8,16,24)` on certified arch — tests
  compounding generalization (cf. iter 28c's K=128 Δ=0.015).
- **30e-TBPTT-deep**: `deq_bptt_k=8` + K∈{4,8,16,32}, compute-neutral vs baseline,
  2× training depth.
- **38-self-refinement** (orthogonal to the inner DEQ, documented in
  opg_doc.tex §app:refinement): enable `num_refinements=1` (already plumbed
  in `train_gpt.py`) and ramp `_refine_mix_alpha` up after 85% of wallclock.

### Phase 7.6 — contraction improvement + arch simplification + expressiveness:
- Improve FP convergence quality (K=128 Δ → target < 0.005)
- Simplify model architecture (fewer modules, fewer params, same or better val_bpb)
- Improve expressiveness within 1-Lip constraints (GroupSort, wider experts, etc.)
- **Pre-requisite**: code must be aligned with opg_doc.tex before starting
  refinement benefit with solver-quality noise.

### Phase 7.8 — Lyapunov Stability Theory for FP convergence (relax Banach contraction)

**Motivation:** Banach contraction (Lip(T_x) < 1 everywhere) is *sufficient* for
FP convergence but extremely restrictive — it forces τ_max ≈ 0.003-0.04, orthogonal
expert parameterization, bounded-domain projections, and 1-Lip activations.  These
constraints provably guarantee convergence but sacrifice model expressiveness
(iter 39b: perfect FP with val_bpb +0.22 regression).

**Lyapunov Stability Theory** provides a *weaker but still sufficient* guarantee:
instead of requiring ‖J_T‖ < 1 globally, we need only a **Lyapunov function**
V(z) ≥ 0 with V(z*) = 0 such that V(T(z)) < V(z) for all z ≠ z*.  This allows:
- Local expansion in some directions (σ_max > 1) as long as V decreases overall
- State-dependent contraction rates (tight where needed, loose where safe)
- More expressive architectures that still converge to a unique FP

**Natural Lyapunov candidates for DEQ:**
- V(z) = ‖z - T(z)‖² (residual norm — already tracked as `deq_residual`)
- V(z) = ‖z - z*‖² (distance to FP — requires FP estimate, e.g. from previous step)
- V(z) = z^T P z for learned PSD matrix P (quadratic Lyapunov)
- Learned V_φ(z) via auxiliary ICNN (input-convex neural network)

**Verification:** At each DEQ iteration k, check V(z_{k+1}) < V(z_k).  If violated,
fall back to damped update z_{k+1} = z_k + α(T(z_k) - z_k) with α chosen to
guarantee V-decrease (line search or conservative α).

**10 highest-ROI changes (priority order):**

| # | Change | What it relaxes | Expected ROI | Risk |
|---|---|---|---|---|
| L1 | **Remove orthogonal constraint → spectral-norm cap σ_max ≤ c** | All-σ=1 isometry → σ_max ≤ c with c∈[1, 2]. Experts can amplify important directions while still bounded. Monitor V(z) = ‖z-T(z)‖² decrease per DEQ iter. | HIGH — orthogonal kills ~50% of rank capacity (singular value freedom). c=1.5 recovers it while V-decrease holds empirically. | If V increases, need adaptive α damping. |
| L2 | **Raise τ_max to 1.0 with Lyapunov safeguard** | τ < 1 (contraction) → τ ≤ 1.0 with runtime V-decrease check. When τ=1, T_x = G_θ (full expressiveness, no shell attenuation). Fall back to τ < 1 only if V fails to decrease. | HIGH — τ≈0.9 already works (iter 35); τ=1.0 removes the (1-τ)b(x_0) anchor entirely, testing whether G_θ alone has a stable FP. | Loss of guaranteed convergence; need runtime fallback. |
| L3 | **Restore learnable RMSNorm inside T_x** | Π_R-only (1-Lip) → RMSNorm (unbounded Lip near ‖x‖→0). RMSNorm's learnable scale provides capacity (iter 37b: +0.069 val_bpb from removing it). Lyapunov doesn't need bounded Lip; it needs V-decrease. | HIGH — directly recovers the capacity lost in iter 37b/39. | Near-zero-norm states could cause V spikes; add ‖z‖ floor. |
| L4 | **Remove BallProjection Π_R clamps** | Bounded domain → unbounded. Π_R clips large states; removing it lets the model use the full representation space. Lyapunov V(z) can still decrease on unbounded domains. | MEDIUM — Π_R rarely activates (R=2√d is generous); removing it simplifies code and removes 1-Lip overhead. | Unbounded states → need V to handle large ‖z‖. Use V = ‖z-T(z)‖² which is norm-agnostic. |
| L5 | **Allow state-dependent τ(z) instead of exogenous τ(x_0)** | Exogenous τ(b(x_0)) → τ(z). Under Banach, τ(z) leaks gradients (unbudgeted ∇_z τ term). Under Lyapunov, τ(z) is fine as long as V decreases — the model can adapt contraction strength to the current state. | MEDIUM — lets τ shrink in unstable regions and grow in stable ones. Gradient ∇_z τ now contributes usefully to learning. | If τ(z) oscillates, V may not decrease monotonically. Clamp τ∈[0.1, 1.0]. |
| L6 | **Remove the 0.5 scale in G_θ** | G = 0.5·(Δ_attn + Δ_mlp) → G = Δ_attn + Δ_mlp (pooled router weights already sum to 1). The 0.5 was a contraction aid; Lyapunov doesn't need it. Doubles effective output magnitude. | MEDIUM — straightforward capacity gain. Already tested in iter 39 (worked for FP, hurt val_bpb only because τ was too small). | May need to adjust initial τ or learning rate. |
| L7 | **Lyapunov-adaptive β (DEQ solver momentum)** | Fixed β=0.20 → β_k adapted per DEQ iteration based on V-decrease rate. If V(z_k) decreases fast, increase β (aggressive); if slow, decrease β (conservative). Similar to line-search in optimization. | MEDIUM — current β=0.20 is conservative. Adaptive β could halve the iterations needed for convergence, improving throughput. | Adds per-iteration overhead (V evaluation). Use V = ‖residual‖² which is already computed. |
| L8 | **Spectral monitoring instead of spectral capping** | σ_max ≤ 1 (enforced) → σ_max tracked + logged (not enforced). Use σ_max as a diagnostic; let Lyapunov V-decrease be the convergence guarantee. Training naturally keeps σ moderate if the loss landscape favors convergence. | MEDIUM — removes the NS overhead (20 matmuls × 6 weights per Block.forward) while keeping the safety net of V monitoring. | Uncontrolled σ growth → solver divergence. Need hard fallback (clamp σ if V fails). |
| L9 | **Restore gated-product MLP activation** | leaky_relu(0.5) × fc (1-Lip) → leaky_relu(0.5)² × fc (not 1-Lip but more expressive). The squared gate was removed in iter 37 for Lip certification. Under Lyapunov, the squared gate is fine as long as V decreases. | LOW-MEDIUM — iter 37 showed only -0.006 val_bpb from removing it. May not be worth the complexity. | Minor: squared gate can saturate; monitor expert entropy. |
| L10 | **Train a lightweight Lyapunov certificate V_φ(z)** | Empirical V = ‖residual‖² → learned V_φ(z) via small ICNN (input-convex NN). Train V_φ alongside the model with loss L_V = max(0, V_φ(T(z)) - V_φ(z) + ε). Provides a tighter stability certificate than the residual heuristic. | LOW — theoretical elegance but significant implementation complexity. ICNN adds params + compute. Only justified if L1-L8 leave convergence unreliable. | ICNN training instability; V_φ might not generalize across the state space. Defer until L1-L8 are validated. |

**Execution order:**
1. L1 + L6 together (relax experts + remove 0.5 scale) — biggest capacity gain with
   minimal risk. Add V = ‖z - T(z)‖² monitoring per DEQ iter (already tracked as
   `deq_residual`; just verify it decreases monotonically).
2. L3 (restore RMSNorm) — directly recovers iter 37b capacity loss.
3. L2 + L5 (raise τ + state-dependent τ) — unlocks the full contraction shell.
4. L7 + L8 (adaptive β + monitoring-only σ) — throughput gains.
5. L4 + L9 (remove Π_R + restore squared gate) — diminishing returns.
6. L10 (learned Lyapunov) — only if empirical V is insufficient.

**Success criteria:**
- val_bpb ≤ iter 35 baseline (1.9197) — must RECOVER lost expressiveness
- K=128 residual monotonically decreasing (Lyapunov V-decrease)
- No solver divergence over 10 consecutive training runs
- Throughput ≥ iter 35 baseline (no NS overhead)

**Theoretical reference:**
- Lyapunov stability for discrete dynamical systems: Khalil, "Nonlinear Systems" §4.4
- Lyapunov-based DEQ analysis: Winston & Kolter, "Monotone Operator Equilibrium Networks" (ICML 2020)
- Input-convex neural networks: Amos, Xu & Kolter (ICML 2017)

### Phase 7+ (deferred): throughput unroll+compile, scaling-law grid, FSQ/rank sweeps

Removed from the active queue to keep focus on Phase 6 doc-alignment.  Will
be restored (or redesigned) once the certified contraction architecture is
established and validated. See prior git history for the full pre-cleanup
queue if needed.

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

#### HARD CONSTRAINTS (doc-alignment + process, added 2026-04-16)

These four rules override ordinary promotion policy.  Violating any of them is
a process bug, not a design choice.

1. **Doc alignment is a HARD CONSTRAINT.**  Every module specified in
   `opg_doc.tex` §3–6 MUST land in `train_gpt.py` (exogenous injection, τ-shell,
   Π_R on state and post-mix norms, per-expert σ_max≤1, 1-Lip activations,
   L2-distance attention, **single pooled router over E=E_attn+E_mlp**, etc.).
   Capacity/throughput regressions within the carry-forward policy
   (`Δval_bpb ≤ 0.03`, K=128 Δ ≤ 0.5) are acceptable tax for certification.
   No doc-specified structure may be marked "optional".
2. **Update this document after every iteration.**  A commit that lands an
   iter without refreshing (a) the Phase 6 queue row, (b) the H33 audit
   table, and (c) any hypothesis statuses affected is incomplete — the doc
   is the autoresearch system's memory.
3. **On launch, set a single 20-min `ScheduleWakeup`.**  When training is
   kicked off, always create ONE wakeup at ~1200s to inspect `run.log` and
   update this document.  Don't chain wakes up front; dynamically pace
   further wakes (270s when cache-warm and close to completion, 1200–1800s
   when genuinely idle).  Never sleep past 300s with no specific signal to
   watch — that burns cache without purpose.
4. **`f_theta` must be deterministic within a DEQ solve.**  RevDEQ's O(1)
   backward reconstructs the forward during backward; any stateful
   parametrization (spectral-norm power iteration, running stats, dropout,
   stochastic noise) that writes to buffers inside `forward()` breaks
   reconstruction and silently corrupts gradients.  Such updates MUST live
   in explicit pre/post-solve hooks (e.g., `update_uv_()` called once per
   optimizer step before the DEQ solve), never inside the `forward()`
   invoked from `RevDEQFunction`.
5. **No stale tensors across microbatches.**  Any tensor stored on a module
   attribute (e.g., `_balance_loss`) that has a `grad_fn` MUST be
   recomputed in each microbatch — NOT reused from a previous one.
   After `.backward()` frees microbatch N's graph, microbatch N+1
   accessing the stale tensor crashes with "backward through graph a
   second time."  The principled fix for regularization losses is:
   compute them ONCE per step from the final forward pass's output
   (post-solve), not inside the DEQ loop (2K× per step).
7. **torch.compile: no inplace mutation of freshly-allocated tensors.**
   `new_zeros` + index assignment (`t[..., :D] = val`) inside a compiled
   graph corrupts AOT autograd's alias tracker — produces garbage shape
   dimensions.  Always use functional ops: `torch.cat` + `F.pad` instead.
   This applies to any pattern where a tensor is allocated then partially
   filled inside the compiled forward.
6. **torch.compile: one compiled output, one backward consumer.**  When a
   compiled module produces a single tensor split into two backward
   paths, AOT autograd may fail.  Either (a) return pre-split outputs
   INSIDE the compiled graph, or (b) exclude the multi-consumer op via
   `@dynamo_disable`.  Applies whenever a compiled forward's output
   is sliced into views feeding separate loss branches.

#### Defaults and gates

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
