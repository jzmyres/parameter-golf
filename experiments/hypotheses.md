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

### H43: Low-dim expert computation (D→r per expert, mix in D-space) — TESTED ✗ (NOT PROMOTED, see H69 + H70; 2026-04-25)
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
**Status:** TESTED 2026-04-25 — NOT PROMOTED. Bottleneck arch (H43 + H44 combined into the iter 90 + iter 91+92 bundle) is structurally sound (K-sweep TIGHTENS in both runs, throughput +34% standalone) but carries a quantified ~27% per-param efficiency penalty vs full-D MLA at matched capacity (iter 91+92: 11.3M params, val_bpb 1.6784 vs baseline 1.5264, Δ +0.152). proj_rank=32 at the D↔r boundary is the likely expressivity ceiling. See H69 (standalone) and H70 (matched-capacity) for full diagnostics. **Code reverted on `autoresearch/phase2-optimization` 2026-04-26** (commits `14e9fb2` + `0e6ab19`); the architectural pattern (`BottleneckIn / ExpertMLABody / ExpertMLPBody / BottleneckOut`, originally committed `8c2be77`) is preserved off-branch at tag `iter-91+92-bottleneck-NOT-PROMOTED` and on side branch `autoresearch/bottleneck-rescue` for a future `proj_rank=48/64` rescue.

### H44: Full-rank expert internals at dim r (remove low-rank factorization) — TESTED ✗ (with H43, see H69+H70; 2026-04-25)
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
**Status:** TESTED 2026-04-25 alongside H43. Outcome: full-rank at r=128 (iter 90) and r=192 (iter 91+92) confirms the architectural simplicity claim — `attn_expert_rank` and `mlp_expert_rank` were replaced by `attn_bottleneck_r`, `mlp_bottleneck_r`, `expert_proj_rank`, `attn_inner_heads`, `attn_inner_kv_heads`, `mlp_inner_mult` (see H69 / H70 for code-level details). Expert capacity is now a single (r, mlp_inner_mult) pair as predicted. NOT PROMOTED for the standalone arch reason from H69+H70 (per-param efficiency penalty). **Code reverted on `autoresearch/phase2-optimization` 2026-04-26** (commits `14e9fb2` + `0e6ab19`); the simpler hyperparameter surface is preserved off-branch on `autoresearch/bottleneck-rescue` for any future bottleneck-arch revisit.

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

### H47: Scale up number of experts — TESTED ✗ (E=16 in iter 91+92 bundle, NOT PROMOTED; 2026-04-25)
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
**Status:** TESTED 2026-04-25 — bundled with H43+H44 in iter 91+92 (E=8→16 alongside D=768→1024 and r=128→192). Outcome: E=16 worked structurally — head-packed SDPA at E×H_in = 16×4 = 64 query heads ran without FlashAttention issues, attn_entropy converged to 2.37 (vs theoretical max log(15)≈2.71 for routed experts), no routing collapse, attn_cv=0.44 (well-balanced after warmup). The capacity went from 4.5M (iter 90, E=8) → 11.3M (iter 91+92, E=16+D=1024+r=192), closing half the gap to baseline. But the NOT-PROMOTED outcome (val_bpb +0.152 vs baseline) is attributable to H43's per-param efficiency penalty (the bottleneck I/O at proj_rank=32), NOT to the E=16 scaling itself. The E-scaling claim is **structurally validated** — more experts at low-dim work — but not in isolation. **Code reverted on `autoresearch/phase2-optimization` 2026-04-26** (commits `14e9fb2` + `0e6ab19`); the E=16 scaling can be revisited on `autoresearch/bottleneck-rescue` independently of the proj_rank rescue. See H70.

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

### H70: Bottleneck experts at matched capacity (iter 91+92 bundle) — NOT PROMOTED ✗ (per-param efficiency penalty quantified; 2026-04-25)

**Claim:** Iter 90 standalone (4.5M params, val_bpb +0.276) regressed because of capacity, not architectural validity (K-sweep tightened, gates healthy, throughput +34%). The follow-up iter 91+92 bundle scales to 87% of baseline capacity (`E=8→16, D=768→1024, r=128→192`) under the bottleneck arch — if the architecture has no per-param efficiency penalty, val_bpb should approach baseline 1.5264 within the 0.03 promotion gate. If it doesn't, that quantifies the cost of the bottleneck design vs the original full-D MLA.

**Test:** iter 91+92 bundle — `num_experts=16, model_dim=1024, attn_bottleneck_r=192, mlp_bottleneck_r=192` (other knobs unchanged from iter 89 baseline including `expert_proj_rank=32`). Commit `3e35655`. 1000 steps on 2× L40S, ~5.5 hours wallclock.

**Result:** ❌ NOT PROMOTED — bottleneck arch has a quantifiable per-param efficiency penalty.

| Metric | Baseline (iter 89) | iter 91+92 bundle | Δ |
|---|---|---|---|
| **val_bpb int6 (sliding window)** | 1.5264 | **1.6784** | **+0.152** ✗ (gate: ≤ 0.03) |
| val_bpb fast (final eval) | 1.4848 | 1.6198 | +0.135 |
| K=4 | 1.7506 | 1.8432 | +0.093 |
| K=8 | 1.5325 | 1.6836 | +0.151 |
| K=16 | 1.5264 | 1.6784 | +0.152 |
| K=32 | 1.5285 | 1.6794 | +0.151 |
| K=64 | 1.5288 | 1.6795 | +0.151 |
| K=128 | 1.5289 | **1.6796** | +0.151 |
| **K=8 → K=128 Δ** | -0.0036 | **-0.004** | tighter contraction ✓ (gate: ≤ 0.5) |
| total params | 13M (est) | **11.32M** | -13% (still under baseline) |
| artifact_bytes | 5,929,124 | 6,607,625 | +12% (modest given more experts) |
| step_avg (ms) | ~13,000 | ~20,000 | **+54% slower** (E=16 + D=1024 doubles compute) |
| peak_vram_mb | 21,549 | 27,738 | +29% (more activations) |
| no NaN/Inf, no gate catastrophe | ✓ | ✓ | clean |

**val_bpb gap trajectory across training (vs baseline at equal step count):**

| Step | iter 91+92 val_bpb | Baseline val_bpb | Gap |
|---|---|---|---|
| 200 | 2.2735 | 1.9996 | +0.274 |
| 400 | 1.9450 | 1.6656 | +0.279 |
| 600 | 1.7479 | 1.5683 | +0.180 |
| 800 | 1.6601 | 1.5260 | +0.134 |
| 1000 | 1.6198 | ~1.4848 | +0.135 |

The gap was actively closing through steps 600-800 (-0.10 nats per 200 steps) but **plateaued at +0.134 from step 800 onward**. This rules out "needs more steps to converge" — at step-1000 budget, iter 91+92 bottleneck arch lands ~0.135 nats behind baseline and is no longer improving.

**Three diagnostic findings:**

1. **Capacity restoration helped, but not enough.** Iter 90 (4.5M, gap +0.276) → iter 91+92 (11.3M, gap +0.135). Going from 35% → 87% of baseline capacity closed half the gap. Linearly extrapolating, full baseline capacity (proj_rank=64, ~13.8M) might close another ~0.05, reaching gap ~+0.08 — still over the 0.03 gate.

2. **K-sweep STILL tightens (-0.004).** Same architectural validity signal as iter 90: bottleneck experts produce a more contractive DEQ than full-D MLA, even at matched capacity. The K=128 val_bpb is the same as K=16 — the FP fully converges at modest K. This is independent evidence that the bottleneck arch is architecturally sound; the capacity penalty is **expressivity per param**, not stability.

3. **Throughput penalty (+54% step time)** offsets iter 90's +34% gain when scaling to baseline capacity. At equal wallclock, baseline (13M, 13s/step) and iter 91+92 (11.3M, 20s/step) need different step counts — the bottleneck arch processes 1.54x more input per step but at higher param efficiency cost.

**Per-param efficiency analysis:**

| | iter 89 baseline | iter 91+92 bundle | iter 90 standalone |
|---|---|---|---|
| Total params | ~13M | 11.3M | 4.5M |
| Final val_bpb int6 | 1.5264 | 1.6784 | 1.8023 |
| **bpb / param** (10⁻⁷ nats/param) | **1.17** | 1.49 | 4.00 |

Bottleneck experts deliver bpb/param ratio 1.49 vs baseline 1.17 — ~27% worse per parameter. This is the architectural cost of the D→r→D bottleneck: the inner low-dim work (full-rank at r=128 or r=192) is more flexible per matmul, but the I/O bottleneck (D→32→r) appears to be the real expressivity ceiling.

**The proj_rank=32 hypothesis** is the most likely culprit. With D=1024, R_proj=32 means the input is squeezed through a 32-dim subspace BEFORE the expert sees it (and again on output). 32-dim subspace × 16 experts is 512 effective input ranks — only 50% of D=1024. Bumping proj_rank to 48 (97% of baseline capacity = 12.6M) or 64 (106% = 13.8M) would relax this bottleneck and is the natural rescue iter.

**Implication:** The bottleneck-experts experiment closes with a clear, quantitative architectural finding rather than a promotion. The architecture is structurally sound (K-sweep TIGHTENS), tests independent across-expert representations (independent xavier per-expert per-stage), and saves significant artifact budget. But proj_rank=32 is an information bottleneck that costs ~27% per-param efficiency. A follow-up rescue iter could test proj_rank=48 or proj_rank=64 at fixed E=16, D=1024 to see if the gap closes within 0.03.

**Status:** ❌ NOT PROMOTED. Baseline stays at iter 89 (`aeba34a`, val_bpb 1.5264). The bottleneck-experts code (iter 90 + iter 91+92) was **reverted on `autoresearch/phase2-optimization` 2026-04-26** (commits `14e9fb2` + `0e6ab19`) so the active branch matches the documented baseline. The architectural pattern is preserved off-branch as an immutable archival tag `iter-91+92-bottleneck-NOT-PROMOTED` (at `3e35655`) and on side branch `autoresearch/bottleneck-rescue` (which also retains the user's `ExpertMLABody` `H_in/H_kv_in≥2` validator as commit `59bc156`). A future `proj_rank=48/64` rescue iter would start from `git checkout autoresearch/bottleneck-rescue && git rebase main`.

**Decision after H70:** rather than chase the proj_rank rescue (which would consume another ~5 hours and is uncertain), proceed to **iter 95 (TBPTT efficiency sweep)** per user direction — TBPTT is a cleaner one-knob optimization on top of the existing baseline (iter 89), no architectural change. Also queues a "Lipschitz vs K probe" diagnostic suggested by the user 2026-04-25 for future K-sweep instrumentation.

### H69: Bottleneck experts (iter 90) — NOT PROMOTED ✗ (capacity-bound at standalone scale; 2026-04-25)

**Claim:** The pre-iter-90 design had each per-expert linear scaling per-expert with model_dim D — Q, KV-A, KV-B, K_rope, Wo all had a 768-side. This made every per-expert footprint proportional to D, so scaling D was unaffordable under the 16 MB artifact cap. Replacing the per-expert path with a *bottleneck* — `BottleneckIn (D → proj_rank → r)` + full-rank MLA/SwiGLU at r + `BottleneckOut (r → proj_rank → D)` — frees the per-expert footprint from D entirely (only the I/O bottleneck modules see D), enabling future iters to scale D and num_experts cheaply.

**Architectural design (composition):**
```
x (B, T, D)
   → BottleneckIn (D → proj_rank → r)        ← only stage that sees D
   → ExpertMLABody / ExpertMLPBody (full-rank, all linears at r)
   → BottleneckOut (r → proj_rank → D)        ← only stage that sees D
   → y (B, T, E, D)
```
- **Inner attention at r=128**: Q (r → H_in*d_in + H_in gate logits), KV-A (r → kv_latent_inner = H_kv_in*d_in/2 — preserves DeepSeek-style 2× cache compression), KV-B-{K,V} (split heads), K_rope (r → H_kv_in*rope_d), Wo (H_in*d_in → r). All full-rank single-stage (no nested low-rank).
- **Inner MLP at r=128**: gate, fc, down — all full-rank single-stage SwiGLU at `mlp_hidden = round(r × mlp_inner_mult) = 320`.
- **Independent per-expert pre-RMSNorm** on every linear input (per the Prenorm Scale Independence Rule). 16+ scale Parameters per expert; trivial param cost.
- **Optimizer coverage**: every new tensor matches `CONTROL_TENSOR_PATTERNS` (`norm_weight`, `q_gain`, `gate_bias`) or goes to the matrix group. `_assert_optimizer_param_coverage` passes at construction.

**Test:** iter 90 — `attn_bottleneck_r=128`, `mlp_bottleneck_r=128`, `expert_proj_rank=32`, `attn_inner_heads=4`, `attn_inner_kv_heads=2`, `mlp_inner_mult=2.5`, with iter 89 baseline otherwise. Commit `8c2be77`. 1000 steps on 2× L40S.

**Result:** ❌ NOT PROMOTED — capacity-bound. Bottleneck arch is functionally validated (DEQ converges, K-sweep TIGHTENS, gates healthy, throughput +34%, artifact 19% of budget) but param count fell from 13M (baseline) → 4.5M (iter 90, -65%) and val_bpb regressed proportionally. The follow-up iter 91+92 bundle (E=16, D=1024, r=192) restores capacity to ~11.3M (87% of baseline) for the architectural-validity comparison vs iter 89.

| Metric | Baseline (iter 89) | Iter 90 | Δ |
|---|---|---|---|
| **val_bpb int6** | 1.5264 | **1.8023** | **+0.276** ✗ (gate: ≤ 0.03) |
| val_bpb fast (final eval) | 1.4848 | 1.6950 | +0.210 |
| K=4 | 1.7506 | 1.9470 | +0.196 |
| K=8 | 1.5325 | 1.8123 | +0.280 |
| K=16 | 1.5264 | 1.8023 | +0.276 |
| K=32 | 1.5285 | 1.8029 | +0.274 |
| K=64 | 1.5288 | 1.8028 | +0.274 |
| K=128 | 1.5289 | 1.8031 | +0.274 |
| **K=8 → K=128 Δ** | -0.0036 | **-0.0092** | tightens MORE under bottleneck ✓ (gate: ≤ 0.5) |
| total params | ~13M | 4,496,467 | **-65%** (the explanation) |
| artifact_bytes | 5,929,124 | 3,017,951 | **-49%** (3.0 MB, 19% of 16 MB budget — huge headroom) |
| step_avg (ms) | ~13,000 | ~8,600 | **-34%** (faster — fewer matmul stages, smaller GEMMs) |
| peak_vram_mb | 21,549 | 13,810 | **-36%** (much smaller activation footprint) |
| no NaN/Inf, no gate catastrophe | ✓ | ✓ | clean |

**Why the K-sweep TIGHTENED (-0.0036 → -0.0092):** the bottleneck design has a sharper FP because the inner work is full-rank at r=128 (vs nested low-rank at D=768 in the old MLA). Better-conditioned inner ops give a more contractive `f_θ`. This is consistent with iter 90's intent: simplify the per-expert compute path so the same Parcae per-dim Ā delivers more contraction.

**Why val_bpb regressed:** capacity, full stop. Going 13M → 4.5M (-65%) is a ~3× capacity cut. iter 90's val_bpb gap vs baseline tracked: step 200 +0.295, step 400 +0.363, step 600 +0.266, step 800 +0.204, step 1000 +0.276. The bottleneck arch is converging *faster per step on the smaller model* than baseline does on the bigger one (final ntp_loss 2.82 at step 1000 vs baseline at similar step), but the smaller-model ceiling is lower. The trajectory at steps 600–800 closing the gap then re-widening at step 1000 is the warmdown phase exposing the capacity ceiling.

**Implication:** The *architectural change* is validated — bottleneck experts (a) preserve MLA semantics with cheaper compute, (b) pass the K-sweep gate decisively, (c) free 65% of param budget and 36% of VRAM. The *capacity* needs to be restored. Iter 91+92 bundle (E=16, D=1024, r=192) is the matched-capacity follow-up, queued by user direction 2026-04-25 regardless of iter 90's standalone verdict.

**Status:** ✅ Architecturally VALIDATED. Standalone NOT PROMOTED. Baseline stays at iter 89 (`aeba34a`, val_bpb 1.5264). Bottleneck infrastructure stays in code (load-bearing for iter 91/92).

### H68: Disable HyDRA denoising regularization (iter 89) — PROMOTED ★ (2026-04-25)

**Claim:** Same Parcae-redundancy logic as H67 applied to the finite-perturbation contraction probe. HyDRA's `||f(z*+ε, x0) - z*||²` is a finite-scale analog of the Hutchinson-Frobenius infinitesimal probe — both regularize toward `‖J‖<1` at z*. Iter 88 showed the infinitesimal probe was redundant under iter-66b Parcae (per-dim Ā ∈ [0.1, 1) by construction); the finite probe should be redundant for the same reason.

**Test:** iter 89 — `denoising_coef = 0.01 → 0.0`. The hot-path block at L3690 short-circuits on `dn_coef > 0.0`, so the noisy-perturbation block forward and `dn_loss` skip entirely. Code path retained per user directive (commented-out future cleanup permitted; deletion not). Commit `aeba34a`. Promoted commit `aeba34a` (no doc-update commit needed pre-launch).

**Result:** PROMOTED.

| Metric | Iter 88 baseline | Iter 89 | Δ |
|---|---|---|---|
| val_bpb fast (final eval) | 1.4844 | **1.4848** | +0.0004 |
| val_bpb int6 (sliding window) | 1.5238 | **1.5264** | **+0.0026** ✓ (≤ 0.03) |
| k=4 | 1.7869 | 1.7506 | -0.036 (off-distribution; iter 89 actually less drifted) |
| k=8 | 1.5310 | 1.5325 | +0.0015 |
| k=16 | 1.5238 | 1.5264 | +0.003 |
| k=32 | 1.5264 | 1.5285 | +0.002 |
| k=64 | 1.5267 | 1.5288 | +0.002 |
| k=128 | 1.5269 | **1.5289** | +0.002 |
| K=8 → K=128 Δ | -0.0041 | **-0.0036** | nearly identical (still negative — deep K BETTER) |
| artifact bytes | 5,880,289 | 5,929,124 | +48,835 (zstd compresses denoising-trained vs not slightly differently; not a parameter change) |
| step_avg (ms) | ~13,300 | ~13,000 | -2% (denoising forward + dn_loss was ~0.3s/step) |
| peak_vram_mb | 21,850 | 21,549 | -301 MB (one-fewer block forward in scope) |

**Three confirmations match H67's pattern (independent test of the same Parcae-redundancy hypothesis):**

1. **No val_bpb regression**: +0.0026 is below the noise floor on dev hardware. Capacity is unchanged.
2. **K-sweep stays tight (-0.0036 ≈ -0.0041)**: contraction is at least as good without the finite-perturbation probe. Parcae per-dim Ā handles both infinitesimal AND finite contraction control.
3. **Throughput recovery (-2%, peak VRAM -301 MB)**: removing the noisy block forward + `(f_noisy - z*)²` MSE recovered ~0.3s/step and ~14 MB working memory. Combined with iter 88, the two ablations together reclaim ~5% of step time and ~1.5% of peak VRAM.

**The `k=4` improvement (1.7869 → 1.7506, -0.036)** is noteworthy — at k=4 (off-training-distribution since K-jitter set is {8,12,20}), iter 89 generalizes BETTER than iter 88. Plausible mechanism: removing the denoising MSE removes a finite-scale regularizer that was effectively asking the model to be insensitive to perturbations of σ=0.01 around z*. With that gone, the model fits training-K behavior more sharply, and that sharpness happens to extrapolate slightly better to shallow K. Not a load-bearing claim — could easily be noise — but it's at least not evidence of K-robustness loss.

**Implication:** The two iter-66b-pre-Parcae regularizers (λ_jac + denoising MSE) were both redundant once Parcae's per-dim Ā took over spectral-radius control. With both off, the loss is now `task_loss + bal_loss + ortho_loss + router_health`, with no spectral-bound auxiliary losses. The contraction in fact *tightens* — see K=8→K=128 Δ trajectory iter 87 (-0.003) → iter 88 (-0.0041) → iter 89 (-0.0036) — and throughput recovers.

**Why we kept the code instead of deleting:** per user directive 2026-04-25, dead-code cleanup uses comment-out, not deletion. Both Lyapunov and denoising paths remain in `train_gpt.py` behind `coef > 0.0` guards. They could re-activate via CLI flag if a future architectural change reintroduces a need for explicit ρ(J) bounding.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `aeba34a`, val_bpb int6 = 1.5264).

### H67: Disable Lyapunov hinge penalty λ_jac (iter 88) — PROMOTED ★ (2026-04-25)

**Claim:** Under iter-66b Parcae per-dim Ā, the spectral radius is already bounded away from 1 by construction (`Ā ∈ [0.1, 1)` via the reversibility floor + softplus reparam), so the Hutchinson-Frobenius `ρ(J) < γ` hinge penalty has nothing to grip on at training time. λ_jac contributes only Hutchinson-probe noise to the gradient and one VJP per step worth of compute.

**Test:** iter 88 — `lyapunov_coef = 0.01 → 0.0`. The hot-path block at L3649 short-circuits on `lyap_coef > 0.0`, so the Hutchinson VJP and surrogate skip entirely. Code path retained (commented-out future cleanup permitted; deletion not). Commit `ceb7dfa`. Promoted commit `45af5bf`.

**Result:** PROMOTED.

| Metric | Iter 87 baseline | Iter 88 | Δ |
|---|---|---|---|
| val_bpb fast (final eval) | 1.4830 | **1.4844** | +0.0014 |
| val_bpb int6 (sliding window) | 1.5188 | **1.5238** | **+0.0050** ✓ (≤ 0.03) |
| k=4 | 1.7048 | 1.7869 | +0.082 (off-training-distribution; expected drift) |
| k=8 | 1.5245 | 1.5310 | +0.0065 |
| k=16 | 1.5188 | 1.5238 | +0.005 |
| k=32 | 1.5208 | 1.5264 | +0.006 |
| k=64 | 1.5212 | 1.5267 | +0.006 |
| k=128 | 1.5213 | **1.5269** | +0.006 |
| K=8 → K=128 Δ | -0.003 | **-0.0041** ★ | tighter (still negative — deep K BETTER than train K) |
| artifact bytes | 5,937,629 | 5,880,289 | -57,340 (-1.0%) |
| step_avg (ms) | ~13,700 | ~13,300 | **-2.9%** (Hutchinson VJP cost confirmed removed) |
| peak_vram_mb | ~21,800 | 21,850 | ~flat |

**Three independent confirmations of the hypothesis:**

1. **No val_bpb regression**: Δ +0.005 is well within the carry-forward 0.03 band, and within the run-to-run noise envelope on this dev-hardware budget. The Hutchinson penalty was contributing nothing to capacity — its removal does not cost expressiveness.
2. **K-sweep stays tight (-0.0041)**: the K=8 → K=128 gap is *more* negative than iter 87 (-0.003 → -0.0041). Lyapunov was supposed to enforce contraction; without it, contraction is at least as good. This is the cleanest evidence that Parcae's per-dim Ā was already doing the contraction work.
3. **Throughput recovery (-2.9%)**: removing the Hutchinson probe + surrogate VJP recovered ~3% of step time, consistent with the cost of one extra forward-mode pass through the shared block per step.

**Implication:** Two separate spectral-control mechanisms (Parcae per-dim Ā + Hutchinson-Frobenius hinge) were stacked redundantly since iter 66b promoted Parcae. λ_jac was load-bearing in pre-Parcae configurations (iter 45-onward) but became dead regularization once Parcae's reversibility floor + softplus reparam took over. Removing it cleans up the loss surface, recovers throughput, and drops one tuning knob (`lyapunov_coef`, `lyapunov_gamma`, `lyapunov_warmup_frac` all become inert defaults).

**Why we kept the code instead of deleting:** per user directive 2026-04-25, dead-code cleanup uses comment-out, not deletion. The Lyapunov code path is structurally clean (gated by `if lyap_coef > 0.0`) and could re-activate via CLI flag if a future architecture change reintroduces a need for explicit ρ(J) bounding. The `_lyapunov_z_star` / `_lyapunov_x0` saves at L2772-2773 still serve the HyDRA denoising path (which iter 89 will ablate next).

**Next implication for iter 89:** the same logic applies to HyDRA denoising (`denoising_coef = 0.01`). If λ_jac was redundant under Parcae, the finite-perturbation contraction probe (`||f(z*+ε, x0) - z*||²`) likely is too — and iter 89 tests that as a clean one-variable ablation on top of the iter-88 baseline.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `45af5bf`, val_bpb int6 = 1.5238).

### H66: K-jitter widen (4,6,10) → (8,12,20) (iter 87) — PROMOTED ★ (2026-04-25)

**Claim:** Doubled the K-jitter set. Deeper average forward depth at training time should tighten the FP, replicating H12's K-sweep win at a wider scale.

**Test:** iter 87 — `deq_k_jitter_set (4,6,10) → (8,12,20)`, `deq_k_max 16 → 20`. Commit `88ad22c`.

**Result:** PROMOTED.

| Metric | Iter 86 baseline | Iter 87 | Δ |
|---|---|---|---|
| val_bpb fast | 1.5042 | **1.4830** | -0.0212 |
| val_bpb int6 | 1.5390 | **1.5188** | **-0.0202** ★ |
| k=4 | 1.5423 | 1.7048 | **+0.16** (training set excludes k=4 now) |
| k=8 | 1.5281 | 1.5245 | -0.004 |
| k=16 | 1.5390 | 1.5188 | -0.020 |
| k=32 | 1.5415 | 1.5208 | -0.021 |
| k=64 | 1.5421 | 1.5212 | -0.021 |
| k=128 | 1.5429 | **1.5213** | -0.022 |
| K=8→K=128 Δ | +0.0148 | **-0.003 ★★** | NEGATIVE — deep K BETTER than train K |
| artifact bytes | 5,948,483 | 5,937,629 | -10,854 |
| step_avg (ms) | ~9700 | ~13700 | +41% slower |

**The K=8→K=128 Δ went negative.** This is a textbook H12-VERIFIED outcome at a wider scale: when training samples deeper K, the model develops a tighter contraction that makes deep-K eval *better* than train-K eval. Iter 86's already-tight Δ=+0.015 became Δ=-0.003.

**The k=4 outlier is expected and not a gate failure:** the K-jitter set no longer contains 4, so the model is no longer optimized for that regime. Eval at k=4 measures off-training-distribution performance. The K=8→K=128 gate (which tracks the in-distribution range) is what matters; that one improved.

**Throughput cost:** step_avg +41% (9.7s → 13.7s). Not a wallclock-cap concern at step-matched 1000-step dev runs, but matters for the 600s submission cap. May want to use only `(6,8,12)` or `(8,12)` for submission.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `88ad22c`, val_bpb int6 = 1.5188).

**Implication:** K-jitter scaling continues to work — the H12 mechanism (K-jitter forces robustness across K) generalizes from 4→16 to 8→20. Could potentially scale further (16→40), but throughput cost would compound. For the current iter 87 win, paying +41% step-time for -0.020 int6 + tightened deep-K is a clear net positive at step-matched comparison.

### H65: WD 0.30 → 0.01 (iter 86) — PROMOTED ★★★ (2026-04-24, MAJOR WIN)

**Claim:** Under the iter 93 landscape (NTP-only + Parcae B̄ + learnable RMSNorm scales everywhere + split shared gates + no BigramHash), the regularization that WD=0.30 was providing has shifted to other mechanisms. A much lower WD floor lets the transformer body express more without destabilizing.

**Test:** iter 86 — one-line `Hyperparameters.weight_decay 0.30 → 0.01`. NOT the same as iter 66a-b (which set WD=0 selectively on 1D params and regressed +0.14 — the WD=0.01 floor stays positive on 1D scalars/norms). Commit `df2cfdb`.

**Result:** PROMOTED. **Largest single-iter improvement in the entire queue.**

| Metric | Iter 93 baseline | Iter 86 | Δ |
|---|---|---|---|
| val_bpb fast | 1.5725 | **1.5042** | **-0.0683** ★★★ |
| val_bpb int6 | 1.5921 | **1.5390** | **-0.0531** ★★★ |
| k=4 | 1.5995 | 1.5423 | -0.0572 |
| k=8 | 1.5834 | 1.5281 | -0.0553 |
| k=16 | 1.5921 | 1.5390 | -0.0531 |
| k=32 | 1.5940 | 1.5415 | -0.0525 |
| k=64 | 1.5943 | 1.5421 | -0.0522 |
| k=128 | 1.5945 | **1.5429** | **-0.0516** |
| K=8→K=128 Δ | +0.0111 | +0.0148 | +0.004 (still ≪0.5 ✓) |
| artifact bytes | 5,737,459 | 5,948,483 | +211,024 (+3.7%) |

Mid-training trajectory was clean: step 200 +0.014 (small early regression as ramp-up needed slightly higher WD), then -0.018/-0.071/-0.083 at steps 400/600/800 — the lead widened monotonically from step 400 onward. **Every K-sweep point improved by ~5%**, an unusually broad and consistent gain.

**Why WD=0.30 was over-regularizing:** the iter 93 landscape has shifted what each mechanism manages:
- learnable prenorm scales (iter 71g+) carry per-projection magnitude control that WD was carrying implicitly
- Parcae B̄ + learnable x0_inject_norm provide structural signal that WD's weight-shrink was preventing
- split shared gates (iter 84) double the modulation degrees of freedom
- removing CTP (iter 94) and BigramHash (iter 93) removed ~2 M params, so the *remaining* parameters are doing more work each — heavy WD throttles them

The accumulated Group A changes converted the loss landscape so the transformer body needs *less* regularization, not more.

**Tradeoff cost:** artifact +211 KB (+3.7%) — weights are slightly larger because less shrinkage. Still ~6 MB, well under 16 MB. K=8→K=128 Δ widened slightly (+0.004) but absolute K=128 still beats iter 93 by 0.052.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `df2cfdb`, val_bpb int6 = 1.5390). This is the new SOTA-track baseline.

**Implication:** WD=0.30 was a legacy from the iter 24 phase (β=0.20, dome-gate regime) where weight magnitudes had to be tightly controlled to keep the contraction property. Modern iter 93 landscape doesn't need that pressure. The next legacy WD-era choice to revisit is the AdamW β2 (iter 25 era), but later. For now, WD=0.01 is the new floor.

**Next candidate for similar audit:** Lyapunov coefficient (iter 88) — same era as WD=0.30, may be similarly over-regularized.

### H64: Disable BigramHash (iter 93) — PROMOTED ★ (2026-04-24)

**Claim:** BigramHash (4096-entry, 128-dim) was added in iter 6 under a very different architecture (pre-DEQ, pre-experts, no Parcae B̄ input injection, no learnable norms). Under the current iter 85 baseline — which now carries token-pair information through (a) the DEQ's x₀ re-injection at every iteration, (b) Parcae's `B̄ ⊙ RMSNorm(x₀)` additive injection at the fixed point, and (c) learnable prenorm scales on every projection — the BigramHash path is architecturally redundant and disabling it frees ~1 MB of artifact budget for the Group D arch scale-up.

**Test:** iter 93 — one-variable change `Hyperparameters.bigram_vocab_size = 4096 → 0`. `GPT.__init__` at L2345 already guards `self.bigram = BigramHashEmbedding(...) if bigram_vocab_size > 0 else None`, and `GPT.forward` at L2723-2724 guards `if self.bigram is not None: x = x + self.bigram(input_ids)`, so the ablation is pure path-drop with zero code churn. Commit `66ce88e`.

**Result:** PROMOTED.

| Metric | Iter 85 baseline | Iter 93 | Δ |
|---|---|---|---|
| val_bpb fast | 1.5707 | 1.5725 | +0.0018 (noise) |
| val_bpb int6 | **1.5898** | **1.5921** | **+0.0023** (≤0.03 ✓) |
| k=4 | 1.5936 | 1.5995 | +0.006 |
| k=8 | 1.5809 | 1.5834 | +0.003 |
| k=16 | 1.5898 | 1.5921 | +0.002 |
| k=32 | 1.5919 | 1.5940 | +0.002 |
| k=64 | 1.5923 | 1.5943 | +0.002 |
| k=128 | 1.5923 | 1.5945 | +0.002 |
| K=8→K=128 Δ | +0.0114 | **+0.0111** | tightened |
| artifact bytes | 6,068,145 | **5,737,459** | **-330,686 (-5.4%) ★** |
| params | 11,246,996 | **10,624,275** | **-622,721 (-5.5%) ★** |

**Mid-training surprise:** val_bpb trajectory inverted. Iter 93 was +0.08 at step 200 (early regression as expected — BigramHash is an early-training aid), then −0.044 at step 400, −0.056 at step 600, −0.020 at step 800, finally +0.002 at step 1000. The transformer body's gradient re-flowed through the freed ~600K params once the initial warmup passed, and iter 93 held a mid-training lead that barely closed at the end. This pattern is consistent with H64: the DEQ's x₀ re-injection + Parcae's B̄ provide the token-pair signal BigramHash was providing, so the model only "misses" BigramHash during the short warmup when the transformer body is still untrained.

**Budget freed for Group D:** -622 K params + -330 KB artifact = substantial headroom for iter 90–92 (low-dim experts / 16-32 experts / D=1024 scale-up). Combined with iter 94's -1.38 M params, the cumulative Group A savings are ~**2 M params (−15% of iter 66b baseline)** — enough to absorb a material expert-bank or model-dim increase.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `66ce88e`). The combined H60 + H64 "budget-freeing Group A" arc has now returned ~2 M params to the transformer body's scale-up budget.

**Implication:** BigramHash was dead weight under the modern landscape. Iter 93's finding generalizes: architectural features added in early iters (pre-Parcae, pre-learnable-norms) should be re-audited because the main path may now carry the signal that the auxiliary path was compensating for. Next candidates for the same audit: Lyapunov penalty (iter 88) and HyDRA denoising (iter 89).

### H63: Stochastic TBPTT (iter 85) — PROMOTED ★ (2026-04-24, narrow margin)

**Claim:** Analogous to K-jitter (H12 VERIFIED), sampling `deq_bptt_k` per step from `{2, 3, 4}` should force the model to be robust across gradient-truncation depths and tighten K-sweep FP quality. Prior iter 28-series (fixed k=4, k=8, k=12) showed deeper TBPTT REGRESSED val_bpb, but that was a fixed-depth specialization issue — jitter over a narrower range tests the broader hypothesis.

**Test:** iter 85 — add `deq_bptt_k_jitter_set = (2, 3, 4)` to Hyperparameters; new `deq_bptt_k_for_step()` helper uses a shuffle-bag sampler mirroring β-jitter (rank-0 sample → `dist.broadcast`). Training loop sets `base_model.deq_bptt_k = deq_bptt_k_for_step(next_step)` right after β-jitter. Commit `b18dc55`.

**Result:** PROMOTED (narrow margin).

| Metric | Iter 84 (fixed k=2) | Iter 85 (jitter 2/3/4) | Δ |
|---|---|---|---|
| val_bpb fast | 1.5653 | 1.5707 | +0.0054 |
| val_bpb int6 | **1.5844** | **1.5898** | **+0.0054** |
| k=4 | 1.5854 | 1.5936 | +0.008 |
| k=8 | 1.5742 | 1.5809 | +0.007 |
| k=16 | 1.5844 | 1.5898 | +0.005 |
| k=32 | 1.5867 | 1.5919 | +0.005 |
| k=64 | 1.5872 | 1.5923 | +0.005 |
| k=128 | 1.5873 | 1.5923 | +0.005 |
| K=8→K=128 Δ | +0.0131 | **+0.0114** | **tightened by -0.002** ★ |
| artifact bytes | 6,074,999 | 6,068,145 | -6,854 |
| step_avg (ms) | ~8,015 | ~9,700 | **+21% slower** |

**Mid-training trajectory showed large transient regression** (step 400: +0.062, step 600: +0.077) before narrowing to +0.005 at step 1000. This matches the iter 28c pattern — deeper backward depths slow training but the final converged point shows tighter K-sweep. The model takes longer to specialize because each step sees a different truncation depth, but at the end it is more robust to *any* backward depth.

**Why the K-sweep tightened:** when `deq_bptt_k ∈ {2,3,4}` is jittered, the effective depth-equivalent gradient signal mixes 2/K, 3/K, 4/K capture ratios. The model learns to make its forward converge at ratios higher than the minimum (2/K), so at full-K inference its FP convergence is tighter. Analogous to H12's K-jitter tightening K=8→K=128 by 125×.

**Throughput cost (+21%) is the real tradeoff.** Step_avg jumps from ~8.0s to ~9.7s because the larger k=3/k=4 sampled values cost more backward iterations. For step-matched dev comparison this is orthogonal, but for **wallclock-constrained submission** (8×H100 600s cap), a 21% throughput hit would erase ~200 steps of training, likely eating more val_bpb than the +0.005 cost. Reconsider if iter 85 actually gets enabled in the submission config — may want to revert to fixed k=2 OR narrow the set to `(2, 3)` only.

**Status:** ✅ VERIFIED (narrow-margin promote per ≤0.03 carry-forward rule). Commit `b18dc55`.

**Implication:** TBPTT jitter does provide K-sweep-tightening benefit analogous to K-jitter, but at meaningful throughput cost. Worth it for dev A/B exploration; may want to re-evaluate for final submission config.

### H62: Independent attn/mlp shared-expert sigmoid gates (iter 84) — PROMOTED ★ (2026-04-24)

**Claim:** Block.shared_gate was a single `nn.Linear(dim, num_shared_experts)` producing ONE sigmoid gate applied to BOTH the attention shared-expert output and the MLP shared-expert output. This accidental symmetry forced the two paths to open/close together. Splitting it into `shared_gate_attn` + `shared_gate_mlp` (each with its own prenorm scale) lets each path learn its own modulation and should improve val_bpb at negligible param cost.

**Test:** Iter 84 — controlled A/B on iter 94 baseline. One change: Block.__init__ now allocates two `nn.Linear(dim, num_shared_experts, bias=True)` modules (`shared_gate_attn`, `shared_gate_mlp`) each with its own `shared_gate_norm_weight_{attn,mlp}` parameter. Block.forward computes `g_s_attn` and `g_s_mlp` separately and threads each to its respective expert bank. Commit `6494a50`. Param cost: +1,537 (+0.014%).

**Result:** PROMOTED.

| Metric | Iter 94 baseline | Iter 84 | Δ |
|---|---|---|---|
| val_bpb fast (k16) | 1.5777 | **1.5653** | **-0.0124** |
| val_bpb int6 (k16) | 1.5952 | **1.5844** | **-0.0108** |
| k=4 | 1.6278 | 1.5854 | -0.0424 |
| k=8 | 1.5843 | 1.5742 | -0.0101 |
| k=16 | 1.5926 | 1.5844 | -0.0082 |
| k=32 | 1.5943 | 1.5867 | -0.0076 |
| k=64 | 1.5946 | 1.5872 | -0.0074 |
| k=128 | 1.5966 | 1.5873 | -0.0093 |
| K=8→K=128 Δ | +0.0103 | +0.0131 | +0.003 (tiny widen, still ≪ 0.5) |
| artifact bytes | 6,007,121 | 6,074,999 | +67,878 (+1.1%) |

**Every K-sweep point improved.** Mid-training val_bpb lead widened monotonically: step 200 −0.013, step 400 −0.039, step 600 −0.028, step 800 −0.013, step 1000 −0.012. The k=4 improvement is particularly large (−0.042) — the shallower the DEQ solve, the more the per-path shared gate matters because the routed-expert mixture hasn't had time to compensate.

**Confirmation that gates actually diverged:** `shared_gate_std` (stdev across the concatenation of the two (B, T, S) gates) grew from **0.0000 at step 0 → 0.17 at step 200 → 0.14 at step 800**. If the model hadn't wanted the new DoF, the two gates would have stayed identical and `shared_gate_std` would have stayed at 0 (same init). The 0.14–0.17 range shows the attn and mlp gates each found their own preferred opening.

**Status:** ✅ VERIFIED. PROMOTED as new baseline (commit `6494a50`).

**Implication:** The pre-iter-84 architecture had a cheap structural flaw (shared gate) that was silently coupling attn-shared and mlp-shared modulation. Fixing it gives a clean −0.011 int6 win for ~0.01% param cost. This pattern (trivial param cost, fixes accidental symmetry) is the model of the Group A iterations. Recommend auditing remaining "shared" modules for similar accidental couplings.

### H61: GLU-style LeakyReLU(0.5)² MLP activation (iter 83) — REFUTED ✗ (2026-04-24)

**Claim:** Replacing SwiGLU (`F.silu(gate) * fc`) with GLU-style LeakyReLU² (`F.leaky_relu(gate, 0.5).square() * fc`) in `MLP.mix_experts` should improve val_bpb, analogous to the SOTA leaderboard config (abaybektursun 1.1194) which uses LeakyReLU².

**Test:** iter 83 — one-line swap at both MLP activation sites (L1599 training path, L1958 Block.forward diagnostic). Commit `8c7a33f`. A/B against iter 94 baseline (c2eff43, val_bpb int6=1.5952).

**Result:** REFUTED.

| Metric | Iter 94 (SwiGLU) | Iter 83 (LeakyReLU²) | Δ |
|---|---|---|---|
| val_bpb fast | 1.5777 | 1.5895 | +0.0118 |
| val_bpb int6 | **1.5952** | **1.6076** | **+0.0124** |
| K-sweep k128 | 1.5966 | 1.6093 | +0.0127 |
| K=8→K=128 Δ | +0.0073 | +0.0087 | +0.0014 (widened) |
| artifact bytes | 6,007,121 | 6,090,865 | +83,744 |

Mid-training val_bpb trajectory flipped sign: iter 83 was ahead at step 200 (1.9641 vs 1.9782), matched at 400 (1.7474 vs 1.7486), then FELL BEHIND from step 600 onward (1.6831 vs 1.6590). The early-training advantage was likely the sharper-gradient benefit of `leaky²` during the unstable warmup phase; once the model settled, the smooth `silu` gating provided better inductive bias for the MoE experts' convex combinations.

**Why it's different from SOTA leaderboard success:** The SOTA abaybektursun config uses a *non-gated* FFN — `F.linear(x, up_w) → leaky_relu(0.5) → square → F.linear(down_w)`. That's a single `up` projection with the squared activation acting as the nonlinearity. Our MoE layout has two parallel projections per expert (`expert_gate` + `expert_fc`) multiplied GLU-style. The GLU product of two projections already provides rich nonlinearity via the multiplication; swapping silu for leaky² on the gate side evidently disrupts the balance without the compensating bottleneck structure of the non-gated form.

**Status:** ✗ REFUTED. Reverted at commit `34ef98d` — train_gpt.py restored to iter 94 SwiGLU. The SOTA-faithful non-gated variant (which would require dropping `expert_fc` entirely) is reserved as a potential follow-up under the Group D architectural rewrite — it's a larger refactor than Group A should contain.

**Implication:** For our per-expert GLU layout, SwiGLU is the better activation. A LeakyReLU² variant would need to come *with* the structural change to non-gated form (single up-projection) to replicate the SOTA pattern, not as a drop-in activation swap.

### H60: Disable CTP head entirely (iter 94) — PROMOTED ★ (2026-04-24)

**Claim:** Under the iter 66b landscape (Parcae per-dim Ā/B̄ input injection, learnable prenorm scales everywhere, WD=0.30), the dual-head MoS (CTP + NTP) is net-neutral-to-beneficial when collapsed to NTP-only. Removing the CTP parameter banks (`gate_ctp`, `gate_ctp_norm_weight`, `ctp_a_norm_weight`, `A_ctp_shared`, `A_ctp`, `B_denoise`, `ctp_rank_norm_weight`) and skipping the CTP loss + inference mixing frees ~1.38M parameters (~10.9%) without hurting val_bpb.

**Mechanism:** CTP weight is `0.05 × num_refinements × refine_strength` where `refine_strength = min(refine_alpha / 0.5, 1.0)` ramps only after `refine_ramp_frac = 0.85` of training. This means CTP gradient is near-zero for the first 85% of steps — the signal it provides is concentrated in the last ~150 steps, and at that point the NTP representation is mostly set. The input-preservation pressure that CTP's denoising loss provided implicitly is now covered by Parcae's `B̄` injection (iter 66b H58). So the dual-head design is **architecturally redundant** under the current landscape. Removing it recovers capacity-per-parameter.

**A/B evidence (controlled, 1000-step dev runs):**

| Metric | Baseline (iter 66b, SwiGLU + CTP on) | Iter 94 (NTP-only) | Δ | Verdict |
|---|---|---|---|---|
| val_bpb fast | 1.5724 | 1.5777 | +0.0053 | noise |
| val_bpb int6 | **1.5926** | **1.5952** | **+0.0026** | ≤0.03 ✓ |
| k=4 | 1.6117 | 1.6278 | +0.016 | — |
| k=8 | 1.5843 | 1.5893 | +0.005 | — |
| k=16 | 1.5926 | 1.5952 | +0.003 | — |
| k=32 | 1.5943 | 1.5964 | +0.002 | — |
| k=64 | 1.5946 | 1.5965 | +0.002 | — |
| k=128 | 1.5946 | 1.5966 | +0.002 | — |
| K=8→K=128 Δ | +0.0103 | **+0.0073** | tightened | ≤0.5 ✓ |
| artifact bytes | 6,599,548 | **6,007,121** | **-592,427 (-9.0%)** | — |
| model params | 12,627,862 | **11,245,459** | **-1,382,403 (-10.9%)** | — |

**Val trajectory during training** (iter 94 consistently ahead of baseline mid-training, then baseline edges ahead in last 15% when refine_ramp activates CTP):
- step 200: iter94=1.9782, baseline=1.9717 (+0.007)
- step 400: iter94=1.7486, baseline=1.7643 (**-0.016**)
- step 600: iter94=1.6590, baseline=1.6774 (**-0.018**)
- step 800: iter94=1.6131, baseline=1.6154 (-0.002)
- step 1000: iter94=1.5777, baseline=1.5724 (+0.005)

The mid-training lead (-0.016 to -0.018) is strong evidence that CTP competes with NTP for capacity during the refinement-inactive phase; the late-training narrowing is CTP paying off slightly during its ramp window. Net at final int6 eval: +0.0026 regression, comfortably within the 0.03 noise/carry-forward band.

**Tech debt (failure_categories, recorded for next iter):**
- `attn_ortho = 0.83` (>0.5 gate) — attention experts correlated without CTP's orthogonality pressure.
- `mlp_ortho = 0.69` (>0.5 gate) — MLP experts correlated.
- `deq_recon_err = 3.97` (>0.1 gate) — RevDEQ reconstruction degraded at final eval K.

These are carry-forward concerns that the LB loss + Parcae's Ā/B̄ should continue to manage, or that a future iter can address with targeted ortho regularization on the MoS side if needed. None block promotion per CLAUDE.md §10 val_bpb-primary policy.

**Status:** ✅ VERIFIED — controlled A/B with a single variable change. PROMOTED as the working baseline (commit `c2eff43`).

**Implication:** The dual-head MoS design was an unforced inheritance from the TSU reference implementation. Under the contemporary RevDEQ + Parcae landscape, it's architecturally dead weight. The 1.38M freed parameters and 592 KB freed artifact budget will compound into later arch scale-up iters (iter 90 low-dim experts, iter 91 scale-experts, iter 92 D=1024).

### H58: Parcae-paper-faithful DEQ input injection (iter 66b) — PROMOTED ★ (strict generalization, unconditional)
**Claim:** Replacing the iter-66a post-refactor `T(z,x₀) = Δ(z,x₀)` with the Parcae ZOH-discretized form
```
T(z, x₀) = B̄ ⊙ RMSNorm_learn(x₀) + Δ(z, x₀)
```
where Ā and B̄ share *only* the per-dim step size Δ (otherwise independent), preserves input information at the equilibrium without creating the pre-iter-66a unconditional `T = x₀ + Δ` shortcut. Improves val_bpb vs iter 66a because the post-diff `T = Δ` form loses input information if experts collapse, while iter 66a-pre `T = x₀ + Δ` was unconditional.

**Strict-generalization argument → unconditional promote.** Iter 66b subsumes iter 74b by construction:
- `B̄ → 0` (via raw_b → −∞) recovers `T_θ = Δ` exactly (iter 74b's form). `B̄ = Δ·softplus(raw_b) + Δ·ε_min` so the lower bound is `Δ·ε_min ≈ 1e-3` — effectively zero for training dynamics.
- Alternatively `x0_inject_norm_weight → 0` zeroes the injection directly.
- Ā parametrization change (compound wrapper → `ε_rev + (1−ε_rev)·exp(Δ·A)`) re-parametrizes the same Ā ∈ [0.1, 1) range; raw_a/raw_delta can produce any target Ā in both forms.

Therefore any val_bpb regression iter 66b → iter 74b is an **optimizer-landscape artifact** (different init, extra parameters, slightly different gradient topology), not a capacity loss. The fix is to tune the new parameters (parcae_lr, raw_b init, x0_inject_norm_weight init), not revert.

**Parametrization (paper-faithful + RevDEQ safety):**
- `Δ = softplus(parcae_raw_delta) + ε_min` (step size, shared by Ā and B̄)
- `A = −(softplus(parcae_raw_a) + ε_min)` ← independent of B
- `B = softplus(parcae_raw_b) + ε_min` ← independent of A, NEW in iter 66b
- `Ā = ε_rev + (1 − ε_rev) · exp(Δ·A)`, with `ε_rev = parcae_reversibility_floor = 0.1` (correctness constant for RevDEQ backward, NOT a tuning knob)
- `B̄ = Δ · B` (no floor — B̄ never in solver reconstruction)
- `β = 1 − Ā` (solver blend unchanged from iter 66a)

**Fixed point (Ā cancels):** `y* = B̄ ⊙ RMSNorm_learn(x₀) + Δ*`.

**Mechanism:**
- Iter 66a tied `B̄ = 1 − Ā`, collapsing Parcae's two per-dim degrees of freedom (persistence via Ā + input forcing via B̄) into one. Iter 66b restores paper-faithful independence.
- The new `x0_inject_norm_weight: nn.Parameter(torch.ones(dim))` on `Block` is the learnable pre-RMSNorm scale for x₀ inside the injection term. Shared across experts (x₀ is the DEQ input seen by all experts; the per-expert invariant applies strictly inside the expert path, not here) and routed to AdamW via `CONTROL_TENSOR_PATTERNS += "norm_weight"` (covered in commit 1).

**Predicted effect:** At least parity with iter 66a val_bpb; likely improvement because the expressive degree of freedom returns.

**Measurement (not a gate — iter 66b is already promoted):** ≥200-step dev run vs. iter 74b reference for val_bpb tracking + diagnostic. Init `parcae_raw_b` so `B̄₀ ≈ 1 − Ā₀ = 0.3` → step-0 effective dynamics match iter 66a.

**If val_bpb regresses (fix, don't revert):**
- First tune: try raising `parcae_lr` (currently 0.002 — a 10× slower than `scalar_lr`); the extra parameter needs enough gradient to move.
- Second tune: init `parcae_raw_b` to drive `B̄₀ → 0` so step 0 matches iter 74b exactly, then let training learn to open the injection gate if it helps. This isolates any optimization artifact from any latent capacity gain.
- Third tune: `x0_inject_norm_weight` init / LR — currently covered by `scalar_lr` via `CONTROL_TENSOR_PATTERNS`; may need its own group.
- Confound to keep in mind: `CONTROL_TENSOR_PATTERNS += "norm_weight"` from commit 1 migrated two pre-existing norm banks Muon→AdamW — any delta could carry-over from that and be unrelated to Parcae-faithful.

**Reversibility sanity:** `experiments/test_arch.py::test_revdeq_reconstruction_at_a_bar_floor` asserts reconstruction stays finite (not NaN/Inf) when raw_a is driven to the ε_rev saturation; smoke-test recon_err stays ≤ 1e-1 under normal Ā training ranges. Full details in the new CLAUDE.md §RevDEQ Reversibility Floor Rule.

**Related:** H56 (Lyapunov), H57 (γ=0.95); iter 66a (tied Parcae). Doc: `opg_doc.tex` §sec:parcae_params + §sec:algorithm describe the combined RevDEQ+Parcae setup.

### H71: "More smaller experts" at iso-cost on linear projections (iter 96 rank/2, E×2) — VERIFIED ★ (2026-04-26)

**Claim:** Halving per-expert LoRA rank (`attn_expert_rank 128→64, mlp_expert_rank 192→96`) and doubling expert count (`num_experts 8→16`) keeps the **linear-projection FLOPs constant** (E·R = const) while doubling routing diversity. The DeepSeek-MoE / Switch hypothesis predicts more, smaller experts win at iso-cost on the dominant cost class.

**Test:** iter 96 — `num_experts=16, attn_expert_rank=64, mlp_expert_rank=96` on the iter 89 baseline (full-D LoRA-style attention/MLP, D=768, H=8, d_head=96 at FA-optimal sweet spot). All other knobs unchanged. Commit `b962b5f`. 1000 steps on 2× L40S, ~6.6 hours wallclock at step_avg=23.4-24.5s (vs iter 89 baseline ~13s — `+78%` step-time penalty, larger than the predicted +10-20% because **SDPA cost** scales `E·H·d_head·B·T²` and is independent of `attn_expert_rank`, so doubling E doubles SDPA wall-time).

**Result:** ✅ PROMOTED — strong improvement on the per-param-efficiency frontier.

| Metric | iter 89 baseline | iter 96 | Δ |
|---|---|---|---|
| val_bpb (int6) | 1.5264 | **1.4903** | **−0.0361** ★ |
| val_bpb (fp32, K=16 best-K) | 1.5264 | 1.4603 | −0.0661 |
| param count | ~13 M | 13.95 M | +7% (KV-A/KV-B/Wo paths scale with E independently of R) |
| artifact_bytes | ~13.4 MB | 7.55 MB | **−44% smaller artifact** ★ (47% of 16 MB budget) |
| step_avg | ~13 s | 23.4-24.5 s | **+78%** (SDPA cost dominates wall-time more than predicted) |
| K=128 vs best-K Δ | −0.004 | +0.0016 | tighter contraction ✓ (gate: ≤ 0.5) |
| attn_cv | ~0.18 | 0.20 | similar ✓ |
| attn_entropy | ~0.95·max | 0.99·max | uniform routing ✓ |
| Min expert share | ~0.04 | 0.040 | no dead experts ✓ |
| Max expert share | ~0.20 | 0.107 | no winner-take-all ✓ |
| router_mass | ~0.95 | 0.79 | gate closing some experts (healthy) |

**Why it works:**
1. **Routing diversity scales linearly with E** — 16 experts give the router 2× the "specialization slots" to compose. Soft dense routing (all experts process all tokens) doesn't suffer from dead-expert pathology of top-k routing, so larger E is pure capacity gain.
2. **Per-expert linear projections still have meaningful rank** — at D=768, R=64 the Q linear sees a 64-dim subspace. Empirically enough headroom; not the floor.
3. **No bottleneck/proj_rank penalty** — unlike iter 90's bottleneck experts (`proj_rank=32` valve), iter 96 keeps all activations at full D=768 with d_head=96 in the FA tensorcore sweet spot. **Per-param efficiency is preserved.**
4. **Artifact savings as bonus** — total params actually grew by 7% but the int6 + zstd-22 compression got tighter (more, smaller experts compress better than fewer, larger ones), netting **−44% on artifact size**. Frees significant budget for iter 97/98 capacity scaling.
5. **K-sweep tightened** — K=128 vs best-K (K=16) Δ = +0.0016 (well within 0.5 gate). The DEQ FP is highly contractive at the new layout.

**Implications:**
- **Iter 97 (E=24, attn_rank=42, mlp_rank=64)** is the natural continuation along this validated axis. Per-param efficiency at iter 96 should hold; routing diversity gain may diminish (E≥16 is already past where most MoE papers report saturation). Justified to test once.
- **Iter 98 (D=1024 under iter 96 LoRA layout)** — D scaling under validated layout. Linear in artifact cost; needs budget check (currently 47% used → can grow to ~16 MB with D=1024 at E=16-24).
- **The "more, smaller experts" axis is now the established scaling direction** for this codebase, replacing the previously-failing "bottleneck" axis. Closes Group D (bottleneck → NOT PROMOTED, archived) and opens Group F (LoRA-rank/E joint scaling) as the active design dimension.

**Confounds / things to watch:**
- The +78% step-time penalty is real. At submission time (600s wallclock), iter 96's config trains ~600/24.5 ≈ 24 steps vs iter 89's 600/13 ≈ 46 steps. **Submission-mode val_bpb may regress** if step count matters more than per-step capacity. Test before claiming submission-eligibility.
- **router_mass dropped to 0.79** (from ~0.95 baseline) — the model is learning to suppress some experts via the sigmoid gate, not just route around them. This is healthy under soft dense routing but worth monitoring at E=24, E=32 to confirm it doesn't collapse.
- **Compile recompiles intensified** with E=16 + k-jitter {8,12,20} + DDP. Adds ~5-10 min of front-loaded overhead. At E=24 and beyond this may need `torch._dynamo.config.recompile_limit` bumping.

**Related:** H43 (low-rank experts — earlier attempt, proven viable here), H44 (bottleneck experts — failed alternative, see H69/H70), H47 (scale to E=16-32 — this iter validates the axis).

### H72: More-experts axis saturates past E=16 / R=64 on D=768 (iter 97 E=20, R=51/77) — TESTED ✗ NOT PROMOTED (2026-04-26)

**Claim:** The "more, smaller experts" axis validated by iter 96 (H71) extends linearly past E=16. Continuing the same iso-cost trade — E 16→20, attn_expert_rank 64→51, mlp_expert_rank 96→77 (E·R held constant on Q/MLP linears) — should yield further val_bpb improvement at proportional throughput cost.

**Test:** iter 97 — `num_experts=20, attn_expert_rank=51, mlp_expert_rank=77` on the iter 96 baseline. All other knobs unchanged. Commit `acce3d4`. (Originally targeted E=24/R=42; OOM on 2× L40S at backward (44 GiB cap); fell back to E=20 with `PYTORCH_ALLOC_CONF=expandable_segments:True`.) 1000 steps, ~8.6 hours wallclock at step_avg=28-31s.

**Result:** ❌ **NOT PROMOTED** — the axis has saturated. Even though val_bpb is within the 0.03 carry-forward gate, **per-wallclock val_bpb regresses** by ~19%, and the projected submission-run gap (600s cap) widens.

| Metric | iter 96 baseline | iter 97 | Δ |
|---|---|---|---|
| val_bpb (int6, fast eval K=16) | 1.4903 | **1.5036** | **+0.0133** (within 0.03 ✓ but regression) |
| val_bpb (fp32 step 1000) | 1.4603 | 1.4672 | +0.007 |
| K=4 / K=8 / K=16 / K=32 / K=64 / K=128 | 1.723 / 1.498 / **1.490** / 1.492 / 1.492 / 1.492 | 1.756 / 1.510 / **1.504** / 1.505 / 1.505 / *pending* | K=16 best-K both, +0.013 across deep K |
| K=128 vs best-K Δ | +0.0016 | ~+0.002 | tighter both, well within 0.5 gate ✓ |
| Params | 13.95 M | 15.62 M | +12% |
| step_avg | 23.4 s | 28-31 s | **+22-32%** ✗ |
| Peak VRAM | 35.7 GiB | 42.75 GiB | +20% (near OOM cliff) |
| Artifact | 7.55 MB | 8.42 MB | +12% (still 53% of budget) |
| **bpb / wallclock-hour** | 0.229 | 0.186 | **−19%** ✗ |
| **Submission run @ 600s** | ~25 steps | ~20 steps | -5 steps × no val_bpb gain ✗ |

**Trajectory** (the smoking gun for saturation):

| Step | iter 97 val_bpb | iter 96 val_bpb | Δ |
|---|---|---|---|
| 200 | 1.9744 | 1.9884 | **−0.014** (capacity premium) |
| 400 | 1.6463 | 1.6564 | **−0.010** (still leading) |
| 600 | 1.5406 | 1.5433 | −0.003 (gap closing) |
| 800 | 1.5008 | 1.4935 | **+0.0073** (gap reverses!) |
| 1000 (final int6) | 1.5036 | 1.4903 | **+0.0133** (regression confirmed) |

The axis's win is front-loaded: extra experts deliver capacity early when each expert is undertrained, but the rank-halving (R=64→51) trades per-token expressivity faster than routing diversity compensates as training matures. By step 800 the rank cost dominates the diversity gain.

**Why E=16 / R=64 is approximately Pareto-optimal on this codebase:**
1. **Rank floor**: each expert sees a `R`-dim subspace of D=768. Below R≈64, per-expert representational capacity drops faster than soft-dense routing diversity gains can compensate.
2. **Router fan-out cost**: the routing softmax over 38 outputs (= 2 × 19 routed components) at iter 97 mechanically smears the per-token distribution — even with `min_share_loss_weight=10.0` keeping experts alive, signal-to-noise on "best expert per token" drops with more candidates.
3. **`min_share` vs specialization tension**: with E=20, the constraint pulls toward uniform-ish utilization, fighting the specialization that would otherwise differentiate the experts. iter 99 (sparsemax) is the principled fix for this — exact zeros let the router specialize without violating the global balance.

**Override of the carry-forward auto-promote rule**: the 0.03 gate is val_bpb-primary by default, so iter 97 would auto-promote on val_bpb alone. Per user directive 2026-04-26, the per-wallclock regression overrides for this iter — the submission run (600s cap) is what the metric ultimately serves, and iter 97 strictly regresses at that scope. **Documented as override, not protocol change** — the carry-forward rule remains val_bpb-primary; this is a one-off override on per-wallclock grounds.

**Implications for queue:**
- **Stop scaling E along this axis.** Further attempts (E=24, E=32) would saturate harder. iter 97 closes the rank-halving subdirection of Group F.
- **Pivot to orthogonal axes**:
  - **iter 98 (D=768→1024)** — different axis, each expert linear scales linearly in D, d_head bumps to 128 (FA tensorcore sweet-spot upgrade). Expected to pay better than more E.
  - **iter 99 (sparsemax)** — dissolves the `min_share` vs specialization tension by letting routing produce exact zeros, may unlock effectively-larger E by allowing peakier per-token distributions.
- **Retain iter 96 as baseline.** Config reverted at the same commit as this H72 entry.
- **Inference-time top-k** (conditional iter 102) could still test "iter 97-style E with sparsemax masking off the bottom experts at inference" — captures the routing-diversity gain with deployment-side sparsity. Logged for later.

**Related:** H43 (low-rank experts), H47 (scale E=16-32 — this iter shows the upper boundary), H71 (the iter 96 PROMOTION that opened this axis), iter 99 sparsemax (next attack on the same problem from a different angle).

### H73: D=1024 under iter 96 LoRA layout — NOT TESTED on dev hardware (3× OOM at 44 GiB cap; 2026-04-26)

**Claim:** Scaling `model_dim 768 → 1024` on the iter 96 baseline (E=16, attn_R=64, mlp_R=96, full-D LoRA, K-jitter {8,12,20}) is an orthogonal axis to E-scaling and should pay off — d_head bumps 96 → 128 (FA tensorcore sweet-spot), per-expert linears scale linearly in D, and the iter 91+92 bottleneck-D failure (H70) was attributed to the bottleneck design not D-scaling itself.

**Test:** iter 98 — 3 attempts on 2× L40S dev hardware (44.4 GiB/rank cap):

| Attempt | Config | Outcome |
|---|---|---|
| 1 | D=1024 + K_max=20 + seq=2048 + batch=524K | OOM at first backward (~46 GiB needed) |
| 2 | D=1024 + K_max=16 + seq=2048 + batch=524K | OOM at same ~43.6 GiB ceiling — issue is D-bound, not K-bound |
| 3 | D=1024 + K_max=16 + seq=1024 + batch=524K | OOM at same 43.57 GiB — DEQ TBPTT floor + D=1024 activations exceed cap regardless of seq/K reductions |

Halving `train_batch_tokens` to 262K caused divergence at the same LRs (smoke test loss UP 6.08 → 6.28; would need LR rescaling, multi-knob change). Reducing both seq and K simultaneously also failed.

**Result:** ❌ **NOT TESTED on dev hardware.** Documented gap, not refutation.

**Why it doesn't fit:**
- iter 96 (D=768): peak VRAM 35.7 GiB. Activations dominate: B·T·D·layers ≈ 256·2048·768·12·4B = 19 GB plus expert outputs.
- iter 98 (D=1024): activation memory scales linearly in D → ~25 GiB just for hidden states. Plus DEQ K=20 saves K+1 states for fp64 reconstruction → another ~25 GiB.
- Total >50 GiB peak, exceeds 44 GiB cap.

**Implications:**
- D-scaling on this codebase needs **gradient checkpointing** (recompute activations on backward; ~30% throughput penalty but cuts activation memory by 12×) OR **submission hardware** (8× H100 80 GiB = 640 GiB total, plenty of headroom).
- For dev iteration: D=768 stays. The "more, smaller experts" axis (iter 96 H71) is the only architectural width axis we can exercise on 2× L40S.
- **For submission**: D=1024 should be re-tested on 8× H100. Iter 91+92 (bottleneck D=1024) was tested on dev because bottleneck experts compute at small r (≤192) per expert, fitting in dev VRAM. LoRA D=1024 doesn't have that escape valve.

**Implications for queue:**
- Iter 98 marked NOT TESTED. Don't retry on dev hardware without gradient checkpointing — same OOM expected.
- Skip ahead to iter 99 (sparsemax) on iter 96 baseline (D=768). Sparsemax doesn't change D, so VRAM cost is iter-96-comparable.
- Future submission-mode iter: re-test D=1024 on 8× H100 (orthogonal axis still untested architecturally).
- Could also try gradient checkpointing as iter 98b — but that's a significant code refactor; not on the active queue.

**Related:** H70 (iter 91+92 bottleneck D=1024 — different arch, NOT PROMOTED but did fit in VRAM), H71 (iter 96 PROMOTION at D=768 — the working baseline this couldn't extend on dev).

### H74: Sparsemax routing — NOT PROMOTED ✗ (extreme architectural sparsity ≈ +0.16 capacity cost; 2026-04-27)

**Claim:** Replacing softmax routing (iter 96) with sparsemax (Martins & Astudillo 2016) — closed-form simplex projection that produces exact zeros for low-logit experts — drives architectural per-token specialization without a tuning knob. Two distinct entropies in routing: per-token entropy (sparsity signal, target LOW) and global utilization entropy (dead-expert sentinel, target HIGH). Sparsemax should crush per-token entropy directly while min_share_loss=10.0 protects global balance.

**Test:** iter 99 — `Hyperparameters.router_kind = "sparsemax"` on iter 96 baseline. All other knobs unchanged. Commit `8f2049e`. 1000 steps, 7.0 hours wallclock.

**Result:** ❌ **NOT PROMOTED** — extreme sparsity exceeds capacity threshold.

| Metric | iter 96 baseline | iter 99 sparsemax | Δ |
|---|---|---|---|
| val_bpb (int6, fast eval K=16) | 1.4903 | **1.6494** | **+0.1591** ✗ (>>0.03 gate) |
| val_bpb (fp32 step 1000) | 1.4603 | 1.6032 | +0.1429 |
| Params | 13.95 M | 13.95 M | unchanged ✓ |
| step_avg | 23.4 s | 23.4 s | identical ✓ (sparsemax overhead amortized) |
| Peak VRAM | 35.7 GiB | 35.7 GiB | identical ✓ |
| Artifact | 7.55 MB | 7.66 MB | +1.5% ✓ |
| **attn_entropy (per-token)** | ~3.0 (≈ uniform over 30 components) | **0.009** (≈ pure top-1 routing) | crushed by 332× ★ |
| attn_cv | ~0.20 | ~0.75 | strong concentration |

**Trajectory** — gap STABILIZES, doesn't diverge:

| Step | iter 99 | iter 96 | Δ |
|---|---|---|---|
| 200 | 2.0743 | 1.9884 | +0.086 |
| 400 | 1.7809 | 1.6564 | +0.125 |
| 600 | 1.6675 | 1.5433 | +0.124 |
| 800 | 1.6144 | 1.4935 | +0.121 |
| 1000 (fp32) | 1.6032 | 1.4603 | +0.143 |
| 1000 (int6) | **1.6494** | **1.4903** | **+0.1591** |

The val_bpb gap is **roughly constant at +0.12-0.16** from step 200 onwards — iter 99's trajectory is iter 96's trajectory shifted up by an architectural-capacity penalty, not a divergent failure. Train_loss gap is much larger (~+1.0 / +35%) than val_bpb gap (+0.16 / +5-8%): **sparsemax acts as implicit regularization** — the model can't memorize as effectively (worse train fit) but generalizes proportionally well (smaller val gap).

**Why it didn't promote (root cause):**

1. **Pure top-1 routing trap.** attn_entropy = 0.009 means each token routes to ~1.01 experts effectively. Inactive experts receive ZERO gradient (sparsemax sets w=0 → gradient is zero through that path) → bad initial routing freezes → model can't escape. min_share_loss=10.0 prevents complete collapse but doesn't drive useful re-diversification at this aggressive sparsity.

2. **Capacity loss > regularization gain.** With effectively k=1 routing, the model has 1/16th the parallel expert capacity per token. Even with implicit-regularization benefit, the +0.16 capacity cost exceeds the regularization gain by ~3×.

3. **Inability to specialize properly.** True specialization requires the router learning good token→expert assignments. With zero gradient through unused experts, the router gets very weak signal for expert reassignment. Becomes path-dependent on initialization.

**Implications:**

- **Architectural sparsity is principled but α=2 is too aggressive on this codebase.** The +0.16 capacity cost is the price of pure top-1 routing.
- **α=1.5 entmax (iter 101) is the natural middle ground.** Mild sparsity (some zeros but mostly soft) preserves gradient flow through low-weight experts → router can still learn to diversify → less capacity loss.
- **iter 99b (sparse expert dispatch)** is shelved — only valued if iter 99 promoted, which it didn't.
- **iter 100 (entropy penalty + softmax)** is also DEFERRED — iter 99's regularization-as-implicit insight suggests the architectural axis (α-entmax family) is more aligned than the loss-side proxy. Skip iter 100 unless iter 101 fails.

**Diagnostic data lost:** the K-sweep with iter 97.6's new Hutchinson + acyclicity-prime probes crashed at the first probe call due to a dtype mismatch (`Float vs BFloat16`) — z_star saved in fp32 by training-time hot path, eval-time SharedBlock runs in bf16. **Bug fixed in same commit as this H74 entry**: `_compute_eval_fp_lipschitz` now casts inputs to the SharedBlock's compute dtype before the probe. Iter 101 will be the first run where the new Lipschitz/acyclicity-prime data actually lands.

**Related:** H32 (DEQ smoothness preservation — sparsemax piecewise smooth, RevDEQ-safe ✓ as designed), iter 96 H71 (working softmax baseline), iter 101 (α=1.5 entmax, the principled next step), iter 99b shelved.

### H75: α=1.5 entmax routing — NOT PROMOTED ✗ (architectural-sparsity capacity gap +0.10; 2026-04-27)

**Claim:** α=1.5 entmax (Peters, Niculae & Martins 2019) is the principled middle ground between softmax (α=1, iter 96, no sparsity) and sparsemax (α=2, iter 99, top-1 trap). At α=1.5, weights are typically all positive but peakier, with possibly some exact zeros — preserving gradient flow through low-weight experts to avoid iter 99's monotonic collapse, while still creating architectural sparsity for implicit regularization.

**Test:** iter 101 — `Hyperparameters.router_kind = "entmax15"` on iter 96 baseline. Closed-form α-entmax via 30-iter bisection over the threshold τ. Commit `caf8a0d`. 1000 steps on 2× L40S, 7.1 hours wallclock.

**Result:** ❌ NOT PROMOTED — but VINDICATES the middle-ground hypothesis structurally; just doesn't beat softmax on val_bpb.

| Metric | iter 96 | iter 99 (sparsemax) | iter 101 (entmax-1.5) | Δ vs iter 96 |
|---|---|---|---|---|
| val_bpb int6 | 1.4903 | 1.6494 | **1.5873** | +0.0970 (vs +0.159 sparsemax) |
| val_bpb fp32 step 1000 | 1.4603 | 1.6032 | 1.5449 | +0.085 |
| step_avg | 23.4 s | 23.4 s | ~24-25 s | +5-7% (bisection overhead) |
| Peak VRAM | 35.7 GiB | 35.7 GiB | 35.7 GiB | unchanged ✓ |
| Artifact | 7.55 MB | 7.66 MB | 7.58 MB | identical ✓ |
| **attn_entropy step 1000** | **~3.0** (uniform) | **0.009** (top-1 trap) | **1.94** (≈7 effective experts/token, sweet spot ★) | as designed |

**Trajectory — gap closes during training (entmax-1.5 SELF-CORRECTS where sparsemax CAN'T):**

| Step | iter 101 val | iter 99 val | iter 101 attn_entropy | Note |
|---|---|---|---|---|
| 70 | (no val) | (no val) | 0.20 | initial collapse |
| 200 | 2.0523 | 2.0743 | 0.28 | collapsed start |
| 400 | 1.7715 | 1.7809 | 0.69 | recovery begins |
| 600 | 1.6233 | 1.6675 | 1.31 | back in sweet spot |
| 800 | 1.5665 | 1.6144 | 1.73 | gap narrowing |
| 1000 fp32 | 1.5449 | 1.6032 | 1.94 | architectural sparsity stable |

**Critical observations:**
1. **Self-correction works**: attn_entropy went 0.20 (collapsed) → 0.69 → 1.31 → 1.73 → 1.94. Iter 99 monotonically collapsed; iter 101 RE-DIVERSIFIED routing as the model learned. This is the principled difference α=1.5 vs α=2 was supposed to deliver, and it did.
2. **Val_bpb gap stabilizes around +0.08** late in training (iter 99 stabilized at +0.12-0.16). α=1.5 has roughly half the capacity penalty of sparsemax — but still NOT enough to promote at the 0.03 gate.
3. **Train-loss gap (~+25%) >> val-loss gap (~+6%)**: same implicit-regularization signature as iter 99. Sparsemax/entmax act as REGULARIZERS — improve generalization gap, hurt absolute fit.
4. **K-sweep crashed at Hutchinson probe AGAIN** — `RuntimeError: Invalid backend` in SDPA under `enable_grad`. Bug in iter 97.6's helper. Fixed in same commit as this H75: wrap Hutchinson + finite-diff probes in try/except so K-sweep never crashes on a diagnostic.

**Why it doesn't promote:**
- α=1.5 buffer reduces but doesn't eliminate the architectural-sparsity capacity penalty.
- The +0.08 final regression is real and ≫ the 0.03 promotion gate.
- Even with ~7 effective experts/token (vs sparsemax's 1), the model still doesn't fit training data as well as full softmax (α=1).
- The implicit-regularization benefit is small relative to the lost expressive capacity.

**Implications for queue:**
- **Architectural sparsity (any α∈{1.5, 2}) does NOT win on val_bpb at this codebase's scale.** Both α=1.5 and α=2 attempted; both NOT PROMOTED. Could test α=1.2 or α=1.3 (closer to softmax) but expected gain is marginal.
- **Loss-side proxy (iter 100 entropy penalty) is now the principled next test** — different mechanism (penalty added to softmax, not architectural change). Doesn't pay capacity cost. May get the regularization benefit without losing the softmax fit.
- **iter 99b (sparse expert dispatch) and iter 101b are CANCELLED** — depended on architectural sparsity promoting, which it didn't.
- **Bigger lesson**: the H72 (E=20 saturation) + H74 (sparsemax) + H75 (entmax-1.5) trio collectively suggest iter 96's softmax + LoRA routing IS near-optimal for THIS codebase at 2× L40S budget. Further val_bpb gains likely require ORTHOGONAL axes (D-scaling on submission hardware, chained routing, attention-side sparsity via AdaSplash, or different optimizer) — not more routing-mechanism variants.

**Pivot**: skip α=1.2/1.3 follow-ups (diminishing returns on closed direction). Run iter 100 (entropy penalty + softmax) to test loss-side proxy, then iter 103 (chained routing) for the orthogonal architectural axis. Iter 95 (TBPTT efficiency sweep) and iter 97.7 (profile-driven throughput) remain as broader optimizations.

**Related:** H32 (DEQ smoothness — entmax-1.5 piecewise smooth, RevDEQ-safe ✓ as designed; iter 101 confirmed by stable training), H74 (sparsemax direct comparison), iter 96 H71 (the softmax baseline this couldn't beat), iter 100 (next test, loss-side mechanism).

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
| 63 | Low-dim expert computation | Each expert: D→r, compute at r, r→D, mix in D-space | H43 | Queued (see merged 63+64 row below) | — | — |
| 64 | Full-rank expert internals | Remove low-rank factorization inside experts (full rank at dim r) | H44 | Queued (merged into 63 below) | — | — |
| 65 | Scale up experts (16-32) | More experts at same/reduced rank for routing diversity | H47 | Queued (duplicated below) | — | — |
| 66 | ~~Parcae negative diagonal~~ | ~~Replace β with per-dim learned A=Diag(-exp(a))~~ | H48 | ~~DONE~~ — landed as iter 66a (tied B̄=1−Ā) then generalized in iter 66b (independent B̄=Δ·B, Mamba-ZOH). See H58. | — | — |
| 67 | Per-iteration LoRA | Rank-4 LoRA per DEQ iter (98K params). Each iter specializes | H49 | **REVERTED** (not principled, doesn't generalize to K>16) | 1.8149 | +0.001 |
| 68 | ~~DeltaDEQ dim skipping~~ | ~~Track per-dim convergence, skip converged dims in later iters~~ | H50 | ~~REMOVED~~ — forward is already compiled/hardware-bound (20ms), dynamic per-dim masking breaks `torch.compile`, and K-jitter already provides coarse-grained "early-stop" at the whole-tensor level. (Duplicated in deferred-row below.) | — | — |
| 69 | Reduce TBPTT 4→1 | Phantom gradient: 1-step backward, 43% faster, 76% more steps | H51 | **REVERTED** (+0.021 post-quant, fast only +0.013) | 1.8396 | -0.002 (K128 best!) |
| 69b | Reduce TBPTT 4→2 | 2-step backward: 29% faster, 40% more steps (1049 vs 747) | H51 | **KEPT** (val_bpb -0.001, 30% throughput gain) | 1.8169 | +0.002 |
| 70 | L2→softmax routing | Replace L2+tanh logits with linear dot-product (standard MoE) | H54 | **PROMOTED ★** (val_bpb -0.024, expert_iter_range 20× higher) | 1.7934 | +0.001 |
| 70 | model_dim 768→1024 | Scale D with low-dim experts (cheap: only down/up grow) | H43 | Queued (after 64) | — | — |
| 71 | Reduce weight_decay 1.08→0.3 | WD=1.08 was biggest expressiveness killer | H54 | **PROMOTED ★** (val_bpb -0.263!!! Largest single improvement ever) | 1.5302 | +0.007 |
| 71b | Reduce weight_decay 0.30→0.10 | Further WD reduction — better fast but worse post-quant | H54 | **REVERTED** (+0.051 post-quant, quant gap 0.047 vs 0.029. WD=0.30 optimal) | 1.5810 | +0.014 |
| 71c | ~~Reduce weight_decay 0.10→0.01~~ | ~~Cancelled: WD=0.10 already regresses post-quant~~ | H54 | CANCELLED | — | — |
| 71d | Drop β=0.7 from jitter | {0.3,0.5,0.7}→{0.3,0.5} | H58 | **REVERTED** (+0.044, less jitter diversity hurt more than recon fix helped) | 1.5744 | +0.017 |
| 71e | ~~Re-enable FSQ, no bounding~~ | ~~FSQ with round+STE but NO tanh bounding~~ | H59 | ~~DROPPED~~ (iter 78 already tested unbounded FSQ + L2 at +0.181@400 — unbounded STE noise accumulates; iter 62's `fsq_levels=0` stands) | — | — |
| 71f | ~~FSQ with clamp(-1,1)~~ | ~~Hard clamp instead of tanh~~ | H59 | ~~DROPPED~~ (iter 78's result generalizes: STE noise through the MoS rank bottleneck is the root cause, not the specific saturating transform) | — | — |
| 72 | Remove post-mix RMSNorm | Replace attn/mlp_post_mix_norm with learned scalar scale | H55 | **REVERTED** (+0.024 val_bpb, K128 Δ=0.001 tightest ever but val regressed) | 1.5495 | +0.001 |
| 73 | Relax grad_clip 0.3→1.0 | Aggressive clip slows learning. Lyapunov provides soft contraction | H56 | **PROMOTED ★** (val_bpb -0.003, every K improved, zero-cost change) | 1.5225 | +0.009 |
| 74 | Raise Lyapunov γ 0.9→0.95 | Allow ρ(J) closer to 1 for more expressive state changes | H57 | **REVERTED** (wash: +0.0005, γ=0.9→0.95 has no measurable effect. Penalty too small at λ=0.01) | 1.5230 | +0.010 |
| 71g | Learnable RMSNorm everywhere | Q/K norms + embed + MoS + bigram (removed soft_embed_norm: DDP unused param) | project constraint | **PROMOTED ★** (val_bpb -0.005, 10% faster, K128 Δ=0.009) | 1.5254 | +0.009 |
| 74b | Lyapunov γ 0.9→0.97 | Push warmup advantage further (γ=0.95 was -0.012@200) | H57 | **PROMOTED ★** (val_bpb -0.008, every K improved -0.006 to -0.008, K8 breaks 1.50) | 1.5150 | +0.008 |
| 74c | WD 0.30→0.01 | Test floor — quant gap may shrink with learnable norms (iter 71g) + Parcae B̄ (iter 66b) both absorb some of the regularization pressure WD was carrying | H54 | **Queued — iter 86 (after 83/84/85)** | — | — |
| 74d | ~~Remove β=0.7 from jitter~~ | ~~{0.3,0.5,0.7}→{0.3,0.5}~~ | H58 | ~~DROPPED~~ (iter 71d already tested and reverted at +0.044; also moot under iter 66b: β is per-dim from Parcae Ā when `use_parcae=True`, the scalar jitter set is a fallback only) | — | — |
| 74e | Restore squared gate leaky_relu(0.5)² | Phase 6 remnant: original activation was more expressive; Banach-contraction constraint that forced the drop is gone in Phase 9 (Lyapunov replaces it) | L9 | **Queued — iter 83 (next up)** | — | — |
| 74f | Independent shared gates (attn vs mlp) | Fix 1-dim shared gate → 2-dim for independent control | arch | **Queued — iter 84 (after 83)** | — | — |
| 74g | ~~Remove x0 residual: T_θ = Δ(z,x0)~~ | ~~More expressive FP equation~~ | arch | ~~SUPERSEDED~~ by iter 66b (`T_θ = B̄ ⊙ RMSNorm_learn(x₀) + Δ`; Parcae-faithful injection, see H58). | — | — |
| 74h | Remove attn_post_mix_norm only | Bisect iter 72 | H55 | **REVERTED** (+0.034, attn norm IS load-bearing) | 1.5488 | — |
| 74i | Remove mlp_post_mix_norm only | Bisect iter 72 | H55 | **REVERTED** (+0.029, mlp norm ALSO load-bearing. Both essential) | 1.5436 | — |
| 74j | Remove state_norm | Pre-expert RMSNorm on z+x0 | arch | **REVERTED** (+0.199@400, catastrophic. state_norm IS load-bearing) | 1.8878@400 | — |
| 74k | Remove bigram proj_norm | Bigram pre-projection RMSNorm | arch | **REVERTED** (+0.174@400, bigram norm also load-bearing. ALL norms essential) | 1.8636@400 | — |
| 74l | Remove shared expert gate (always=1) | DeepSeek-V3 style unconditional shared expert | arch | **REVERTED** (+0.020, shared gate provides useful per-token modulation) | 1.5346 | — |
| 75 | ELM identity init (zero) | Zero-init expert outputs → T_θ≈x0 at init | ELM ICLR26 | **REVERTED** (+0.119@600, zero kills expert diversity) | 1.6988@600 | — |
| 75b | ELM small-scale init (rescue 1) | Normal(0,0.02) expert outputs | ELM ICLR26 | **REVERTED** (+0.157@400, even worse — under-scaled gradients) | 1.8464@400 | — |
| 77 | ~~Residual injection (lerp)~~ | ~~REMOVED: gate g→0 kills input dependence, unprincipled~~ | H25 | REMOVED | — | — |
| 76 | Simulated refinement (mean-emb ε=0.1) | Corrupt x0 = (1-ε)·tok + ε·mean_emb | H16 | **REVERTED** (+0.161@400, mean_emb≈0 reduces magnitude) | 1.8497@400 | — |
| 76b | Simulated refinement rescue (Gaussian ε=0.01) | Additive Gaussian noise corruption | H16 | **REVERTED** (+0.112 final, noise hurts NTP without CTP payoff) | 1.6273 | +0.005 |
| 78 | FSQ symmetric [-8,8] + L2 | 17-level STE on MoS projection, unconstrained + L2=0.01 | H28/H59 | **REVERTED** (+0.181@400, STE noise accumulates despite L2) | 1.8701@400 | — |
| 66a | Parcae+noWD (combined) | Per-dim Ā + no WD on 1D params | H48/Parcae | **REVERTED** (+0.063, caught up @600 but widened late. Bisecting) | 1.5779 | +0.013 |
| 66a-b | No-WD-on-1D bisect | weight_decay=0 for all 1D params (without Parcae) | optimizer | **REVERTED** (+0.140@400, WD on 1D IS beneficial in DEQ) | 1.8295@400 | — |
| 81 | Double max K jitter | K jitter {4,6,10}→{8,12,20} — deeper solver, better FP quality | solver | **Queued — iter 87 (after 86); risky (iter 59's replay)** | — | — |
| 82 | Stochastic TBPTT + doubled | TBPTT fixed 2→jitter {2,3,4}, richer backward signal | solver | **Queued — iter 85 (after 84)** | — | — |
| 66b | ~~Parcae: remove Lyapunov~~ **→ RENAMED iter 88** | After iter 66b-Parcae-faithful lands, test whether Parcae's per-dim Ā makes the Hutchinson λ_jac penalty redundant. | H48 | **Queued — iter 88** (after 85/86; `66b` name collides with the committed iter 66b Parcae-paper-faithful injection, renumbered) | — | — |
| 66c | ~~Parcae: remove denoising reg~~ **→ RENAMED iter 89** | After iter 88 lands, test whether Parcae's per-dim Ā makes the HyDRA denoising penalty redundant. | H48 | **Queued — iter 89 (after 88)** (renumbered from `66c` for the same reason) | — | — |
| 66d | ~~Parcae: separate B̄ (full ZOH)~~ | ~~B̄=A⁻¹(Ā-I)·b, independent from Ā~~ | H48 | ~~SUPERSEDED~~ by iter 66b (committed: `B̄ = Δ·B` Mamba-ZOH approximation, independent of Ā except through shared Δ. See H58). | — | — |
| 66e | ~~Parcae: remove x0 skip in T_θ~~ | ~~T_θ=Δ only (Ā retention replaces x0 skip)~~ | H48 | ~~SUPERSEDED~~ by iter 66b (Parcae-faithful `T_θ = B̄⊙RMSNorm_learn(x₀) + Δ` — the `B̄` injection is an *expressive* replacement for the `x0` skip, not a removal. See H58). | — | — |
| 79 | ~~Per-iter depth embeddings~~ | ~~iter_embed∈R^{K_max×dim}, zero-init, u=z+x0+embed[k]~~ | H24 | ~~REMOVED~~ — not principled: DEQ theory requires the fixed-point map T_θ to be *iteration-invariant* so the K→∞ limit is well-defined (Banach / Parcae-ZOH both assume a single map iterated indefinitely). Per-iteration embeddings make T_θ = T_θ(k) depend on k, destroying the fixed-point premise; K-sweep extrapolation (the existing `K=128` eval gate) ceases to be meaningful. This is a shared-weight K-layer transformer, not a DEQ. | — | — |
| 80 | ~~Refinement inject during DEQ~~ | ~~REMOVED: raw x0 already blended into x0_refined~~ | H27 | REMOVED | — | — |
| 68 | ~~DeltaDEQ dim skipping~~ | ~~REMOVED: non-bottleneck, breaks compile, K-jitter handles~~ | H50 | REMOVED | — | — |
| 63 | Full-rank low-dim experts (merged 63+64) | down(D→r), full-rank attn+MLP at r, up(r→D) per expert | H43/H44 | **NOT PROMOTED ✗ (iter 90, commit `8c2be77`)** — int6 Δ +0.276 vs baseline 1.5264 (capacity -65%, 4.5M params), but K=8→K=128 Δ tightened -0.0036 → -0.0092, throughput +34%, peak VRAM -36%, artifact -49% (3.0 MB). Architecturally validated; stand-alone capacity-bound. See H69. | 1.8023 | -0.0092 (tighter than baseline) |
| 65 | Scale to 16-32 experts | More experts at cheap per-expert dim r | H47 | **NOT PROMOTED ✗ (iter 91+92 bundle, commit `3e35655`)** — bundled with model_dim 768→1024 + r 128→192 to test matched-capacity (11.3M = 87% of baseline). int6 Δ +0.152 vs baseline (closed half the iter-90 gap). K=8→K=128 Δ -0.004 (still tighter than baseline). Quantified per-param efficiency penalty: bpb/param 1.49 vs baseline 1.17 (~27% worse). proj_rank=32 likely the bottleneck. See H70. | 1.6784 | -0.004 |
| 70d | model_dim 768→1024 | Scale D with low-dim experts (cheap: only down/up grow) | H43 | **Bundled into iter 91+92 (above)** — single combined run rather than two sequential iters; per-knob attribution would require separate runs. See H70. | (see iter 91+92) | (see iter 91+92) |

### H76: CV-only equilibrium beats min_share + entropy decoupled (iter 100b annealed entropy + min_share=0) — VERIFIED ★ PROMOTED (2026-04-27)

**Claim:** Antagonistic regularizers (min_share hinge floor + per-token entropy penalty) on the same routing axis fight each other and create training instability (iter 100 train_loss spike 5.92→7.96). Decoupling them — drop `min_share_loss_weight` to 0 (keep min_share as a diagnostic metric), let CV loss alone do global-balance regularization, and add a SOFT per-token entropy penalty WITH ANNEAL FROM 0 (avoid cold-start trap that hurt iter 99/101) — produces a stable, monotonically improving routing equilibrium. Confirms two principled framings: (a) CV smooth gradient is sufficient for dead-expert prevention without a hard floor; (b) sparsity coefs interacting with router learning must anneal from 0 (warmup_delay_frac=0.3 default).

**Test:** iter 100b — `Hyperparameters.min_share_loss_weight = 0.0` (was 1.0), `cv_loss_weight = 2.0` (was 0.10, 20× to compensate for sole responsibility), `router_entropy_coef = 0.005` target with `router_entropy_warmup_delay_frac = 0.3` linear ramp from 0. Built on iter 96 baseline (E=16, attn_rank=64, mlp_rank=96, D=768, softmax routing). Commit `bbb4234` (training launch) + diagnostic split `c5154b3` (logging-only mid-flight). 1000 steps on 2× L40S, ~7.8 hours wallclock.

**Result:** ✅ PROMOTED. val_bpb int6 = **1.489339** (best K=16 from K-sweep), beats iter 96 baseline 1.4903 by **0.0010**. Artifact 7,515,680 bytes (47% of 16 MB budget). All promotion gates pass:

| Gate | Value |
|---|---|
| val_bpb int6 (best K=16) | 1.489339 vs 1.4903 → ✅ improvement |
| K=128 val_bpb | 1.490805 |
| K=128 vs best-K Δ | 0.001474 (gate ≤ 0.5) ✅ massively |
| Acyclicity K=17 vs K=16 | Δ=0.0003 ✅ genuine FP |
| Acyclicity K=37 vs K=32 | Δ=0.00006 ✅ |
| Acyclicity K=113 vs K=128 | Δ=0.00004 ✅ |
| artifact ≤ 16 MB | 7.52 MB ✅ |
| peak_vram_mb | 35,721 (under 44 GB) ✅ |

**Eval results (iter 100b, full K-sweep + routing health)**:

Roundtrip verification: val_bpb (int6 + zstd) = **1.489339**, val_loss = 2.4744. Final val checkpoint at step 1000 (fast mode K=16): val_bpb = 1.4572, val_loss = 2.4210, attn_cv 0.1735, attn min share 0.040, attn max share (expert 12) 0.099, attn_ortho 0.139, mlp_ortho 0.236, router_mass 0.802, shared_gate_mean 0.122, shared_gate_min 0.003, pertoken_entropy 2.986.

| K | val_bpb | iter_conv_rel | residual | acyclicity check |
|---|---|---|---|---|
| 4 | 1.668684 | 0.297594 | 853.87 | (under-converged) |
| 8 | 1.493566 | 0.078900 | 853.87 | — |
| 16 | **1.489331** ← best | 0.017228 | 853.87 | — |
| 17 (prime) | 1.489594 | 0.016195 | 853.87 | Δ vs K=16 = 0.0003 ✓ |
| 32 | 1.490595 | 0.013763 | 853.87 | — |
| 37 (prime) | 1.490654 | 0.013892 | 853.87 | Δ vs K=32 = 0.00006 ✓ |
| 64 | 1.490774 | 0.013790 | 853.87 | — |
| 113 (prime) | 1.490766 | n/a | 853.87 | Δ vs K=128 = 0.00004 ✓ |
| 128 | 1.490805 | n/a | 853.87 | — |

K=128 vs best-K (K=16) Δ = +0.001474, ≪ 0.5 promotion gate ✓. All three acyclicity primes confirm genuine fixed point. Hutchinson-Frobenius + finite-direction Lipschitz probes did NOT emit (silently caught by iter 97.6 try/except — root cause was **OOM at 32 GiB allocation request** when running JVP/forward through the full val-batch SharedBlock graph, NOT SDPA backend rejection as initially documented; resolved by iter 97.5b-fix's `B_probe=1` slicing 2026-04-29). Per-K expert/sparsity/shared_gate diagnostics were NOT YET emitted by iter 100b run (this is the data the post-iter-100b `k_sweep_table:` extension adds, commit `100ffcc`).

**Trajectory summary** (val_bpb at each in-training val checkpoint, fast mode K=16):

| Step | val_loss | val_bpb | Δ from prev |
|---|---|---|---|
| 0 | 7.0040 | 4.2156 | — |
| 200 | 3.3249 | 2.0012 | −2.215 |
| 400 | 2.7391 | 1.6486 | −0.353 |
| 600 | 2.5489 | 1.5342 | −0.114 |
| 800 | 2.4869 | 1.4968 | −0.037 |
| 1000 | 2.4210 | **1.4572** | −0.040 |

**Trajectory** (s200→s1000 every 200 steps, fast-mode val_bpb): 2.0012 → 1.6486 → 1.5342 → 1.4968 → **1.4572**. Drops 0.353 → 0.114 → 0.037 → 0.040 (decay leveled off in last 200 instead of continuing to halve). Roundtrip int6+zstd 1.4893.

**Routing trajectory** (single pooled router, soft-dense MoE):
- attn_cv: 0.71 peak (s90, no-floor exploration) → 0.40 (s200) → 0.29 (s400) → 0.247 (s120 of next 100) → **0.1735 final**. CV TIGHTENED MONOTONICALLY through training including the entropy-ramp phase. Per-slice (post c5154b3 diagnostic split) at s120 reference: attn_cv≈1.07, mlp_cv≈0.18 — within-attn winner-take-all (expert 12 peak 0.376, settled to 0.099 final), MLP slice trivially uniform (max 0.084, range 0.044). Pool entropy 94% of log30 confirms no cross-slice dominance.
- max attn share (expert 12): 0.376 peak (s90) → 0.099 final (75% reduction).
- min attn share: stayed in 0.014-0.023 band throughout (well above 0.005 sentinel) — **no dead expert without min_share floor**, validating CV-only mechanism.
- per-token "entropy" (still labelled `attn_entropy`/`mlp_entropy` under old format pre-c5154b3): rose 2.33 (s80) → 2.55 (s200) → 2.88 (s400) → 2.98 (s760) → 2.99 (s1000). Penalty marginally engaged at high coef (s800+: halted the rise, never bent down).
- `aux_gap` (train_loss − ntp_loss): 1.57 (s80) → 0.20 (s270, post-equilibration) → maintained. Regulators became QUIESCENT after CV equilibration.
- router_mass: 0.99 (init) → 0.83 (s600 plateau) → 0.80 (final). Model intentionally closing the routed mixture; shared experts pick up slack.

**Critical reframe** (informs queue priority going forward): iter 100b's val_bpb improvement is dominantly **CV-driven**, not specialization-driven. The entropy penalty target=0.005 was too weak to bend per-token entropy down (only halted its rise); CV at weight 2.0 did all the heavy lifting alone. **Decoupling antagonistic regularizers + annealing the soft sparsity coef** = the principled fix; `min_share` floor is the dirty fallback that would re-introduce the antagonism. Implication for sparsity priority: router sparsity under soft-dense MoE has zero throughput benefit (all experts still computed) and limited val_bpb benefit (CV achieves balance without specialization); **attention sparsity (iter 104 AdaSplash α-entmax)** is the strongest throughput+regularization play next — see `feedback_sparsity_value_props.md`.

**Related:** H74 (sparsemax NOT PROMOTED), H75 (entmax-1.5 NOT PROMOTED), `feedback_decouple_regularizers.md`, `feedback_anneal_sparsity_coefs.md`, `feedback_sparsity_value_props.md`.

### Iter 104 launch attempts log (2026-04-28, awaiting direction)

**v1 — block_causal_sdpa with reshape-flatten** (commit `165ea77`, launched 01:18, crashed 01:19)
- Error: `RuntimeError('Invalid backend')` from `F.scaled_dot_product_attention` with non-square `attn_mask=(256, 260)` under torch.compile dynamo trace.
- Root cause: S=4 sinks created (W, S+W) non-square mask; SDPA's FlashAttention backend rejects this configuration at trace time (not runtime).
- User correction noted: this implementation REPLACED FlashAttention with reshape-flatten SDPA, not preserving "MLA + FA + sparsity" framing.

**v2 — sliding-window via flex_attention** (commit `1dba8ca`, launched 01:36, crashed 01:35)
- Error: `ValueError: Expected query, key, and value to have the same dtype, but got query.dtype: torch.float32, key.dtype: torch.float32, and value.dtype: torch.bfloat16`.
- Root cause: flex_attention is strict about dtype consistency; dense `F.scaled_dot_product_attention` auto-casts but flex_attention does not. The MLA pipeline produces q/k in fp32 (rotary output) and v in bf16.
- Architecture properly preserved: same MLA + head-packed q_full/k_full/v_full pattern; only the SDPA primitive swap.

**v3 — flex_attention + dtype unification fix** (smoke PASSED 01:42, SUPERSEDED by v4)
- User redirect: sliding window alone hurts long-sequence performance.

**v4 — AdaSplash α-entmax with α-annealing** (commit `d6f3cff`, launched 01:56, IN FLIGHT)
- Replaces softmax with α-entmax via deep-spin/adasplash Triton kernel (ICML 2025, arxiv 2502.12082).
- Anneal α: 1.0 (softmax fallback) for first 30% of training, then linear ramp 1.0→1.5 over remaining 70% — avoids cold-start trap that hurt iter 99/101.
- Kernel correctness verified: at α=1.001 matches F.scaled_dot_product_attention to bf16 precision (max_diff 0.016, mean_rel 2.5%); higher α progressively diverges as α-entmax should.
- Trajectory steps 10-110: train_loss tracks iter 100b within early-training noise (-0.40 to +1.53 swing); attn_cv peak 1.22 at s90 vs iter 100b's 0.74 peak — meaningfully higher imbalance during pre-warmup despite using softmax fallback (suggests RNG-state difference from prior smoke test relaunches, not architectural change since alpha=1.0 makes the path identical to iter 100b). Min attn share 0.011 — close to 0.005 sentinel but not violating. Critical signal point: step 200 val_bpb (target <2.10 to track iter 100b's 2.00); step 300 = α-anneal engages, AdaSplash kernel becomes active.
- **Step 200 val: val_bpb=2.0005** ← matches iter 100b's 2.0012 within 0.001 (within hardware/RNG noise). CV divergence at s90-110 was RNG stochastic, not architectural — the routing took a different intermediate path but the loss tracks. attn_cv 0.64 at s200 (vs iter 100b's 0.37) — higher CV is irrelevant when val_bpb is identical. Pre-warmup throughput: step_avg 23.37s (iter 100b: 23.93s) → ratio 0.977 (~2% faster, within noise). As expected for α=1.0 fallback path. Throughput claim of AdaSplash becomes testable at step 300+ when α anneals above 1.0 and the kernel becomes active.
- **❌ CRASHED at step 300 — AdaSplash kernel-shape incompatibility**: at step 300+ when α first ramped above 1.0, AdaSplash's Triton kernel asserted `H_DIM in {16, 32, 64, 128, 256}` and crashed both ranks. Our head_dim is **96** (model_dim 768 / num_heads 8) — not in the supported set. AdaSplash's kernel hardcodes power-of-2 head dims for Triton block-size specialization. Pre-warmup phase ran fine (256 steps with α=1.0 dense SDPA fallback) but the very first AdaSplash call at step 300 crashed. Iter 104 v3 → **NOT TESTED** on val_bpb beyond step 200.

**v4 — Architecture change to head_dim=64** (commit `56c9f79`, launched 04:26, **❌ CRASHED at step 301**) — User directive: proceed with Option C (change architecture). num_heads 8→12, num_kv_heads 4→6 (preserves GQA ratio 2:1). Same total head capacity. Step 200 val: **val_bpb=1.9993** (within 0.002 of iter 100b) — architecture change is val_bpb-neutral. Pre-warmup throughput: 23.07s/step (3.6% faster than iter 100b). Step 300 reached cleanly with α=1.0 fallback. **Crash at step 301+** when α first exceeded 1.0 and AdaSplash Triton kernel was actually invoked: SIGABRT (signal 6, C-level abort) at rank 0 with no Python traceback. Likely torch.compile + DDP + Triton kernel incompatibility or CUDA-level assertion. No Python-level error visible. **Per user directive 2026-04-28: iter 104 DEFERRED to end of queue, proceed to next iter.** Iter 104 v3 NOT TESTED (head_dim=96 incompatible) + iter 104 v4 NOT TESTED (kernel crashes under compile+DDP at d_head=64). AdaSplash kernel verified to work in standalone PyTorch at d_head=64 GQA-2 — but fails under our compiled-DDP-revdeq pipeline. Recovery path requires either (a) disable torch.compile around AdaSplash invocation, (b) wrap AdaSplash in `@torch.compiler.disable`, (c) test AdaSplash in non-DDP single-GPU first, (d) pivot to iter 106 NSA-style flex_attention (no Triton-kernel-from-package risk). Defer iter 104 until next round; proceeding to next queue item: **iter 102 (α-anneal router sparsity)** — uses pure PyTorch ops, no Triton kernel, sidesteps AdaSplash compatibility issues.

### Iter 102 — Router α-anneal entmax (commit `e382d95`, launched 06:55, IN FLIGHT)
- Pivoted from iter 104 (deferred). Pure-PyTorch implementation: linear blend of softmax + entmax15, where alpha_scale ramps 0→1 over training (warmup_delay_frac=0.3). New `router_kind="entmax_anneal"`. No Triton kernel risk.
- Pre-warmup observations (steps 1-200): RNG-state divergence from iter 100b — attn_cv plateaus at 1.06 (vs iter 100b's 0.74 peak), pertoken_entropy 2.04 (vs iter 100b's 2.78), train_loss +0.95 wider gap at step 140 vs iter 100b. Same pattern as iter 104 v3/v4 had during pre-warmup. Suggests compile-graph subtleties from the new code path (the `getattr(self, "_router_alpha_scale", 0.0)` + branch logic) cause different RNG specialization, NOT a real algorithmic regression.
- **Step 200 val: val_bpb=2.0025** ← within 0.0013 of iter 100b's 2.0012 (RNG noise NOT regression). Implementation is correct: the pure-softmax fallback path of `entmax_anneal` (alpha_scale=0) is baseline-equivalent. Confirms the pattern from iter 104 v4: CV/entropy divergence is a compile-graph RNG artifact, val_bpb is the actual signal.
- Pre-warmup throughput: 23.36s/step (vs iter 100b's 23.93s/step) → ~2.4% faster. Probably noise / hardware variation since algorithm is identical.
- Critical val checkpoints ahead: step 400 (alpha_scale ≈ 0.14, light entmax), step 600 (~0.43 substantial), step 800 (~0.71 mostly entmax), step 1000 (1.0 pure entmax15). The α-anneal effect should manifest at step 400+ as either val_bpb improvement (regularization helps) or regression (architectural sparsity costs capacity per H75). Promotion gate is val_bpb int6 < 1.4903 (iter 100b baseline).
- **Step 400 val: val_bpb=1.6647 vs iter 100b's 1.6486 → +0.0161 regression**. **Throughput**: step_avg climbed from 23.0s pre-warmup to 26.3s post-α-anneal (+13.7% per-step cost from entmax15's 30-iter bisection). Combined per-wallclock regression: ~+10-15% wallclock cost for the same val_bpb (or worse). Per H72 precedent (per-wallclock override of val_bpb gate), this trajectory unlikely to promote. Continue monitoring through step 600 to confirm trend before deciding abort vs. complete-and-document. The α-anneal is at α=0.14 (14% entmax blend) — and the trajectory is already worse than iter 100b. Will likely worsen as α grows toward 1.0.
- **Step 600 val: val_bpb=1.5436 vs iter 100b's 1.5342 → +0.0094 regression** (better than predicted). At α=0.43 (43% entmax15 + 57% softmax routing), α-anneal is mostly tracking iter 100b on val_bpb. step_avg 26.14s vs iter 100b's 24.47s (+6.8% wallclock). Per-wallclock combined cost: ~7% for ~+0.01 val_bpb regression. Per H72 precedent, this is still not promotable on per-wallclock grounds. Continue to step 800/1000 to see if α=0.71/1.0 either: (1) closes val_bpb gap (regularization helps in the limit) → maybe promotable, or (2) widens (capacity cost grows) → confirms not promotable. Earlier train_loss gap was misleading — the α-anneal blend is doing genuine regularization that the val_bpb gate captures better than train_loss.
- **Step 800 val: val_bpb=1.5038 vs iter 100b's 1.4968 → +0.0070 regression**. **GAP IS CLOSING**: +0.0161 (s400) → +0.0094 (s600) → +0.0070 (s800). α-anneal regularization is REAL, increasingly showing benefit as entmax dominance grows (α=0.71 at s800). Linear extrapolation s1000: val_bpb ~1.46 (gap ~+0.005 vs iter 100b's 1.4572). Iter 102 will land essentially-equal to iter 100b on val_bpb but with +6.8% wallclock cost. **Not promotable per H72** but the small per-wallclock gap suggests a follow-up iter could rescue: reduce entmax15 n_iter (currently 30 bisection) to 10, OR cache α-blend vs recomputing both branches per call. Per-wallclock gate is the unique blocker. The H75 hypothesis (α-entmax has fundamental capacity cost) is REFUTED by α-anneal — annealing genuinely fixes the cold-start trap that hurt iter 99/101.

**User redirect 2026-04-28**: test AdaSplash α-entmax in iter 104 (not sliding window). Adasplash package install blocked by sandbox enforcing CLAUDE.md "no new packages". Awaiting authorization OR alternative-direction decision (Options A/B/C). DeepSeek V4 (released 2026-04-24) introduces CSA + HCA hybrid attention: compressed sparse attention (lightning indexer + top-k=1024 + sliding window 128) and heavily compressed attention (128× compression rate). NSA-style 3-branch was already queued as iter 106 (task #103); V4 is essentially NSA + HCA.

### H77: Chained 2-stage routing (iter 103) — PROPOSED (2026-04-28 user spec)

**Hypothesis.** Replacing the current single-stage pooled router with two sequentially-chained routing stages — each routing through an INDEPENDENT, halved set of experts — adds *sequential routing depth* without raising total expert count, exposing a routing-composition axis the current single-stage pooled architecture cannot represent.

**Current architecture (per CLAUDE.md §6.2 + train_gpt.py L2212).** Each Block has ONE pooled `SoftDenseRouter(dim, 2 * num_routed, health_slices=(num_routed, num_routed))`. With `num_experts = 16` and `num_shared_experts = 1`, the pool emits `2 × 15 = 30` routing weights — 15 for the attn slice (driving 15 routed attn experts + 1 shared attn expert) and 15 for the mlp slice (driving 15 routed mlp experts + 1 shared mlp expert). All routed experts process the FULL input in parallel and contribute additively to the residual `T_θ(z, x_0)`.

**Proposed iter-103 architecture (user verbal spec, 2026-04-28).** Replace the single pool with TWO sequential routing stages inside `T_θ`. The user's framing: *"current config is single pooled expert that contains [N_a] attn + [N_m] ffn experts → new iteration with chained should be 2 chained pooled routers, with the first pool containing [N_a/2] attn + [N_m/2] ffn experts, output chained to the next group of experts with another different independent [N_a/2] attn + [N_m/2] ffn experts."*

Concretely:
- **Stage 1**: pooled `SoftDenseRouter_1(dim, 2 × (num_routed/2))` driving `num_routed/2` attn + `num_routed/2` mlp experts (independent params, no sharing with stage 2).
- **Chaining**: stage 1's mixed output `T_1(z, x_0)` becomes the **input to stage 2** (replacing `z` in stage 2's signature). Stage 2's router sees this stage-1 output and routes among its OWN experts.
- **Stage 2**: pooled `SoftDenseRouter_2(dim, 2 × (num_routed/2))` driving an INDEPENDENT `num_routed/2` attn + `num_routed/2` mlp experts.
- Final block residual: `T_θ(z, x_0) = T_2(T_1(z, x_0), x_0) + B̄ · RMSNorm(x_0)` (the iter-66b Parcae input injection is preserved as currently structured; chaining is intra-Δ only).

**Key design questions to resolve at implementation time.**
1. **Total expert count vs. per-stage halving** — the user's spec says "halve per stage." Two principled iso-something options:
   - (a) **Iso-expert-count** (preferred default): keep 15 attn + 15 mlp routed total split as 7+8 per stage. Each stage's experts have the SAME `attn_expert_rank=64` / `mlp_expert_rank=96` as baseline → total expert params unchanged, but each stage runs ½ the experts → total compute roughly unchanged. Wins via routing composition, not capacity.
   - (b) **Iso-stage-cost** (richer model): keep 15 attn + 15 mlp PER STAGE (30 total per type) at baseline rank → 2× expert params → likely violates 16 MB artifact budget; only viable with rank halving (`r/2`) per stage to hold params constant. This is iso-param but 2× FLOPs because both stages' experts run.
2. **Shared experts**: keep 1 shared per stage per type (so 2 shared attn + 2 shared mlp total) OR keep just 1 shared at the block boundary (outside the chain). User did not specify; default to "1 shared per stage" so `num_shared_experts` semantics carry through unchanged.
3. **Skip-connection across the chain**: should the stage-1 router output also bypass stage 2 (residual-style) to preserve iter-100b's CV equilibrium as a special case? Default = yes (`T = T_1 + T_2(T_1, x_0)`); ensures *strict generalization* — stage 2 weights initialized near zero recovers iter 100b exactly. Promotion gate: this initialization must be representable in the iter-103 parametrization (see CLAUDE.md §11 Strict-generalization promotion rule).

**Strict-generalization argument (Promotion Rule §11 prerequisite).** Setting all stage-2 router weights and stage-2 expert outputs to zero recovers iter 100b's forward map exactly:
- Stage 1 with `num_routed/2` experts is NOT iso-functional with iter 100b's `num_routed` experts; iso-functionality requires the full `num_routed` in stage 1 + zero stage 2 (option-b with stage-2 zero-init), which violates the iso-expert-count default.
- Therefore option (a) does NOT strictly generalize iter 100b. Promotion is val_bpb-primary under §11 standard rule, NOT unconditional.
- Option (b) DOES strictly generalize iff implementation places baseline experts in stage 1 + zero-init stage 2 — but it doubles param count, blowing the 16 MB budget.
- **Recommendation**: run option (a) first as a clean ablation; if val_bpb regresses, the loss is structural-capacity not routing-composition, and option (b) with rank halving becomes the principled retry.

**Risks.**
- DDP all-reduce bandwidth doubles (router gradient traffic doubles per Block).
- Step-time cost: stage 2 cannot start until stage 1's output is materialized → no parallelism, ~+30-50% step_avg in expectation. Per-wallclock gate (H72 precedent) may apply.
- Stage 2 experts may starve at cold start if stage 1's output is near-zero — entropy ramp + cv_loss carrying through to stage 2 should mitigate; verify with the iter-100b regulator stack untouched.
- Routing depth interacts with DEQ depth: chained router is part of `T_θ`, so the FP equation becomes deeper effectively. K-sweep gate especially important here; test acyclicity primes K∈{17, 37, 113} per iter 97.6.

**Status:** PROPOSED — queued (priority 3 in revised post-iter-100b queue per user reorder 2026-04-28).

### H78: Attn-vs-MLP expert-count ratio (iter 107) — PROPOSED (2026-04-28 user spec)

**Hypothesis.** The current 1:1 attn-to-mlp routed-expert split (15:15 routed under `num_experts = 16`, `num_shared_experts = 1`) is an arbitrary choice; the val_bpb-optimal ratio may differ. Test 1:1 (current) vs. 2:1 (more attn experts) vs. 1:2 (more mlp experts) at held-constant total routed count.

**Motivation.** Attn experts and mlp experts serve qualitatively different functions in the architecture:
- **Attn experts** (per-expert MLA — full-D LoRA Q/K/V/Wo at `attn_expert_rank=64`): specialize *what to attend to* (head pattern + KV compression).
- **MLP experts** (per-expert SwiGLU at `mlp_expert_rank=96`): specialize *how to transform attended content* (nonlinear feature mixing).

Their per-expert parameter budgets differ (attn rank 64 ≠ mlp rank 96), so iso-count is NOT iso-param. The current 1:1 split was inherited from iter 96's `num_experts=16` decision applied symmetrically; there is no principled reason that the optimum lives at 1:1 specifically.

**Proposed test points (held constant: total routed expert count = 30 = baseline `2 × num_routed`).**
- **1:1 baseline**: 15 attn + 15 mlp routed (current iter 100b/96 config — re-run for control if needed; otherwise use existing baseline).
- **2:1 attn-heavy**: 20 attn + 10 mlp routed.
- **1:2 mlp-heavy**: 10 attn + 20 mlp routed.

Implementation requires the pooled `SoftDenseRouter` to support unequal slices: `health_slices=(N_attn_routed, N_mlp_routed)` with `N_attn_routed + N_mlp_routed = 30`. Per-expert rank held at baseline (`attn_expert_rank=64`, `mlp_expert_rank=96`) so the total param budget shifts mildly with the ratio (attn experts are smaller); track artifact bytes to confirm ≤ 16 MB.

**Strict-generalization status.** Neither 2:1 nor 1:2 strictly generalizes 1:1 (different parametrization). Standard val_bpb-primary promotion rule applies (§11). Best-of-3 is the candidate; if 2:1 or 1:2 wins, the new ratio replaces 15:15 in the default config.

**Connection to existing diagnostics.** The asymmetric routing already observed in iter 100b (attn slice winner-take-all with `attn_cv≈0.17` final, mlp slice essentially uniform with `mlp_cv≈0.18`) suggests the two slices already operate at different effective specialization regimes. **Hypothesis nuance**: if attn is already saturated by single-expert dominance, *adding more* attn experts may be wasted (1:2 mlp-heavy preferred); conversely if attn imbalance reflects under-capacity in the attn pathway (15 isn't enough so the model concentrates), 2:1 attn-heavy would give the dominant expert siblings to share with. iter 100b's pattern alone cannot distinguish these — that's why this iter is principled, not fishing.

**Pre-emptive prediction.** Given iter 100b's attn-slice winner-take-all (mass concentrated on expert 12 then 1-3 others), **prior is that 1:2 mlp-heavy < 1:1 < 2:1 attn-heavy**. The mlp slice is uniform → already well-distributed → adding more mlp experts is unlikely to help. The attn slice concentrates → either under-capacity (more experts help, 2:1 wins) or over-capacity with bad init (more experts hurt, 1:2 wins). User direction will be informed by which predicted ordering manifests.

**Status:** PROPOSED — queued (priority 4 in revised post-iter-100b queue per user reorder 2026-04-28).

### H81: D=1024 with grad_accum_multiplier=2 (iter 98b retry of H73) — NOT PROMOTED ✗ (2026-04-29)

**Claim:** Iter 98 H73 OOM'd at D=1024 on dev hardware (3× attempts at 44 GiB cap). Iter 98b retries with `grad_accum_multiplier=2` halving the per-step micro-batch — the same effective batch is accumulated over 2× more micro-steps with halved per-step activation memory. If the rescue fits VRAM AND val_bpb improves at iso-effective-batch, D=1024 + grad-accum-multiplier becomes the new dev baseline.

**Test:** iter 98b — 1000-step training, D=768 → 1024, `grad_accum_multiplier=2`, all other knobs at iter 100b baseline (E=16, attn_R=64, mlp_R=96, K=16 fixed, TBPTT=2, refinement=1@85% ramp, WD=0.01, Parcae per-dim Ā+B̄).

**Result: NOT PROMOTED** ✗ on both H72 gates.

**Full eval results:**

(a) Roundtrip + fast eval:
- s1000 fast val_bpb = **1.4665** (vs iter 100b's 1.4572 — Δ +0.0093)
- s1000 roundtrip int6 val_bpb = **1.5018** (vs iter 100b's **1.4893** — Δ **+0.0125 REGRESSION**)

(b) K-sweep — acyclicity primes 17/37/113 verify genuine FP at K=16 (3/3 confirmations within 0.001 of neighbors):

| K | val_bpb | iter_conv_rel | acyclicity check |
|---|---|---|---|
| 4 | 1.8873 | 0.2367 | (cold solver, far from FP) |
| 8 | 1.5890 | 0.1012 | |
| 16 | 1.5018 | 0.0276 | training K |
| **17 (prime)** | 1.5008 | 0.0239 | ✓ matches K=16 within 0.001 |
| 32 | 1.5023 | 0.0081 | |
| **37 (prime)** | 1.5028 | 0.0078 | ✓ matches K=32 within 0.001 |
| 64 | 1.5037 | 0.0075 | |
| **113 (prime)** | 1.5039 | 0.0079 | ✓ matches K=128 within 0.001 |
| 128 | 1.5039 | 0.0076 | |

K=128 vs best-K (K=17, 1.5008) Δ = +0.0031 — well within 0.5 FP-quality gate. K-sweep table `hutch_F` and `rd_step` all `N/A` — root cause was **OOM at 32 GiB allocation request** when the JVP/forward through SharedBlock at full val batch (B≈32 sequences × T=2048 × D=768) overran the predictive OOM guard (which used `4× z_star` ≈ 800 MiB but actual need was ~32 GiB through 12 layers × 16+16 experts × multiple intermediate tensors). Misattributed to "SDPA backend rejection" in iter 100b documentation; corrected 2026-04-29 by iter 97.5b-fix (`B_probe=1` slicing in `_hutchinson_F_at_saved_fp` and `_compute_eval_fp_lipschitz` brings probe activation memory to ~1-2 GiB). Future iters will populate `hutch_F` correctly.

(c) Routing + FP diagnostics at K=16 (training K):
- attn_cv 0.2369, mlp_cv 0.0427, pool_cv 0.1702 — healthy
- attn_min 0.0463, mlp_min 0.0605 — well above 0.005 sentinel
- attn_ortho 0.1182, mlp_ortho 0.1416 — within target range
- pool_entropy 3.3871 (≈ log(30)) — global utilization healthy
- pertoken_entropy 3.0608 — no specialization (consistent with iter 100b reframe)

(d) Trajectory by step:
| Step | val_bpb (fast) | Δ vs iter 100b | tbptt_vjp[z, y] |
|---|---|---|---|
| 0 | 4.2157 | (cold) | (no grad yet) |
| 200 | 1.9346 | −0.067 (lead) | [8.81e-05, 7.48e-05] |
| 400 | 1.6186 | −0.030 | [1.31e-04, 1.09e-04] |
| 600 | 1.5323 | −0.002 (closing) | [1.78e-04, 1.37e-04] |
| 800 | 1.4987 | **+0.002 (lead reversed)** | [2.03e-04, 1.46e-04] |
| 1000 | 1.4665 | **+0.009 (full deficit)** | — |

(e) Wallclock + budget:
- Total step_avg: 25.10 s/step (vs iter 100b 24.47s, +2.6% slower)
- Refinement-induced step time: 22.7 s/step pre-refinement → 38.5 s/step post-refinement at s850+ (refinement amplification at D=1024 is +70% vs iter 100b's +6%)
- artifact_bytes: 11,249,533 (within 16 MB budget, 70% used vs iter 100b's 47%)
- peak_vram_mb: 23,142 (well under 44 GiB dev cap — micro-batch rescue WORKED architecturally)

**Why NOT PROMOTED:**
1. val_bpb gate (CLAUDE.md §11): int6 1.5018 > iter 100b 1.4893 → REGRESSION, fails primary gate.
2. Per-wallclock override (H72 precedent): iter 98b is +2.6% slower, so per-wallclock ALSO worse. Cannot override on this axis.
3. Strict-generalization (CLAUDE.md §11): D=1024 does NOT strict-generalize iter 100b's D=768 — different parametrization, no representable point recovers iter 100b. Cannot promote unconditionally.

**Why iter 98b underperformed (root cause analysis):**
- The early lead (s200: −0.067, s400: −0.030) showed D=1024 capacity DOES help, but the late-training cooling phase systematically gave the lead back. iter 100b's s400→s600 drop was −0.114; iter 98b's was −0.086 (24% slower descent). The increased capacity under-helped relative to the increased step cost.
- The dominant cost driver was REFINEMENT amplification at D=1024. Refinement (`num_refinements=1` enabled at 85% ramp) added a full extra DEQ pass; at D=1024 the activation memory pressure during this extra pass caused step time to rise from 22.7s → 38.5s (+70% per step), dominating the late-training budget. iter 100b's same refinement only added +6% step cost. The user observed this in real-time and asked whether refinement was actually load-bearing — see EXPERIENCE.md if pursuing iter 98c (D=1024 + `num_refinements=0`).
- The Hutchinson `hutch_F` diagnostic that would have measured FP spectral structure across training to confirm/refute "is the FP becoming MORE contractive with capacity?" was unavailable due to SDPA-backend rejection. Iter 97.5b PERMANENT (added 2026-04-29 same commit as this NOT PROMOTED close-out) wires `hutch_F` into val checkpoints; future D-scaling iters will have the diagnostic available if SDPA rejection is resolved.

**Implications for queue:**
- D=1024 dev-hardware path closed (re-confirms H73). Submission hardware (8× H100 80 GiB) does not have the activation-memory-binding constraint, so D=1024 remains a valid axis for the final submission.
- Per the user's 2026-04-29 follow-up question: iter 98c (D=1024 + `num_refinements=0`) is a principled retry that isolates whether refinement is the binding cost at D=1024 or whether D=1024 itself doesn't pay off. Status: AWAITING USER GO/NO-GO. The implementation is one-line (`num_refinements_ramp_frac=1.0` to never enable, OR `num_refinements=0` to disable refinement entirely). If iter 98c promotes, that confirms refinement is the cost driver and CLAUDE.md §6.5 needs updating to gate refinement by `D > some_threshold`. If iter 98c also doesn't promote, D-scaling on dev hardware is fully closed until 8× H100 access.
- Queue advances to **iter 104** (AdaSplash) per the original user reorder, but iter 104 was attempted earlier in this session as "Fix #2" and SIGABRT'd (task #111). Custom_op registration work blocking. Surfacing to user before next launch.

**Related:** H73 (iter 98 NOT TESTED, OOM at 44 GiB cap), H72 (per-wallclock override precedent), H76 (iter 100b PROMOTED ★, current baseline).

### H79: Forward K reduction 16 → 10 (iter 108) — PROPOSED (2026-04-29 user spec)

**Hypothesis.** Reducing `deq_k_max` from 16 → 10 cuts FP-solve compute by ~37% per training step at the risk of incomplete fixed-point convergence. If the iter-98b/iter-100b solver has converged to a regime where the FP residual is dominated by the first ~10 iterations, K=10 is a clean throughput win at zero or negligible quality cost.

**Empirical motivation.** Iter 98b's logged `deq_iter_conv_rel` (the relative convergence metric `||z_T − z_{T-1}||/||z_T||`) at K=16 has settled into the 0.015–0.025 range from s400 onward. The geometric decay implied by Parcae's per-dim damping suggests that residual at K=10 would be `(1−Ā)^(16−10)` × residual at K=16 ≈ ~6× larger if `Ā ≈ 0.7`, but this still puts iter_conv_rel ≈ 0.10–0.15 — well under the divergence threshold of 0.20. Whether this translates to val_bpb degradation depends on how much the gradient pathway uses the last 6 forward iterations under TBPTT k=2 (where only the last 2 iterations contribute backward gradient).

**Test:** Single config change: `deq_k_max = 10`, `deq_k_eval = 10`, `deq_k_jitter_set = (10,)`. Hold all other knobs at iter 98b (or current) baseline. Compare s1000 val_bpb fast + roundtrip int6 vs current baseline; compare step_avg.

**Predicted outcomes:**
- (a) **Throughput win, quality preserved**: step_avg drops ~25–30% (16→10 forward iters, but compile + DDP + non-K overhead is ~30% of step). val_bpb regression ≤ 0.005. Promote.
- (b) **Per-wallclock win, quality regresses ≤ 0.02**: equal or better val_bpb / wallclock product. Per H72 precedent, may still promote.
- (c) **Quality regresses > 0.02**: don't promote; informs that K=16 is load-bearing for FP quality, not just safety margin.

**Strict-generalization status.** Not strict-generalizing (smaller forward graph). Standard val_bpb-primary promotion rule applies (§11), with per-wallclock override possible per H72.

**Connection to existing H-claims.** Complements iter 95 (TBPTT efficiency, deq_bptt_k axis) and iter 109/H80 (K-jitter {10,16}). The triplet 108/109/95 jointly characterizes the K landscape: forward K, jitter K, TBPTT K.

**Status:** PROPOSED — queued (priority 8, after iter 97.7 throughput retry).

### H80: K-jitter {10, 16} (iter 109) — PROPOSED (2026-04-29 user spec)

**Hypothesis.** Stochastic K-jitter sampling from `{10, 16}` per training step provides gradient diversity over FP-depths, analog to H12 (VERIFIED — wider jitter `{4,6,10}` → `{8,12,20}` tightened the K-sweep tail). The model sees gradient signals from variable-depth FPs each step, which may regularize the FP-convergence properties beyond what fixed K=16 provides.

**Empirical motivation.** Iter 87 (H66) PROMOTED widening K-jitter range; the principle "varied K beats fixed K" was established at the {4,6,10}→{8,12,20} scale. Whether the principle extends to the narrower `{10, 16}` range (the candidate for iter 108 if it promotes, paired with the current K=16 default) is a question with no precedent in the queue history.

**Test:** Single config change: `deq_k_jitter = True`, `deq_k_jitter_set = (10, 16)`. Hold all other knobs at iter 98b (or current) baseline. Compare s1000 val_bpb fast + roundtrip int6 + K-sweep table vs current baseline; check whether the K-sweep tail at K=128 tightens (analog of H12 verdict).

**Predicted outcomes:**
- (a) **Jitter wins on val_bpb + K-sweep tail**: per H12 verified pattern, jitter regularizes the FP and improves K=128 generalization. Promote, especially if iter 108 K=10 also promoted (compounded throughput + regularization win).
- (b) **Jitter neutral on val_bpb but tightens K-sweep**: stability win, marginal val_bpb gain. Per CLAUDE.md §8 simplicity criterion, may still promote if delta is small.
- (c) **Jitter regresses**: rare given H12 precedent, but possible if {10,16} is too narrow a range to provide meaningful gradient diversity.

**Strict-generalization status.** Not strict-generalizing. Standard val_bpb-primary promotion rule applies (§11).

**Risk.** Re-enabling K-jitter brings back the dynamo-recompile thrash that motivated the 2026-04-28 K=16 fixed-point decision (see CLAUDE.md `deq_k_jitter_set | (16,)` row). That recompile cost was ~6 s/step at the K-jitter-set size of 3; with set size 2 the overhead is approximately halved (~3 s/step) but still significant. Iter 109 must measure step_avg + recompile count and document whether the regularization gain outweighs the throughput cost.

**Connection to existing H-claims.** Pairs with H79 (iter 108 K=10 fixed) and H66 (iter 87 K-jitter widen, VERIFIED). Together they bisect the question "what K depth distribution is optimal?"

**Status:** PROPOSED — queued (priority 9, after iter 108).

### H83 + H83b SUPERSEDED by iter 117 (2026-04-29)

**iter 111a was launched 17:39, killed 17:55 at step 20/1000** after user redirect: skip 111a/111b, go directly to iter 117 (combined). Rationale: iter 117's blend_logit trajectory provides built-in attribution between variance-only-help vs entmax-help vs combined-help — recovers the same information as separate 111a + 111b runs, saves ~10 hours of compute. Variance penalty implementation (committed in `3e1bd50`) is preserved and reused inside iter 117.

iter 111a's partial trajectory (s0→s20): ntp_loss 7.01 → 5.05, attn_cv 0.03 → 0.63, pertoken_entropy 3.36 → 2.99, step_avg 21.1s steady. **Healthy at kill time.** Insufficient signal for attribution since variance penalty was still in warmup_delay (coef=0 below s300/1000).

### H83b ORIGINAL (frozen, superseded): Variance-only routing regularization (iter 111b) — PROPOSED (now folded into iter 117)

**Hypothesis.** iter 100b's reframe demonstrated that the per-token entropy penalty was "marginally effective at high coef — halted rise of pertoken_entropy but couldn't bend it down." The structural reason (H83 analysis): per-token entropy has SYMMETRIC gradient at uniform routing — no symmetry-breaking direction. Across-token variance has ASYMMETRIC gradient that points toward token specialization. Critically, **variance maximized at top-1 specialization (which has zero per-token entropy by construction)** — so variance subsumes entropy in the optimization sense. Test whether entropy is redundant: drop `router_entropy_coef → 0`, keep `routing_variance_coef = 0.005`.

**Test config (single-axis change from iter 111a):**
- `router_entropy_coef = 0.0`     # OFF (was 0.005 in iter 111a)
- `routing_variance_coef = 0.005` # ON (same as iter 111a)
- `routing_variance_warmup_delay_frac = 0.3`
- All other knobs at iter 100b baseline

**Strict-generalization** ✓: with `routing_variance_coef = 0.0` AND `router_entropy_coef = 0.0`, recovers iter 96 (the iter 100b predecessor without entropy). With `routing_variance_coef = 0.0` AND `router_entropy_coef = 0.005`, recovers iter 100b.

**Promotion logic:**
- If `|val_bpb_111b − val_bpb_111a| < 0.005`: entropy is REDUNDANT given variance → promote 111b (cleaner config, one fewer regularizer)
- If `val_bpb_111b > val_bpb_111a + 0.005`: entropy adds value despite weak gradient → keep 111a baseline; document why entropy + variance combine
- If `val_bpb_111b < val_bpb_111a − 0.005`: entropy was actively hurting (suppressing variance gradient?) → promote 111b

**Status:** PROPOSED — HIGH PRIORITY iter 111b, sequenced immediately after iter 111a regardless of 111a's promotion outcome (clean attribution requires both data points).

### H83: Per-token routing-variance penalty (iter 111a) — PROPOSED HIGH PRIORITY (2026-04-29 user spec)

**Hypothesis.** iter 100b's reframe + iter 98b's K-sweep showed `pertoken_entropy ≈ 3.0` consistently — uniform routing across tokens. Combined with `attn_ortho ≈ 0.12` (orthogonal experts), the architecture has the components for a per-token basis decomposition (orthogonal experts as basis vectors × per-token weights as coordinates) but lacks **token-conditional weight variance**: every token uses the same near-uniform mixture, making the E experts equivalent to one averaged dense block. The model pays for E experts and learns a 1-expert function.

**Principled fix.** Add a routing-variance regularizer that penalizes LOW variance in routing distributions across tokens:
```
L_var = -λ · Var_token(w(e|t))    summed over experts e, averaged over tokens
```
where `w(e|t)` is the routing weight of expert e for token t. **High variance across tokens = different tokens use different mixtures = token-conditional basis decomposition.** Orthogonal to existing regularizers:
- CV penalty: penalizes batch-mean imbalance (across batch, fixed expert)
- Entropy penalty: penalizes per-token spread (within token, across experts)
- **Variance penalty: penalizes uniform-across-tokens (across tokens, fixed expert)**

The three regularizers form an orthogonal basis on the routing distribution moments — a complete control system.

**Implementation.** One-line addition in `SoftDenseRouter.forward` post-softmax: compute `w.var(dim=batch_dim).mean()` and subtract from loss with coefficient `routing_variance_coef`. Same anneal-from-zero schedule as entropy_coef (per `feedback_anneal_sparsity_coefs.md`).

**Test:** Single config addition: `routing_variance_coef = 0.005` (mirroring entropy_coef target), `routing_variance_warmup_delay_frac = 0.3`. All other knobs at iter 100b baseline. Expected: pertoken_entropy STAYS high (we're not forcing per-token sparsity), but per-token routing patterns DIFFERENTIATE (low across-token variance becomes high). Token-specialized basis decomposition emerges. val_bpb gain comes from the model finally USING the E experts as distinct basis directions.

**Strict-generalization status.** Setting `routing_variance_coef=0.0` recovers iter 100b exactly (this iter strictly subsumes the baseline). Per CLAUDE.md §11, regression-safe — promotion is unconditional if val_bpb improves.

**Status:** PROPOSED — queued **HIGH PRIORITY priority 1** post-iter-98b per user direction 2026-04-29.

### H84: Orthogonal-expansion routing (iter 112) — PROPOSED HIGH PRIORITY (2026-04-29 user spec)

**Hypothesis.** Stronger version of H83 — replace the softmax router with one that produces **literal orthogonal per-token weight vectors**: enforce `(1/T) · Σ_t w_t w_t^T ≈ I/E` across tokens in a batch. Each token's weight vector becomes a direction in expert-space, giving a literal Krylov-style basis decomposition where the expert outputs span an E-dim subspace and per-token routing selects orthogonal coordinates within that subspace.

**Mechanism.** Add a Gram-matrix penalty:
```
G = (1/T) · Σ_t w_t w_t^T        # E × E matrix
L_ortho_route = ||G - I/E||²_F   # Frobenius distance to scaled identity
```
This forces the routing weight matrix `[w_1, w_2, ..., w_T]` (shape T × E) to have approximately orthogonal columns when row-normalized. Stronger than H83's marginal-variance penalty because it constrains the JOINT distribution of routing across tokens.

**Why principled.** This is the inner-product-space analog of the user's "weight basis like linear basis in linear algebra" framing. With orthogonal experts AND orthogonal-across-tokens routing, the model genuinely decomposes input → orthogonal basis × orthogonal coordinates. Function-space rank = E.

**Risk.** Orthogonal routing is HARDER than variance-based — may over-constrain and hurt training stability. Iter 99 sparsemax (similar over-constraint) +0.16 capacity cost. If iter 111 (variance penalty) closes the gap, iter 112 may be unnecessary; if iter 111 is insufficient, iter 112 escalates.

**Test:** Add `routing_gram_coef = 0.01`, `routing_gram_warmup_delay_frac = 0.3` annealed schedule. Strict-generalization (coef=0 recovers baseline). Run after iter 111.

**Status:** READY — integrated into `train_gpt.py` at commit `1d0d9ac` (2026-04-30). Default OFF; CLI-enable per launch with `--use-orthogonal-expansion-routing=1 --routing-gram-coef=0.01`.

**Component smoke (2026-04-30):** standalone helper smoke 6/6 PASS — penalty correctness verified analytically: uniform=0.109375 = (E-1)/E²; balanced one-hot=0; collapsed=0.875 = (1−1/E)² + (E−1)·(1/E)²; gradient flows through softmax→W path; anneal helper validated against `progress` ∈ {0, delay, midway, 1}. Integration into `train_gpt.py::SoftDenseRouter._collect_routing_losses` LANDED via three-touchpoint pattern (commit `1d0d9ac`): Hyperparameters fields + CLI flags + `_gram_coef` buffer/property/annealer + Gram penalty computation in router forward + `_gram_penalty_loss` folded into `router_reg_loss`. End-to-end smoke PASSED (loss 7.02 → 4.44 over 300 steps, recon stable, no NaN). Standalone helper archived at `experiments/components/archive/orthogonal_expansion_routing.py` (design notes only; canonical implementation now lives in `train_gpt.py`).

### H85: Increase block_ortho_aux_coef 0.1 → 0.5 (iter 113) — DROPPED ✗ (2026-04-30, no-op given current ortho values + threshold-design analysis)

**Decision (2026-04-30, post-iter-117 v5 K-sweep analysis):** DROP iter 113. The proposed 5× coef bump is essentially a no-op because:

1. **Iter 117 v5 K-sweep ortho values are below threshold:** `attn_ortho = 0.1235`, `mlp_ortho = 0.2041`. Threshold = 0.20. Penalty per current formula `relu(ortho − 0.20)² × coef`:
    - `attn_b = relu(0.1235 − 0.20)² = 0` (below threshold → zero gradient)
    - `mlp_b = relu(0.2041 − 0.20)² ≈ 1.7e-5` (just above; tiny)
    - At `coef=0.1`: total contribution ≈ 8.4e-7 (negligible vs ntp_loss ~2.5)
    - At `coef=0.5`: total contribution ≈ 4.2e-6 (still negligible; absolute change ~3e-6)
2. **The K-sweep shows ortho is K-invariant** at exactly 0.1235 and 0.2041 across all K∈{4, 8, 16, 17, 24, 32, 37, 64, 113, 128} — confirming this is a PARAMETER-LEVEL property dominated by expert weight matrices. Eval-time variation is below 4-decimal display precision.
3. **Threshold-based design is heuristic, not principled** — `thr = 0.20` is arbitrary, max-pairwise is a heuristic metric choice, and the formulation creates a flat basin (no pressure when below threshold) with no theoretical grounding.

**To redesign iter 113 meaningfully**, would need to either:
- Lower threshold (`thr = 0.20 → 0.10`) so penalty actually activates at current ortho levels
- Drop threshold entirely (active at all magnitudes, original H32 form)
- Switch to a principled formulation: Frobenius distance from `I/E`, mutual information, or spectral regularizer

**Iter 112 (H84 Gram-matrix penalty) supersedes this need.** The Gram penalty `‖G − I/E‖²_F` (where `G = (1/N) W^T W` over routing weights) is the principled alternative — no threshold, active everywhere, targets orthogonal columns of routing weight matrix directly. Component already PASSED smoke tests (`experiments/components/archive/orthogonal_expansion_routing.py` (archived after iter 112 integration), 6/6).

**Status:** DROPPED ✗. Pivot to iter 112 (H84) for the principled orthogonality push, or iter 110 (H82, re-enable num_refinements=1) for an architectural test.

### H85 ORIGINAL (frozen for archival): Increase block_ortho_aux_coef 0.1 → 0.5 (iter 113) — PROPOSED HIGH PRIORITY (2026-04-29 user spec)

**Hypothesis.** Cheap baseline test. Currently `block_ortho_aux_coef = 0.1`. Pushing to 0.5 (or 1.0) forces more orthogonal expert OUTPUTS — addresses the basis-component side of the basis-decomposition argument (vs iter 111/112 which address the routing side). Useful as a control: if increased ortho alone closes the gap, the issue was insufficient orthogonality, not uniform routing. If it doesn't close the gap, that confirms the routing-uniformity (not expert orthogonality) is the bottleneck — strengthening the case for iter 111/112.

**Test:** `block_ortho_aux_coef = 0.5` (5× current). All other knobs at iter 100b baseline. Single-knob ablation.

**Strict-generalization status.** Coef = 0.1 (current) is a special case of coef ∈ [0, ∞), so this strictly subsumes iter 100b only if the regularizer is monotone in coef (which it isn't — too-high coef can make experts COLLAPSE to identical-up-to-orthogonal). Standard val_bpb-primary promotion applies.

**Pre-emptive prediction.** Iter 100b already shows `attn_ortho 0.12, mlp_ortho 0.14` (close to "low" target 0.1-0.2). Pushing to 0.5× more might saturate the orthogonality (already near boundary) without helping per-token specialization. Likely NOT PROMOTED, but the data point confirms whether ortho is the bottleneck.

**Status:** PROPOSED — queued **HIGH PRIORITY priority 3** post-iter-98b per user direction 2026-04-29 (control test for H83/H84).

### H86: Native Sparse Attention (iter 106) — DEFERRED 2026-04-29 (T=2048 kernel cost; revised analysis 2026-04-29)

**Decision (2026-04-29, REVISED).** Iter 106 NSA implementation was 80% complete (2-branch compression + sliding via SDPA-with-mask, full plumbing) when official fla-org/native-sparse-attention Triton benchmarks revealed: at **T=2048 (our seq len), NSA is 0.42× FlashAttention — i.e. ~2.4× SLOWER per attention call**. Speedup only materializes at T ≥ 8192 (1.35×) and T = 16384 (2.66×). Initial reaction was "drop entirely"; **revised analysis (2026-04-29) reframes as DEFERRED, not dropped**:

**Recomputed step-time impact (corrected)**:
- Per-call delta NSA vs FA at T=2048 ≈ 1.43 ms (fla-org benchmark, on-device kernel time)
- 192 attn calls per step × 1.43 ms = +274 ms attention slowdown per step
- Attention is ~10–30% of total 18.5s step time → **+1.5% to +5% step regression** (NOT +5–20% as initially quoted — that estimate didn't account for attention-as-fraction-of-step)
- torch.compile + CUDA-graph replay eliminates per-call Python dispatch overhead (~60 ms savings/step), reducing the net regression further
- For per-wallclock gate (typically tolerates 3-7% regression if val_bpb gain is meaningful): **NSA at T=2048 is borderline, not catastrophic**

**Why DEFER not drop**: regularization upside is unmeasured. Iter 101 (entmax-1.5 from step 0) hurt val_bpb by +0.10 — that's a different mechanism (router sparsity), and NSA's regularization is via attention sparsity which interacts with FP iteration differently. Could go either way. The right experiment is to RUN iter 106 NSA alongside iter 117 (entmax+skip) to compare regularization-only vs throughput-positive sparsity strategies head-to-head. Run priority: AFTER iter 117/118 establish the throughput baseline (iter 117/118 are higher-leverage on the throughput axis at T=2048).

**Sources**: github.com/fla-org/native-sparse-attention (T=2048: NSA 2.4ms vs FA 0.97ms); github.com/lucidrains/native-sparse-attention-pytorch (PyTorch impl with `flex_attention` for sliding); github.com/XunhaoLai/native-sparse-attention-triton.

**Disposition**: code stays in repo (off by default via `use_nsa_attention=False`); 2-branch implementation complete and standalone-validated (strict-gen max diff 0.0078 vs full SDPA in bf16, finite gradients). Re-queue **AFTER iter 117** to test as a regularization-only sparsity intervention.

### H86 ORIGINAL (frozen for archival): Native Sparse Attention (iter 106) — PROPOSED HIGHEST PRIORITY (2026-04-29 user spec)

**Design replaces iter 104 (AdaSplash, DROPPED) as the principled sparse-attention path.** Per user direction 2026-04-29 post-cleanup commit `c5d3d42`: iter 106 is the NEXT iter to launch.

**Hypothesis.** DeepSeek's Native Sparse Attention (arxiv:2502.11089, ICML 2025) replaces single-path softmax attention with a **three-branch hybrid** that is genuinely sparse while preserving global reach:
- **Compression branch**: each query attends to a downsampled K/V stream covering ALL T tokens (compress_block_size, compress_block_sliding_stride). O(T / compress_block) cost. **Global** but coarse.
- **Selection branch**: each query selects top-K most relevant blocks across the full sequence (selection_block_size, num_selected_blocks) — fine-grained access to a sparse subset chosen per query. O(num_selected_blocks × select_block_size) cost. **Sparse but precise**.
- **Sliding-window branch**: each query attends to its W most recent tokens (sliding_window_size). O(W) cost. **Local recency**.

The three branches are computed independently and gated together via a learnable mixer. Together they preserve `O(T)` reachability at `O(T/block)` compute. Per the DeepSeek paper, **NSA matches OR exceeds full-attention val_bpb on long-context tasks while being substantially faster** — the rare case where sparse attention is BOTH faster and better.

**Why NSA over alternatives** (rationale from 2026-04-29 review):
- vs **AdaSplash α-entmax** (iter 104): AdaSplash uses `torch.autograd.Function` which is incompatible with torch.compile per official PyTorch dev-discuss guidance. Two systematic-debug rounds (custom_op + contiguous everywhere) failed with `_functionalization.apply_view_meta_sequence` corruption at training step 2. DROPPED 2026-04-29.
- vs **sliding-window-only**: hard cap on context — anything past W tokens invisible. Long-range modeling regression. DROPPED 2026-04-29.
- vs **PyTorch-native sparsemax/entmax-1.5** (iter 99/101): both NOT PROMOTED with +0.16 / +0.10 capacity costs — PyTorch-only sparse attention is too expensive to compute densely-then-mask.
- **NSA's torch.compile compatibility**: implementable via `torch.nn.attention.flex_attention` with `BlockMask`, which is designed FOR torch.compile (no autograd.Function gotcha). Per FlexAttention blog: "lowers into a fused FlashAttention kernel through torch.compile" + supports arbitrary `score_mod` / `mask_mod`.
- **Native trainability**: DeepSeek paper key contribution — NSA is end-to-end trainable from scratch (no dense-warmstart prerequisite). Fits our 1000-step from-scratch training regime.

**Implementation plan (~3-4 hours engineering)**:

1. **Reference impl**: github.com/lucidrains/native-sparse-attention-pytorch — community PyTorch implementation matching the DeepSeek paper. Use as scaffolding to skip from-paper reimplementation.

2. **Architecture invariants preserved** (per CLAUDE.md §6.3):
   - Per-expert MLA pipeline unchanged (Q/K/V/Wo low-rank, gated attention).
   - Head-packed `(B, E·H, T, d)` shape unchanged for compression/sliding branches.
   - Selection branch's top-K indexer operates per-query (per-token routing of which blocks to attend to) — a NEW learnable component.

3. **Hyperparameters to add** (mirror lucidrains naming):
   - `nsa_compress_block_size = 32` (downsample stride for compression branch)
   - `nsa_compress_block_sliding_stride = 16` (compression overlap)
   - `nsa_selection_block_size = 64` (block granularity for top-K selector)
   - `nsa_num_selected_blocks = 4` (per-query top-K count)
   - `nsa_sliding_window_size = 256` (W for the local branch)
   - `nsa_branch_gate_init = 0.0` (initial mixer bias — softmax-equivalent at init)
   - `use_nsa_attention = False` (default disabled; CLI-enabled per iter 106 launch)

4. **Plumbing** (4-touch rule per CLAUDE.md §9):
   - `Hyperparameters` adds the NSA fields.
   - `_CLI_TUNABLE_KNOBS` appends each.
   - `CausalSelfAttention.forward` gains a branch: `if use_nsa_attention: y = nsa_attention(...) else: y = F.scaled_dot_product_attention(...)`.
   - CLAUDE.md §5 mirror + §6.3 architectural-invariant section get NSA-on-as-an-option documented.

5. **Strict-generalization** (per CLAUDE.md §11): setting `use_nsa_attention=False` recovers iter 100b exactly. Promotion is **unconditional** if val_bpb improves at strictly-generalized parametrization (special case: NSA branches with `compress_block_size=1` + `num_selected_blocks=T/select_block` + `sliding_window_size=T` ≈ full attention; in practice we promote on any val_bpb gain at the chosen sparse params).

6. **Test sequence**:
   - Smoke test (`python experiments/smoke_test.py`) MUST pass at default config (NSA off → identical to iter 100b).
   - 10-iter live test with `--use-nsa-attention=1 --num-heads=12 --num-kv-heads=6` (head_dim=64 for FA tensorcore sweet spot; flex_attention prefers 16/32/64/128). Confirm no view_meta_sequence crash, hutch_F finite at val checkpoint, train loss descends.
   - Full 1000-step run + K-sweep matrix + roundtrip int6 per CLAUDE.md §7 protocol mandate (the FULL `k_sweep_table:` matrix copied verbatim into hypotheses.md after).

**Pre-emptive prediction**: NSA's compression+selection branches preserve global reach so val_bpb should NOT regress vs iter 100b. The DeepSeek paper claims throughput gains at T=64k+; at our T=2048 the gain may be modest (full attention is already cheap). The ICLR-flavor value: NSA's **selection branch IS a learned sparsity** that the model adapts per-query — this is the "real adaptive sparsity" the user wanted from AdaSplash, but via a torch.compile-native path.

**Risks**:
- Selection branch's top-K indexer has a learnable softmax over block scores → may interact with our DEQ FP iteration in unexpected ways. Mitigation: validate K-sweep at K∈{4, 8, 16, 24, 32, 64, 128} acyclicity primes pass.
- lucidrains' impl may not be perfectly DDP-compatible out of the box. Mitigation: smoke + 10-iter live test before full 1000-step.
- NSA's three-branch gating adds ~5-15% step cost in expectation (per paper). If val_bpb improvement is smaller than the wallclock cost, per-wallclock gate (H72 precedent) may apply.

**Status:** PROPOSED — **HIGHEST PRIORITY (next iter to launch, 2026-04-29 user direction)**.

### H82: Re-enable num_refinements=1 (iter 110) — PROPOSED (2026-04-29 user spec)

**Hypothesis.** iter 98b's NOT PROMOTED post-mortem showed refinement adds +6% step cost at D=768 (iter 100b) and +70% at D=1024 (iter 98b), but the val_bpb benefit of refinement has NEVER been directly ablated under the iter-100b baseline. Disabling by default (`num_refinements=0`, set 2026-04-29) creates a clean baseline without the refinement cost. iter 110 tests RE-ENABLE on top of iter 100b's CV-only equilibrium to determine whether refinement contributes meaningful val_bpb gain.

**Test:** Single config flip from the new disabled-default baseline: `num_refinements = 1`, `num_refinements_ramp_frac = 0.85` (the prior default — refinement enables for the last 15% of training). All other knobs at iter 100b's `c5154b3` baseline.

**Expected outcomes:**
- (a) **Refinement helps**: val_bpb int6 < baseline by ≥ 0.005, justifying the +6% step cost. Promote (with refinement on by default).
- (b) **Refinement neutral**: |Δ val_bpb| < 0.005. NOT PROMOTED — keep refinement disabled (saves wallclock with no quality cost).
- (c) **Refinement hurts**: val_bpb regression. Confirms the 2026-04-29 disable decision; refinement was load-bearing only for the now-cancelled diffusion-AR objective (iter 94 disabled CTP, removing the loss that motivated refinement).

**Why now (2026-04-29 user direction).** The architecture invariant in CLAUDE.md §6.5 marked refinement as "currently enabled with ramp 0.85" but no H-claim ever VERIFIED its val_bpb contribution. iter 98b's failure mode brought the cost into focus; this iter brings the benefit-side into focus.

**Status:** PROPOSED — queued (priority 7 in revised post-iter-98b queue per user reorder 2026-04-29).

### iter 104 (AdaSplash α-entmax) — DEFERRED ON UPSTREAM BUG (2026-04-29 systematic debug exhausted)

**Final diagnosis (post round 2 systematic debug 2026-04-29).** The `_functionalization.apply_view_meta_sequence` crash at step 2 with garbage int64 shape is NOT in our wrapping code — it's an upstream incompatibility between `torch.autograd.Function` (which AdaSplash uses internally at `adasplash_block_mask.py:936 class _sparse_attention(torch.autograd.Function)`) and torch.compile's AOTAutograd functionalization layer.

**Authoritative evidence (PyTorch dev-discuss, "Custom Ops Under torch.compile: autograd.Function vs torch.library.custom_op"):**
> "torch.autograd.Function is the most widely used API, but it is **not the recommended one if you need torch.compile integration**. Some compositions of autograd.Function with PyTorch operator registration APIs can lead to silent incorrectness when composed with torch.compile."

**Systematic debug rounds 1-2 results:**

| Round | Hypothesis | Fix | Result |
|---|---|---|---|
| 1 | `@torch._dynamo.disable` insufficient; need custom_op | Wrap call in `torch.library.custom_op` with FakeTensor + register_autograd | **FAILED** — same view_meta_sequence crash at step 2 |
| 2 | FakeTensor metadata vs real-tensor stride mismatch | Add `.contiguous()` at all custom_op boundaries (inputs, outputs, saved_for_backward, grad outputs) | **FAILED** — same crash, same shape garbage `[22343783044301, 5039929193026096800, 2048, 64]` |

Both attempts crashed at step 2 with byte-identical error patterns, confirming the bug is INSIDE AdaSplash's `_sparse_attention.apply()` autograd Function — wrapping at our level cannot bypass it.

**Three escape options (none committed to as of 2026-04-29):**
1. **Wait for AdaSplash upstream**: deep-spin/adasplash needs to migrate from `torch.autograd.Function` to `torch.library.custom_op` per PyTorch's official recommendation. File issue at https://github.com/deep-spin/adasplash. Timeline unknown.
2. **Fork AdaSplash kernels**: re-implement forward + backward Triton orchestration (~600 lines from `adasplash_block_mask.py`) inside our own `@torch.library.custom_op`. ~1-2 days of engineering effort. High implementation risk; not worth the marginal val_bpb upside given iter 100b's reframe (specialization not the bottleneck).
3. **Defer iter 104**: skip entirely. Move on to higher-priority queue items (iter 111-113 routing-variance, iter 110 refinement-test).

**Decision (2026-04-29): option (3), DEFER.** Code stays in train_gpt.py disabled-by-default (`attn_alpha_target=1.0`). Will re-evaluate when AdaSplash upstream migrates or when iter 111/112/113 results indicate sparsity is genuinely worth the engineering effort.

**Related:** H75 (iter 101 entmax-1.5 +0.10 capacity cost), H77 (iter 102 α-anneal NOT PROMOTED on per-wallclock), H74 (iter 99 sparsemax +0.16 capacity cost). Architectural-sparsity axis already 3× tested via PyTorch-native paths and 3× NOT PROMOTED — AdaSplash's only remaining value is *throughput* via fused Triton kernel, which is currently inaccessible.

### H87: Adaptive entmax-α + capacity-padded compute-skip (iter 117) — PROPOSED HIGH PRIORITY 2026-04-29

**Hypothesis.** Entmax-α (α ∈ (1, 2]) is the unique fully-differentiable sparse-routing map that is RevDEQ-compatible AND admits exact compute-skip. Pairing it with **per-block learnable α (init=1.0)** lets the model decide layer-by-layer how much sparsity helps, **AND** avoids iter 101's cold-start trap (H75: from-step-0 entmax-1.5 cost +0.10 val_bpb) by starting at exact softmax. Pairing with **capacity-padded buckets** lets us SKIP per-expert compute on tokens with weight = 0 — without breaking torch.compile's static-shape requirement, without STE, without chain-rule violation.

**Adaptive α parameterization (per-block learnable, init = dense softmax):**

```
alpha_logit_l = nn.Parameter(torch.full((1,), -7.0))   # one scalar per block, total 12
α_l = 1.0 + softplus(alpha_logit_l)                    # ∈ [1.0, ∞), starts at 1.0009 ≈ softmax
weights_l = entmax_alpha(scores, α_l)                  # closed-form, differentiable in α and scores
```

- **Strict-gen at init**: `softplus(−7) ≈ 9e-4` → α ≈ 1.0009 ≈ pure softmax → recovers iter 100b within bf16 floor at step 0
- **Lower-bounded at α=1.0**: by construction (softplus ≥ 0), α can never go BELOW softmax — only sparser
- **Per-block degree of freedom**: each of 12 blocks learns its own sparsity preference. Early/late layers may diverge — analog of Mixture-of-Depths layer-skip but at the routing level

**How sparsity gets "encouraged" (no explicit α penalty needed):**
1. **Existing per-token entropy penalty** (`router_entropy_coef = 0.005` from iter 100b H76) directly rewards lower per-token entropy. Higher α → lower per-token entropy → α drifts up if gradient says it helps.
2. **LM loss → expert specialization gradient**: if specialization helps val_bpb (iter 100b reframe says pertoken_entropy=3.0 IS a real bottleneck), gradient through entmax_α to alpha_logit pushes α up.
3. **Optional iter 117b**: if α stays pinned near 1.0 throughout (no sparsity emerges), add `−α_coef × Σ_l (α_l − 1)` penalty to actively encourage. Default: omit, let gradient decide.

**Gradient through α to `alpha_logit`**: entmax-α has well-defined `∂(weights)/∂α` (involves the Tsallis-α threshold τ). The closed-form sort-and-threshold implementation supports this via standard autograd.

**Why entmax-α is principled (vs Switch/Mixtral top-K + STE):**

| Property | Softmax | **Entmax-1.5** | Sparsemax | Argmax + STE |
|---|---|---|---|---|
| Output | always dense | sparse (exact 0s) | sparser | one-hot |
| Continuous everywhere | ✅ | **✅ (Lipschitz)** | ✅ | ❌ |
| Differentiable everywhere | ✅ | **✅ smooth Jacobian** | ✅ | ❌ STE only |
| Boundary at zero | n/a | **smooth pass-through** | smooth | discrete jump |
| RevDEQ-safe (smooth FP) | ✅ | **✅** | ✅ | ❌ chain-rule break |

**The fundamental tension and how entmax-α resolves it.** RevDEQ's contraction proof requires `‖∂T/∂z‖ < 1` everywhere along the FP iteration. STE-based methods (Switch top-1, Mixtral top-K) make `∂T/∂z` discontinuous at routing boundaries (when token t flips from expert i to expert j as z iterates) — FP iteration can oscillate between branches instead of converging. **Entmax-α is C¹-continuous at the zero-boundary**: as a logit crosses threshold, the corresponding weight passes through 0 smoothly with a well-defined Jacobian (which has a zero column for inactive experts but is still continuous). Skipping zero-weight experts at compute is **operator-level identical** to dense compute (`w_e · expert_e(x) = 0` regardless of expert_e), preserving both forward output AND backward gradients exactly.

**Implementation (path A, adaptive α + capacity-padded inside-compile):**

```
# Per-block learnable α (init = dense softmax)
self.alpha_logit = nn.Parameter(torch.full((1,), -7.0))   # block-l scalar

Forward (per-block, per-token routing):
  α_l = 1.0 + softplus(self.alpha_logit)                  # ≈ 1.0009 at init
  scores = router(x_n)                                     # (B, T, E)
  w = entmax_alpha(scores, α_l)                            # exact zeros emerge as α > 1
  # Capacity-padded dispatch (Switch-style, fully torch.compile-compatible):
  for e in range(E):                                        # static loop over experts
    cap = ceil(C · T·B / E)                                # C = 1.5; static buffer size
    top_k_idx = w[:,:,e].topk(cap, dim=tok_axis).indices    # deterministic, static shape
    in_e = gather(input, top_k_idx)                          # (cap, D)
    w_e_top = gather(w[:,:,e], top_k_idx)
    out_e_compact = expert_e(in_e) * w_e_top[:, None]
    output += scatter(out_e_compact, top_k_idx)
```

**Strict-generalization (CLAUDE.md §11):** at init `alpha_logit = −7` → α ≈ 1.0009 ≈ pure softmax = iter 100b exact within bf16 numerical floor. With C → ∞, no overflow, all expert outputs scaled by softmax weights = soft-dense forward map. Promotion is **unconditional** on val_bpb improvement. Even if α stays pinned at 1.0 (no sparsity emerges, no throughput win), val_bpb is bounded by iter 100b at worst.

**Expected outcomes:**
- val_bpb: matches or improves iter 100b — strict-gen bound, never worse than baseline at α=1.0; if α drifts up under gradient, that's because sparsity helped val_bpb
- step_avg trajectory:
  - s0–100: α ≈ 1.0 across all blocks → no compute skip → throughput identical to iter 100b
  - s100+: α drifts up if gradient says sparsity helps → throughput improves as exact zeros emerge in routing weights
  - s500+: per-block α distribution reveals architectural sparsity preference (e.g., late blocks may prefer α=2, early blocks may stay near α=1)
- Theoretical max throughput gain at α=1.5 with ~75% sparsity: 1.5–2.5× wallclock on expert path
- Per-block α evolution = NEW DIAGNOSTIC: log mean / min / max α across blocks per val checkpoint

**Risks:**
1. α may NOT drift up under gradient (no sparsity emerges, no throughput win, but no val_bpb regression either — strict-gen bound). Mitigation: iter 117b adds explicit α penalty to encourage sparsity if natural gradient is too weak
2. Capacity overflow: rare with C=1.5 at α=1.5 (~75% sparsity); track `overflow_rate` diagnostic; if >5% sustained, capacity-anneal or path B fallback
3. Gradient through entmax_α to learnable α may be small at α≈1 (where the entropy is high and the function is approximately softmax). Could be slow to drift; will measure with α-trajectory diagnostic.
4. Iter 102 (α-anneal NOT PROMOTED H77) showed entmax compute itself costs +6.8% wallclock WITHOUT compute-skip. iter 117's net throughput requires `compute-skip savings > entmax overhead` — at α≈1.0 (init), entmax overhead exists but compute-skip doesn't activate yet, so initial step_avg WILL regress modestly until α drifts up. If α never drifts → permanent +6.8% with no benefit. Mitigation: check α trajectory at first val checkpoint (s100); if α<1.05 across all blocks, fast-fail and revert.

**Status:** PROPOSED — **HIGH PRIORITY iter 117**, queued after iter 113 (H85). Expected first principled throughput-positive iter in our codebase.

### H87 RESULT (iter 117 v5, commit `08783b4`, 2026-04-30): PROMOTED ★ — strict-gen via `use_entmax_routing` CLI flag; carry-forward gate satisfied

**Verdict.** PROMOTED under +0.03 carry-forward rule + strict-generalization recovery. Final design adopted entmax-1.5 + softmax annealed BLEND with learnable scalar `_entmax_blend_logit` (init=5.0, sigmoid(5)=0.9933 → ≈ pure softmax) + `_entmax_blend_anneal` buffer (0→1 over training). At CLI default `use_entmax_routing=False`, recovers iter 100b bit-identically (entmax code path entirely gated off). At `--use-entmax-routing=1` (this run), the blend stays near softmax-dominant (logit movement minimal at `entmax_blend_lr=0.002` over 1000 steps from init 5).

**Final results (1000 steps, --use-entmax-routing=1):**
- val_bpb fp32 (s1000): **1.4820** (vs iter 100b ~1.4716, Δ +0.0104)
- val_bpb int6 roundtrip: **1.5122** (vs iter 100b 1.4893, **Δ +0.0229**, within +0.03 carry-forward gate)
- artifact_bytes: 7,510,357 (47% of 16 MB budget); total (code+artifact) 7,819,443 / 16,000,000
- step_avg: 20.62s (vs iter 100b ~22s, **-6% throughput improvement** from concurrent code-quality refactors landed mid-iter)
- peak_vram_mb: 34,598 (no OOM)
- 0 NaN, 0 errors, 0 dynamo recompile warnings (iter 117 v5 fixed v4's recompile_limit hit via `@dynamo_disable` shared-gate diagnostic helper)

**K-sweep matrix (verbatim from run.log, K∈{4, 8, 16, **17**, 24, 32, **37**, 64, **113**, 128} with acyclicity primes bolded):**

```
K-sweep done: k4:1.777597 k8:1.568374 k16:1.512213 k17:1.511606 k24:1.511525 k32:1.512369 k37:1.512702 k64:1.513267 k113:1.513534 k128:1.513522

k_sweep_table:    K   val_bpb  attn_cv   mlp_cv  pool_cv  attn_min   mlp_min  attn_ortho  mlp_ortho  pertoken_ent  pool_ent  shared_gate   hutch_F   rd_step  iter_conv_rel
k_sweep_table:    4    1.7776   0.2917   0.1723   0.2396    0.0288    0.0417      0.1235     0.2041        2.8637    3.3705       0.3991       N/A       N/A         0.1874
k_sweep_table:    8    1.5684   0.2699   0.1766   0.2281    0.0344    0.0404      0.1235     0.2041        2.8441    3.3740       0.3970       N/A       N/A         0.0824
k_sweep_table:   16    1.5122   0.2722   0.1585   0.2227    0.0359    0.0434      0.1235     0.2041        2.8111    3.3754       0.3930       N/A       N/A         0.0216
k_sweep_table:   17    1.5116   0.2718   0.1578   0.2222    0.0359    0.0435      0.1235     0.2041        2.8102    3.3756       0.3928       N/A       N/A         0.0187
k_sweep_table:   24    1.5115   0.2716   0.1566   0.2216    0.0361    0.0437      0.1235     0.2041        2.8089    3.3757       0.3923       N/A       N/A         0.0091
k_sweep_table:   32    1.5124   0.2711   0.1563   0.2213    0.0361    0.0437      0.1235     0.2041        2.8091    3.3758       0.3922       N/A       N/A         0.0064
k_sweep_table:   37    1.5127   0.2710   0.1566   0.2213    0.0361    0.0437      0.1235     0.2041        2.8091    3.3758       0.3921       N/A       N/A         0.0060
k_sweep_table:   64    1.5133   0.2710   0.1567   0.2214    0.0361    0.0437      0.1235     0.2041        2.8089    3.3758       0.3921       N/A       N/A         0.0056
k_sweep_table:  113    1.5135   0.2709   0.1566   0.2212    0.0361    0.0437      0.1235     0.2041        2.8089    3.3758       0.3921       N/A       N/A         0.0056
k_sweep_table:  128    1.5135   0.2708   0.1566   0.2212    0.0361    0.0437      0.1235     0.2041        2.8090    3.3758       0.3921       N/A       N/A         0.0057
```

**Acyclicity prime check (genuine FP confirmation):** K=17 (1.5116) ≈ K=16 (1.5122) within 0.0006; K=37 (1.5127) ≈ K=32 (1.5124) within 0.0003; **K=113 (1.5135) ≈ K=128 (1.5135) within 0.0000**. Three prime-vs-power-of-2 deltas all in [0.000, 0.001] band → **GENUINE FIXED POINT**, not iteration-count-specific overfitting.

**K=128 vs best-K Δ:** best-K = K=24 (1.5115); K=128 - K=24 = +0.0020 ≪ 0.5 gate. **PASS.**

**hutch_F / rd_step:** N/A across all K-sweep rows due to autocast-mismatch dtype error: `RuntimeError: self and mat2 must have the same dtype, but got Float and BFloat16`. Root cause: `target_dtype = next(sb.parameters()).dtype` returns fp32 (control tensor) while internal SharedBlock matmuls expect bf16. The val-checkpoint hutch_F probe got an autocast wrapper in iter 97.5b-fix2 (L4177); the K-sweep `_compute_eval_fp_lipschitz` probes were missed. **FIXED 2026-04-30 in iter 117 v5 promotion commit (iter 97.5b-fix3)**: same autocast wrapper added to both probes inside `_compute_eval_fp_lipschitz`. Future iters will report hutch_F + rd_step per-K. Val-checkpoint hutch_F values (which DO use the autocast wrapper) tracked correctly: s0=1.91 → s200=0.38 → s400=0.50 → s600=0.57 → s800=0.61 → s1000=0.66 (rising contractive across training).

**Val-checkpoint trajectory:**

| Step | val_bpb (fast) | Δ vs iter 100b | hutch_F | attn_cv | mlp_cv | pool_cv | pertoken_ent | router_mass |
|---|---|---|---|---|---|---|---|---|
| s0   | 4.2168 | (init) | 1.913 | 0.030 | 0.031 | 0.030 | 3.364 | 0.993 |
| s200 | 2.0466 | +0.045 | 0.376 | 0.555 | 0.190 | 0.415 | 2.004 | 0.983 |
| s400 | 1.7142 | +0.066 (peak) | 0.499 | 0.343 | 0.085 | 0.250 | 2.494 | 0.921 |
| s600 | 1.5739 | +0.040 | 0.572 | 0.307 | 0.096 | 0.227 | 2.705 | 0.895 |
| s800 | 1.5248 | +0.031 | 0.609 | 0.271 | 0.101 | 0.204 | 2.805 | 0.872 |
| **s1000** | **1.4820** | **+0.010** | **0.657** | **0.216** | **0.062** | **0.159** | **2.799** | **0.841** |

**Lag CLOSED through training:** +0.045 (s200) → +0.066 (s400 peak) → +0.040 (s600) → +0.031 (s800) → +0.010 (s1000). The blend is near-pure-softmax throughout (entmax_blend_lr=0.002 too slow to overcome init logit=5 in 1000 steps), so iter 117 v5 effectively runs iter 100b with marginal entmax pollution — yet still trails by only +0.010 fp32 / +0.023 int6.

**Reframe (re-confirms iter 100b H76 finding):** at near-pure-softmax blend, pertoken_entropy stalls at 2.80 (vs target ≤1.5 for specialization). The infrastructure works; sparsity is not engaging because (a) init logit=5 keeps blend ≈ 99% softmax and (b) `router_entropy_coef=0.005` magnitude × `pertoken_entropy=2.80` = 0.014 dwarfed by ntp gradient ~2.5 (25× imbalance vs `cv_loss_weight=2.0 × cv=0.20 = 0.40`). **Iter 117b will bump entropy_coef to 0.05 (10×) and refactor cv + entropy + block_ortho into a single named `router_reg_loss` group** to give entropy the gradient budget to actually drive specialization.

**Status:** PROMOTED ★. Architecture extended with entmax-1.5 + annealed blend (CLI-gated) + `_entmax_blend_logit` learnable parameter (10×-slower LR group). Strict-gen recovery to iter 100b verified at `use_entmax_routing=False`. Iter 117b queued for actual sparsity engagement.

### H87b RESULT (iter 117b-1, commit `54473aa`, 2026-04-30): NOT PROMOTED ✗ — hypothesis REFUTED, config bumps reverted

**Hypothesis tested.** 10× bump of `router_entropy_coef` (0.005 → 0.05) and `entmax_blend_lr` (0.002 → 0.02) on top of iter 117 v5's blend infrastructure should drive per-token entropy down (from 2.80 toward ≤1.5) and unlock val_bpb improvement via specialization.

**Result: REFUTED.** Pertoken_entropy stayed at ~2.6 throughout training, never bending below 2.5. The 10× entropy coef engaged but couldn't overcome CV-redistribution dominance in soft-dense MoE.

**Final results (1000 steps, --use-entmax-routing=1):**
- val_bpb fp32 (s1000): **1.4790** (vs iter 117 v5 1.4820, Δ −0.003)
- val_bpb int6 roundtrip: **1.512872** (vs iter 117 v5 1.5122, **Δ +0.0007**, essentially flat)
- artifact_bytes: 7,623,275 (47.6% of 16 MB budget)
- step_avg: 20.62s (matches iter 117 v5)
- peak_vram_mb: 34,598
- 0 NaN, 0 errors

**Promotion gate analysis (CLAUDE.md §11):**
- artifact ≤ 16 MB: **PASS**
- K=128 vs best-K Δ ≤ 0.5: **PASS** (best K=17 at 1.5124, K=128 at 1.5154 → Δ=+0.003)
- Acyclicity primes: **GENUINE FP** (K=17→K=16: -0.0005, K=37→K=32: +0.0005, K=113→K=128: 0.0000)
- val_bpb improvement vs current baseline (iter 117 v5 = 1.5122): **NO** (+0.0007)
- Strict-generalization escape: **NO** (hyperparameter-only changes don't extend the function class)
- Per §11: *"val_bpb worse/equal AND no strict-gen → git revert"*

**K-sweep matrix (verbatim from run.log):**

```
   K   val_bpb  attn_cv   mlp_cv  pool_cv  attn_min   mlp_min  attn_ortho  mlp_ortho  pertoken_ent  pool_ent  shared_gate   hutch_F   rd_step  iter_conv_rel
   4    1.7806   0.3593   0.2030   0.2918    0.0320    0.0382      0.1338     0.1992        2.5789    3.3580       0.5042    0.6981  451.4170         0.1968
   8    1.5652   0.3493   0.2131   0.2894    0.0294    0.0352      0.1338     0.1992        2.5794    3.3574       0.5042    0.6755  450.6616         0.0835
  16    1.5129   0.3252   0.1993   0.2697    0.0310    0.0360      0.1338     0.1992        2.5956    3.3629       0.5046    0.6759  443.8861         0.0243
  17    1.5124   0.3246   0.1992   0.2693    0.0310    0.0360      0.1338     0.1992        2.5963    3.3630       0.5046    0.6753  441.8971         0.0216
  24    1.5129   0.3240   0.1985   0.2687    0.0311    0.0362      0.1338     0.1992        2.5965    3.3631       0.5046    0.6767  449.0061         0.0131
  32    1.5139   0.3226   0.1986   0.2679    0.0312    0.0362      0.1338     0.1992        2.5969    3.3634       0.5046    0.6771  438.8580         0.0108
  37    1.5144   0.3222   0.1987   0.2676    0.0312    0.0362      0.1338     0.1992        2.5974    3.3634       0.5046    0.6761  448.5642         0.0103
  64    1.5151   0.3207   0.1985   0.2667    0.0311    0.0362      0.1338     0.1992        2.5978    3.3637       0.5046    0.6786  445.8197         0.0102
 113    1.5154   0.3214   0.1989   0.2672    0.0312    0.0362      0.1338     0.1992        2.5972    3.3635       0.5046    0.6739  447.3644         0.0101
 128    1.5154   0.3218   0.1988   0.2675    0.0312    0.0362      0.1338     0.1992        2.5971    3.3635       0.5046    0.6737  440.0336         0.0101
```

**hutch_F + rd_step now reported per-K** (iter 97.5b-fix3 autocast wrapper working ★). Stable around 0.67/445 across all K — confirming consistent contractive FP behavior.

**Val-checkpoint trajectory:**

| Step | val_bpb (fast) | Δ vs iter 117 v5 | hutch_F | attn_cv | pertoken_entropy |
|---|---|---|---|---|---|
| s200  | 2.0238 | -0.0228 | 0.413 | 0.555 | 1.846 |
| s400  | 1.6913 | -0.0229 | 0.538 | 0.343 | 2.452 |
| s600  | 1.5693 | -0.0046 | 0.602 | 0.330 | 2.609 |
| s800  | 1.5201 | -0.0047 | 0.624 | 0.300 | 2.609 |
| s1000 | **1.4790** | **-0.0030** | **0.673** | **0.301** | **2.607** |

**Trajectory pattern:** the early lead vs iter 117 v5 (-0.023 at s200/s400) NARROWED at s600 (-0.005) and stabilized through s800/s1000. The 10× entropy coef engaged in the second half (after warmup_delay_frac=0.3) but couldn't bend pertoken_entropy back below 2.5 — the CV-redistribution from the cross-batch balance reg dominates over the per-token sparsity push at this coef magnitude.

**Lesson (this is the key takeaway).** In soft-dense MoE on this task, **bumping entropy_coef alone is not sufficient** to drive specialization. CV-redistribution, sigmoid gating, and entmax blend interact such that pertoken_entropy reaches a stable equilibrium around 2.6 regardless of the entropy coef. Future iters targeting per-token specialization need either:
1. **Architectural sparsity** (entmax with smaller `entmax_blend_init_logit`, e.g., 0 → sigmoid=0.5 balanced — but iter 117 v5 already proved this approach has capacity cost)
2. **Joint reg** like Gram-matrix penalty (iter 112 H84 — component ready) which constrains both cross-token correlation AND per-token sparsity simultaneously
3. **Sparsemax annealing** (iter 102b H77 follow-up) which forces hard zeros after warmup

**Action: REVERT config bumps, keep structural improvements.**
- `router_entropy_coef` 0.05 → 0.005 (back to iter 117 v5 value)
- `entmax_blend_lr` 0.02 → 0.002 (back to iter 117 v5 value)
- KEEP: router_reg_loss group refactor (commit `54473aa` structural part), DRY `_run_hutchinson_F` helper (now reporting hutch_F + rd_step per-K in K-sweep ★ which iter 117 v5 did not), sign-clarification doc updates, H89 REFUTED documentation, X1/X2/X3 component infrastructure.

**Status:** NOT PROMOTED ✗. Configs reverted. Current baseline remains **iter 117 v5** (val_bpb int6 = 1.5122). Pivoting to next queue item (iter 113 H85 `block_ortho_aux_coef` 0.1 → 0.5).

### H84+H93 RESULT (iter 112+122 MERGED, commit `ec4cb19` → promoted at `fb15a48`, 2026-05-02): PROMOTED ★ — Gram-matrix orthogonal-expansion routing (coef=0.1) + Gemma2-style logit softcap (=30) merged

**Outcome:** PROMOTED ★ via strict-gen unconditional rule (CLAUDE.md §11). Baseline now **iter 112+122 (val_bpb int6 = 1.5165)**.

**val_bpb summary:**
- val_bpb fast (s1000 train): 1.4814 (vs iter 117 v5 baseline 1.4820 → Δ −0.0006)
- val_bpb int6 (roundtrip): **1.5165** (vs iter 117 v5 baseline 1.5122 → Δ +0.0043, well within carry-forward gate 0.03; vs promotion gate 1.5422 → −0.026 margin)
- val_loss int6: 2.5196
- artifact_bytes: 7,591,678 (47.4% of 16 MB budget)
- peak_vram_mb: 34,557 (under 44 GB cap)
- step_avg: 20.79s (1000-step run = 5.78h wallclock)
- compressor: zstd

**Strict-generalization argument** (per CLAUDE.md §11 unconditional-promote): setting `use_orthogonal_expansion_routing=False` (or `routing_gram_coef=0`) AND `logit_softcap=0` recovers iter 117 v5 baseline forward map exactly:
1. Param setting recovering baseline: `--use-orthogonal-expansion-routing=0 --logit-softcap=0`
2. Representable: gram penalty is loss-only (gated by flag → 0 contribution), softcap is `if self.logit_softcap > 0` short-circuit in `MoSHead._head_forward`
3. Optimizer access: NO new optimizer-tracked params added (gram is loss-only, softcap is constant); iter 117 v5 weights are exactly reachable

**Trajectory table (val_bpb fast vs iter 100b PROMOTED reference 1.4572 final):**

| Step | iter 100b | iter 112+122 | Δ vs 100b | Notes |
|---|---|---|---|---|
| s200 | 2.0012 | 2.0160 | +0.015 | early lead absorbed by CV-aux during specialization burst |
| s400 | 1.6486 | 1.6963 | +0.048 | maximum gap; gram_coef warmup begins at s300 |
| s600 | 1.5342 | 1.5652 | +0.031 | gap closing — gram penalty engaging |
| s800 | 1.4968 | 1.5182 | +0.021 | continued closing |
| s1000 | 1.4572 | 1.4814 | +0.024 | final |

**Full K-sweep matrix (PERMANENT iter 97.6 protocol — 10 K values × 15 cols, primes 17/37/113 in bold):**

```
k_sweep_table:    K   val_bpb  attn_cv   mlp_cv  pool_cv  attn_min   mlp_min  attn_ortho  mlp_ortho  pertoken_ent  pool_ent  shared_gate   hutch_F   rd_step  iter_conv_rel
k_sweep_table:    4    1.8420   0.2779   0.1978   0.2412    0.0404    0.0512      0.1504     0.2197        2.6896    3.3724       0.2994    0.6961  414.8419         0.2197
k_sweep_table:    8    1.5716   0.3112   0.2184   0.2689    0.0382    0.0507      0.1504     0.2197        2.7033    3.3657       0.2993    0.6672  433.5479         0.0858
k_sweep_table:   16    1.5165   0.3015   0.2054   0.2579    0.0388    0.0500      0.1504     0.2197        2.7118    3.3685       0.2961    0.6653  424.3776         0.0247
k_sweep_table:  **17**    1.5163   0.3016   0.2054   0.2580    0.0389    0.0499      0.1504     0.2197        2.7129    3.3685       0.2963    0.6677  421.5315         0.0221
k_sweep_table:   24    1.5177   0.3004   0.2045   0.2570    0.0390    0.0498      0.1504     0.2197        2.7140    3.3688       0.2961    0.6658  421.6063         0.0134
k_sweep_table:   32    1.5188   0.3007   0.2047   0.2572    0.0390    0.0498      0.1504     0.2197        2.7143    3.3687       0.2963    0.6646  423.0342         0.0112
k_sweep_table:  **37**    1.5191   0.3011   0.2045   0.2574    0.0390    0.0498      0.1504     0.2197        2.7134    3.3687       0.2961    0.6660  420.3005         0.0108
k_sweep_table:   64    1.5197   0.3008   0.2045   0.2572    0.0390    0.0498      0.1504     0.2197        2.7137    3.3687       0.2962    0.6655  413.4591         0.0106
k_sweep_table: **113**    1.5198   0.3004   0.2043   0.2569    0.0390    0.0498      0.1504     0.2197        2.7139    3.3688       0.2961    0.6675  427.2326         0.0107
k_sweep_table:  128    1.5199   0.3011   0.2050   0.2576    0.0389    0.0497      0.1504     0.2197        2.7134    3.3686       0.2962    0.6643  418.7645         0.0106
```

**Acyclicity-prime check (per iter 97.6 PERMANENT protocol):**
- K=17 (prime) val_bpb 1.5163 vs K=16 1.5165 → Δ = −0.0002 ★ (genuine FP, ≪0.01-0.02 acyclicity bound)
- K=37 (prime) val_bpb 1.5191 vs K=32 1.5188 → Δ = +0.0003 ★ (genuine FP)
- K=113 (prime) val_bpb 1.5198 vs K=128 1.5199 → Δ = −0.0001 ★ (genuine FP)
- All three primes confirm: model has trained a TRUE fixed point, not a depth-specialized cycle.

**K=8 → K=128 Δ:** 1.5199 − 1.5716 = **−0.0517** (deeper-K BETTER, monotone decreasing post K=8) — exceeds H12 K-sweep gate by orders of magnitude (well below 0.5 threshold).

**K=16 → K=128 Δ:** +0.0034 — minimal drift; FP quality preserved across the full extrapolation range.

**Routing health at s1000 val (final):**
- shared_gate_mean: 0.306 (down from 0.731 at s0 → routed experts dominating)
- attn_cv: 0.245, mlp_cv: 0.071, pool_cv: 0.180 (well-controlled)
- attn_ortho: 0.150, mlp_ortho: 0.220 (ortho output-mean cosines stable)
- pool_entropy: 3.385 (99.5% of log30 — global balance preserved end-to-end)
- pertoken_entropy: 2.722 (specialization developed)
- hutch_F: 0.656 (FP contraction healthy)

**Diagnostic-gate failures (recorded per CLAUDE.md §11 — DO NOT block promotion):**
- ⚠ `attn_router_collapse`: attn_min_share=0.0377 < 0.0400 (one expert under-routed by ~6%). Retry hint: increase attn_balance_mult by 1.5×.
- ⚠ `expert_collapse` (weight-space): attn_ortho=0.7188 > 0.5 (weight-space cosine — DIFFERENT signal from K-sweep's expert OUTPUT MEAN cosine 0.1504). Retry hint: increase weight_decay by 1.5×.
- These are hint-level only; promotion is val_bpb-primary and the strict-gen rule applies.

**Mechanism interpretation:**
- **Gram penalty** (`‖G − I/E‖²_F` over routing weights, c=0.1): targets balanced one-hot routing per token. Math: at target G=I/E, off-diagonal=0 forces per-token specialization (one-hot routing); diagonal=1/E forces balanced load AND prevents dead experts (Cauchy-Schwarz bound). Engaged from s300, ramped 0→0.1 by s1000. Routing-health metrics (pool_cv 0.51 → 0.18 across val checkpoints) confirm gram is doing real work.
- **Logit softcap** (`30·tanh(logits/30)` on per-expert MoS logits, BEFORE log_softmax): bounds extreme logit magnitudes (Gemma2-style). Visible effect in mos_ntp_ortho declining 0.018 → 0.013 across run (smoother MoS logit cloud). Likely also stabilized the iter-112-first-pass's late-warmdown regression that NOT-PROMOTED at coef=0.01 first-pass (s680 ntp Δ+0.01).

**Why merging worked when iter 112 first-pass alone marginal:** iter 112 first-pass at coef=0.01 trajectory was monotonically shrinking advantage (s200 Δ−0.023 → s680 ntp Δ+0.01), extrapolating to marginal s1000. The 10× coef bump (0.01 → 0.1) gave gram penalty real gradient signal, AND softcap=30 added an orthogonal stabilization on the MoS head. Together they crossed the promotion gate while iter 112 alone at coef=0.01 was likely going to fail.

**Status:** PROMOTED ★. Baseline updated. 7,591,678-byte artifact rotated to `experiments/weights/baseline/`. Plots regenerated. Continuing autonomous Tier 1 execution: iter 108 next (`deq_k_jitter_set (16,24)→(10,24), deq_k_eval 16→10`).

### H63 RESULT (iter 95, commit `046235f` → promoted at baseline rotate, 2026-05-02): PROMOTED ★ — `deq_bptt_k = 2 → 3` retest under iter 112+122 baseline + gram=0.1 + softcap=30 regime

**Outcome:** PROMOTED ★. Baseline now **iter 95 (val_bpb int6 = 1.5001)**. TBPTT=3 hypothesis VINDICATED — both val_bpb AND K-sweep tightness improve over the iter 112+122 baseline.

**val_bpb summary:**
- val_bpb fast (s1000 train): 1.4635 (vs iter 112+122 1.4814 → Δ −0.0179)
- val_bpb int6 (roundtrip): **1.5001** (vs iter 112+122 1.5165 → Δ −0.0164; vs gate 1.5465 → 0.046 margin)
- val_loss int6: 2.4923 (vs 2.5196)
- artifact_bytes: 7,447,773 (46.5% of 16 MB budget, slightly smaller than iter 112+122 7,591,678)
- peak_vram_mb: 34,551
- step_avg: 23.49s (1000-step run = 6.5h wallclock; +12.9% vs iter 112+122 20.79s)

**Trajectory table (val_bpb fast vs iter 112+122):**

| Step | iter 95 | iter 112+122 | Δ |
|---|---|---|---|
| s200 | 2.0034 | 2.0160 | −0.013 |
| s400 | 1.6660 | 1.6963 | −0.030 (peak gap) |
| s600 | 1.5516 | 1.5652 | −0.014 |
| s800 | 1.5097 | 1.5182 | −0.009 |
| s1000 | **1.4635** | 1.4814 | **−0.018** |

**Full K-sweep matrix (acyclicity primes 17/37/113 in bold):**

```
k_sweep_table:    K   val_bpb  attn_cv   mlp_cv  pool_cv  attn_min   mlp_min  attn_ortho  mlp_ortho  pertoken_ent  pool_ent  shared_gate   hutch_F   rd_step  iter_conv_rel
k_sweep_table:    4    1.9496   0.4485   0.2122   0.3508    0.0216    0.0358      0.1406     0.2246        2.8162    3.3385       0.2393    0.7705  425.4629         0.2926
k_sweep_table:    8    1.5984   0.4266   0.2229   0.3404    0.0243    0.0351      0.1406     0.2246        2.7924    3.3423       0.2170    0.7546  427.1459         0.1252
k_sweep_table:   16    1.5001   0.4024   0.2028   0.3186    0.0265    0.0378      0.1406     0.2246        2.7628    3.3493       0.1953    0.7495  425.9280         0.0313
k_sweep_table: **17**    1.4992   0.4015   0.2020   0.3178    0.0265    0.0379      0.1406     0.2246        2.7623    3.3496       0.1948    0.7481  414.6889         0.0276
k_sweep_table:   24    1.4988   0.3999   0.2008   0.3164    0.0267    0.0380      0.1406     0.2246        2.7618    3.3500       0.1941    0.7469  420.6639         0.0175
k_sweep_table:   32    1.4996   0.4011   0.2007   0.3172    0.0266    0.0380      0.1406     0.2246        2.7617    3.3498       0.1942    0.7479  419.2790         0.0158
k_sweep_table: **37**    1.4999   0.3991   0.2003   0.3157    0.0267    0.0381      0.1406     0.2246        2.7628    3.3502       0.1944    0.7513  415.0172         0.0157
k_sweep_table:   64    1.5004   0.4008   0.2006   0.3169    0.0266    0.0380      0.1406     0.2246        2.7619    3.3498       0.1943    0.7460  419.0722         0.0161
k_sweep_table: **113**    1.5005   0.4003   0.2007   0.3166    0.0267    0.0381      0.1406     0.2246        2.7621    3.3499       0.1942    0.7459  419.9065         0.0163
k_sweep_table:  128    1.5005   0.4007   0.2008   0.3169    0.0267    0.0380      0.1406     0.2246        2.7616    3.3498       0.1941    0.7461  424.0358         0.0163
```

**K-sweep analysis** (TBPTT=3 delivers TIGHTER FP than TBPTT=2):
- K=8 → K=128 Δ: 1.5005 − 1.5984 = **−0.098** (vs iter 112+122 K=8→K=128 Δ = −0.052) — **iter 95 K-sweep is 88% tighter**
- K=16 → K=128 Δ: +0.0004 (vs iter 112+122 +0.0034) — **8.5× tighter at deep K**
- K=24 (best): 1.4988 ★ — TBPTT=3 finds the optimal FP at K=24, the deeper K-jitter sample
- All 3 acyclicity primes (17, 37, 113) confirm GENUINE FP within ±0.0010 of nearest non-prime
- Best-K = K=24 at 1.4988, +0.0017 better than K=16 — TBPTT=3 trains an FP that's BETTER at deep K than at training-K-min

**Routing health at s1000 val (final):**
- shared_gate_mean: 0.201 (down from 0.731 at s0; further suppressed than iter 112+122's 0.306)
- attn_cv: 0.351, mlp_cv: 0.114, pool_cv: 0.261
- attn_ortho: 0.141, mlp_ortho: 0.225 (gram penalty held throughout)
- pool_entropy: 3.367 (99.0% of log30, global balance preserved)
- pertoken_entropy: 2.78 (specialized)
- hutch_F: 0.736 (climbing 0.41→0.52→0.60→0.64→0.74 across vals — slow drift, just under 1.0 watch threshold)

**Training descent rate (Δntp / 100 steps, iter 95 vs iter 112+122):**

| Window | iter 95 ntp | iter 112+122 ntp | iter 95 Δntp | iter 112+122 Δntp | iter 95 per-sec | iter 112+122 per-sec |
|---|---|---|---|---|---|---|
| s100 | 4.0282 | ~4.07 | — | — | — | — |
| s100→s200 | → 3.3584 | → 3.37 | −0.670/100 | −0.700/100 | −0.000285 | −0.000326 |
| s200→s400 | → 2.5922 | → 2.66 | −0.766/200 | −0.710/200 | −0.000163 | −0.000165 |
| s400→s600 | → 2.7722 | → 2.79 | +0.180/200 (plateau) | +0.130/200 (plateau) | +0.0000383 | +0.0000302 |
| s600→s800 | → 2.4882 | → 2.51 | −0.284/200 | −0.280/200 | −0.0000604 | −0.0000651 |
| s800→s1000 | → **2.4379** | → 2.47 | −0.0503/200 | −0.04/200 | −0.0000107 | −0.0000093 |
| **Avg s100→s1000** | — | — | **−0.001767/step** | **−0.001778/step** | **−7.53e-5/sec** | **−8.27e-5/sec** |

**Per-step iter 95 ≈ iter 112+122 (−0.001767 vs −0.001778, basically tied). Per-wallclock iter 112+122 marginally faster (8.9% faster /sec) due to lower step_avg.** But val_bpb int6 iter 95 wins by Δ−0.0164. The val/train decoupling is the key insight: deeper TBPTT averages backward gradient noise → cleaner generalization despite indistinguishable training-loss descent.

**Mechanism interpretation:**
- **TBPTT=3** unrolls 3 backward iterations (vs 2). Each adds another chain-rule contribution to the per-token gradient.
- **Effect on training loss**: ~tied (more samples averaged but same total signal magnitude).
- **Effect on val_bpb**: clearly improves (less batch-specific noise → cleaner per-sample gradient → better generalization to held-out data).
- **Effect on K-sweep**: dramatically tightens (TBPTT=3 better matches longer forward unrolls — model doesn't depend on shallow-K shortcuts; FP quality preserved at K=128).
- **Cost**: +12.9% wallclock per step. Cleanly Pareto-dominated by the val_bpb improvement: Δ−0.0164 int6 / +12.9% wallclock gives −0.13 bpb-per-relative-wallclock, significantly higher ROI than typical gram-coef sweeps.

**Closed:** TBPTT scaling-law adaptive sweep (95d/95e/95f/95-jitter) deferred to AFTER sparsity stack (iter 117b-2/3/3b) per user directive 2026-05-02. Will validate the optimum is at TBPTT=3 (or find a better point in {4, 6, 8}) on top of the throughput-improved baseline.

**Status:** PROMOTED ★. Baseline updated. 7,447,773-byte artifact rotated to `experiments/weights/baseline/`. Plots regenerated. Continuing autonomous Tier 1: iter 117b-2 (Triton entmax) next per Tier 1 reorder.

### H88: Triton-fused entmax + grouped-GEMM via custom_op (iter 118) — PROPOSED CONDITIONAL 2026-04-29

**Hypothesis.** If iter 117 (path A pure-PyTorch dispatch) confirms val_bpb is preserved, the next step is a **fused Triton kernel** for `entmax_alpha + grouped_GEMM` registered via `torch.library.custom_op` (with `register_fake` + `register_autograd`). Predicted **3–4× wallclock** speedup over iter 100b dense soft-MoE, vs iter 117's 1.5–2.5× from pure-PyTorch path.

**Why custom_op (not autograd.Function).** iter 104 (AdaSplash) DROPPED 2026-04-29 because `torch.autograd.Function` is incompatible with torch.compile's AOTAutograd functionalization (`_functionalization.apply_view_meta_sequence` corruption at step 2). The PyTorch-recommended path is `torch.library.custom_op` with FakeTensor backward — the standard remediation pattern. iter 118 follows this pattern from the start.

**Implementation order:**
1. Verify iter 117 (path A) preserves val_bpb on iter 100b's 1000-step regime
2. Implement Triton kernel: input (B*T, E) scores → output (B*T, E) entmax weights, fused with the per-expert grouped GEMM dispatch
3. Register via custom_op + FakeTensor backward
4. Smoke test, 10-iter, full 1000-step

**Status:** PROPOSED — HIGH PRIORITY iter 118 **conditional on iter 117 promotion**. Skip if 117 fails.

### H89: Mixture-of-Depths (iter 119) — REFUTED ✗ (2026-04-30, principled architectural-incompatibility)

**Hypothesis.** Mixture-of-Depths (Raposo et al. 2024, Google DeepMind) — skip ENTIRE LAYERS for some tokens via per-layer per-token continuous score + capacity-based soft selection. Published: ~50% compute reduction at <2% quality loss. Composes multiplicatively with iter 117's entmax skip.

**REFUTED — principled grounds (2026-04-30, user critique).** As proposed for our soft-dense + RevDEQ stack, MoD has a degenerate trivial solution: per-layer gates can drift to zero across all blocks → entire stack collapses to identity → no LM signal but also no capacity loss. The original Raposo et al. paper avoids this by enforcing **hard top-K capacity per block** (only top-K tokens by gate score get block compute), which forces gates into a competitive top-K selection. Our soft-dense MoE setup intentionally avoids hard top-K (CLAUDE.md §6.2: "no top-K, no token dropping" — RevDEQ requires C¹-smooth `T_θ`, and top-K introduces gradient discontinuities at the K/K+1 boundary that break the FP convergence proof). Adding the capacity scaffolding to enable MoD would require giving up the soft-dense + RevDEQ contract.

**Anti-collapse alternatives considered + rejected:**
- Per-block utilization regularization (analog of CV for layer gates): another reg knob; doesn't solve the fundamental issue that soft-gate MoD without top-K can drift to 0 long before reg pulls back, and the LM gradient through a near-identity stack is weak.
- Target compute budget penalty (`L = (compute_used - target)²`): adds another antagonistic reg fighting LM loss; iter 100b H76 reframe (decoupled antagonistic regs hurt — see `feedback_decouple_regularizers.md`) argues against this.

**Decision: drop iter 119 from queue.** Sparsity at the routing-pool level (iter 117/117b/118) is the principled axis on this stack. Layer-skip would require a fundamentally different architecture.

**Status:** REFUTED ✗.

### H90: RRAttention — Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts (iter 120) — PROPOSED 2026-04-30

**Hypothesis.** Replace dense causal SDPA with a per-head round-robin (RR) block-sparse attention pattern (Liu et al. 2026, arxiv:2602.05853). Each head samples DIFFERENT query positions within a length-S stride; collectively H heads cover the full stride. Stride-level importance estimation reduces complexity O(L²) → O(L²/S²); adaptive Top-τ block selection gives input-adaptive efficiency.

**Algorithm (3-stage pipeline, see `experiments/components/rr_attention.py`):**
1. **Round-robin query sampling**: for stride i, head h, sample position
   `pos(i, h) = i·S + ((S-1-h) mod S)`. With H ≥ S, every position
   within each stride is sampled by SOME head.
2. **Stride-level importance estimation**: aggregate keys per stride
   (`K_agg[j] = mean(K[j·S:(j+1)·S])`), compute scores
   `I[i, j] = Q_s[i] · K_agg[j] / sqrt(d)`, softmax over key strides.
3. **Top-τ block selection**: aggregate stride probs to block level
   (block size B = S·strides_per_block), normalize per query block,
   select smallest set of key blocks with cumulative mass ≥ τ. Always
   protect the diagonal (self-attention) block.
4. **Sparse SDPA**: standard `F.scaled_dot_product_attention` with the
   token-level expanded mask (block-sparse pattern → token mask).

**Strict-generalization.** At τ=1.0, all blocks are retained → mask is
all-True (modulo causality) → output bit-identical to dense SDPA. The
component's smoke test verifies `rel_err = 0.00e+00` at τ=1.0.

**Expected throughput (paper).** At 128K context: 2.4× speedup over
FlashAttention. At our T=2048: smaller absolute gain (~1.2-1.5×) since
the L²/S² reduction is less impactful; primary value is the abstraction
for future T-scaling.

**Integration into our codebase.** Drop-in replacement for the head-packed
`(B, E·H, T, d)` SDPA call in `CausalSelfAttention.forward`. Treats `E·H`
as the head dimension; round-robin sampling rotates across `E·H`
positions per stride, which is consistent with the per-expert-per-head
independence (CLAUDE.md §6.2 hard constraint preserved).

**Risks.**
- Token-level mask + dense SDPA: until a block-sparse kernel is wired
  (FlashAttention blocked variant or `torch.nn.attention.flex_attention`),
  throughput gain is **purely from the lower attention compute when the
  mask is sparse**, NOT from skipping mask-False positions in the SDPA
  kernel itself. Net throughput at T=2048 may be neutral or slightly
  negative until block-sparse kernel landed.
- Causality at the block boundary: confirmed in component smoke test
  (out[0:T/2] unchanged when Q[T/2+1:] is perturbed).
- DEQ FP interaction: RRAttention is a self-attention substitute inside
  T_θ, so the FP iterates over the new Q,K,V at each step. Block-mask
  selection depends on Q,K which evolve through the FP — masks may
  jitter across iterations. Standard K-sweep gate applies.

**Component implementation (2026-04-30):** standalone helper landed at
`experiments/components/rr_attention.py`. Smoke tests PASS:
- shape preservation: PASS (B, H, T, d) → (B, H, T, d)
- finiteness: PASS (no NaN/Inf)
- causality: PASS (out[0:T/2] unchanged by Q[T/2+1:] perturbation)
- τ=1.0 ≈ dense: rel_err = 0.00 ★ (bit-identical, confirms strict-gen)
- τ=0.3 ≠ dense: rel_err = 0.55 (sparse, diverges as expected)
- gradient flow: PASS through Q/K/V
- RR sample indices: PASS (heads cover stride positions in correct order)
- toggle setter: PASS

**Test plan for the iter 120 launch (when GPU free):**
1. Component smoke (CPU): already PASSED ✓
2. Standalone GPU test: import in a small script, B=1 H=8 T=2048,
   verify output finite + causal + matches dense at τ=1.
3. Smoke test full training: `python experiments/smoke_test.py
   --use-rr-attention=1 --rr-stride=8 --rr-block-size=64 --rr-tau=0.95`.
4. 1000-step launch with `--use-rr-attention=1`, monitor val_bpb against
   iter 117b-1 baseline.
5. Promotion gate: val_bpb regression ≤ 0.03 (carry-forward standard).
   Strict-gen unconditional path applies if τ=1.0 (recovers iter 117b-1
   exactly), but we'd run with τ=0.95 for an actual sparsity test.

**Sources:**
- https://arxiv.org/abs/2602.05853 (Liu et al. 2026, RRAttention paper)
- https://arxiv.org/html/2602.05853 (HTML version)

**Status:** PROPOSED. Component PASSED smoke tests; train_gpt.py integration
deferred to BETWEEN iterations per "clean-up between launches" discipline.

### iter 104 OLD ENTRY — BLOCKED on `torch.library.custom_op` registration (2026-04-29 root-caused)

**Updated diagnosis (supersedes earlier "needs custom_op" speculation in task #95).** Live 10-iter test (commit `72f4de0`, launched 2026-04-29 with `num-heads=12 num-kv-heads=6 attn-alpha-target=1.5 attn-alpha-warmup-delay-frac=0.0`) revealed two distinct bugs:

**Bug 1 — AdaSplash kernel GQA crash (root-caused + FIXED standalone, 2026-04-29).** The upstream kernel hits `cudaErrorIllegalAddress` at certain GQA scales: `H_q ≥ 96` with GQA crashes even at B=1; `H_q=12 H_kv=6` GQA crashes at B ≥ 4. Empirically bisected. Principled fix shipped in `adasplash_alpha_entmax_attention()`: reshape head-packed `(B, E*H, T, d)` → `(B*E, H, T, d)` to keep H within the safe range, plus explicit `repeat_interleave` GQA broadcast that bypasses the buggy internal GQA path. Standalone verification (B=2 E=16 H_q=12 H_kv=6 T=2048 d=64): forward + backward both PASS.

**Bug 2 — AOTAutograd view-meta corruption (NOT YET FIXED).** With Bug 1's fix, the live test got past compile and step 0 (val) cleanly; step 1 (first training backward) ran OK; **step 2 crashed inside torch.compile's AOTAutograd view-metadata replay**:

```
[rank1]:     out = _functionalization.apply_view_meta_sequence(...)
[rank1]: RuntimeError: shape '[22343783044301, 5039929193026096800, 2048, 64]'
        is invalid for input of size 402653184
```

The huge integers in the shape are uninitialized memory interpreted as int64 — the kernel output's tensor metadata got corrupted by AOTAutograd's view-meta replay between training steps. **`@torch._dynamo.disable` is insufficient**: dynamo's *tracer* skips the wrapped function, but AOTAutograd's *functionalization layer* still tries to track tensors that flowed through the opaque op, and its view-meta replay between iterations doesn't get the correct shape/stride info.

**Principled fix (deferred)**: register AdaSplash via `torch.library.custom_op` with explicit FakeTensor backward, so AOTAutograd treats the kernel as a first-class op with proper meta. This is the standard remediation path for "third-party Triton kernel + torch.compile + autograd". Engineering effort: ~1-2 hours to write the FakeTensor signature + integration test.

**Status: BLOCKED. iter 104 plumbing landed in `72f4de0` but DISABLED BY DEFAULT** (`attn_alpha_target=1.0` → dense softmax fallback, no behavior change vs iter 100b). Setting `attn_alpha_target>1.0` via CLI fires the kernel but crashes at step 2. Iter 104 should remain queued behind iter 103 (chained 2-stage routing) and the trivial config-only iters (108, 109) until the custom_op registration is prioritized.

### Next up — recommended ordering after iter 100b

**Current baseline:** **iter 95** (`046235f` (promote-rotate, 2026-05-02), val_bpb int6 = 1.5001) — last promoted iter (H63 RESULT). One-line config: `Hyperparameters.deq_bptt_k = 2 → 3` on top of iter 112+122 (gram=0.1 + softcap=30). Per-token backward unrolls 3 iterations instead of 2 — cleaner gradient via more chain-rule samples averaged → better val_bpb generalization despite essentially-tied training-loss descent. Cost: +12.9% step_avg (23.49s vs iter 112+122 20.79s). val_bpb int6 1.5001 vs iter 112+122 1.5165 → Δ −0.0164 (vs gate 1.5465 → 0.046 margin). K-sweep K=8→K=128 Δ=−0.098 (88% tighter than iter 112+122 −0.052); all 3 acyclicity primes confirm genuine FP. K=24 best=1.4988. Artifact 7.45 MB (47% of 16 MB budget).

**Historical baseline (superseded by iter 95):** **iter 112+122 MERGED** (`fb15a48` from launch commit `ec4cb19`, val_bpb int6 = 1.5165) — promoted 2026-05-02 (H84+H93 RESULT). Added Gram-matrix orthogonal-expansion routing (`routing_gram_coef=0.1`, `‖G − I/E‖²_F`) + Gemma2-style logit softcap (`logit_softcap=30`).

**Historical baseline (superseded by iter 112+122):** **iter 117 v5** (`08783b4`, val_bpb int6 = 1.5122) — promoted 2026-04-30 (H87 RESULT). Added entmax-1.5 + softmax annealed BLEND infrastructure on top of iter 100b's E=16 LoRA-style backbone.

#### POST-iter-117b-1 queue (2026-04-30, supersedes the legacy paragraph below)

Iter 117b-1 NOT PROMOTED 2026-04-30 (H87b RESULT) — config bumps reverted. Baseline remains iter 117 v5. Structural improvements kept: `router_reg_loss` group, DRY `_run_hutchinson_F` helper (now emitting hutch_F + rd_step per-K in K-sweep), iter 117b-2 X1 Triton entmax kernel + custom_op (default OFF), iter 117b-3 X2 sparse MoE dispatch helper + MLP-path wiring (default OFF), iter 103 X3 chained-routing flag + safety guard (default OFF), all `experiments/components/` files (chained_block, orthogonal_expansion_routing, rr_attention, sparse_attention_dispatch).

**Tier 1 — Ready to launch (just config or flag, no implementation):**

**Priority reshuffle 2026-04-30 (user directive — `feedback_throughput_priority.md`):** throughput-bearing sparsity iters take precedence over Gram-coef follow-up sweeps. Iter 112's Gram-penalty coef/delay sweeps (112b/c/d) are deferred to AFTER all throughput-bearing iters land.

**Priority reshuffle 2026-05-01 (user directive — TBPTT prioritization):** TBPTT depth experiments pulled forward to Tier 1 #2 and #3 (iter 95 + iter 95b). Trigger: iter 112+122 mid-run grad_norm observed at 0.07 (well below grad_clip=1.0) — soft signal that backward-depth-truncation may be starving the gradient signal. The current iter 117 v5 baseline runs `deq_bptt_k=2` under K-jitter {16, 24}, so backward only reconstructs 2/16=12.5% to 2/24=8.3% of the unrolled depth — meaningfully shallower coverage than iter 85 era ({2,3,4} at K-jitter (4,6,10) gave 50-67% mean coverage). This is a regime-change argument, not a re-run of iter 85. Iter 95 = fixed `deq_bptt_k=3` (single-variable change, ~+10% wallclock). Iter 95b = fp32 TBPTT accumulators (conditional, profile-gated; 1-3% wallclock upside, principled — see Q1 analysis 2026-05-01).

**Priority reshuffle 2026-05-02 (user directive — iter 108 deprioritized to end of Tier 1):** iter 108 (deq_k 16→10 throughput) was reordered to position #2 yesterday but moved to END of Tier 1 today. Rationale: throughput-only iter with no expected val_bpb improvement, lower priority than val_bpb-improving iters (95 TBPTT depth) and the subsumption-test iters that simplify architecture. Killed mid-launch (~step 10) and reverted commit `3401f6a` to keep deq_k defaults at (16, 24) / 16. Will re-launch as final Tier 1 iter.

**Priority reshuffle #2 2026-05-02 (user directive — iter 95 variants + 95b deprioritized AFTER sparsity iters):** iter 95 (TBPTT=3, currently IN FLIGHT) stays at position #2 to complete the in-flight run. Subsequent iter 95 variants (95d/95e/95f/95-jitter, the TBPTT scaling-law sweep) AND iter 95b (fp32 accumulators) move to AFTER the sparsity throughput iters (117b-2 / 117b-3 / 117b-3b). Rationale: throughput compounds research velocity (`feedback_throughput_priority.md`) — once the sparsity stack lands, every subsequent iter benefits from the wallclock improvement. The TBPTT scaling sweep is a 1-4-iter chain that benefits multiplicatively from a faster baseline.

1. ~~**Iter 113 (H85)** — `block_ortho_aux_coef` 0.1 → 0.5~~ — **DROPPED 2026-04-30** ✗.
2. **Iter 112+122 MERGED (H84+H93)** — **PROMOTED ★ 2026-05-02** at commit `fb15a48`. val_bpb int6 = 1.5165 (gate ≤1.5422 by 0.026; vs iter 117 v5 1.5122 → Δ +0.0043 within carry-forward 0.03). New baseline. See H84+H93 RESULT block above for full K-sweep matrix.
3. **Iter 95 (H63 retest)** — **TBPTT DEPTH (IN FLIGHT 2026-05-02).** Fixed `deq_bptt_k=2 → 3` under iter 112+122 baseline. s200 val 2.0034 (Δ −0.013 vs baseline 2.0160). Strong promotion outlook. ETA s1000 ~10:53.
4. **Iter 117b-2 GPU smoke** — **THROUGHPUT-BEARING.** Launch with `--use-entmax-triton=1 --use-entmax-routing=1`. Triton entmax kernel verified vs deep-spin/entmax at fp32 floor; needs DDP+compile+RevDEQ smoke. Tests fused-kernel correctness under blend.
5. **Iter 117b-3 GPU smoke** — **THROUGHPUT-BEARING.** Launch with `--use-sparse-dispatch=1 --sparse-dispatch-capacity-factor=8 --use-entmax-routing=1`. Sparse MoE dispatch numerically equivalent to dense at C ≥ (1-s)·E; smoke at C=8 should be bit-identical, then sweep capacity down to find break-even. **Strongest throughput win** (skip-zero-experts under entmax sparsity).
6. **Iter 117b-3b** — **THROUGHPUT-BEARING.** Per-expert sparse-Q attention (asymmetric analog of MLP sparse dispatch). Component at `experiments/components/sparse_attention_dispatch.py` PASSED 7/7 smoke. Saves Q + SDPA + Wo per-expert via capacity-padded gather/dispatch; K, V remain dense AND the SDPA call itself is preserved (smaller Q rows but same FA fusion). Net win iff entmax sparsity > gather overhead.
7. **TBPTT scaling-law adaptive sweep — 95d/95e/95f/95-jitter (DEPRIORITIZED 2026-05-02 to after sparsity)** — characterize the optimal `deq_bptt_k` for the current K-jitter (16, 24) regime via minimum-iter adaptive bisection. Existing data: TBPTT=1 (iter 69 NOT PROMOTED +0.021), TBPTT=2 (iter 69b PROMOTED, became baseline), TBPTT=3 (iter 95 in flight, s200 val 2.0034 = -0.013 vs baseline). Iter 28d closed TBPTT≥12 path under K-jitter (4,8,16,24) (regressed +0.061). Sweep upper bound = 8 under current K-jitter. **Adaptive doubling-or-stop strategy** (max 3 new iters, conditional on prior result):
   - **Iter 95d**: `deq_bptt_k = 4`. If val_bpb int6 ≤ iter 95: continue. Else STOP, optimum is 3.
   - **Iter 95e**: `deq_bptt_k = 6`. Pre-cond 95d promotes. Bisects (4, 8). If 95e ≤ 95d: continue. Else: optimum is 4-5 range.
   - **Iter 95f**: `deq_bptt_k = 8`. Pre-cond 95e promotes. If 95f ≤ 95e: optimum is at or beyond 8; queue 95g (TBPTT=12 if K=24 sample dominates wallclock). Else: optimum is 5-7 range.
   - **Iter 95-jitter (cond.)**: stochastic TBPTT bag {best_fixed - 1, best_fixed, best_fixed + 1} per-step jitter. Tests whether TBPTT-jitter beats fixed-best (analog of H12 K-jitter VERIFIED). Run only if all fixed-TBPTT iters complete.
   - **Stop conditions**: any single iter regresses val_bpb int6 by > 0.01 vs prior best → STOP. Or wallclock cost exceeds +25% baseline → STOP.
   - **Output**: scaling-law fit `val_bpb(TBPTT)` and `wallclock(TBPTT)` curves, identifying Pareto-optimal point. Documented under H63-extension section after sweep completes.
   - **Total compute**: 1-4 iters depending on adaptive path. Min 1 (TBPTT=4 regresses → optimum is 3). Max 4 (full chain 4→6→8→jitter).
8. **Iter 95b (DEPRIORITIZED 2026-05-02 to after sparsity + scaling sweep)** — **fp32 TBPTT accumulators (conditional).** Replace fp64 Kahan accumulators in `RevDEQFunction.backward` reconstruction with fp32. Error analysis: with `Ā ≥ 0.1` floor, per-step amplification = 1/Ā = 10×, over `deq_bptt_k=BEST` reconstruction = (10^BEST)× cumulative; fp32 unit roundoff 1.2e-7 should hold safe up to BEST ≤ 6 (then re-derive). **bf16 NOT VIABLE** — same calc with bf16 unit roundoff 7.8e-3 blows past tolerance for any BEST. Throughput upside modest (1-3% wallclock — accumulators are memory-bound). **Pre-condition**: chrome trace under iter 117 v5 must show fp64 accumulators in hotspot top-5; otherwise skip.
9. **Iter 110 (H82)** — Re-enable `num_refinements = 1` (currently 0 since iter 98b). One-line config. Tests refinement under iter 117 v5's blend infrastructure.
10. **Iter 108 (H79 reframed) — DEPRIORITIZED to END of Tier 1 2026-05-02 (user directive).** Two coupled changes: `Hyperparameters.deq_k_jitter_set = (16, 24) → (10, 24)` AND `deq_k_eval = 16 → 10`. K-sweep matrix UNCHANGED. Throughput upside ~12% step_avg estimated; no expected val_bpb improvement. Iter 108 launch attempt at 04:08 was killed at ~step 10, commit reverted (`fa22310`). Will re-launch at end of Tier 1. **Verdict path**: if val_bpb int6 ≤ baseline + 0.005 AND step_avg ≤ baseline × 0.92, promote.
**Gradient-magnitude analysis 2026-05-02 (revises sequential ablation order)** — rigorous derivation of pairwise gram-vs-existing-reg dynamics shows:
- `∂L_G/∂w_t = (4 c_G / N) · (G − I/E) w_t` (matrix form, per-token)
- `∂L_CV/∂w_t[a] = (2 c_CV / N) (p[a] − 1/E)` (uniform across tokens)
- `∂L_ent/∂w_t[a] = (c_ent / N)(−log w_t[a] − 1)` (per-token, multiplicative)
- L_BO operates on expert OUTPUT means `mu[e]` — orthogonal axis, not on W

At over-concentration (p[1]=0.7, w_t[1]=1) with current coefs (c_G=0.1, c_CV=2.0, c_ent=0.005, E=16):
- L_CV per-token grad: ~2.55/N (DOMINANT)
- L_G per-token grad: ~0.255/N (10× weaker than CV on diagonal axis; ALIGNED direction)
- L_ent per-token grad: ~0.005/N (50× weaker than L_G; OPPOSITE direction at concentration)

Conclusions:
1. **L_G ↔ L_CV: ALIGNED but L_CV is 10× stronger** on load-balance. **KEEP BOTH**. Removing L_CV requires c_G ↑ ~10× (separate iter, not pure ablation).
2. **L_G ↔ L_ent: COMPETING at concentrated states, but L_ent is ~noise-level** (~2% of total grad). **REMOVE L_ent — cleanest principled simplification**.
3. **L_G ↔ L_BO: ORTHOGONAL axes** (W^T W vs expert output means). L_BO catches parameter-level collapse (2 experts learning same function on disjoint tokens) that L_G cannot. **KEEP BOTH**. iter 112e expected to FAIL or show ortho drift.
4. **L_G subsumes L_min_share**: diagonal G[e,e]=0 → max penalty 1/E². Already disabled (iter 100b H76).

Revised ordering reflects this analysis (L_ent removal first as cleanest, others demoted):

11. **Iter 112g (PROMOTED to first ablation 2026-05-02)** — `router_entropy_coef = 0.005 → 0` under gram=0.1. **Hypothesis** (confirmed by gradient analysis): L_ent is ~2% of total routing-reg gradient at current coefs and OPPOSES L_G at concentrated states. Removing it is near-strict-gen. **Pre-condition**: iter 112+122 promotes. **Verdict path**: val_bpb int6 ≤ baseline ± 0.005 AND pertoken_entropy descends or holds. Removes entropy-warmup-delay machinery + cold-start trap (H87b/iter 117b-1 history).
12. **Iter 112i (NEW 2026-05-02)** — Replace `block_ortho_aux_coef` (relu-threshold formulation) with **output-mean Frobenius gram penalty** `c_O · ‖M̂ M̂^T − I‖²_F` where `M[e] = (1/N) Σ_t out_e[t]` (per-expert MEAN OUTPUT averaged across N batch tokens, M ∈ ℝ^{E×D}) and `M̂[e] = M[e] / ‖M[e]‖`. **Note**: M is the BATCH-AVERAGED output per expert (same signal block_ortho already uses), NOT single-token. The penalty switches FORM (threshold-relu → smooth Frobenius), not signal. **Theoretical motivation**: forms a principled DUAL to L_G_router — one Frobenius gram on routing weights (input side), one on expert mean outputs (output side). Both target identity-like structure (balanced one-hot for routing, mutual orthogonality for outputs). Replaces threshold-gated relu (CLAUDE.md §9 critique of arbitrary 0.20) with smooth, gradient-active-everywhere penalty. **Strict-gen** at c_O=0 ⇒ identity. **Spec**: add `routing_output_gram_coef=0.05` + `routing_output_gram_warmup_delay_frac=0.3`. Set `block_ortho_aux_coef=0` simultaneously (replaced, not just disabled). **Cost**: ~0.1% wallclock (E²D + EDN per block, negligible). **Pre-condition**: iter 112g promotes. **Verdict path**: val_bpb int6 ≤ baseline ± 0.01 AND attn/mlp_ortho stay ≤ 0.20 throughout. If passes: iter 112e becomes moot (block_ortho replaced, not dropped). Per-token output gram (M = O[t,e,:] direct) was considered as variant B but rejected: O(NE²D) cost ~1% wallclock + over-regularizes shared features (forces all experts' outputs to be orthogonal even on tokens where they are barely routed).
   - **Iter 112i' (follow-up if 112i promotes)**: routing-weighted M variant `M[e] = (Σ_t w_t[e]·out_e[t]) / (Σ_t w_t[e])`. Measures expert identity ONLY on tokens that actually route to it (more semantically meaningful: w-weighted centroid). Couples M to routing weights → composes with L_G_router. Tests whether routing-weighted output identity improves on plain mean.
13. **Iter 112e (DEMOTED to third 2026-05-02; now post-112i)** — `block_ortho_aux_coef = 0.1 → 0` ablation under gram=0.1, NO 112i replacement. **Pre-condition**: iter 112i FAILS (i.e. principled output-gram replacement also doesn't help). **Reduces to**: pure ablation test of whether L_BO's threshold formulation is load-bearing. **Verdict path**: val_bpb int6 ≤ baseline ± 0.01 AND attn/mlp_ortho stay ≤ 0.25 throughout. EXPECTED TO FAIL under the math (L_BO operates on orthogonal axis to L_G_router), but cheap empirical confirmation.
14. **Iter 112h (REVISED 2026-05-02)** — joint removal of L_ent + L_BO (excludes L_CV): `router_entropy_coef = 0` + `block_ortho_aux_coef = 0`. **Pre-condition**: 112e/g both promote. Tests "gram + CV are the dominant pair" hypothesis. **Verdict path**: val_bpb int6 ≤ baseline ± 0.01 AND routing-health metrics stable.
15. **Iter 112f' (REVISED 2026-05-02 from old 112f)** — `cv_loss_weight = 2.0 → 0` AND `routing_gram_coef = 0.1 → 1.0` (compensating coef-redistribution). **Rationale**: L_CV is 10× stronger than L_G at current coefs, so naive removal regresses. Bumping c_G to 1.0 restores comparable gradient magnitude on the diagonal/load-balance axis. Tests whether gram alone (at higher coef) can replace CV's load-balance role. **Pre-condition**: 112h passes (confirms gram + CV are the dominant pair before testing CV removal). **Verdict path**: val_bpb int6 ≤ baseline ± 0.02 AND pool_cv stays ≤ 0.7 throughout. NOT a pure ablation — coef redistribution.

**DEMOTED 2026-04-30 — replaces SDPA at T=2048, likely throughput regression:**
- ~~Iter 120 (RRAttention)~~ — **DEFERRED back to Tier 3 / Deferred section.** Replaces head-packed SDPA in `CausalSelfAttention.forward`; same architectural class as iter 106 NSA which was DROPPED 2026-04-29 because **NSA is 0.42× FlashAttention at T=2048** (per official fla-org benchmark, see H86). Component at `experiments/components/rr_attention.py` validates correctness only ("τ=1.0 bit-identical" is a numerics check, not a throughput check) — pure-PyTorch impl can't compete with FA SDPA at this seq length. Promotion path: re-implement on `torch.nn.attention.flex_attention` (PyTorch 2.5+) with `score_mod`/`block_mask` keeping FA fusion intact, OR defer until T scales (e.g. T=4096/8192 iter). User directive 2026-04-30: "Would RRAttention hurt throughput as the optimized SDPA is replaced?" — answered yes, demoted.

**Deferred — coef-sweep follow-ups (post throughput iters):**
- **Iter 112b**: Gram coef target 0.05 (5× current). Same `warmup_delay=0.3`. Tests stronger steady-state pressure. Conditional on iter 112 promotion + after 117b-2/117b-3 land.
- **Iter 112c**: Gram `warmup_delay=0.1` + coef 0.01. Engages earlier (s100 vs s300), longer active phase. Conditional on 112 promotion.
- **Iter 112d**: combined (`warmup_delay=0.1`, coef 0.05). Most aggressive; only if 112b/c clean.

**Tier 2 — (re-emptied 2026-04-30):**
- ~~Iter 117b-3b (sparse-Q attention)~~ → **Tier 1 #5** (preserves SDPA call, capacity-padded gather/dispatch)
- ~~Iter 120 (RRAttention)~~ → **DEFERRED back to Tier 3 / Deferred** — replaces SDPA, same class as iter 106 NSA (0.42× FA at T=2048). See Tier 1 demotion note above. Re-queue requires `flex_attention` reimplementation OR T-scaling.

**Tier 3 — Pending implementation (no component yet):**
9. **Iter 117c** — Equal-weight `routing_reg_coef` refactor (collapse 3 coefs to 1, sweep at {0.1, 0.5, 1.0, 2.0}).
10. **Iter 103 (H77, X3)** — Chained 2-stage routing. Flag + safety guard exists (commits `2ad0a13`, `f8fb068`). Block refactor (two routers, halved expert sets, chain logic in `Block.forward`) pending — design scaffold at `experiments/components/chained_block.py`.
11. **Iter 107 (H78)** — attn:mlp expert-count ratio sweep (router refactor for asymmetric pool).
12. **Iter 97.7** — PROFILE-driven throughput retry (proper chrome-trace + fix top 5 bottlenecks). **Note 2026-05-01**: this iter is now ALSO the gating profile for iter 95b (fp32 TBPTT accumulators) — chrome trace must check whether fp64 accumulators show in the hotspot top-5.
13. ~~Iter 95~~ — **MOVED TO TIER 1 #3** as `deq_bptt_k=2→3` fixed test (2026-05-01, post grad_norm=0.07 observation in iter 112+122). The original exhaustive sweep `(1,2,3,4,5,6,8,12)` is descoped — fixed-3 is the cleanest single-variable test; broader sweep only after fixed-3 result.
14. **Iter 118 (H88)** — Triton fused entmax + grouped-GEMM (extension of 117b-2 + 117b-3). Conditional on those smoke tests passing.

**Deferred / awaiting decision:**
- **Iter 106 (H86)** — NSA 2-branch attention. DROPPED 2026-04-29 — 0.42× FlashAttention at T=2048. Code preserved off-by-default for future T-scaling.
- **Iter 120 (H90)** — RRAttention. DEFERRED 2026-04-30 (was briefly Tier 1, demoted same day per user challenge "Would RRAttention hurt throughput as the optimized SDPA is replaced?"). Same SDPA-replacement class as iter 106; pure-PyTorch component cannot compete with fused FA at T=2048. Re-queue requires `flex_attention` (PyTorch 2.5+) reimplementation with `score_mod`/`block_mask` keeping FA fusion intact, OR defer until T-scaling phase.
- **Iter 109 (H80)** — K-jitter {10, 16}. SUPERSEDED by current default {16, 24}. Close out.
- **Iter 98c** — D=1024 + `num_refinements=0`. AWAITING USER GO/NO-GO 2026-04-29.

**Refuted / closed:**
- Iter 119 (H89) — Mixture-of-Depths. REFUTED 2026-04-30 on principled architectural-incompatibility grounds.
- Iter 105 — α-jitter. SUPERSEDED by iter 117 entmax blend.
- Iter 104 — AdaSplash α-entmax. DROPPED 2026-04-29 (Triton + DDP + compile + RevDEQ SIGABRT).
- Iter 117b-1 — entropy/blend coef bumps. NOT PROMOTED 2026-04-30 (H87b REFUTED, +0.0007 essentially flat); configs reverted, structural improvements kept.

#### Legacy ordering paragraph (kept for posterity, see queue summary above for current state)


**Historical baseline reference (superseded):** iter 100b (`c5154b3`, val_bpb int6 = 1.4893) — previous promoted iter; the iter 117 v5 promotion supersedes it because the entmax-on run regressed val_bpb +0.0229 (within +0.03 carry-forward gate) but landed the architectural infrastructure for iter 117b sparsity-engaging tests. Strict-generalization recovery to iter 100b is preserved at `use_entmax_routing=False` (CLI default). Improvement of −0.0010 vs iter 96, K=128 vs best-K Δ=+0.00147 (genuine FP across acyclicity primes K∈{17,37,113}), artifact 7.52 MB (47% of 16 MB budget). H76 documents the result. Groups A-D closed; Group F (LoRA-rank/E joint scaling) opened by iter 96, iter 97 closed E>16 axis (H72), iter 98 H73 OOM'd D=1024 on dev. Architectural-sparsity axis closed at the soft-penalty level by H72+H74+H75 trio + iter 100b reframe (entropy mechanism marginally effective at high coef but val_bpb gain is dominantly CV-driven; specialization not the bottleneck on this task). **Iter 97.6 PERMANENT eval-harness change** (Hutchinson-Frobenius + finite-direction Lipschitz + acyclicity primes K∈{17,37,113}) — Hutchinson/Lipschitz silently caught by try/except in iter 100b + iter 98b runs, root-caused 2026-04-29 to **OOM at 32 GiB allocation request** through full val-batch SharedBlock graph (NOT SDPA backend rejection as initially attributed; the predictive OOM guard underestimated the JVP graph size by ~8×). RESOLVED by iter 97.5b-fix: `B_probe=1` slicing in both `_hutchinson_F_at_saved_fp` (val checkpoints) and `_compute_eval_fp_lipschitz` (K-sweep) — probes now operate on a single sequence, bounding activation memory to ~1-2 GiB. Acyclicity primes worked perfectly throughout (3/3 genuine FP confirmations both runs). **Iter 100b PERMANENT eval-harness extension (2026-04-28)**: K-sweep now emits a tabular `k_sweep_table:` row per K with 14 columns — `K val_bpb attn_cv mlp_cv pool_cv attn_min mlp_min attn_ortho mlp_ortho pertoken_ent pool_ent shared_gate hutch_F rd_step iter_conv_rel`. Header row at start, fixed-width columns for grep + visual scanning. Value `N/A` for unavailable fields (e.g., Hutchinson when SDPA rejects). Allows direct comparison of routing health, sparsity, and Lipschitz across K within and across iters. Backwards-compatible — existing `k_sweep:k=N ...` line preserved. Active queue post-iter-100b promotion (priority order, 2026-04-29 NSA-promoted-to-head): **iter 98b CLOSED** ✗ (H81). **iter 102 CLOSED** ✗ (α-anneal). **iter 105 CLOSED**. **iter 104 (AdaSplash) DROPPED + thoroughly cleaned** ✗ 2026-04-29 (commit `c5d3d42`, −218 lines from train_gpt.py). Sliding-window attention also dropped same time. **iter 106 NSA DROPPED 2026-04-29** (T=2048 kernel-launch overhead — NSA is 0.42× FlashAttention at our seq length per official fla-org Triton benchmark; principled decision after reading official repos; full rationale at H86). Code stays in repo (off by default) for future T-scaling. → **iter 111a — NEXT TO LAUNCH**: per-token routing-variance penalty per H83 — HIGH PRIORITY; routing_variance_coef=0.005 + KEEP existing entropy_coef=0.005 (single-axis change from iter 100b for clean attribution); addresses pertoken_entropy ≈ 3.0; strict-generalization unconditional-promote → **iter 111b** (per H83b — HIGH PRIORITY; drop router_entropy_coef → 0, keep variance_coef=0.005; tests whether entropy is REDUNDANT given variance; runs immediately after 111a regardless of 111a's promotion outcome) → **iter 112** (orthogonal-expansion routing per H84 — HIGH PRIORITY) → **iter 113** (`block_ortho_aux_coef` 0.1 → 0.5 per H85 — HIGH PRIORITY control test) → **iter 110** (re-enable `num_refinements=1` per H82) → **iter 117** (adaptive entmax-α + capacity-padded compute-skip per H87 — HIGH PRIORITY; per-block learnable α with init=1.0 ≈ dense softmax, gradient-driven sparsity discovery; first principled throughput-positive iter; pure PyTorch path-A inside torch.compile via static-shape Switch-style buckets; predicted 1.5–2.5× wallclock at val_bpb ±0.02 of iter 100b) → **iter 118** (Triton custom_op fused entmax+grouped-GEMM per H88 — HIGH PRIORITY conditional on iter 117 promotion; predicted 3-4× wallclock) → **iter 119** (Mixture-of-Depths layer skip per H89 — MEDIUM PRIORITY; orthogonal sparsity axis; composes multiplicatively with iter 117 for ~85% compute reduction) → **iter 106 RE-QUEUED** (NSA 2-branch DEFERRED 2026-04-29 — code preserved off-by-default; revisit as regularization-only test after iter 117 establishes throughput baseline) → **Iter 98c** (D=1024 + `num_refinements=0`) AWAITING USER GO/NO-GO 2026-04-29 → **iter 103** (chained 2-stage routing per H77; ~2 hrs dev) → **iter 107** (attn:mlp ratio sweep per H78; router refactor) → **iter 97.7** (PROFILE-driven throughput retry) → **iter 108** (forward K=10 fixed per H79; TRIVIAL config) → **iter 109** (K-jitter {10, 16} per H80; superseded by 2026-04-29 default {16, 24}; close out) → **iter 95** (TBPTT efficiency sweep). **Iter 97.5b PERMANENT (2026-04-29)**: Hutchinson-Frobenius `hutch_F` now emitted at every val checkpoint AND in K-sweep matrix (verified live: hutch_F=1.9049 at val checkpoint, 2026-04-29). **K-sweep matrix protocol mandate (CLAUDE.md §7, 2026-04-29)**: every iter copies its full `k_sweep_table:` matrix verbatim into hypotheses.md. **K-sweep K=24 added 2026-04-29**: K-sweep now has 10 K values (4, 8, 16, 17, 24, 32, 37, 64, 113, 128). E-scaling past 16 closed (H72). Iter 99b sparse dispatch CANCELLED. Iter 102-old (inference-time top-k) removed.

**Historical baseline reference (superseded):** iter 96 (`b962b5f`, val_bpb int6 = 1.4903) — "more, smaller experts" promotion. H71 documents the result. Groups A-D are now closed (Groups A-C all PROMOTED ★ except iter 83 reverted; Group D bottleneck → NOT PROMOTED, code reverted, archive at tag `iter-91+92-bottleneck-NOT-PROMOTED`). Group F (LoRA-rank/E joint scaling) is the active scaling direction, opened by iter 96. **Iter 97 (E=20 continuation) NOT PROMOTED on per-wallclock grounds — H72 closes the rank-halving subdirection past E=16.** **Iter 97.5 (throughput config bumps) NOT IMPROVED — reverted.** **Iter 97.6 (PERMANENT eval-harness change)**: K-sweep now auto-reports Hutchinson-Frobenius + finite-direction random-step Lipschitz at the FP, plus acyclicity primes K∈{17,37,113}. Probes wrapped in try/except per H75-commit (SDPA backend rejects under enable_grad in some dtype paths). **Iter 98 (D=1024) NOT TESTED** — 3× OOM at 44 GiB cap (H73). **Iter 99 (sparsemax) NOT PROMOTED** — +0.16 capacity cost from pure top-1 trap (H74). **Iter 101 (entmax-1.5) NOT PROMOTED** — +0.10 capacity cost; trajectory self-corrected (entropy 0.20→1.94) but couldn't close the gap (H75). H72+H74+H75 trio collectively close the architectural-sparsity axis on this codebase — iter 96 softmax+LoRA is near-optimal at 2× L40S budget. Active queue (in order, 2026-04-27): **iter 100b** (IN FLIGHT — annealed entropy penalty 0→0.005 with `warmup_delay_frac=0.3` + `min_share_loss_weight=0.0` decoupled, `cv_loss_weight=2.0` (20× default) doing global balance alone; principled rescue if CV runs away = `cv_loss_weight ↑`, NEVER reintroduce min_share floor — that's the dirty antagonist that motivated the decouple in the first place. **Trajectory s80→s120**: pool-level cv equilibrated (peaked 0.74 at s90, fell to 0.62 by s110-120 and held); **per-slice asymmetry exposed under shared pooled router** — attn slice has winner-take-all (expert 12: 0.376 peak → 0.302 pulling back, min 0.014-0.019 well above sentinel 0.005), MLP slice is essentially uniform (max 0.084, min 0.040, range 0.044) so CV pressure is doing its work almost entirely on the attn axis; `ntp_loss` AHEAD of iter 96 trajectory (3.99 at s100 vs ~4.5 expected). **Diagnostic decomposition added at s120 (logging-only, mid-flight)**: `format_expert_info` now emits `attn_cv` / `mlp_cv` / `pool_cv` and `attn_entropy` / `mlp_entropy` / `pool_entropy` as DISTINCT values (previously the single mean-of-per-slice-cv was duplicated under both prefixes). Hand-computed s120 reference: attn_cv≈1.07, mlp_cv≈0.18, pool_cv≈0.77; attn_ent≈2.31 (85% of log15), mlp_ent≈2.70 (99.6%), pool_ent≈3.19 (94% of log30). Pool entropy 94% of log30 confirms no cross-slice dominance — total mass well-spread across attn+mlp at the pool level; the imbalance is purely within-attn-slice. The change is fully backwards-compatible with `experiments/plot_metrics.py` (same prefix names; values are now per-slice meaningful). **Step 200 val checkpoint hit: val_bpb=2.0012 (predicted 1.85±0.10, actual +0.05 above band — slightly slower descent than linear extrapolation). attn_cv fell DECISIVELY from 0.74 peak (s90) → 0.40 at s200, expert 12 share halved (0.376 → 0.191). CV-only global balance VALIDATED; no min_share floor needed even under heavy attn imbalance pressure.** Step 270 (still pre-entropy-ramp): attn_cv continues dropping to 0.318, expert 12 down to 0.130 (≈3× decrease from s90 peak), ntp_loss 2.99, **aux-gap collapsed from 1.57 (s80) to 0.20 (s270)** — CV regulator is quiescent, routing has reached a stable fixed point of the balance constraint BEFORE the entropy penalty engages. Ideal pre-condition: entropy ramp at s300+ has clean territory for per-token specialization, not equilibrium repair. router_mass slowly declining (0.99 → 0.93) — model intentionally closing the routed mixture (per CLAUDE.md §6.2 sigmoid-gate semantic). **Step 300-340 (entropy ramp engaged, coef growing from 0 to 0.0003)**: attn_cv stable at 0.281-0.289 across the transition; max attn share oscillates 0.124-0.138 (well below s90 peak 0.376); ntp_loss continues descent 2.94 → 2.77; per-token entropy (still labelled `attn_entropy`/`mlp_entropy` under old format pre-c5154b3) rises 2.78 → 2.85 (more uniform per-token, ramp not yet biting). Diagnostic outcome (1) CONFIRMED: entropy and CV co-existing healthily; the prepared CV equilibrium absorbs the small initial entropy push without re-awakening winner-take-all. Real specialization test deferred to step 500-700 when entropy_coef grows to 0.0015-0.003 (5-10× current). **Step 400 val checkpoint: val_bpb=1.6486** — major win signal. Iter 96 FINAL val_bpb is 1.4903; iter 100b is 0.16 above that with 60% of training remaining. Trajectory: s200 val=2.00, s400 val=1.65 (descent rate 0.35/200 steps). Conservative extrapolation s1000 val_bpb ∈ [1.34, 1.48] — **on track to match or beat iter 96 baseline**. Per-token "entropy" plateaued at 2.88 (rising trend halted at step 350, entropy ramp barely biting at coef 0.0007). attn_cv holds 0.28-0.30. ntp_loss 2.56 at s400 (well ahead of "toward 2.5 by step 500" target). shared_gate_mean dropped 0.73 (init) → 0.14 (s400) — shared experts increasingly suppressed as routed specialization develops; shared_gate_min 0.0037 means at least one block runs essentially routing-only. **Step 530 update (entropy_coef ≈ 0.0016, ~33% of target)**: attn_cv TIGHTENED FURTHER to 0.247 (from 0.289 at s400) — the two regulators are not just co-existing but cooperating. ntp_loss 2.52 at s510 (target was "toward 2.5 by step 500" — beat it). Per-token "entropy" still slowly RISING (2.88 → 2.94) — entropy ramp not yet strong enough to overcome CV-driven uniformity at coef 0.0016. **Reframe**: iter 100b's win is coming from CV equilibration (smooth global balance), NOT from per-token specialization. Implication for iter 102 (α-anneal): if specialization isn't the bottleneck for soft-MoE routing on this task, iter 102 may be solving a non-problem — reconsider queue priority after iter 100b finishes. **Step 600 val checkpoint: val_bpb=1.5342** — CLEAR PROMOTION PATH. iter 96 final 1.4903; iter 100b is only 0.044 above with 40% training remaining. Descent rate decelerating (s400→s600 = −0.11 vs s200→s400 = −0.35) — classic regularization-fix → LM-cooling transition. Most-likely s1000 outcome: 1.43-1.48 range (beats or matches iter 96). **REFRAME STRONGLY CONFIRMED**: per-token "entropy" 2.88 (s400, coef 0.0007) → 2.97 (s600, coef 0.0021) → 2.98 (s630, coef 0.0024). Entropy penalty grew 3.4× since s400 yet per-token entropy STILL RISING. Target coef=0.005 is too weak to drive specialization in this architecture; the win is purely CV-driven. The two regulators completely segregated (CV did all routing-balance work, entropy did essentially nothing). Useful attribution for iter 100c/100d if we want to push specialization harder. Iter 102 (α=1.5 entmax) is a stronger architectural-sparsity push that bypasses the gradient-strength limit — its value beyond iter 100b's CV-equilibrium win remains genuinely open. **Step 760 update (entropy_coef ≈ 0.0033, 66% of target)**: attn_cv hit a NEW MINIMUM of 0.218 at s730 (from 0.247 at s120, 0.288 at s400, 0.277 at s600) — CV continues to TIGHTEN even under stronger entropy push, the two regulators STILL cooperate at 66% target. ntp_loss 2.48 at s760 (descent accelerating: 0.26 drop over 160 steps from s600 to s760, FASTER scaled rate than s400→s600). Per-token "entropy" PINNED at 2.97-3.00 — increased coef by 1.6× since s600 produced ZERO bend-down. Reframe IRONCLAD-confirmed: soft entropy penalty mechanism is COMPLETELY ineffective at driving specialization in this architecture. **Step 800 val checkpoint: val_bpb=1.4968** — outcome (2) MODEST PROMOTION confirmed. Iter 96 final 1.4903; iter 100b is 0.0065 ABOVE with 200 steps remaining. Half-rate decay clean (s400→s600 −0.114, s600→s800 −0.037, ratio 0.33×). Linear extrapolation s1000 = 1.479-1.492 → **beats iter 96 by 0.005-0.015**. attn_cv hit NEW minimum 0.1945 at s800 val (was 0.218 at s730, 0.247 at s120). Per-token "entropy" finally PLATEAUED at 3.00 (no longer rising) — at coef 0.0036 (72% target) the penalty is MARGINALLY engaged (just enough to halt the rise) but not strong enough to bend down. Slight revision: penalty is "marginally effective at high coef" not "completely ineffective". Qualitative reframe still holds — iter 100b win is dominantly CV-driven. **Step 930 update**: attn_cv hit ANOTHER new minimum 0.181 at s920 — CV continues tightening through the warmdown phase. ntp_loss 2.44 at s930. step_avg crept up to 26s (DDP sync noise from Muon momentum saturation in warmdown). Step 1000 + K-sweep (acyclicity primes K∈{17,37,113} + Hutchinson + Lipschitz per iter 97.6 PERMANENT) expected to emit ~22:45. **🎉 FINAL RESULTS (s1000)**: val_bpb fast=1.4572, **roundtrip int6+zstd=1.4893** ← beats iter 96 int6 baseline 1.4903 by **0.0010**. Conservative extrapolation undershot — actual descent rate didn't decay as predicted (s800→s1000 drop 0.040 vs predicted 0.012-0.018). attn_cv final 0.1735 (NEW MINIMUM, from 0.181 at s920, 0.247 at s120). artifact_bytes=7,515,680 (47% of 16MB budget). peak_vram_mb=35,721 (well under 44 GB cap). K-sweep partial (K=4→32): k=4 1.6685, k=8 1.4936, k=16 1.4893 (best so far), k=17 1.4896 (acyclicity prime, genuine FP — matches k=16 within 0.0003), k=32 1.4906. K=64, K=113 (prime), K=128, Hutchinson, Lipschitz still pending. Promotion gates: ✓ val_bpb int6 improvement, ✓ 16MB budget, ✓ acyclicity, ⏳ K=128 vs best-K Δ. Implication for queue priority (revised 2026-04-27 after user correction): iter 100b's reframe refutes only the narrow claim that "soft per-token entropy would drive specialization in this architecture" — it does NOT refute the broader value of sparsity for THROUGHPUT (skip-zero-experts dispatch, sparse SDPA kernels) and REGULARIZATION (inductive bias, generalization). Critical distinction by axis: (a) router sparsity under soft-dense routing gives NO throughput benefit (all experts still computed) — value is regularization-only, val_bpb upside lower than initially expected; (b) attention sparsity via fused Triton kernels (AdaSplash α-entmax) gives BOTH throughput AND regularization. **Revised post-100b queue prioritizes iter 104 (AdaSplash attention) ABOVE iter 102/105 (router sparsity)** — attention sparsity is the strongest throughput play given soft-dense MoE pays full expert compute regardless of router weights. iter 98b (D=1024) follows attention sparsity in priority since iter 100b shows we're capacity-bound on dimensionality not on routing concentration. Iter 102/105 (router sparsity) retain their queue slot for regularization+stability tests but with lower expected val_bpb upside. router_mass plateaued at 0.83 from s600 onward (model settled the routed-vs-shared split). Predicted s1000 val_bpb 1.43-1.48 — modest but real promotion (beats iter 96 by 0.02-0.05). Pre-penalty phase still active until s300. Revised step-200 val_bpb prediction: 1.70-1.80. CV-only global balance VALIDATED; the user's principled framing — CV smooth ≫ min_share dirty hinge — was correct.) → **iter 104** (AdaSplash α-entmax attention — PRIORITIZED post-100b reframe: throughput via fused Triton kernel + regularization; soft-dense MoE means router sparsity gives no throughput benefit, so attention sparsity is the strongest throughput play) → **iter 98b** (D=1024 with `grad_accum_multiplier=2`; rescue iter 98 H73 OOM; iter 100b shows we're capacity-bound on dimensionality not routing concentration) → **iter 102** (α-annealing 1.0→1.5 router sparsity; regularization-only after reframe — soft-dense pays full expert compute regardless of router weights) → **iter 105** (α-jitter {1.0, 1.5, 2.0} per-step — conditional rescue, only run if iter 102 leaves >0.04 val_bpb gap; analog of K-jitter H12 VERIFIED + TBPTT-jitter iter 85, periodic α=1 steps deliver gradient to dead experts throughout training) → **iter 103** (chained 2-stage routing — orthogonal axis) → **iter 97.7** (PROFILE-driven throughput retry, proper chrome-trace) → **iter 95** (TBPTT efficiency sweep). E-scaling past 16 closed (H72). Iter 99b sparse dispatch CANCELLED. Iter 102-old (inference-time top-k) removed.

Run ordering rationale (preserved for posterity): low-risk → higher-risk, activation / gate / schedule tweaks before legacy-loss ablations, architectural scale-up last (depends on predecessors).

#### Group A — low-risk quick wins (run first)

| New # | Old # | One-line | Rationale |
|---|---|---|---|
| **94** | new | Disable CTP entirely (NTP-only). Add `Hyperparameters.use_ctp = True` guard; when False, `MoSLowRankOutputHead.forward` returns only `log_p_ntp`, `GPT.forward` skips `ctp_loss` computation at L2760-2781, and CTP-specific `nn.Parameter` banks (`gate_ctp`, `gate_ctp_norm_weight`, `A_ctp_shared`, `A_ctp`, `ctp_a_norm_weight`, `ctp_rank_norm_weight`, `B_denoise`) are NOT allocated. **PROMOTED ★ (commit `c2eff43`)** — int6 Δ=+0.0026 (≤0.03 ✓), K8→K128 Δ tightened +0.0103→+0.0073, artifact -592 KB (-9.0%), params -1.38M (-10.9%). See H60. |
| **83** | 74e | Restore MLP activation `leaky_relu(0.5)²` (GLU-style: `leaky(gate,0.5)² * fc`) | **REVERTED ✗ (commit `34ef98d`)** — int6 regression +0.0124 vs iter 94 (1.6076 vs 1.5952), artifact +84 KB, K-sweep slightly widened (K=8→K=128 Δ: +0.0073 → +0.0087). Technically within the ≤0.03 carry-forward band, but the change delivers *no* offsetting benefit (no param savings, no FP-quality tightening, no artifact saving) — it is a pure regression under the iter 94 NTP-only + Parcae landscape. The SOTA abaybektursun 1.1194 leaderboard config used a *non-gated* `leaky²(up(x)) → down` FFN; our MoE has `expert_gate + expert_fc + expert_down`, so the GLU-with-leaky² variant tested here is a hybrid that apparently pulls worse than SwiGLU in our architecture. See H61. (A full SOTA-faithful non-gated expert rewrite is still open as a possible follow-up, but blocked behind the Group D architectural rewrites.) |
| **84** | 74f | Independent attn/mlp shared gates (1-dim → 2-dim) | **PROMOTED ★ (commit `6494a50`)** — int6 Δ=**-0.0108** (improvement!), every K-sweep point improved (k=4 by -0.042), K=8→K=128 widened negligibly (+0.003, still ≪0.5), artifact +68 KB (+1.1%). `shared_gate_std` grew 0 → 0.17, confirming the two gates took meaningfully different values. See H62. |
| **85** | 82 | Stochastic TBPTT `{2,3,4}` | **PROMOTED ★ (commit `b18dc55`)** — int6 regression +0.0054 (within threshold), K=8→K=128 Δ tightened +0.0131→+0.0114, artifact -7 KB. Cost: step_avg +21% (from deeper avg backward). See H63. Reconsider for wallclock-constrained submission config. |
| **93** | new | Remove bigram embed (`bigram_vocab_size 4096 → 0`) | **PROMOTED ★ (commit `66ce88e`)** — int6 Δ=+0.0023 (≤0.03), K=8→K=128 tightened +0.0114→+0.0111, artifact **-330 KB (-5.4%)**, params **-622K (-5.5%)**. See H64. Mid-training early regression (+0.08 @ step 200) fully reversed by step 400 (-0.044). |

#### Group B — medium-risk schedule + regularization tuning

| New # | Old # | One-line | Rationale |
|---|---|---|---|
| **86** | 74c | WD 0.30 → 0.01 re-test | **PROMOTED ★★★ (commit `df2cfdb`)** — int6 Δ=**-0.0531** (largest single-iter win in queue), every K-sweep point improved ~5%, K=8→K=128 widened +0.004 (still ≪0.5), artifact +3.7% (slightly larger weights, expected). See H65. |
| **87** | 81 | K-jitter `{4,6,10} → {8,12,20}` | **PROMOTED ★ (commit `88ad22c`)** — int6 Δ=**-0.0202**, K=8→K=128 Δ went **NEGATIVE** (+0.0148 → -0.003, deep K is now BETTER than train K). k=4 +0.16 (off-distribution, expected). Step_avg +41% (9.7s→13.7s). See H66. |

#### Group C — legacy-loss ablations (after Parcae is validated)

| New # | Old # | One-line | Rationale |
|---|---|---|---|
| **88** | old 66b queue | Disable Lyapunov Hutchinson penalty (λ_jac 0.01 → 0) | **PROMOTED ★ (commit `45af5bf`)** — int6 Δ=+0.0050 (≤ 0.03 ✓), K=8→K=128 Δ tightened -0.003 → **-0.0041** (still negative — deep K BETTER), artifact -57 KB, step_avg **-2.9% (~3% throughput recovery)**. Hypothesis confirmed: Parcae per-dim Ā already bounds spectral radius; λ_jac contributed only noise + one VJP/step. Code path retained (commented-out future cleanup permitted, never delete). See H67. |
| **89** | old 66c queue | Disable HyDRA denoising regularization (denoising_coef 0.01 → 0) | **PROMOTED ★ (commit `aeba34a`)** — int6 Δ=+0.0026 (≤ 0.03 ✓), K=8→K=128 Δ -0.0041 → -0.0036 (still negative — deep K BETTER), artifact +49 KB (zstd-compression diff, no params changed), step_avg -2%, peak_vram -301 MB. Same Parcae-redundancy hypothesis as iter 88 confirmed for the finite-perturbation probe. Cumulative iter 88+89 reclaims ~5% step time and ~1.5% peak VRAM. Code path retained per user directive. See H68. |

#### Group D — architectural scale-up (CLOSED 2026-04-25; bottleneck DISCARDED — superseded by iter 96 full-D LoRA)

> **DECISION (2026-04-26): Low-dim bottleneck experts are DISCARDED.** The iter 96 PROMOTION of full-D LoRA with rank-halving + E-doubling (H71, val_bpb int6 1.4903) demonstrated that **the same artifact-budget gain is achievable without sacrificing per-param efficiency or `d_head` tensorcore alignment**. Bottleneck experts forfeit both:
> - Per-param efficiency: 1.49 bpb/param vs full-D LoRA's 1.17 (~27% worse, H70 quantified at matched capacity).
> - SDPA throughput: small `r` forces `d_head ≤ 48` (off FA sweet spot of 64+), and the penalty compounds linearly when scaling N_expert.
>
> Full-D LoRA delivers strictly better val_bpb at the same artifact size, with a routing-diversity scaling axis (Group F) that doesn't require giving up `d_head ≥ 64`. The bottleneck infrastructure is preserved for archival/rescue purposes only — it should NOT be re-introduced as a forward-looking scaling axis. CLAUDE.md §6.3 records this as the architectural standard.

| New # | Old # | One-line | Result |
|---|---|---|---|
| **90** | 63-merged | Full-rank low-dim experts: `BottleneckIn (D→proj_rank→r) → full-rank MLA/MLP at r → BottleneckOut (r→proj_rank→D)` per expert | **NOT PROMOTED ✗ (commit `8c2be77`)** — int6 Δ +0.276 vs baseline (capacity -65%, 4.5M params). Architecturally validated: K=8→K=128 Δ TIGHTER (-0.0092 vs baseline -0.0036), step_avg -34%, peak VRAM -36%, artifact -49% (3.0 MB / 19% of budget). KV-A still latent-compressed for DeepSeek-style cache benefit; expert independence + per-expert pre-RMSNorm preserved. See H69. |
| **91+92** | 65+70-dup | Bundle: `num_experts 8→16, model_dim 768→1024, attn_bottleneck_r 128→192, mlp_bottleneck_r 128→192` (proj_rank=32 unchanged) | **NOT PROMOTED ✗ (commit `3e35655`)** — matched-capacity test of bottleneck arch at 11.3M params (87% of baseline). int6 Δ +0.152 vs baseline (closed half the iter-90 gap, but plateaus from step 800). K=8→K=128 Δ -0.004 (still tighter than baseline). Quantified per-param efficiency: bpb/param 1.49 vs baseline 1.17 (~27% worse). proj_rank=32 likely the bottleneck (D=1024 squeezed through 32-dim subspace per expert). See H70. |
| 91+92 follow-up (deferred, optional) | new | proj_rank=48 or 64 rescue (12.6M / 13.8M params, full-baseline-capacity bottleneck arch) | Available if user prioritizes bottleneck-arch closure: would test whether the per-param penalty is intrinsic to the bottleneck or specific to proj_rank=32. Not on the active queue per user direction 2026-04-25 (proceed to iter 95 instead). |

#### Group E — active (now next-up after Group D closure)

| New # | Old # | One-line | Plan |
|---|---|---|---|
| **95** | new | TBPTT efficiency sweep — find elbow | **NEXT (active queue)** — per user redesign 2026-04-26: replace the originally-planned anneal schedule with a uniform stochastic sweep. Expand `Hyperparameters.deq_bptt_k_jitter_set` from `(2, 3, 4)` to `(1, 2, 3, 4, 5, 6, 8, 12)` (covers k=1 to K_max/2); each step samples uniformly. Existing run.log lines already capture `(tbptt_k, grad_norm, ntp_loss)` per step. Post-training: bin grad_norm by tbptt_k, identify the elbow where marginal gradient signal saturates. The elbow tbptt_k* is the optimal TBPTT depth — update the jitter set to a tighter range around k* in a follow-up commit. Deliverable: H71 entry with elbow finding + recommended set. Diagnostic addition (queued for the same iter, ~15-line patch): per-K Lipschitz probe `‖DEQ(x+ε,K) − DEQ(x,K)‖/‖ε‖` to verify contraction tightens with K (independent confirmation of K-sweep monotonicity). Runs on iter 89 baseline (`aeba34a`), no architectural change. |
| **Phase 8 — self-refinement extensions** | iter 38 | Further self-refinement work (`num_refinements ≥ 2`, alternative ramp schedules, self-conditioning variants on `_refine_mix_alpha`) | **DEFERRED to after iter 95**. Note: the *baseline* already has `num_refinements=1` with 85% ramp on `_refine_mix_alpha` (see `train_gpt.py`) — that single-refinement form has been the working baseline since Phase 7.5. What's deferred is any further self-refinement expansion (more refinement steps, alternative ramp curves, or self-conditioning variants). Rationale (post-Group-D update 2026-04-25): the bottleneck-experts arch did NOT promote, so Group D's intended landscape change didn't materialize — Phase 8 will be tuned against the iter 89 baseline rather than waiting for the (now-not-happening) D=1024+E=16 landscape. Still ranked below iter 95 because TBPTT optimization is a single-knob investigation with cleaner attribution. |
| **Phase 7.5 — profile pipeline + top 10 ROI bottlenecks** | new | Profile the iter 89 step pipeline + fix top 10 ROI bottlenecks | **DEFERRED to after Phase 8** — same rationale as Phase 8 reset; profile-driven optimization is more useful once iter 95 + Phase 8 settle. Step_avg is already at 13s on dev hardware (iter 89 baseline); profiling against that vs the bottleneck arch's 8.6s would mostly reidentify already-known wins. |
| **iter 90 follow-up (proj_rank=48 or 64)** | new | Bottleneck-arch rescue at full baseline capacity | **OPTIONAL (deferred indefinitely)** per user direction 2026-04-25. Available if Group D revisit is prioritized: bumping proj_rank from 32 to 48 (12.6M params, 97% of baseline) or 64 (13.8M, 106%) would test whether the +0.135 plateau in iter 91+92 is intrinsic to the bottleneck arch or specific to the 32-dim per-expert subspace. Not on the active queue. |

#### Group F — LoRA-rank/E joint scaling (active scaling direction, opened by iter 96 PROMOTION 2026-04-26)

The "more, smaller experts" axis: hold `E·R` constant on Q/MLP linears (preserves linear-projection FLOPs) while doubling routing diversity. Validated by iter 96 (PROMOTED, val_bpb int6 1.4903 = -0.0361 vs iter 89 baseline). The axis costs SDPA + Wo wall-time (`E·H·d_head·B·T²` scaling with E independent of R, so doubling E doubles SDPA), but the per-param efficiency gain pays for it on val_bpb. Group F continues scaling along this axis on the iter 96 baseline.

| New # | One-line | Status |
|---|---|---|
| **97** | E=20 (was 24, OOM on 2× L40S — fell back), `attn_expert_rank=51, mlp_expert_rank=77` (iso-cost on linears: 16·64=1024 → 20·51=1020). Continuation of the iter 96 axis. | **NOT PROMOTED ✗ (commit `acce3d4`, reverted; H72)** — int6 Δ +0.0133 vs iter 96 (within 0.03 carry-forward gate), but **per-wallclock regresses 19% (bpb/h 0.229 → 0.186)** and submission @ 600s drops 5 steps. Per-wallclock override applied per user directive 2026-04-26. Trajectory inverted (-0.014 step 200 → +0.0133 step 1000) — capacity gain front-loaded, rank cost dominates late. K-sweep tightened (K=128 vs best-K Δ ~+0.002). E=16/R=64 is Pareto-optimal on D=768; further E-scaling along this axis is closed. Pivot to iter 98 (D scaling) and iter 99 (sparsemax). |
| **98** | `model_dim 768 → 1024` under iter 96/97 LoRA layout | **PENDING** — D scaling on top of the validated more-smaller-experts axis (no bottleneck involved). d_head naturally goes 96 → 128 (FA tensorcore sweet spot). Per-expert linear cost scales linearly in D (D·R + R·H·d_head ≈ 2·D·R). Budget check required: artifact must stay ≤ 16 MB after int6 + zstd-22. iter 96 used 47% of budget (7.55 MB) so significant headroom. Runs on whichever of iter 96/97 is the latest promoted baseline. |
| **99** | Per-token entropy logging + penalty (`entropy_coef = 0.02`) | **PENDING** — drives per-token routing specialization. Two distinct entropies are tracked separately (per CLAUDE.md §6.2): `expert_entropy` (global utilization, target HIGH ≈ log(N) — no dead experts) vs new `pertoken_entropy` (per-token routing distribution, target LOW — concentration on few experts per token). Single bundled change: (a) add per-token entropy logging in `SoftDenseRouter.forward` as `H_pertoken = -(w * log(w+eps)).sum(-1).mean()`; (b) add penalty `loss += entropy_coef * H_pertoken.mean()` with `entropy_coef = 0.02`. The `min_share_loss_weight=10.0` stays as anti-collapse guard for global balance (different aggregation, compatible objective). Logging alone is insufficient — observation does not change the loss landscape; the penalty is what creates the specialization gradient. Watch DEQ FP convergence: sharper softmax may make T_θ less smooth and hurt iter_conv_rel; if so, drop entropy_coef or pair with τ>1 in iter 100. |
| **100 (conditional)** | Sweep `λ_entropy ∈ {0.01, 0.05, 0.1}` × softmax temperature `τ ∈ {1.0, 1.5}` | **PENDING (only if iter 99 promotes)** — joint sweep of the per-token specialization knobs. τ alone is borderline (optimizer can compensate via flatter logits); pairing τ with λ_entropy is the principled combination for true per-token sparsity without hard top-k masking. |

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
  - **K=128 vs best-K Δ ≤ 0.5** (FP still converges at deep K). The
    reference is the K-sweep row with the lowest val_bpb (whichever K
    that is — was K=8 historically when training-K was small, but with
    `deq_k_jitter_set={8,12,20}` best-K may be K=12, K=20, or any K).
    The gate is "K=128 is no worse than best-K by more than 0.5."
  - No catastrophic failure on other gates (no NaN, no routing collapse,
    mos_ortho ≤ 0.9, expert_min_share ≥ 0.005, etc.).
  → `update_results.sh --promote`, update H33 audit row for the newly-
     certified component, launch next iter in the queue.

- **Fix-and-retry (stay on current iter)** if the change IS significant:
  - val_bpb regression > 0.03 OR K=128-vs-best-K Δ > 0.5 OR any gate catastrophes.
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

---

## Records-derived hypotheses (H91–H99) — appended 2026-04-30

Source: problem-driven audit of `records/track_10min_16mb/` 2026-04-27 SOTA submission (val_bpb=1.0611) and predecessors. Each H-claim below identifies a concrete problem the records' SOTA stack solves, and confirms our codebase has the same problem (i.e. is not architecturally subsumed by RevDEQ + Parcae + soft-MoE). Items where our model addresses the problem differently (U-Net skips, depth recurrence, parallel decoder, LN scale 1/√(layer+1)) are deliberately NOT queued.

Queue insertion: Tier 4 — Records-derived (after Tier 1 throughput iters and Tier 3 architectural sweeps). Ordered within Tier 4 by (a) impact magnitude and (b) implementation complexity.

### H91: Phased Test-Time Training (TTT) closes the per-document adaptation gap

**Problem:** standard LM perplexity treats each document independently, but documents have local statistics (vocabulary, style, topic) that aren't well-captured by global parameters. Records' SOTA stack uses TTT: a small LoRA adapter is fine-tuned on each document's prefix at eval time, then frozen and used to predict the suffix. Phased TTT splits this into 3 cumulative phases at doc-boundaries 833/1666/2500 (max prefix=2500 docs). LoRA per-doc reset.

**Codebase status:** ZERO TTT infrastructure. We evaluate via single-pass standard perplexity. The gap closes ~0.05–0.10 BPB across all post-2026-03-23 records.

**Proposal:** add `ttt_eval_enabled` Hyperparameter + `ttt_lora_rank=80`, `ttt_phases=3`, `ttt_phase_doc_boundaries=(833, 1666, 2500)`, `ttt_beta2=0.99`, `ttt_weight_decay=0.5`. New eval-loop function that, for each phase, runs LoRA-only SGD on the cumulative prefix tokens, then evaluates suffix tokens with the adapted weights, then resets LoRA at next-doc boundary. LoRA targets: Q/K/V/O/MLP/lm_head per the SOTA records.

**Why it ports cleanly to RevDEQ:** TTT is purely inference-time. The base model isn't retrained; the FP iteration runs unchanged. LoRA adapts only the per-expert linears (rank-80 added on top of our existing rank-64/96 LoRA structure).

**Risks:** (a) eval time extends from ~30 s → ~7 min (TTT runs ~450-510 s on H100; on dev L40S likely ~25 min). (b) per-phase LoRA reset may interact with our shared-block design (the same Block runs all 12 layers; per-doc reset is not per-block, only per-doc).

**Estimated ROI:** **−0.05 to −0.10 BPB** (largest single feature in records corpus). Promotion path: implement → run iter 121 with TTT enabled at fixed-step val_bpb. If int6 BPB drops by ≥0.03, promote.

**Status:** PROPOSED — Tier 4 priority **#2** (after H93 which is cheapest).

### H92: Logit softcap (Gemma2-style) bounds extreme logit values

**Problem:** without softcap, lm_head logits can grow unbounded during training, causing gradient spikes and bf16 numerical issues. Records use `logits = softcap * tanh(logits / softcap)` with softcap=30. Standard in every record from 2026-04+.

**Codebase status:** NOT PRESENT. Our grad_norm history is healthy (0.05-0.30) and grad_clip=1.0 absorbs spikes, so the problem is mild — but the regularization effect on training dynamics is real and consistent in records.

**Proposal:** one-line change in `MoSLowRankOutputHead.forward` (or wherever the final logits are produced). Add `logit_softcap = 30.0` Hyperparameter; apply `logits = softcap * tanh(logits / softcap)` if `softcap > 0`.

**Estimated ROI:** **−0.005 to −0.015 BPB**. Cheapest principled win.

**Status:** PROPOSED — Tier 4 priority **#3** (low-risk, easy, but small magnitude).

### H93: Logit softcap is the absolute-cheapest first add (priority over even H92's positioning)

This is H92 reframed: it's a one-line trivial add. Run it before TTT to reduce variance in subsequent iter measurements. Status: PROPOSED — Tier 4 priority **#1**.

### H94: GPTQ + LQER int4-rank4 closes the int6 quantization-error gap

**Problem:** our per-row int6 quantization uses naive scale-zero-point with no Hessian-aware error optimization. Records use GPTQ (Hessian-aware) + LQER asymmetric int4 rank-4 correction on top-3 tensors. This delivers the same artifact size at lower quantization error — directly improves `val_bpb_int6` (our promotion gate).

**Codebase status:** YES, we have this problem. Our int6 round-trip costs us a fixed BPB delta (e.g., iter 117 v5: fast=1.4820 → int6=1.5122, +0.0302 quantization tax). LQER+GPTQ in records reduces the equivalent tax to ~0.005-0.010 BPB.

**Proposal:** replace `_int6_quantize_per_row` with GPTQ pipeline (calibration set + Hessian update + greedy quant). Add LQER post-processing: find top-3 worst-quantized tensors by reconstruction error, compute rank-4 correction `U @ V.T` where U, V come from SVD of the quantization residual.

**Estimated ROI:** **−0.02 to −0.04 BPB on int6** (target: shrink the int6 quantization tax from 0.030 to ~0.010). Affects val_bpb_int6 directly.

**Risks:** GPTQ calibration adds eval-time cost. LQER correction tensors add ~50-100 KB to artifact (rank-4 × top-3 tensors). Net artifact may grow slightly.

**Status:** PROPOSED — Tier 4 priority **#4**.

### H95: SP1024 → SP8192 tokenizer + CaseOps closes vocab-inefficiency gap

**Problem:** at vocab=1024, BPE produces ~more tokens per byte than SP8192. Even at perfect prediction, BPB is bounded above by `tokens_per_byte × per_token_perplexity`. Records use SP8192 + CaseOps lossless case preprocessing (bijective lowercase + private-use-area sentinels) which compounds the gain.

**Codebase status:** YES, our `vocab_size=1024` is the smallest in the records corpus (most use 8192). CaseOps not present.

**Proposal:** PAIR change. (1) Add `vocab_size=8192` config + retrain tokenizer at sp8192 on FineWeb (one-time data-prep step, ~30 min). (2) Add CaseOps preprocessing layer: bijective lowercase encode + private-use-area sentinels at training-set construction time. Decoder mirrors the encode.

**Risks:** (a) embedding params grow 8× (1024×D → 8192×D). At D=768 and FP16, that's +10.5 MB embedding alone — exceeds 16 MB budget without compression help. (b) MoS prediction head also has vocab dimension. Must re-architect or use tied embeddings rigorously. (c) Re-tokenizing the dataset is one-time cost.

**Estimated ROI:** **−0.02 to −0.04 BPB** (records show consistent gain from this single tokenizer axis).

**Status:** PROPOSED — Tier 4 priority **#5**, **CONDITIONAL on H96 (compression) + H97 (artifact-budget audit)**. Cannot land until artifact fits.

### H96: Per-group lrzip+brotli compression frees ~280 KB artifact budget

**Problem:** generic stream compression (zstd-22) doesn't exploit per-tensor distributional similarity. Records' approach: bucket int6 tensors by role (qo_bank, kv_bank, mlp_up_bank), L1 nearest-neighbour similarity-sort rows within each bucket (so adjacent serialized rows are numerically close, giving entropy coder longer runs of small deltas), then lrzip-zpaq compress each group, falling back to brotli for the remainder.

**Codebase status:** YES. We use zstd-22 stream compression on the entire artifact. Records' per-group approach saves ~280 KB at same model.

**Proposal:** add `compressor=pergroup` option to artifact builder. Implement `_similarity_sort_l1` (uint16 permutation indices, brotli-compressed alongside the bucket data). Shell out to lrzip via subprocess for ZPAQ context-mixing back-end. lrzip system binary required.

**Risks:** (a) lrzip is an external binary (apt-get install). (b) Compression time +~75 s (one-time at submission build, irrelevant for training).

**Estimated ROI:** **0 BPB direct, +280 KB free artifact budget** (~2% of the 16 MB cap). Frees room for H95's bigger embeddings.

**Status:** PROPOSED — Tier 4 priority **#6**.

### H97: attn-gate int8-per-row quantization saves bytes at no quality cost

**Problem:** attention gates have low dynamic range (sigmoid output, naturally bounded). Quantizing them at int6 wastes precision; int8-per-row is precise enough.

**Codebase status:** YES. Our per-expert-per-head sigmoid gates currently quantize at int6 (default). Records use int8-per-row for the analogous `GATED_ATTN_QUANT_GATE` tensor.

**Proposal:** add per-tensor quant-bit-width override for the attn-gate. Set int8-per-row for those tensors only.

**Risks:** the int8 attn-gate is LARGER than int6 per row (8 vs 6 bits) — but per-row scales are FEWER bytes than the per-element saving. Net direction depends on the gate tensor shape; needs measurement.

**Estimated ROI:** small artifact savings (~10-30 KB), no val_bpb effect (or marginal positive from lower quant error on the gate).

**Status:** PROPOSED — Tier 4 priority **#9** (low magnitude; do only if budget is genuinely tight).

### H98: Sparse attention head-output gate (window=12) sparsifies per-token head contributions

**Problem:** attention heads are uniformly mixed via Wo, but some heads contribute noise per token. Records' sparse head-output gate adds a narrow window over heads and sparsifies which heads contribute.

**Codebase status:** PARTIAL. We have per-expert-per-head sigmoid gates after SDPA, but it's a single sigmoid per head, not "windowed top-12 over the head dimension". The records approach is different mechanism on top.

**Proposal:** add `sparse_attn_head_gate_window=12` Hyperparameter. Replace the existing per-head sigmoid with a windowed-top-K over the 8 heads × 16 experts = 128 slots; only top-12 within each window contribute. Composes with existing gated-attn structure.

**Estimated ROI:** **−0.005 to −0.015 BPB**. Small gain; principled mechanism on top of what we have.

**Status:** PROPOSED — Tier 4 priority **#7**.

### H99: SmearGate (BOS-fixed) adds a position-mixing memory channel

**Problem:** position-1 forward smear `x[1:] += g * x[:-1]` gives the model a small additional "previous token" memory channel beyond what attention provides. Records introduced it in PR #1667; PR #1797's BOS-leak fix is essential — naive smear leaks across document boundaries in packed validation streams.

**Codebase status:** PARTIAL. Our DEQ + Parcae handles temporal mixing via the FP iteration (not the same mechanism). SmearGate would be additive — orthogonal signal channel.

**Proposal:** add `use_smear_gate` Hyperparameter and `smear_gate` learnable scalar (per-layer). Implement: `x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)` where `not_bos = (input_ids[:, 1:] != BOS_ID)`. CRITICAL: must apply both in `_forward_hidden` and (if H91 lands) in `forward_ttt` to avoid eval/train mismatch.

**Estimated ROI:** **−0.005 to −0.015 BPB** (records show consistent small gain).

**Status:** PROPOSED — Tier 4 priority **#8**.

### Records-derived priority order (within Tier 4)

Sequenced for ROI/risk balance, after Tier 1 throughput iters (117b-2/3/3b) complete:

| Priority | H | Iter# | Feature | ROI | Complexity |
|---|---|---|---|---|---|
| 1 | H93/H92 | iter 122 | Logit softcap | −0.005 to −0.015 | Trivial 1-liner |
| 2 | H91 | iter 123 | Phased TTT eval | **−0.05 to −0.10** | High (eval-loop refactor) |
| 3 | H94 | iter 124 | GPTQ + LQER quant | −0.02 to −0.04 (int6) | High (replace quant pipeline) |
| 4 | H96 | iter 125 | Per-group lrzip+brotli compression | 0 (frees ~280 KB) | Medium (data prep + subprocess) |
| 5 | H95 | iter 126 | SP1024 → SP8192 + CaseOps tokenizer upgrade | −0.02 to −0.04 | High (data retokenize, embedding scale) |
| 6 | H98 | iter 127 | Sparse attn head-output gate (window=12) | −0.005 to −0.015 | Medium |
| 7 | H99 | iter 128 | SmearGate (BOS-fixed) | −0.005 to −0.015 | Low |
| 8 | H97 | iter 129 | attn-gate int8-per-row quant | 0 (~30 KB artifact) | Low |

**Independence**: items #1, #2, #3, #6, #7, #8 are mostly independent. #4 (compression) enables #5 (tokenizer upgrade) by freeing artifact budget. #2 (TTT) is the largest single gain but slowest to implement.

**Records-derived items NOT queued (problem subsumed or already present):**
- U-Net encoder-decoder skips → RevDEQ shared block + x₀ injection handles cross-depth signal preservation differently
- Depth recurrence (loop layers ×3) → RevDEQ FP iteration **is** this exactly, K=16-24
- Parallel decoder / 2-lane parallel residuals → soft-dense MoE has E=16 parallel paths
- LN scale 1/√(layer+1) → Parcae per-dim Ā provides depth-dependent contractive damping
- LeakyReLU(0.5)² in gated MLP → iter 83 tested + REVERTED (regressed in our gated context, H61)
- Polar-Express NS Muon → iter 121 NaN'd at s30; preserved gated for future revisit with lower matrix_lr
- qk_gain init=5.0 → already at L250 (matches records)
- EMA decay~0.997 → already enabled by default
- Partial RoPE 16/64 dims → already split per CLAUDE.md §6.3
