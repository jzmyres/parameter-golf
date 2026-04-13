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

### H5: Routing collapse is caused by expert count alone
**Observation:**
- (12exp, rank128, WD=0.06, β=0.35) → collapsed at step 400
- (8exp, rank128, WD=0.06, β=0.35) → stable 743 steps
- (16exp, rank64, WD=0.09, β=0.10) → stable 372 steps
- (16exp, rank128, WD=0.18, β=0.10) → collapsed at step 400
**CAN claim:** routing collapsed under these specific (E, R, WD, β) combinations.
**CANNOT claim:** "12+ experts is inherently unstable" — never tested at high WD. Higher WD might stabilize (per verified H9).
**CANNOT claim:** "collapse scales as E×R²" — only observed at low WD.
**To isolate:** test (8exp vs 12exp) at identical (WD=0.72, β=0.20, rank128).
**Confounds:** WD, β, and routing regularization all differed across tests.

### H6: β has a U-shaped optimum (at WD=0.18)
**Observation:** β=0.05 (1.820), β=0.10 (1.796), β=0.20 (1.818). Best at 0.10.
**Confounds:** Different β values may interact with K jitter range. Need: re-test at optimal WD with wider K jitter.

### H7: WD=0.18 is optimal for val_bpb
**Observation:** WD=0.09 (1.796), WD=0.18 (1.754), WD=0.36 (1.779). Best at 0.18.
**Confounds:** β was held at 0.10; optimal β may differ at each WD (per H9). Need: 2D sweep.

### H8: Expert rank is more valuable than model dim at constant throughput
**Observation:** dim=896/rank96 worse than dim=768/rank128. dim=1024/rank80 smoke FAILED.
**Confounds:** Only tested 2 dim values. Need: systematic grid at constant compute.

### H10: Model exploits training K rather than finding a good fixed point
**Observation:** First valid K-sweep (iter 12b) showed K=8 val_bpb 1.778 but K=64 val_bpb 1.825 (FP worse).
**Confounds:** Only tested at WD=0.36/β=0.10/K∈{4,8}. Need: test if wider K jitter (H12) fixes it.

### H11: Bigram hash table size is low-value relative to transformer body capacity
**Observation:** Records use 2048-4096 bigram vocab vs our 65536. Reducing freed 71% of params.
**Confounds:** Reduced simultaneously with dim increase. Need: controlled test of bigram size at fixed everything else.

---

## PROPOSED (untested)

### H17: WD controls training stability (prevents collapse) — PRIORITY
**Claim:** Higher WD → smaller weight magnitudes → smaller Jacobian → prevents training degeneracy (routing collapse, gradient explosion, loss divergence). WD is the lever for STABILITY, independent of convergence speed.
**Mechanism:** WD shrinks ||W|| → shrinks spectral_norm(∂f/∂z) → contraction condition β×||∂f/∂z|| < 1 is easier to satisfy → training stays in the stable basin.
**Prediction:** A config that collapses at WD=X should become stable at WD=2X without changing β.
**Partial evidence:** H9 verified (WD=0.36/β=0.20 FAILED → WD=0.72/β=0.20 PASSED). But H9 only tested one β value.
**To verify:** Test a config that collapsed (e.g., 12exp rank128) at progressively higher WD {0.36, 0.72, 1.44} with β held constant. If it stabilizes, WD→stability is confirmed.
**Isolation:** β must be held constant. If β also changes, the effect is confounded.

### H18: β controls DEQ convergence speed — TESTED
**Claim:** Higher β → faster convergence → better FP quality.
**Test:** Iter 13 (β=0.20) vs iter 14a (β=0.30), both at WD=0.72. Single variable change.
**Evidence:**
- β=0.30 residuals 40-50% lower at every K (converges ~2× faster per iteration) ✓
- β=0.30 gg_iter[15]=0.067 vs β=0.20's 0.082 (gate closer to zero) ✓
- BUT: β=0.30 K=8→K=64 degradation +0.020 vs β=0.20's +0.009 (2× worse FP quality) ✗
**Verdict:** ⚠️ PARTIALLY TRUE — β controls convergence SPEED (verified) but NOT FP quality. Higher β converges faster TO A DIFFERENT (WORSE) fixed point.
**Mechanism (revised):** The coupled-state relaxation `z_{n+1} = (1-β)z_n + β·f(z_n)` with finite K doesn't reach the true FP — it reaches a β-dependent intermediate. Lower β intermediates are closer to the optimal FP.
**Implication:** Use the LOWEST β that converges within the K budget. Higher β is NOT better even when WD stabilizes it. β=0.20 at WD=0.72 beats β=0.30 at WD=0.72 on FP quality.

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
**Status:** OBSERVED but initial interpretation was premature. The dome-formation delay may be β-dependent, not WD-dependent.
**Revised test:** Compare dome formation SPEED at (WD=0.72/β=0.30) vs (WD=1.44/β=0.30) — does higher WD make the dome form earlier in training?

### H12: Wider K jitter fixes FP quality degradation
**Claim:** Training at K∈{4,8,12,16} forces the model to optimize FP quality at all K, making K-sweep monotone.
**Test:** Iter 13 — WD=0.72/β=0.20 + K jitter {4,8,12,16}
**Evidence:** K=8→K=16 Δ reduced from +0.025 (iter 12b) to +0.0002 (125× improvement). K=8→K=64 Δ reduced from +0.047 to +0.009 (5× improvement). Strong evidence but NOT fully verified — small residual degradation at K=32/64 remains.

### H13: The optimal (WD, β) pair lies on a diagonal
**Claim:** As WD increases, optimal β increases proportionally.
**Test:** Planned — need 2D sweep data
**Partial evidence:** H19 suggests WD_min ∝ β². The diagonal relationship exists for STABILITY (min WD per β), but may differ for PERFORMANCE (optimal WD per β for best val_bpb).

### H14: Router sigmoid gate improves expert utilization
**Claim:** Input-dependent sigmoid gate on softmax routing weights helps DEQ convergence.
**Test:** Queued (arch exploration block)

### H15: Quant-noise injection eliminates post-quant degradation
**Claim:** Quantization noise during DEQ iterations makes FP robust to int6.
**Test:** Queued (advanced training block)

### H16: Single-step diffusion CTP enriches embedding gradients
**Claim:** Noisy soft-embed input + CTP denoising gives gradient to more embedding rows.
**Test:** Queued (advanced training block)

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
| **13** | **WD=0.72, β=0.20, K jitter {4,8,12,16}** | **TBD** | **running** | **H9, H12** |

## Iteration Schedule (upcoming)

**Phase 1: Verify WD→stability and β→convergence (H17, H18) — PRIORITY**

| Iter | Config change | Tests | Depends on |
|---|---|---|---|
| **13** | WD=0.72, β=0.20, K jitter {4,8,12,16} | H17 (stable?), H12 (K jitter→FP) | — |
| **14a** | **WD=0.72, β=0.30** (hold WD, increase β) | **H18, H19** (gate went flat → β too high for WD=0.72) | **running** |
| **14b** | **WD=1.44, β=0.30** (double WD to match β) | **H19** (does higher WD restore dome-shaped gate at β=0.30?) | iter 14a |
| 14c | WD=0.72, β=0.10 (decrease β at same WD) | H18 control: slower convergence? | if needed |
| 15 | WD=1.44, β=0.20 (hold β=0.20, increase WD) | H17 (more WD at proven-good β) | if needed |

**Decision point after Phase 1:**
- H17 verified → WD is the stability lever, set to minimum stable value
- H18 verified → β is the convergence lever, set to max that doesn't hurt val_bpb
- Lock (WD, β) pair for all subsequent experiments

**Phase 2: Stability-enabled scaling (test H5 revisited)**

| Iter | Config change | Tests | Depends on |
|---|---|---|---|
| 16 | 12exp rank128 at locked high WD | H5 revisited (collapse addressable by WD?) | Phase 1 |
| 17 | Gate statistics infrastructure | Observability for arch exploration | iter 16 |

**Phase 3: Architecture exploration**

| Iter | Config change | Tests | Depends on |
|---|---|---|---|
| 18 | Router sigmoid gate (input-dependent, init open) | H14 | iter 17 |
| 19 | Injection mechanism exploration | New hypothesis | iter 18 |
| 20 | Post-norm vs pre-norm | New hypothesis | iter 19 |

**Phase 4: Advanced training objectives**

| Iter | Config change | Tests | Depends on |
|---|---|---|---|
| 21 | Quant-noise injection in DEQ iterations | H15 | iter 20 |
| 22 | Single-step diffusion CTP | H16 | iter 21 |

**Phase 5: Scaling law experiments**

| Iter | Config change | Tests | Depends on |
|---|---|---|---|
| 23-27 | Grid: vary (dim, rank, experts) at locked config | Scaling law | iter 22 |

### Permanent protocol for all iterations
- K jitter: {4, 8, 12, 16} (train at varying K to force good FP)
- K-sweep: {4, 8, 16, 32, 64} with fast eval (256 seqs) + per-K diagnostics
- Pre-commit: /simplify → coderabbit → pr-review-toolkit → superpowers review
- Save full-precision weights (model_full.pt) before quantization
- Update this hypothesis log after each iteration
- Stability over task performance
