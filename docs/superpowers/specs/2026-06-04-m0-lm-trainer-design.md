# M0 LM Trainer — Design Spec

- Date: 2026-06-04
- Status: Draft for review
- Aligns to: `reports/opg_doc.tex` (the project spec); this design also drives an `opg_doc.tex` update.

## Context & goal

Replace the bloated rich `train_gpt.py` (~10.4k lines, 7× over the upstream 1500-line
hard-stop) with an **organized** implementation that **preserves and optimizes the core
improvement** of `opg_doc.tex`: *recurrent depth buys task utility under a fixed memory
budget*. Two explicit goals drive every decision:

- **G1 — minimize resource** (training memory, params, artifact ≤16 MB, FLOPs/600 s, KV).
- **G2 — maximize expressiveness** (val_bpb, effective depth, representational rank).

These form a **Pareto frontier** (you cannot independently max both); each mechanism is
kept only if it advances *expressiveness per unit resource*, judged by the metrics below.

## Architecture (retained core, lean-but-complete)

**M0 reversible recurrence (the core; floor-free, explicit inverse).** Tied, token-injected,
additive-coupling reversible recurrence over `s_k=(a_k,b_k)` with input embedding `x0`:

```
a_{k+1} = a_k + F_θ(RMSNorm(b_k + x0))
b_{k+1} = b_k + G_θ(RMSNorm(a_{k+1} + x0))
```

Inverse is exact and explicit (each update depends only on the *other* stream — strictly
cleaner than RevFFN's cross-attention coupling, which needs a fixed-point iteration):

```
b_k = b_{k+1} − G_θ(RMSNorm(a_{k+1} + x0))
a_k = a_{k+1} − F_θ(RMSNorm(b_k + x0))
```

Invertibility is algebraic (no Parcae floor / contraction requirement — confirmed by
RevFFN's no-floor design). `F_θ`/`G_θ` are pre-norm blocks: **MLA attention + SwiGLU MoE**.

- **Full reversible BPTT backward** (custom `autograd.Function`): reconstructs `(a,b)`
  backward via the explicit inverse → **O(1) activation memory in K** (G1). Full (not
  truncated) BPTT — truncation measurably *lowers* effective depth φ (Iso-Depth: 0.46→0.38).
- **Readout:** final recurrent state `z_K` (= `0.5(a_K+b_K)`) → **MoS output head** (mixture
  of softmaxes over the tied embedding basis). **No halting readout** (it serves P1
  *measurement* nestedness, not G1/G2).
- **MLA attention** in `F_θ`/`G_θ`: low-rank joint KV compression (G1: KV cache + params);
  `kv_latent_dim` is an erank-tuned knob (start `dim//2`; DeepSeek-V2 goes far lower).
- **MoE SwiGLU experts** (the effective-depth basis): low-rank (LoRA-style) experts shared
  across recurrence steps; per-step routing → combinatorial *effective depth* recovering the
  diversity weight-tying loses (MoEUT, Sparse-Looped-MoE, Relaxed-Recursive). **Router is a
  control-experiment knob `router_type ∈ {relu, softmax}`** (see Experiments). MoE is
  **hard-on** (mandated for G2). Routing is **smooth** (no hard top-k) → reversibility-safe.
- **Low-rank everywhere it pays:** experts, MLP, MLA Q/KV, MoS components are low-rank with
  rank set per-matrix on the **erank** frontier (Roy & Vetterli 2007; ARSVD). Small matrices
  (norms, gates, **router**, the tiny vocab=1024 embedding) stay full-rank.
- **Tied embeddings** (input↔output basis the MoS mixes over) — small win at vocab 1024, kept.
- **int6 + zstd ≤16 MB artifact** (G1, required); post-hoc (QAT-late dropped).
- **Optimizer:** Muon (matrices) + AdamW (embeddings/scalars/router), PE-NS, warmdown LR.

**Dropped** (don't serve G1/G2; were P1-obscuring or FP/diagnostic cruft): halting readout,
Parcae damping/floor, Lyapunov, fixed-point consistency, DEQ-implicit/IFT backward, spectral
probes, Dirichlet-UCB complexity (→ ReLU/softmax routing), QAT-late, rich diagnostics,
Bigram/FSQ/CTP/NSA/smear.

## Principled metrics (the two goals)

| Goal | Primary | Supporting |
|---|---|---|
| **G1 resource** | `R_act(K)` = peak act-mem(K)/mem(K₀) → must be ≈1 (reversibility) | params + **artifact bytes (≤16 MB gate)**; KV bytes/token + MLA ratio; **FLOPs/token + active-expert fraction** (MoE compute); peak VRAM; wall-clock/step (600 s) |
| **G2 expressiveness** | `val_bpb` (FineWeb SP1024) | **φ recurrence-equivalence exponent** (effective depth; fit over r∈{1,2,4,8}); **erank** (representation + per-matrix rank sufficiency); MoS output rank; MoE route-depth diversity; paired depth-gain `G_T` (synthetic) |
| **Joint (Pareto)** | bpb/param · bpb/KV-byte · bpb at constant act-mem · **Δφ per FLOP** | — |

## Control experiments (the empirical gates)

1. **Router: ReLU vs softmax** — matched everything else; measure val_bpb, φ, erank
   (G2) and FLOPs / active-expert fraction, reconstruction error (G1/correctness). ReLU
   (ReMoE-style, adaptive-L1 sparsity + load-balance) wins only if it cuts active-fraction
   without losing val_bpb/φ at equal reconstruction fidelity.
2. **Effective depth φ** over r∈{1,2,4,8}; target φ→1; verify MoE raises φ vs no-MoE.
3. **MLA latent-dim / erank sweep** — find the smallest `kv_latent_dim` before erank/val_bpb degrade.
4. **Reversibility:** reconstruction error `<1e-5` (fp64); grad-equivalence (reversible
   backward vs ordinary autograd) — hard correctness gate; `R_act(K)` ≈ const.
5. (optional) MoS-component-count, tied-vs-untied embedding.

## Evaluation

- **FineWeb10B SP1024 → val_bpb** (primary; the 600 s / 16 MB resource benchmark — keep).
- **Synthetic depth-hard (S5/parity, `p1_synthetic`)** for effective-depth/φ/paired-gain
  (the comparator class the recurrent-depth literature uses).
- Promotion: post-int6 val_bpb improves ∧ artifact ≤16 MB; φ and erank reported; reversibility gate green.

## Hyperparameters (grounded starting points)

`model_dim=768`, `num_heads=8`, `num_kv_heads=4`, `mlp_mult=3.0`, MLA `kv_latent_dim≈dim/2`
(erank-tuned), `num_experts=16` low-rank (rank erank-tuned), recurrence K-set `{32,64,128}`,
full reversible BPTT, MoS components (start small, e.g. 2–4 at vocab 1024). Citations recorded
in `opg_doc.tex` Related Work.

## Build approach & repo organization

- **Hybrid build:** clean LM-scaffold chassis (from the 1,126-line launch snapshot `a15093a`,
  minimized) + ported int6 artifact + the M0 reversible core (forward from `p1_synthetic`,
  clean explicit-inverse reversible backward) + MLA/MoE/MoS modules.
- **Archive** the rich arch + rich-only components/tests to read-only `legacy/` (tag
  `legacy/rich-revdeq-moe` already preserves it). **`src/` reorg deferred** to a separate change.
- **Alignment:** update `opg_doc.tex` (§Models simple readout; MLA/MoE/MoS retained, removed
  from §Goal exclusions; two-goal metrics; Related Work). Fix stale CLAUDE.md ref-impl paths.

## Line budget (resolved 2026-06-04)

**No hard line cap on the research `train_gpt.py`** (user directive; the CLAUDE.md ≤1500
invariant was removed). The model carries the mandated mechanisms (MLA + MoE + MoS + reversible
backward), so the upstream ≤1500-line note is treated as a *reference-baseline* aspiration only.
Target: lean code with each mechanism in a clearly-bounded module (est. ~2,000–2,500 lines), not
a line count.

## Testing strategy

Reversibility reconstruction `<1e-5` (fp64); reversible-backward grad-equivalence vs ordinary
autograd; `R_act(K)` constancy; φ/erank computation unit tests (known inputs); int6 artifact
roundtrip; DDP shape smoke; BPB smoke; ReLU/softmax router parity smoke; updated audit/contract tests.

## Related work (also added to opg_doc.tex)

- **Reversible×MoE:** RevFFN (arXiv 2512.20920) — reversible blocks + MoE, 49% VRAM, no floor.
- **Reversible coupling:** RevNet (Gomez 2017), Reformer (Kitaev 2020).
- **Smooth-sparse routing:** ReMoE/ReLU routing (arXiv 2412.14711), DSelect-k (NeurIPS 2021); Soft-MoE.
- **Expressivity recovery for tied depth:** MoEUT (arXiv 2405.16039), Sparse-Looped-MoE
  (arXiv 2605.09165), Relaxed Recursive Transformers (arXiv 2410.20672).
- **Effective depth metric φ:** Iso-Depth Looped LM Scaling (arXiv 2604.21106).
- **Recurrent depth / tied:** Universal Transformer (arXiv 1807.03819), Parcae (arXiv 2604.12946),
  Huginn (arXiv 2502.05171), weight-sharing inductive bias (Saunshi et al. 2025).
- **MLA:** DeepSeek-V2 (arXiv 2405.04434) — low-rank joint KV compression.
- **MoS:** Breaking the Softmax Bottleneck (Yang et al., ICLR 2018, arXiv 1711.03953).
- **Effective rank:** Roy & Vetterli (2007); ARSVD (arXiv 2504.20078).

## Risks

- ReLU-router sparsity may not hold expressiveness at small scale → control experiment #1 gates it.
- Reversible backward correctness is the critical risk → grad-equivalence test is a hard gate.
- MoE compute (even smooth-sparse) may strain 600 s → FLOPs/active-fraction metric gates it.
- 1500-line budget vs mandated mechanisms → flagged open decision above.
