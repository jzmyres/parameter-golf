# ADR 0002: Final Minimal P1 OPG Design

- Date: 2026-06-04
- Status: Accepted

This ADR **defines the interface** the P1 implementation conforms to. Dated run
results are NOT recorded here — they live in
[`experiments/docs/hypotheses.md`](../../experiments/docs/hypotheses.md) (see Evidence).

## Issue

The previous Pure Finite Reversible OPG framing correctly rejected global
fixed-point pressure, but the active implementation still mixed too many axes —
dense MoE, Dirichlet-UCB routing, MLA-style expert attention, MoS head,
quantization, Parcae-style damping, and language-model scoring. A stable `16x1`
diagnostic run produced negative paired depth gains, and that result did not
localize whether the failure came from the harness, the depth signal, block
expressivity, routing, or deployment features. We need an attribution-first
minimal design with a depth-hard positive control before any small-LM work.

## Decision

Make the final P1 design minimal and attribution-first:

- Use **M0**: tied, token-injected, additive-coupling reversible recurrence with
  standard MHA/GQA + SwiGLU and a halting readout. The P2 target is full
  reversible BPTT.
- Establish P1 on a **Tier-1 depth-hard synthetic task** before small-LM work,
  with a vanilla looped-transformer **positive control** and an **M_clk** clock
  diagnostic run through the same harness.
- Exclude from the P1 science path: Dirichlet-UCB routing, MLA, MoS,
  quantization, distillation, UFID, anytime losses, DEQ/implicit backward,
  consistency, spectral pressure, and Lyapunov pressure.
- Keep fixed-point diagnostics and `lyapunov_coef` rejection as operating
  constraints for the current codebase.
- Stage P3 MoE-basis recovery only after P1 holds, in a separate harness or an
  explicit ablation.

## Interface (the implementation conforms to this)

- **P1 harness** = `experiments/p1_synthetic.py`. The complete CLI surface is
  `--variant {control, m0, mclk}`. It intentionally exposes **no** MoE, cache, or
  quantization switches and rejects them.
- **Model contract:** `control` = vanilla looped transformer (positive control);
  `m0` = additive-coupling reversible recurrence (MHA/GQA + SwiGLU, halting
  readout); `mclk` = `m0` plus per-step clock injection. `m0`/`mclk` MUST
  reconstruct the initial state to `< 1e-5` (structural reversibility).
- **Promotion gate:** depth gain is reported as a **paired 95% confidence
  interval** (`G_nll` low/high), never a mean alone; promotion requires the
  **lower bound positive** plus a controlled `NDR_epsilon`. The Tier-1 harness
  uses ordinary autograd for detectability + reconstruction; it does **not** claim
  activation-memory scaling until the memory-saving backward path is implemented
  and measured (that is the P2 claim, tracked independently).
- **Shell runner** `experiments/run_p1_synthetic_pipeline.sh` is parameterized
  only over task/budget knobs (`TASK, SEQ_LEN, ITERATIONS, TRAIN_DEPTHS,
  EVAL_DEPTHS, PAIRS, MODEL_DIM, NPROC`) so a failed positive control can be
  localized without adding model machinery.
- **Feedback stages** live in `experiments/p1_feedback_stages.py` +
  `experiments/run_feedback_stage_pipeline.sh`, separate from the P1 runner:
  S+1 (MoE/static-MoE route-depth NMI / AEBR / utilization), S+2 (hidden-state
  cache *proxy* — explicitly not an autoregressive-KV proof), S+3 (fake-int6
  sensitivity). They are mechanism/deployment evidence only and do **not** promote
  P1 while the S+0 positive control fails.

## Evidence

Run results are tracked in `experiments/docs/hypotheses.md`
(§ "Current Feedback-Stage Verification"), not inlined here. **Standing status
(2026-06-04):** P1 is **not promoted** — the S+0 positive-control depth-gain CI
crosses/falls below zero and `NDR_epsilon ≈ 0.5`, while `M0`/`M_clk` retain
near-precision structural reconstruction.

## Consequences

The current rich `train_gpt.py` architecture is **not** the final research
target; it remains prior diagnostic evidence and a deployment-feature pool, to be
archived. Future P1 work starts from the Tier-1 harness + positive control + M0 +
M_clk, then moves to small language modeling only after the synthetic
detectability gate passes.

The standing scientific next step is **detectability repair** of the Tier-1 task /
positive-control capacity — not reintroducing MoE, fixed-point pressure, KV-cache,
or int6 as explanations for a failed S+0 gate. Those are reintroduced only after
P1 passes, each as a one-variable ablation with its own promotion gate.

Note (readout confound, from pre-commit review): the halting readout pools over
all intermediate states, so it can suppress depth gain by concentrating weight on
early states. Detectability-repair work should check halting-gate saturation
before attributing a failed control to task hardness.
