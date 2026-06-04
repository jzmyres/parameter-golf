# Parameter Golf Context

This file records shared project language that is broader than a single
iteration ledger entry. Operational rules remain in `CLAUDE.md` and
`EXPERIENCE.md`; experiment-specific evidence remains in
`experiments/docs/hypotheses.md`.

## Baseline Vocabulary

| Term | Meaning |
|---|---|
| Active Baseline | The current in-repo comparator, owned by `train_gpt.py::Hyperparameters` and the latest promoted/current logs. It is not automatically the final research design; current dense-MoE/Dirichlet-UCB/MLA/MoS runs are prior diagnostic evidence until the final minimal P1 path is implemented. |
| Final Minimal P1 Design | The canonical target in `reports/opg_doc.tex`: a tied, token-injected, additive-coupling reversible recurrent model `M0` with standard MHA/GQA plus SwiGLU, halting readout, a depth-hard positive-control protocol, and a P2 target of full reversible BPTT. The current Tier-1 harness tests P1 detectability and structural reconstruction; it does not claim activation-memory scaling until the memory-saving backward path is implemented and measured. |
| P1 Depth Utility | Primary project claim: paired task gain `G_T(K_lo,K_hi)>0` at fixed memory on data that requires depth. P1 must pass before MoE-basis or deployment-feature claims matter. |
| P2 Reversible Memory | Independent memory claim: reversible reconstruction makes activation memory approximately constant in recurrent depth, verified by activation slope and reconstruction error. |
| P3 MoE Basis | Secondary claim, staged after P1: a MoE basis recovers tying's lost depth diversity under matched budget. It is not part of the P1 core. |
| Depth-Hard Synthetic | Tier-1 detectability task with known depth demand. The primary task is iterated `S5` permutation composition; the implemented alternate is parity (`--task s5\|parity`). Graph reachability with tunable diameter is planned, not yet implemented. |
| P1 Synthetic Harness | `experiments/p1_synthetic.py`, the executable Tier-1 path for the final-minimal design. Variants `control`, `m0`, and `mclk` are the complete active interface. It intentionally rejects MoE, cache, and quantization switches so the core runner only answers the P1 detectability question. It reports paired confidence intervals for depth gain; a mean-only gain is not gate evidence. |
| Feedback-Stage Harness | `experiments/p1_feedback_stages.py` plus `experiments/run_feedback_stage_pipeline.sh`. It executes S+1 MoE/static-MoE, S+2 hidden-state cache proxy, and S+3 fake-int6 diagnostics without widening the P1 promotion runner. |
| Positive Control | A vanilla looped transformer with input injection and stored activations, run through the same Tier-1 harness. If it does not scale, the harness or task is broken. |
| Clock Diagnostic | `M_clk`, the `M0` model with explicit recurrent-step embedding. It is an upper bound on depth-conditioned computation and a diagnostic ceiling, not a default model commitment. |
| Halting Readout | A readout that mixes recurrent states with a saturating gate so a deeper budget can reproduce a shallower output, giving nested achievable risk when the gate can saturate. |
| Record Baseline | Historical Parameter Golf reproductions and records, especially runs preserved under `records/`, `records_baseline/`, and their logs. These are challenge comparators, not necessarily the current architecture. |
| Comparable Baseline | External research baselines that use recurrent depth, implicit depth, looped transformers, or sparse looped layers to trade extra computation for fewer unique parameters. |
| Fixed-Point Consistency | Legacy fallback pressure from the iter163/iter172 line. It is not part of the active loss and startup validation rejects attempts to re-enable prefix anchors or positive multi-K consistency. |
| Fixed-Point Diagnostics | Advisory convergence measurements such as spectral radius, residual, and iterate convergence. They can explain stability or cache-readiness, but they are not active training pressure. |
| Fixed-Point Pressure | Any training loss or launch knob that directly pushes the recurrent trajectory toward contraction or convergence, including nonzero `lyapunov_coef`. This is distinct from diagnostics and is rejected for the active finite-horizon baseline. |
| Paired Depth Gain | Evaluate the same validation sequences at `(K_lo,K_hi)` and report `G_T`; for P1 it is a gate only on depth-hard data with a passing positive control, paired confidence interval lower bound above zero, and controlled `NDR_epsilon`. |
| Effective Recurrent Depth | Diagnostic realized-transition metric. Legacy fields include `ED_update` and `ED_logit`; final-minimal P1 treats task gain and recurrence-equivalence exponent as the headline depth evidence. |
| Route-Depth Diagnostics | Router-node metrics such as `route_depth_nmi_mean`, `route_depth_nmi_max`, and `expert_util_mean`. They are S+1 mechanism diagnostics, not P1 gates. |
| Expert Slot | One serial expert position inside the recurrent block. Each slot owns an attention expert bank and an MLP expert bank. |
| Expert Layout | The allocation of expert modules as `experts_per_slot x expert_slots` inside one recurrent block. This is an expressiveness/throughput axis, not Lipschitz or fixed-point control. |
| Expert Slot Order | The within-slot ordering of attention and MLP expert updates: parallel, attention then MLP, or MLP then attention. |
| Exact Multi-Depth Cache | Autoregressive inference cache that preserves each recurrent depth's token states or K/V tensors. It is the exact finite-horizon cache policy and is expected to scale linearly with recurrent depth. |
| Terminal Hidden-State Cache | The first constant-KV candidate: store previous tokens' terminal `z_K` state, or K/V derived from that state, and measure quality gap against the exact multi-depth cache. |

## Current Comparison Question

The current research question is whether the final minimal P1 design can turn
extra recurrent computation into paired task gain on depth-hard data under a
fixed memory budget. The current codebase still contains the richer
dense-MoE/Dirichlet-UCB/MLA/MoS implementation; that path is diagnostic
evidence and implementation debt. Use `experiments/p1_synthetic.py` only for
Tier-1 P1 evidence from `M0`, the clock diagnostic, and the positive-control
harness. Use `experiments/p1_feedback_stages.py` only for staged mechanism and
deployment diagnostics; those rows do not compensate for a failed S+0 gate.
Do not treat small-LM K-sweeps or later-stage ablations as project-level
confirmation until P1 passes.
The baseline inventory for recurrent-depth and
looped-model comparisons lives in `experiments/docs/recurrent_depth_baselines.md`,
with replication state tracked under `baselines/`.
