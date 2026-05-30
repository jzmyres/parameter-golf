# Recurrent-Depth Baselines

This document lists baselines comparable to the current Parameter Golf direction: fixed-point, looped, recurrent-depth, or adaptive-depth Transformer-family models that trade additional compute for fewer unique parameters. Replication state is tracked in `baselines/REPLICATION_REPORT.md`; source pins are tracked in `baselines/manifest.yaml`; search provenance is in `baselines/search_ledger.md`.

## Local Comparators

| Baseline | Type | Current evidence | Why it matters |
|---|---|---|---|
| Active RevDEQ + Parcae | Active Baseline | `experiments/training_logs/baseline.log`: 13.90M params, K-jitter `(16,24,32,64,96,128)`, final full-validation `val_bpb=1.462898`, int6 artifact 7.40 MB. DDP debug smoke passed on 2 GPUs. | Current code path: reversible fixed-point solve, Parcae-style damping/input injection, recursive multi-K prefix anchors. |
| 2026-03-25 ValCalib/GPTQ/XSA | Record Baseline | `experiments/training_logs/records_baseline_2026-03-25.log`: 26.99M params, exact int6 roundtrip `val_bpb=1.21287138`, exact sliding-window `val_bpb=1.18916331`, 12.77 MB submission. | Strong challenge record with different architecture; useful for absolute BPB gap but not mechanism-matched. |

## Core Comparable External Baselines

| Baseline | Source | Mechanism | Replication stance |
|---|---|---|---|
| Deep Equilibrium Models | Paper: <https://arxiv.org/abs/1909.01377>; code: <https://github.com/locuslab/deq> | Solves for a fixed point rather than stacking unique layers; sequence code includes DEQ-Transformer LM examples. | Source smoke passed. DDP adapter passed with a dependency-light DEQ import; full solver import needs SciPy/legacy setup. |
| TorchDEQ | Paper: <https://arxiv.org/abs/2310.18605>; code: <https://github.com/locuslab/torchdeq> | Modern PyTorch DEQ solver library. | Source smoke passed. 2-rank DDP adapter passed using a tiny TorchDEQ fixed-point model. |
| Reversible Deep Equilibrium Models | Paper: <https://arxiv.org/abs/2509.12917>; code: <https://github.com/sammccallum/revdeq> | Reversible DEQ training claims exact gradients and fewer function evaluations than classical implicit differentiation. | Source smoke passed. PyTorch DDP blocked because upstream is JAX/Equinox and declares Python >=3.12. |
| Universal Transformer | Paper: <https://arxiv.org/abs/1807.03819>; code: <https://github.com/tensorflow/tensor2tensor> | Reuses a Transformer block recurrently through depth, optionally with adaptive per-position halting. | Source smoke passed. PyTorch DDP blocked because upstream training is old TensorFlow/T2T. |
| MoEUT | Paper: <https://arxiv.org/abs/2405.16039>; module: <https://github.com/robertcsordas/moeut>; training code: <https://github.com/RobertCsordas/moeut_training_code> | Sparse MoE capacity compensates for the expressivity loss of Universal Transformer-style layer sharing. | Source smoke passed for module and training repo. DDP adapter passed for the module with a tiny MoEUT LM and for the training-code checkout with a generic adapter. |
| Parcae | Paper: <https://arxiv.org/abs/2604.12946>; code: <https://github.com/sandyresearch/parcae> | Stable looped language models with loop depth as a separate scaling-law axis. | Source smoke passed. DDP adapter passed after importing `parcae_lm`/`recpre`; paper-scale sweeps remain compute-blocked. |
| Huginn / recurrent-pretraining | Paper: <https://arxiv.org/abs/2502.05171>; code: <https://github.com/seal-rg/recurrent-pretraining> | Large recurrent-depth latent reasoning LM with variable test-time compute. | Source smoke passed. DDP adapter passed with a dependency-light source-file import; full package import needs Lightning/Transformers stack. |
| Iso-Depth Looped LM Scaling | Paper: <https://arxiv.org/abs/2604.21106>; code: <https://github.com/kschwethelm/looped-lm-scaling> | Measures recurrence value via a recurrence-equivalence exponent `phi`, separating true loop capacity from token-budget effects. | Source smoke passed. DDP adapter passed after importing `nanochat`; full 116-run sweep is compute-blocked locally. |
| Hyperloop Transformers | Paper: <https://arxiv.org/abs/2604.21254> | Hyper-connected looped middle-block Transformer designed to train looped LMs efficiently and recover performance with roughly half the unique parameters. | Included as a must-track SOTA baseline. DDP blocked because no official public code target was found. |
| Retrofitted Recurrence | Paper: <https://arxiv.org/abs/2511.07384>; code: <https://github.com/mcleish7/retrofitting-recurrence> | Converts pretrained non-recurrent LMs into depth-recurrent models with recurrence curricula. | Source smoke passed. DDP adapter passed. |
| MoDR | Paper: <https://openreview.net/pdf?id=9Pba4rcQbE>; code: <https://github.com/zhangxjohn/MoDr> | Mixture-of-depth-recurrent branches for test-time reasoning, routing among recurrence depths. | Source smoke passed. DDP adapter passed. |
| Mixture-of-Recursions | Paper: <https://arxiv.org/abs/2507.10524>; code: <https://github.com/raymin0223/mixture_of_recursions> | Adaptive token-level recursive depth. | Source smoke passed. DDP adapter passed after importing `model`. |
| Think-at-Hard | Paper: <https://arxiv.org/abs/2511.08577>; code: <https://github.com/thu-nics/TaH> | Selective latent iterations for hard tokens; dynamic-depth reasoning comparator. | Source smoke passed. DDP adapter passed. |

## Paper-Only / Conceptual Comparators

| Baseline | Source | Mechanism | Replication stance |
|---|---|---|---|
| Ouro / Scaling Latent Reasoning | Paper: <https://arxiv.org/abs/2510.25741>; project: <https://ouro-llm.github.io/> | Looped LMs with latent iterative computation and released weights. | `blocked_no_code`: no cloneable training repo found. |
| Reasoning with Latent Thoughts | Paper: <https://arxiv.org/abs/2502.17416> | Shows looped Transformers can recover effective depth for synthetic and downstream reasoning. | `blocked_no_code`. |
| Sparse Looped-MoE | Paper: <https://arxiv.org/abs/2605.09165> | Sparse MoE layers recover functional diversity across loop passes and improve early exits. | `blocked_no_code`; highly relevant to this repo's expert-routing direction. |
| Relaxed Recursive Transformers | Paper: <https://arxiv.org/abs/2410.20672> | Layer sharing with layer-wise LoRA relaxations. | `blocked_no_code`; useful prior for relaxing repeated-layer rigidity. |
| Dynamic Nested Depth | Paper: <https://arxiv.org/abs/2510.11001> | Routes selected tokens back through nested depth for extra processing. | `blocked_no_code`. |
| Parallel Loop Transformer | Paper: <https://arxiv.org/abs/2510.24824> | Parallelizes across loop dimension and shares first-loop KV cache to reduce loop latency. | `blocked_no_code`. |
| Thinking Deeper, Not Longer | Paper: <https://arxiv.org/abs/2603.21676> | Depth-recurrent Transformer study for compositional generalization. | `blocked_no_code`. |
| Loop, Think, & Generalize | Paper: <https://arxiv.org/abs/2604.07822> | Controlled recurrent-depth Transformer study for systematic generalization and depth extrapolation. | `blocked_no_code`. |
| Mechanistic Analysis of Looped Reasoning LMs | Paper: <https://arxiv.org/abs/2604.11791> | Analysis of cyclic fixed points, input injection, and normalization in looped LMs. | `blocked_no_code`; analysis-only comparator. |

## Comparison Axes

| Axis | Local RevDEQ + Parcae | External baseline signal |
|---|---|---|
| Parameter reuse | One shared RevDEQ-style block is iterated to a fixed point. | DEQ, RevDEQ, UT, Parcae, Huginn, Ouro, Hyperloop, and iso-depth work all treat recurrent depth as a parameter-saving axis. |
| Stability mechanism | Parcae damping/injection, reversible backward, multi-K prefix consistency, `rho_F` diagnostics. | DEQ uses implicit fixed-point solvers; RevDEQ emphasizes reversible gradients; Parcae and Hyperloop emphasize stable looped LM training. |
| Expressivity recovery | Dense-soft routed experts inside the repeated block. | MoEUT, Sparse Looped-MoE, MoDR, Mixture-of-Recursions, and Think-at-Hard point to sparse/dynamic depth or experts as recovery mechanisms. |
| Replication target | FineWeb SP1024 BPB under 16 MB artifact constraints. | Most external papers target WikiText/PTB, C4/SlimPajama, synthetic reasoning, or large pretraining sweeps, so local smoke validates code viability rather than paper-scale BPB parity. |

## Screening Notes

- `chenllliang/DnD-Transformer` was screened out: it is an image-generation project, not the depth-recurrent Transformer paper.
- `isl-org/MiDaS` was screened out: it is monocular depth estimation, not a recurrent-depth Transformer baseline.
- ALBERT-style cross-layer sharing is background but not a looped/test-time recurrent-depth baseline.
- RWKV, RetNet, SSMs, and RecurrentGemma are sequence-recurrent families, not recurrent-depth parameter-sharing baselines for this comparison.
