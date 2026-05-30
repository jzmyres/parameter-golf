# Recurrent-Depth Baseline Search Ledger

Search pass: 2026-05-24. Scope: Transformer-family models that reuse depth, loop hidden-state computation, solve a fixed point, or allocate recurrent/test-time depth to recover capability under parameter sharing. Sequence-recurrent alternatives such as RWKV/RetNet/SSMs are out of scope unless they explicitly use recurrent depth as the parameter-saving mechanism.

## Queries Run

- `Hyperloop Transformers recurrent loop transformers`
- `Hyperloop Transformers arXiv code GitHub mHC Transformer`
- `recurrent depth transformer language model looped transformer parameter`
- `looped language models recurrent depth transformer code`
- `Huginn recurrent depth language model code GitHub`
- `Scaling up Test-Time Compute with Latent Reasoning recurrent depth code`
- `Retrofitting Language Models with Looped Transformers code`
- `Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence GitHub`
- `Mixture-of-Depth-Recurrent Transformer MoDR GitHub`
- `MoEUT training code GitHub`
- `Mixture of Recursions Transformer GitHub`
- `Think-at-Hard dynamic latent thinking GitHub`
- `Relaxed Recursive Transformers arxiv code`
- `Dynamic Nested Depth DND arxiv code`
- `Parallel Loop Transformer Efficient Test-Time Computation Scaling arxiv`
- `Thinking Deeper Not Longer Depth-Recurrent Transformers GitHub`
- `Loop Think Generalize Implicit Reasoning Recurrent-Depth Transformers`
- `Mechanistic Analysis of Looped Reasoning Language Models arxiv`
- `Sparse Layers are Critical to Scaling Looped Language Models code`

## Included Code Targets

| Baseline | Code target | Pin |
|---|---|---|
| DEQ | <https://github.com/locuslab/deq> | `1fb7059d6d89bb26d16da80ab9489dcc73fc5472` |
| TorchDEQ | <https://github.com/locuslab/torchdeq> | `4f6bd5fa66dd991cad74fcc847c88061764cf8db` |
| RevDEQ | <https://github.com/sammccallum/revdeq> | `c3ae05e47f30e92ea0eb41aa979e95c7c22a9939` |
| Universal Transformer/T2T | <https://github.com/tensorflow/tensor2tensor> | `bafdc1b67730430d38d6ab802cbd51f9d053ba2e` |
| MoEUT | <https://github.com/robertcsordas/moeut> | `87f04973d3db3ad7cb4ca67a0750241e40210d3c` |
| MoEUT training code | <https://github.com/RobertCsordas/moeut_training_code> | `521093a88007524b7ec9112be78769eb0d7c465f` |
| Parcae | <https://github.com/sandyresearch/parcae> | `6e4519556e5a9d444793c56aa9fd9ac2965f097b` |
| Huginn/recurrent-pretraining | <https://github.com/seal-rg/recurrent-pretraining> | `1ea7220ec7eb42d13e89db0663df254d0bcdc28e` |
| Iso-depth looped LM | <https://github.com/kschwethelm/looped-lm-scaling> | `ad7d8f3055501e4c84ced7797f772dce2f8e2107` |
| Retrofitted Recurrence | <https://github.com/mcleish7/retrofitting-recurrence> | `e21042a6f4708b9f1ad2ca1f767948debe5a9289` |
| MoDR | <https://github.com/zhangxjohn/MoDr> | `4d77e558bcabfb551258daaacbee47ced3661987` |
| Mixture-of-Recursions | <https://github.com/raymin0223/mixture_of_recursions> | `53d0fee43632b53fb9bddd4acf9af7a2eba43bb6` |
| Think-at-Hard | <https://github.com/thu-nics/TaH> | `4ba7145499d8c4635a3effa5c66bfafd99c2f5bd` |

## Paper-Only or Blocked Code Targets

| Baseline | Source | Reason |
|---|---|---|
| Hyperloop Transformers | <https://arxiv.org/abs/2604.21254> | No official public code found during this pass. |
| Ouro / Scaling Latent Reasoning | <https://arxiv.org/abs/2510.25741>, <https://ouro-llm.github.io/> | Project page/model cards, no cloneable training repo found. |
| Reasoning with Latent Thoughts | <https://arxiv.org/abs/2502.17416> | No official public code found. |
| Sparse Looped-MoE | <https://arxiv.org/abs/2605.09165> | No official public code found. |
| Relaxed Recursive Transformers | <https://arxiv.org/abs/2410.20672> | No official standalone code target confirmed. |
| Dynamic Nested Depth | <https://arxiv.org/abs/2510.11001> | No official code target found. |
| Parallel Loop Transformer | <https://arxiv.org/abs/2510.24824> | No official code target found. |
| Thinking Deeper, Not Longer | <https://arxiv.org/abs/2603.21676> | No official code target found. |
| Loop, Think, & Generalize | <https://arxiv.org/abs/2604.07822> | No official code target found. |
| Mechanistic Analysis of Looped Reasoning LMs | <https://arxiv.org/abs/2604.11791> | Analysis paper; no official code target found. |

## Screened False Positives and Near-Misses

- `chenllliang/DnD-Transformer` is an image-generation project, not the depth-recurrent Transformer paper.
- `isl-org/MiDaS` is monocular depth estimation, not Mixture of Depthwise Experts in recurrent-depth Transformers.
- ALBERT-style sharing is relevant background but not a looped/test-time recurrent-depth baseline.
- RWKV, RetNet, SSMs, and RecurrentGemma are sequence-recurrent or hybrid sequence models, not depth-recurrent parameter-sharing baselines for this comparison.
