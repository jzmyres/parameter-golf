# Recurrent-Depth Baseline Replication Report

Generated for the 2026-05-24 baseline-review pass; last updated after the train_gpt-native 10-step DDP sweep.

> **Evidence provenance (read first).** The pass/fail counts and metrics below are a
> **snapshot of the 2026-05-24 run on 2× NVIDIA L40S GPUs**. `baselines/runs/` is
> gitignored and regenerable, so the authoritative, tracked evidence lives in
> [`runs_snapshot_2026-05-24/`](runs_snapshot_2026-05-24/README.md). Regenerating on a
> GPU-less host (e.g. a SLURM login node) yields `blocked_hardware`, which is expected.
> A 2026-05-30 login-node rerun already overwrote the consolidated
> `runs/ddp_smoke_report.json` and the `parcae-fixed-k16` native row; the surviving
> authoritative artifacts (full source smoke, three per-baseline DDP retries, five
> native rows) are in the snapshot. The full 12/13 adapter DDP run reproduces only on a
> 2-GPU node. See the `#committed-report-tracked-evidence` rule in `EXPERIENCE.md`.

## Scope Boundary

- **Comparable training result** means the row launched `train_gpt.py` itself under `torchrun --standalone --nproc_per_node=2` for 10 optimizer steps, using the repo data loader, optimizer, model shell, DDP path, and logging.
- **External source/DDP smoke** means pinned upstream code was fetched/imported, or a tiny adapter DDP loop ran. Those checks validate source availability and launchability only; they are not counted as comparable `train_gpt.py` training baselines.
- `PROFILE_SKIP_KSWEEP=1` was set for the native sweep so post-training roundtrip/K-sweep diagnostics did not dominate the smoke wallclock. The training loop and final fast validation still ran through `train_gpt.py`.
- Several SOTA recurrent/looped Transformer papers are relevant but not implemented inside `train_gpt.py`; they are listed below as paper baselines with blocked or proxy status rather than pass/fail comparable runs.

## Environment

| Item | State |
|---|---|
| Coordinator env | `opgbaselines` at `/project/ylin/mzhong4/conda/envs/opgbaselines` for fetch/source/DDP orchestration. |
| Native DDP runtime env | `opg` for `train_gpt.py`; PyTorch DDP on 2 visible NVIDIA L40S GPUs. |
| Native comparable sweep output | `baselines/runs/train_gpt_comparable_10step_report.json`. |
| Native comparable logs | `baselines/runs/train_gpt_comparable/<variant>/*.log`. |
| Source smoke output | `baselines/runs/smoke_report.json`. |
| External adapter DDP output | `baselines/runs/ddp_smoke_report.json` (pre-existing launchability smoke, not the native 10-step comparison). |

## Train GPT Comparable 10-Step Sweep

- Generated JSON: `baselines/runs/train_gpt_comparable_10step_report.json`.
- Result counts: `train_gpt_10step_passed`=6.
- All six train_gpt-native comparable variants reached at least 10 optimizer steps under 2-rank DDP.
- `val_bpb` below is the debug-profile fast validation after 10 steps. It is only a smoke comparability signal, not paper-scale replication or final promotion evidence.

| Variant | Status | Steps | Train Loss | Fast Val BPB | Duration | Peak VRAM | Baseline role | Log |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `active-revdeq-parcae` | `train_gpt_10step_passed` | 10/10 | 7.1107 -> 6.8809 | 3.9943 | 427.4s | 36845 MB | native-current: Current train_gpt.py comparator: reversible fixed-point loop, Parcae-style per-dimension damping/input injection, K-jitter, and recursive prefix-anchor consistency. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/active-revdeq-parcae/20260524T224105Z.log` |
| `parcae-fixed-k16` | `train_gpt_10step_passed` | 10/10 | 7.1080 -> 6.9012 | 3.7456 | 366.6s | 35620 MB | looped-fixed-depth: Pins the train-time recurrence depth to K=16 while keeping Parcae damping. This is the closest train_gpt.py-native proxy for fixed-depth looped/Universal-Transformer comparisons. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/parcae-fixed-k16/20260524T224826Z.log` |
| `parcae-no-recursive-anchors` | `train_gpt_10step_passed` | 10/10 | 7.0084 -> 7.6205 | 4.1133 | 309.5s | 34845 MB | parcae-ablation: Keeps the Parcae loop and stochastic K curriculum but removes the local recursive multi-K consistency target. This isolates the active repo's extra contraction/anchor mechanism from the Parcae-style baseline. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/parcae-no-recursive-anchors/20260524T225444Z.log` |
| `scalar-revdeq-fixed-k16` | `train_gpt_10step_passed` | 10/10 | 7.0090 -> 6.3349 | 3.4584 | 269.9s | 33501 MB | deq-family: Disables Parcae and uses the scalar DEQ beta path with a fixed K=16 solve. This is the closest local model to a classic shared-transition DEQ/loop baseline in train_gpt.py. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/scalar-revdeq-fixed-k16/20260524T230005Z.log` |
| `scalar-revdeq-kjitter` | `train_gpt_10step_passed` | 10/10 | 7.0089 -> 6.3181 | 3.4054 | 292.0s | 33501 MB | deq-family: Disables Parcae while preserving the active stochastic K curriculum. This separates learned damping from the recurrent-depth exposure schedule. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/scalar-revdeq-kjitter/20260524T230455Z.log` |
| `entmax-router-parcae` | `train_gpt_10step_passed` | 10/10 | 7.1133 -> 7.1563 | 3.9214 | 467.2s | 36893 MB | sparse-loop-proxy: Uses the implemented entmax routing path with linear scoring while keeping the Parcae loop. This is only a local sparse-routing proxy, not a paper-faithful MoEUT or Sparse Looped-MoE implementation. | `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/entmax-router-parcae/20260524T231000Z.log` |

## Native Proxy Mapping

| Paper/baseline family | train_gpt-native proxy status | Notes |
|---|---|---|
| Parameter Golf active RevDEQ + Parcae baseline | `active-revdeq-parcae` exact local row completed 10/10. | Current Hyperparameters comparator: 12-layer shared RevDEQ-style block, Parcae per-dim damping, K-jitter 16/24/32/64/96/128, recursive prefix anchors. |
| Parameter Golf 2026-03-25 ValCalib/GPTQ/XSA record baseline | No train_gpt implementation; manifest mode `blocked_no_training_code`. | Historical challenge comparator with a stronger BPB but different non-DEQ architecture and larger parameter count. |
| Deep Equilibrium Models | `active-revdeq-parcae`, `scalar-revdeq-fixed-k16`, `scalar-revdeq-kjitter` completed 10/10 as local proxy row(s). | Implicit-depth fixed point baseline with Transformer sequence examples; paper-faithful LM replication needs WikiText/PTB/One Billion Word setup and older training recipes. |
| TorchDEQ | `active-revdeq-parcae`, `scalar-revdeq-fixed-k16`, `scalar-revdeq-kjitter` completed 10/10 as local proxy row(s). | Modern DEQ library and DEQ Zoo reference; useful for solver/interface comparison more than direct Parameter Golf BPB replication. |
| Reversible Deep Equilibrium Models | `active-revdeq-parcae`, `scalar-revdeq-fixed-k16`, `scalar-revdeq-kjitter` completed 10/10 as local proxy row(s). | Closest conceptual baseline to the current reversible fixed-point training path; local read-only ARWDEQ reference also exists outside this repo. |
| Universal Transformer | `parcae-fixed-k16`, `scalar-revdeq-fixed-k16` completed 10/10 as local proxy row(s). | Depth-recurrent Transformer ancestor with optional adaptive halting; direct training replication is blocked by old TensorFlow/T2T stack unless isolated further. |
| MoEUT: Mixture-of-Experts Universal Transformers | `entmax-router-parcae` completed 10/10 as local proxy row(s). | Directly comparable sparse/shared-depth Transformer baseline: MoE capacity compensates for recurrent layer sharing. |
| Parcae: Scaling Laws For Stable Looped Language Models | `active-revdeq-parcae`, `parcae-fixed-k16`, `parcae-no-recursive-anchors` completed 10/10 as local proxy row(s). | Most directly aligned with the active Parcae-style damping/injection story; paper scripts target scaling-law sweeps, not 16MB FineWeb challenge constraints. |
| Ouro / Scaling Latent Reasoning via Looped Language Models | No train_gpt implementation; manifest mode `blocked_no_code`. | Public page advertises looped models and released weights, but code is not available as a cloneable training repository. |
| Reasoning with Latent Thoughts: On the Power of Looped Transformers | No train_gpt implementation; manifest mode `blocked_no_code`. | Important conceptual baseline for looped effective depth and latent reasoning; not a direct code replication target. |
| How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models | `active-revdeq-parcae`, `parcae-fixed-k16`, `parcae-no-recursive-anchors`, `scalar-revdeq-kjitter` completed 10/10 as local proxy row(s). | Scaling-law diagnostic baseline with recurrence-equivalence exponent phi; full 116-run sweep is compute-blocked locally. |
| Sparse Layers are Critical to Scaling Looped Language Models | `entmax-router-parcae` completed 10/10 as local proxy row(s). | Highly relevant to this repo's expert-routing direction: sparse MoE layers recover functional diversity across loop passes. |
| MoEUT training code | `entmax-router-parcae` completed 10/10 as local proxy row(s). | Official training-code companion for MoEUT; included because the module repo points here for training. |
| Huginn / Scaling up Test-Time Compute with Latent Reasoning | External code smoke only; no paper-faithful train_gpt implementation. | Official large-scale recurrent-depth training/inference code for Huginn-0125; full run used Frontier-scale compute. |
| Hyperloop Transformers | No train_gpt implementation; manifest mode `blocked_no_code`. | Hyper-connected looped middle-block Transformer; reports matching or outperforming depth-matched Transformer and mHC baselines with roughly 50 percent fewer parameters. |
| Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence | External code smoke only; no paper-faithful train_gpt implementation. | Retrofitting path for converting pretrained non-recurrent LMs into depth-recurrent models with recurrence curricula. |
| MoDR: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning | External code smoke only; no paper-faithful train_gpt implementation. | Dynamic multi-branch routing over Huginn-style recurrence using LoRA branches and hard-gate routing. |
| Mixture-of-Recursions | External code smoke only; no paper-faithful train_gpt implementation. | Adaptive token-level recursive depth; closely related to loop count allocation and dynamic compute. |
| Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models | External code smoke only; no paper-faithful train_gpt implementation. | Selective latent iterations only for hard tokens; relevant dynamic-depth comparator even though it is more inference-adaptive than parameter-sharing-first. |
| Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA | No train_gpt implementation; manifest mode `blocked_no_code`. | Layer-tying with depth-wise LoRA relaxations; useful prior for making repeated layers less rigid. |
| DND: Boosting Large Language Models with Dynamic Nested Depth | No train_gpt implementation; manifest mode `blocked_no_code`. | Routes critical tokens back through nested depth for an extra processing pass; included as adaptive-depth comparator. |
| Parallel Loop Transformer for Efficient Test-Time Computation Scaling | No train_gpt implementation; manifest mode `blocked_no_code`. | Targets loop latency by parallelizing across loop dimension and sharing first-loop KV cache. |
| Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization | No train_gpt implementation; manifest mode `blocked_no_code`. | Synthetic compositional reasoning baseline for stable 20+ step latent recurrence. |
| Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers | No train_gpt implementation; manifest mode `blocked_no_code`. | Controlled recurrent-depth Transformer study for systematic generalization and depth extrapolation. |
| A Mechanistic Analysis of Looped Reasoning Language Models | No train_gpt implementation; manifest mode `blocked_no_code`. | Analysis-only but important for understanding cyclic fixed points, input injection, and normalization in looped LMs. |

## External Source And Adapter Smoke

- Source smoke counts: `adapted_eval_run`=2, `blocked_no_code`=10, `smoke_passed`=13.
- External adapter DDP smoke counts: `blocked_no_code`=10, `blocked_no_training_code`=1, `blocked_non_pytorch_ddp`=2, `ddp_smoke_passed`=12.
- These checks are retained for reproducibility bookkeeping. They are not used as train_gpt-comparable 10-step baseline results.

| Baseline | Source status | Adapter/native smoke status | Evidence | Next action |
|---|---|---|---|---|
| Parameter Golf active RevDEQ + Parcae baseline | `adapted_eval_run` | `ddp_smoke_passed` | No fetchable pinned GitHub source in manifest. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/active-revdeq-parcae/ddp_smoke_20260524T220459Z.log` | Use native 10-step result above for train_gpt comparison; longer runs needed for quality claims. |
| Parameter Golf 2026-03-25 ValCalib/GPTQ/XSA record baseline | `adapted_eval_run` | `blocked_no_training_code` | No fetchable pinned GitHub source in manifest. blocked: historical one-file record snapshot has no maintained DDP training entrypoint | Keep as historical/reference comparator only. |
| Deep Equilibrium Models | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/deq-locuslab/ddp_smoke_20260524T220940Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| TorchDEQ | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/torchdeq/ddp_smoke_20260524T220228Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Reversible Deep Equilibrium Models | `smoke_passed` | `blocked_non_pytorch_ddp` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. blocked: RevDEQ is JAX/Equinox and declares Python >=3.12; PyTorch DDP does not apply | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Universal Transformer | `smoke_passed` | `blocked_non_pytorch_ddp` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. blocked: upstream training is TensorFlow/T2T, not PyTorch DDP | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| MoEUT: Mixture-of-Experts Universal Transformers | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/moeut/ddp_smoke_20260524T220234Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Parcae: Scaling Laws For Stable Looped Language Models | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/parcae/ddp_smoke_20260524T220301Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Ouro / Scaling Latent Reasoning via Looped Language Models | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no public cloneable training-code target found | Track paper; revisit if official code/checkpoints become public. |
| Reasoning with Latent Thoughts: On the Power of Looped Transformers | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official public code target found | Track paper; revisit if official code/checkpoints become public. |
| How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/iso-depth-looped-lm/ddp_smoke_20260524T220308Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Sparse Layers are Critical to Scaling Looped Language Models | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no public code target found | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| MoEUT training code | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/moeut-training-code/ddp_smoke_20260524T220314Z.log` | Use mapped train_gpt proxy above for smoke-level local comparison; do not claim paper-faithful replication. |
| Huginn / Scaling up Test-Time Compute with Latent Reasoning | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/huginn-recurrent-pretraining/ddp_smoke_20260524T221218Z.log` | Optional: build a paper-specific env and bounded upstream trainer; current adapter is only launchability evidence. |
| Hyperloop Transformers | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official public code target found | Track paper; revisit if official code/checkpoints become public. |
| Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/retrofitting-recurrence/ddp_smoke_20260524T220327Z.log` | Optional: build a paper-specific env and bounded upstream trainer; current adapter is only launchability evidence. |
| MoDR: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/modr/ddp_smoke_20260524T220333Z.log` | Optional: build a paper-specific env and bounded upstream trainer; current adapter is only launchability evidence. |
| Mixture-of-Recursions | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/mixture-of-recursions/ddp_smoke_20260524T220339Z.log` | Optional: build a paper-specific env and bounded upstream trainer; current adapter is only launchability evidence. |
| Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models | `smoke_passed` | `ddp_smoke_passed` | Exact pinned commit present; README files: README.md; Python syntax compilation passed. Log: `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/think-at-hard/ddp_smoke_20260524T220345Z.log` | Optional: build a paper-specific env and bounded upstream trainer; current adapter is only launchability evidence. |
| Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official standalone code target confirmed | Track paper; revisit if official code/checkpoints become public. |
| DND: Boosting Large Language Models with Dynamic Nested Depth | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official code target found | Track paper; revisit if official code/checkpoints become public. |
| Parallel Loop Transformer for Efficient Test-Time Computation Scaling | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official code target found | Track paper; revisit if official code/checkpoints become public. |
| Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official code target found | Track paper; revisit if official code/checkpoints become public. |
| Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official code target found | Track paper; revisit if official code/checkpoints become public. |
| A Mechanistic Analysis of Looped Reasoning Language Models | `blocked_no_code` | `blocked_no_code` | No fetchable pinned GitHub source in manifest. blocked: no official code target found | Track paper; revisit if official code/checkpoints become public. |

## Dedicated Paper Notes

### Parameter Golf active RevDEQ + Parcae baseline

- **Registry id:** `active-revdeq-parcae`
- **Paper/project:** local:opg_doc.tex
- **Code:** local:train_gpt.py @ `local`
- **Framework:** `pytorch`; license `repo LICENSE`
- **Mechanism/relevance:** Current Hyperparameters comparator: 12-layer shared RevDEQ-style block, Parcae per-dim damping, K-jitter 16/24/32/64/96/128, recursive prefix anchors.
- **Source replication status:** `adapted_eval_run`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step status:** exact/local row `active-revdeq-parcae` completed 10/10 steps; fast val BPB `3.9943`; log `/project/ylin/mzhong4/research/opg/parameter-golf/baselines/runs/train_gpt_comparable/active-revdeq-parcae/20260524T224105Z.log`.

### Parameter Golf 2026-03-25 ValCalib/GPTQ/XSA record baseline

- **Registry id:** `record-2026-03-25`
- **Paper/project:** local:records_baseline
- **Code:** local:records_baseline/record_2026-03-25.py @ `local`
- **Framework:** `pytorch`; license `repo LICENSE`
- **Mechanism/relevance:** Historical challenge comparator with a stronger BPB but different non-DEQ architecture and larger parameter count.
- **Source replication status:** `adapted_eval_run`
- **External DDP/adaptation status:** `blocked_no_training_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** source/reference exists but no maintained bounded DDP training entrypoint is available.

### Deep Equilibrium Models

- **Registry id:** `deq-locuslab`
- **Paper/project:** https://arxiv.org/abs/1909.01377
- **Code:** https://github.com/locuslab/deq.git @ `1fb7059d6d89bb26d16da80ab9489dcc73fc5472`
- **Framework:** `pytorch-legacy`; license `MIT-style repo license`
- **Mechanism/relevance:** Implicit-depth fixed point baseline with Transformer sequence examples; paper-faithful LM replication needs WikiText/PTB/One Billion Word setup and older training recipes.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `active-revdeq-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9943), `scalar-revdeq-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4584), `scalar-revdeq-kjitter` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4054).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### TorchDEQ

- **Registry id:** `torchdeq`
- **Paper/project:** https://arxiv.org/abs/2310.18605
- **Code:** https://github.com/locuslab/torchdeq.git @ `4f6bd5fa66dd991cad74fcc847c88061764cf8db`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Modern DEQ library and DEQ Zoo reference; useful for solver/interface comparison more than direct Parameter Golf BPB replication.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `active-revdeq-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9943), `scalar-revdeq-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4584), `scalar-revdeq-kjitter` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4054).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Reversible Deep Equilibrium Models

- **Registry id:** `revdeq`
- **Paper/project:** https://arxiv.org/abs/2509.12917
- **Code:** https://github.com/sammccallum/revdeq.git @ `c3ae05e47f30e92ea0eb41aa979e95c7c22a9939`
- **Framework:** `jax-equinox`; license `repo license`
- **Mechanism/relevance:** Closest conceptual baseline to the current reversible fixed-point training path; local read-only ARWDEQ reference also exists outside this repo.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `blocked_non_pytorch_ddp`
- **train_gpt-native 10-step proxy:** `active-revdeq-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9943), `scalar-revdeq-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4584), `scalar-revdeq-kjitter` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4054).
- **Replication blocker:** upstream stack is not PyTorch DDP, so the project DDP smoke requirement does not apply directly.

### Universal Transformer

- **Registry id:** `universal-transformer-t2t`
- **Paper/project:** https://arxiv.org/abs/1807.03819
- **Code:** https://github.com/tensorflow/tensor2tensor.git @ `bafdc1b67730430d38d6ab802cbd51f9d053ba2e`
- **Framework:** `tensorflow1-tensor2tensor`; license `Apache-2.0`
- **Mechanism/relevance:** Depth-recurrent Transformer ancestor with optional adaptive halting; direct training replication is blocked by old TensorFlow/T2T stack unless isolated further.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `blocked_non_pytorch_ddp`
- **train_gpt-native 10-step proxy:** `parcae-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.7456), `scalar-revdeq-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4584).
- **Replication blocker:** upstream stack is not PyTorch DDP, so the project DDP smoke requirement does not apply directly.

### MoEUT: Mixture-of-Experts Universal Transformers

- **Registry id:** `moeut`
- **Paper/project:** https://arxiv.org/abs/2405.16039
- **Code:** https://github.com/robertcsordas/moeut.git @ `87f04973d3db3ad7cb4ca67a0750241e40210d3c`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Directly comparable sparse/shared-depth Transformer baseline: MoE capacity compensates for recurrent layer sharing.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `entmax-router-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9214).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Parcae: Scaling Laws For Stable Looped Language Models

- **Registry id:** `parcae`
- **Paper/project:** https://arxiv.org/abs/2604.12946
- **Code:** https://github.com/sandyresearch/parcae.git @ `6e4519556e5a9d444793c56aa9fd9ac2965f097b`
- **Framework:** `pytorch-lightning`; license `repo license`
- **Mechanism/relevance:** Most directly aligned with the active Parcae-style damping/injection story; paper scripts target scaling-law sweeps, not 16MB FineWeb challenge constraints.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `active-revdeq-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9943), `parcae-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.7456), `parcae-no-recursive-anchors` (train_gpt_10step_passed, 10/10 steps, fast val BPB 4.1133).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Ouro / Scaling Latent Reasoning via Looped Language Models

- **Registry id:** `ouro-looplm`
- **Paper/project:** https://arxiv.org/abs/2510.25741
- **Code:** https://ouro-llm.github.io/ @ `none`
- **Framework:** `pytorch`; license `model cards list apache-2.0 for released weights`
- **Mechanism/relevance:** Public page advertises looped models and released weights, but code is not available as a cloneable training repository.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### Reasoning with Latent Thoughts: On the Power of Looped Transformers

- **Registry id:** `latent-thoughts-looped-transformer`
- **Paper/project:** https://arxiv.org/abs/2502.17416
- **Code:** none_found @ `none`
- **Framework:** `unknown`; license `n/a`
- **Mechanism/relevance:** Important conceptual baseline for looped effective depth and latent reasoning; not a direct code replication target.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models

- **Registry id:** `iso-depth-looped-lm`
- **Paper/project:** https://arxiv.org/abs/2604.21106
- **Code:** https://github.com/kschwethelm/looped-lm-scaling.git @ `ad7d8f3055501e4c84ced7797f772dce2f8e2107`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Scaling-law diagnostic baseline with recurrence-equivalence exponent phi; full 116-run sweep is compute-blocked locally.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `active-revdeq-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9943), `parcae-fixed-k16` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.7456), `parcae-no-recursive-anchors` (train_gpt_10step_passed, 10/10 steps, fast val BPB 4.1133), `scalar-revdeq-kjitter` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.4054).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Sparse Layers are Critical to Scaling Looped Language Models

- **Registry id:** `sparse-looped-moe`
- **Paper/project:** https://arxiv.org/abs/2605.09165
- **Code:** none_found @ `none`
- **Framework:** `unknown`; license `n/a`
- **Mechanism/relevance:** Highly relevant to this repo's expert-routing direction: sparse MoE layers recover functional diversity across loop passes.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** `entmax-router-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9214).
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### MoEUT training code

- **Registry id:** `moeut-training-code`
- **Paper/project:** https://arxiv.org/abs/2405.16039
- **Code:** https://github.com/RobertCsordas/moeut_training_code.git @ `521093a88007524b7ec9112be78769eb0d7c465f`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Official training-code companion for MoEUT; included because the module repo points here for training.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** `entmax-router-parcae` (train_gpt_10step_passed, 10/10 steps, fast val BPB 3.9214).
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Huginn / Scaling up Test-Time Compute with Latent Reasoning

- **Registry id:** `huginn-recurrent-pretraining`
- **Paper/project:** https://arxiv.org/abs/2502.05171
- **Code:** https://github.com/seal-rg/recurrent-pretraining.git @ `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Official large-scale recurrent-depth training/inference code for Huginn-0125; full run used Frontier-scale compute.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Hyperloop Transformers

- **Registry id:** `hyperloop-transformers`
- **Paper/project:** https://arxiv.org/abs/2604.21254
- **Code:** none_found @ `none`
- **Framework:** `pytorch-probable`; license `n/a`
- **Mechanism/relevance:** Hyper-connected looped middle-block Transformer; reports matching or outperforming depth-matched Transformer and mHC baselines with roughly 50 percent fewer parameters.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence

- **Registry id:** `retrofitting-recurrence`
- **Paper/project:** https://arxiv.org/abs/2511.07384
- **Code:** https://github.com/mcleish7/retrofitting-recurrence.git @ `e21042a6f4708b9f1ad2ca1f767948debe5a9289`
- **Framework:** `pytorch`; license `Apache-2.0`
- **Mechanism/relevance:** Retrofitting path for converting pretrained non-recurrent LMs into depth-recurrent models with recurrence curricula.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### MoDR: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning

- **Registry id:** `modr`
- **Paper/project:** https://openreview.net/pdf?id=9Pba4rcQbE
- **Code:** https://github.com/zhangxjohn/MoDr.git @ `4d77e558bcabfb551258daaacbee47ced3661987`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Dynamic multi-branch routing over Huginn-style recurrence using LoRA branches and hard-gate routing.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Mixture-of-Recursions

- **Registry id:** `mixture-of-recursions`
- **Paper/project:** https://arxiv.org/abs/2507.10524
- **Code:** https://github.com/raymin0223/mixture_of_recursions.git @ `53d0fee43632b53fb9bddd4acf9af7a2eba43bb6`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Adaptive token-level recursive depth; closely related to loop count allocation and dynamic compute.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models

- **Registry id:** `think-at-hard`
- **Paper/project:** https://arxiv.org/abs/2511.08577
- **Code:** https://github.com/thu-nics/TaH.git @ `4ba7145499d8c4635a3effa5c66bfafd99c2f5bd`
- **Framework:** `pytorch`; license `repo license`
- **Mechanism/relevance:** Selective latent iterations only for hard tokens; relevant dynamic-depth comparator even though it is more inference-adaptive than parameter-sharing-first.
- **Source replication status:** `smoke_passed`
- **External DDP/adaptation status:** `ddp_smoke_passed`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication caveat:** current upstream-code result is import-plus-tiny-DDP launchability; paper-faithful training still needs a dedicated env/data recipe.

### Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA

- **Registry id:** `relaxed-recursive-transformers`
- **Paper/project:** https://arxiv.org/abs/2410.20672
- **Code:** none_found @ `none`
- **Framework:** `pytorch-probable`; license `n/a`
- **Mechanism/relevance:** Layer-tying with depth-wise LoRA relaxations; useful prior for making repeated layers less rigid.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### DND: Boosting Large Language Models with Dynamic Nested Depth

- **Registry id:** `dynamic-nested-depth`
- **Paper/project:** https://arxiv.org/abs/2510.11001
- **Code:** none_found @ `none`
- **Framework:** `pytorch-probable`; license `n/a`
- **Mechanism/relevance:** Routes critical tokens back through nested depth for an extra processing pass; included as adaptive-depth comparator.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### Parallel Loop Transformer for Efficient Test-Time Computation Scaling

- **Registry id:** `parallel-loop-transformer`
- **Paper/project:** https://arxiv.org/abs/2510.24824
- **Code:** none_found @ `none`
- **Framework:** `pytorch-probable`; license `n/a`
- **Mechanism/relevance:** Targets loop latency by parallelizing across loop dimension and sharing first-loop KV cache.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization

- **Registry id:** `thinking-deeper-not-longer`
- **Paper/project:** https://arxiv.org/abs/2603.21676
- **Code:** none_found @ `none`
- **Framework:** `unknown`; license `n/a`
- **Mechanism/relevance:** Synthetic compositional reasoning baseline for stable 20+ step latent recurrence.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers

- **Registry id:** `loop-think-generalize`
- **Paper/project:** https://arxiv.org/abs/2604.07822
- **Code:** none_found @ `none`
- **Framework:** `unknown`; license `n/a`
- **Mechanism/relevance:** Controlled recurrent-depth Transformer study for systematic generalization and depth extrapolation.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

### A Mechanistic Analysis of Looped Reasoning Language Models

- **Registry id:** `mechanistic-looped-reasoning`
- **Paper/project:** https://arxiv.org/abs/2604.11791
- **Code:** none_found @ `none`
- **Framework:** `analysis`; license `n/a`
- **Mechanism/relevance:** Analysis-only but important for understanding cyclic fixed points, input injection, and normalization in looped LMs.
- **Source replication status:** `blocked_no_code`
- **External DDP/adaptation status:** `blocked_no_code`
- **train_gpt-native 10-step proxy:** none implemented; no comparable training run claimed.
- **Replication blocker:** no public cloneable training-code target was found during the 2026-05-24 search pass.

## Current Blockers

- Paper-faithful replication is compute/data blocked for large recurrent-depth scaling-law baselines on the local two-L40S setup.
- Hyperloop Transformers, Ouro, Sparse Looped-MoE, DND, Parallel Loop Transformer, and several reasoning-focused loop papers remain paper-only or no-public-code rows in this workspace.
- Non-PyTorch projects are not PyTorch DDP failures: RevDEQ is JAX/Equinox; Universal Transformer/T2T is TensorFlow 1.x/T2T.
- External adapter passes are intentionally shallow and should not be used for quality comparison against `train_gpt.py`.
