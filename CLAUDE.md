# Parameter Golf — Autoresearch Project

> **Before each iteration, READ `EXPERIENCE.md` §1 index** for the dated incident-driven rules. The §9 audit checklist below cites those anchors — consult them before adding new code of the same shape. New audit/development rules go into `EXPERIENCE.md` §1, not into this file (see §8).

## 0. Pre-Action Memory (READ BEFORE ANY ACTION)

> **MANDATORY**: Before making any code change, doc edit, training launch, or commit, read every memory file referenced below. The auto-memory `MEMORY.md` index is loaded with the system prompt, but individual files are NOT — read them with the Read tool. New memory files MUST be added to this section in the same commit that creates them, or they will not be read on future sessions.

**Memory directory**: `/home/mzhong4/.claude/projects/-project-ylin-mzhong4-research-opg-parameter-golf/memory/`

**User profile** — read first to frame interactions:
- `user_profile.md` — ML researcher, Parameter Golf, RevDEQ/MoE/MLA focus, L40S dev hardware

**Feedback (durable rules — overrides default behavior)** — read before any action:
- `feedback_hypotheses_sync.md` — sync hypotheses.md in real time
- `feedback_pre_commit_review.md` — /simplify + 3 reviews
- `feedback_simplify_before_commit.md` — /simplify each iter
- `feedback_dry_fixes.md` — fix all instances of a bug class
- `feedback_arch_exploration.md` — architecture > hyperparameter tuning
- `feedback_compile_training.md` — disable torch.compile for dev
- `feedback_always_ddp.md` — always all GPUs DDP
- `feedback_wakeup_cadence.md` — 5-min after failure → 20-min after 3 healthy
- `feedback_lipschitz_in_ksweep.md` — Lipschitz + acyclicity primes in K-sweep
- `feedback_per_wallclock_override.md` — val_bpb may yield to per-wallclock
- `feedback_sparsity_value_props.md` — sparsity: val_bpb / throughput / reg
- `feedback_uv_install.md` — `uv pip install`
- `feedback_check_gpu_free.md` — pgrep + nvidia-smi preflight
- `feedback_conda_run_buffering.md` — `conda run --no-capture-output`
- `feedback_throughput_priority.md` — throughput iters take priority
- `feedback_sdpa_replacement_at_T2048.md` — SDPA replacements regress at T=2048
- `feedback_diagnosis_context.md` — record full active config when closing
- `feedback_ntp_descent_rate_metric.md` — ntp descent rate is permanent H-claim metric
- `feedback_cumulative_vs_instantaneous_metrics.md` — cumulative averages lie about steady state; compute per-step deltas
- `feedback_routing_metric_axes.md` — load-balance ≠ sparsity; CV is balance, pertoken_entropy is sparsity
- `feedback_grad_enabled_vs_requires_grad.md` — dispatch on `is_grad_enabled() AND requires_grad`
- `feedback_profile_before_throughput.md` — chrome trace, not log fragments
- `feedback_decouple_regularizers.md` — antagonistic regs: one as metric
- `feedback_anneal_sparsity_coefs.md` — sparsity coefs anneal from 0
- `feedback_deq_convergence.md` — DEQ as fixed point
- `feedback_expert_collapse.md` — full-dim low-rank experts
- `feedback_mla_preferred.md` — MLA, not MHA/GQA
- `feedback_mos_routing.md` — MoS pure softmax
- `feedback_refinement_decoupled.md` — refinement separate forward pass

**Project (current state, lessons)** — read for context on running work:
- `project_autoresearch.md` — autoresearch setup + protocol
- `project_group_f_lessons.md` — H71-H75 architectural sparsity closed
- `project_experiment_results.md` — experiment log + insights
- `project_phase0_learnings.md` — Phase 0 takeaways
- `project_phase4_throughput_first.md` — throughput micro-opts first
- `project_deq_depth_insight.md` — DEQ effective depth = 1
- `project_architecture_ideas.md` — XSA, GPTQ-lite, Late QAT, DiffAttn, DeltaNet
- `project_fix_later.md` — deferred tech debt

**When adding a new memory file**: append it under the right section here AND to `MEMORY.md`. Both must be touched in the same commit, otherwise future sessions miss the directive.

## 1. Project at a Glance

OpenAI Parameter Golf challenge (March 18 – April 30, 2026; $1M OpenAI compute prize): train the best small LM that fits in a **16 MB artifact** (code + compressed model), trains in **≤10 minutes** on 8×H100 SXM GPUs, evaluated by **val_bpb** (bits-per-byte) on the FineWeb validation set. Lower is better.

**Hard constraints** (enforced):
- Artifact ≤ 16,000,000 bytes (code + compressed model).
- Submission training ≤ 600 s on 8×H100 SXM (must opt in via `--max-wallclock-seconds=600` — never the default).
- FineWeb validation set; SentencePiece BPE tokenizer, vocab = 1024.

**Baseline pointer.** This repo's working baseline (true int6, dev hardware) is tracked in `experiments/hypotheses.md` (latest promoted iter row). Update that line in the same commit that promotes a new baseline.

**Leaderboard SOTA reference (not our baseline).** val_bpb = 1.1194 (abaybektursun, 2026-03-23). Techniques: LeakyReLU(0.5)², TTT, Parallel Muon, XSA, GPTQ-lite, EMA. Our autoresearch baseline is independent.

## 2. Environment & Files

- **Conda env**: `conda activate opg` (must be active before any command).
- **Python deps**: `requirements.txt`. **Do not install new packages by default.** When the user explicitly authorizes a new package install, use `uv` (NOT `pip`): `uv pip install <pkg>` (user directive 2026-04-28). Update `requirements.txt` in the same commit as the package usage.
- **Data (READ-ONLY)**: `./data/datasets/fineweb10B_sp1024/`; tokenizer at `./data/tokenizers/fineweb_1024_bpe.model`.

**Key files.**
- `train_gpt.py` — base training script; the only model file the agent modifies.
- `EXPERIENCE.md` — incident archive (§1) and lessons-learned (§2). Cited from §9 audit checklist.
- `experiments/hypotheses.md` — hypothesis log; read before each iter, update after.
- `experiments/update_results.sh` — log rotation + plot regeneration; run after every iteration.
- `results.tsv` — experiment log (untracked by git).
- `run.log` — latest training console capture (untracked).
- `records/` — historical leaderboard submissions (READ-ONLY).
- `opg_doc.tex` — paper draft; co-update on architectural changes (see §9 Doc-Code Invariant).

**Reference implementations (READ-ONLY).**
- **RevDEQ**: `/home/mzhong4/work/research/rdeq/WIP-ARWDEQ/code/arwdeq/qwen3_utmoe_revdeq.py` — RevDEQ solver with custom autograd, fp64 accumulators, Kahan compensation; gated low-rank input injection `z_hat = z + α·B(A(LN(x)))`; warm start `y0 = z0 = x`.
- **TSU (CTP/NTP/MoS)**: `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py` — `MoSLowRankOutputHead` (frozen expert + trainable low-rank experts, dual B matrices); `_derive_ctp_labels`; `sample_dirichlet_lowrank_topk`; `TSUEmbedding`.

## 3. Agent Non-Negotiables

- **Modify only**: `train_gpt.py`, focused tests, project docs (incl. `EXPERIENCE.md`, `experiments/hypotheses.md`). Wider scope requires explicit user approval.
- **Never modify**: `data/`, tokenizer code, evaluation harness code, `records/`, dependency files, package manifests.
- **No new packages by default.** Authorized installs use `uv pip install <pkg>` (NOT `pip install`). User directive 2026-04-28.
- **DDP-safe**: all code MUST work with both single-GPU and multi-GPU `torchrun` (dev: 1–2× L40S; full: 8× H100). No GPU-count-specific logic without proper `world_size` handling.
- **Submission runs honor 600 s wallclock + 16 MB artifact** (see §1).
- **Expert health is final-only**, computed on **normalized per-component** expert shares; total routed mass is tracked separately and must not be conflated with balance.

## 4. How to Run

**Default (1000 iters, no wallclock cap, all visible GPUs)** — this is the canonical step-count-governed dev run:
```bash
conda activate opg
torchrun --standalone --nproc_per_node=gpu train_gpt.py
```
`--nproc_per_node=gpu` auto-detects all visible GPUs. Wallclock cap is OFF unless explicitly set. To run shorter: `--iterations=N`.

**Submission run (8× H100 SXM, 600 s competition cap)** — must opt in:
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --max-wallclock-seconds=600
```

**Explicit GPU count** (if `gpu` alias is unsupported):
```bash
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

**Single-GPU debug only**: `python train_gpt.py` (works but does not scale).

**Evaluate**: `grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log`.

**Fair-comparison principle.** Compare at equal **step count** (not wall-clock); the default 1000-iter run enforces this. Wall-clock only matters for the submission.

## 5. Current Architecture

Single source of truth: `train_gpt.py::Hyperparameters`. The tables below MUST match `Hyperparameters` defaults; edits to either land in the same commit (see §9 Single-source-of-truth).

### Architecture
| Parameter | Value |
|---|---|
| num_layers | 12 |
| model_dim | 768 (H73) |
| num_heads | 8 |
| num_kv_heads | 4 |
| num_experts | 16 (H72) |
| num_shared_experts | 1 (DeepSeek shared expert, always-on with sigmoid gate) |
| mlp_mult | 3.0 (hidden = D × 3 / num_experts via low-rank experts) |
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| grad_accum_multiplier | 1 (>1 halves per-step activation memory at constant effective batch; no LR/WD rescaling) |
| vocab_size | 1024 |
| tie_embeddings | yes |
| deq_beta | 0.50 (fallback when `use_parcae=False`) |
| use_parcae | True (per-dim Ā and B̄; supersedes scalar β/jitter when active) |
| parcae_init_a_bar | 0.7 (Ā₀; β₀ = 1−Ā₀ = 0.3) |
| parcae_init_b_bar | 0.3 (B̄₀ ≈ 1−Ā₀ at step 0) |
| parcae_reversibility_floor | 0.1 (correctness constant — Ā ≥ bound for RevDEQ backward; NOT a tuning knob — see §10) |
| deq_bptt_k | 3 (iter 95 2026-05-02: TBPTT=2→3 to add ~50% backward gradient depth under WD=0.01 + K-jitter (16,24) regime; triggered by grad_norm 0.07 headroom obs in iter 112+122) |
| num_refinements | 0 (default off; iter 110 queued as clean re-enable ablation; §6.5 still defines architectural form) |
| use_ctp | False (NTP-only; CTP param banks not allocated; H60) |
| use_nsa_attention | False (CLI-enable. When True: 2-branch NSA mixer (compression block-pool + sliding-window). Selection branch deferred to iter 106b. H86.) |
| nsa_compress_block_size | 32 |
| nsa_compress_block_sliding_stride | 16 |
| nsa_selection_block_size | 64 (iter 106b knob) |
| nsa_num_selected_blocks | 0 (0 disables selection branch) |
| nsa_sliding_window_size | 256 |
| nsa_branch_gate_init | 0.0 (uniform softmax at init) |

### Optimizer
| Parameter | Value |
|---|---|
| matrix_lr | 0.022 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| embed_lr | 0.6 |
| parcae_lr | 0.002 (applied to `parcae_raw_a`, `parcae_raw_delta`, `parcae_raw_b`) |
| entmax_blend_lr | 0.002 (H87b) |
| muon_momentum | 0.99 |
| muon_backend_steps | 7 (PE-NS empirical elbow; see `EXPERIENCE.md#variance-reg-ns-cascade`) |
| use_polar_express_ns | True (default ON; `--use-polar-express-ns=0` reverts to stock NS for A/B) |
| muon_momentum_warmup_start | 0.92 |
| muon_momentum_warmup_steps | 800 |
| weight_decay | 0.01 (applied to both AdamW and Muon groups) |
| grad_clip_norm | 1.0 |
| warmdown_frac | 0.72 |

### Routing & Expert Ranks
| Parameter | Value |
|---|---|
| router_scoring | linear (dot-product logits) |
| mos_balance_mult | 50.0 (multiplies MoS-NTP balance loss in `_collect_routing_losses`; the principled fix for `mos_*_min_share` failures, H26) |
| min_share_loss_weight | 0.0 (CV loss alone covers global balance; `min_share` stays a diagnostic — sentinel `< 0.005` sustained → intervene; H76) |
| cv_loss_weight | 2.0 (compensates dropped min_share floor; H76) |
| router_entropy_coef | 0.005 (iter 117 v5 baseline; 10× bump in 117b-1 NOT PROMOTED — H87b) |
| router_entropy_warmup_delay_frac | 0.3 (linear ramp 0→target after 30 % of wallclock; cold-start trap mitigation per `feedback_anneal_sparsity_coefs.md`) |
| use_entmax_routing | False (CLI-enable. When True: `sigmoid(blend) * softmax + (1−sigmoid(blend)) * entmax_1p5` with learnable scalar `blend_logit`. Strict-gen at `entmax_blend_init_logit=+5`. H87.) |
| entmax_blend_init_logit | 5.0 (sigmoid(5) ≈ 0.9933 ⇒ ≈ pure softmax at init; CLI override `--entmax-blend-init-logit=…`) |
| use_orthogonal_expansion_routing | False (CLI-enable. Adds Gram-matrix penalty `λ · ‖G − I/E‖²_F` over the routing-weight tensor. Strict-gen at `routing_gram_coef=0`. H84.) |
| routing_gram_coef | 0.01 (effective when `use_orthogonal_expansion_routing=True`; annealed 0 → target) |
| routing_gram_warmup_delay_frac | 0.3 (same shape as `router_entropy_warmup_delay_frac`) |
| logit_softcap | 0.0 (CLI-enable. When > 0, applies `softcap·tanh(logits/softcap)` to per-expert MoS logits before log_softmax. Strict-gen at 0. Records use 30.0; H93.) |
| attn_expert_rank | 64 |
| mlp_expert_rank | 96 |
| bigram_vocab_size | 0 (BigramHash disabled; H64) |
| bigram_dim | 128 |
| deq_beta_jitter | True (sample β from {0.3, 0.5, 0.7} per step when `use_parcae=False`) |
| deq_k_jitter_set | (16, 24) (FP found at K=16 per iter 98b K-sweep; K=24 adds wider FP-depth jitter — analog of H12 VERIFIED. Iter 131 tried (32,48) — NOT-PROMOTED 2026-05-03 due to +62.6% wallclock cost without val_bpb improvement) |
| lyapunov_coef | 0.0 (λ_jac disabled — Parcae per-dim Ā already bounds spectral radius) |
| lyapunov_gamma | 0.97 |
| lyapunov_warmup_frac | 0.05 |
| denoising_coef | 0.0 (HyDRA denoising disabled — Parcae-redundancy logic) |
| denoising_noise_std | 0.01 |

### Quantization & Techniques
- int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings
- BigramHash(4096+) + OrthoInit
- Sliding-window eval (stride = 64)
- SWA disabled — see `EXPERIENCE.md#disabled-techniques` for the reason trail.

## 6. Architectural Invariants

These define the model class. Each rule states *what* the architecture must be; rationale is in EXPERIENCE.md §2 (Lessons Learned) where applicable.

### 6.1 RevDEQ — Reversible Deep Equilibrium Model
Paper: arxiv:2509.12917. Reference impl: see §2.

- Output = fixed point of a learned function. **Two decoupled loops**:
  - **DEQ solver** (`num_layers` coupled-state iters): `y_{n+1} = (1−β)·y_n + β·f(z_n, x₀)`; `z_{n+1} = (1−β)·z_n + β·f(y_{n+1}, x₀)`.
  - **Refinement** (`num_refinements`): predict → soft_embed → re-solve. Total block calls = `(1 + num_refinements) × num_layers × 2`.
- **Warm start**: `z₀ = x` (one-hot token embedding); on refinement, `z₀ = x0_refined`.
- **fp64 accumulators** for add/sub — required for exact reversibility.
- **`deq_recon_err` under TBPTT is "distance travelled", NOT decision-grade.** Under default `deq_bptt_k < num_layers` the metric reports forward-FP travel, not reconstruction error. Do not gate divergence / promotion on it. Set `deq_bptt_k = 0` for true reconstruction (~1e-12 floor). See `EXPERIENCE.md#deq-recon-err-interpretation`.
- Fixed-point behavior is desired: monitor `||z_T − z_{T−1}|| / ||z_T||` (i.e. `deq_iter_conv_rel`, which IS informative under TBPTT) and residuals; improve if it does not harm expert health or val_bpb.
- Smoke test asserts: recon near precision *(only valid when smoke runs full BPTT)* · loss decreasing · convergence not exploding · no NaN/Inf · expert routing healthy.

### 6.2 Soft Dense Routing (Dense MoE on ALL components)
Paper: Soft MoE (arxiv:2308.00951). Mixtape (NeurIPS 2019) for MoS softmax.

- ALL experts process ALL tokens — no top-k, no token dropping. **HARD architectural constraint** — discrete-decision dispatch (top-K gather, threshold skip, capacity drop, argmax routing) breaks RevDEQ reversibility. See H101.
- **Router routes on component INPUT** (pre-computation), consistent across all components.
- **Full-dim low-rank experts**: every expert operates on the FULL hidden dim. Use low-rank matrices (`dim → rank → dim`); do NOT partition dimensions across experts.
- **Expert independence (HARD CONSTRAINT)**: every expert is fully independent — **zero shared trainable parameters** within the expert computation path. All projections (Q/K/V/Wo/gate/fc/down/MoS A-bank), all learned norms (RMSNorm scales) are per-expert (shape `(E, ...)`). Only the **router** itself and **non-learned ops** (RoPE tables, activations) may be shared. Adding a new param to the expert path: shape MUST start with `E`. See §9 "Prenorm scale independence (HARD)".
- **Attn/MLP routing**: `softmax(allocation) × sigmoid(gate)` (`SoftDenseRouter`). Weights sum to ≤ 1 (NOT renormalized). The sigmoid gate lets the model globally suppress the mixture (`T(z, x₀) ≈ 0` when all paths close).
- **MoS routing**: pure softmax (convex combination summing to 1).
- **Applied to**: attention output, MLP hidden, MoS output heads.
- **Expert-health metrics** (min-share / CV) are computed on **renormalized** per-component shares; total routed mass is logged separately.
- **Two distinct routing entropies** — *global utilization* (`H_global` over batch-averaged shares; HIGH = no dead experts; sentinel) and *per-token concentration* (`H_pertoken` averaged over tokens; LOW = specialization). Target: HIGH global AND LOW per-token. Full definitions, axes, and failure modes in §7 metrics table.
- **Regularization** — three orthogonal regs grouped under `router_reg_loss` in `_collect_routing_losses`; coefs are independent (do not collapse magnitudes in a single commit):
  - **Per-token specialization**: `+entropy_coef · H_pertoken` (POSITIVE sign drives `H_pertoken → 0`).
  - **Global balance / dead-expert prevention**: `min_share_loss_weight` (floor) + `cv_loss_weight` (CV penalty). Drives `H_global → log(N)`.
  - **Expert orthogonality**: `block_ortho_aux_coef × ‖cos_sim‖` between expert OUTPUT means.
- Fully differentiable, no discrete decisions.

### 6.3 Per-Expert MLA + Gated Attention (full-D LoRA-style — architectural standard)
Papers: DeepSeek-V2 MLA (arxiv:2405.04434); Gated Attention (arxiv:2505.06708, NeurIPS 2025 Best Paper).

**Architectural standard**: full-D LoRA-style — every expert linear is rank-`R` factored (`D → R → H·d_head` etc.) but **all activations and SDPA run at full `model_dim`** with `d_head` in the FlashAttention tensorcore sweet spot (64+). The rank `R` constrains *parameter count per expert*, not the *attention compute width*.

- Each expert has its own complete MLA pipeline (no shared params — see §6.2):
  - Per-expert Q: `dim → expert_rank → H·d_head + H` (gate logits appended).
  - Per-expert KV compression: `dim → kv_rank → kv_latent_dim`.
  - Per-expert KV decompression: per-expert RMSNorm + per-expert `kv_latent → K_nope, V`.
  - Per-expert K_rope: `dim → kr_rank → H_kv·rope_dim`.
  - Per-expert Wo: `D → wo_rank → D` (mixes heads per expert).
- **Head-packed SDPA**: expert index extends head dimension (`E·H` query heads, `E·H_kv` KV heads) for one FlashAttention call. GQA ratio preserved.
- **Decoupled RoPE**: split heads into RoPE and non-RoPE components.
- **Gated Attention**: query-dependent per-expert-per-head sigmoid gate after SDPA. Gate logits from per-expert Q projection (appended to Q output); each token gets its own gate value per head per expert.

**Optional sparse-attention path — NSA (iter 106, default off).** When `use_nsa_attention=True`, head-packed SDPA is replaced by a 2-branch Native Sparse Attention mixer (arxiv:2502.11089): compression (block-pool K/V + rectangular causal mask) + sliding-window (last W tokens, band causal). Per-expert-per-head softmax gate `nsa_branch_gate` shape `(E·H, 2)` mixes branches. Gated attention preserved. Selection branch (top-K per-query) deferred via `nsa_num_selected_blocks=0`. Strict-gen at `block_size=stride=1, window=T, gate_init=0` recovers SDPA within bf16 floor.

**Discarded alternative — bottleneck experts.** Do NOT re-introduce bottleneck-style experts (BottleneckIn/Out around a small `r`) as a scaling axis. Per-param efficiency and SDPA throughput both regress vs full-D LoRA. Full rationale + the archival reference: `EXPERIENCE.md#bottleneck-experts-closed`.

### 6.4 FSQ in MoS Head
Paper: FSQ (arxiv:2309.15505).

- FSQ via STE in an intermediate projection space inside the MoS output head.
- Rank flexible (tune for 16 MB budget). With V = 1024, even full-rank projections are affordable.
- **Currently disabled** (iter 62, H53): `fsq_levels = 0` bypasses FSQ; the low-rank MoS projection alone is sufficient. Code retained for re-enabling.

### 6.5 Diffusion-AR (Autoregressive + Iterative Refinement)
Reference impl: see §2.

- **Refinement loop** (decoupled from DEQ solver):
  - Step 0: DEQ solve with `x₀ = tok_emb(input_ids)` (clean one-hot).
  - Step 1+: predict from `z*` → build soft embedding → DEQ solve with `x₀ + soft_embed`.
  - Soft embedding: average CTP[i] and NTP[i−1] logits, top-k sparse embed, EMA blending.
  - `num_refinements` controls predict→refine cycles.
- **NTP-only MoS prediction** (iter 94 baseline): `Hyperparameters.use_ctp = False`. CTP param banks not allocated; `MoSHead.forward` returns `(log_p_ntp, log_p_ntp)`; `GPT.forward` sets `ctp_loss = 0`; `_get_soft_embedding` uses `p_mix = p_ntp`. Historical dual-head design preserved behind `--use-ctp=1`. Iter-94 metrics, removed param banks, and rationale: see H60 in `experiments/hypotheses.md`.
- NTP MoS head: 2 shared + 1 specialized expert, xavier init, rank = 256.

## 7. Autoresearch Protocol

### Setup
1. Branch `autoresearch/<tag>` from `main`.
2. Read `train_gpt.py`, `results.tsv`, `git log` for full context.
3. Verify `./data/datasets/fineweb10B_sp1024/` exists.
4. Initialize `results.tsv` with header.
5. Run iter 0 (converged baseline) → establish baseline val_bpb.

### Experiment Loop
1. Read git state: `git log --oneline -20` + `results.tsv`.
2. **Read `experiments/hypotheses.md`** — review relevant hypotheses, state what this iter tests.
3. Make ONE focused change to `train_gpt.py`.
4. Run `python experiments/smoke_test.py` — must pass before long training (loss decreases · DEQ recon < 1.0 relative · `||z_T − z_{T−1}||` not diverging · no NaN/Inf grads).
5. Write/update tests (TDD — tests BEFORE implementation).
6. `git commit` the change (after §8 pre-commit chain + §9 audit).
7. Run training; redirect output to `run.log`.
8. Read results: `grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log`. If empty → crash; `tail -n 50 run.log`, attempt fix.
9. Log to `results.tsv` (do NOT commit `results.tsv`).
10. **Always** run `bash experiments/update_results.sh` (rotates `current.log`/`current/weights` → `previous`, copies `run.log` → `current.log`, regenerates plots).
11. Apply §11 Promotion Rules.
12. **Update `experiments/hypotheses.md`** — record results, statuses, confounds. The H-claim section MUST include: (a) roundtrip int6 val_bpb + val_loss; (b) the FULL `k_sweep_table:` matrix verbatim from run.log (header + one row per K; 15 cols including acyclicity primes 17/37/113 in bold) — non-optional, no exceptions; (c) trajectory table (val_bpb per val checkpoint with Δ); (d) acyclicity-prime check (Δ vs nearest power-of-2 in 0.01-0.02 = genuine FP); (e) **ntp_loss descent rate** (per-step Δntp / 10 steps over windows s30-s100, s100-s200, s200-s400, s400-s600, s600-s800, s800-s1000) AND **per-wallclock equivalent** (Δntp/sec = Δntp/step ÷ step_avg) — both reported alongside baseline comparison. Permanent metric per user directive 2026-05-02. All numbers must be grep-able from run.log. See `feedback_hypotheses_sync.md`.
13. Track consecutive non-improvements. **STOP after 100** and seek user guidance.

### Logging, Weights & Plotting (every iteration)
- **Training logs**: `experiments/training_logs/{baseline,previous,current}.log`.
- **Model weights**: `experiments/weights/{baseline,previous,current}/`.
- **Metrics comparison**: `python experiments/plot_metrics.py` → `experiments/metrics_comparison.png` (4×3 grid: train_loss · val_bpb · step_avg_ms / DEQ residual · recon_err · iter_conv / expert_usage · entropy · ortho / summary text).
- **Progress plots**: `python experiments/plot_progress.py` → `experiments/progress.png`, `progress_full.png`.

#### Required routing-health metrics (ALL must be reported every train+val log line)

| Metric | Target | Decomposition |
|---|---|---|
| **`pertoken_entropy`** | **LOW** ≈ 1.0 nat (specialization) | single pool-level value |
| **`attn_entropy` / `mlp_entropy` / `pool_entropy`** | **HIGH** ≈ log(N_routed) per slice; pool ≈ log(2·N_routed) | per-slice renormalized + full pool |
| **`attn_min_share` / `mlp_min_share`** | **≥ 0.005** (dead-expert sentinel) | per slice |
| **`attn_cv` / `mlp_cv` / `pool_cv`** | **LOW** ≈ 0.2-0.3 per slice | per slice + full pool (cross-slice dominance) |
| **`attn_ortho` / `mlp_ortho`** | **LOW** ≈ 0.1-0.2 (`max\|cos_sim\|` of expert OUTPUT means) | per slice |
| **`router_mass`** | 0.7-0.95 (mean `sigmoid(gate)`) | single value |
| **`hutch_F`** (FP spectral) | LOW + decreasing across val | single value at FP; skipped on OOM/SDPA-grad-reject |

Definitions, axis interpretations, the iter-99 / iter-100b decomposition rationale, and the `k_sweep_table:` 14-column schema (incl. legacy `k_sweep:` line for plot back-compat) live in `EXPERIENCE.md#routing-health-metrics`.

- Detailed comparison: 2 configs only — baseline vs current.
- **Prioritize architecture exploration** over hyperparameter tuning; cite papers/repos.

### Time Budget
- ≤ 10 min wall-clock per experiment. If a run exceeds 15 min, kill and treat as failure. ~6 experiments/hour on dev hardware.

### `results.tsv` format (tab-separated)
```
commit	val_bpb	artifact_bytes	status	description
a1b2c3d	1.142760	15900000	keep	baseline converged config
b2c3d4e	1.140000	15800000	keep	added RevDEQ fixed-point layer
c3d4e5f	1.150000	16100000	discard	MLA attention (artifact too large)
d4e5f6g	0.000000	0	crash	soft routing OOM
```

## 8. Development Practices

### TDD
Tests before implementation for each architectural change. Test categories: shape · gradient (especially through RevDEQ fixed-point) · quantization roundtrip (int6 + zstd) · DDP correctness · artifact size ≤ 16 MB.

### Pre-Commit Review Chain
Before EVERY commit:
1. `/simplify`
2. `coderabbit:review`
3. `pr-review-toolkit:review-pr`
4. `superpowers:requesting-code-review`
5. Run §9 audit checklist.
6. Apply simple, first-principled fixes.
7. `git commit`.

### Hypothesis Log (`experiments/hypotheses.md`)
- **READ BEFORE each iteration** — review relevant hypotheses; check for known fixes.
- **UPDATE AFTER each iteration** — record verdict, update statuses, note confounds.
- Design experiments to test ONE hypothesis with a single controlled variable.
- Statuses: VERIFIED (controlled test), OBSERVED (confounded evidence), PROPOSED (untested), REFUTED (controlled disproof).
- Use as troubleshooting manual when DEQ diverges, routing collapses, or K-sweep degrades (e.g. H9: double WD when β is too high).
- Stability over task performance — prefer verified-stable over slightly-better-but-unverified.

### Simplicity Criterion
All else equal, simpler wins. A 0.001 bpb improvement adding 20 lines of hacky code is probably not worth it. Removing code with equal results is a keep. Favor removing complexity over adding it.

### Where new audit/development rules go
New incident-driven rules go into **`EXPERIENCE.md` §1**, not into this file. Procedure:
1. Add `### <slug>` section to `EXPERIENCE.md` §1 using the section template at the top of that file.
2. Add one row to the §9 audit checklist: `- **<rule>** — <one-sentence summary + grep cmd if any> → \`EXPERIENCE.md#<slug>\``.
3. Update the index table at the top of `EXPERIENCE.md` §1.

### Where new memory files go
New auto-memory files (feedback / project / user / reference) go in `/home/mzhong4/.claude/projects/-project-ylin-mzhong4-research-opg-parameter-golf/memory/`. Procedure (all four touches in the same commit, otherwise future sessions miss the file):
1. Write the file with the standard frontmatter (name / description / type).
2. Append a one-line entry to `MEMORY.md` (the index loaded into the system prompt).
3. Append a one-line entry to **CLAUDE.md §0 Pre-Action Memory** under the right category (User / Feedback / Project) so future sessions actively read it.
4. If the new file supersedes or modifies an existing rule, update the affected file in place rather than creating a duplicate.

## 9. Code-Quality Audit Checklist

Run before every commit that touches `train_gpt.py` OR `CLAUDE.md`. Each row is one-line enforcement; click the anchor for the full incident.

> **META-PRINCIPLE — Read the definition before the value; compute from raw when in doubt; treat user pushback as a clue you missed something.** → [`EXPERIENCE.md#reading-derived-metrics`](EXPERIENCE.md#reading-derived-metrics)

- **Single-source-of-truth (Hyperparameters)** — every tunable knob lives in `train_gpt.py::Hyperparameters` and is plumbed from there. Tests assert against `args.<field>`, not literals. CLAUDE.md §5 mirrors edits in the same commit. → [`EXPERIENCE.md#config-drift`](EXPERIENCE.md#config-drift)
- **Permutation consistency** — `grep -n 'permute(' train_gpt.py | grep -v '#'`; related groups (e.g. `(E,B,T,H,d) → (B,E·H,T,d)`) must use the same index tuple. A single outlier is almost certainly a silent transposition. → [`EXPERIENCE.md#permutation-consistency`](EXPERIENCE.md#permutation-consistency)
- **Optimizer coverage** — every `requires_grad` parameter must land in exactly one optimizer group (asserted at construction time via `_assert_optimizer_param_coverage`). Manual flatten/reshape of a parameter bank also needs an einsum-equivalence test (shape-only tests miss transposed-storage bugs).
- **Dead-code audit** — when removing a feature (e.g. `gg_gate`, `SmearGate`), remove ALL paths/tests/doc refs/tracking infrastructure in the same commit; `grep -rn '<removed_name>' .` returns zero. → [`EXPERIENCE.md#dead-code-tracking`](EXPERIENCE.md#dead-code-tracking)
- **Module-alias audit** — when merging N modules to a shared instance, verify `named_parameters()` filters match the canonical name and per-instance loops use `id()` dedup. → [`EXPERIENCE.md#router-alias`](EXPERIENCE.md#router-alias)
- **Hot-path sync prohibition** — `grep -n '\.item()\|\.cpu()' train_gpt.py`; ZERO hits between the `for micro_step` loop and `train_loss /= grad_accum_steps`. Comments saying "no .item()" don't count — automated grep is the source of truth. → [`EXPERIENCE.md#hot-path-sync`](EXPERIENCE.md#hot-path-sync)
- **Explicit boundary (incl. compile-wrapper writes)** — wrapper / device / optional-telemetry crossings get an explicit unwrap (use `_unwrap_compiled_module()` once per scope for `base_model.shared_block` writes), `?` fallback in shell summaries under `set -euo pipefail`, AND a contract test. GPU-stay covered by Hot-path sync row above. → [`EXPERIENCE.md#explicit-boundary`](EXPERIENCE.md#explicit-boundary)
- **Custom autograd inputs** — every grad-needing tensor is an explicit `apply(...)` arg with a matching `backward(...)` slot. Use `.clone().requires_grad_()` (NOT `.detach()`) for re-instantiated leaves. Optional inputs go on `ctx`, not `save_for_backward`. → [`EXPERIENCE.md#custom-autograd-input`](EXPERIENCE.md#custom-autograd-input)
- **Routing-predicate migration** — when changing any predicate that classifies params/modules (optimizer groups, quantization tiers, `CONTROL_TENSOR_PATTERNS`, `FP16_KEEP_PATTERNS`, EMA keys, gradient hooks, diagnostic buckets), enumerate before/after in the commit message + add a contract test. → [`EXPERIENCE.md#routing-predicate-migration`](EXPERIENCE.md#routing-predicate-migration)
- **Identifier uniqueness across wrappers** — no name may be both a method and an attribute on sibling classes in the same call graph. `grep -n '\.<new_name>\b' train_gpt.py tests/ experiments/` before adding. → [`EXPERIENCE.md#identifier-uniqueness`](EXPERIENCE.md#identifier-uniqueness)
- **Prenorm scale independence (HARD)** — `grep -n '_norm_weight' train_gpt.py`; every learned scale conditions exactly one linear weight. Shape follows the linear (E-prefixed for per-expert; bare D for shared linears that route to experts but aren't themselves per-expert). → [`EXPERIENCE.md#prenorm-scale-independence`](EXPERIENCE.md#prenorm-scale-independence)
- **Doc-Code Invariant** — when `opg_doc.tex` describes an algorithm and `train_gpt.py` implements a different (better) variant, the doc MUST note the deviation in a "Practical implementation" paragraph. Pseudocode is theoretical; code is the source of truth. → [`EXPERIENCE.md#doc-code-invariant`](EXPERIENCE.md#doc-code-invariant)
- **Diagnostic-gate component awareness** — flag-gated code paths must gate diagnostic emission AND retry prescriptions on the same flag (component-specific levers, e.g. `mos_balance_mult` for MoS collapse, NOT global `weight_decay`). → [`EXPERIENCE.md#diagnostic-gate-component-awareness`](EXPERIENCE.md#diagnostic-gate-component-awareness)
- **Hyperparameter fan-out** — every knob lives in `Hyperparameters`, reachable via `_parse_cli_overrides`, consumer reads `args.<field>` (no shadowing literal). Four-touch rule for new knobs: (1) `Hyperparameters` field, (2) `args.<field>` read, (3) CLAUDE.md §5 row, (4) `opg_doc.tex` parameter table. Document effective vs documented magnitude when they differ. → [`EXPERIENCE.md#hyperparameter-fanout`](EXPERIENCE.md#hyperparameter-fanout)
- **CLAUDE.md size budget** — `wc -c CLAUDE.md` < 40 000. Iter-history annotations ("iter X NOT PROMOTED because Y") route to `experiments/hypotheses.md` H## or `EXPERIENCE.md` §2; CLAUDE.md keeps invariants only. → [`EXPERIENCE.md#claude-md-size-budget`](EXPERIENCE.md#claude-md-size-budget)
- **Cumulative-vs-instantaneous metric distinction (HARD)** — `step_avg = train_time/step` is cumulative. Compute per-step delta `Δ_t = train_time[t] − train_time[t−1]` for throughput decisions before s50. Loss/grad: take latest, not cumulative. → [`EXPERIENCE.md#cumulative-metric-misread`](EXPERIENCE.md#cumulative-metric-misread)
- **Routing-reg input invariant (HARD)** — all routing regs operate on combined `p = softmax × sigmoid(gate)`. `grep -nE 'share / share\.sum\(' train_gpt.py` — zero hits in routing-reg paths. MoS exempt (§6.2). See H100 in `experiments/hypotheses.md`.
- **No-top-K-dispatch (HARD)** — RevDEQ forbids discrete-decision routing/dispatch. `grep -nE 'topk\(.*expert|capacity_factor.*ceil|argmax.*router' train_gpt.py` — every match must be flag-gated OFF under RevDEQ. Permitted: soft routing, Sinkhorn, Gumbel-softmax, ε-skip with `ε ≤ ε_bf16`. See H101.
- **`is_grad_enabled` vs `requires_grad` (HARD)** — fast-path dispatch checks `torch.is_grad_enabled() AND any(requires_grad)`. `nn.Parameter.requires_grad` is True permanently → flag-only check defeats kernels in no_grad code (RevDEQ FP iter). See `feedback_grad_enabled_vs_requires_grad.md`.

## 10. RevDEQ Specifics

These rules govern the DEQ block math directly and must update `train_gpt.py` + `opg_doc.tex` + targeted tests in the same commit.

### DEQ Input Conditioning
The DEQ block may add an output-side `x₀` term only if it is scaled by a learnable, per-dim coefficient derived from the Parcae ZOH discretization (`B̄ = Δ·B`, softplus-reparametrized). Unconditional residuals like `T = x₀ + Δ` are prohibited — they create shortcut fixed points regardless of expert dynamics. The current form (iter 66b) implemented in `Block.forward` is:

```
T_θ(z, x_0) = B̄ ⊙ RMSNorm_learn(x_0) + Δ(z, x_0)
```

`B̄` is independent of the solver's `β = 1 − Ā` except through the shared per-dim step size `Δ`. Pass it explicitly through `RevDEQFunction.apply` (see §9 "Custom autograd inputs"). Any change to this equation must update `train_gpt.py`, `opg_doc.tex` §eq:Tx_new, and the regression tests in `tests/test_gate_init_defaults.py` in the same commit.

### RevDEQ Reversibility Floor
Any per-dim coefficient that divides the RevDEQ backward reconstruction MUST be lower-bounded by a correctness constant. The reverse step is:

```
y_n = (y_{n+1} − β ⊙ T_θ(z_{n-1}, x_0)) / (1 − β)
```

with `(1 − β) = Ā`. Worst-case backward amplification per iter is `1/Ā`; over `K = num_layers` steps the bound scales as `(1/Ā)^K · ε_fp64`. For `K = 12` and fp64 (`ε ≈ 1e-15`), `Ā ≥ 0.1` keeps the bound at `10^12 · 1e-15 = 1e-3`, within smoke-test tolerances (1e-1). We enforce:

```
Ā = ε_rev + (1 − ε_rev) · exp(Δ·A),     ε_rev = parcae_reversibility_floor = 0.1
```

`ε_rev` is NOT a tuning knob. Changing it requires updating in the same commit:
1. `GPT.parcae_reversibility_floor` in `train_gpt.py`.
2. `experiments/test_arch.py::test_revdeq_reconstruction_at_a_bar_floor`.
3. `experiments/smoke_test.py` tolerance.
4. The `opg_doc.tex` §Parcae-params remark citing the value.

### Backward-only floors
Coefficients that do NOT appear in the backward reconstruction (notably `B̄`, which lives inside `T_θ` and never in `(y_{n+1} − β·T)/(1−β)`) MUST NOT carry a floor. `B̄ = Δ·B` stays paper-faithful and unbounded — adding a floor silently biases training away from the Mamba-ZOH form.

## 11. Promotion Rules

Promotion policy is **val_bpb-primary**: if val_bpb improved AND artifact ≤ 16 MB, promote.

**Procedure on improvement.**
1. Run §8 pre-commit chain + §9 audit; keep only after issues resolved.
2. Promote: `bash experiments/update_results.sh --promote`.
3. **Always review + `/simplify` before committing improvements** — keeps code clean.

**Diagnostic-gate failures DO NOT block promotion.** Failures in ortho / K-sweep / gg / inj / recon_err diagnostics are recorded as `validated_with_diagnostic_fail` in `meta_json["status"]` and surfaced as retry prescriptions for the *next* iter. Promotion is gated only on val_bpb improvement + 16 MB budget. Mechanism: the train script writes `run_valid=true` whenever val_bpb is recorded; diagnostic failures populate `failure_categories` + `retry_hint.json` for the next iter but don't block `--promote`.

**Strict-generalization unconditional promote** — when the new iter's functional class **strictly subsumes** the baseline's (there exists a setting of the new params where the iter *exactly* recovers the baseline's forward map), promote unconditionally regardless of val_bpb delta. Any regression is by construction an optimization-landscape artifact (init, LR, gradient topology), not a capacity loss — tune the new params, do NOT revert. If val_bpb regresses, tune in this order: (i) LR of new params, (ii) init (try matching baseline at step 0), (iii) gradient flow paths.

Required in the commit message: (1) the exact param setting that recovers the baseline forward map, (2) a 1–2 line argument that this setting is representable in the new parametrization, (3) how the optimizer's access to that point is preserved (LR, WD, init range).

Full rule, examples, and anti-examples (look-alikes to reject): [`EXPERIENCE.md#strict-generalization`](EXPERIENCE.md#strict-generalization).

**Reverting non-improvements.** If val_bpb is equal or worse AND the change does NOT strictly generalize the baseline → `git revert` to previous good state (weights stay in `previous/`).

## 12. Submission Process

1. Run 3 seeds (e.g. 42, 1337, 2024) on 8× H100.
2. Compute mean and std of val_bpb.
3. Create `records/track_10min_16mb/YYYY-MM-DD_<name>/`.
4. Include: `README.md`, `submission.json`, `train_gpt.py`, training logs.
5. Submit PR to `main`.
