# Parameter Golf — Autoresearch Project

## Project Overview
OpenAI Parameter Golf challenge: train the best small LM that fits in a **16MB artifact** (code + compressed model), trains in **≤10 minutes** on 8×H100 SXM GPUs, evaluated by **val_bpb** (bits-per-byte) on FineWeb validation set. Lower is better.

Challenge: March 18 – April 30, 2026. Prize: $1M in OpenAI compute credits.

## Environment
- **Conda env**: `conda activate opg` (MUST be activated before any command)
- **Python deps**: see `requirements.txt`
- **Data**: `./data/datasets/fineweb10B_sp1024/`, tokenizer at `./data/tokenizers/fineweb_1024_bpe.model`

## Key Files
- `train_gpt.py` — Base training script (reference). Agent modifies a working copy.
- `data/` — Dataset and tokenizer (READ-ONLY, never modify)
- `records/` — Historical leaderboard submissions (READ-ONLY reference)
- `results.tsv` — Experiment log (untracked by git)
- `run.log` — Latest training console capture (untracked by git)
- `experiments/update_results.sh` — Log rotation + plot regeneration (run after EVERY iteration)

## Agent Non-Negotiables
- Only modify `train_gpt.py`, focused tests, and project docs unless the user explicitly approves a wider scope.
- Never modify `data/`, tokenizer code, evaluation harness code, `records/`, dependency files, or package manifests.
- Do not install new packages.
- Submission-like training must honor the 600 second wall-clock cap and 16 MB artifact limit.
- Code must remain DDP-safe for single-GPU and multi-GPU `torchrun`.
- Expert health is final-only and computed on normalized per-component expert shares; total routed mass is tracked separately.

## Reference Implementations (READ-ONLY)
- **RevDEQ**: `/home/mzhong4/work/research/rdeq/WIP-ARWDEQ/code/arwdeq/qwen3_utmoe_revdeq.py`
  - RevDEQ solver with custom autograd.Function, fp64 accumulators, Kahan compensation
  - Gated low-rank input injection: `z_hat = z + alpha * B(A(LN(x)))`
  - Warm start: y0=x, z0=x (initialize solver state from input)
- **TSU (CTP/NTP/MoS)**: `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py`
  - MoSLowRankOutputHead: frozen expert + trainable low-rank experts, dual B matrices
  - CTP labels: `_derive_ctp_labels(input_ids, k=0)` → current token prediction
  - Dirichlet sampling: `sample_dirichlet_lowrank_topk()` for soft embedding
  - TSUEmbedding: low-rank V→rank→d_model via SVD init

## Current SOTA
- **Leaderboard SOTA**: val_bpb = 1.1194 (abaybektursun, 2026-03-23). Key techniques: LeakyReLU(0.5)², TTT, Parallel Muon, XSA, GPTQ-lite, EMA. **Reference only** — our autoresearch baseline is independent.
- **This repo's working baseline** (true int6, dev hardware): tracked in `experiments/hypotheses.md` (latest promoted iter row). Update this line in the same commit that promotes a new baseline.

## Training Budget
- **Default script behavior**: 1000 iterations on DDP with all available GPUs; wallclock cap disabled (step-count governs).
  - `Hyperparameters.iterations = 1000`
  - `Hyperparameters.max_wallclock_seconds = 0`  # 0 = disabled
  - Canonical invocation: `torchrun --standalone --nproc_per_node=gpu train_gpt.py` (auto-detects all GPUs).
- **Submission runs (8xH100 SXM competition)**: 600-second wallclock cap — MUST pass `--max-wallclock-seconds=600` on the command line. The 600s hard cap comes from the competition constraint; it is never the default.
- **Step-matched dev runs**: the default (1000 iters, no wallclock) IS a step-matched dev run. To run shorter, override `--iterations=N`.
- **Fair comparison principle**: when configs have different throughput, compare at equal STEP COUNT
  (not wall-clock). A larger model needs proportionally more steps. Wall-clock matters for
  competition submission; step count matters for architectural comparison. The default (1000 iters,
  no wallclock) implements this by construction.

## Current Architecture (single source of truth: `train_gpt.py` `Hyperparameters`)
The values below MUST match `Hyperparameters` defaults in `train_gpt.py`. If you edit one, edit the other in the same commit (see "Config Single-Source-of-Truth" under Development Practices).

### Architecture
| Parameter | Value |
|---|---|
| num_layers | 12 |
| model_dim | 768 |
| num_heads | 8 |
| num_kv_heads | 4 |
| num_experts | 8 |
| num_shared_experts | 1 (DeepSeek shared expert, always-on with sigmoid gate) |
| mlp_mult | 3.0 (hidden = 768 × 3 / num_experts via low-rank experts) |
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| vocab_size | 1024 |
| tie_embeddings | yes |
| deq_beta | 0.50 (fallback when `use_parcae=False`) |
| use_parcae | True (per-dim Ā and B̄; supersedes scalar beta/jitter when active) |
| parcae_init_a_bar | 0.7 (initial Ā per dim; β₀ = 1-Ā₀ = 0.3) |
| parcae_init_b_bar | 0.3 (B̄₀ ≈ 1-Ā₀ at step 0; iter 66b continuity with iter 66a) |
| parcae_reversibility_floor | 0.1 (correctness constant — Ā ≥ this bound for RevDEQ backward safety, NOT a tuning knob) |
| deq_bptt_k | 2 (truncated BPTT: backward reconstructs only last 2 DEQ iters) |
| num_refinements | 1 |
| use_ctp | False (iter 94: CTP head disabled — NTP-only; CTP param banks not allocated) |

### Optimizer
| Parameter | Value |
|---|---|
| matrix_lr | 0.022 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| embed_lr | 0.6 |
| parcae_lr | 0.002 (applied to `parcae_raw_a`, `parcae_raw_delta`, `parcae_raw_b`) |
| muon_momentum | 0.99 |
| muon_momentum_warmup_start | 0.92 |
| muon_momentum_warmup_steps | 800 |
| weight_decay | 0.01 (iter 86: 0.30 → 0.01 re-test under the iter 93 landscape; applied to both AdamW and Muon groups) |
| grad_clip_norm | 1.0 |
| warmdown_frac | 0.72 |

### Routing & Expert Ranks
| Parameter | Value |
|---|---|
| router_scoring | linear (dot-product logits, iter 70) |
| attn_expert_rank | 128 |
| mlp_expert_rank | 192 |
| bigram_vocab_size | 0 (iter 93: BigramHash disabled; see H64) |
| bigram_dim | 128 |
| deq_beta_jitter | True (sample β from {0.3, 0.5, 0.7} per step when `use_parcae=False`) |
| deq_k_jitter_set | (4, 6, 10) (DEQ iteration counts sampled per step) |
| lyapunov_coef | 0.01 (λ_jac: Hutchinson-Frobenius penalty weight) |
| lyapunov_gamma | 0.97 (target spectral radius threshold) |
| lyapunov_warmup_frac | 0.05 (ramp over first 5% of wallclock) |
| denoising_coef | 0.01 (HyDRA denoising regularization weight) |
| denoising_noise_std | 0.01 (Gaussian noise σ for denoising penalty) |

### Quantization & Techniques
- int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings
- BigramHash(4096+) + OrthoInit
- SWA disabled (iter 1: dragged gates toward identity at 1h budget)
- Sliding window eval (stride=64)

## How to Run

### Default — DDP on all available GPUs, 1000 iterations, no wallclock cap
```bash
conda activate opg
torchrun --standalone --nproc_per_node=gpu train_gpt.py
```
`--nproc_per_node=gpu` auto-detects all visible GPUs (2×L40S on dev, 8×H100 on full). 1000 iterations is the step-count-governed default — wallclock is OFF unless explicitly set.

### Explicit GPU count (if the `gpu` alias is not supported on your launcher)
```bash
torchrun --standalone --nproc_per_node=2 train_gpt.py   # 2 GPUs
torchrun --standalone --nproc_per_node=8 train_gpt.py   # 8 GPUs
```

### Shorter dev iterations / smoke
```bash
torchrun --standalone --nproc_per_node=gpu train_gpt.py --iterations=200
```

### Submission-like run (8×H100 SXM, 600 s competition hard cap)
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --max-wallclock-seconds=600
```
The 600 s wallclock cap is NEVER the default — it is the competition constraint and must be opted into explicitly so step-count-governed dev runs can't accidentally submit.

### Single-GPU debug (no DDP, only if really needed)
```bash
python train_gpt.py   # world_size=1; runs but does not scale
```

### Evaluate results
```bash
grep "val_bpb:" run.log
grep "peak_vram_mb:\|artifact.*bytes" run.log
```

## Autoresearch Protocol

### Setup
1. Create branch: `git checkout -b autoresearch/<tag>` from main
2. Read `train_gpt.py`, `results.tsv`, and `git log` for full context
3. Verify data exists in `./data/datasets/fineweb10B_sp1024/`
4. Initialize `results.tsv` with header row
5. Run iteration 0 (converged baseline) to establish baseline val_bpb

### Experiment Loop
1. Read git state: `git log --oneline -20` + `results.tsv`
1a. **Read `experiments/hypotheses.md`** — review relevant hypotheses, state what this iteration tests
2. Make ONE focused change to `train_gpt.py`
2a. Run `python experiments/smoke_test.py` — MUST PASS before long training. Checks:
   - Loss decreases (not diverging)
   - DEQ reconstruction error stays < 1.0 (relative) and does not increase
   - DEQ iter convergence ||z_T - z_{T-1}|| does not diverge
   - No NaN/Inf gradients
3. Write/update tests (TDD — tests BEFORE implementation)
4. `git commit` the change
5. Run: redirect output to `run.log` (do NOT flood context)
6. Read results: `grep "val_bpb:\|peak_vram_mb:\|artifact.*bytes" run.log`
7. If grep empty -> crash. Run `tail -n 50 run.log`, attempt fix
8. Log to `results.tsv` (do NOT commit results.tsv)
9. **ALWAYS** run `bash experiments/update_results.sh` to rotate logs+weights and regenerate plots:
   - Rotates `current.log` → `previous.log`, `current/weights` → `previous/weights`
   - Copies `run.log` → `current.log`
   - Regenerates all plots
10. **PROMOTION POLICY (val_bpb-primary)**: if val_bpb improved AND artifact ≤ 16MB:
    - Run code review + `/simplify`, then keep
    - Promote to baseline: `bash experiments/update_results.sh --promote`
    - **ALWAYS review + `/simplify` before committing improvements** to keep code clean
    - **Diagnostic-gate failures (ortho, K-sweep, gg, inj, recon_err) DO NOT block promotion** —
      they're recorded as `validated_with_diagnostic_fail` in `meta_json["status"]` and
      surfaced as retry prescriptions for the *next* iteration. Promotion is gated only on
      val_bpb improvement + 16MB budget. This unblocks autoresearch when gate calibration is
      fighting val_bpb. To enable, the train script writes `run_valid=true` whenever val_bpb
      is recorded; diagnostic failures populate `failure_categories` + `retry_hint.json` for
      the next iter, but don't block --promote.
    - **EXCEPTION — strict-generalization unconditional promote**: when the new iter's
      functional class *strictly subsumes* the baseline's (i.e. there exists a setting of
      the new learnable params where the iter *exactly* recovers the baseline's forward
      map — e.g. `B̄ → 0` in iter 66b recovering iter 74b's `T_θ = Δ`), promote the new
      iter unconditionally regardless of val_bpb delta. Any val_bpb regression is by
      construction an optimization-landscape artifact (new-params init, LR mismatch,
      gradient topology), not a capacity loss — the fix is to tune the new params, not
      to revert the more-general form. See "Strict-Generalization Promotion Rule" below.
11. If val_bpb equal or worse AND the change does NOT strictly generalize the baseline
    -> `git revert` to previous good state (weights stay in previous/)
12. **Update `experiments/hypotheses.md`** — record results, update hypothesis statuses, note confounds
13. Track consecutive non-improvements. **STOP after 100 consecutive non-improvements** and seek user guidance

### Logging, Weights & Plotting (REQUIRED every iteration)
- **Training logs**: `experiments/training_logs/`
  - `baseline.log` — best config (promoted via `--promote`)
  - `previous.log` — last iteration (auto-rotated)
  - `current.log` — this iteration
- **Model weights**: `experiments/weights/`
  - `baseline/` — best config weights (promoted via `--promote`)
  - `previous/` — last iteration weights (auto-rotated)
  - `current/` — this iteration weights (written by train_gpt.py)
- **Metrics comparison**: After each iteration, run `python experiments/plot_metrics.py` to generate `experiments/metrics_comparison.png`
  - 4x3 grid with FULL TRAINING CURVES (not just final values):
    - Row 1: Train Loss curve, Val BPB curve, Step Avg (ms) curve
    - Row 2: DEQ Residual curve, DEQ Recon Error curve, DEQ Iter Convergence curve
    - Row 3: Expert Usage (per expert) curve, Expert Entropy curve, Expert Orthogonality curve
    - Row 4: Summary text comparing ALL final values between baseline and current
- **Progress plot**: After each iteration, run `python experiments/plot_progress.py` to update `experiments/progress.png` and `experiments/progress_full.png`
- **All metrics tracked**: train_loss, val_loss, val_bpb, step_avg_ms, deq_residual, deq_recon_err, deq_iter_conv, expert_usage (per expert), expert_entropy, expert_ortho
- Only track 2 configs for detailed comparison: baseline (best) vs current experiment
- **Prioritize architecture exploration** over hyperparameter tuning; search for relevant papers/techniques

### Time Budget
- **10 minutes max** per experiment (wall clock training)
- If a run exceeds 15 minutes, kill and treat as failure
- ~6 experiments/hour on dev hardware

### results.tsv Format (tab-separated)
```
commit	val_bpb	artifact_bytes	status	description
a1b2c3d	1.142760	15900000	keep	baseline converged config
b2c3d4e	1.140000	15800000	keep	added RevDEQ fixed-point layer
c3d4e5f	1.150000	16100000	discard	MLA attention (artifact too large)
d4e5f6g	0.000000	0	crash	soft routing OOM
```

## Optimization Goal

Find the architecture that satisfies ALL constraints below while yielding **optimal val_bpb**.
Optimize for **throughput** (steps/sec) and **convergence rate** (bpb/step) — both directly improve final perf within the fixed time budget.

When proposing architecture improvements:
- Include references to papers and git repos for verification
- Test specific claims (e.g., where to place gates, what initialization helps)
- Measure throughput impact — a 5% bpb improvement that costs 20% throughput may net negative

## Architectural Constraints (MUST SATISFY)

### 1. RevDEQ (Reversible Deep Equilibrium Model)
- Paper: https://arxiv.org/abs/2509.12917
- Reference: `/home/mzhong4/work/research/rdeq/WIP-ARWDEQ/code/arwdeq/qwen3_utmoe_revdeq.py`
- Model output defined as fixed point of a learned function
- **Two decoupled loops**:
  - **DEQ solver** (`num_layers`): coupled-state iterations to approximate fixed point
    - `y_{n+1} = (1-beta)*y_n + beta*f(z_n, x0)`, `z_{n+1} = (1-beta)*z_n + beta*f(y_{n+1}, x0)`
  - **Refinement** (`num_refinements`): predict → soft_embed → re-solve cycles
    - Each full DEQ solve produces ONE prediction; that prediction builds the soft embedding for the next solve
    - Total block calls = (1 + num_refinements) × num_layers × 2
- **Warm start**: z₀ = x (initially one-hot token embedding); on refinement steps, z₀ = x0_refined
- **fp64 accumulators** for add/subtract operations — ensures exact reversibility
- Reconstruction error should stay near numerical precision (verified by smoke test); large spikes usually indicate a diagnostics precision mismatch.
- **Fixed-point behavior is a desired goal**: monitor relative convergence (e.g. `||z_T - z_{T-1}||/||z_T||`) and residuals, and improve if it does not harm expert health or val_bpb.
- Smoke test checks: recon near precision, total loss decreasing, convergence not explosively diverging, no NaN/Inf, expert routing health

### 2. Soft Dense Routing (Dense MoE on ALL components)
- Paper: Soft MoE (arxiv:2308.00951) — adapted for dense routing
- ALL experts process ALL tokens — no top-k selection, no token dropping
- **Router routes on component INPUT** (pre-computation), consistent across all components
- **Full-dim low-rank experts**: every expert operates on the FULL model hidden dimension.
  Use low-rank matrices (dim→rank→dim) to control parameter count.
  Do NOT partition dimensions across experts (no `expert_size = dim // num_experts`).
- **Expert independence (HARD CONSTRAINT)**: every expert must be fully independent —
  **zero shared trainable parameters** within the expert computation path. Knowing one
  expert's weights must tell you nothing about any other expert. Specifically:
  - All projections (Q, K, V, Wo, gate, fc, down, MoS output heads/vocab projections) must be per-expert
  - All learned norms (including RMSNorm scale weights before expert projections) must be per-expert
  - Only the **router** (which routes TO experts, not inside them) and **non-learned
    operations** (RoPE cos/sin tables, activation functions) may be shared
  - When adding a new parameter to the expert path, it MUST have shape `(E, ...)`.
    A `grep -n` for shared `nn.Module` or `nn.Parameter` without `E` in the expert
    forward should find nothing.
- **Attn/MLP routing**: softmax allocation × sigmoid gate (SoftDenseRouter).
  Weights sum to ≤ 1 (NOT renormalized). The sigmoid gate allows the model to
  globally suppress the expert mixture (`T(z,x0) ≈ 0` when all expert paths close).
  More expressive than forcing sum=1.
  Expert-health min-share/CV metrics are computed on renormalized per-component
  shares; total routed mass is logged separately and must not be conflated with balance.
- **MoS routing**: pure softmax (convex combination summing to 1).
  Per Mixtape paper ("Breaking the Softmax Bottleneck Efficiently", NeurIPS 2019).
- **Applied to ALL components**: attention output, MLP hidden, MoS output heads
- **Regularization** (per-token sparsity + global balance + orthogonality):
  - **Per-token sparsity**: L1 on routing weights (each token concentrates on fewer experts)
  - **Global balance**: MSE between mean expert usage and uniform target (per-component)
  - **Expert orthogonality**: |cos_sim| between expert weight groups → 0 (not ±1)
- Fully differentiable, no discrete routing decisions

### 3. Per-Expert MLA (Multi-head Latent Attention) with Gated Attention
- Paper (MLA): DeepSeek-V2 (arxiv:2405.04434)
- Paper (Gated Attn): "Gated Attention for Large Language Models" (arxiv:2505.06708)
  - Repo: https://github.com/YuchuanTian/GatedAttn (NeurIPS 2025 Best Paper)
- **Each expert has its own complete MLA pipeline** (no shared params — see constraint #2):
  - Per-expert Q: low-rank `dim → expert_rank → H*d_head + H` (gate logits appended)
  - Per-expert KV compression: low-rank `dim → kv_rank → kv_latent_dim`
  - Per-expert KV decompression: per-expert RMSNorm + per-expert `kv_latent → K_nope, V`
  - Per-expert K_rope: low-rank `dim → kr_rank → H_kv*rope_dim`
  - Per-expert Wo: low-rank output projection `D → wo_rank → D` (mixes heads per expert)
- **Head-packed SDPA**: expert index extends head dimension (E×H query heads,
  E×H_kv KV heads) for a single FlashAttention call. GQA ratio H/H_kv preserved.
- Decoupled RoPE: split heads into RoPE and non-RoPE components
- **Gated Attention**: query-dependent per-expert-per-head sigmoid gate after SDPA
  - Gate logits from per-expert Q projection (appended to Q output)
  - Each token gets its own gate value per head per expert

### 4. FSQ (Finite Scalar Quantization) in MoS Head
- Paper: FSQ (arxiv:2309.15505)
- Apply FSQ via STE in an intermediate projection space within the MoS output head
- Rank is flexible — tune as long as 16MB artifact size is met
- With V=1024, even full-rank projections are affordable (~917K params = 3.5MB fp16)
- **Currently disabled** (iter 62, H53): `fsq_levels=0` bypasses FSQ — the low-rank MoS projection alone is sufficient within the 16MB budget. Code machinery retained for potential re-enabling.

### 5. Diffusion-AR (Autoregressive + Iterative Refinement)
- Reference: `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py`
- **Refinement loop** (decoupled from DEQ solver):
  - Step 0: DEQ solve with x₀ = tok_emb(input_ids) (clean one-hot)
  - Step 1+: predict from z* → build soft embedding → DEQ solve with x₀ + soft_embed
  - Soft embedding: average CTP[i] and NTP[i-1] logits, top-k sparse embed, EMA blending
  - `num_refinements` controls how many predict→refine cycles (default: 1)
- **NTP-only MoS prediction** (iter 94, 2026-04-24): the dual-head CTP+NTP
  design was ablated. `Hyperparameters.use_ctp = False` is the baseline.
  CTP param banks (`gate_ctp`, `gate_ctp_norm_weight`, `ctp_a_norm_weight`,
  `A_ctp_shared`, `A_ctp`, `B_denoise`, `ctp_rank_norm_weight`) are not
  allocated; `MoSHead.forward` returns `(log_p_ntp, log_p_ntp)` when
  disabled; `GPT.forward` sets `ctp_loss = 0`; `_get_soft_embedding`
  uses `p_mix = p_ntp` during refinement. Promoted at iter 94 — int6
  val_bpb 1.5952 vs baseline (iter 66b) 1.5926, K=8→K=128 Δ tightened
  from +0.0103 to +0.0073, artifact -592 KB (-9.0%), params -1.38M
  (-10.9%). See H60 in experiments/hypotheses.md.
  - The refinement soft-embedding in `_get_soft_embedding` now uses
    `p_ntp` only (no CTP mixing); the refinement mechanism still
    operates through the DEQ solve — what was removed is the CTP
    output-head parameter banks and its auxiliary loss gradient.
  - NTP MoS head still uses 2 shared + 1 specialized expert, xavier
    init, rank=256.
  - Historical (pre-iter-94) dual-head design: preserved behind
    `--use-ctp=1` CLI flag for A/B re-testing if needed.

### 6. Parameter Golf Hard Constraints (ENFORCED)
- Artifact size <= 16,000,000 bytes (code + compressed model)
- Training time <= 600 seconds on 8xH100 SXM for submission-like runs
- Must use FineWeb validation set for evaluation
- Tokenizer: SentencePiece BPE, vocab=1024

### 7. DDP Compatibility
- All code MUST work with both single-GPU and multi-GPU (torchrun DDP)
- Dev on 1-2x L40S, validate on 8xH100
- Never use GPU-count-specific logic without proper world_size handling

## Development Practices

### TDD (Test-Driven Development)
- Write tests BEFORE implementation for each architectural change
- Test categories:
  - Shape tests: verify tensor dimensions through the model
  - Gradient tests: verify gradients flow (especially through RevDEQ fixed-point)
  - Quantization roundtrip tests: verify model survives int6+zstd
  - DDP tests: verify multi-GPU correctness
  - Artifact size tests: verify <= 16MB after compression

### Pre-Commit Review Chain
Before EVERY commit, run this chain:
1. `/simplify` — clean up code
2. `coderabbit:review` — AI code review
3. `pr-review-toolkit:review-pr` — comprehensive PR review
4. `superpowers:requesting-code-review` — verify requirements
5. Apply simple, first-principled fixes to valid issues
6. Then `git commit`

### Hypothesis Log (`experiments/hypotheses.md`)
- **READ BEFORE each iteration** — review relevant hypotheses, state what's being tested, check for known fixes
- **UPDATE AFTER each iteration** — record the verdict, update hypothesis statuses, note confounds
- **Design experiments to test ONE hypothesis** with a single controlled variable
- **Status levels**: VERIFIED (controlled test), OBSERVED (confounded evidence), PROPOSED (untested), REFUTED (controlled disproof)
- A hypothesis is only VERIFIED when a dedicated experiment tests it with all else equal
- **Use as troubleshooting manual** — when DEQ diverges, routing collapses, or K-sweep degrades, consult the log for known fixes (e.g., H9: double WD when β is too high)
- **Stability over task performance** — prefer verified-stable configs over slightly-better-but-unverified ones

### Simplicity Criterion
- All else equal, simpler is better
- A 0.001 bpb improvement adding 20 lines of hacky code? Probably not worth it
- Removing code and getting equal results? Definitely keep
- Favor removing complexity over adding it

### Config Single-Source-of-Truth
Every tunable architectural knob (`num_layers`, `num_heads`, `num_experts`, `model_dim`, `mlp_mult`, ranks, etc.) MUST appear exactly once — in `train_gpt.py`'s `Hyperparameters` class — and be plumbed from there to every constructor. Defaults inside sub-module `__init__` signatures are allowed only as a fallback; the authoritative value for any run is `Hyperparameters.<field>`.

Rules:
1. **Add to Hyperparameters first**, then thread through `GPT.__init__` → `Block.__init__` → leaf modules. Never introduce a new knob whose only home is a constructor default.
2. **Tests assert against the config, not a literal**. `assert model.num_experts == args.num_experts` is allowed; `assert model.num_experts == 6` (or `>= 2`) is forbidden — the first catches drift, the second hides it.
3. **CLAUDE.md "Current Architecture" table is mirror-only**. Edits to `Hyperparameters` and edits to that table MUST land in the same commit. Do not update one without the other.
4. **`opg_doc.tex` §2.1 and the "Current SOTA"/"working baseline" lines are dated artifacts**. A PR that mutates `Block.forward`, the DEQ equation, or the promoted baseline MUST update these in the same commit, or open a `TODO(paper)` ticket noting the divergence.

Rationale (incident from 2026-04-15 review): five drift defects shipped together — a stale 27b comment in iter 27d code, CLAUDE.md's config table two phases behind real code, `num_experts` hardcoded in three classes, a test silently relaxed from `== 6` to `>= 2`, and `opg_doc.tex` describing a removed `gg_gate`. All five share one root cause: configuration was duplicated across files with no single source of truth, so each editor only updated the file in front of them. This subsection codifies the fix.

### DEQ Input Conditioning Rule
The DEQ block may add an output-side `x0` term only if it is scaled by a learnable, per-dim coefficient derived from the Parcae ZOH discretization (`B̄ = Δ·B`, softplus-reparametrized). Unconditional residuals like `T = x0 + Δ` are prohibited: they create shortcut fixed points regardless of what the expert dynamics learn. The current form implemented in `Block.forward` (iter 66b) is:
```
T_θ(z, x_0) = B̄ ⊙ RMSNorm_learn(x_0) + Δ(z, x_0)
```
where `B̄` is independent of the solver's `β = 1 − Ā` except through the shared per-dim step size Δ. Any change to this equation must update `train_gpt.py`, `opg_doc.tex` §eq:Tx_new, and the regression tests in `tests/test_gate_init_defaults.py` (`test_zero_expert_delta_returns_b_bar_times_norm_x0`, `test_direct_block_without_b_bar_skips_b_bar_term`, and `test_revdeq_default_trains_parcae_b_bar`) in the same commit.

`B̄` MUST be passed explicitly through `RevDEQFunction.apply(..., beta, b_bar, ...)`; storing it on a module attribute is insufficient because custom autograd can only return gradients for explicit inputs.

### RevDEQ Reversibility Floor Rule
Any per-dim coefficient that divides the RevDEQ backward reconstruction MUST be lower-bounded by a correctness constant documented as such in code. The RevDEQ reverse step is:
```
y_n = (y_{n+1} − β ⊙ T_θ(z_{n-1}, x_0)) / (1 − β)
```
where `(1 − β) = Ā`. Worst-case backward-amplification per iteration is `1/Ā`, so over `K = num_layers` backward steps the error bound scales as `(1/Ā)^K · ε_fp64`. For `K = 12` and fp64 (ε ≈ 1e-15), `Ā ≥ 0.1` keeps the bound at `10^12 · 1e-15 = 1e-3`, which smoke-test tolerances (1e-1) expect. We therefore enforce:
```
Ā = ε_rev + (1 − ε_rev) · exp(Δ·A),      ε_rev = parcae_reversibility_floor = 0.1
```
`ε_rev` is NOT a tuning knob. Changing it requires updating in the same commit:
1. `GPT.parcae_reversibility_floor` in `train_gpt.py`.
2. The reversibility contract test in `experiments/test_arch.py::test_revdeq_reconstruction_at_a_bar_floor`.
3. The smoke-test tolerance in `experiments/smoke_test.py`.
4. The `opg_doc.tex` §Parcae-params remark that cites the specific value.

Coefficients that do NOT appear in the backward reconstruction (notably `B̄`, which lives inside `T_θ` and never in the `(y_{n+1} − β·T)/(1−β)` step) MUST NOT carry a floor. `B̄ = Δ·B` stays paper-faithful unbounded — adding a floor silently biases training away from the paper's Mamba-ZOH form.

Rationale (iter 66b pre-commit review): iter 66a's `min_a_bar = 0.1` was the value with the right safety margin for `K = 12` DEQ iterations in fp64; the post-diff refactor bundled it with an extra `decay_floor`/`min_rate` compound wrapper that made Ā non-paper-faithful. Iter 66b strips the compound wrapper and retains only the single `ε_rev` floor — explicitly a correctness constant.

### Permutation Consistency Audit Rule
When multiple tensors of identical shape undergo `permute()+reshape()` to the same target layout (e.g., `(E,B,T,H,d)` → `(B,E*H,T,d)`), ALL must use identical permutation indices. Before committing any attention/expert tensor reshaping code:
```bash
grep -n 'permute(' train_gpt.py | grep -v '#'
```
Verify all groups of related permutes use the same index tuple. A single outlier is almost certainly a bug (cf. k_rope permute incident, commit ec1048b).

### Tensor Layout and Optimizer Coverage Rule
Any manual flatten/reshape of a parameter bank MUST have a focused equivalence test against an index-explicit formulation, such as `einsum`. Shape-only tests are insufficient because they do not catch transposed storage semantics (e.g., `(E,D,R)` flattened as if it were `(E,R,D)`).

Any manual optimizer grouping MUST assert that every `requires_grad` parameter appears in exactly one optimizer group. This is mandatory when adding learnable norms, heads, gates, or other parameters outside the shared block.

### Dead Code Audit Rule
When a feature is removed (e.g., gg_gate, SmearGate, tie_attn_mlp_router), the SAME commit must:
1. Remove ALL code paths that reference it (CLI args, constructor params, CONTROL_TENSOR_PATTERNS, diagnostic tracking)
2. Remove ALL doc references (CLAUDE.md tables, comments saying "removed")
3. Update ALL tests that assert on the removed feature
4. `grep -rn` for the removed name across the entire codebase — if any match remains, it's incomplete
5. Remove ALL tracking infrastructure (fields, methods, aggregation, prescriptions) that recorded values for the removed feature — constant-valued tracking is dead code too

Rationale (incident from 2026-04-18 review): gg_gate and inj_lin were removed but their tracking infrastructure survived (~100 lines), recording constant 1.0 values with zero diagnostic signal. Tautological validation checks on always-1.0 values could never fire.

### Module Alias Audit Rule
When replacing N independent modules with a single shared instance (e.g., `attn_router` + `mlp_router` → single `router`):
1. Verify all `named_parameters()` filters still match the canonical name (PyTorch deduplicates by identity)
2. Verify all per-instance loops (bias_update, health_scale, diagnostics) use `id()` dedup
3. Remove duplicate set/get operations on the same instance

Rationale (incident from 2026-04-17 review): iter 35 pooled-router merge created module aliases (`attn_router` = `mlp_router` = `router`). The optimizer param filter still looked for `attn_router.router.weight` — a name PyTorch never generates for an alias. The router silently trained with Muon at 4.4x the intended LR and with weight decay, making `router_lr` a no-op. Same root cause as the 2026-04-15 incident: partial migration left stale references.

### Hot-Path Sync Prohibition
The training loop (gradient accumulation + optimizer step) MUST NOT contain any `.item()`, `.cpu()`, or Python-scalar branching on GPU tensors. All control flow must use pure-tensor math (e.g., `torch.relu`, `torch.where`). GPU→CPU syncs are only permitted at log sites (guarded by `will_log_train`).

**Verification**: Before committing any training-loop code (especially Lyapunov/regularization): `grep -n '\.item()\|\.cpu()' train_gpt.py` and verify ZERO hits between the `for micro_step` and `train_loss /= grad_accum_steps` lines. Comments claiming "no .item()" are not sufficient — automated grep is the source of truth.

Rationale (incident from 2026-04-18 review): Lyapunov penalty code at line 3037 had a comment "Pure-tensor math — no .item() GPU-CPU syncs" immediately followed by `if scale_t.item() > 0:`. Three more `.item()` syncs were found in the same block. Comments lie, code doesn't.

### Doc-Code Invariant
When `opg_doc.tex` describes an algorithm and `train_gpt.py` implements a different (better) variant, the doc MUST note the deviation in a "Practical implementation" paragraph. The pseudocode represents the theoretical formulation; the implementation note is the source of truth for code.

### Compile-Wrapper Write Rule
All attribute writes to `base_model.shared_block` (or any potentially-compiled module) MUST go through `_unwrap_compiled_module()`. The pattern is: `sb = _unwrap_compiled_module(base_model.shared_block)` once per training-loop scope, then use `sb` for all reads/writes.

### Explicit Boundary Rule
Any code that crosses a wrapper, device, or optional-telemetry boundary MUST make that boundary explicit:
1. Mutate model state only on the semantic owner after unwrapping compile/DDP/DataParallel wrappers with `_unwrap_compiled_module()`.
2. Keep GPU tensors on GPU in the training hot path; use tensor math (`torch.relu`, `torch.where`, `clamp`) instead of Python branches on CUDA scalar tensors.
3. Treat optional summary metrics as optional under `set -euo pipefail`; shell summaries must preserve `?` fallbacks instead of aborting when a grep has no matches.
4. Add or update a contract test for the boundary. Comments and local intent are not enforcement.

Rationale (incident from 2026-04-23 review): validation wrote `_deq_k_override` to a wrapper instead of the underlying `GPT`, a Lyapunov CUDA scalar branch risked hot-path synchronization, and `update_results.sh` could abort on an incomplete log before printing the intended `val_bpb=?` fallback. All three bugs crossed an implicit boundary without a test.

### Custom Autograd Input Rule
Every tensor that should receive gradients through a custom `torch.autograd.Function` MUST be an explicit `apply(...)` input and have a corresponding gradient slot in `backward(...)`. Reading a trainable tensor from module state inside the function's forward/backward is not enough: autograd has no edge to chain through unless the tensor crosses the function boundary.

When a saved tensor is re-instantiated as a leaf inside a `for _ in range(K)` loop in `backward()` (for truncated-BPTT or K-step unrolled reverse solves), use `.clone().requires_grad_(flag)` — NOT `.detach().requires_grad_(flag)` — so that successive iterations and multiple legs (y-leg, z-leg) within one iteration do not share underlying storage. `.detach()` returns a view; a future in-place edit to the leaf inside `f_theta` then silently aliases across legs. Cost is one small-tensor allocation per step; the robustness is permanent.

Optional tensor inputs (e.g. `b_bar` when `use_parcae=False`) MUST be stored on `ctx` as a plain attribute (`ctx.b_bar_saved = tensor_or_None`), not embedded in `save_for_backward` via an empty-tensor sentinel. `save_for_backward` is semantically a list of *real* tensors; iterating it elsewhere and assuming non-empty shapes must remain safe.

Rationale (incident from 2026-04-23 review): `B̄ = Δ·B` was computed in `GPT._deq_solve()` and stored on a `Block` attribute, but `RevDEQFunction.apply(...)` only accepted `beta` and block parameters. The reverse pass re-ran `Block.forward`, so outputs numerically depended on `B̄`, but `backward(...)` never returned a `grad_b_bar`; task-loss gradients to `parcae_raw_b` were therefore silently absent. Contract tests must compare custom-backward gradients against explicit unroll gradients for every non-parameter tensor input (`beta`, `b_bar`, etc.). The clone-vs-detach clause comes from the 2026-04-24 pre-commit review of the same iter: two `.detach().requires_grad_()` leaves (y-leg, z-leg) aliased the same storage; the current backward is correct because autograd tracks by tensor identity, but any future in-place edit of `b_bar` inside `f_theta` would silently corrupt the other leg.

### Prenorm Scale Independence Rule (HARD CONSTRAINT)
**All learnable prenorm scales are independent — no two distinct linear inputs inside the training graph may share the same learned scale Parameter.** The RMS statistic (`RMSUnit(x) = x / sqrt(mean(x²) + ε)`) is parameter-free and may be reused freely; the learned multiplicative scale that follows it is owned exclusively by the linear input it conditions.

Formally: for every linear map `y = W·(RMSUnit(x) ⊙ g) + b` inside `T_theta` or its auxiliary heads, `g` is an `nn.Parameter` that conditions exactly one linear weight (`W`). The same `(W, g)` pair may appear in multiple forward paths (e.g. the main expert mixer and the orthogonality diagnostic), but `g` must never condition a second, distinct weight tensor.

**Shape follows the linear, not the rule.** The rule is about *independence*, not shape:
- A per-expert linear (per-expert Q/K/V/O banks, per-expert MLP gate/fc, per-expert MoS A-banks) owns a scale of shape `(E, D)` or `(E, rank)` — leading dim `E` matches the linear's per-expert dimension.
- A shared linear that the expert path routes *through* (router score, router gate, shared-expert gate, MoS routing gates) owns a scale of shape `(D,)` — there is no `E` dimension because the linear itself is not per-expert. This is still "independent": the router's score-scale and gate-scale are two distinct Parameters, even though both are shape `(D,)`.

Rationale: a shared learned prenorm scale couples unrelated linear maps and violates the expert/projection independence invariant. The regression this rule guards against is the pre-iter-66b state where a single `state_norm.weight` was multiplied before the router, every attention projection, every MLP input, the shared gate, and every MoS A-bank — eight+ distinct linear maps tied to one learned vector. The fix is one Parameter per linear input: `router.score_norm_weight`, `router.gate_norm_weight`, `attn.q_down_norm_weight`, `attn.kv_a_norm_weight`, `attn.kr_a_norm_weight`, `attn.k_nope_in_norm_weight`, `attn.v_in_norm_weight`, `mlp.gate_in_norm_weight`, `mlp.fc_in_norm_weight`, `shared_gate_norm_weight`, `mos_head.gate_ctp_norm_weight`, `mos_head.gate_ntp_norm_weight`, `mos_head.ctp_a_norm_weight`, `mos_head.ntp_a_norm_weight`, and any per-rank post-projection scales on the expert path.

**Enforcement.** Before committing any code that adds a learned prenorm scale, run:
```
grep -n '_norm_weight' train_gpt.py
```
For each scale, verify every usage multiplies the SAME linear weight — i.e. the scale conditions one `W`. A scale that appears against two distinct `W` Parameters in forward is a violation. The contract tests `test_expert_path_parameters_are_expert_independent` and `TestOptimizerCoverage.test_all_trainable_parameters_are_grouped_once` together catch missing scales and check per-expert shapes, but the grep audit is the first line of defense against cross-linear sharing.

### Identifier Uniqueness Across Wrappers Rule
An identifier that appears on more than one class in the same call graph (e.g. `GPT`/`Block`, `GPT`/`MoSHead`, `Block`/`MLP`) MUST NOT be a method on one class and an attribute on another. Two distinct semantics for one name — even when Python dispatch resolves them today — is a latent collision: a future proxy, wrapper, or `_unwrap_compiled_module` call can promote or demote the lookup and silently flip semantics.

Checklist for every new attribute or method on a class used inside the training graph:
1. `grep -n '\.<name>\b' train_gpt.py tests/ experiments/` — confirm no pre-existing method/attribute with the same base name on a sibling class.
2. If the name must be reused (e.g. a cached value mirroring a computed property), suffix the cache or delete the duplicate if it is dead state.
3. When adding a method whose name conflicts with a pre-existing attribute on a sibling class, rename (or delete) the attribute in the same commit and update every reader.

Rationale (incident, iter 66b pre-commit review, 2026-04-24): `_parcae_b_bar` was simultaneously a `GPT` method (computes `Δ·B` on the fly) and a `Block` attribute (threaded by `_deq_solve`). The assignment and clear in `_deq_solve` became dead once all readers passed `b_bar` explicitly, but the shared name made dead writes read like live state; one reviewer mis-escalated to a correctness CRITICAL because they could not distinguish the two. The `/simplify` pass dropped the Block attribute entirely — the method on `GPT` is now the sole holder of that name.

### Routing Predicate Migration Rule
When adding or changing any predicate that classifies parameters, modules, or tensors into regimes — optimizer groups, quantization tiers, compile targets, `CONTROL_TENSOR_PATTERNS`, `FP16_KEEP_PATTERNS`, EMA keys, gradient hooks, diagnostic buckets — the same commit MUST:
1. Enumerate every pre-existing item the new predicate matches, and record the before/after regime for each in the commit message.
2. Add (or extend) a contract test that asserts at least one representative pre-existing item lands in the intended regime — not the legacy one.
3. If the new predicate claims *full coverage* (e.g. "every trainable param"), add an assertion at construction time that raises if any required item is missing or duplicated (as `_assert_optimizer_param_coverage` does).

Rationale (incidents 2026-04-15 / -17 / -18 / -23, plus the pre-commit review of iter 66a): every prior silent-migration bug in this repo traces to a classification predicate that changed in one place while pre-existing objects quietly flipped regimes elsewhere. Partial migrations compound: the router-alias bug silently disabled `router_lr`; the `shared_block.named_parameters()` scope silently froze four RMSNorm scales (`bigram.proj_norm.weight`, `mos_head.input_norm.weight`, `final_norm.weight`, `embed_norm.weight`) for an entire phase of training; adding `"norm_weight"` to `CONTROL_TENSOR_PATTERNS` silently migrated `kv_norm_weight` and `hidden_norm_weight` from Muon/int6 to AdamW/fp32. Making the enumeration explicit in the commit and enforced in a test is the simplest first-principled fix.

### Strict-Generalization Promotion Rule
When an iteration's functional class **strictly subsumes** the baseline — i.e. there exists a setting of the new learnable parameters under which the iteration's forward map is *exactly* identical to the baseline's (not approximately, not "close enough") — the iteration MUST be promoted unconditionally, regardless of any val_bpb delta measured on a short dev-hardware run.

**Examples of strict generalization** (this repo):
- Iter 66b's `T_θ = B̄ ⊙ RMSNorm_learn(x₀) + Δ` strictly generalizes iter 74b's `T_θ = Δ`: setting `parcae_raw_b → −∞` (or `x0_inject_norm_weight → 0`) recovers `T_θ = Δ` exactly (up to an ε_min·Δ injection term that the optimizer absorbs).
- Changing a tied parameter to independent (e.g. iter 66a's tied `B̄ = 1 − Ā` → iter 66b's independent `B̄ = Δ·B`) generalizes as long as the tied setting is representable in the new parametrization.
- Adding a learnable gate initialized open + residual bypass: `y = (1−g)·x + g·f(x)` with `g = σ(·)` init to `0` gives `y = x` at init, recovering identity (the absence of `f`).

**Examples that are NOT strict generalization** (look-alikes to reject):
- A re-parametrization with a different output range even in the limit (e.g. adding `clamp(·, 0, c)` with `c < ∞` to a previously-unbounded output — the new form cannot represent outputs outside `[0, c]`).
- Adding a new loss term with strictly positive coefficient (the baseline is only recovered by setting the coefficient to exactly zero — not representable if the coefficient is a positive softplus).
- Replacing a full-rank projection with a low-rank factorization (low-rank cannot represent all full-rank maps).

**Required in the commit message for a strict-generalization iter:**
1. The exact setting of the new parameters that recovers the baseline's forward map.
2. A 1-2 line argument that this setting is representable in the new parametrization (e.g. "softplus(raw_b) can get arbitrarily close to 0; `Δ·ε_min ≈ 1e-3` is below training-noise scale"; or "`init_alpha = 0.0` reproduces baseline exactly").
3. How the optimizer's access to this baseline-equivalent point is preserved (LR, weight decay, init range should be such that the optimizer can *reach* the baseline-equivalent point if that's the minimum).

**What to do if val_bpb regresses after a strict-generalization promote:**
- Do NOT revert. The new iter cannot be worse in capacity than the baseline — any regression is an optimization-landscape artifact.
- Tune the new parameters in decreasing order of suspicion: (i) learning rate of the new params, (ii) initialization of the new params (try initializing so the step-0 forward map *exactly* matches the baseline), (iii) gradient flow paths if the new params sit behind a chain of reparametrizations.
- Record the diagnostic in hypotheses.md but continue running the next queued iter on top of the new baseline.

Rationale (iter 66b, 2026-04-23): reverting a strict generalization is categorically the wrong move — it strictly loses expressive capacity the baseline could not reach. The iter-66b / iter-74b choice is a training-dynamics question, not a capacity question; treating it as the latter lets autoresearch bounce between equivalent-or-better configurations forever on noise-floor val_bpb swings. Codifying "strict-generalization → unconditional promote" prevents that loop.

## Submission Process (when ready)
1. Run 3 seeds (e.g., 42, 1337, 2024) on 8xH100
2. Compute mean and std of val_bpb
3. Create folder in `records/track_10min_16mb/YYYY-MM-DD_<name>/`
4. Include: README.md, submission.json, train_gpt.py, training logs
5. Submit PR to main
