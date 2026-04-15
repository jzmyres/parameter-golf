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
- **8xH100 SXM (competition)**: 600 seconds (10 min) — original competition constraint
- **2xL40S (dev)**: 1200 seconds (20 min) — relaxed for development hardware

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
| mlp_mult | 3.0 (hidden = 768 × 3 / num_experts via low-rank experts) |
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| vocab_size | 1024 |
| tie_embeddings | yes |
| deq_beta | 0.20 |
| num_refinements | 1 |

### Optimizer
| Parameter | Value |
|---|---|
| matrix_lr | 0.022 |
| scalar_lr | 0.02 |
| router_lr | 0.005 |
| tied_embed_lr | 0.03 |
| embed_lr | 0.6 |
| muon_momentum | 0.99 |
| muon_momentum_warmup_start | 0.92 |
| muon_momentum_warmup_steps | 800 |
| weight_decay | 1.08 (iter 24; applied to both AdamW and Muon param groups) |
| grad_clip_norm | 0.3 |
| warmdown_frac | 0.72 |

### Quantization & Techniques
- int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings
- SmearGate + BigramHash(4096+) + OrthoInit
- SWA every 50 steps, start_frac=0.4-0.5
- Sliding window eval (stride=64)

## How to Run

### Dev mode (2x L40S — primary dev hardware)
```bash
conda activate opg
# Single GPU:
python train_gpt.py
# 2 GPUs:
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### Full mode (8xH100 — final validation only)
```bash
conda activate opg
torchrun --standalone --nproc_per_node=8 train_gpt.py
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
    - **Hard-gate failures (ortho, K-sweep, gg, inj, recon_err) DO NOT block promotion** —
      they're tracked as tech debt and prescribed fixes for the *next* iteration. Promotion
      is gated only on val_bpb improvement + 16MB budget. This unblocks autoresearch when
      gate calibration is fighting val_bpb. To enable, the train script writes
      `run_valid=true` whenever val_bpb is recorded; gate failures populate
      `failure_categories` + `retry_hint.json` for the next iter, but don't block --promote.
11. If val_bpb equal or worse -> `git revert` to previous good state (weights stay in previous/)
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
- **Attn/MLP routing**: pure softmax (SoftDenseRouter)
- **MoS routing (exception)**: pure softmax only (convex combination summing to 1), NO sigmoid gates.
  Per Mixtape paper ("Breaking the Softmax Bottleneck Efficiently", NeurIPS 2019).
  The softmax bottleneck is broken by the mixture of softmaxes itself, not by gating.
- **Applied to ALL components**: attention output, MLP hidden, MoS output heads
- **Regularization** (per-token sparsity + global balance + orthogonality):
  - **Per-token sparsity**: L1 on routing weights (each token concentrates on fewer experts)
  - **Global balance**: MSE between mean expert usage and uniform target (per-component)
  - **Expert orthogonality**: |cos_sim| between expert weight groups → 0 (not ±1)
- Fully differentiable, no discrete routing decisions

### 3. Multi-head Latent Attention (MLA) with Gated Attention
- Paper (MLA): DeepSeek-V2 (arxiv:2405.04434)
- Paper (Gated Attn): "Gated Attention for Large Language Models" (arxiv:2505.06708)
  - Repo: https://github.com/YuchuanTian/GatedAttn (NeurIPS 2025 Best Paper)
- Low-rank KV compression: project to latent space, cache compressed, decompress on-the-fly
- Decoupled RoPE: split heads into RoPE and non-RoPE components
- **Gated Attention**: query-dependent per-head sigmoid gate after SDPA
  - Gate logits from expanded Q projection: `c_q outputs dim + num_heads`
  - Each token gets its own gate value per head (NOT a fixed scalar)
  - The paper claims: (1) mitigates attention sinks, (2) enables larger LR, (3) improves stability
  - **Test these claims** — verify the gate position (after SDPA, before output proj) improves perf
  - If a different gate position works better, document the finding

### 4. FSQ (Finite Scalar Quantization) in MoS Head
- Paper: FSQ (arxiv:2309.15505)
- Apply FSQ via STE in an intermediate projection space within the MoS output head
- Rank is flexible — tune as long as 16MB artifact size is met
- With V=1024, even full-rank projections are affordable (~917K params = 3.5MB fp16)

### 5. Diffusion-AR (Autoregressive + Iterative Refinement)
- Reference: `/home/mzhong4/work/research/tsu/WIP-TSU/code/model.py`
- **Refinement loop** (decoupled from DEQ solver):
  - Step 0: DEQ solve with x₀ = tok_emb(input_ids) (clean one-hot)
  - Step 1+: predict from z* → build soft embedding → DEQ solve with x₀ + soft_embed
  - Soft embedding: average CTP[i] and NTP[i-1] logits, top-k sparse embed, EMA blending
  - `num_refinements` controls how many predict→refine cycles (default: 1)
- **Dual-head MoS prediction** (REQUIRED):
  - **CTP (Current Token Prediction)**: predict current token (denoising)
  - **NTP (Next Token Prediction)**: predict next token (standard AR)
  - MoS with shared experts + specialized experts per head
  - CTP weight scales with num_refinements: `0.1 × num_refinements`
    (at refinement 0, input is clean one-hot — nothing to denoise)
  - All experts trainable (no frozen expert), xavier init (no SVD bias)
  - Track and plot CTP and NTP losses separately

### 6. Parameter Golf Hard Constraints (ENFORCED)
- Artifact size <= 16,000,000 bytes (code + compressed model)
- Training time <= 600 seconds on 8xH100 SXM (competition), <= 1200 seconds on 2xL40S (dev)
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

## Submission Process (when ready)
1. Run 3 seeds (e.g., 42, 1337, 2024) on 8xH100
2. Compute mean and std of val_bpb
3. Create folder in `records/track_10min_16mb/YYYY-MM-DD_<name>/`
4. Include: README.md, submission.json, train_gpt.py, training logs
5. Submit PR to main
