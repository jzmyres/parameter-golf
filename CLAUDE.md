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

## Current SOTA
val_bpb = 1.1194 (abaybektursun, 2026-03-23)
Key techniques: LeakyReLU(0.5)², TTT, Parallel Muon, XSA, GPTQ-lite, EMA

## Training Budget
1.5x of original = 900 seconds on 8xH100 SXM (relaxed from 600s)

## Converged Best-Known Config (Iteration 0 Baseline)
This is the consensus of the top 3 leaderboard entries. Use as starting point.

### Architecture
| Parameter | Value |
|---|---|
| num_layers | 10 |
| model_dim | 512 |
| num_heads | 8 |
| num_kv_heads | 4 |
| mlp_mult | 3 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| vocab_size | 1024 |
| tie_embeddings | yes |

### Optimizer
| Parameter | Value |
|---|---|
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 |
| momentum_warmup_start | 0.92 |
| momentum_warmup_steps | 1500 |
| muon_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| warmdown_iters | 3000 |

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
9. If val_bpb improved AND artifact <= 16MB -> run `/simplify`, then keep
10. If val_bpb equal or worse -> `git revert` to previous good state
11. Track consecutive non-improvements. **STOP after 100 consecutive non-improvements** and seek user guidance
12. Run `python experiments/plot_metrics.py` and `python experiments/plot_progress.py` to update plots

### Logging & Plotting (REQUIRED every iteration)
- **Training logs**: Save full stdout/stderr to `experiments/training_logs/`
  - `baseline.log` — the current best config (re-run when best changes)
  - `current.log` — the current experiment being tested
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

## Architectural Constraints (MUST SATISFY)

### 1. RevDEQ (Reversible Deep Equilibrium Model)
- Paper: https://arxiv.org/abs/2509.12917
- The main transformer backbone MUST use RevDEQ: model output defined as fixed point of a learned function
- **Coupled-state iteration** with relaxation beta=0.5:
  - `y_{n+1} = (1-beta)*y_n + beta*f(z_n, x0)`
  - `z_{n+1} = (1-beta)*z_n + beta*f(y_{n+1}, x0)`
- **fp64 accumulators** for add/subtract operations (paper recommendation) — ensures exact reversibility
- **Reconstruction error MUST be < 1e-8** (verified by smoke test before every run)
- **Convergence regularization**: 1.0 * ||z_T - z_{T-1}||²/||z_T||² added to loss to encourage fixed-point convergence
- Track: equilibrium residual `||z - f(z)||`, reconstruction error, and iter convergence `||z_T - z_{T-1}||`
- Smoke test (`experiments/smoke_test.py`) MUST pass before any long training run:
  - Recon error < 1e-8 (HARD)
  - Iter convergence must not diverge > 3x from initial (trending toward equilibrium)
  - Loss must not increase by > 0.5
  - No NaN/Inf gradients

### 2. Soft Dense Routing (Dense MoE — no sparsity)
- Inspired by Soft MoE (arxiv:2308.00951) but fully dense — ALL experts process ALL tokens
- No top-k selection, no token dropping, no sparse gating
- Routing weights via softmax over experts, with **sigmoid gating** to break convex constraint:
  - Per-expert sigmoid gate AFTER softmax routing: `effective_weight = sigmoid(gate_i) * softmax_weight_i`
  - This allows the model to "skip" experts by driving sigmoid gate toward 0
  - Learned gate scalars per expert, initialized near 0 (sigmoid ≈ 0.5)
- Fully differentiable, no discrete routing decisions

### 3. Multi-head Latent Attention (MLA) with Gated Attention — DeepSeek
- Replace standard MHA/GQA with MLA
- Low-rank KV compression: project to latent space, cache compressed, decompress on-the-fly
- Decoupled RoPE: split heads into RoPE and non-RoPE components
- Absorb decompression into subsequent linear layers where possible
- **Gated Attention** (arxiv:2505.06708): apply head-specific sigmoid gate after SDPA
  - Query-dependent sparse gating scores modulate attention output per head
  - Mitigates attention sinks, improves long-context performance
  - Enables larger learning rates and better training stability
  - Negligible parameter overhead (one gate vector per head)

### 4. FSQ (Finite Scalar Quantization) with Low-Rank Non-Linear Bottleneck
- Apply FSQ in a low-rank intermediate space within the model
- Use non-linearity (SiLU/GELU) before/after FSQ to recover expressiveness lost by low rank
- FSQ discretizes continuous values to a finite set of scalars
- The low-rank bottleneck compresses representations before quantization
- Non-linearity after FSQ expands back to full expressiveness

### 5. Diffusion-AR (Autoregressive + Single-Step Diffusion)
- Each DEQ iteration incorporates a diffusion-like denoising step
- The model refines soft token predictions across iterations
- AR-like: step i+1 depends on step i via shared KV state
- Diffusion-like: each step starts from a noisy soft distribution and denoises
- Dual signal: current token prediction (CTP) + next token prediction (NTP)

### 6. Parameter Golf Hard Constraints (ENFORCED)
- Artifact size <= 16,000,000 bytes (code + compressed model)
- Training time <= 600 seconds on 8xH100 SXM
- Must use FineWeb validation set for evaluation
- Tokenizer: SentencePiece BPE, vocab=1024

### 5. DDP Compatibility
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

### Simplify Before Commit
- Run `/simplify` on every successful experiment before committing
- Keep code clean and minimal — complexity must earn its keep

### Simplicity Criterion
- All else equal, simpler is better
- A 0.001 bpb improvement adding 20 lines of hacky code? Probably not worth it
- Removing code and getting equal results? Definitely keep
- Favor removing complexity over adding it

## Submission Process (when ready)
1. Run 3 seeds (e.g., 42, 1337, 2024) on 8xH100
2. Compute mean and std of val_bpb
3. Create folder in `records/track_10min_16mb/YYYY-MM-DD_<name>/`
4. Include: README.md, submission.json, train_gpt.py, training logs
5. Submit PR to main
