# Parameter Golf — Autoresearch Project

> **Before each iteration, READ `EXPERIENCE.md` §1 index** for the dated incident-driven rules. The §9 audit checklist below cites those anchors — consult them before adding new code of the same shape. New audit/development rules go into `EXPERIENCE.md` §1, not into this file (see §8).

## 0. Pre-Action Memory (READ BEFORE ANY ACTION)

> **MANDATORY**: Before making any code change, doc edit, training launch, or commit, read every memory file referenced below. The auto-memory `MEMORY.md` index is loaded with the system prompt, but individual files are NOT — read them with the Read tool. New memory files MUST be added to this section in the same commit that creates them, or they will not be read on future sessions.

**Memory directory**: `/home/mzhong4/.claude/projects/-project-ylin-mzhong4-research-opg-parameter-golf/memory/`

**User profile** — read first to frame interactions:
- `user_profile.md` — ML researcher, Parameter Golf, RevDEQ/MoE/MLA focus, L40S dev hardware

**Feedback (durable rules — overrides default behavior)** — read before any action:
- `feedback_hypotheses_sync.md` — Update hypotheses.md IN REAL TIME for any queue/baseline/status change
- `feedback_pre_commit_review.md` — Pre-commit chain: /simplify + 3 code reviews + fix valid issues
- `feedback_simplify_before_commit.md` — /simplify before each iteration commit
- `feedback_dry_fixes.md` — When fixing a bug class, audit ALL instances in same commit
- `feedback_arch_exploration.md` — Prioritize architecture over hyperparameter tuning
- `feedback_compile_training.md` — Disable torch.compile for dev runs
- `feedback_always_ddp.md` — Always use all GPUs with DDP, never single-GPU
- `feedback_wakeup_cadence.md` — 5-min wakeups after failure → 20-min after 3 healthy checks
- `feedback_lipschitz_in_ksweep.md` — Lipschitz + acyclicity primes permanent in K-sweep
- `feedback_per_wallclock_override.md` — val_bpb gate can be overridden on per-wallclock grounds (H72)
- `feedback_sparsity_value_props.md` — Score sparsity iters on val_bpb / throughput / reg as 3 orthogonal axes; soft-dense routing → router sparsity has zero throughput benefit without sparse dispatch
- `feedback_uv_install.md` — Authorized installs use `uv pip install` (user directive 2026-04-28)
- `feedback_check_gpu_free.md` — Preflight `pgrep` + `nvidia-smi` before launching ANY GPU job (silent contention wastes time + corrupts profiles)
- `feedback_profile_before_throughput.md` — Throughput optimization needs chrome trace; not log fragments
- `feedback_decouple_regularizers.md` — Antagonistic regularizers → keep one as metric, the other as loss
- `feedback_anneal_sparsity_coefs.md` — Sparsity coefs anneal from 0; warmup_delay_frac=0.3 default
- `feedback_deq_convergence.md` — DEQ as true fixed point; K=64 extrapolation; stability over task perf
- `feedback_expert_collapse.md` — Full-dim low-rank experts; no split-dim; router on pre-attention input
- `feedback_mla_preferred.md` — MLA is preferred attention; do not replace with MHA/GQA
- `feedback_mos_routing.md` — MoS uses pure softmax routing (no sigmoid gates)
- `feedback_refinement_decoupled.md` — Refinement step must be separate independent forward pass

**Project (current state, lessons)** — read for context on running work:
- `project_autoresearch.md` — Autoresearch setup + experiment loop protocol
- `project_group_f_lessons.md` — H71-H75 trio: architectural sparsity closed; iter 96 near-optimal
- `project_experiment_results.md` — Ongoing experiment log + insights
- `project_phase0_learnings.md` — Phase 0 takeaways: hyperparameter landscape
- `project_phase4_throughput_first.md` — Throughput micro-opts first; numerical equivalence required
- `project_deq_depth_insight.md` — DEQ effective depth = 1; soft sparsity could unlock real depth
- `project_architecture_ideas.md` — XSA, GPTQ-lite, Late QAT, DiffAttn, DeltaNet
- `project_fix_later.md` — Deferred tech debt

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

**Fair-comparison principle.** When configs have different throughput, compare at equal **step count**, not wall-clock. The default (1000 iters, no wallclock) implements this by construction; wall-clock matters only for the competition submission.

## 5. Current Architecture

Single source of truth: `train_gpt.py::Hyperparameters`. The tables below MUST match `Hyperparameters` defaults; edits to either land in the same commit (see §9 Single-source-of-truth).

### Architecture
| Parameter | Value |
|---|---|
| num_layers | 12 |
| model_dim | 768 (iter 96 baseline. Iter 98b D=1024 attempt NOT PROMOTED on dev hardware 2026-04-29 — micro-batch halving fit D=1024 in VRAM but val_bpb int6 1.5018 vs iter 100b 1.4893 (Δ +0.0125 regression) AND step_avg 25.1s vs 24.5s (+2.6% slower per-wallclock; refinement at D=1024 amplified to +70% step cost). D-scaling deferred to 8× H100 submission hardware. See H73 retry section in hypotheses.md.) |
| num_heads | 8 |
| num_kv_heads | 4 |
| num_experts | 16 (iter 96 baseline H71; iter 97 E=20 attempt NOT PROMOTED on per-wallclock grounds, see H72 — E-scaling past 16 closed) |
| num_shared_experts | 1 (DeepSeek shared expert, always-on with sigmoid gate) |
| mlp_mult | 3.0 (hidden = D × 3 / num_experts via low-rank experts) |
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| grad_accum_multiplier | 1 (iter 98b retired this default 2 → 1 after NOT PROMOTED. Field retained for future use; setting >1 multiplies base grad_accum_steps to halve per-step activation memory at constant effective batch. No LR/WD rescaling needed.) |
| vocab_size | 1024 |
| tie_embeddings | yes |
| deq_beta | 0.50 (fallback when `use_parcae=False`) |
| use_parcae | True (per-dim Ā and B̄; supersedes scalar beta/jitter when active) |
| parcae_init_a_bar | 0.7 (initial Ā per dim; β₀ = 1−Ā₀ = 0.3) |
| parcae_init_b_bar | 0.3 (B̄₀ ≈ 1−Ā₀ at step 0; iter 66b continuity with iter 66a) |
| parcae_reversibility_floor | 0.1 (correctness constant — Ā ≥ this bound for RevDEQ backward safety, NOT a tuning knob) |
| deq_bptt_k | 2 (truncated BPTT: backward reconstructs only last 2 DEQ iters) |
| num_refinements | 0 (2026-04-29 user directive: disable refinement by default after iter 98b showed +70% step cost at D=1024 / +6% at D=768 with no directly-ablated val_bpb benefit; iter 110 queued to test re-enable as a clean ablation. CLAUDE.md §6.5 still defines the architectural form for when re-enabled.) |
| use_ctp | False (iter 94: CTP head disabled — NTP-only; CTP param banks not allocated) |
| use_nsa_attention | False (iter 106 default off; CLI-enabled per launch. When True, replaces dense causal SDPA with the 2-branch NSA mixer — compression branch (mean-pool K/V over `nsa_compress_block_size` blocks at `nsa_compress_block_sliding_stride` stride) + sliding-window branch (last `nsa_sliding_window_size` tokens). Selection branch (3rd NSA branch) deferred to iter 106b: gated by `nsa_num_selected_blocks > 0`, currently 0. Per-expert-per-head learnable softmax mixer `nsa_branch_gate (E*H, 2)` initialized to `nsa_branch_gate_init=0.0` (uniform mix). See H86 in hypotheses.md.) |
| nsa_compress_block_size | 32 (only used when `use_nsa_attention=True`) |
| nsa_compress_block_sliding_stride | 16 |
| nsa_selection_block_size | 64 (selection branch deferred — iter 106b knob) |
| nsa_num_selected_blocks | 0 (0 disables selection branch; 4 enables it for iter 106b) |
| nsa_sliding_window_size | 256 |
| nsa_branch_gate_init | 0.0 (logits init to 0 → uniform softmax over branches at init) |

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
| mos_balance_mult | 50.0 (iter 26-lb-loss: hardcoded multiplier promoted to a Hyperparameter; multiplies MoS-NTP balance loss inside `_collect_routing_losses` — the principled fix for `mos_*_min_share` failures since H26) |
| min_share_loss_weight | 0.0 (iter 100b: dropped to 0; CV loss alone provides smooth global-balance regularization without the hard-floor antagonism that hurt iter 100. min_share remains a diagnostic metric — sentinel: `min_share < 0.005` sustained → intervene. See H76.) |
| cv_loss_weight | 2.0 (iter 100b: 0.10 → 2.0, 20× to compensate for dropping min_share floor; CV-only global balance VALIDATED at this weight. See H76.) |
| router_entropy_coef | 0.005 (iter 100b: per-token entropy penalty for soft per-token specialization) |
| router_entropy_warmup_delay_frac | 0.3 (iter 100b: anneal entropy_coef from 0 over training, ramping linearly 0→target after 30% of wallclock; avoids cold-start trap that hurt iter 99/101 architectural sparsity attempts. See H76 + `feedback_anneal_sparsity_coefs.md`.) |
| attn_expert_rank | 64 (iter 96 baseline, unchanged through iter 100b) |
| mlp_expert_rank | 96 (iter 96 baseline, unchanged through iter 100b) |
| bigram_vocab_size | 0 (iter 93: BigramHash disabled; see H64) |
| bigram_dim | 128 |
| deq_beta_jitter | True (sample β from {0.3, 0.5, 0.7} per step when `use_parcae=False`) |
| deq_k_jitter_set | (16, 24) — 2026-04-29 user directive: K-jitter RE-ENABLED with set {16, 24} after iter 98b K-sweep showed val_bpb is essentially CONVERGED at K=16 (K=16: 1.5018, K=128: 1.5039, Δ=+0.0021 — FP found at K=16). The previous K=24 OOM (profile_v8 2026-04-28) has been re-confirmed as RevDEQ-memory-O(1)-in-K via the iter 97.5b-fix diagnostic loop; no OOM concern at K=24. Adding K=24 to both training-jitter and the K-sweep matrix tests whether wider FP-depth jitter regularizes (analog of H12 VERIFIED). `deq_k_jitter=True`, `deq_k_max=24`, `deq_k_eval=16`. |
| lyapunov_coef | 0.0 (iter 88: λ_jac disabled — Parcae per-dim Ā already bounds spectral radius) |
| lyapunov_gamma | 0.97 (target spectral radius threshold) |
| lyapunov_warmup_frac | 0.05 (ramp over first 5% of wallclock) |
| denoising_coef | 0.0 (iter 89: HyDRA denoising disabled — same Parcae-redundancy logic as iter 88) |
| denoising_noise_std | 0.01 (Gaussian noise σ for denoising penalty) |

### Quantization & Techniques
- int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings
- BigramHash(4096+) + OrthoInit
- SWA disabled (iter 1: dragged gates toward identity at 1 h budget)
- Sliding-window eval (stride = 64)

## 6. Architectural Invariants

These define the model class. Each rule states *what* the architecture must be; rationale is in EXPERIENCE.md §2 (Lessons Learned) where applicable.

### 6.1 RevDEQ — Reversible Deep Equilibrium Model
Paper: arxiv:2509.12917. Reference impl: see §2.

- Output = fixed point of a learned function. **Two decoupled loops**:
  - **DEQ solver** (`num_layers` coupled-state iters): `y_{n+1} = (1−β)·y_n + β·f(z_n, x₀)`; `z_{n+1} = (1−β)·z_n + β·f(y_{n+1}, x₀)`.
  - **Refinement** (`num_refinements`): predict → soft_embed → re-solve. Total block calls = `(1 + num_refinements) × num_layers × 2`.
- **Warm start**: `z₀ = x` (one-hot token embedding); on refinement, `z₀ = x0_refined`.
- **fp64 accumulators** for add/sub — required for exact reversibility.
- **`deq_recon_err` semantic under TBPTT (NOT decision-grade).** Whenever `deq_bptt_k < num_layers` (the default since iter 28-tbptt) the reverse loop stops at iteration `K_fwd − K_bwd`, not at `z_0`. The logged metric (`((z_rec − z_0).norm() + (y_rec − z_0).norm()) / ||z_0||`) then measures how far the forward FP *travelled* in the un-reconstructed iterations — a "distance travelled" gauge, **not** a numerical reconstruction error. Expect values O(1) once the FP is non-trivial; do not gate divergence / promotion on its absolute magnitude. To measure true RevDEQ reconstruction error (target near fp64 precision, ~1e-12), set `deq_bptt_k = 0` (full BPTT) and re-run; only that regime makes recon_err comparable to the fp64 floor.
- Fixed-point behavior is desired: monitor `||z_T − z_{T−1}|| / ||z_T||` (i.e. `deq_iter_conv_rel`, which IS informative under TBPTT) and residuals; improve if it does not harm expert health or val_bpb.
- Smoke test asserts: recon near precision *(only valid when smoke runs full BPTT)* · loss decreasing · convergence not exploding · no NaN/Inf · expert routing healthy.

### 6.2 Soft Dense Routing (Dense MoE on ALL components)
Paper: Soft MoE (arxiv:2308.00951). Mixtape (NeurIPS 2019) for MoS softmax.

- ALL experts process ALL tokens — no top-k, no token dropping.
- **Router routes on component INPUT** (pre-computation), consistent across all components.
- **Full-dim low-rank experts**: every expert operates on the FULL hidden dim. Use low-rank matrices (`dim → rank → dim`); do NOT partition dimensions across experts.
- **Expert independence (HARD CONSTRAINT)**: every expert is fully independent — **zero shared trainable parameters** within the expert computation path. All projections (Q/K/V/Wo/gate/fc/down/MoS A-bank), all learned norms (RMSNorm scales) are per-expert (shape `(E, ...)`). Only the **router** itself and **non-learned ops** (RoPE tables, activations) may be shared. Adding a new param to the expert path: shape MUST start with `E`. See §9 "Prenorm scale independence (HARD)".
- **Attn/MLP routing**: `softmax(allocation) × sigmoid(gate)` (`SoftDenseRouter`). Weights sum to ≤ 1 (NOT renormalized). The sigmoid gate lets the model globally suppress the mixture (`T(z, x₀) ≈ 0` when all paths close).
- **MoS routing**: pure softmax (convex combination summing to 1).
- **Applied to**: attention output, MLP hidden, MoS output heads.
- **Expert-health metrics** (min-share / CV) are computed on **renormalized** per-component shares; total routed mass is logged separately.
- **Two distinct entropies — do not confuse them.** Both are reported under similar names but they answer different questions:
  - **Global utilization entropy** (`expert_entropy` / `attn_entropy` / `mlp_entropy` in logs): `H_global = -Σ_e p̄_e log p̄_e` where `p̄_e` is the batch-averaged renormalized share for expert `e`. **HIGH = uniform usage across batch = no dead experts ✓.** This is the dead-expert sentinel; it should stay close to `log(N_routed)`.
  - **Per-token routing entropy** (NEW under iter 99: `attn_pertoken_entropy` / `mlp_pertoken_entropy` in logs): `H_pertoken = mean_token(-Σ_e w(e|token) log w(e|token))` averaged over tokens. **LOW = each token concentrates weight on a few experts = specialization ✓.** This is the per-token sparsity / specialization signal; we want it driven down.
  - You can have HIGH global *and* LOW per-token simultaneously — that's the target regime: every expert gets used somewhere in the batch (no waste), but each individual token uses only a few experts strongly (specialization).
  - HIGH per-token entropy + HIGH global entropy = uniform smoothing, no specialization (current pre-iter-99 state).
  - LOW per-token entropy + LOW global entropy = winner-take-all collapse with dead experts (bad — `min_share_loss_weight` exists to prevent this).
- **Regularization** (each enforces a distinct objective; do not collapse them in commits):
  - **Per-token sparsity / specialization**: per-token entropy penalty (iter 99: `entropy_coef * H_pertoken`). Drives `H_pertoken → 0`. Optionally paired with softmax temperature `τ > 1`.
  - **Global balance / dead-expert prevention**: `min_share_loss_weight` (forces every expert above a min share floor) + `cv_loss_weight` (penalizes CV across experts). Drives `H_global → log(N)`.
  - **Expert orthogonality**: `|cos_sim| → 0` between expert outputs (post-mix `mu_e`).
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

**Optional sparse-attention path — NSA (iter 106, default off).** When `use_nsa_attention=True`, the head-packed `(B, E·H, T, d)` SDPA call is replaced by a two-branch Native Sparse Attention mixer (arxiv:2502.11089): (1) **compression branch** mean-pools K/V over fixed-size sliding blocks then attends with a rectangular causal mask; (2) **sliding-window branch** attends to the last W tokens with a band-causal mask. A per-expert-per-head learnable softmax gate (`nsa_branch_gate` shape `(E·H, 2)`) mixes the two outputs. Gated attention (post-SDPA per-head sigmoid) is preserved unchanged. The third NSA branch (selection — top-K per-query block selection) is deferred to iter 106b via `nsa_num_selected_blocks=0`. **Strict-generalization (CLAUDE.md §11):** `nsa_compress_block_size=1`, `nsa_compress_block_sliding_stride=1`, `nsa_sliding_window_size=T`, `nsa_branch_gate_init=0` recovers full causal SDPA exactly within bf16 numerical floor — promotion is unconditional on val_bpb improvement.

**Discarded alternative — bottleneck experts (iter 90, 91+92)**. The "low-dim bottleneck" rewrite (BottleneckIn `D→proj_rank→r` + ExpertBody at small `r` + BottleneckOut `r→proj_rank→D`) was tested as Group D and NOT PROMOTED. Empirically it underperformed full-D LoRA on **per-param efficiency** (`bpb/param 1.49 vs full-D LoRA's 1.17 — ~27% worse`, H70) AND on **SDPA throughput** (forces `d_head ≤ 48` at any `r ≤ 192` with `H_in ≥ 4`, off the FA tensorcore sweet spot of 64+). Both penalties compound when scaling N_expert. **Do not re-introduce bottleneck-style experts as a scaling axis.** The bottleneck infrastructure is preserved for archival reference at git tag `iter-91+92-bottleneck-NOT-PROMOTED` (commit `3e35655`) and side branch `autoresearch/bottleneck-rescue` (`proj_rank=48/64` rescue workspace). The iter 96 PROMOTED axis — full-D LoRA with rank-halving / E-doubling at iso-cost on linears — supersedes it (H71). See §6.2 for the routing semantics that this expert layout feeds into.

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
12. **Update `experiments/hypotheses.md`** — record results, update statuses, note confounds. The H-claim section MUST include a FULL eval result subsection (user directive 2026-04-28, K-sweep matrix mandate strengthened 2026-04-29): (a) roundtrip int6 val_bpb + val_loss; (b) **K-sweep matrix — ALWAYS print the FULL `k_sweep_table:` matrix (header row + one data row per K) verbatim from run.log, NOT just summary text**. The 14 columns are: K, val_bpb, attn_cv, mlp_cv, pool_cv, attn_min, mlp_min, attn_ortho, mlp_ortho, pertoken_ent, pool_ent, shared_gate, hutch_F, rd_step, iter_conv_rel. This protocol is non-optional — every iter that completes a K-sweep gets its matrix copied into hypotheses.md, no exceptions. Acyclicity primes (17 / 37 / 113) are bolded. (c) trajectory table — val_bpb at each in-training val checkpoint with Δ from prev; (d) acyclicity prime check results (Δ vs nearest power-of-2 in 0.01-0.02 = genuine FP). Numbers must be grep-able from run.log. See `feedback_hypotheses_sync.md`.
13. Track consecutive non-improvements. **STOP after 100** and seek user guidance.

### Logging, Weights & Plotting (every iteration)
- **Training logs**: `experiments/training_logs/{baseline,previous,current}.log`.
- **Model weights**: `experiments/weights/{baseline,previous,current}/`.
- **Metrics comparison**: `python experiments/plot_metrics.py` → `experiments/metrics_comparison.png` (4×3 grid: train_loss · val_bpb · step_avg_ms / DEQ residual · recon_err · iter_conv / expert_usage · entropy · ortho / summary text).
- **Progress plots**: `python experiments/plot_progress.py` → `experiments/progress.png`, `progress_full.png`.

#### Required routing-health metrics (ALL must be reported every train+val log line)

| Metric | Target | What it tracks | Pool prefix? |
|---|---|---|---|
| **`pertoken_entropy`** (sparsity, iter 99+) | **LOW** ≈ 1.0 nat | Per-token routing concentration: `−Σ_e w(e\|token) log w(e\|token)` averaged over tokens. LOW = each token uses few experts strongly = specialization | **NO** — single pooled router-output value; pool-level by construction |
| **`attn_entropy` / `mlp_entropy` / `pool_entropy`** (global utilization, iter 100b+) | **HIGH** ≈ log(N_routed) per slice; pool ≈ log(2·N_routed) | Global cross-batch entropy `−Σ_e p̄_e log p̄_e` computed on (a) the per-slice renormalized batch-averaged shares (attn / mlp) and (b) the full unrenormalized pool. Comparing slice values against pool diagnoses cross-slice dominance: pool >> max(attn_slice, mlp_slice) means total mass is well-spread across attn+mlp; pool ≈ slice means one slice carries most mass | **YES** — three distinct values, attn ≠ mlp ≠ pool |
| **`min_expert_contribution`** | **≥ 0.005** (0.5%) | `min_e p̄_e` — smallest batch-averaged share across the routed-expert pool. Sentinel for dead experts. Should report explicitly per pool slice (attn min, mlp min) | **YES** — `attn_min_share` / `mlp_min_share` differ because the per-component shares differ across pool slices |
| **`attn_cv` / `mlp_cv` / `pool_cv`** (coefficient of variation, iter 100b+) | **LOW** ≈ 0.2-0.3 per slice | Per-slice CV uses the renormalized within-slice distribution; pool CV uses the full 2R unrenormalized distribution. Per-slice CVs measure within-role imbalance independently; pool CV captures BOTH within-slice imbalance AND any tilt of total mass between attn and mlp slices. Diagnostic: large gap between attn_cv and mlp_cv = role-asymmetric routing (e.g., iter 100b s120 attn_cv≈1.07, mlp_cv≈0.18 — attn winner-take-all, MLP uniform). Large pool_cv with small per-slice CVs = cross-slice dominance | **YES** — three distinct values |
| **`ortho`** (expert orthogonality) | **LOW** ≈ 0.1-0.2 | `max\|cos_sim\|` between expert OUTPUT means. Low cosine = experts represent different directions | **YES** — `attn_ortho` / `mlp_ortho` differ because attention experts and MLP experts produce DIFFERENT outputs even with shared router; per-pool computation is required |
| **`router_mass`** | 0.7-0.95 typical | Mean `sigmoid(gate)` value — total routed contribution per token. Drops as the model gates the mixture down | NO — single gate, single value |
| **`hutch_F`** (FP spectral, iter 97.5b PERMANENT 2026-04-29) | LOW + decreasing across val checkpoints | Hutchinson-Frobenius estimator at the saved DEQ FP `z*`: `rho_F = sqrt(E[mean(jvp²)]) ≈ ||J||_F / sqrt(dim)` for `J = ∂T_θ/∂z`. Distinguishes contractive attractor (rho_F < 1, decreasing with training) from marginal stability (rho_F ≈ 1) or trivial dynamics (rho_F → 0). Emitted on val log lines AND in the K-sweep table per-K. Probe runs at `B_probe=1` slice of saved z*/x0 (set by iter 97.5b-fix 2026-04-29 to bound activation memory to ~1-2 GiB; previously the unsliced full-val-batch JVP OOM'd at 32 GiB on the 44 GB dev cap, misattributed to "SDPA backend rejection" in earlier iter 100b documentation). Silently skipped (no `hutch_F:` field in log) on OOM-pred guard / runtime OOM / SDPA-grad-incompatibility | NO — single global value at the FP |

**Prefix convention** (updated 2026-04-27, iter 100b): the SoftDenseRouter is a SINGLE pooled router shared across attn and mlp components (per CLAUDE.md §6.2 and the iter 35 router consolidation). Logging now decomposes routing-distribution metrics into THREE values: `attn_*` (per-slice renormalized within attn experts), `mlp_*` (per-slice renormalized within mlp experts), and `pool_*` (full 2R distribution without slice renormalization). The three values diagnose distinct phenomena: per-slice within-role imbalance (attn_cv vs mlp_cv asymmetry), cross-slice dominance (pool_cv vs max(per-slice)), and the relationship between them. Metrics derived from **expert outputs** (usage arrays, orthogonality, min_share per slice) keep their `attn_*`/`mlp_*` prefix as before. `pertoken_entropy` remains a single pool-level value (per-token entropy is a pool-level quantity by construction — each token has ONE distribution).

**K-sweep tabular emission** (PERMANENT 2026-04-28, iter 100b user directive): the eval K-sweep emits a `k_sweep_table:` row per K value with 14 fixed-width columns: `K val_bpb attn_cv mlp_cv pool_cv attn_min mlp_min attn_ortho mlp_ortho pertoken_ent pool_ent shared_gate hutch_F rd_step iter_conv_rel`. A header row precedes the data rows. `N/A` indicates an unavailable field (most commonly Hutchinson/Lipschitz when SDPA backend rejects under enable_grad — known iter 97.6 limitation). The legacy `k_sweep:k=N val_bpb:... attn_gate_iter:[...] router_gate_iter:[...] iter_conv_rel:... residual:...` line is preserved for backward compatibility with `experiments/plot_metrics.py`. Use the `k_sweep_table:` rows for cross-K and cross-iter routing-health comparisons; use `k_sweep:` for per-iteration gate trajectories.

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

Run before every commit that touches `train_gpt.py`. Each row is one-line enforcement; click the anchor for the full incident.

- **Single-source-of-truth (Hyperparameters)** — every tunable knob lives in `train_gpt.py::Hyperparameters` and is plumbed from there. Tests assert against `args.<field>`, not literals. CLAUDE.md §5 mirrors edits in the same commit. → [`EXPERIENCE.md#config-drift`](EXPERIENCE.md#config-drift)
- **Permutation consistency** — `grep -n 'permute(' train_gpt.py | grep -v '#'`; related groups (e.g. `(E,B,T,H,d) → (B,E·H,T,d)`) must use the same index tuple. A single outlier is almost certainly a silent transposition. → [`EXPERIENCE.md#permutation-consistency`](EXPERIENCE.md#permutation-consistency)
- **Optimizer coverage** — every `requires_grad` parameter must land in exactly one optimizer group (asserted at construction time via `_assert_optimizer_param_coverage`). Manual flatten/reshape of a parameter bank also needs an einsum-equivalence test (shape-only tests miss transposed-storage bugs).
- **Dead-code audit** — when removing a feature (e.g. `gg_gate`, `SmearGate`), remove ALL paths/tests/doc refs/tracking infrastructure in the same commit; `grep -rn '<removed_name>' .` returns zero. → [`EXPERIENCE.md#dead-code-tracking`](EXPERIENCE.md#dead-code-tracking)
- **Module-alias audit** — when merging N modules to a shared instance, verify `named_parameters()` filters match the canonical name and per-instance loops use `id()` dedup. → [`EXPERIENCE.md#router-alias`](EXPERIENCE.md#router-alias)
- **Hot-path sync prohibition** — `grep -n '\.item()\|\.cpu()' train_gpt.py`; ZERO hits between the `for micro_step` loop and `train_loss /= grad_accum_steps`. Comments saying "no .item()" don't count — automated grep is the source of truth. → [`EXPERIENCE.md#hot-path-sync`](EXPERIENCE.md#hot-path-sync)
- **Explicit boundary (incl. compile-wrapper writes)** — wrapper / device / optional-telemetry crossings get an explicit unwrap (use `_unwrap_compiled_module()` once per scope for `base_model.shared_block` writes), `?` fallback in shell summaries under `set -euo pipefail`, AND a contract test. GPU-stay covered by Hot-path sync row above. → [`EXPERIENCE.md#explicit-boundary`](EXPERIENCE.md#explicit-boundary)
- **Custom autograd inputs** — every grad-needing tensor is an explicit `apply(...)` arg with a matching `backward(...)` slot. Use `.clone().requires_grad_()` (NOT `.detach()`) for re-instantiated leaves. Optional inputs go on `ctx`, not `save_for_backward`. → [`EXPERIENCE.md#custom-autograd-input`](EXPERIENCE.md#custom-autograd-input)
- **Routing-predicate migration** — when changing any predicate that classifies parameters/modules into regimes (optimizer groups, quantization tiers, `CONTROL_TENSOR_PATTERNS`, `FP16_KEEP_PATTERNS`, EMA keys, gradient hooks, diagnostic buckets), enumerate before/after in the commit message + add a contract test. Full-coverage predicates need a construction-time assertion. → [`EXPERIENCE.md#routing-predicate-migration`](EXPERIENCE.md#routing-predicate-migration)
- **Identifier uniqueness across wrappers** — no name may be both a method and an attribute on sibling classes in the same call graph. `grep -n '\.<new_name>\b' train_gpt.py tests/ experiments/` before adding. → [`EXPERIENCE.md#identifier-uniqueness`](EXPERIENCE.md#identifier-uniqueness)
- **Prenorm scale independence (HARD)** — `grep -n '_norm_weight' train_gpt.py`; every learned scale conditions exactly one linear weight. Shape follows the linear (E-prefixed for per-expert; bare D for shared linears that route to experts but aren't themselves per-expert). → [`EXPERIENCE.md#prenorm-scale-independence`](EXPERIENCE.md#prenorm-scale-independence)
- **Doc-Code Invariant** — when `opg_doc.tex` describes an algorithm and `train_gpt.py` implements a different (better) variant, the doc MUST note the deviation in a "Practical implementation" paragraph. Pseudocode is theoretical; code is the source of truth. → [`EXPERIENCE.md#doc-code-invariant`](EXPERIENCE.md#doc-code-invariant)
- **Diagnostic-gate component awareness** — when a feature flag disables a code path (e.g. `use_ctp=False`), the corresponding diagnostic emission MUST be gated on the same flag, and any retry prescription for that component MUST recommend a component-specific lever (e.g. `mos_balance_mult` for MoS routing collapse, NOT global `weight_decay`). `grep -n 'mos_ctp\|use_ctp' train_gpt.py` — every diagnostic spec referencing a CTP-only attribute lives behind a `mos_head.use_ctp` guard. → [`EXPERIENCE.md#diagnostic-gate-component-awareness`](EXPERIENCE.md#diagnostic-gate-component-awareness)
- **Hyperparameter fan-out** — every documented knob lives in `Hyperparameters`, is reachable via `_parse_cli_overrides`, and its consumer reads `args.<field>` (no constructor literal that shadows the dataclass). Four-touch rule for new knobs: (1) `Hyperparameters` field, (2) `args.<field>` read at consumer, (3) CLAUDE.md §5 row, (4) `opg_doc.tex` parameter table or "Practical implementation" note. When effective magnitude differs from documented magnitude (e.g. via balance-mult dedup), document the effective value or fix the multiplication. → [`EXPERIENCE.md#hyperparameter-fanout`](EXPERIENCE.md#hyperparameter-fanout)

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
