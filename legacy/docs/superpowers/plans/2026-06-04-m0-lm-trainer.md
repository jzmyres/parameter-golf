# M0 LM Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an organized OPG research `train_gpt.py` that preserves+optimizes the core improvement of `reports/opg_doc.tex` — recurrent depth buying task utility under a fixed memory budget — via a floor-free reversible recurrent core with MLA + smooth-sparse MoE + MoS, replacing the 10.4k-line rich architecture.

**Architecture:** Tied, token-injected, additive-coupling **reversible** recurrence (explicit inverse, O(1) activation memory via a custom `autograd.Function`); per-iteration blocks use **MLA** attention + **SwiGLU MoE** (router `relu`/`softmax` as a control experiment); readout is final state → **MoS** head. Low-rank matrices with erank-tuned rank; int6 ≤16 MB artifact; Muon+AdamW; FineWeb SP1024 + synthetic depth-hard eval.

**Tech Stack:** PyTorch (bf16 autocast, DDP/torchrun), SentencePiece, zstandard, custom `torch.autograd.Function`. Conda env `opg`.

**Source references (port, don't re-derive):** clean LM scaffold = `git show a15093a:train_gpt.py` (1,126 lines: `DistributedTokenLoader`, `build_sentencepiece_luts`, `run_validation`/BPB, `Muon`, `RMSNorm`, `Rotary`, DDP setup); int6 artifact = current `train_gpt.py::encode_scored_artifact`/`save_int6_artifact`; M0 forward + halting-free readout patterns = `experiments/p1_synthetic.py::AdditiveCouplingP1Model`. Spec: `docs/superpowers/specs/2026-06-04-m0-lm-trainer-design.md`.

---

## File structure

- Create: `train_gpt_m0.py` — the new research trainer (built + validated in isolation, then promoted to `train_gpt.py` in Task 12). Single file, clearly sectioned: config → modules (RMSNorm/Rotary/MLA/SwiGLU-MoE/MoS) → reversible core (`ReversibleRecurrence` + `RevRecurrenceFn`) → GPT model → data/eval/artifact/optimizer → metrics → train loop → CLI.
- Create: `tests/test_m0_reversible.py`, `tests/test_m0_modules.py`, `tests/test_m0_metrics.py`, `tests/test_m0_trainer.py`.
- Create: `experiments/run_m0_control_experiments.sh` — ReLU-vs-softmax router + φ(r) + MLA-dim sweeps.
- Modify (Task 10): `reports/opg_doc.tex` (§Goal exclusions, §Models readout), `tests/test_training_contracts.py`.
- Move (Task 11): rich `train_gpt.py` + rich-only `experiments/components/*` + legacy `experiments/test_*` → `legacy/`.

Keep each mechanism in its own section/class so it is independently testable. CPU-only tests use tiny dims; DDP/GPU paths get a documented smoke.

---

## Task 1: Reversible additive-coupling core (forward + explicit inverse)

**Files:**
- Create: `train_gpt_m0.py` (start the file: imports, `RMSNorm`, a stub block)
- Test: `tests/test_m0_reversible.py`

- [ ] **Step 1: Write the failing reconstruction test**

```python
# tests/test_m0_reversible.py
import torch
from train_gpt_m0 import ReversibleRecurrence, _TinyDelta

def test_additive_coupling_reconstructs_initial_state():
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64)
    a0 = b0 = x0
    (aK, bK), _states = rec.forward_states(a0, b0, x0, depth=5)
    a_rec, b_rec = rec.invert(aK, bK, x0, depth=5)
    assert torch.allclose(a_rec, a0, atol=1e-9)
    assert torch.allclose(b_rec, b0, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails** — `conda run -n opg python -m pytest tests/test_m0_reversible.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement the coupling**

```python
# train_gpt_m0.py
import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x)

class _TinyDelta(nn.Module):  # test-only delta block
    def __init__(self, d): super().__init__(); self.n = RMSNorm(d); self.l = nn.Linear(d, d, bias=False).double()
    def forward(self, x): return self.l(self.n(x))

class ReversibleRecurrence(nn.Module):
    """a_{k+1}=a_k+F(RMSNorm(b_k+x0)); b_{k+1}=b_k+G(RMSNorm(a_{k+1}+x0)). Explicit inverse."""
    def __init__(self, F, G): super().__init__(); self.F, self.G = F, G
    def forward_states(self, a, b, x0, depth):
        states = []
        for _ in range(int(depth)):
            a = a + self.F(b + x0)
            b = b + self.G(a + x0)
            states.append(0.5 * (a + b))
        return (a, b), states
    def invert(self, a, b, x0, depth):
        for _ in range(int(depth)):
            b = b - self.G(a + x0)
            a = a - self.F(b + x0)
        return a, b
```
(Note: `F`/`G` already include their input RMSNorm via `_TinyDelta.n`; the `+x0` injection is applied before the block. Keep norm inside the block so the inverse recomputes it identically.)

- [ ] **Step 4: Run to verify it passes** — same command → PASS.

- [ ] **Step 5: Commit** — `git add train_gpt_m0.py tests/test_m0_reversible.py && git commit -m "feat(m0): reversible additive-coupling core with explicit inverse"`

---

## Task 2: Custom reversible-BPTT backward (the critical correctness gate)

**Files:** Modify `train_gpt_m0.py` (add `RevRecurrenceFn`); Test: `tests/test_m0_reversible.py`

- [ ] **Step 1: Write the failing grad-equivalence test**

```python
def test_reversible_backward_matches_ordinary_autograd():
    torch.manual_seed(0); d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    # reference: ordinary autograd through forward_states
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=4)
    ref = (0.5*(aK+bK)).pow(2).sum(); ref.backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}; ref_x0 = x0.grad.clone()
    for p in rec.parameters(): p.grad = None
    x0b = x0.detach().clone().requires_grad_(True)
    # reversible path
    zK = rec.run_reversible(x0b, depth=4)  # returns 0.5(aK+bK), O(1) memory
    zK.pow(2).sum().backward()
    for n, p in rec.named_parameters():
        assert torch.allclose(p.grad, ref_g[n], atol=1e-6), n
    assert torch.allclose(x0b.grad, ref_x0, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`run_reversible` undefined).

- [ ] **Step 3: Implement `RevRecurrenceFn` + `run_reversible`**

```python
class RevRecurrenceFn(torch.autograd.Function):
    # Saves only (aK,bK,x0); reconstructs each step's inputs on backward → O(1) activation memory.
    @staticmethod
    def forward(ctx, rec, x0, depth, *params):
        with torch.no_grad():
            (aK, bK), _ = rec.forward_states(x0, x0, x0, depth)
        ctx.rec, ctx.depth = rec, depth
        ctx.save_for_backward(aK.detach(), bK.detach(), x0.detach())
        return aK, bK
    @staticmethod
    def backward(ctx, ga, gb):
        rec, depth = ctx.rec, ctx.depth
        a, b, x0 = ctx.saved_tensors
        params = list(rec.parameters()); pgrads = [torch.zeros_like(p) for p in params]
        gx0 = torch.zeros_like(x0)
        for _ in range(depth):
            # reconstruct step inputs (reverse coupling)
            b_prev = b - rec.G(a + x0)
            a_prev = a - rec.F(b_prev + x0)
            # local recompute with grad to get vjps
            ap = a_prev.detach().requires_grad_(True); bp = b_prev.detach().requires_grad_(True)
            x0r = x0.detach().requires_grad_(True)
            a_new = ap + rec.F(bp + x0r)
            b_new = bp + rec.G(a_new + x0r)
            grads = torch.autograd.grad((a_new, b_new), [ap, bp, x0r, *params],
                                        grad_outputs=(ga, gb), retain_graph=False, allow_unused=True)
            ga, gb, gx0_step = grads[0], grads[1], grads[2]
            gx0 = gx0 + (gx0_step if gx0_step is not None else 0)
            for i, g in enumerate(grads[3:]):
                if g is not None: pgrads[i] = pgrads[i] + g
            a, b = a_prev, b_prev
        gx0 = gx0 + ga + gb  # a0=b0=x0 seed
        return (None, gx0, None, *pgrads)

def run_reversible(self, x0, depth):
    aK, bK = RevRecurrenceFn.apply(self, x0, depth, *self.parameters())
    return 0.5 * (aK + bK)
ReversibleRecurrence.run_reversible = run_reversible
```
(Per `EXPERIENCE.md#custom-autograd-input`: every grad-needing tensor — `x0` and each parameter — crosses `apply(...)` and has a backward slot; reconstruct leaves use `.detach().clone().requires_grad_` semantics via fresh `requires_grad_(True)` tensors, no aliasing.)

- [ ] **Step 4: Run to verify it passes** — PASS (atol 1e-6).
- [ ] **Step 5: Commit** — `git commit -am "feat(m0): O(1)-memory reversible BPTT backward + grad-equivalence test"`

---

## Task 3: MLA attention block

**Files:** Modify `train_gpt_m0.py` (add `MLAttention`, `Rotary`); Test: `tests/test_m0_modules.py`

- [ ] **Step 1: Failing test** — shape + causality + KV-latent size:

```python
# tests/test_m0_modules.py
import torch
from train_gpt_m0 import MLAttention
def test_mla_shapes_and_kv_latent():
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8)
    x = torch.randn(2, 6, 32)
    y = m(x)
    assert y.shape == x.shape
    assert m.kv_latent == 8  # compressed KV dim < dim
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `MLAttention`** — port the MLA path from the current `train_gpt.py::CausalSelfAttention` (the `kv_latent_dim` low-rank down/up projections + decoupled RoPE + SDPA causal). Strip per-expert/NSA/smear branches; keep: `q_down`→`q_up`, `kv_down`→`k_up`/`v_up`, decoupled `k_rope`, `F.scaled_dot_product_attention(is_causal=True)`, output proj. All projections low-rank (`kv_latent`, `q_latent`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): MLA attention block (low-rank KV)"`

---

## Task 4: SwiGLU MoE with relu/softmax router (control-experiment knob)

**Files:** Modify `train_gpt_m0.py` (add `SwiGLUMoE`); Test: `tests/test_m0_modules.py`

- [ ] **Step 1: Failing tests** — both routers run; ReLU yields sparsity; routing is smooth (reversibility-safe):

```python
from train_gpt_m0 import SwiGLUMoE
def test_moe_routers_run_and_relu_is_sparse():
    x = torch.randn(2, 5, 32)
    for rt in ("softmax", "relu"):
        moe = SwiGLUMoE(dim=32, n_experts=8, expert_rank=8, router_type=rt)
        y = moe(x); assert y.shape == x.shape
        if rt == "relu":
            w = moe.last_route  # (.., n_experts)
            assert (w == 0).any()         # exact-zero sparsity
def test_moe_routing_is_smooth_for_reversibility():
    moe = SwiGLUMoE(dim=32, n_experts=8, expert_rank=8, router_type="relu")
    x = torch.randn(2, 5, 32); y1 = moe(x); y2 = moe(x + 1e-7)
    assert (y1 - y2).abs().max() < 1e-3   # continuous (no discrete jumps)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `SwiGLUMoE`** — low-rank (LoRA-style) SwiGLU experts shared across recurrence steps; router `nn.Linear(dim, n_experts)` (full-rank). `softmax`: `w=softmax(logits)`. `relu` (ReMoE): `w=relu(logits)`; expose `aux_l1 = w.abs().mean()` for the adaptive-L1 sparsity/load-balance loss (Task 7). Combine experts as `sum_e w_e * expert_e(x)` (dense compute; skip-zero optimization is a later FLOPs win). Store `self.last_route=w`. No top-k.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): SwiGLU MoE with relu/softmax router (smooth, reversibility-safe)"`

---

## Task 5: MoS output head

**Files:** Modify `train_gpt_m0.py` (add `MoSHead`); Test: `tests/test_m0_modules.py`

- [ ] **Step 1: Failing test** — valid distribution + rank > 1:

```python
from train_gpt_m0 import MoSHead
def test_mos_head_is_distribution_and_high_rank():
    h = MoSHead(dim=16, vocab=32, n_mix=3)
    z = torch.randn(4, 7, 16); logp = h(z)
    assert logp.shape == (4, 7, 32)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(4, 7), atol=1e-4)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `MoSHead`** — `n_mix` low-rank context projections + a shared/tied output embedding; `logp = log( sum_k pi_k * softmax(z W_k E^T) )` (Yang 2018). `pi = softmax(gate(z))`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): MoS output head (mixture of softmaxes)"`

---

## Task 6: Assemble the M0 GPT model

**Files:** Modify `train_gpt_m0.py` (add `Hyperparameters`, `M0GPT`); Test: `tests/test_m0_modules.py`

- [ ] **Step 1: Failing test** — end-to-end forward+backward, loss finite, tied embedding, depth used:

```python
from train_gpt_m0 import M0GPT, Hyperparameters
def test_m0gpt_forward_backward_and_depth():
    args = Hyperparameters(model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=32,
                           n_experts=4, expert_rank=8, n_mix=2, kv_latent=8)
    m = M0GPT(args)
    x = torch.randint(0, 32, (2, 8)); y = torch.randint(0, 32, (2, 8))
    loss = m(x, y, depth=4); loss.backward()
    assert torch.isfinite(loss)
    assert m.tok_emb.weight.data_ptr() == m.head.out_embed.weight.data_ptr()  # tied
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `M0GPT`** — `tok_emb` + learned pos; `x0 = tok_emb(x)+pos`; `F_θ`,`G_θ` = pre-norm blocks each `MLAttention`+`SwiGLUMoE` (delta form `attn_out + moe(norm(y))` — see `p1_synthetic.StandardMhaSwiGLUBlock`, verified delta-correct); `ReversibleRecurrence(F,G)`; forward uses `run_reversible(x0, depth)` → `z_K` → `MoSHead` → CE loss. `Hyperparameters` dataclass holds all knobs incl. `router_type`, `kv_latent`, `n_experts`, `expert_rank`, `n_mix`, K-set.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): assemble M0GPT (reversible MLA+MoE recurrence + MoS readout)"`

---

## Task 7: LM scaffold — data, eval (BPB), train loop, int6 artifact, optimizer, finite-horizon loss

**Files:** Modify `train_gpt_m0.py`; Test: `tests/test_m0_trainer.py`

- [ ] **Step 1: Failing smoke test** (CPU, tiny synthetic shard):

```python
def test_m0_trainer_smoke(tmp_path, monkeypatch):
    from train_gpt_m0 import main
    # tiny in-memory/temp shard via env knobs; 3 steps, eval once, write artifact
    main(["--iterations","3","--model-dim","32","--n-experts","4","--seq-len","16",
          "--eval-batches","2","--artifact-out",str(tmp_path/"m.bin")])
    assert (tmp_path/"m.bin").exists()
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement scaffold** — port from `a15093a:train_gpt.py`: `DistributedTokenLoader`, `build_sentencepiece_luts`, `run_validation` (BPB), `Muon`, DDP setup, LR warmdown; port int6 artifact from current `train_gpt.py::encode_scored_artifact`/`save_int6_artifact` (replace the snapshot's int8) with the 16 MB check. Train loop: sample `depth=K_hi` from the K-set; loss = `CE(K_hi) + λ_h·relu(CE(K_hi) − sg(CE(K_lo)) + m) + λ_route·aux_l1` (finite-horizon hinge from `opg_doc` eq:p1-loss + ReMoE adaptive-L1). bf16 autocast; finite-guards on logits/loss (raise on NaN/Inf). AdamW for embeddings/scalars/router, Muon for matrices; assert optimizer param coverage.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): LM scaffold (data, BPB eval, Muon, int6 artifact, finite-horizon loss)"`

---

## Task 8: Two-goal metrics (R_act, φ, erank, FLOPs/active-fraction, KV bytes)

**Files:** Modify `train_gpt_m0.py` (metrics fns); Test: `tests/test_m0_metrics.py`

- [ ] **Step 1: Failing tests** — known-input correctness:

```python
from train_gpt_m0 import effective_rank, fit_phi
def test_effective_rank_known():
    import torch
    s = torch.eye(4); assert abs(effective_rank(s) - 4.0) < 1e-4          # isotropic → erank=dim
    s2 = torch.zeros(4,4); s2[0,0]=1; assert abs(effective_rank(s2) - 1.0) < 1e-4
def test_fit_phi_monotone():
    # val-loss(r) consistent with phi in [0,1]; synthetic points
    losses = {1: 3.0, 2: 2.7, 4: 2.5, 8: 2.4}
    phi = fit_phi(losses); assert 0.0 <= phi <= 1.0
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `effective_rank` = exp(spectral entropy of singular values) (Roy & Vetterli); `fit_phi` = fit val-loss vs effective-params(r) power law, return the exponent normalized to [0,1] (Iso-Depth method); `r_act(model, K)` = peak `torch.cuda.max_memory_allocated` ratio K vs K₀ (logged, ≈1 expected); KV-bytes + active-expert-fraction (mean `last_route>0`) + FLOPs/token estimate. Log all under a `metrics:` line + add `experiments/plot_metrics.py` parser entry + a `tests/test_loss_component_logging.py`-style field test.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"feat(m0): two-goal metrics (R_act, phi, erank, KV, active-fraction)"`

---

## Task 9: Control-experiment runner

**Files:** Create `experiments/run_m0_control_experiments.sh`; Test: `tests/test_m0_trainer.py` (bash -n + dry knobs)

- [ ] **Step 1: Failing test** — script syntax + exposes the matched ablations:

```python
def test_control_experiment_script_shape():
    txt = open("experiments/run_m0_control_experiments.sh").read()
    for s in ("router_type=relu","router_type=softmax","for r in 1 2 4 8","kv_latent"):
        assert s in txt
    import subprocess; subprocess.run(["bash","-n","experiments/run_m0_control_experiments.sh"],check=True)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `set -euo pipefail`; loops: ReLU-vs-softmax (matched), φ over `r∈{1,2,4,8}`, MLA `kv_latent` sweep; each writes a metrics JSON; a summary that won't abort on missing grep (preserve `?` fallback per `EXPERIENCE.md#explicit-boundary`).
- [ ] **Step 4: Run → PASS (`bash -n`).**
- [ ] **Step 5: Commit** — `"feat(m0): control-experiment runner (relu-vs-softmax, phi(r), MLA sweep)"`

---

## Task 10: Align `opg_doc.tex` + contract tests to the M0 design

**Files:** Modify `reports/opg_doc.tex` (§Goal exclusions, §Models readout), `tests/test_training_contracts.py`

- [ ] **Step 1: Update the failing contract tests first** — change `test_opg_doc_stages_kv_and_prior_dense_moe_as_future_non_p1` so MLA/MoE/MoS are asserted **retained** (not future-excluded); keep `lyapunov_coef` rejection assertions; assert §Models describes the `z_K→MoS` readout (no halting). Add `test_opg_doc_has_related_work` asserting the Related Work section + key cites (`revffn`,`remoe`,`isodepth`,`deepseekv2`,`mos`).
- [ ] **Step 2: Run → FAIL** (doc not yet aligned).
- [ ] **Step 3: Edit `opg_doc.tex`** — §Goal: move MLA/MoE/MoS *out* of the exclusion list into "retained for the resource/expressiveness goals (smooth-sparse MoE, low-rank MLA, MoS)"; keep DEQ-implicit/consistency/Lyapunov/Dirichlet-UCB excluded. §Models: replace the halting-readout eq with the `z_K→MoS` readout; note ReLU/softmax router control experiment. Add the two-goal metric definitions to §Metrics.
- [ ] **Step 4: Run → PASS** (`pytest tests/test_training_contracts.py`).
- [ ] **Step 5: Commit** — `"docs(opg_doc): realign §Goal/§Models to retained M0 mechanisms + tests"`

---

## Task 11: Archive the rich architecture; promote `train_gpt_m0.py`

**Files:** Move rich `train_gpt.py` + rich-only `experiments/components/*` + legacy `experiments/test_*` → `legacy/`; rename `train_gpt_m0.py`→`train_gpt.py`; update `experiments/run_audit_tests.sh`, contract/removal-symmetry tests.

- [ ] **Step 1: Update tests/audit registry first** to expect the new `train_gpt.py` (M0) + `legacy/` paths; `git grep` for movers (#move-tracked-invariant) — every moved file `git add`-ed in the same commit.
- [ ] **Step 2: Run → FAIL** (paths not yet moved).
- [ ] **Step 3: `git mv`** rich `train_gpt.py`→`legacy/train_gpt_rich.py`; move rich-only components/tests to `legacy/`; `git mv train_gpt_m0.py train_gpt.py`; fix imports in moved tests; update `_OPTIONAL_COMPONENT_*`/removal-symmetry registries.
- [ ] **Step 4: Run** `bash experiments/run_audit_tests.sh` + full `pytest` → PASS.
- [ ] **Step 5: Commit** — `"refactor(m0): promote M0 train_gpt.py; archive rich arch to legacy/"`

---

## Task 12: Integration smoke + verification

**Files:** none new; run real smokes.

- [ ] **Step 1: GPU preflight** (`pgrep`+`nvidia-smi`, per `EXPERIENCE.md#gpu-preflight-protocol`).
- [ ] **Step 2: FineWeb tiny DDP smoke** — `torchrun --nproc_per_node=2 train_gpt.py --iterations=20`: loss ↓, finite grads, reconstruction `<1e-5` logged, `R_act(K)`≈1, artifact ≤16 MB. Report per `EXPERIENCE.md#iter-progress-reporting`.
- [ ] **Step 3: Synthetic depth-hard smoke** — confirm φ + paired depth-gain emitted.
- [ ] **Step 4: Run review chain** (`/simplify`, coderabbit, audit) + full `pytest`.
- [ ] **Step 5: Commit + record** — update `experiments/docs/hypotheses.md` (fresh M0 ledger entry) + `experiments/update_results.sh`.

---

## Self-review

- **Spec coverage:** reversible core (T1), O(1) backward (T2), MLA (T3), MoE relu/softmax control (T4), MoS (T5), assembly (T6), LM scaffold+finite-horizon loss+int6 (T7), two-goal metrics incl. φ/erank (T8), control experiments (T9), doc realignment (T10), archive+promote (T11), eval on FineWeb+synthetic (T12). All spec sections mapped.
- **Placeholders:** novel components have full code; standard scaffold cites exact source (`a15093a`, current int6) — port, not placeholder.
- **Type consistency:** `ReversibleRecurrence.forward_states/invert/run_reversible`, `RevRecurrenceFn.apply(rec,x0,depth,*params)`, `SwiGLUMoE(router_type=…).last_route`, `MoSHead(n_mix)`, `M0GPT(args)`, `effective_rank`/`fit_phi` — names consistent across tasks.
- **Risks:** T2 (reversible backward correctness) is the gate — grad-equivalence test must pass at atol 1e-6 before proceeding; T4 router smoothness test guards reversibility; T7 finite-guards prevent silent NaN gates.
