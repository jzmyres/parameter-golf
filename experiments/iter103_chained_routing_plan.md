# iter 103 / H77 — Chained N-Stage Routing: Implementation Plan

**Status:** plan only · 2026-05-06
**Author intent:** drop-in component in `experiments/components/`, default OFF,
toggle-able via `Hyperparameters` + CLI, supports unified vs typed-chain (attn-first / mlp-first / split).

---

## 1. Hypothesis recap

The current Block has a **single unified routed-expert pool** of size `2 × num_routed`,
where the same `SoftDenseRouter` produces both attn weights `w_attn[..., :num_routed]`
and mlp weights `w_mlp[..., num_routed:]`. Both attn and mlp experts see the same
`h = RMSUnit(z + x_0)`.

H77 proposes splitting this pool into **N sequential stages**, each with its own router
and experts. Stage k's input is stage k-1's output, so later stages compute on
*transformed* representations rather than the raw block input. This adds a
*sequential routing-composition* axis that the unified pool can't represent.

Variants to test:
1. **`unified`** (current) — single mixed stage. Strict-gen recovery target.
2. **`attn_first_2stage`** — stage 0 = attn experts only, stage 1 = mlp experts only.
   Mimics standard transformer (attn → ffn) but inside the DEQ inner step.
3. **`mlp_first_2stage`** — flipped order: mlp first, then attn on mlp output.
4. **`split_2stage`** — H77 original spec: each stage has half attn + half mlp.
5. **`split_4stage`** — N=4 fine-grained chain (attn / mlp / attn / mlp), useful
   for VRAM-scaling tests.

---

## 2. Architecture spec

### 2.1 Stage I/O contract

Each stage k computes `Δ_k` from its (parameter-free RMS-unit-normed) input and
adds it residually:

```
stage_in_0 = z + x_0                       # standard block input
h_k        = RMSUnit(stage_in_k)           # parameter-free preconditioner
Δ_k        = AttnMix_k(h_k) + MlpMix_k(h_k)  # missing terms zeroed out
stage_in_{k+1} = stage_in_k + Δ_k          # residual chain
block_out  = Σ_k Δ_k + ParcaeInjection(x_0)  # unchanged Parcae path
```

**Why residual chain rather than replacement:** preserves the existing iter-100b
strict-gen path. With a single stage the chain reduces to `Δ_total = Δ_0` exactly.

### 2.2 Per-stage owned modules

Each stage k owns:
- `router_k: SoftDenseRouter` over its routed experts (attn slice + mlp slice).
- `attn_k: CausalSelfAttention` (only if `attn_total_k > 0`).
- `mlp_k: MLP` (only if `mlp_total_k > 0`).
- `shared_gate_attn_k: nn.Linear` + `shared_gate_norm_weight_attn_k: Parameter`
  (only if `shared_attn_k > 0`).
- `shared_gate_mlp_k: nn.Linear` + `shared_gate_norm_weight_mlp_k: Parameter`
  (only if `shared_mlp_k > 0`).
- `attn_post_mix_norm_k`, `mlp_post_mix_norm_k`: per-stage RMSNorms.

Stages with zero attn or zero mlp simply skip those terms — the corresponding
`Δ_k` is partial (e.g., attn-only stage emits `Δ_k = AttnMix_k(h_k)` and no mlp
contribution).

### 2.3 Param independence rule

No parameter is shared across stages. Each stage's router/attn/mlp/shared-gate
weights are constructed independently. This is enforced by a unit test
(`test_no_param_sharing_across_stages`) that walks `named_parameters()` and
verifies each `id(p)` appears in at most one stage.

### 2.4 Reversibility / RevDEQ contract

Block output is still `B̄ ⊙ RMSUnit(x_0) ⊙ x0_inject_norm_weight + Δ_total`, where
`Δ_total = Σ_k Δ_k`. The fixed-point structure is unchanged: `T_θ(z, x_0)` is still
a single function of `(z, x_0)` with the same Parcae injection signature, so:

- `RevDEQFunction.apply(...)` keeps the same input list.
- The reverse pass (fp64 add/sub reconstruction) sees the same `B̄`, `β`,
  `Δ_total` it does today.
- `parcae_reversibility_floor` semantics are unchanged.

**No backward-only floor** is introduced anywhere in the chain.

### 2.5 Activation-memory analysis

**Forward peak (per stage k):**
```
peak_k ≈ B·T·(E_attn_k + E_mlp_k)·D
```
With sequential execution and proper buffer release (no retained intermediates
between stages), the block-level forward peak is:
```
peak_block = max_k(peak_k)   instead of   Σ_k peak_k
```
For the unified pool: `peak = B·T·30·D`. For `attn_first_2stage` (15 + 15 typed):
`peak = B·T·15·D` — **halved.** For `split_4stage` (4+4+4+4): peak quartered.

**Backward peak:** autograd retains all stage-k tensors needed for chain rule, so
backward holds `Σ_k peak_k` of saved tensors. The win is **forward-only**. Useful
for VRAM-bounded scaling of `num_experts` per stage *during forward*, but doesn't
halve total training memory.

**Combined with RevDEQ:** the per-DEQ-iter forward peak is the bottleneck inside
the K-loop. Halving forward peak doubles the room for `K` jitter, `num_experts`,
or `seq_len` *for forward-only memory cost.* Backward TBPTT memory is bounded by
`deq_bptt_k`, which is a separate axis.

---

## 3. File layout

```
experiments/components/
  chained_routing.py        # NEW — the component (≈350 LOC)
  README.md                 # update inventory table + integration cookbook
  __init__.py               # add export

experiments/
  test_chained_routing.py   # NEW — component tests (≈250 LOC, 21+ unit tests)

train_gpt.py                # MODIFIED in 7 surgical sites (≈40 LOC delta)
```

### 3.1 `experiments/components/chained_routing.py`

**Public API:**
```python
@dataclass(frozen=True)
class StageSpec:
    attn_experts: int = 0      # routed
    mlp_experts: int = 0       # routed
    shared_attn: int = 0
    shared_mlp: int = 0
    def attn_total(self) -> int: ...
    def mlp_total(self) -> int: ...

PRESETS: dict[str, list[dict]] = {
    "unified": [...],
    "attn_first_2stage": [...],
    "mlp_first_2stage": [...],
    "split_2stage": [...],
    "split_4stage": [...],
}

def parse_stages(preset_or_json: str | None) -> list[StageSpec] | None:
    """None / "" / "none" → None (caller falls back to unified Block.forward).
    Else parses preset name or JSON list-of-dicts. Raises ValueError on bad input."""

def set_chained_routing_enabled(enabled: bool) -> None: ...
def chained_routing_enabled() -> bool: ...

class ChainedExpertStack(nn.Module):
    def __init__(
        self, *, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
        kv_latent_dim, attn_expert_rank, mlp_expert_rank, stages,
        router_scoring="linear", router_pertoken_entropy_coef=0.0,
        use_entmax_routing=False, entmax_blend_init_logit=5.0,
        use_nsa_attention=False, nsa_*=...,
        # CLASS INJECTION (avoids circular import):
        soft_dense_router_cls, causal_self_attention_cls, mlp_cls, rms_norm_cls,
    ): ...
    def forward(self, z_in: Tensor, x0: Tensor, rms_unit_fn) -> Tensor:
        """Returns Δ_total. Block.forward owns the Parcae x_0 injection."""
```

**Class injection** (the four `*_cls` kwargs) is the standard pattern in
`experiments/components/` to avoid circular imports between component code and
`train_gpt.py`. The component file imports nothing from `train_gpt`; `train_gpt`
passes the existing class objects in.

**Forward implementation outline:**
```python
def forward(self, z_in, x0, rms_unit_fn):
    cumulative_state = z_in
    delta_total = torch.zeros_like(z_in)
    for k, spec in enumerate(self.stages):
        h = rms_unit_fn(cumulative_state + x0)
        w_attn, w_mlp = self._route(self.routers[k], h, spec)  # None if absent
        attn_mix = self._attn_mix(self.attns[k], h, w_attn, spec, ...)  # None if absent
        mlp_mix  = self._mlp_mix(self.mlps[k], h, w_mlp, spec, ...)
        stage_delta = 0
        if attn_mix is not None: stage_delta = stage_delta + self.attn_post_norms[k](attn_mix)
        if mlp_mix  is not None: stage_delta = stage_delta + self.mlp_post_norms[k](mlp_mix)
        cumulative_state = cumulative_state + stage_delta
        delta_total = delta_total + stage_delta
    return delta_total
```

### 3.2 `experiments/test_chained_routing.py`

Test classes (21+ tests, all CPU):

| Class | Tests |
|---|---|
| `TestStageSpec` | validate negative counts; reject empty stage; `attn_total`/`mlp_total` correctness |
| `TestParseStages` | None/""/"none" → None; each preset round-trips; JSON pass-through; invalid input raises |
| `TestModuleToggle` | default OFF; `set_chained_routing_enabled(True/False)` round-trip |
| `TestChainedStackForwardShapes` | for each of {unified, attn_first, mlp_first, split_2stage}: `forward` returns `(B,T,D)`; `loss.backward()` produces non-zero grad in *every* stage |
| `TestStageParameterIndependence` | walk `named_parameters()` of `split_2stage`; assert no `id(p)` appears in two stages |
| `TestUnifiedPresetParamCount` | unified preset constructs exactly 1 attn + 1 mlp + 1 router; expert counts match canonical 16/16 |
| `TestTypedChainStageFlow` | attn-first: stage 0 has attn-only, stage 1 has mlp-only; mlp-first: mirror |

Tests use small dims (`dim=32, T=4, num_heads=4, num_kv_heads=2`) on CPU to stay
fast (<10s for the suite).

### 3.3 `train_gpt.py` changes (surgical, ≈40 LOC delta total)

1. **`Hyperparameters`** (line ~520):
   - Add: `chained_stages_preset: str | None = None`
   - Update existing comment block on `use_chained_routing` to point at the preset flag.
2. **`_CLI_TUNABLE_KNOBS`** (line ~656): add `"chained-stages-preset"`.
3. **`Block.__init__`** signature (line ~3199): accept `chained_stages_preset: str | None = None`.
4. **`Block.__init__`** body (after `self.mlp = MLP(...)`):
   - `self.chained_stack = None`
   - If preset is set: lazy-import `ChainedExpertStack` + `parse_stages`, build
     the stack, store on `self.chained_stack`. The unified Block modules are
     *still constructed* (so toggling preset off doesn't lose state-dict keys).
5. **`Block.forward`** (line ~3429): early-dispatch: if `self.chained_stack is
   not None`, compute `delta = self.chained_stack(z_in, x0, _rms_unit)`, apply
   Parcae injection, return — skipping the unified path.
6. **`GPT.__init__`** signature + body: thread `chained_stages_preset` through to
   the `Block(...)` call.
7. **`main()`** (line ~5050): replace the existing `use_chained_routing`
   NotImplementedError guard with a deprecation message redirecting to
   `--chained-stages-preset=<name>`. Call `set_chained_routing_enabled(...)`
   before model construction.
8. **`GPT(...)` call site in main** (line ~5328): add
   `chained_stages_preset=getattr(args, "chained_stages_preset", None)`.

**Strict-gen invariant:** with `chained_stages_preset=None`, every line above is
a no-op or default-None — `Block.forward` runs the original code path, no params
are added, no behavior changes. This is how the `124 / 124` existing test suite
must keep passing.

### 3.4 `experiments/components/__init__.py` + `README.md`

- `__init__.py`: add `from . import chained_routing  # noqa`.
- `README.md` inventory table: add row for `chained_routing.py` with status
  "default-off optional path".

---

## 4. Test budget + integration smokes

### 4.1 New unit tests
- 21+ component tests — see §3.2.

### 4.2 Existing test suite
- All 125 / 125 must still pass with `chained_stages_preset=None` (current
  default). One known-affected test: `experiments/test_cli_parser.py::TestCliParser::test_default_parity` —
  **expected to need update** (the parity table includes the full CLI knob list,
  so adding `chained-stages-preset` requires extending the expected dict).
  Update is mechanical (one new key with default `None`).

### 4.3 Integration smokes (CPU)
- `--chained-stages-preset=unified` + `--iterations=2`: smoke that the dispatch
  path runs end-to-end through `RevDEQFunction`. CPU is fine since this is
  correctness, not speed.
- Each non-unified preset (`attn_first_2stage`, `mlp_first_2stage`,
  `split_2stage`): same 2-iteration CPU smoke.

### 4.4 Integration smoke (GPU, post-CPU pass)
- `--chained-stages-preset=attn_first_2stage --iterations=20` on 2× L40S.
  Verify: no NaN, loss descends, K-sweep flat at K≥16, artifact ≤ 16 MB,
  step time within 1.3× of unified (the chain has slightly more wrapping
  overhead due to per-stage routers).

---

## 5. Diagnostics & logging

### 5.1 Per-stage telemetry

The current Block emits `attn_cv`, `mlp_cv`, `pool_cv`, `attn_entropy`, etc. as
a single set of values for the unified router. For chained the natural extension
is per-stage:

```
attn_cv_s0:0.45 mlp_cv_s0:0.30 attn_cv_s1:0.20 mlp_cv_s1:0.15 ...
```

**Decision:** in the first implementation, log the existing keys (`attn_cv`,
etc.) as the **summed-across-stages** value (or stage-0 for typed chains where
later stages don't have that side). This keeps `experiments/plot_metrics.py`
unchanged. Per-stage breakdown can be added in a follow-up — flagged as a
known limitation in the plan, not a blocker.

### 5.2 Diagnostic gate prescriptions

The `router_collapse` / `expert_collapse` gates currently inspect the single
pool. For chained, each stage's router has its own `_cv_loss_raw` and
`_pertoken_entropy_loss`. The gate aggregator (`_collect_routing_losses`) needs
to walk all routers in the stack, not just `self.router`. **This is a wiring
change**, not a behavior change — all tensors already exist per-router.

---

## 6. Risk matrix

| Risk | Severity | Mitigation |
|---|---|---|
| RevDEQ reverse-pass mismatch with multi-stage Δ_total | high | Δ_total is still a function of `(z_in, x0, b_bar)` only → reverse pass sees the same inputs. Unit test: numerical equivalence of forward+backward at unified preset vs. existing Block. |
| `torch.compile` recompiles per-stage | medium | Each stage uses the same ops as Block today; dynamo should fuse the loop. If recompile pressure appears, mark `ChainedExpertStack.forward` `@dynamo_disable` (mirror the existing `@dynamo_disable` helpers). |
| DDP buckets straddle stages | medium | DDP buckets parameters by registration order; `nn.ModuleList` registration is deterministic. No special handling needed. |
| State-dict keys change for chained vs unified | low | Old checkpoints are not expected to load into a chained model — this is a new architecture, not a config tweak. Preset=None recovers the old key set. |
| Per-stage loss aggregation correctness | medium | Add a test that asserts `_router_reg_loss = Σ_k coef × per-stage-loss` matches a hand-computed reference. |
| Activation-VRAM win is theoretical | low | Add a profiling smoke that compares forward peak under `unified` vs `attn_first_2stage` at the same total expert count. Don't claim VRAM win in the commit message until measured. |

---

## 7. Implementation steps (recommended order)

| Step | Deliverable | Why this order |
|---|---|---|
| 1 | `experiments/components/chained_routing.py` skeleton (no train_gpt changes). | Component is standalone-importable; can be unit-tested in isolation. |
| 2 | `experiments/test_chained_routing.py` | TDD for component before integration. Catch class-injection / shape bugs. |
| 3 | Run new tests — all pass on CPU. | Confidence gate before touching `train_gpt.py`. |
| 4 | `train_gpt.py` Hyperparameters + CLI flag (no Block change). | Smallest delta first; verify --help works and `test_cli_parser` updates land. |
| 5 | `train_gpt.py` Block.__init__ + Block.forward dispatch. | The actual behavior change. With preset=None, no observable difference. |
| 6 | Run full existing test suite (125 tests). | Verify strict-gen recovery: nothing breaks at default config. |
| 7 | CPU integration smoke (2-iter run with each preset). | Catch DDP-free correctness issues. |
| 8 | GPU smoke (20-iter run with `attn_first_2stage`). | Verify GPU + RevDEQ + autocast end-to-end. |
| 9 | `experiments/components/README.md` inventory + integration cookbook update. | Doc closing the loop. |
| 10 | `experiments/hypotheses.md` queue: add iter 103 row with run plan. | Research ledger sync (per CLAUDE.md standing directive). |

---

## 8. Out of scope (explicit)

- **Per-stage loss diagnostics.** First version logs aggregated values; per-stage
  CV / entropy / usage emission lands as a follow-up after the first comparison
  run shows whether chained routing is worth the diagnostic complexity.
- **State-dict migration from unified → chained.** Old checkpoints don't load
  into a chained model. Acceptable because we're testing a new architecture, not
  evolving an existing one.
- **VRAM win measurement.** The plan claims forward-peak halving on first
  principles. Actual measured win goes in the iter 103 hypotheses entry, not
  the implementation commit.
- **`split_4stage` GPU run.** The 4-stage preset is supported and unit-tested,
  but the first GPU comparison run is `attn_first_2stage` vs `unified` (highest
  signal-per-cost).

---

## 9. Hypotheses-log entry (draft, lands with iter 103 run)

```markdown
| iter 103 / H77 | Chained N-stage routing component (attn_first_2stage). | TBD |
val_bpb …; vs unified preset (controlled comparison at same total expert count
and same wall budget). Forward-peak VRAM measured at … vs unified ….
Routing health per-stage: stage 0 attn_cv=…, stage 1 mlp_cv=….
| TBD |
```

The actual claim depends on what the run produces. The pre-registered
*hypothesis* is: chained typed-stage routing at iso-expert-count adds a routing-
composition axis that improves val_bpb (or matches it at lower forward VRAM).

---

## 10. Estimated effort

| Phase | LOC | Wall time (eng) |
|---|---|---|
| Component code (`chained_routing.py`) | ~350 | 2–3 h |
| Component tests (`test_chained_routing.py`) | ~250 | 1–2 h |
| `train_gpt.py` integration (8 sites) | ~40 | 1 h |
| Test suite update (cli_parser parity) | ~5 | 15 min |
| CPU + GPU integration smokes | n/a | 20 min compile + 8 min × 4 presets |
| Doc updates | ~50 | 30 min |

**Total:** roughly half a day of focused implementation, plus the actual GPU
comparison run (1000 steps × ~24 s/step = ~6.5 h on 2× L40S).

---

## 11. Notes on prior implementation attempt (rolled back 2026-05-06)

I drafted the component file + test file and wired the integration before the
"plan only" directive arrived. All 21 component tests passed on CPU, and 145 of
146 in the full suite (the one failure was the expected
`test_cli_parser.py::test_default_parity` parity-table extension). The work
was **fully reverted**; the 125 / 125 baseline test count is restored. The plan
above describes that implementation, not an unrelated design.
