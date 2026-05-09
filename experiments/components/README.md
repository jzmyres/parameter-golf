# Components — pluggable patches for `train_gpt.py`

Per user directive 2026-04-30: "The patch should be written in separate files
like components to be easily added / integrated later."

This directory holds active drop-in component modules that `train_gpt.py` can
import without pulling in stale experiment scaffolds. Each active component is
a focused single-purpose module that:

1. Is **standalone-importable** (imports only from PyTorch + stdlib).
2. Has **gated activation** via a module-level flag set from `Hyperparameters`.
3. Defaults to **OFF** so absence of the import in `train_gpt.py` is a no-op.
4. Carries **its own correctness tests** in `experiments/test_*.py`.

## Active Component Inventory

| Component | Status | Integration step | Tests |
|---|---|---|---|
| `fused_routed_down.py` | active optional path | imported by `train_gpt.py` when `use_unified_routed_down=True` | component smoke test |
| `chained_routing.py` | active optional path (design: [`../iter103_chained_routing_plan.md`](../iter103_chained_routing_plan.md)) | selected by `--chained-stages-preset=<preset>` | `experiments/test_chained_routing.py` |

## Archived Prototypes

Historical prototypes, duplicate scaffolds, and deferred ideas live in
`experiments/components/archive/`. They are retained for research reference but
are not active integration targets. If an archived component becomes active
again, move it back to this directory and update this inventory in the same
change.

## Integration pattern (cookbook)

For a new component `foo.py`:

1. **Write the component** — a single Python module with:
   - All Triton kernels / autograd functions / nn.Modules that the feature needs.
   - A module-level toggle (default False) and `set_foo_enabled(bool)` setter.
   - A test in `experiments/test_foo.py` that runs CPU-only where possible.
2. **Test the component** standalone (CPU first, GPU when free).
3. **Wire into `train_gpt.py`** by adding three things:
   - `from experiments.components.foo import set_foo_enabled, foo_helper`
   - `Hyperparameters.use_foo = False` field + `--use-foo` CLI flag.
   - In `main()`: `set_foo_enabled(args.use_foo)` BEFORE model construction.
   - In the consumer (`Block.forward`, `MLP.forward`, etc.): conditional
     dispatch on the flag.
4. **Smoke test** the integration end-to-end.

This keeps `train_gpt.py` lean (the canonical model file) and components
modular (new features land as new files, not 200-line additions to the
5000-line training script).

## Worktree usage (optional)

For substantial integrations that touch `train_gpt.py` heavily (e.g. iter
103 chained routing's Block refactor), develop on a separate branch via
`git worktree add ../parameter-golf-iter103 autoresearch/iter103-chained`
to leave the main branch's `train_gpt.py` untouched while the running
iter on `autoresearch/phase2-optimization` continues. Merge back via PR
or fast-forward once the integration smoke-tests green.
