# Components — pluggable patches for `train_gpt.py`

Per user directive 2026-04-30: "The patch should be written in separate files
like components to be easily added / integrated later."

This directory holds drop-in component modules that future iters can import
from `train_gpt.py` rather than inlining substantial new logic. Each
component is a focused single-purpose module that:

1. Is **standalone-importable** (imports only from PyTorch + stdlib).
2. Has **gated activation** via a module-level flag set from `Hyperparameters`.
3. Defaults to **OFF** so absence of the import in `train_gpt.py` is a no-op.
4. Carries **its own correctness tests** in `experiments/test_*.py`.

## Component inventory

| Component | Status | Integration step | Tests |
|---|---|---|---|
| `triton_entmax.py` | TODO extract | already-inlined in train_gpt.py (iter 117b-2 X1 commit `81b5404`) | `experiments/test_entmax_triton.py` |
| `sparse_dispatch.py` | TODO extract | already-inlined in train_gpt.py (iter 117b-3 X2 commit `93c4d6c`) | `experiments/test_sparse_dispatch.py` (Phase A.0–A.5 PASS) |
| `chained_block.py` | TODO new | iter 103 / H77 X3 step 3+ | TODO new |

The "TODO extract" entries refer to code already committed in `train_gpt.py`
that COULD be moved here for cleanliness; doing so is optional and pure
refactor (no behavioral change). The "TODO new" entries are the unfinished
pieces that should be written here FIRST, then integrated into
`train_gpt.py` once tested.

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
