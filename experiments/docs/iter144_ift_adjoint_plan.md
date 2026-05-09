# iter144 IFT Adjoint Plan

## Goal

Test a default-off implicit-function-theorem (IFT) adjoint for the RevDEQ
backward pass. The goal is to recover a better gradient signal for the
truncated input/embedding path without changing the forward fixed-point solve,
the reversible storage contract, or the active default training behavior.

## Current Gradient Issue

The forward solve runs a finite RevDEQ iteration and the default backward uses
truncated BPTT over the last `deq_bptt_k` reconstruction steps. When
`deq_bptt_k < K`, the backward pass does not propagate task gradient through
the early solver states. This is memory-efficient and reversible, but it cuts
part of the input-state gradient. IFT is the principled DEQ correction for the
limit case where the terminal state is treated as an equilibrium.

## Mathematical Contract

Let the transition body be:

```text
z_star = T_theta(z_star, x0, b_bar)
```

For a loss `L(z_star)`, the implicit adjoint solves:

```text
lambda = dL/dz_star + J_T(z_star)^T lambda
```

where `J_T` is the Jacobian of `T_theta` with respect to `z`. In code, solve
this with a bounded Neumann/VJP iteration:

```text
lambda_0 = dL/dz_star
lambda_{i+1} = dL/dz_star + vjp_z(T_theta, z_star, lambda_i)
```

After `deq_ift_iters` iterations, use one final VJP of
`T_theta(z_star, x0, b_bar)` with cotangent `lambda` to produce gradients for
`x0`, `b_bar`, and the parameters of `T_theta`.

## Default-Off Interface

Add exactly one training knob:

```text
deq_ift_iters = 0
```

- `0` preserves the current truncated-BPTT backward bit-for-bit at the control
  flow level.
- `>0` enables pure IFT backward for the RevDEQ fixed-point output.
- The implementation should thread this through `Hyperparameters`, CLI
  parsing, `GPT.__init__`, `_deq_solve`, and `RevDEQFunction.apply`.

Do not change default `deq_bptt_k`, K-jitter, Parcae defaults, or forward solve
behavior in this iteration.

## Parcae Beta Compatibility

IFT is compatible with the current Parcae RevDEQ architecture, but pure IFT
does not give a task gradient to the solver relaxation beta.

Parcae has two distinct roles:

- `b_bar = Delta * B` is an input to the transition map `T_theta`; it remains
  part of the implicit gradient and should receive task gradient.
- `beta = 1 - A_bar` controls the solver blend:

```text
z_{k+1} = (1 - beta) z_k + beta T_theta(z_k, x0, b_bar)
```

At a true fixed point with `beta > 0`, the equilibrium condition reduces to
`z_star = T_theta(z_star, x0, b_bar)`. The equilibrium itself is independent of
the solver relaxation beta. Therefore pure IFT should return `grad_beta=None`
for task loss. Beta can still be tuned by explicit stability/solver objectives,
but not by the pure equilibrium task gradient.

Do not mix TBPTT beta gradients into the first IFT experiment. A hybrid beta
gradient is possible, but it changes the estimator and risks double-counting
finite-K solver effects.

## Implementation Shape

In `RevDEQFunction.backward`, branch on `deq_ift_iters`:

- If `deq_ift_iters == 0`, run the existing truncated-BPTT path unchanged.
- If `deq_ift_iters > 0`, treat the saved terminal state as `z_star` and run
  the pure IFT path.
- Use detached cloned leaves for saved tensors that require gradients during
  backward, especially `b_bar`, to avoid storage aliasing across VJP calls.
- Keep all adjoint iterations on device. Do not introduce hot-path CPU syncs.
- Return zero/`None` gradients for solver-only inputs as appropriate:
  `z_init_grad = 0`, `grad_beta = None`, and `None` for non-tensor config
  arguments.

Add lightweight diagnostics on the unwrapped module:

```text
_deq_ift_iters_last_bwd
_deq_ift_adj_norm_last_bwd
_deq_ift_adj_rel_last_bwd
```

These should be optional diagnostics only and must not affect promotion logic.

## Test Plan

- Default-off regression: `deq_ift_iters=0` follows the same backward branch and
  preserves existing RevDEQ gradient tests.
- Linear fixed-point test: for `T(z, x) = A z + W x` with spectral radius below
  1, compare the IFT gradient to the closed-form implicit gradient.
- Parcae slot test: with IFT enabled, `b_bar` receives gradient and solver
  `beta` does not receive task gradient.
- Integration smoke: tiny GPT forward/backward with `deq_ift_iters=1` and `2`
  on CPU or the smallest available CUDA shape.
- CLI/default test: parser exposes `--deq-ift-iters`, default is `0`, and saved
  metadata records the configured value.

## Promotion Criteria

Run iter144 only after the current root-cause rescue/contraction queue is not
blocked. Compare equal-step and equal-wallclock results against the current
default. Promote only if final full-validation BPB improves or is neutral within
the active promotion tolerance and wallclock cost is justified by task
performance. If it only improves diagnostics without BPB, keep it default-off.
