# ADR 0001: Finite-Horizon OPG Main Path

Date: 2026-06-02

## Status

Superseded by [ADR 0002: Final Minimal P1 OPG Design](0002-final-minimal-p1-design.md).

## Context

`reports/opg_doc.tex` reframes the project as finite-horizon recurrent OPG, not
as a model trained primarily to converge to one fixed point. The old
prefix-anchor consistency path improved historical fixed-point diagnostics but
can encourage premature convergence and reduce measured marginal utility from
deeper recurrence.

## Decision

Make finite-horizon OPG the active implementation path:

- Sample deep training depths with mass `{32: 0.20, 64: 0.40, 128: 0.40}`.
- Optimize task loss at `K_hi`.
- Add a one-sided scale hinge against a shallow stop-gradient endpoint.
- Keep router and expert-diversity regularizers as health terms.
- Reject legacy prefix anchors and positive multi-K consistency coefficients at startup.
- Report paired `G_T`, `NDR_epsilon`, and `HardGain` on identical validation sequences.
- Treat fixed-point diagnostics as advisory signals, not training pressure or
  finite-horizon promotion gates.
- Define effective recurrent depth through layout-agnostic realized-transition
  metrics (`ED_update`, `ED_logit`). Use route-depth NMI/utilization only as
  explanatory diagnostics; raw gate trajectories are not promotion metrics.
- Reject nonzero `lyapunov_coef` at startup; the refuted Lyapunov pressure
  path is not a valid launch-time substitute for diagnostic evidence.
- For constant-KV inference, start with a terminal hidden-state cache candidate
  and compare it against exact multi-depth cache before changing the training
  loss or model API.

Residual-aware routing is implemented with a reversible-safe detached
displacement-from-input router feature. True `z_k - z_{k-1}` routing is deferred
because RevDEQ backward reconstruction would otherwise depend on previous-state
history that is unavailable during inversion.

## Consequences

Fixed-point diagnostics remain useful health/fallback/cache-readiness
measurements, but they are not the main promotion gate. Future work that
restores fixed-point pressure must show a measured instability or terminal-cache
quality failure rather than assuming that global contraction maximizes
expressivity.
