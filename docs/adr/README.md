# Architecture Decision Records

This directory is the **design-history spine**: one immutable, numbered ADR per
major design *epoch*. Each ADR **defines the interface** the implementation must
conform to and records *why* the design changed (the issue the prior epoch's
experiments revealed).

## Conventions

- **Immutable + numbered.** An ADR is a dated snapshot of a decision. Do not
  rewrite a past ADR; to change a decision, add a new numbered ADR and mark the
  old one `Superseded by NNNN`.
- **Fixed template:** `Status` · `Issue` (what prior results revealed) ·
  `Decision` · `Interface` (the contract the code conforms to) · `Evidence`
  (link to results, never inlined) · `Consequences`.
- **Authority split.** [`CLAUDE.md`](../../CLAUDE.md) is the single binding
  *directive* (how the agent works). ADRs define the *implementation interface*.
  Raw experimental results live in
  [`experiments/docs/hypotheses.md`](../../experiments/docs/hypotheses.md) and
  `legacy/docs/hypotheses_archive.md`; ADRs link to them, never copy them.
- **Consistency.** A change to an ADR's Interface and the code that implements it
  must land in the same commit (ADR interface ↔ code).

## Index (track changes + reasons)

| ADR | Design epoch | Status | Key result | Issue that forced the next change |
|-----|--------------|--------|------------|------------------------------------|
| — | RevDEQ + dense-MoE rich architecture (legacy `train_gpt.py`) | Archived | best val_bpb ≈ 1.463 (iter172) | Mixed too many axes; depth utility not isolatable; file 7× over the 1500-line hard-stop |
| [0001](0001-finite-horizon-opg-main-path.md) | Finite-horizon OPG main path | Superseded by 0002 | finite-horizon training path landed | Prefix-anchor/multi-K consistency encouraged premature convergence; results still didn't localize the depth-gain failure |
| [0002](0002-final-minimal-p1-design.md) | Final minimal P1 (M0 + Tier-1 harness) | Accepted | S+0 positive-control depth-gain CI crosses zero (not promoted) | (open) detectability repair of the Tier-1 task / positive-control capacity |
| [0003](0003-repository-organization.md) | Repository organization (single `legacy/`, active-only tree, `results/`) | Accepted | active suite 470 passed; `experiments/` test-free; evidence has a tracked home | (open, Phase B) port the rich contract/audit/unit tests to M0 + re-point the audit registry |

Detailed run evidence for each row: `experiments/docs/hypotheses.md` (active) and
`legacy/docs/hypotheses_archive.md` (history).
