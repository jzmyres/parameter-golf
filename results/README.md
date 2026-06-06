# Results — frozen run evidence

Durable, **tracked** home for per-run evidence. Training logs themselves are
gitignored and ephemeral (background-task stdout lands in session-scoped
`/tmp`, never the repo), so a result that a committed doc cites must be frozen
here as a small snapshot — never left as a regenerable/ephemeral artifact
(CLAUDE.md *committed-report tracked-evidence* rule).

## Convention

One directory per run that is worth citing:

```
results/<YYYY-MM-DD>_<short_tag>/
├── summary.md        # config + final metrics table + the K-sweep (human-readable)
└── run_metrics.txt   # the trimmed metric lines from the run (the raw evidence;
                      #   no weights, no full log — just the gate/diagnostic lines)
```

- `summary.md` is the citable artifact: it states the exact config, the
  two-goal metric outcomes, and the depth/extrapolation sweep.
- `run_metrics.txt` is the frozen raw evidence those numbers resolve to.
- The `experiments/docs/hypotheses.md` ledger links here for full numbers;
  this directory holds the evidence, the ledger holds the verdict.

This is the same pattern as `baselines/snapshots/*.json` (frozen evidence) and
`records/` (frozen official submissions), applied to development runs.
