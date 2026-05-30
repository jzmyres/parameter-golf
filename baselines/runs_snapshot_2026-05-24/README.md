# Frozen evidence snapshot — 2026-05-24

`baselines/runs/` is gitignored and regenerable, so the numbers in
`../REPLICATION_REPORT.md` are backed by this tracked snapshot instead. These
files are the surviving authoritative artifacts from the 2026-05-24 review pass
on 2× NVIDIA L40S GPUs.

| File | Provenance |
|---|---|
| `smoke_report.json` | Intact 2026-05-24 source smoke (13 `smoke_passed`, 10 `blocked_no_code`, 2 `adapted_eval_run`). Authoritative. |
| `ddp_smoke_active_retry.json`, `ddp_smoke_deq_retry.json`, `ddp_smoke_huginn_retry.json` | 2026-05-24 2-GPU (`gpu_count=2`) per-baseline DDP smokes; `active-revdeq-parcae`, `deq-locuslab`, `huginn` `ddp_smoke_passed`. |
| `train_gpt_comparable_10step_report.json` | Holds the 5 surviving 2026-05-24 native passes (merged by id). The `parcae-fixed-k16` row was overwritten to `blocked_hardware` by a later 0-GPU login-node rerun (see caveat below). |

## Caveat: partially-lost evidence

On 2026-05-30 the runners were re-invoked on a CPU-only SLURM **login node**. With
no `nvidia-smi`, `gpu_count()` returned 0 and the partial reruns overwrote the
consolidated `runs/ddp_smoke_report.json` (now a single `blocked_hardware` row)
and the `parcae-fixed-k16` row in the native report. The full consolidated
12/13 `ddp_smoke_passed` run is therefore reproducible only on a 2-GPU node; this
snapshot preserves the source smoke, the three per-baseline DDP retries, and the
five surviving native rows. This incident motivated the
`#committed-report-tracked-evidence` and `#smoke-must-exercise-target` rules in
`EXPERIENCE.md`.
