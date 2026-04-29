"""DEPRECATED — single-step micro-profiler.

Removed 2026-04-28 (Phase 9 cleanup, Item 3): the per-expert micro-instrumentation
in this script reached into bottleneck-era module names (`BottleneckIn` /
`ExpertMLABody` / `BottleneckOut` and the `expert_proj_rank` knob) that no longer
exist after the iter 96 full-D-LoRA promotion. Rather than maintain a parallel
micro-profiler that drifts from the production model, use the chrome-trace
wrapper which always exercises the SAME module the training loop builds.

For per-step throughput profiling, use ``experiments/profile_train.py``:

    torchrun --standalone --nproc_per_node=gpu experiments/profile_train.py \\
        --iterations=25 --profile-warmup=5 --profile-active=15

It wraps the real training loop in ``torch.profiler``, dumps a chrome trace,
and prints a top-N self-CUDA-time op summary.
"""
import sys


def main() -> int:
    sys.stderr.write(
        "profile_step.py is deprecated — per-expert micro-profiling assumed "
        "the bottleneck-era module layout (removed in iter 96). Use "
        "experiments/profile_train.py for throughput profiling.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
