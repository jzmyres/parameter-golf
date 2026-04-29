"""Profile train_gpt.py for ~20 training steps to identify bottlenecks.

Wraps the main training loop in torch.profiler. Captures CPU + CUDA + stack
traces. Dumps a chrome trace and prints a top-30 op summary by self-CUDA-time.

Usage:
    conda run -n opg torchrun --standalone --nproc_per_node=gpu \
        experiments/profile_train.py --iterations=25 --profile-warmup=5 \
        --profile-active=15 > profile.log 2>&1

Output:
    experiments/profile_chrome_trace.json  (load in chrome://tracing)
    profile.log                             (top-N op table to stdout)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so we can import train_gpt's main.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

# Override iterations small + skip wallclock to keep the profile run short.
# We patch sys.argv so train_gpt's argparse picks up these.
if "--iterations" not in " ".join(sys.argv):
    sys.argv += ["--iterations=25"]

# Profiler config: warmup=5 steps (let dynamo finish compiling), active=15 steps
# (the actual measured window), repeat=1. The wait phase is 0.
PROFILE_WARMUP = int(os.environ.get("PROFILE_WARMUP", "5"))
PROFILE_ACTIVE = int(os.environ.get("PROFILE_ACTIVE", "15"))
TRACE_PATH = str(_ROOT / "experiments" / "profile_chrome_trace.json")
SUMMARY_TOP_N = int(os.environ.get("PROFILE_TOP_N", "30"))

import torch
from torch.profiler import profile, ProfilerActivity, schedule
import train_gpt  # type: ignore

# Patch train_gpt.main — wrap its training step with a profiler step boundary.
# Strategy: use a global profiler handle and call .step() at every train log
# emission point. To avoid hooking into all the internals, we monkey-patch
# torch.cuda.synchronize() to also call profiler.step() — every train step
# already has a synchronize call after gradient-accumulation completes.
#
# Cleaner alternative: use `with profile(...)` wrapping main(). The schedule
# auto-advances based on torch.profiler.step() calls — for a step-driven
# trace we need to call .step() once per training step. Simplest hook: wrap
# the gradient accumulation loop. But that's deep in main(). Use synchronize.

_profiler: profile | None = None


def _make_step_callback():
    """Returns a function that, when called once per training step,
    advances the profiler schedule. We rely on the fact that train_gpt.py
    calls torch.cuda.synchronize() at predictable points."""
    state = {"step": 0}

    def _on_step():
        if _profiler is not None:
            try:
                _profiler.step()
            except Exception as e:
                print(f"[profile] step callback failed: {type(e).__name__}: {e}",
                      flush=True)
        state["step"] += 1

    return _on_step


# Hook torch.nn.utils.clip_grad_norm_: train_gpt calls it exactly once per
# training step (line 4318), right before the per-optimizer .step() loop.
# This gives clean per-step boundaries for the profiler's schedule. Hooking
# log0 would also work but it's a local function inside train_gpt.main(),
# not a module attribute, so it's not patchable from outside.
import torch.nn.utils as _torch_nn_utils

_real_clip_grad_norm = None


def _hooked_clip_grad_norm(*args, **kwargs):
    out = _real_clip_grad_norm(*args, **kwargs)
    if _profiler is not None:
        try:
            _profiler.step()
        except Exception as e:
            print(f"[profile] step callback failed: {type(e).__name__}: {e}",
                  flush=True)
    return out


def main_with_profile():
    global _profiler, _real_clip_grad_norm
    print(f"[profile] starting profiler: warmup={PROFILE_WARMUP} active={PROFILE_ACTIVE}",
          flush=True)
    # PROFILE_SKIP_KSWEEP=1 (default) tells train_gpt to early-exit AFTER the
    # train loop, BEFORE the int6 roundtrip + K-sweep. The OOM-prone Hutchinson
    # / Lipschitz probes (task #102) drag the profile run by 10+ min without
    # contributing any training-step ops. The env var is checked by ALL ranks
    # in train_gpt.main right at the post-train boundary (see L4574-area), so
    # all ranks exit together — no NCCL hang.
    os.environ.setdefault("PROFILE_SKIP_KSWEEP", "1")

    sched = schedule(wait=0, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE, repeat=1)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        _profiler = prof
        _real_clip_grad_norm = _torch_nn_utils.clip_grad_norm_
        _torch_nn_utils.clip_grad_norm_ = _hooked_clip_grad_norm
        try:
            train_gpt.main()
        except SystemExit:
            pass
        finally:
            _profiler = None
            _torch_nn_utils.clip_grad_norm_ = _real_clip_grad_norm

    # Export chrome trace + summary on rank 0 only (avoid duplicate writes).
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0:
        try:
            prof.export_chrome_trace(TRACE_PATH)
            print(f"[profile] chrome trace written to {TRACE_PATH}",
                  flush=True)
        except Exception as e:
            print(f"[profile] export_chrome_trace failed: {type(e).__name__}: {e}",
                  flush=True)
        try:
            print("=" * 100)
            print(f"TOP {SUMMARY_TOP_N} OPS BY SELF CUDA TIME")
            print("=" * 100)
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=SUMMARY_TOP_N,
                top_level_events_only=False,
            ))
            print("=" * 100)
            print(f"TOP {SUMMARY_TOP_N} OPS BY SELF CPU TIME")
            print("=" * 100)
            print(prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=SUMMARY_TOP_N,
                top_level_events_only=False,
            ))
        except Exception as e:
            print(f"[profile] table failed: {type(e).__name__}: {e}",
                  flush=True)


if __name__ == "__main__":
    main_with_profile()
