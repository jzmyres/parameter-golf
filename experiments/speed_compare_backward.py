"""DEPRECATED — speed comparison between deq_backward modes.

Removed 2026-04-28 (Phase 9 cleanup, Item 4 + Item 2): the ``deq_backward``
knob was eliminated entirely; only ``revdeq`` is supported. The
``unroll`` / ``autograd`` backward modes (which this script compared
against ``revdeq``) no longer exist. There is nothing to compare.

For per-step throughput profiling, use ``experiments/profile_train.py``
which wraps the main training loop in ``torch.profiler`` and emits a
chrome trace.
"""
import sys


def main() -> int:
    sys.stderr.write(
        "speed_compare_backward.py is deprecated — deq_backward modes were "
        "removed (only revdeq remains). Use experiments/profile_train.py "
        "for throughput profiling.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
