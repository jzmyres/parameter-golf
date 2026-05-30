"""Shared helpers for the baseline smoke/sweep runners.

`ddp_smoke_baselines.py` and `train_gpt_comparable_sweep.py` both need GPU
detection, subprocess-with-timeout plumbing, failure classification, and a JSON
report envelope. Per the project's sibling-fanout DRY gate (>=3 shared sites
across >=3 files), those live here as a single source of truth so the two
runners cannot drift.

Note on failure-gate policy: the two runners intentionally classify failures
the same way (via `classify_failure`) but gate differently. `ddp_smoke` exercises
*external* code, where a missing dependency is an expected "blocked" state; the
native sweep exercises *this* repo's `train_gpt.py`, where the same signal is a
real failure. The shared classifier removes the drift; the divergent gate is
documented at each call site.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Anchored so it only matches a runner's own rank-0 `step:N/M ...` line, never an
# echoed command string or an upstream library log that mentions "step:".
STEP_RE = re.compile(r"^step:(\d+)/(\d+)", re.M)

# nvidia-smi can wedge on a contended/broken driver; never block a sweep forever.
_NVIDIA_SMI_TIMEOUT_S = 15

# Number of trailing non-empty log lines inspected for failure classification.
# Keying on the terminal traceback (not a whole-log substring scan) avoids
# mislabelling a real crash as "blocked" just because the word "ImportError"
# appears somewhere upstream in the log.
_CLASSIFY_TAIL_LINES = 40


def gpu_count() -> int:
    """Return the number of visible GPUs.

    A genuinely absent `nvidia-smi` (FileNotFoundError) means zero GPUs and is
    returned as 0. A driver that errors or hangs is NOT silently reported as
    zero: we warn on stderr so a wedged driver surfaces instead of masquerading
    as a benign "0 GPUs / blocked_hardware".
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL, timeout=_NVIDIA_SMI_TIMEOUT_S
        )
    except FileNotFoundError:
        return 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"WARNING: nvidia-smi failed ({type(exc).__name__}); treating as 0 GPUs", file=sys.stderr)
        return 0
    return len([line for line in out.splitlines() if line.strip().startswith("GPU ")])


def classify_failure(log_text: str, timeout: bool) -> str:
    """Classify a non-zero / timed-out run from the *terminal* portion of its log.

    Only the last `_CLASSIFY_TAIL_LINES` non-empty lines are inspected so that a
    dependency/OOM error mentioned mid-log does not downgrade an unrelated real
    failure to a benign "blocked" status.
    """
    if timeout:
        return "failed_timeout"
    tail_lines = [line for line in log_text.splitlines() if line.strip()][-_CLASSIFY_TAIL_LINES:]
    tail = "\n".join(tail_lines).lower()
    if "modulenotfounderror" in tail or "no module named" in tail or "importerror" in tail or "distributionnotfound" in tail:
        return "blocked_dependency"
    if "out of memory" in tail or "cuda oom" in tail or "outofmemoryerror" in tail:
        return "blocked_compute"
    return "failed"


def observed_steps(log_text: str) -> int:
    seen = [int(m.group(1)) for m in STEP_RE.finditer(log_text)]
    return max(seen) if seen else 0


def run_logged(
    command,
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: int,
    env: dict | None = None,
    shell: bool = False,
    header_lines: list[str] | None = None,
) -> tuple[int | None, bool, float]:
    """Run `command`, streaming stdout+stderr to `log_path`, with a hard timeout.

    On timeout the whole process group is SIGTERM'd (then SIGKILL'd) so orphaned
    torchrun workers cannot survive. Returns (returncode, timed_out, duration_s);
    returncode is None only when the process had to be killed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    returncode: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        for line in header_lines or []:
            log.write(line if line.endswith("\n") else line + "\n")
        display = command if isinstance(command, str) else " ".join(command)
        log.write(f"$ {display}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=shell,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(proc)
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
    return returncode, timed_out, time.monotonic() - start


def _terminate_group(proc: "subprocess.Popen") -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def report_envelope(**extra) -> dict:
    """Common JSON-report header shared by both runners."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "gpu_count": gpu_count(),
    }
    payload.update(extra)
    return payload
