"""Smoke test wrapper around `train_gpt.py main()`.

Runs the actual training script with a short-budget config and parses the
emitted train log for diagnostic trends. The DRY win: any change to the
training loop, optimizer setup, autocast, or diagnostic emission is
picked up automatically — there's no parallel implementation to drift
out of sync.

Hard requirements verified:
1. Both recon errors stay bounded (`tbptt_recon < 1e-1`, `deq_recon_err`
   reported informationally; bf16 ceiling sits between K_fwd=16 and 24).
2. Convergence `||z_T - z_{T-1}|| / ||z_T||` (relative) does not blow up.
3. NTP loss decreases over training.
4. No NaN/Inf in `grad_norm` line.

Env overrides:
- `SMOKE_K=N`   — override `_deq_k_override` (forward iterations).
- `SMOKE_KBWD=N` — override `Hyperparameters.deq_bptt_k` (gradient depth).
- `SMOKE_STEPS=N` — override iteration count (default 300).
"""
import math
import os
import re
import shlex
import subprocess
import sys


_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _build_cmd(steps: int, kbwd: int | None, k_fwd: int | None) -> list[str]:
    """Construct the torchrun command. Identical to production training
    (same compile, batch, seq_len, optimizer, autocast, diagnostics) —
    only the iteration count is shortened. `--nproc_per_node=gpu` lets
    torchrun auto-detect every visible GPU, matching CLAUDE.md §4 default.
    Validation is skipped via a giant `val-loss-every` since smoke only
    needs the train-time recon + descent signals."""
    parts = [
        "torchrun", "--standalone", "--nproc_per_node=gpu",
        "train_gpt.py",
        f"--iterations={steps}",
        "--val-loss-every=1000000",
    ]
    if kbwd is not None:
        parts.append(f"--deq-bptt-k={kbwd}")
    if k_fwd is not None:
        # When jitter is off, the training loop's K_fwd = `args.deq_k_max`
        # (`train_gpt.py::deq_k_for_step`). Disable jitter and pin
        # deq-k-max to the requested K to fix the forward iteration count.
        parts += [f"--deq-k-max={k_fwd}", "--deq-k-jitter=0"]
    return parts


def _parse_train_log(log_path: str) -> dict:
    """Pull every train-log line's diagnostic fields into parallel lists."""
    fields = {
        "step": [], "ntp": [], "grad_norm": [],
        "deq_recon_err": [], "tbptt_recon": [], "deq_fp_travel": [],
        "deq_iter_conv_rel": [],
    }
    pat = {
        "step": rf"^step:(\d+)/",
        "ntp": rf"ntp_loss:{_FLOAT}",
        "grad_norm": rf"grad_norm:{_FLOAT}",
        "deq_recon_err": rf"deq_recon_err:{_FLOAT}",
        "tbptt_recon": rf"tbptt_recon:{_FLOAT}",
        "deq_fp_travel": rf"deq_fp_travel:{_FLOAT}",
        "deq_iter_conv_rel": rf"deq_iter_conv_rel:{_FLOAT}",
    }
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("step:"):
                continue
            if "ntp_loss:" not in line:
                continue
            for k, p in pat.items():
                m = re.search(p, line)
                if m is None:
                    fields[k].append(math.nan)
                elif k == "step":
                    fields[k].append(int(m.group(1)))
                else:
                    fields[k].append(float(m.group(1)))
    return fields


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def _check(rows: dict) -> bool:
    """Apply the smoke threshold checks. Returns True on PASS."""
    ok = True
    n = len(rows["step"])
    if n < 2:
        print("FAIL: train log has fewer than 2 logged steps")
        return False

    ntps = [v for v in rows["ntp"] if _is_finite(v)]
    if len(ntps) < 2:
        print("FAIL: ntp_loss missing in train log")
        ok = False
    else:
        q = max(len(ntps) // 4, 1)
        first_q = sum(ntps[:q]) / q
        last_q = sum(ntps[-q:]) / q
        if last_q > first_q:
            print(f"FAIL: NTP not decreasing (first_q={first_q:.4f} -> last_q={last_q:.4f})")
            ok = False
        else:
            print(f"NTP descent: {ntps[0]:.4f} -> {ntps[-1]:.4f} (Δ={ntps[-1]-ntps[0]:+.4f})")

    tbptt = [v for v in rows["tbptt_recon"] if _is_finite(v)]
    if tbptt:
        if any(v > 1e-1 for v in tbptt):
            print(f"FAIL: tbptt_recon > 1e-1 (max={max(tbptt):.2e}) — reversibility broken")
            ok = False
        if len(tbptt) >= 2 and tbptt[-1] > max(tbptt[0] * 5, 1e-10):
            print(f"FAIL: tbptt_recon diverging ({tbptt[0]:.2e} -> {tbptt[-1]:.2e})")
            ok = False
        print(f"tbptt_recon: {tbptt[0]:.2e} -> {tbptt[-1]:.2e} (max={max(tbptt):.2e})")

    x0 = [v for v in rows["deq_recon_err"] if _is_finite(v)]
    if x0:
        print(f"x0_recon (deq_recon_err): {x0[0]:.2e} -> {x0[-1]:.2e} (informational; full BPTT only)")

    fp = [v for v in rows["deq_fp_travel"] if _is_finite(v)]
    if fp:
        print(f"deq_fp_travel: {fp[0]:.3e} -> {fp[-1]:.3e} (||z_K - z_init||/||z_init||; expressivity proxy)")

    convs = [v for v in rows["deq_iter_conv_rel"] if _is_finite(v)]
    if len(convs) >= 2:
        ratio = convs[-1] / max(convs[0], 1e-6)
        if ratio > 2.0:
            print(f"FAIL: iter_conv_rel growing ({convs[0]:.4f} -> {convs[-1]:.4f}, ratio={ratio:.1f}x)")
            ok = False
        print(f"iter_conv_rel: {convs[0]:.4f} -> {convs[-1]:.4f}")

    grads = [v for v in rows["grad_norm"] if _is_finite(v)]
    if any(math.isnan(v) or math.isinf(v) for v in rows["grad_norm"] if v is not None):
        print("FAIL: grad_norm contained NaN or Inf in train log")
        ok = False
    if grads:
        print(f"grad_norm: {grads[0]:.3f} -> {grads[-1]:.3f}")

    return ok


def smoke_test():
    # Default 50 steps matches "smoke" wallclock budget at full-batch prod
    # config (~25s/step on L40S → ~20min). Override `SMOKE_STEPS=N` for
    # longer. Production train_log_every=10 → 5 log points at default,
    # enough to verify descent direction + recon stability.
    steps = int(os.environ.get("SMOKE_STEPS", "50"))
    kbwd = os.environ.get("SMOKE_KBWD")
    k_fwd = os.environ.get("SMOKE_K")
    cmd = _build_cmd(
        steps=steps,
        kbwd=int(kbwd) if kbwd is not None else None,
        k_fwd=int(k_fwd) if k_fwd is not None else None,
    )
    log_path = "run.log"
    print(f"smoke cmd: {shlex.join(cmd)}")
    print(f"smoke log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"FAIL: train_gpt.py exited with code {proc.returncode}")
        # Surface the last few log lines for debugging.
        with open(log_path, "r", encoding="utf-8") as f:
            tail = f.readlines()[-30:]
        sys.stdout.write("".join(tail))
        return False

    rows = _parse_train_log(log_path)
    print(f"\n--- Smoke Test Results ({len(rows['step'])} log points) ---")
    return _check(rows)


if __name__ == "__main__":
    ok = smoke_test()
    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED — fix before running full training")
    sys.exit(0 if ok else 1)
