#!/usr/bin/env python
"""Principled Iso-Depth recurrence-equivalence exponent ``phi`` (train-r sweep).

EXPENSIVE MULTI-TRAIN EXPERIMENT — NOT a per-run metric. This harness trains the
M0 model FROM SCRATCH once per recurrence count ``r`` in a sweep (default
``1,2,4,8,16`` -> FIVE full training runs), all at the SAME data / steps / seed /
config, captures each model's final validation loss ``L(r)``, and fits the
Iso-Depth joint scaling law (arXiv:2604.21106)::

    L(r) = E + A * (N_once + r^phi * N_rec)^(-alpha)

via :func:`train_gpt.fit_phi_isodepth`. ``N_once`` is the non-recurrent parameter
count (embeddings + readout, applied once) and ``N_rec`` is the recurrent
(looped-block) parameter count; the ``N_once + r^phi N_rec`` split encodes "loops
vs unique blocks", so phi=1 means looping r times buys the capacity of r unique
blocks and phi=0 means looping buys nothing. Because the split itself models this,
LOOPED MODELS ALONE identify phi — no untied baseline is needed (paper reference
phi ~ 0.46).

This is THE recurrence-effectiveness metric. The per-run ``phi_eval`` emitted by
``train_gpt.py --k-eval-sweep`` is only a CHEAP EVAL-DEPTH PROXY (the inference-K
log-slope of ONE trained model), not this train-r phi.

Each ``r`` trains at a FIXED recurrence depth: we pass ``--k-set <r>`` (a single
value makes ``K_hi=r`` every step and ``k_eval=r``) and DISABLE the
no-degradation hinge with ``--lambda-h 0`` so each model is a clean pure-depth-r
train (the hinge mixes a shallow/deep comparison that would confound the scaling
law). The runs are spawned as subprocesses of ``train_gpt.py`` (matching
``experiments/run_m0_control_experiments.sh``), using ``torchrun`` when
``NPROC>1`` and ``python`` otherwise.

Fit robustness: a 4-parameter law over only ~5 r is a small-sample, weakly
identified fit (phi/alpha/A partially trade off). Recovery is near-exact on clean
data but noise-fragile, especially at low phi. For tight confidence intervals,
add more r and/or repeat the sweep over seeds and report phi's spread.

Usage (GPU 7, default sweep)::

    CUDA_VISIBLE_DEVICES=7 ITERATIONS=2000 python experiments/measure_phi.py

Knobs via env or CLI (CLI wins). ``MODEL`` args are passed through to
``train_gpt.py`` verbatim after a ``--`` separator, e.g.::

    R_LIST=1,2,4,8,16 ITERATIONS=2000 SEQ_LEN=512 NPROC=1 \
      python experiments/measure_phi.py -- --model-dim 768 --n-experts 16

Output block::

    phi_sweep: r=<r> val_loss:<L> val_bpb:<bpb>     (one per r)
    phi_isodepth: <phi:.4f> alpha:<..> E:<..> rmse:<..> n_once:<..> n_rec:<..>
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (  # noqa: E402
    M0GPT,
    build_arg_parser,
    fit_phi_isodepth,
    hyperparameters_from_args,
    recurrence_param_counts,
)

# ``final val_loss:<L> val_bpb:<bpb>`` is the line train_gpt.py prints once at the
# end of every run (rank-0). NaN-tolerant so a diverged run surfaces as NaN (then
# fit_phi_isodepth drops it) rather than failing the regex.
_FLOAT_NAN = r"([-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf))"
_FINAL_RE = re.compile(rf"^final val_loss:{_FLOAT_NAN}\s+val_bpb:{_FLOAT_NAN}", re.M)


def parse_final_val_loss(output: str):
    """Return ``(val_loss, val_bpb)`` from a train_gpt.py run's stdout/stderr.

    Reads the LAST ``final val_loss:<L> val_bpb:<bpb>`` line (a run prints it
    once; tail-most is the authoritative final). Returns ``(None, None)`` when
    the line is absent (a crashed/incomplete run) so the caller can record the
    failure without poisoning the fit.
    """
    matches = list(_FINAL_RE.finditer(output))
    if not matches:
        return None, None
    m = matches[-1]
    return float(m.group(1)), float(m.group(2))


def build_train_command(r, *, iterations, seq_len, seed, device, nproc,
                        artifact_out, passthrough):
    """Build the train_gpt.py subprocess argv for a single fixed-depth-r run.

    A single-element ``--k-set <r>`` pins ``K_hi=r`` every step (and ``k_eval``
    defaults to ``max(k_set)=r``), and ``--lambda-h 0`` disables the
    no-degradation hinge so each run is a clean pure-depth-r train. ``torchrun``
    is used iff ``nproc > 1`` (DDP-safe), else plain ``python``. ``passthrough``
    forwards arbitrary ``train_gpt.py`` model/config flags verbatim.
    """
    if nproc > 1:
        argv = ["torchrun", "--standalone", f"--nproc_per_node={nproc}",
                "train_gpt.py"]
    else:
        argv = [sys.executable, "train_gpt.py"]
    argv += [
        "--k-set", str(r),
        "--lambda-h", "0",
        "--iterations", str(iterations),
        "--seq-len", str(seq_len),
        "--seed", str(seed),
        "--device", device,
        "--artifact-out", artifact_out,
    ]
    argv += list(passthrough)
    return argv


def summarize_phi_sweep(losses_by_r, n_once, n_rec, *, bpb_by_r=None):
    """Emit the ``phi_sweep:`` per-r lines + the final ``phi_isodepth:`` line.

    ``losses_by_r`` is ``{r: final_val_loss}`` (None entries — failed runs — are
    skipped in the per-r emit and absent from the fit). ``bpb_by_r`` is the
    parallel ``{r: val_bpb}`` map (optional; printed for readability). Returns the
    :func:`train_gpt.fit_phi_isodepth` result dict so callers/tests can assert it.
    """
    bpb_by_r = bpb_by_r or {}
    for r in sorted(losses_by_r):
        loss = losses_by_r[r]
        if loss is None:
            print(f"phi_sweep: r={r} val_loss:FAILED val_bpb:FAILED")
            continue
        bpb = bpb_by_r.get(r)
        bpb_s = f"{bpb:.4f}" if bpb is not None else "nan"
        print(f"phi_sweep: r={r} val_loss:{loss:.4f} val_bpb:{bpb_s}")
    fit = fit_phi_isodepth(
        {r: v for r, v in losses_by_r.items() if v is not None},
        n_once, n_rec)
    reason = fit.get("reason")
    line = (f"phi_isodepth: {fit['phi']:.4f} alpha:{fit['alpha']:.4f} "
            f"E:{fit['E']:.4f} rmse:{fit['rmse']:.4g} "
            f"n_once:{n_once} n_rec:{n_rec} n_points:{fit['n_points']}")
    if reason:
        line += f" reason:{reason}"
    print(line)
    return fit


def _build_model_for_param_split(passthrough):
    """Build an M0GPT under the same config as the runs, to read (n_once, n_rec).

    Parses the model/config flags from ``passthrough`` with train_gpt.py's own
    parser and builds the config through ``hyperparameters_from_args`` (the SAME
    mapping the trainer uses), so the split is computed from the EXACT config the
    sweep trains. The recurrence count r does NOT change the parameter set (the
    same block is looped r times), so one build suffices for the whole sweep.
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(passthrough))
    model = M0GPT(hyperparameters_from_args(args))
    return recurrence_param_counts(model)


def _env_default(name, default):
    return os.environ.get(name, default)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Iso-Depth train-r phi sweep (EXPENSIVE: trains one model per r).")
    p.add_argument("--r-list", default=_env_default("R_LIST", "1,2,4,8,16"),
                   help="Comma-separated recurrence counts r (default 1,2,4,8,16).")
    p.add_argument("--iterations", type=int,
                   default=int(_env_default("ITERATIONS", "2000")))
    p.add_argument("--seq-len", type=int,
                   default=int(_env_default("SEQ_LEN", "512")))
    p.add_argument("--seed", type=int, default=int(_env_default("SEED", "1337")))
    p.add_argument("--nproc", type=int, default=int(_env_default("NPROC", "1")),
                   help="GPUs per run; >1 uses torchrun DDP.")
    p.add_argument("--device", default=_env_default("DEVICE", "cuda"))
    p.add_argument("--out-dir",
                   default=_env_default("OUT_DIR",
                                        "experiments/training_logs/measure_phi"))
    # MODEL/config passthrough: everything after `--` goes verbatim to train_gpt.py.
    p.add_argument("passthrough", nargs="*",
                   help="train_gpt.py model/config flags (after a `--`).")
    args = p.parse_args(argv)

    # CUDA_VISIBLE_DEVICES defaults to GPU 7 per the M0 task conventions (NEVER
    # empty: an empty string hides all GPUs). Respect an explicit user setting.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

    r_list = [int(x) for x in args.r_list.split(",") if x.strip()]
    if len(r_list) < 3:
        print(f"measure_phi WARN: only {len(r_list)} r in the sweep; the "
              f"4-parameter law needs >=3 finite points to fit phi.",
              file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)

    # One param-split build under the sweep's config (r does not change the param set).
    n_once, n_rec = _build_model_for_param_split(args.passthrough)
    print(f"measure_phi: r_list={r_list} iterations={args.iterations} "
          f"seq_len={args.seq_len} seed={args.seed} nproc={args.nproc} "
          f"n_once={n_once} n_rec={n_rec}")

    losses_by_r = {}
    bpb_by_r = {}
    for r in r_list:
        log_path = os.path.join(args.out_dir, f"r{r}.log")
        artifact = os.path.join(args.out_dir, f"r{r}.int6.bin")
        cmd = build_train_command(
            r, iterations=args.iterations, seq_len=args.seq_len, seed=args.seed,
            device=args.device, nproc=args.nproc, artifact_out=artifact,
            passthrough=args.passthrough)
        print(f"measure_phi_run_start: r={r} -> {log_path}\n  cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + "\n" + proc.stderr
        with open(log_path, "w") as f:
            f.write(output)
        loss, bpb = parse_final_val_loss(output)
        if loss is None:
            print(f"measure_phi_run_FAILED: r={r} (no 'final val_loss:' in output; "
                  f"see {log_path}); rc={proc.returncode}", file=sys.stderr)
        losses_by_r[r] = loss
        bpb_by_r[r] = bpb
        print(f"measure_phi_run_done: r={r} val_loss={loss} val_bpb={bpb}")

    summarize_phi_sweep(losses_by_r, n_once, n_rec, bpb_by_r=bpb_by_r)


if __name__ == "__main__":
    main()
