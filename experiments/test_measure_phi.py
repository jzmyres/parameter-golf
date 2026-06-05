"""Unit tests for the Iso-Depth train-r phi harness (experiments/measure_phi.py).

CPU-only and TRAINING-FREE: the subprocess train_gpt.py runs are stubbed (canned
``final val_loss:`` lines), so these tests exercise the parse + fit + emit WIRING
without launching any training. The numeric phi recovery of the fit itself is
covered by tests/test_m0_metrics.py::test_fit_phi_isodepth_*.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.measure_phi import (  # noqa: E402
    build_train_command,
    parse_final_val_loss,
    summarize_phi_sweep,
)
from train_gpt import fit_phi_isodepth  # noqa: E402


# --- parse_final_val_loss -------------------------------------------------
def test_parse_final_val_loss_reads_last_line():
    out = (
        "step:1/10 train_loss:6.9\n"
        "final val_loss:3.1000 val_bpb:1.8000\n"  # an earlier (stale) line
        "...rerun...\n"
        "final val_loss:2.5500 val_bpb:1.4200\n"  # the authoritative final
    )
    assert parse_final_val_loss(out) == (2.55, 1.42)


def test_parse_final_val_loss_missing_returns_none():
    assert parse_final_val_loss("crashed, no final line\nTraceback...") == (None, None)


def test_parse_final_val_loss_nan_tolerant():
    loss, bpb = parse_final_val_loss("final val_loss:nan val_bpb:nan")
    assert math.isnan(loss) and math.isnan(bpb)


# --- build_train_command --------------------------------------------------
def test_build_train_command_single_gpu_fixed_depth_and_no_hinge():
    cmd = build_train_command(
        4, iterations=100, seq_len=16, seed=7, device="cpu", nproc=1,
        artifact_out="/tmp/r4.bin", passthrough=["--model-dim", "32"])
    # python (not torchrun) for nproc=1.
    assert cmd[0].endswith("python") or "python" in cmd[0]
    assert "train_gpt.py" in cmd
    # FIXED depth r: single-element --k-set 4 (K_hi=r every step, k_eval defaults to r).
    assert cmd[cmd.index("--k-set") + 1] == "4"
    # Hinge DISABLED so each run is a clean pure-depth-r train.
    assert cmd[cmd.index("--lambda-h") + 1] == "0"
    # passthrough forwarded verbatim.
    assert cmd[cmd.index("--model-dim") + 1] == "32"


def test_build_train_command_multi_gpu_uses_torchrun():
    cmd = build_train_command(
        8, iterations=100, seq_len=16, seed=7, device="cuda", nproc=2,
        artifact_out="/tmp/r8.bin", passthrough=[])
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=2" in cmd
    assert cmd[cmd.index("--k-set") + 1] == "8"


# --- summarize_phi_sweep (parse->fit->emit wiring) ------------------------
_N_ONCE = 2_000_000
_N_REC = 6_000_000


def _law(r, phi, alpha=0.3, E=2.5, A=50.0, n_once=_N_ONCE, n_rec=_N_REC):
    return E + A * (n_once + (r ** phi) * n_rec) ** (-alpha)


def test_summarize_phi_sweep_emits_isodepth_line_matching_fit(capsys):
    # Canned per-r losses generated from the law with KNOWN phi=0.6 (simulating
    # the parsed `final val_loss:` of each stubbed training run).
    losses = {r: _law(r, 0.6) for r in (1, 2, 4, 8, 16)}
    bpb = {r: 1.5 - 0.01 * i for i, r in enumerate((1, 2, 4, 8, 16))}
    fit = summarize_phi_sweep(losses, _N_ONCE, _N_REC, bpb_by_r=bpb)

    # The returned fit is exactly fit_phi_isodepth on the same inputs.
    ref = fit_phi_isodepth(losses, _N_ONCE, _N_REC)
    assert abs(fit["phi"] - ref["phi"]) < 1e-9
    assert abs(fit["phi"] - 0.6) < 0.05  # noiseless recovery

    captured = capsys.readouterr().out
    # One phi_sweep line per r, ascending.
    for r in (1, 2, 4, 8, 16):
        assert f"phi_sweep: r={r} val_loss:" in captured
    # The final phi_isodepth line carries phi/alpha/E/rmse/n_once/n_rec.
    assert "phi_isodepth:" in captured
    iso_line = next(ln for ln in captured.splitlines()
                    if ln.startswith("phi_isodepth:"))
    assert f"{fit['phi']:.4f}" in iso_line
    assert f"n_once:{_N_ONCE}" in iso_line
    assert f"n_rec:{_N_REC}" in iso_line


def test_summarize_phi_sweep_skips_failed_runs(capsys):
    # A None loss (a crashed run with no `final val_loss:`) is reported FAILED and
    # excluded from the fit; the remaining finite r still fit.
    losses = {r: _law(r, 0.6) for r in (1, 2, 4, 8)}
    losses[16] = None  # crashed run
    fit = summarize_phi_sweep(losses, _N_ONCE, _N_REC)
    assert fit["n_points"] == 4  # only the 4 finite r enter the fit
    captured = capsys.readouterr().out
    assert "phi_sweep: r=16 val_loss:FAILED" in captured


def test_summarize_phi_sweep_too_few_points_emits_nan_with_reason(capsys):
    losses = {1: _law(1, 0.6), 2: _law(2, 0.6)}  # only 2 points
    fit = summarize_phi_sweep(losses, _N_ONCE, _N_REC)
    assert math.isnan(fit["phi"])
    captured = capsys.readouterr().out
    iso_line = next(ln for ln in captured.splitlines()
                    if ln.startswith("phi_isodepth:"))
    assert "nan" in iso_line.lower()
    assert "reason:" in iso_line
