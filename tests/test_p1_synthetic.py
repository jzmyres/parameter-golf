"""Contracts for the final-minimal P1 synthetic harness."""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_s5_composition_table_has_identity_and_associativity():
    from experiments.p1_synthetic import build_s5_tables

    table, identity = build_s5_tables()
    for p in range(120):
        assert int(table[p, identity]) == p
        assert int(table[identity, p]) == p

    triples = [(0, 1, 2), (3, 7, 11), (19, 23, 29), (37, 41, 43)]
    for a, b, c in triples:
        left = int(table[table[a, b], c])
        right = int(table[a, table[b, c]])
        assert left == right


def test_mean_ci95_numeric_matches_known_inputs():
    """The promotion-gate statistic must be numerically correct, not just present."""
    import math

    from experiments.p1_synthetic import _mean_ci95

    # vector [1,2,3,4,5] -> total=15, sum_sq=55, n=5; mean=3.0, sample var=2.5.
    mean, lo, hi = _mean_ci95(15.0, 55.0, 5.0)
    half = 1.96 * math.sqrt(2.5 / 5.0)
    assert abs(mean - 3.0) < 1e-9
    assert abs(lo - (3.0 - half)) < 1e-9
    assert abs(hi - (3.0 + half)) < 1e-9
    # n == 1 -> degenerate zero-width CI at the mean.
    assert _mean_ci95(5.0, 25.0, 1.0) == (5.0, 5.0, 5.0)
    # n <= 0 -> NaN (an empty sample has no mean/CI; it must NOT read as a real 0).
    m0, l0, h0 = _mean_ci95(0.0, 0.0, 0.0)
    assert math.isnan(m0) and math.isnan(l0) and math.isnan(h0)


def _reconstructs(variant: str) -> float:
    from experiments.p1_synthetic import AdditiveCouplingP1Model, build_s5_tables, sample_s5_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, identity = build_s5_tables(device)
    model = AdditiveCouplingP1Model(
        dim=32,
        num_heads=4,
        mlp_mult=1.0,
        seq_len=4,
        variant=variant,
        num_classes=120,
    ).to(device)
    model.eval()
    tokens, _ = sample_s5_batch(4, 4, table, identity, device)
    return model.reconstruction_error(tokens, depth=3)


def test_additive_coupling_m0_reconstructs_initial_state():
    assert _reconstructs("m0") < 1e-5


def test_additive_coupling_mclk_reconstructs_initial_state():
    # Guards the clock-injection asymmetry: a forward/reverse mismatch in the
    # per-step step embedding would break mclk reconstruction while m0 still passes.
    assert _reconstructs("mclk") < 1e-5


def test_degenerate_budgets_are_rejected():
    from experiments.p1_synthetic import parse_args

    for argv in (
        ["--eval-batches", "0"],
        ["--batch-size", "0"],
        ["--eval-batch-size", "0"],
        ["--iterations", "0"],
    ):
        try:
            parse_args(argv)
        except SystemExit:
            continue
        raise AssertionError(f"degenerate budget accepted (would fabricate a gate): {argv}")


def test_multi_batch_eval_streams_and_respects_ci_invariants(tmp_path):
    # eval_batches > 1 exercises the streaming sum/sum-of-squares accumulation that
    # the single-batch tests never touch, and pins the gate's structural invariants.
    from experiments.p1_synthetic import main

    out = tmp_path / "p1_multi.json"
    main(
        [
            "--variant", "m0",
            "--iterations", "2",
            "--batch-size", "8",
            "--eval-batch-size", "8",
            "--eval-batches", "3",
            "--seq-len", "4",
            "--model-dim", "32",
            "--num-heads", "4",
            "--train-depths", "1,2",
            "--eval-depths", "1,2",
            "--pairs", "1,2",
            "--output-json", str(out),
        ]
    )
    pair = json.loads(out.read_text())["pairs"][0]
    assert pair["G_nll_ci_low"] <= pair["G_nll"] <= pair["G_nll_ci_high"]
    assert 0.0 <= pair["NDR_epsilon"] <= 1.0
    assert pair["n_examples"] == 8 * 3  # eval_batch_size * eval_batches


def test_tiny_p1_synthetic_run_writes_gate_metrics(tmp_path):
    from experiments.p1_synthetic import main

    out = tmp_path / "p1_synthetic.json"
    main(
        [
            "--variant",
            "mclk",
            "--task",
            "s5",
            "--iterations",
            "2",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "4",
            "--eval-batches",
            "1",
            "--seq-len",
            "4",
            "--model-dim",
            "32",
            "--num-heads",
            "4",
            "--train-depths",
            "1,2",
            "--eval-depths",
            "1,2",
            "--pairs",
            "1,2",
            "--output-json",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["variant"] == "mclk"
    assert data["k_sweep"]
    assert data["pairs"]
    assert {
        "K_lo",
        "K_hi",
        "G_nll",
        "G_nll_ci_low",
        "G_nll_ci_high",
        "G_acc",
        "G_acc_ci_low",
        "G_acc_ci_high",
        "NDR_epsilon",
        "n_examples",
    } <= set(data["pairs"][0])
    assert data["reconstruction_error"] is not None


def test_non_p1_cli_surface_is_rejected():
    from experiments.p1_synthetic import parse_args

    rejected = [
        ["--variant", "moe"],
        ["--variant", "static_moe"],
        ["--num-experts", "2"],
        ["--emit-cache-diagnostics", "1"],
        ["--eval-int6", "1"],
    ]
    for argv in rejected:
        try:
            parse_args(argv)
        except SystemExit:
            continue
        raise AssertionError(f"non-P1 CLI surface accepted: {argv}")


def test_gate_summary_contains_only_core_p1_fields(tmp_path):
    from experiments.p1_synthetic import main

    out = tmp_path / "p1_synthetic_core.json"
    main(
        [
            "--variant",
            "m0",
            "--iterations",
            "1",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "4",
            "--eval-batches",
            "1",
            "--seq-len",
            "4",
            "--model-dim",
            "32",
            "--num-heads",
            "4",
            "--train-depths",
            "1,2",
            "--eval-depths",
            "1,2",
            "--pairs",
            "1,2",
            "--output-json",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert "k_sweep" in data
    assert "pairs" in data
    assert "reconstruction_error" in data
    for removed in (
        "expert_utilization",
        "mechanism_diagnostics",
        "cache_diagnostics",
        "int6_k_sweep",
        "int6_pairs",
    ):
        assert removed not in data


def test_tiny_parity_task_run_uses_binary_head(tmp_path):
    from experiments.p1_synthetic import main

    out = tmp_path / "p1_parity.json"
    main(
        [
            "--task",
            "parity",
            "--variant",
            "control",
            "--iterations",
            "2",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "4",
            "--eval-batches",
            "1",
            "--seq-len",
            "4",
            "--model-dim",
            "32",
            "--num-heads",
            "4",
            "--train-depths",
            "1,2",
            "--eval-depths",
            "1,2",
            "--pairs",
            "1,2",
            "--output-json",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["task"] == "parity"
    assert data["variant"] == "control"
    assert data["k_sweep"]
