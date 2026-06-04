"""Contracts for the non-core feedback-stage diagnostics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tiny_common(out: Path) -> list[str]:
    return [
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


def test_s1_moe_stage_writes_mechanism_diagnostics(tmp_path):
    from experiments.p1_feedback_stages import main

    out = tmp_path / "s1_moe.json"
    main(["--stage", "s1", "--variant", "moe", "--num-experts", "3", *_tiny_common(out)])
    data = json.loads(out.read_text())
    assert data["stage"] == "s1"
    assert data["variant"] == "moe"
    assert data["k_sweep"]
    assert data["pairs"]
    assert "mechanism_diagnostics" in data
    diag = data["mechanism_diagnostics"]
    assert diag["kind"] == "s1_moe_mechanism_diagnostic"
    assert 0.0 <= diag["route_depth_nmi"] <= 1.0
    assert diag["AEBR"] >= 1.0
    assert 0.0 <= diag["expert_utilization"] <= 1.0


def test_s1_static_moe_uses_same_trainable_surface_as_stage_moe():
    from experiments.p1_feedback_stages import _parse_args, _model_for_args

    moe_args = _parse_args(["--stage", "s1", "--variant", "moe", "--num-experts", "3"])
    static_args = _parse_args(["--stage", "s1", "--variant", "static_moe", "--num-experts", "3"])
    moe = _model_for_args(moe_args)
    static = _model_for_args(static_args)
    assert sum(p.numel() for p in moe.parameters() if p.requires_grad) == sum(
        p.numel() for p in static.parameters() if p.requires_grad
    )


def test_s2_cache_stage_marks_proxy_as_not_autoregressive_kv(tmp_path):
    from experiments.p1_feedback_stages import main

    out = tmp_path / "s2_m0.json"
    main(["--stage", "s2", "--variant", "m0", *_tiny_common(out)])
    data = json.loads(out.read_text())
    cache = data["cache_diagnostics"]
    assert cache["kind"] == "hidden_state_proxy_not_autoregressive_kv"
    assert cache["alpha_exact_multi_depth_proxy"] == 1.0
    assert cache["alpha_terminal_hidden_proxy"] == 0.0
    assert cache["alpha_shared_hidden_proxy"] == 0.0
    assert cache["quality_gap"] is None
    assert "not_tested" in cache["quality_gap_status"]


def test_s3_int6_stage_writes_quantization_gap(tmp_path):
    from experiments.p1_feedback_stages import main

    out = tmp_path / "s3_m0.json"
    main(["--stage", "s3", "--variant", "m0", *_tiny_common(out)])
    data = json.loads(out.read_text())
    assert data["stage"] == "s3"
    assert data["variant"] == "m0"
    assert data["int6_k_sweep"]
    assert data["int6_pairs"]
    assert isinstance(data["quantization_gap_best_loss"], float)


def test_feedback_stage_file_keeps_dirichlet_ucb_deleted():
    text = (ROOT / "experiments" / "p1_feedback_stages.py").read_text().lower()
    assert "dirichlet_ucb" not in text
    assert "ucb_beta" not in text


def test_s2_s3_reject_moe_variants():
    from experiments.p1_feedback_stages import _parse_args

    for stage in ("s2", "s3"):
        try:
            _parse_args(["--stage", stage, "--variant", "moe"])
        except SystemExit:
            continue
        raise AssertionError(f"{stage} accepted non-m0 diagnostic variant")


def test_moe_stage_backward_touches_router_and_experts():
    from experiments.p1_feedback_stages import _model_for_args, _parse_args

    args = _parse_args(
        [
            "--stage",
            "s1",
            "--variant",
            "moe",
            "--seq-len",
            "4",
            "--model-dim",
            "32",
            "--num-heads",
            "4",
            "--num-experts",
            "3",
        ]
    )
    model = _model_for_args(args)
    tokens = torch.randint(0, 120, (2, 4))
    target = torch.randint(0, 120, (2,))
    loss = torch.nn.functional.cross_entropy(model(tokens, 2), target)
    loss.backward()
    names_with_grad = {name for name, param in model.named_parameters() if param.grad is not None}
    assert any("router" in name for name in names_with_grad)
    assert any("experts" in name for name in names_with_grad)
