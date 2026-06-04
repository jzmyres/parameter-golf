"""Architecture-agnostic effective-depth diagnostic contracts."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_effective_rank_from_signatures_detects_identical_vs_orthogonal_steps():
    from train_gpt import _effective_rank_from_signatures

    identical = torch.ones(4, 8)
    orthogonal = torch.eye(4)

    assert torch.isclose(_effective_rank_from_signatures(identical), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(_effective_rank_from_signatures(orthogonal), torch.tensor(4.0), atol=1e-4)


def test_route_depth_metrics_accept_single_and_multi_slot_layouts():
    from train_gpt import _route_depth_metrics_from_weight_tracks

    single_slot = torch.tensor(
        [
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ],
        dtype=torch.float32,
    )
    multi_slot = torch.stack([single_slot, single_slot.flip(-1)], dim=1)

    one = _route_depth_metrics_from_weight_tracks(single_slot)
    many = _route_depth_metrics_from_weight_tracks(multi_slot)

    for stats in (one, many):
        assert set(stats) >= {
            "route_depth_nmi_mean",
            "route_depth_nmi_max",
            "expert_util_mean",
        }
        assert 0.0 <= float(stats["route_depth_nmi_mean"]) <= 1.0
        assert 0.0 <= float(stats["route_depth_nmi_max"]) <= 1.0
        assert 0.0 <= float(stats["expert_util_mean"]) <= 1.0

    assert float(one["route_depth_nmi_mean"]) > 0.80
    assert float(many["route_depth_nmi_mean"]) > 0.80


def test_k_sweep_table_includes_layout_agnostic_effective_depth_columns():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r", encoding="utf-8").read()

    for column in (
        "ED_update",
        "ED_logit",
        "route_depth_nmi_mean",
        "route_depth_nmi_max",
        "expert_util_mean",
        "expert_output_erank_mean",
    ):
        assert f'"{column}"' in src


def test_eval_forward_populates_effective_depth_probe_on_tiny_model():
    from train_gpt import GPT, router_diagnostics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(
        vocab_size=64,
        num_layers=3,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        rope_base=10000.0,
        qk_gain_init=1.0,
        bigram_vocab_size=0,
        bigram_dim=8,
        kv_latent_dim=16,
        num_experts=2,
        attn_expert_rank=4,
        mlp_expert_rank=4,
        num_refinements=0,
        use_ctp=False,
        finite_horizon_scale_coef=0.0,
    ).to(device)
    model.eval()
    x = torch.randint(0, 64, (1, 8), device=device)
    model._effective_depth_probe_pending = True
    with torch.no_grad(), router_diagnostics(enabled=True, step_tag=3):
        _ = model.forward_logits(x)

    ed_update = getattr(model, "_effective_depth_update_t", None)
    ed_logit = getattr(model, "_effective_depth_logit_t", None)
    assert isinstance(ed_update, torch.Tensor)
    assert isinstance(ed_logit, torch.Tensor)
    assert torch.isfinite(ed_update)
    assert torch.isfinite(ed_logit)
    assert float(ed_update.detach().cpu()) >= 1.0


def test_fp_metrics_are_advisory_not_finite_horizon_failures():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_gpt.py"), "r", encoding="utf-8").read()

    assert "FP ADVISORY" in src
    assert "rho_F=" not in _failure_append_block(src)
    assert "iter_conv_rel=" not in _failure_append_block(src)


def _failure_append_block(src: str) -> str:
    marker = "if _failures:"
    assert marker in src
    return src.split(marker, 1)[0].rsplit("_failures.append", 1)[-1]
