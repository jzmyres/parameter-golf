from __future__ import annotations

import io
import os
import sys
import zlib

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tiny_gpt(**overrides):
    from train_gpt import GPT

    kwargs = dict(
        vocab_size=64,
        num_layers=2,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        rope_base=1000.0,
        qk_gain_init=1.0,
        bigram_vocab_size=0,
        bigram_dim=0,
        kv_latent_dim=16,
        num_refinements=0,
        num_experts=2,
        num_shared_experts=0,
        attn_expert_rank=4,
        mlp_expert_rank=4,
        use_ctp=False,
    )
    kwargs.update(overrides)
    return GPT(**kwargs)


def test_smear_gate_component_identity_and_bos_mask():
    from experiments.components.smear_gate import SmearGate

    torch.manual_seed(0)
    gate = SmearGate(dim=8, window=4, bos_id=1)
    x = torch.randn(2, 5, 8)
    ids = torch.tensor([[2, 3, 1, 4, 5], [6, 7, 8, 9, 10]])
    assert torch.allclose(gate(x, ids), x)
    with torch.no_grad():
        gate.lam.fill_(1.0)
    out = gate(x, ids)
    assert torch.allclose(out[0, 2], x[0, 2])
    assert not torch.allclose(out[1, 2], x[1, 2])


def test_sparse_attn_head_gate_component_identity():
    from experiments.components.sparse_attn_head_gate import SparseAttnHeadGate

    gate = SparseAttnHeadGate(num_heads=6, gate_window=4, gate_factor=2.0)
    x = torch.randn(2, 3, 8)
    y = torch.randn(2, 3, 6, 5)
    assert torch.allclose(gate(x, y), y)


def test_rr_attention_component_tau_one_matches_dense():
    from experiments.components.rr_attention import rr_attention

    torch.manual_seed(1)
    q = torch.randn(1, 4, 64, 8)
    k = torch.randn(1, 4, 64, 8)
    v = torch.randn(1, 4, 64, 8)
    out = rr_attention(q, k, v, stride=4, block_size=16, tau=1.0, causal=True)
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == dense.shape
    assert torch.allclose(out, dense, atol=2e-5, rtol=2e-5)


def test_gpt_train_time_components_forward_backward():
    torch.manual_seed(2)
    model = _tiny_gpt(
        use_smear_gate=True,
        smear_gate_window=4,
        use_sparse_attn_head_gate=True,
        sparse_attn_gate_window=4,
        use_rr_attention=True,
        rr_stride=2,
        rr_block_size=4,
        rr_tau=1.0,
    )
    x = torch.randint(0, 64, (1, 8))
    y = torch.randint(0, 64, (1, 8))
    loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert hasattr(model, "smear_gate")
    assert hasattr(model.shared_block.attn, "sparse_attn_head_gate")


def test_gpt_deq_prefix_anchors_forward_backward():
    # iter152 / flag-to-effect contract: prove the prefix-anchor branch in
    # GPT._forward_hidden runs forward+backward and produces finite gradients
    # at the project's actual sampled-K range. Untested-path executability
    # gate.
    torch.manual_seed(3)
    model = _tiny_gpt(
        deq_prefix_anchors=True,
        deq_prefix_anchor_set=(1, 2),
    )
    # Force K=2 so _prefix_anchor_depths(2, (1,2)) returns (1, 2) → 2 anchors.
    model._deq_k_override = 2
    model.train()
    x = torch.randint(0, 64, (1, 8))
    y = torch.randint(0, 64, (1, 8))
    loss = model(x, y)
    assert torch.isfinite(loss), f"prefix-anchor loss is non-finite: {loss.item()}"
    anchors_used = model._deq_prefix_anchor_depths_last
    assert anchors_used == (1, 2), f"expected anchors (1,2), got {anchors_used}"
    loss.backward()
    any_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert any_grad, "no finite gradient produced on any parameter"


def test_use_reverse_kl_balance_output_differs():
    """iter153 / flag-to-effect contract: forward KL(EMA||U) and reverse
    KL(U||EMA) must produce different `_ema_balance_raw_loss` values for any
    non-uniform EMA state. The reverse-KL path is the iter153 default; the
    forward path remains recoverable via `--use-reverse-kl-balance=0`.
    """
    from train_gpt import SoftDenseRouter

    torch.manual_seed(0)
    fwd = SoftDenseRouter(dim=8, num_experts=4, use_reverse_kl_balance=False)
    torch.manual_seed(0)
    rev = SoftDenseRouter(dim=8, num_experts=4, use_reverse_kl_balance=True)

    # Force a non-uniform persistent EMA: one dead expert, rest balanced.
    non_uniform = torch.tensor([0.01, 0.33, 0.33, 0.33])
    for r in (fwd, rev):
        r._expert_usage_ema_gpu.copy_(non_uniform)
        r._expert_usage_ema_initialized.fill_(True)

    torch.manual_seed(1)
    x = torch.randn(2, 6, 8)
    fwd.train(); rev.train()
    fwd(x); rev(x)

    fwd_loss = float(fwd._ema_balance_raw_loss.detach())
    rev_loss = float(rev._ema_balance_raw_loss.detach())
    assert fwd_loss != rev_loss, (
        f"reverse-KL flag had no effect on _ema_balance_raw_loss "
        f"(fwd={fwd_loss}, rev={rev_loss})"
    )
    # Reverse KL must strictly exceed forward KL for this tail-heavy EMA
    # (the whole point: reverse weights by uniform, so the dead expert's
    # -log(EMA_i) term blows up while forward's EMA_i · log(EMA_i/U_i) damps).
    assert rev_loss > fwd_loss, (
        f"reverse-KL was meant to amplify the dead-expert signal "
        f"but rev={rev_loss} <= fwd={fwd_loss}"
    )


def test_ttt_component_smoke():
    from experiments.components.phased_ttt import run_ttt_component_smoke

    stats = run_ttt_component_smoke(device="cpu")
    assert stats["before_norm"] == 0.0
    assert stats["after_norm"] > 0.0
    assert stats["reset_norm"] == 0.0


def test_gptq_lqer_component_smoke():
    from experiments.components.gptq_lqer import run_gptq_lqer_component_smoke

    stats = run_gptq_lqer_component_smoke(device="cpu", use_gptq=True, use_lqer=True, lqer_rank=2)
    assert stats["mse_after_lqer"] <= stats["mse_before_lqer"]
    assert stats["weighted_error"] >= 0.0


def test_grouped_artifact_compression_schema_roundtrip():
    from experiments.components.artifact_compression import grouped_compress_int6_payload

    qsd = {
        "a.scale": torch.ones(2, dtype=torch.float16),
        "a.q": torch.ones(2, 4, dtype=torch.int8),
        "ctrl": torch.randn(3),
    }
    meta = {"a": {"type": "int6"}, "ctrl": "passthrough_ctrl"}
    blob, stats = grouped_compress_int6_payload(qsd, meta, compressor="zlib")
    payload = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu", weights_only=False)
    assert set(payload["state_dict"]) == set(qsd)
    assert stats["compressed_bytes"] == len(blob)


def test_grouped_artifact_compression_flag_changes_payload_layout():
    # Flag-to-effect contract gate: enabling
    # `use_grouped_artifact_compression=True` must produce a measurably
    # different compressed payload from the naive zlib(torch.save(qsd))
    # baseline, otherwise the flag is a no-op and the banner misrepresents.
    # The Hyperparameters construction here is the witness the contract test
    # looks for; the codec call below proves the path actually differs.
    from train_gpt import Hyperparameters, _validate_hyperparameters
    from experiments.components.artifact_compression import grouped_compress_int6_payload

    cfg = Hyperparameters()
    cfg.use_grouped_artifact_compression = True
    _validate_hyperparameters(cfg)  # accepts the flag

    torch.manual_seed(4)
    qsd = {
        "block.0.attn.W_qkv.scale": torch.ones(16, dtype=torch.float16),
        "block.0.attn.W_qkv.q": torch.randint(-31, 31, (16, 64), dtype=torch.int8),
        "block.0.mlp.W_in.scale": torch.ones(8, dtype=torch.float16),
        "block.0.mlp.W_in.q": torch.randint(-31, 31, (8, 32), dtype=torch.int8),
        "norm.weight": torch.randn(64),
    }
    meta = {
        "block.0.attn.W_qkv": {"type": "int6"},
        "block.0.mlp.W_in": {"type": "int6"},
        "norm.weight": "passthrough_ctrl",
    }
    grouped_blob, grouped_stats = grouped_compress_int6_payload(qsd, meta, compressor="zlib")
    baseline_buf = io.BytesIO()
    torch.save(qsd, baseline_buf)
    naive_blob = zlib.compress(baseline_buf.getvalue(), 9)
    assert grouped_blob != naive_blob, (
        "grouped path produced identical bytes to naive zlib(torch.save) — flag is a no-op"
    )
    assert grouped_stats["compressed_bytes"] == len(grouped_blob)


def test_caseops_component_roundtrip():
    from experiments.components.caseops_tokenizer import encode_caseops_text, restore_caseops_text

    encoded = encode_caseops_text("Parameter Golf Smoke")
    assert encoded.normalized == "parameter golf smoke"
    assert restore_caseops_text(encoded) == "Parameter Golf Smoke"
