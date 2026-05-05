"""Test the submission artifact: size limit and quantization roundtrip fidelity.

Verifies that the model, after the same quantization + compression pipeline
used for scoring, still meets the 16MB size limit and that the quantized
model's loss closely matches the unquantized model's loss.

This is the test for what actually gets submitted and scored — the
quantization, serialization, and compression all go through the SAME helpers
that `train_gpt.main()` uses (`save_int6_artifact` / `load_int6_artifact`),
so the test cannot drift away from the real save path.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAX_ARTIFACT_BYTES = 16_000_000  # 16MB hard limit
MAX_QUANT_DEGRADATION = 0.05     # max allowed val_loss increase from quantization


def _cuda_available_or_skip(name: str) -> bool:
    if torch.cuda.is_available():
        return True
    print(f"=== SKIP: {name} requires CUDA ===")
    return False


def _build_model():
    """Build model with the production Hyperparameters.

    Threads every architecture field through the GPT constructor so the
    instantiated model matches what `train_gpt.main()` builds — otherwise
    the artifact-size test verifies a different (likely smaller) model
    than the one actually scored. Iter 96 baseline: full-D LoRA experts
    (no bottleneck rewrite), revdeq backward only.
    """
    from train_gpt import GPT, Hyperparameters
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim, num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank, mlp_expert_rank=args.mlp_expert_rank,
        deq_beta=args.deq_beta,
        deq_bptt_k=args.deq_bptt_k,
        router_scoring=args.router_scoring,
        router_entropy_coef=args.router_entropy_coef,
        use_entmax_routing=args.use_entmax_routing,
        entmax_blend_init_logit=args.entmax_blend_init_logit,
        entmax_blend_warmup_delay_frac=args.entmax_blend_warmup_delay_frac,
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts,
        use_smear_gate=args.use_smear_gate,
        smear_gate_init=args.smear_gate_init,
        smear_gate_bos_id=args.smear_gate_bos_id,
        logit_softcap=args.logit_softcap,
        lyapunov_coef=args.lyapunov_coef,
        lyapunov_gamma=args.lyapunov_gamma,
        use_parcae=args.use_parcae,
        parcae_init_a_bar=args.parcae_init_a_bar,
        parcae_init_b_bar=args.parcae_init_b_bar,
        use_ctp=args.use_ctp,
        ctp_weight=args.ctp_weight,
        router_load_cv_coef=args.router_load_cv_coef,
        mos_load_cv_coef=args.mos_load_cv_coef,
        cv_target=args.cv_target,
        mos_cv_target=args.mos_cv_target,
        expert_diversity_kind=args.expert_diversity_kind,
        expert_output_diversity_coef=args.expert_output_diversity_coef,
        expert_diversity_every=args.expert_diversity_every,
        expert_diversity_max_tokens=args.expert_diversity_max_tokens,
        mos_output_diversity_coef=args.mos_output_diversity_coef,
        regularizer_warmup_frac=args.regularizer_warmup_frac,
        use_nsa_attention=args.use_nsa_attention,
        nsa_compress_block_size=args.nsa_compress_block_size,
        nsa_compress_block_sliding_stride=args.nsa_compress_block_sliding_stride,
        nsa_sliding_window_size=args.nsa_sliding_window_size,
        nsa_branch_gate_init=args.nsa_branch_gate_init,
    ).cuda()
    return model, args


def _train_few_steps(model, num_steps=20):
    """Train a few steps so weights are non-trivial."""
    from train_gpt import Hyperparameters
    args = Hyperparameters()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(num_steps):
        x = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        y = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model


def _ntp_loss(model, vocab_size, num_batches=5):
    """Compute average NTP loss on random data."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x = torch.randint(0, vocab_size, (4, 128), device="cuda")
            y = torch.randint(0, vocab_size, (4, 128), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(x, y)
            total_loss += model._ntp_loss
    return total_loss / num_batches


def _quantize_and_compress(model):
    """Run the EXACT save pipeline `train_gpt.main()` uses.

    Returns `(blob, sd_cpu)` — `sd_cpu` is the un-quantized template needed
    by the symmetric loader for shape/dtype reconstruction.
    """
    from train_gpt import save_int6_artifact
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    blob, _qsd, _meta = save_int6_artifact(sd_cpu)
    return blob, sd_cpu


def _decompress_and_load(model, quant_blob, sd_cpu):
    """Inverse of _quantize_and_compress; uses the production load helper."""
    from train_gpt import load_int6_artifact
    deq_state = load_int6_artifact(quant_blob, sd_cpu)
    model.load_state_dict(deq_state, strict=True)
    return model


def test_artifact_size():
    """Test that quantized + compressed artifact fits within 16MB."""
    if not _cuda_available_or_skip("Artifact Size"):
        return None
    print("=== Test: Artifact Size ===")
    model, args = _build_model()
    model = _train_few_steps(model, num_steps=5)

    quant_blob, _ = _quantize_and_compress(model)
    quant_bytes = len(quant_blob)

    # Code size: read train_gpt.py
    code_path = os.path.join(os.path.dirname(__file__), "..", "train_gpt.py")
    with open(code_path) as f:
        code_bytes = len(f.read().encode("utf-8"))

    total_bytes = quant_bytes + code_bytes
    print(f"  Quantized model: {quant_bytes:,} bytes")
    print(f"  Code:            {code_bytes:,} bytes")
    print(f"  Total artifact:  {total_bytes:,} bytes")
    print(f"  Limit:           {MAX_ARTIFACT_BYTES:,} bytes")
    print(f"  Headroom:        {MAX_ARTIFACT_BYTES - total_bytes:,} bytes")

    assert total_bytes <= MAX_ARTIFACT_BYTES, (
        f"FAIL: artifact {total_bytes:,} bytes exceeds {MAX_ARTIFACT_BYTES:,} byte limit"
    )
    print("  PASS: artifact within size limit\n")
    return total_bytes


def test_quantization_roundtrip():
    """Test that quantized model performance matches unquantized within tolerance."""
    if not _cuda_available_or_skip("Quantization Roundtrip Fidelity"):
        return None
    print("=== Test: Quantization Roundtrip Fidelity ===")
    model, args = _build_model()
    model = _train_few_steps(model, num_steps=20)

    # Loss before quantization
    pre_loss = _ntp_loss(model, args.vocab_size)
    print(f"  Pre-quantization NTP loss:  {pre_loss:.4f}")

    # Quantize -> compress -> decompress -> load
    quant_blob, sd_cpu = _quantize_and_compress(model)
    model = _decompress_and_load(model, quant_blob, sd_cpu)

    # Loss after quantization roundtrip
    post_loss = _ntp_loss(model, args.vocab_size)
    print(f"  Post-quantization NTP loss: {post_loss:.4f}")

    degradation = post_loss - pre_loss
    print(f"  Degradation:               {degradation:+.4f}")
    print(f"  Tolerance:                  {MAX_QUANT_DEGRADATION:.4f}")

    assert degradation < MAX_QUANT_DEGRADATION, (
        f"FAIL: quantization degradation {degradation:.4f} exceeds tolerance {MAX_QUANT_DEGRADATION}"
    )
    # Sanity: quantization shouldn't magically improve loss by a lot
    assert degradation > -0.5, (
        f"FAIL: suspicious improvement after quantization ({degradation:+.4f})"
    )
    print("  PASS: quantization roundtrip within tolerance\n")
    return pre_loss, post_loss


def test_roundtrip_deterministic():
    """Test that quantize -> decompress -> re-quantize produces stable sizes."""
    if not _cuda_available_or_skip("Roundtrip Determinism"):
        return None
    print("=== Test: Roundtrip Determinism ===")
    model, _ = _build_model()
    model = _train_few_steps(model, num_steps=5)

    blob1, sd_cpu1 = _quantize_and_compress(model)
    # Load quantized weights back and re-quantize
    model = _decompress_and_load(model, blob1, sd_cpu1)
    blob2, _ = _quantize_and_compress(model)

    # Sizes should be very close (dequantized weights differ slightly)
    size_ratio = len(blob2) / len(blob1) if len(blob1) > 0 else 1.0
    print(f"  First compression:  {len(blob1):,} bytes")
    print(f"  Second compression: {len(blob2):,} bytes")
    print(f"  Size ratio:         {size_ratio:.4f}")

    assert 0.95 < size_ratio < 1.05, (
        f"FAIL: re-quantized size differs by {(size_ratio - 1)*100:+.1f}% "
        f"({len(blob1):,} -> {len(blob2):,})"
    )
    print("  PASS: roundtrip sizes are stable\n")


if __name__ == "__main__":
    test_artifact_size()
    test_quantization_roundtrip()
    test_roundtrip_deterministic()
    print("All submission tests passed!")
