"""Test the submission artifact: size limit and quantization roundtrip fidelity.

Verifies that the model, after the same quantization + compression pipeline
used for scoring, still meets the 16MB size limit and that the quantized
model's loss closely matches the unquantized model's loss.

This is the test for what actually gets submitted and scored.
"""
import io
import os
import sys
import zlib

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAX_ARTIFACT_BYTES = 16_000_000  # 16MB hard limit
MAX_QUANT_DEGRADATION = 0.05     # max allowed val_loss increase from quantization


def _build_model():
    """Build model with default hyperparameters."""
    from train_gpt import GPT, Hyperparameters
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
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
    """Run the same quantization + compression pipeline as train_gpt.py."""
    from train_gpt import mixed_quantize_int6, _COMPRESSOR
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn", "bigram"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    try:
        import zstandard
        if _COMPRESSOR == "zstd":
            quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        else:
            quant_blob = zlib.compress(quant_raw, 9)
    except ImportError:
        quant_blob = zlib.compress(quant_raw, 9)
    return quant_blob, quant_raw, sd_cpu


def _decompress_and_load(model, quant_blob, sd_cpu):
    """Decompress and load quantized weights back into the model."""
    from train_gpt import dequantize_mixed_int6, _COMPRESSOR
    try:
        import zstandard
        if _COMPRESSOR == "zstd":
            decompressed = zstandard.ZstdDecompressor().decompress(quant_blob)
        else:
            decompressed = zlib.decompress(quant_blob)
    except ImportError:
        decompressed = zlib.decompress(quant_blob)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    model.load_state_dict(deq_state, strict=True)
    return model


def test_artifact_size():
    """Test that quantized + compressed artifact fits within 16MB."""
    print("=== Test: Artifact Size ===")
    model, args = _build_model()
    model = _train_few_steps(model, num_steps=5)

    quant_blob, _, _ = _quantize_and_compress(model)
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
    print("=== Test: Quantization Roundtrip Fidelity ===")
    model, args = _build_model()
    model = _train_few_steps(model, num_steps=20)

    # Loss before quantization
    pre_loss = _ntp_loss(model, args.vocab_size)
    print(f"  Pre-quantization NTP loss:  {pre_loss:.4f}")

    # Quantize -> compress -> decompress -> load
    quant_blob, _, sd_cpu = _quantize_and_compress(model)
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
    print("=== Test: Roundtrip Determinism ===")
    model, _ = _build_model()
    model = _train_few_steps(model, num_steps=5)

    blob1, _, sd_cpu1 = _quantize_and_compress(model)
    # Load quantized weights back and re-quantize
    model = _decompress_and_load(model, blob1, sd_cpu1)
    blob2, _, _ = _quantize_and_compress(model)

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
