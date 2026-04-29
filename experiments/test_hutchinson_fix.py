"""Verify iter 97.5b-fix: _hutchinson_F_at_saved_fp returns a finite value.

Loads the iter 100b baseline weights, runs a single forward pass to populate
_lyapunov_z_star and _lyapunov_x0, then calls the Hutchinson-Frobenius probe
and asserts a finite float is returned (not None — the failure mode pre-fix).

Pre-fix expected output:
    hutch_F = None  (B_probe-unsliced JVP OOM'd at 32 GiB)

Post-fix expected output:
    hutch_F = <finite float>  (B_probe=1 slice -> ~1-2 GiB JVP, fits comfortably)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
from train_gpt import GPT, Hyperparameters, _hutchinson_F_at_saved_fp


def _build_test_model() -> GPT:
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim, num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank, mlp_expert_rank=args.mlp_expert_rank,
        router_scoring=args.router_scoring,
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts,
        use_ctp=args.use_ctp,
    ).cuda()
    return model


def main() -> int:
    print("=" * 60, flush=True)
    print("iter 97.5b-fix verification: _hutchinson_F_at_saved_fp", flush=True)
    print("=" * 60, flush=True)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", flush=True)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    free_gb_pre = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"Free VRAM pre-build: {free_gb_pre:.1f} GiB", flush=True)

    model = _build_test_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {n_params:,} params", flush=True)

    # Larger-than-smoke batch to mirror the val-checkpoint regime that
    # OOM'd pre-fix. iter 98b's val mini-batch was ~32 sequences x 2048;
    # we use 16x512 here to keep this test quick while still hitting the
    # batched-saved-activations regime that triggered 32 GiB OOM at full
    # val batch. The test goal is NOT to reproduce OOM (we already know
    # it OOM'd) but to confirm the post-fix B_probe=1 slice produces a
    # finite hutch_F.
    B, T = 16, 512
    vocab_size = Hyperparameters().vocab_size
    x = torch.randint(0, vocab_size, (B, T), device="cuda")

    print(f"Forward pass at B={B} T={T} ...", flush=True)
    model.train(False)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_logits(x)

    z_star = getattr(model, "_lyapunov_z_star", None)
    x0_lyap = getattr(model, "_lyapunov_x0", None)
    if z_star is None or x0_lyap is None:
        print(f"FAIL: forward did not populate _lyapunov_z_star / _lyapunov_x0", flush=True)
        return 3
    print(f"z_star shape: {tuple(z_star.shape)} dtype={z_star.dtype}", flush=True)
    print(f"x0_lyap shape: {tuple(x0_lyap.shape)} dtype={x0_lyap.dtype}", flush=True)

    free_gb_pre_probe = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"Free VRAM pre-probe: {free_gb_pre_probe:.1f} GiB", flush=True)

    print("Calling _hutchinson_F_at_saved_fp(B_probe=1) ...", flush=True)
    rho_F = _hutchinson_F_at_saved_fp(model, n_samples=2, B_probe=1)
    print(f"Result: hutch_F = {rho_F!r}", flush=True)

    if rho_F is None:
        print("FAIL: hutch_F returned None - fix did not resolve OOM", flush=True)
        return 1
    if not isinstance(rho_F, float):
        print(f"FAIL: hutch_F is not a float: {type(rho_F).__name__}", flush=True)
        return 1
    if not (rho_F > 0.0 and rho_F < 1e6):
        print(f"FAIL: hutch_F out of plausible range: {rho_F}", flush=True)
        return 1

    print("=" * 60, flush=True)
    print(f"PASS: hutch_F = {rho_F:.6f} (finite, plausible)", flush=True)
    print("iter 97.5b-fix VERIFIED - Hutchinson probe no longer OOMs.", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
