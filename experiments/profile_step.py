"""Profile a single training step to identify throughput bottlenecks.

Instruments forward_experts, MLP, router, DEQ solver, and backward pass
with CUDA events for accurate GPU timing. Reports ms and % breakdown.

Usage: conda activate opg && python experiments/profile_step.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from train_gpt import (
    GPT, Hyperparameters, router_diagnostics, _rms_norm,
)

def cuda_timer():
    """Return (start, end) CUDA events for precise GPU timing."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e

def profile_forward_experts(attn, x_n):
    """Profile sub-sections of CausalSelfAttention.forward_experts."""
    B, T, D = x_n.shape
    E = attn.num_experts
    H, H_kv = attn.num_heads, attn.num_kv_heads
    d = attn.head_dim
    R_q, R_kv = attn.expert_rank, attn.kv_rank
    N = B * T
    dtype = x_n.dtype
    timings = {}

    # --- Q projection ---
    s, e = cuda_timer()
    s.record()
    x_flat = x_n.reshape(N, D)
    q_down = attn.expert_q_down.to(dtype=dtype)
    q_h = (x_flat @ q_down.reshape(E * R_q, D).t()).view(N, E, R_q).permute(1, 0, 2)
    q_up = attn.expert_q_up.to(dtype=dtype).transpose(1, 2)
    q_and_gate = torch.bmm(q_h, q_up)
    e.record(); torch.cuda.synchronize()
    timings["Q_proj"] = s.elapsed_time(e)

    q_raw = q_and_gate[:, :, :H * d].reshape(E, B, T, H, d)
    gate_logits = q_and_gate[:, :, H * d:].reshape(E, B, T, H, 1)

    # --- Q RMS norm ---
    s, e = cuda_timer()
    s.record()
    q_rope = q_raw[..., :attn.rope_dim]
    q_nope = q_raw[..., attn.rope_dim:]
    q_rope, q_nope = _rms_norm(q_rope), _rms_norm(q_nope)
    e.record(); torch.cuda.synchronize()
    timings["Q_rmsnorm"] = s.elapsed_time(e)

    # --- KV projection ---
    s, e = cuda_timer()
    s.record()
    kv_a = attn.expert_kv_a.to(dtype=dtype)
    kv_b = attn.expert_kv_b.to(dtype=dtype).transpose(1, 2)
    kv_h = (x_flat @ kv_a.reshape(E * R_kv, D).t()).view(N, E, R_kv).permute(1, 0, 2)
    kv_latent = torch.bmm(kv_h, kv_b)
    e.record(); torch.cuda.synchronize()
    timings["KV_compress"] = s.elapsed_time(e)

    # --- KV decompress ---
    s, e = cuda_timer()
    s.record()
    kv_normed = attn.kv_pre_norm(kv_latent.reshape(E * N, attn.kv_latent_dim))
    kv_normed = kv_normed.reshape(E, N, attn.kv_latent_dim)
    ek = attn.expert_k_nope.to(dtype=dtype)
    ev = attn.expert_v.to(dtype=dtype)
    k_nope = torch.bmm(kv_normed, ek.transpose(1, 2)).reshape(E, B, T, H_kv, attn.nope_dim)
    v = torch.bmm(kv_normed, ev.transpose(1, 2)).reshape(E, B, T, H_kv, d)
    e.record(); torch.cuda.synchronize()
    timings["KV_decompress"] = s.elapsed_time(e)

    # --- K_rope + RMS norm ---
    s, e = cuda_timer()
    s.record()
    k_rope_shared = attn.c_k_rope(x_n).reshape(B, T, H_kv, attn.rope_dim)
    k_rope = k_rope_shared.unsqueeze(0).expand(E, -1, -1, -1, -1)
    k_rope, k_nope = _rms_norm(k_rope), _rms_norm(k_nope)
    e.record(); torch.cuda.synchronize()
    timings["K_rope+norm"] = s.elapsed_time(e)

    # --- RoPE + assemble ---
    s, e = cuda_timer()
    s.record()
    cos, sin = attn.rotary(T, x_n.device, q_rope.dtype)
    q_rope_p = q_rope.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, attn.rope_dim)
    from train_gpt import apply_rotary_emb
    q_rope_p = apply_rotary_emb(q_rope_p, cos, sin)
    k_rope_p = k_rope.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, attn.rope_dim)
    k_rope_p = apply_rotary_emb(k_rope_p, cos, sin)
    q_nope_p = q_nope.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, attn.nope_dim)
    q_full = torch.cat([q_rope_p, q_nope_p], dim=-1)
    q_full = q_full * attn.q_gain.to(dtype=dtype)[None, :, None, None]
    k_nope_p = k_nope.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, attn.nope_dim)
    k_full = torch.cat([k_rope_p, k_nope_p], dim=-1)
    v_full = v.permute(1, 0, 3, 2, 4).reshape(B, E * H_kv, T, d)
    e.record(); torch.cuda.synchronize()
    timings["RoPE+assemble"] = s.elapsed_time(e)

    # --- SDPA ---
    s, e = cuda_timer()
    s.record()
    try:
        y = F.scaled_dot_product_attention(
            q_full, k_full, v_full, attn_mask=None, is_causal=True,
            enable_gqa=(H_kv != H),
        )
    except TypeError:
        rep = H // H_kv
        k_use = k_full.repeat_interleave(rep, dim=1)
        v_use = v_full.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q_full, k_use, v_use, attn_mask=None, is_causal=True)
    e.record(); torch.cuda.synchronize()
    timings["SDPA"] = s.elapsed_time(e)

    # --- Gate + reshape ---
    s, e = cuda_timer()
    s.record()
    gate_logits_p = gate_logits.permute(1, 0, 3, 2, 4).reshape(B, E * H, T, 1)
    gate_act = torch.sigmoid(
        gate_logits_p.to(dtype=y.dtype) + attn.gate_bias.to(dtype=y.dtype)[None, :, None, None]
    )
    y = y * gate_act
    y = y.reshape(B, E, H, T, d).permute(0, 3, 1, 2, 4).reshape(B, T, E, D)
    e.record(); torch.cuda.synchronize()
    timings["gate+reshape"] = s.elapsed_time(e)

    return y, timings


def main():
    device = torch.device("cuda")
    args = Hyperparameters()

    # Build model at production config
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        num_experts=args.num_experts, attn_expert_rank=128, mlp_expert_rank=192,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        router_scoring=args.router_scoring,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(device).bfloat16()

    block = model.shared_block
    block.train()

    # Production-like batch: grad_accum micro-batch
    # train_batch_tokens=524288, seq_len=2048, world_size=2, grad_accum=~8
    # micro_batch = 524288 / 2048 / 2 / 8 = 16 sequences
    B, T = 16, 2048
    D = args.model_dim

    print(f"Config: B={B}, T={T}, D={D}, E={args.num_experts}, H={args.num_heads}")
    print(f"  Attn heads: E*H={args.num_experts * args.num_heads}, "
          f"KV heads: E*H_kv={args.num_experts * args.num_kv_heads}")
    print(f"  expert_rank={block.attn.expert_rank}, kv_rank={block.attn.kv_rank}")
    print()

    # Warmup
    x_n = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    x0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        with torch.no_grad():
            block(x_n, x0)
    torch.cuda.synchronize()

    # --- Profile actual Block.forward (end-to-end) ---
    N_RUNS = 10
    block_times = []
    attn_times = []
    for _ in range(N_RUNS):
        z_in = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        x0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            s, e = cuda_timer(); s.record()
            block(z_in, x0)
            e.record(); torch.cuda.synchronize()
            block_times.append(s.elapsed_time(e))

            h = block.state_norm(z_in + x0)
            s2, e2 = cuda_timer(); s2.record()
            block.attn.forward_experts(h)
            e2.record(); torch.cuda.synchronize()
            attn_times.append(s2.elapsed_time(e2))

    block_avg = sum(block_times) / len(block_times)
    attn_avg = sum(attn_times) / len(attn_times)
    print(f"Block.forward (actual):       {block_avg:.3f} ms")
    print(f"  forward_experts (actual):   {attn_avg:.3f} ms")
    print(f"  remainder (MLP+routing+..): {block_avg - attn_avg:.3f} ms")
    print()

    # --- Profile Block.forward components (instrumented) ---
    N_RUNS = 5
    all_timings = {}

    for run in range(N_RUNS):
        z_in = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        x0 = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            # 1. state_norm
            s, e = cuda_timer(); s.record()
            u = z_in + x0
            h = block.state_norm(u)
            e.record(); torch.cuda.synchronize()
            all_timings.setdefault("state_norm", []).append(s.elapsed_time(e))

            # 2. Router
            s, e = cuda_timer(); s.record()
            E = block.num_experts
            w_all = block.router(h, pre_normed=True)
            w_attn = w_all[..., :E].contiguous()
            w_mlp = w_all[..., E:].contiguous()
            e.record(); torch.cuda.synchronize()
            all_timings.setdefault("router", []).append(s.elapsed_time(e))

            # 3. Attention forward_experts (sub-profiled)
            attn_out, attn_timings = profile_forward_experts(block.attn, h)
            for k, v in attn_timings.items():
                all_timings.setdefault(f"attn.{k}", []).append(v)

            # 4. Attention weighted sum + post-norm
            s, e = cuda_timer(); s.record()
            attn_mix = (attn_out * w_attn.unsqueeze(-1)).sum(dim=2)
            attn_mix = block.attn_post_mix_norm(attn_mix)
            e.record(); torch.cuda.synchronize()
            all_timings.setdefault("attn_mix+norm", []).append(s.elapsed_time(e))

            # 5. MLP
            s, e = cuda_timer(); s.record()
            mlp_mix = block.mlp.mix_experts(h, w_mlp, pre_normed=True)
            mlp_mix = block.mlp_post_mix_norm(mlp_mix)
            e.record(); torch.cuda.synchronize()
            all_timings.setdefault("MLP", []).append(s.elapsed_time(e))

            # 6. Residual
            s, e = cuda_timer(); s.record()
            delta = (attn_mix + mlp_mix).to(dtype=z_in.dtype)
            raw_out = x0 + delta
            e.record(); torch.cuda.synchronize()
            all_timings.setdefault("residual", []).append(s.elapsed_time(e))

    # --- Report ---
    print("=" * 65)
    print(f"{'Component':<25} {'Mean (ms)':>10} {'Std':>8} {'%':>6}")
    print("-" * 65)

    means = {k: sum(v) / len(v) for k, v in all_timings.items()}
    total = sum(means.values())

    # Sort by time descending
    for k, mean_ms in sorted(means.items(), key=lambda x: -x[1]):
        std = (sum((x - mean_ms)**2 for x in all_timings[k]) / len(all_timings[k]))**0.5
        pct = 100.0 * mean_ms / total
        print(f"  {k:<23} {mean_ms:>10.3f} {std:>8.3f} {pct:>5.1f}%")

    print("-" * 65)
    print(f"  {'TOTAL':<23} {total:>10.3f} {'':>8} {'100.0':>5}%")
    print()

    # Attention sub-total
    attn_total = sum(v for k, v in means.items() if k.startswith("attn."))
    mlp_total = means.get("MLP", 0)
    print(f"  Attention total:  {attn_total:.3f} ms ({100*attn_total/total:.1f}%)")
    print(f"  MLP total:        {mlp_total:.3f} ms ({100*mlp_total/total:.1f}%)")
    print(f"  SDPA alone:       {means.get('attn.SDPA', 0):.3f} ms "
          f"({100*means.get('attn.SDPA', 0)/total:.1f}%)")
    print(f"  Q+KV proj:        {means.get('attn.Q_proj', 0) + means.get('attn.KV_compress', 0) + means.get('attn.KV_decompress', 0):.3f} ms")
    print(f"  Permute+reshape:  {means.get('attn.RoPE+assemble', 0) + means.get('attn.gate+reshape', 0):.3f} ms")


if __name__ == "__main__":
    main()
