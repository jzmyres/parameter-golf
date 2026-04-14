"""Compare forward+backward wall-clock for deq_backward="revdeq" vs "unroll".

Measures actual per-step compute delta on the current hardware.
If unroll is significantly faster AND VRAM fits, switch the default for training.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import torch

from train_gpt import GPT, Hyperparameters, router_diagnostics


def _load_data(vocab_size=1024, total=65536):
    # Read data path from Hyperparameters so this benchmark works across environments.
    from train_gpt import Hyperparameters
    data_path = getattr(Hyperparameters(), "data_path", "./data/datasets/fineweb10B_sp1024/")
    shard = Path(data_path) / "fineweb_train_000000.bin"
    raw = np.fromfile(str(shard), dtype=np.int16, count=total)
    return torch.from_numpy(raw.astype(np.int64)).clamp(0, vocab_size - 1).cuda()


def _sample(buf, batch, seq):
    starts = torch.randint(0, len(buf) - seq - 1, (batch,))
    x = torch.stack([buf[s:s + seq] for s in starts])
    y = torch.stack([buf[s + 1:s + seq + 1] for s in starts])
    return x, y


def _build_model(args, backward_mode: str) -> GPT:
    return GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        kv_latent_dim=args.kv_latent_dim, num_refinements=args.num_refinements,
        attn_expert_rank=args.attn_expert_rank, mlp_expert_rank=args.mlp_expert_rank,
        deq_beta=args.deq_beta, deq_backward=backward_mode,
        tie_attn_mlp_router=args.tie_attn_mlp_router,
    ).cuda()


def benchmark_mode(backward_mode: str, n_warmup: int = 3, n_iter: int = 10,
                   batch: int = 8, seq: int = 1024) -> dict:
    torch.manual_seed(0)
    args = Hyperparameters()
    model = _build_model(args, backward_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    buf = _load_data(args.vocab_size)

    # Warmup (also triggers any JIT compile)
    model.train()
    for _ in range(n_warmup):
        x, y = _sample(buf, batch, seq)
        with router_diagnostics(enabled=True, step_tag=0):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    losses = []
    for _ in range(n_iter):
        x, y = _sample(buf, batch, seq)
        with router_diagnostics(enabled=True, step_tag=0):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "mode": backward_mode,
        "ms_per_step": 1000.0 * dt / n_iter,
        "peak_mem_gb": peak_mem,
        "final_loss": losses[-1],
    }


if __name__ == "__main__":
    print("Benchmarking revdeq (O(1) memory, 3× forward FLOPs per step)...")
    r_rev = benchmark_mode("revdeq")
    print(f"  revdeq: {r_rev['ms_per_step']:.1f} ms/step  peak={r_rev['peak_mem_gb']:.2f} GB  loss={r_rev['final_loss']:.4f}")

    # Free memory between runs so peak measurement of unroll is clean.
    torch.cuda.empty_cache()

    print("Benchmarking unroll (O(K) memory, 2× forward FLOPs per step)...")
    r_unr = benchmark_mode("unroll")
    print(f"  unroll: {r_unr['ms_per_step']:.1f} ms/step  peak={r_unr['peak_mem_gb']:.2f} GB  loss={r_unr['final_loss']:.4f}")

    print()
    speedup = r_rev["ms_per_step"] / max(r_unr["ms_per_step"], 1e-6)
    mem_ratio = r_unr["peak_mem_gb"] / max(r_rev["peak_mem_gb"], 1e-6)
    print(f"Speedup (unroll vs revdeq):     {speedup:.2f}×")
    print(f"Memory ratio (unroll/revdeq):   {mem_ratio:.2f}×")
    print(f"VRAM headroom on L40S (48GB):    {48.0 - r_unr['peak_mem_gb']:.1f} GB remaining with unroll")
    print()
    if speedup >= 1.20:
        print(f"✓ UNROLL gives ≥1.20× speedup — RECOMMEND switching default to 'unroll'.")
    elif speedup >= 1.05:
        print(f"⚠ UNROLL gives {speedup:.2f}× speedup — modest, judgment call.")
    else:
        print(f"✗ UNROLL gives only {speedup:.2f}× speedup — keep revdeq.")
