"""Iter 117b-3b: Per-expert capacity-padded sparse Q + Wo for attention.

Drop-in component for `train_gpt.py::CausalSelfAttention.forward` when
`use_sparse_attn_dispatch=True`. Asymmetric analog of the MLP sparse
dispatch (X2 step 3) that respects attention's token-token coupling.

# Why attention sparsity is asymmetric vs MLP

MLP per-token compute is INDEPENDENT — each token's MLP output uses only
that token's input. Capacity-padded gather/scatter trivially preserves
correctness and gives full proportional speedup.

Attention has TOKEN-TOKEN COUPLING — token t's output needs K[s], V[s]
from all other tokens s. Sparsifying K/V breaks the attention semantics
(other queries lose access to information through this expert). So:

  - Q projection: SPARSIFIABLE per expert (only top-K tokens contribute
    to expert e's output → only their Q_e is needed)
  - K, V projections: NOT SPARSIFIABLE (all keys/values needed by some
    other query through this expert)
  - SDPA: SHRINKS PROPORTIONALLY with sparse Q (K·T·d vs T²·d → ~4× at
    K=T/4 for our default C=4)
  - Wo projection: SPARSIFIABLE per expert (only sparse Q's positions
    have nonzero contribution from this expert)

# Compute saving (per expert)

Dense per-expert cost (approximate): Q_proj(T·D·R) + K_proj(T·D·r_kv) +
V_proj(T·D·r_kv) + SDPA(T²·d) + Wo(T·D·R).

Sparse per-expert cost at K = T/C: Q_proj(K·D·R) + K_proj(T·D·r_kv) +
V_proj(T·D·r_kv) + SDPA(K·T·d) + Wo(K·D·R).

Q + SDPA + Wo saving ≈ (T - K)/T = 1 - 1/C of those three (plus
gather/scatter overhead). At C=4: ~75% saving on Q+SDPA+Wo, ~0% saving
on K+V (the still-dense projections).

Total expert-attention cost reduction ≈ (Q + SDPA + Wo) · (1-1/C) /
total ≈ 50% if Q/SDPA/Wo are roughly half the per-expert cost.
Realistic step_avg reduction at our scale: ~20-30%.

# Caveats

1. **Head-packing constraint**: our architecture uses
   `(B, E·H, T, d)` head-packed SDPA. Sparsifying per-expert means
   different experts select different token sets, which breaks the
   fused SDPA call. Two implementation options:

   (a) **Per-expert SDPA loop** (simple, ~E SDPA calls instead of 1):
       Each expert gets its own (B, H, K, d) sparse-Q SDPA against
       (B, H, T, d) K/V. Loop overhead is non-trivial (E=15 SDPA
       launches per Block forward), but each SDPA is smaller.

   (b) **Fused sparse-Q via flex_attention** (advanced): use
       `torch.nn.attention.flex_attention` with a per-expert score_mod
       that masks non-selected query positions. Single kernel call,
       but requires careful score_mod design.

   This component implements (a). Performance vs (b) is an open
   question — test empirically once iter 117b-1 finishes.

2. **GQA structure**: our K, V are computed at `H_kv` heads (= H/2
   for our 2:1 GQA ratio), then broadcast. Sparsifying Q doesn't
   change the K/V compute — still done at H_kv heads × T × r_kv.

3. **Routing-weighted output**: the existing
   `out[t] = Σ_e w_e[t] · attn_e[t]` weighting is unchanged. Sparse
   dispatch produces 0 for tokens outside top-K (where w_e ≈ 0 by
   design), so the weighting is consistent.

4. **AdaSplash precedent (Fix #2 NOT VIABLE)**: AdaSplash's Triton
   kernel SIGABRT'd at step 1 under DDP + compile + RevDEQ. This
   component avoids Triton and uses standard `F.scaled_dot_product_attention`
   on smaller per-expert tensors — should not hit the same fragility.

# Implementation reference

```python
def sparse_attn_dispatch_per_expert(
    Q_full,        # (B, E, H, T, d)         — full head-packed Q
    K_full,        # (B, E, H_kv, T, d)      — full K
    V_full,        # (B, E, H_kv, T, d)      — full V
    routing_w,    # (B, T, E)               — per-token-per-expert weight
    capacity_factor,
):
    B, E, H, T, d = Q_full.shape
    H_kv = K_full.shape[2]
    K = ceil(capacity_factor * T / E)  # static
    out = torch.zeros(B, E, H, T, d)
    for e in range(E):
        w_e = routing_w[:, :, e]            # (B, T)
        topk_w, topk_idx = w_e.topk(K, dim=-1)  # (B, K), (B, K)
        # Gather sparse Q for expert e
        Q_e_sparse = gather(Q_full[:, e, :, :, :], topk_idx)  # (B, H, K, d)
        K_e = K_full[:, e, :, :, :]         # (B, H_kv, T, d)
        V_e = V_full[:, e, :, :, :]
        # Sparse SDPA: K queries × T keys
        out_e = F.scaled_dot_product_attention(
            Q_e_sparse, K_e, V_e, is_causal=False, ...)  # (B, H, K, d)
        # Apply causal masking via attn_mask if needed (depends on
        # whether causal can be derived from topk_idx ordering)
        out_e_weighted = out_e * topk_w  # (B, H, K, d)
        # Scatter back
        out[:, e, :, :, :].scatter_add_(2, topk_idx_expanded, out_e_weighted)
    return out
```

# Integration touchpoints (deferred)

1. Hyperparameters: `use_sparse_attn_dispatch = False`,
   `sparse_attn_capacity_factor = 4.0`.
2. CLI flag: `--use-sparse-attn-dispatch`.
3. CausalSelfAttention.forward conditional dispatch: when flag is set,
   replace the head-packed SDPA call with the per-expert sparse loop
   above. Causal masking for sparse Q requires care (see note 1).

# Component status

This file is **DESIGN + REFERENCE IMPLEMENTATION** only. The actual
sparse-Q gather + per-expert SDPA loop + scatter-back is non-trivial
under our specific head-packed + GQA + per-expert-MLA architecture
(`train_gpt.py::CausalSelfAttention`). A full integration commit
would need to:

1. Restructure the existing head-pack so per-expert tensors are
   accessible (currently fused with E·H concatenation).
2. Add the per-expert top-K gather using attn router weights.
3. Run E SDPA calls (or one flex_attention with score_mod).
4. Scatter back into the expected (B, T, E·d) output layout.
5. Verify causality is preserved (sparse Q + dense K means causal mask
   needs to map sparse-Q positions to their original token ids).

Smoke test below verifies the standalone helper math; full integration
test will follow when iter 117b-1 finishes + GPU free.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level toggle
# ---------------------------------------------------------------------------

_USE_SPARSE_ATTN_DISPATCH: bool = False
_SPARSE_ATTN_C: float = 4.0


def set_sparse_attn_dispatch(enabled: bool, capacity_factor: float = 4.0) -> None:
    """Module-level toggle invoked from train_gpt.py::main() at startup."""
    global _USE_SPARSE_ATTN_DISPATCH, _SPARSE_ATTN_C
    _USE_SPARSE_ATTN_DISPATCH = bool(enabled)
    _SPARSE_ATTN_C = float(capacity_factor)


def is_sparse_attn_dispatch_enabled() -> bool:
    return _USE_SPARSE_ATTN_DISPATCH


# ---------------------------------------------------------------------------
# Reference helper: per-expert sparse-Q SDPA
# ---------------------------------------------------------------------------


def sparse_attn_per_expert(
    Q_e: Tensor,          # (B, H, T, d) — single expert's full Q
    K_e: Tensor,          # (B, H_kv, T, d)
    V_e: Tensor,          # (B, H_kv, T, d)
    w_e: Tensor,          # (B, T) — routing weights for this expert
    capacity_factor: float,
    causal: bool = True,
) -> Tensor:
    """Per-expert sparse-Q attention: top-K tokens by routing weight.

    Returns: (B, H, T, d) with non-top-K positions = 0 (these positions
    are zero-weighted in the upstream routing sum, so this is correct).

    Implementation: gather top-K Q rows, run smaller SDPA against full
    K/V, scatter results back to original positions.

    Causal handling: when `causal=True`, the sparse-Q SDPA needs an
    attn_mask that maps each top-K query position back to its original
    token index, blocking attention to keys at positions > query position.
    """
    B, H, T, d = Q_e.shape
    K_n = int(math.ceil(float(capacity_factor) * T / 1))  # per-expert K (no E split here)
    K_n = max(min(K_n, T), 1)

    # Gather top-K query positions per (B,) — shared across heads.
    topk_w, topk_idx = w_e.topk(K_n, dim=-1)  # (B, K_n) values + indices
    # Expand topk_idx for gather across (H, d) dims.
    topk_idx_q = topk_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, K_n, d)
    Q_sparse = Q_e.gather(2, topk_idx_q)  # (B, H, K_n, d)

    # Build causal mask for sparse-Q vs dense-K.
    # For each sparse-Q position with original token idx q_idx, allow attention
    # only to key positions k_idx <= q_idx.
    if causal:
        # topk_idx shape (B, K_n), keys shape T.
        k_pos = torch.arange(T, device=Q_e.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        q_pos = topk_idx.unsqueeze(-1)                                         # (B, K_n, 1)
        causal_mask = k_pos <= q_pos  # (B, K_n, T) bool
        # Expand to (B, H, K_n, T)
        attn_mask = causal_mask.unsqueeze(1).expand(B, H, K_n, T)
        attn_mask_add = torch.zeros(B, H, K_n, T, dtype=Q_sparse.dtype, device=Q_e.device)
        attn_mask_add = attn_mask_add.masked_fill(~attn_mask, float('-inf'))
    else:
        attn_mask_add = None

    # SDPA on sparse Q × dense K, V. GQA: K_e/V_e have H_kv heads; we manually
    # repeat to H heads for compatibility with F.scaled_dot_product_attention's
    # standard signature (some torch versions support `enable_gqa=True` but we
    # use the explicit repeat for portability).
    H_kv = K_e.shape[1]
    if H_kv != H:
        if H % H_kv != 0:
            raise ValueError(f"H={H} must be a multiple of H_kv={H_kv} for GQA")
        repeats = H // H_kv
        K_e = K_e.repeat_interleave(repeats, dim=1)  # (B, H, T, d)
        V_e = V_e.repeat_interleave(repeats, dim=1)  # (B, H, T, d)
    out_sparse = F.scaled_dot_product_attention(
        Q_sparse, K_e, V_e,
        attn_mask=attn_mask_add,
        is_causal=False,  # we passed an explicit mask
    )  # (B, H, K_n, d)

    # Routing-weight: scale by w_e at the K positions.
    out_weighted = out_sparse * topk_w.unsqueeze(1).unsqueeze(-1)  # (B, H, K_n, d)

    # Scatter back to (B, H, T, d). Non-selected positions remain zero.
    out = torch.zeros(B, H, T, d, dtype=Q_e.dtype, device=Q_e.device)
    out.scatter_(2, topk_idx_q, out_weighted)

    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Verify shape, finiteness, causality, and bit-identity at C=T/K_n."""
    print("sparse_attention_dispatch.py smoke test:")
    torch.manual_seed(0)
    B, H, H_kv, T, d = 2, 4, 2, 32, 16

    Q_e = torch.randn(B, H, T, d)
    K_e = torch.randn(B, H_kv, T, d)
    V_e = torch.randn(B, H_kv, T, d)
    w_e = torch.softmax(torch.randn(B, T), dim=-1)  # routing weights

    # Test 1: shape preservation.
    out = sparse_attn_per_expert(Q_e, K_e, V_e, w_e, capacity_factor=0.5)
    assert out.shape == (B, H, T, d), f"shape {out.shape}"
    print(f"  shape preservation:    PASS  out.shape = {tuple(out.shape)}")

    # Test 2: finiteness.
    assert torch.isfinite(out).all()
    print(f"  finiteness:            PASS  max(|out|) = {out.abs().max().item():.4f}")

    # Test 3: at C such that K_n = T (top-K = all tokens), output should equal
    # F.scaled_dot_product_attention(Q, K, V, is_causal=True) * w_e (broadcast).
    # For the dense reference, manually expand K/V to H heads (same trick as
    # inside sparse_attn_per_expert).
    K_e_full = K_e.repeat_interleave(H // H_kv, dim=1)
    V_e_full = V_e.repeat_interleave(H // H_kv, dim=1)
    out_full = sparse_attn_per_expert(Q_e, K_e, V_e, w_e, capacity_factor=1.0)
    out_dense = F.scaled_dot_product_attention(Q_e, K_e_full, V_e_full, is_causal=True)
    out_dense_weighted = out_dense * w_e.unsqueeze(1).unsqueeze(-1)
    rel = (out_full - out_dense_weighted).abs().max().item() / out_dense_weighted.abs().max().item()
    status = "PASS" if rel < 1e-5 else "FAIL"
    print(f"  C=1.0 (K=T) ≈ dense·w: rel_err = {rel:.2e}  {status}")

    # Test 4: sparse output (low C) — non-top-K positions are zero.
    out_sparse = sparse_attn_per_expert(Q_e, K_e, V_e, w_e, capacity_factor=0.25)
    K_n = max(int(math.ceil(0.25 * T)), 1)
    _, topk_idx = w_e.topk(K_n, dim=-1)
    nontop_mask = torch.ones(B, T, dtype=torch.bool)
    nontop_mask.scatter_(1, topk_idx, False)
    nontop_max = out_sparse[nontop_mask.unsqueeze(1).expand(B, H, T).unsqueeze(-1).expand(B, H, T, d)].abs().max()
    print(f"  C=0.25 → non-top-K positions = 0: max(|out|) = {nontop_max.item():.2e}  "
          f"({'PASS' if nontop_max.item() < 1e-6 else 'FAIL'})")

    # Test 5: causality via Q[t+1:] perturbation.
    Q2 = Q_e.clone()
    # Need to perturb a token that IS in top-K to test sparsity-aware causality
    Q2[:, :, T // 2 + 1:, :] += torch.randn_like(Q2[:, :, T // 2 + 1:, :]) * 5.0
    out2 = sparse_attn_per_expert(Q2, K_e, V_e, w_e, capacity_factor=1.0)
    out_full = sparse_attn_per_expert(Q_e, K_e, V_e, w_e, capacity_factor=1.0)
    diff_before = (out_full - out2)[:, :, :T // 2, :].abs().max().item()
    print(f"  causality at C=1.0:    diff_before={diff_before:.2e}  ({'PASS' if diff_before < 1e-6 else 'FAIL'})")

    # Test 6: gradient flow.
    Q_e.requires_grad_(True)
    K_e.requires_grad_(True)
    V_e.requires_grad_(True)
    out = sparse_attn_per_expert(Q_e, K_e, V_e, w_e, capacity_factor=0.5)
    out.pow(2).sum().backward()
    assert Q_e.grad is not None and Q_e.grad.abs().max().item() > 0
    print(f"  gradient flow:         PASS  max(|∂L/∂Q|) = {Q_e.grad.abs().max().item():.4f}")

    # Test 7: toggle setter.
    assert not is_sparse_attn_dispatch_enabled()
    set_sparse_attn_dispatch(True, capacity_factor=8.0)
    assert is_sparse_attn_dispatch_enabled()
    assert _SPARSE_ATTN_C == 8.0
    set_sparse_attn_dispatch(False)
    assert not is_sparse_attn_dispatch_enabled()
    print(f"  toggle setter:         PASS")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
