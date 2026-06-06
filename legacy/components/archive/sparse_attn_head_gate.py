"""Iter 127 / H98: Sparse attention head-output gate (narrow-input per-head sigmoid).

Drop-in for the attention output mixer in `train_gpt.py::CausalSelfAttention.forward`.
Per-head input-dependent sigmoid that gates each head's contribution to the
post-SDPA tensor before the output projection.

Verified against records SOTA (2026-04-27, val_bpb=1.0611):
`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py`
L1060-1063 (forward) + L993-1001 (init). The implementation here matches the
canonical narrow-input semantics exactly.

# Important — naming clarification

The records call this "sparse" because the gate weight matrix is a NARROW LINEAR
`(num_heads, gate_window=12)` instead of the dense `(num_heads, dim)` of standard
GatedAttention (arxiv:2505.06708 G1). The "sparsity" is in the INPUT WIDTH (12
of D), NOT in any top-K selection over heads. Each head still gets a per-token
sigmoid gate; no head is hard-zeroed. Calling it "sparse" is records' terminology.

# Hypothesis (from experiments/docs/hypotheses.md H98)

Records use:

    g[B,T,h] = sigmoid(scale · W_g[h, :] @ x[B,T, :gate_window])
    y[B,T,h,:] = y[B,T,h,:] · g[B,T,h]      # broadcast over head_dim

where:
- `gate_window` (default 12): leading dims of x the gate reads
- `W_g`: (num_heads, gate_window), zero-init by default
- `scale` (default 1.0; SOTA hparam-stack uses 0.5): logit pre-sigmoid scaling

Strict-gen: `W_g = 0` ⇒ g = 0.5 uniformly, which is NOT identity. The records
also support `attn_out_gate` mode where the gate uses `2 * sigmoid(...)` so
zero-init gives g=1 (transparent). For this component we expose both modes
via `gate_factor`.

# Strict-generalization

`gate_factor=2.0`, `W_g=0` ⇒ g = 2·σ(0) = 1 ⇒ y unchanged ⇒ identity.
Use `gate_factor=2.0` for strict-gen at zero-init.
`gate_factor=1.0` reproduces records' "sparse" mode literally (g≈0.5 at init,
which scales y by half — useful as a regularization-style downscale at init,
but breaks strict-gen).

# Integration into train_gpt.py

Three touchpoints:
1. `Hyperparameters.use_sparse_attn_head_gate = False`,
   `sparse_attn_gate_window = 12`, `sparse_attn_gate_scale = 1.0`,
   `sparse_attn_gate_factor = 2.0` (default; 1.0 reproduces records).
2. Instantiate one `SparseAttnHeadGate` per Block (or per CausalSelfAttention
   instance) with `(num_experts * num_heads, gate_window)` shape — note: our
   model uses head-packed SDPA where the head dim is `E·H` not just `H`.
3. Apply in CausalSelfAttention.forward immediately after SDPA emits `y`
   (shape `(B, T, E*H, d_head)` after reshape), before the per-expert Wo
   projection. The records apply gate before output projection (see L1060-1063):

       gate_in = x[..., :gate_window].contiguous()
       g = sigmoid(scale * F.linear(gate_in, attn_gate_w))   # (B, T, num_heads)
       y = y * g[..., None]
       y = F.linear(y.reshape(...), out_w)                    # output projection

Smoke-test: `python experiments/components/archive/sparse_attn_head_gate.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SparseAttnHeadGate(nn.Module):
    """Narrow-input per-head sigmoid gate (records' "sparse_attn_gate" semantics).

    Args:
        num_heads: total head count to gate (in our model, this is `num_experts * num_heads`).
        gate_window: number of leading dims of input the gate reads. Default 12.
        scale: logit pre-sigmoid scaling. Default 1.0; SOTA hparam stack uses 0.5
            for "softer head-output gate".
        gate_factor: post-sigmoid multiplier. 1.0 = records' "sparse_attn_gate"
            literal; 2.0 = "attn_out_gate" mode where W=0 ⇒ g=1 ⇒ identity.
            Default 2.0 (strict-gen friendly).
        init_std: std for normal-init of W_g. 0.0 = zero-init (default; identity at
            init when gate_factor=2.0).

    Forward:
        x: (B, T, D) — input residual stream (shared across heads).
        y: (B, T, num_heads, head_dim) — post-SDPA per-head output.

    Returns:
        y_gated: (B, T, num_heads, head_dim) with per-token per-head gate applied.
    """

    def __init__(
        self,
        num_heads: int,
        gate_window: int = 12,
        scale: float = 1.0,
        gate_factor: float = 2.0,
        init_std: float = 0.0,
    ):
        super().__init__()
        if gate_factor <= 0:
            raise ValueError(f"gate_factor must be > 0, got {gate_factor}")
        self.num_heads = int(num_heads)
        self.gate_window = int(gate_window)
        self.scale = float(scale)
        self.gate_factor = float(gate_factor)
        # W_g: (num_heads, gate_window). Zero-init by default.
        W = torch.empty(self.num_heads, self.gate_window, dtype=torch.float32)
        if init_std > 0:
            nn.init.normal_(W, mean=0.0, std=init_std)
        else:
            nn.init.zeros_(W)
        self.attn_gate_w = nn.Parameter(W)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.shape[-1] < self.gate_window:
            raise ValueError(f"x last dim {x.shape[-1]} < gate_window {self.gate_window}")
        gate_in = x[..., : self.gate_window].contiguous()
        g = self.gate_factor * torch.sigmoid(
            self.scale * F.linear(gate_in, self.attn_gate_w.to(x.dtype))
        )  # (B, T, num_heads)
        return y * g[..., None]


def _smoke_test() -> None:
    print("sparse_attn_head_gate.py smoke test (records-canonical):")
    torch.manual_seed(0)
    B, T, D = 2, 4, 64
    H = 8
    d_head = 16
    W = 12

    # Case 1: gate_factor=2, W=0 → g=1 → identity (strict-gen)
    sg = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=1.0, gate_factor=2.0)
    x = torch.randn(B, T, D)
    y = torch.randn(B, T, H, d_head)
    out = sg(x, y)
    assert torch.allclose(out, y, atol=1e-6), "gate_factor=2 + W=0 must give identity"
    print("  gate_factor=2, W=0 identity: PASS")

    # Case 2: gate_factor=1 (records literal), W=0 → g=0.5 → y * 0.5
    sg = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=1.0, gate_factor=1.0)
    out = sg(x, y)
    assert torch.allclose(out, y * 0.5, atol=1e-6), "gate_factor=1 + W=0 must scale by 0.5"
    print("  gate_factor=1, W=0 (0.5x):   PASS")

    # Case 3: input-dependent gate
    sg = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=1.0, gate_factor=2.0, init_std=0.5)
    x = torch.randn(B, T, D)
    y = torch.randn(B, T, H, d_head)
    out = sg(x, y)
    # Reproduce the canonical math:
    expected_g = 2.0 * torch.sigmoid(F.linear(x[..., :W], sg.attn_gate_w))
    expected = y * expected_g[..., None]
    assert torch.allclose(out, expected, atol=1e-5)
    print("  input-dependent gate:        PASS")

    # Case 4: scale parameter — softer at scale=0.5 (records' SPARSE_ATTN_GATE_SCALE)
    sg_soft = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=0.5, gate_factor=2.0, init_std=1.0)
    sg_hard = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=2.0, gate_factor=2.0, init_std=1.0)
    with torch.no_grad():
        sg_hard.attn_gate_w.copy_(sg_soft.attn_gate_w)  # same weights
    x = torch.randn(B, T, D)
    y = torch.randn(B, T, H, d_head)
    out_soft = sg_soft(x, y)
    out_hard = sg_hard(x, y)
    # With same weights and larger scale, gates saturate more aggressively → larger
    # spread of |y_out| values.
    soft_g = 2.0 * torch.sigmoid(0.5 * F.linear(x[..., :W], sg_soft.attn_gate_w))
    hard_g = 2.0 * torch.sigmoid(2.0 * F.linear(x[..., :W], sg_hard.attn_gate_w))
    assert hard_g.std() > soft_g.std(), "scale=2 should saturate more than scale=0.5"
    print(f"  scale soft/hard saturation:  PASS (std_soft={soft_g.std():.3f} < std_hard={hard_g.std():.3f})")

    # Case 5: gradient flows
    sg = SparseAttnHeadGate(num_heads=H, gate_window=W, scale=1.0, gate_factor=2.0, init_std=0.1)
    x = torch.randn(B, T, D, requires_grad=True)
    y = torch.randn(B, T, H, d_head, requires_grad=True)
    out = sg(x, y)
    out.sum().backward()
    assert sg.attn_gate_w.grad is not None and sg.attn_gate_w.grad.abs().max() > 0
    assert x.grad is not None and x.grad.abs().max() > 0
    assert y.grad is not None and y.grad.abs().max() > 0
    print("  gradient flow:               PASS")

    # Case 6: dtype propagation
    for dt in [torch.float32, torch.bfloat16]:
        sg = SparseAttnHeadGate(num_heads=H, gate_window=W, gate_factor=2.0, init_std=0.1)
        x = torch.randn(B, T, D, dtype=dt)
        y = torch.randn(B, T, H, d_head, dtype=dt)
        out = sg(x, y)
        assert out.dtype == dt, f"dtype mismatch for {dt}"
    print("  dtype (fp32, bf16):          PASS")

    # Case 7: shape preservation across realistic shapes
    for nh in [4, 8, 16, 32, 128]:  # 128 ≈ our E*H = 16*8
        sg = SparseAttnHeadGate(num_heads=nh, gate_window=12, gate_factor=2.0, init_std=0.1)
        x = torch.randn(2, 8, 64)
        y = torch.randn(2, 8, nh, 16)
        out = sg(x, y)
        assert out.shape == y.shape
    print("  shape preservation:          PASS")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
