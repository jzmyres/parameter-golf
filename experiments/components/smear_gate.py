"""Iter 128 / H99: SmearGate (BOS-fixed) — INPUT-DEPENDENT forward-1 token smear.

Drop-in for `Block.forward` (or post-token-embed in the encoder stack).
Implements the records' canonical SmearGate from PR #1667 (modded-nanogpt
@classiclarryd) with the BOS-leak fix from PR #1797.

Verified against records SOTA (2026-04-27, val_bpb=1.0611):
`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py`
L1242–1338. The corrected implementation here is BIT-EQUIVALENT to that file's
`_forward_hidden` SmearGate path.

# Hypothesis (from experiments/docs/hypotheses.md H99)

Records' SmearGate adds an INPUT-DEPENDENT forward-1 position smear:

    g_t = λ · σ(W · x_t[:window])    # scalar gate per token, depends on token
    x_t' = x_t + g_t · x_{t-1} · (input_ids[t] != BOS_ID)    for t > 0

where:
- `window` (default 12): how many of the leading dims of `x_t` the gate reads
- `W: (window, 1)` zero-init learnable weight
- `λ`: scalar zero-init learnable lambda
- σ: sigmoid

# The BOS-leak fix (PR #1797 — non-negotiable)

In packed validation streams, the naive smear leaks the last token of doc N
into the BOS embedding of doc N+1. Mask the previous-token contribution
wherever the current token is BOS:

    not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
    x' = concat([x[:, :1], x[:, 1:] + g · x[:, :-1] · not_bos], dim=1)

Apply symmetrically in `_forward_hidden` and `forward_ttt` so train/eval
stay consistent (records do this — see L1332).

# Strict-generalization

Both `λ=0` AND `W=0` ⇒ `g=0` ⇒ x' = x identity. Default init satisfies both:
zero λ + zero W. Promotion under §11 unconditional-promote rule because at
init the forward map equals iter 117 v5 baseline exactly.

# Integration into train_gpt.py

Three touchpoints:
1. `Hyperparameters.use_smear_gate = False`, `smear_gate_window = 12` fields.
2. Instantiate `SmearGate` once in `GPT.__init__` (or per-Block) with `model_dim`
   and `window`. CastedLinear-equivalent: keep `smear_gate.weight` in fp32 to
   avoid bf16 init imprecision.
3. Apply in `_forward_hidden` after `tok_emb`, before the first encoder layer.
   For TTT (when iter 123 H91 lands), apply identically in `forward_ttt`.

Smoke-test: `python experiments/components/smear_gate.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_USE_SMEAR_GATE: bool = False
_SMEAR_GATE_WINDOW: int = 12
_SMEAR_GATE_BOS_ID: int = 1


def set_smear_gate_enabled(enabled: bool, *, window: int = 12, bos_id: int = 1) -> None:
    """Module-level toggle invoked from `train_gpt.py::main()` at startup."""
    global _USE_SMEAR_GATE, _SMEAR_GATE_WINDOW, _SMEAR_GATE_BOS_ID
    _USE_SMEAR_GATE = bool(enabled)
    _SMEAR_GATE_WINDOW = int(window)
    _SMEAR_GATE_BOS_ID = int(bos_id)


def is_smear_gate_enabled() -> bool:
    return _USE_SMEAR_GATE


class SmearGate(nn.Module):
    """Input-dependent forward-1 position smear with BOS-leak masking.

    Args:
        dim: hidden dim D (must equal x.shape[-1]).
        window: number of leading dims of x_t the gate reads. Default 12.
        bos_id: integer BOS marker. Stored on the module for convenience but
            also passed per-call so callers can override (e.g. cu_seqlens-style).

    Forward:
        x: (B, T, D) hidden states.
        input_ids: (B, T) integer token ids.
        bos_id: integer BOS marker (overrides module default if provided).

    Init: `gate.weight` zero-init AND `lam` zero-init ⇒ identity.

    Records SOTA semantics (2026-04-27 PR #1797):
        g_t = lam * sigmoid(gate(x_t[:window]))         # (B, T-1, 1)
        not_bos_t = (input_ids[t] != bos_id)            # (B, T-1, 1)
        x_t' = x_t + g_t * x_{t-1} * not_bos_t          for t > 0
    """

    def __init__(self, dim: int, window: int = 12, bos_id: int = 0):
        super().__init__()
        if window > dim:
            raise ValueError(f"window={window} must be <= dim={dim}")
        self.dim = int(dim)
        self.window = int(window)
        self.bos_id = int(bos_id)
        # Linear(window, 1, bias=False) — fp32 init, zero-init for identity-at-start.
        self.gate = nn.Linear(self.window, 1, bias=False)
        nn.init.zeros_(self.gate.weight)
        # Global learnable scalar lambda; zero-init for identity-at-start.
        self.lam = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: Tensor, input_ids: Tensor, bos_id: int | None = None) -> Tensor:
        if x.shape[1] < 2:
            return x
        bos = self.bos_id if bos_id is None else int(bos_id)
        sl = self.lam.to(dtype=x.dtype)
        # gate input: leading `window` dims of each token, positions t=1..T-1.
        gate_in = x[:, 1:, : self.window].contiguous()
        # g shape: (B, T-1, 1)
        g = sl * torch.sigmoid(self.gate(gate_in.to(self.gate.weight.dtype))).to(x.dtype)
        # not_bos mask: (B, T-1, 1)
        not_bos = (input_ids[:, 1:] != bos).to(x.dtype).unsqueeze(-1)
        smeared = x[:, 1:] + g * x[:, :-1] * not_bos
        return torch.cat([x[:, :1], smeared], dim=1)


def _smoke_test() -> None:
    print("smear_gate.py smoke test (records-canonical, input-dependent):")
    torch.manual_seed(0)
    B, T, D, W = 2, 8, 16, 12
    BOS = 0

    # Case 1: zero-init (lam=0 AND gate weight=0) → identity
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    x = torch.randn(B, T, D)
    ids = torch.randint(1, 100, (B, T))  # no BOS
    out = sg(x, ids)
    assert torch.allclose(out, x, atol=1e-7), "zero-init must be identity"
    print("  zero-init identity:        PASS")

    # Case 2: lam>0 with zero W → still identity (sigmoid(0)=0.5, but g = lam*0.5,
    # multiplied by x[:-1] which is nonzero — so NOT identity unless we also zero W)
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(0.5)
        # gate weight stays zero
    x = torch.randn(B, T, D)
    ids = torch.randint(1, 100, (B, T))
    out = sg(x, ids)
    # g = 0.5 * sigmoid(0) = 0.25 per token (uniform).
    # Expected: x[:, 1:] += 0.25 * x[:, :-1] (no BOS).
    expected = x.clone()
    expected[:, 1:] = x[:, 1:] + 0.25 * x[:, :-1]
    assert torch.allclose(out, expected, atol=1e-6), "lam=0.5 + W=0 should give g=0.25 uniform"
    print("  lam=0.5 W=0 uniform smear: PASS (g=0.25)")

    # Case 3: nonzero W and lam → input-dependent g per token
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(1.0)
        sg.gate.weight.copy_(torch.randn_like(sg.gate.weight) * 0.5)
    x = torch.randn(B, T, D)
    ids = torch.randint(1, 100, (B, T))
    out = sg(x, ids)
    # Reproduce exactly:
    g = 1.0 * torch.sigmoid(F.linear(x[:, 1:, :W], sg.gate.weight))  # (B, T-1, 1)
    expected = x.clone()
    expected[:, 1:] = x[:, 1:] + g * x[:, :-1]
    assert torch.allclose(out, expected, atol=1e-5), \
        f"input-dependent gate failed; max diff = {(out - expected).abs().max()}"
    print("  input-dependent gate:      PASS")

    # Case 4: BOS at position t blocks the previous-token contribution at t
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(1.0)
        sg.gate.weight.copy_(torch.randn_like(sg.gate.weight) * 0.5)
    x = torch.randn(B, T, D)
    ids = torch.randint(1, 100, (B, T))
    ids[0, 3] = BOS  # BOS at sample 0, position 3
    out = sg(x, ids)
    # Position 3 in sample 0: not_bos = 0 → no contribution from x[0, 2].
    expected_3 = x[0, 3]  # bare
    assert torch.allclose(out[0, 3], expected_3, atol=1e-6), \
        f"BOS at t=3 sample 0 should block smear; out[0,3]={out[0,3,:3]} vs expected={expected_3[:3]}"
    print("  BOS-fix masking:           PASS")

    # Case 5: T=1 → no-op
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    x = torch.randn(B, 1, D)
    ids = torch.randint(1, 100, (B, 1))
    out = sg(x, ids)
    assert torch.equal(out, x), "T=1 should be identity"
    print("  T=1 no-op:                 PASS")

    # Case 6: window=D (gate sees full token) — corner case
    sg = SmearGate(dim=D, window=D, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(1.0)
        sg.gate.weight.copy_(torch.randn_like(sg.gate.weight) * 0.1)
    x = torch.randn(B, T, D)
    ids = torch.randint(1, 100, (B, T))
    out = sg(x, ids)
    g = torch.sigmoid(F.linear(x[:, 1:, :], sg.gate.weight))
    expected = x.clone()
    expected[:, 1:] = x[:, 1:] + g * x[:, :-1]
    assert torch.allclose(out, expected, atol=1e-5)
    print("  window=D full-token gate:  PASS")

    # Case 7: gradient flows to gate AND lam AND x
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(0.1)
        sg.gate.weight.copy_(torch.randn_like(sg.gate.weight) * 0.1)
    x = torch.randn(B, T, D, requires_grad=True)
    ids = torch.randint(1, 100, (B, T))
    out = sg(x, ids)
    out.sum().backward()
    assert sg.lam.grad is not None and sg.lam.grad.abs().item() > 0, "lam must receive gradient"
    assert sg.gate.weight.grad is not None and sg.gate.weight.grad.abs().max().item() > 0, \
        "gate weight must receive gradient"
    assert x.grad is not None and x.grad.abs().max().item() > 0, "x must receive gradient"
    print("  gradient flow:             PASS")

    # Case 8: window > D rejected
    try:
        SmearGate(dim=8, window=12)
        assert False, "should have raised"
    except ValueError:
        pass
    print("  window > dim rejected:     PASS")

    # Case 9: dtype propagation through bf16
    sg = SmearGate(dim=D, window=W, bos_id=BOS)
    with torch.no_grad():
        sg.lam.fill_(0.3)
        sg.gate.weight.copy_(torch.randn_like(sg.gate.weight) * 0.2)
    x = torch.randn(B, T, D, dtype=torch.bfloat16)
    ids = torch.randint(1, 100, (B, T))
    out = sg(x, ids)
    assert out.dtype == torch.bfloat16, f"output dtype mismatch: {out.dtype}"
    print("  bf16 dtype propagation:    PASS")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
