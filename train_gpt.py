"""OPG M0 trainer — clean reversible recurrent-depth GPT.

This module is being built incrementally to replace the rich ``train_gpt.py``.
Task 1 implements ONLY the core science object: a tied, additive-coupling
*reversible* recurrence whose inverse is structural, explicit, and floor-free
(invertibility is algebraic, not a contraction). Later tasks add the
O(1)-memory custom backward, MLA attention, MoE, MoS, and the LM scaffold.

Sections:
    1. Imports
    2. Normalization (RMSNorm)
    3. Reversible recurrence core (ReversibleRecurrence)
    4. O(1)-memory reversible BPTT (RevRecurrenceFn + run_reversible)
    5. Rotary position embedding (Rotary + apply_rotary_emb)
    6. Multi-head Latent Attention (MLAttention)
    7. SwiGLU Mixture-of-Experts (SwiGLUMoE)
    8. Mixture-of-Softmaxes output head (MoSHead)
    9. Test-only delta block (_TinyDelta)
   10. LM scaffold (Hyperparameters + M0GPT)
   11. Muon optimizer (ported from a15093a clean scaffold)
   12. Data loading (DistributedTokenLoader + tiny-synthetic fallback)
   13. BPB evaluation (build_sentencepiece_luts + run_validation)
   14. int6 artifact codec (quantize/save/load, 16 MB check)
   15. Finite-horizon loss + optimizer builder
   16. CLI trainer (main)
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import argparse
import glob
import io
import math
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:  # pragma: no cover - zstd is the prod path
    _COMPRESSOR = "zlib"

# ---------------------------------------------------------------------------
# 2. Normalization
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Learned-scale RMS normalization computed in fp32 then cast back."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.w * normed.type_as(x)


# ---------------------------------------------------------------------------
# 3. Reversible recurrence core
# ---------------------------------------------------------------------------
class ReversibleRecurrence(nn.Module):
    """Additive-coupling reversible recurrence with an explicit, floor-free inverse.

    Forward update (one depth step), conditioned on the injected input ``x0``::

        a_{k+1} = a_k + F(b_k     + x0)
        b_{k+1} = b_k + G(a_{k+1} + x0)

    The inverse runs the same blocks in reverse, undoing each additive coupling::

        b_k     = b_{k+1} - G(a_{k+1} + x0)
        a_k     = a_{k+1} - F(b_k     + x0)

    Reconstruction is exact (no contraction / no floor) because each step only
    adds a quantity that is recomputable from the *other* stream plus ``x0``.
    ``F`` / ``G`` carry their own input RMSNorm, so the inverse recomputes the
    same normalized argument identically.
    """

    def __init__(self, F, G):
        super().__init__()
        self.F, self.G = F, G

    def forward_states(self, a, b, x0, depth):
        states = []
        for _ in range(int(depth)):
            a = a + self.F(b + x0)
            b = b + self.G(a + x0)
            states.append(0.5 * (a + b))
        return (a, b), states

    def invert(self, a, b, x0, depth):
        for _ in range(int(depth)):
            b = b - self.G(a + x0)
            a = a - self.F(b + x0)
        return a, b

    def run_reversible(self, x0, depth):
        """O(1)-memory reversible BPTT: seed ``a0 = b0 = x0``, return midpoint.

        Routes through :class:`RevRecurrenceFn` so the backward reconstructs
        activations instead of storing them. Gradients are identical to
        ``forward_states`` followed by ordinary autograd (see grad-equivalence
        test). ``x0`` and every parameter are explicit ``apply`` inputs.
        """
        aK, bK = RevRecurrenceFn.apply(self, x0, depth, *self.parameters())
        return 0.5 * (aK + bK)


# ---------------------------------------------------------------------------
# 4. O(1)-memory reversible BPTT
# ---------------------------------------------------------------------------
class RevRecurrenceFn(torch.autograd.Function):
    """Custom autograd for the reversible recurrence with O(1) activation memory.

    Forward saves only the terminal state ``(aK, bK)`` (plus the injected
    ``x0``); intermediate states are *not* stored. Backward walks the depth
    loop in reverse: at each step it reconstructs ``(a_prev, b_prev)`` from the
    current ``(a, b)`` via the algebraic inverse, rebuilds that single forward
    step as a fresh local graph, and VJP-propagates the incoming cotangents
    ``(ga, gb)`` to the step inputs (``a_prev``, ``b_prev``, ``x0``) and every
    parameter. This yields gradients numerically identical to ordinary BPTT
    while keeping memory constant in ``depth``.

    Per ``EXPERIENCE.md#custom-autograd-input``: every grad-needing tensor
    (``x0`` and every parameter) is an explicit ``apply(...)`` input with a
    matching gradient slot in ``backward``. Reconstructed states are
    re-instantiated as leaves with ``.clone().requires_grad_(...)`` — never
    ``.detach().requires_grad_(...)`` — so the two coupling legs and successive
    steps never share underlying storage.
    """

    @staticmethod
    def forward(ctx, rec, x0, depth, *params):
        with torch.no_grad():
            (aK, bK), _ = rec.forward_states(x0, x0, x0, depth)
        ctx.rec, ctx.depth = rec, depth
        ctx.save_for_backward(aK.detach(), bK.detach(), x0.detach())
        return aK, bK

    @staticmethod
    def backward(ctx, ga, gb):
        rec, depth = ctx.rec, ctx.depth
        a, b, x0 = ctx.saved_tensors
        params = list(rec.parameters())
        pgrads = [torch.zeros_like(p) for p in params]
        gx0 = torch.zeros_like(x0)
        for _ in range(int(depth)):
            # Algebraic inverse of one forward step (no_grad: reconstruction
            # only, gradients flow through the rebuilt graph below).
            with torch.no_grad():
                b_prev = b - rec.G(a + x0)
                a_prev = a - rec.F(b_prev + x0)
            # Rebuild the single forward step on fresh leaves. Use .clone() so
            # the y-leg/z-leg pair and successive steps do not alias storage.
            # enable_grad: a custom Function's backward runs with grad disabled
            # by default; we need a live local graph to take the per-step VJP.
            with torch.enable_grad():
                ap = a_prev.clone().requires_grad_(True)
                bp = b_prev.clone().requires_grad_(True)
                x0r = x0.clone().requires_grad_(True)
                a_new = ap + rec.F(bp + x0r)
                b_new = bp + rec.G(a_new + x0r)
            grads = torch.autograd.grad(
                (a_new, b_new), [ap, bp, x0r, *params],
                grad_outputs=(ga, gb), retain_graph=False, allow_unused=True,
            )
            ga, gb, gx0_step = grads[0], grads[1], grads[2]
            if gx0_step is not None:
                gx0 = gx0 + gx0_step
            for i, g in enumerate(grads[3:]):
                if g is not None:
                    pgrads[i] = pgrads[i] + g
            a, b = a_prev, b_prev
        # Seed: a0 = b0 = x0, so the cotangents that reached the loop entry
        # (ga, gb) both flow into x0 in addition to the per-step injection.
        gx0 = gx0 + ga + gb
        return (None, gx0, None, *pgrads)


# ---------------------------------------------------------------------------
# 5. Rotary position embedding
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    """Rotary position embedding with a cached cos/sin table.

    Ported from ``train_gpt.py::Rotary``, stripped of the legacy
    ``@dynamo_disable`` decorator and inference-mode cache guard (M0 builds the
    cache lazily and recomputes when ``seq_len`` / device changes). ``dim`` is
    the rotated sub-dimension (the rope half of each head), so the table holds
    ``dim // 2`` frequencies and ``apply_rotary_emb`` rotates ``dim`` channels.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _refresh_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self._cos_cached = freqs.cos()[None, None, :, :].contiguous()
        self._sin_cached = freqs.sin()[None, None, :, :].contiguous()
        self._seq_len_cached = seq_len

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
            self._refresh_cache(seq_len, device)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    """Rotate the last dim of ``x`` (shape ``(..., 2*half)``). Ported verbatim
    from ``train_gpt.py``: the two halves are the real/imag interleave-free
    layout (first half / second half), so ``cos``/``sin`` have width ``half``.
    """
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# ---------------------------------------------------------------------------
# 6. Multi-head Latent Attention (MLA)
# ---------------------------------------------------------------------------
class MLAttention(nn.Module):
    """Clean single-module Multi-head Latent Attention (DeepSeek-V2 style).

    Low-rank joint KV compression with decoupled RoPE, ported and stripped from
    ``train_gpt.py::CausalSelfAttention``. Removed relative to the source:
    per-expert weight banks, NSA / round-robin / sparse-head-gate / smear
    branches, and the attention gate. What remains is the core MLA path:

      1. Q (low-rank):   x -> q_down (dim->q_latent) -> RMS -> q_up
                         (q_latent -> n_heads*head_dim), split into rope/nope.
      2. KV (low-rank):  x -> kv_down (dim->kv_latent) -> RMS statistic, then
                         separate learned K/V input scales -> k_up / v_up
                         (kv_latent -> n_kv_heads*nope_dim / n_kv_heads*head_dim).
      3. Decoupled RoPE: a separate k_rope projection (dim -> n_kv_heads*rope_dim)
                         carries position; RoPE is applied to q_rope and k_rope.
      4. Attention:      causal SDPA with GQA (enable_gqa when n_kv_heads<n_heads).
      5. Output:         o_proj mixes heads back to ``dim``.

    Each head splits into ``rope_dim`` (rotated, position-dependent) plus
    ``nope_dim`` (content-only). bf16-friendly: only RMSNorm statistics use the
    fused ``F.rms_norm`` kernel; no forced fp32 elsewhere.
    """

    def __init__(self, dim, n_heads, n_kv_heads, kv_latent, head_dim,
                 q_latent=None, rope_base=10000.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads (GQA)"
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_latent = kv_latent
        self.q_latent = q_latent if q_latent is not None else kv_latent
        # Decoupled-RoPE split: half of each head carries position (rope), the
        # remainder is content-only (nope). Mirrors the source MLA convention.
        self.rope_dim = head_dim // 2
        self.nope_dim = head_dim - self.rope_dim

        # --- Q: low-rank dim -> q_latent -> n_heads*head_dim ---
        self.q_down = nn.Linear(dim, self.q_latent, bias=False)
        self.q_norm = RMSNorm(self.q_latent)
        self.q_up = nn.Linear(self.q_latent, n_heads * head_dim, bias=False)

        # --- KV: low-rank dim -> kv_latent, then up-project to K_nope / V ---
        self.kv_down = nn.Linear(dim, kv_latent, bias=False)
        self.kv_norm = RMSNorm(kv_latent)
        self.k_up = nn.Linear(kv_latent, n_kv_heads * self.nope_dim, bias=False)
        self.v_up = nn.Linear(kv_latent, n_kv_heads * head_dim, bias=False)

        # --- Decoupled K_rope: dim -> n_kv_heads*rope_dim (position channel) ---
        self.k_rope = nn.Linear(dim, n_kv_heads * self.rope_dim, bias=False)

        # --- Output projection: mix heads back to dim ---
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rotary = Rotary(self.rope_dim, base=rope_base)

    def forward(self, x):
        B, T, _ = x.shape
        H, H_kv, d = self.n_heads, self.n_kv_heads, self.head_dim

        # --- Q (low-rank) -> (B, H, T, head_dim), split rope/nope ---
        q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(B, T, H, d).transpose(1, 2)              # (B, H, T, d)
        q_rope, q_nope = q[..., :self.rope_dim], q[..., self.rope_dim:]

        # --- KV (low-rank latent) -> K_nope, V ---
        kv = self.kv_norm(self.kv_down(x))                   # (B, T, kv_latent)
        k_nope = self.k_up(kv).view(B, T, H_kv, self.nope_dim).transpose(1, 2)  # (B,H_kv,T,nope)
        v = self.v_up(kv).view(B, T, H_kv, d).transpose(1, 2)                   # (B,H_kv,T,d)

        # --- Decoupled K_rope ---
        k_rope = self.k_rope(x).view(B, T, H_kv, self.rope_dim).transpose(1, 2)  # (B,H_kv,T,rope)

        # --- Apply RoPE to the rope sub-dims of Q and K ---
        cos, sin = self.rotary(T, x.device, q_rope.dtype)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos.to(k_rope.dtype), sin.to(k_rope.dtype))

        # --- Assemble full Q, K (rope ++ nope) ---
        q_full = torch.cat([q_rope, q_nope], dim=-1)          # (B, H, T, d)
        k_full = torch.cat([k_rope, k_nope], dim=-1)          # (B, H_kv, T, d)

        # --- Causal SDPA with GQA ---
        y = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=None, is_causal=True,
            enable_gqa=(H_kv != H),
        )                                                     # (B, H, T, d)

        # --- Output projection ---
        y = y.transpose(1, 2).reshape(B, T, H * d)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# 6b. Attention Mixture-of-Experts (MoEUT-style)
# ---------------------------------------------------------------------------
class MLAMoE(nn.Module):
    """Per-token Mixture-of-Experts over low-rank MLA attention experts.

    A MoEUT-style attention MoE used as the attention sub-component of an
    ``F``/``G`` delta block when ``attn_moe=True``. Each of the ``n_experts``
    experts is an independent :class:`MLAttention` (owning its OWN low-rank
    Q / KV-compression / decompression, K-rope, and ``o_proj``), so no trainable
    parameter is shared across experts (expert-independence invariant). The
    per-token combine uses the SAME smooth-routing family as the FFN MoE
    (:class:`SwiGLUMoE`): ``softmax`` (dense soft) or ``relu`` (smooth-sparse,
    exact zeros). NO top-k / argmax / capacity dispatch — every expert is
    always evaluated and combined with continuous per-token weights, so the
    block stays a deterministic *pure function of its input* and the reversible
    recurrence reconstructs it exactly (reversibility-safe).

    DeepSeek shared (always-on, ungated) attention experts are supported via
    ``num_shared_experts``: their outputs are SUMMED into every token
    unconditionally, in addition to the routed combine.

    The routing diagnostics / router-aux machinery mirror :class:`SwiGLUMoE`
    (``last_route`` / ``last_sparsity`` / ``_route_input`` and the entropy /
    load-balance / ReMoE-L1 recompute terms) so the same collector
    (:func:`_collect_moe_aux`) trains the attention router too.
    """

    def __init__(self, dim, n_heads, n_kv_heads, kv_latent, head_dim,
                 n_experts, q_latent=None, rope_base=10000.0,
                 router_type="softmax", num_shared_experts=0,
                 expert_b_init="small"):
        super().__init__()
        assert router_type in ("softmax", "relu"), (
            f"router_type must be 'softmax' or 'relu', got {router_type!r}"
        )
        assert num_shared_experts >= 0
        self.dim = dim
        self.n_experts = n_experts
        self.num_shared_experts = num_shared_experts
        self.router_type = router_type
        self.expert_b_init = expert_b_init

        self.router = nn.Linear(dim, n_experts)
        # Routed experts: one independent MLA per expert (own Q/KV/o weights).
        self.experts = nn.ModuleList([
            MLAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                        kv_latent=kv_latent, head_dim=head_dim,
                        q_latent=q_latent, rope_base=rope_base)
            for _ in range(n_experts)
        ])
        # DeepSeek shared (always-on) attention experts.
        self.shared_experts = nn.ModuleList([
            MLAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                        kv_latent=kv_latent, head_dim=head_dim,
                        q_latent=q_latent, rope_base=rope_base)
            for _ in range(num_shared_experts)
        ])
        if expert_b_init == "zero":
            for e in self.experts:
                nn.init.zeros_(e.o_proj.weight)

        self.last_route = None
        self.last_sparsity = None
        self._route_input = None

    # -- router map / aux terms: identical contract to SwiGLUMoE -------------
    def _router_weights(self, x):
        logits = self.router(x)
        if self.router_type == "softmax":
            return F.softmax(logits, dim=-1)
        return F.relu(logits)

    def _recompute_router_w(self):
        if self._route_input is None:
            return None
        return self._router_weights(self._route_input)

    def router_entropy(self):
        w = self._recompute_router_w()
        return None if w is None else SwiGLUMoE._route_entropy(self, w)

    def load_balance_term(self):
        w = self._recompute_router_w()
        return None if w is None else SwiGLUMoE._load_balance(self, w)

    def aux_l1_loadbalanced(self):
        w = self._recompute_router_w()
        if w is None:
            return None
        active = (w > 0).type_as(w)
        f_e = active.detach().mean(dim=(0, 1))
        mean_mass = w.mean(dim=(0, 1))
        return (f_e * mean_mass).mean()

    def forward(self, x):
        w = self._router_weights(x)                              # (B, T, E)
        self._route_input = x.detach()
        self.last_route = w.detach()
        self.last_sparsity = (w.detach() == 0).type_as(w).mean()

        # Routed: stack the per-expert attention outputs and soft-combine.
        # Each expert is a pure function of x; the combine is continuous in w.
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (B, T, E, dim)
        y = torch.einsum("bte,bted->btd", w.type_as(outs), outs)

        # Shared always-on experts: summed in, ungated.
        for e in self.shared_experts:
            y = y + e(x).type_as(y)
        return y


# ---------------------------------------------------------------------------
# 7. SwiGLU Mixture-of-Experts
# ---------------------------------------------------------------------------
class SwiGLUMoE(nn.Module):
    """Smooth, reversibility-safe SwiGLU Mixture-of-Experts FFN.

    Used as an ``F`` / ``G`` block inside :class:`ReversibleRecurrence`, so the
    routing MUST be a *continuous* function of the input — NO top-k, argmax,
    capacity, or hard-threshold dispatch. A discrete jump in which experts fire
    would break the reversible backward's recomputation stability (the inverse
    must reconstruct the identical forward argument). Every expert is therefore
    always computed and the outputs are combined with smooth per-token weights;
    "sparsity" comes only from the ReLU router emitting *exact zeros*, not from
    skipping any expert.

    Two router types (a planned control experiment, selected by ``router_type``):

      * ``softmax`` — dense-soft baseline: ``w = softmax(router(x))``. All weights
        strictly positive (no sparsity), always sums to 1 per token.
      * ``relu``    — ReMoE-style smooth-sparse: ``w = relu(router(x))``. The
        ReLU yields exact zeros (data-dependent sparsity / load shedding) while
        staying continuous and a.e.-differentiable.

    Experts are low-rank (LoRA-style) SwiGLU: each expert down-projects ``dim``
    to ``expert_rank``, applies SwiGLU there, and up-projects back to ``dim``, so
    the parameter count scales with ``expert_rank`` rather than a full hidden
    width. All experts are evaluated densely as batched tensors and combined as
    ``y = sum_e w[..., e] * expert_e(x)``.

    Detached diagnostics set after each forward (log / controller reads):
      * ``self.last_route``    — detached routing weights, shape ``(B, T, E)``.
      * ``self.last_sparsity`` — detached scalar, the realized sparsity (mean
        fraction of route weights ``== 0``) for the ReMoE adaptive controller to
        read ONCE per optimizer step at the step boundary (never in the hot loop).
      * ``self._route_input``  — detached block input; the router aux terms below
        are recomputed from it under grad (the recurrence forward runs no_grad).

    Composable collapse-prevention auxiliaries (a control-experiment bake-off; all
    OFF unless the train loop gates them with a positive coef). Each is RECOMPUTED
    from the saved detached router input UNDER GRAD (so it trains the router) and
    returned in-graph as a differentiable loss term:
      * :meth:`router_entropy` — mean-token entropy ``H(p)`` of the router dist
        (softmax weights, or relu weights renormalized to a distribution). The
        train loop MAXIMIZES it (``loss += -coef * H``) to spread mass back across
        experts (standard entropy regularization). Higher ``H`` => more uniform.
      * :meth:`load_balance_term` — Switch-Transformer load balance
        ``E * sum_e f_e * P_e`` (Fedus et al. 2021), with ``P_e`` the mean router
        prob mass on expert ``e`` (differentiable) and ``f_e`` the fraction of
        tokens whose argmax (top) expert is ``e`` (detached — argmax is used ONLY
        in this loss factor, NOT in the forward dispatch, so reversibility is
        untouched). Minimized to equalize usage.
      * :meth:`aux_l1_loadbalanced` — ReMoE load-balanced sparsity L1
        ``mean_e( f_e * mean_t route_{t,e} )``, scaled by the adaptive
        ``lambda_route`` (ReMoE arXiv 2412.14711).

    The ReMoE controller is what prevents router collapse for the relu router: a
    *fixed* L1 penalty monotonically drives every routing weight to zero
    (``active_frac -> 0``, MoE off). ReMoE instead adapts ``lambda_route`` to HOLD
    ``last_sparsity`` at a target ``S* = 1 - moe_target_active_frac`` (see
    :func:`_update_lambda_route`). Entropy / load-balance are independent,
    composable alternatives evaluated against this controller in the bake-off.

    bf16-friendly: no forced fp32 except inside ``RMSNorm`` statistics.
    """

    def __init__(self, dim, n_experts, expert_rank, router_type="softmax",
                 num_shared_experts=0, expert_b_init="small"):
        super().__init__()
        assert router_type in ("softmax", "relu"), (
            f"router_type must be 'softmax' or 'relu', got {router_type!r}"
        )
        assert expert_b_init in ("small", "zero"), (
            f"expert_b_init must be 'small' or 'zero', got {expert_b_init!r}"
        )
        assert num_shared_experts >= 0
        self.dim = dim
        self.n_experts = n_experts
        self.expert_rank = expert_rank
        self.router_type = router_type
        self.num_shared_experts = num_shared_experts
        self.expert_b_init = expert_b_init

        # Router is full-rank: dim -> n_experts logits. Routes ONLY the routed
        # experts; the shared experts (below) are always-on and ungated.
        self.router = nn.Linear(dim, n_experts)

        # Low-rank (LoRA-style) SwiGLU experts as batched parameter banks.
        # gate/up share a single down-projection to expert_rank, then SwiGLU
        # (silu(gate) * up) at rank, then up-project back to dim.
        #   x:(B,T,dim) @ w_in:(E,dim,2*rank) -> (B,T,E,2*rank) -> SwiGLU(rank)
        #   -> @ w_out:(E,rank,dim) -> (B,T,E,dim)
        self.w_in = nn.Parameter(torch.empty(n_experts, dim, 2 * expert_rank))
        self.w_out = nn.Parameter(torch.empty(n_experts, expert_rank, dim))
        nn.init.normal_(self.w_in, std=dim ** -0.5)
        # Routed-expert output-proj init (knob ``expert_b_init``): ``small`` keeps
        # a non-zero std so the router + experts get a finite task gradient from
        # step 0 (engagement fix); ``zero`` is the classic LoRA-B init (only
        # sensible with shared experts providing a base). M0GPT may RE-init w_out
        # later for the near-identity start; this is the module-local default.
        if expert_b_init == "zero":
            nn.init.zeros_(self.w_out)
        else:
            nn.init.normal_(self.w_out, std=expert_rank ** -0.5)

        # DeepSeek-style shared (always-on, ungated) experts: a separate bank of
        # ``num_shared_experts`` LoRA-SwiGLU experts whose outputs are SUMMED into
        # every token unconditionally (not multiplied by any router weight). They
        # are ALWAYS small-non-zero initialized so they provide a live base even
        # when ``expert_b_init='zero'`` zeroes the routed experts.
        if num_shared_experts > 0:
            self.shared_w_in = nn.Parameter(
                torch.empty(num_shared_experts, dim, 2 * expert_rank))
            self.shared_w_out = nn.Parameter(
                torch.empty(num_shared_experts, expert_rank, dim))
            nn.init.normal_(self.shared_w_in, std=dim ** -0.5)
            nn.init.normal_(self.shared_w_out, std=expert_rank ** -0.5)
        else:
            self.shared_w_in = None
            self.shared_w_out = None

        self.last_route = None
        self.last_sparsity = None
        # Detached router input saved each forward; the router aux terms (entropy,
        # Switch load-balance, ReMoE-L1) are recomputed from it UNDER GRAD so they
        # train the router even though the recurrence forward runs under no_grad.
        self._route_input = None

    def _route_entropy(self, w, eps=1e-9):
        """Mean-token entropy ``H(p)`` of the router distribution.

        ``w`` is the (B, T, E) routing weight tensor. For softmax routing ``w`` is
        already a per-token distribution; for relu routing we renormalize each
        token's weights to a distribution ``p = w / w.sum(-1)`` before the entropy
        (tokens whose route sums to ``0`` carry no mass and are skipped so they
        contribute ``0`` rather than ``NaN``). ``H(p) = -sum_e p_e log p_e`` per
        token; returns the mean over the live tokens (kept in-graph so the train
        loop can MAXIMIZE it). ``0`` if no token has any mass.
        """
        total = w.sum(dim=-1, keepdim=True)                     # (B, T, 1)
        live = (total.squeeze(-1) > 0).type_as(w)              # (B, T) has-mass
        p = w / total.clamp_min(eps)                            # (B, T, E) dist
        # 0*log(0) := 0 — mask exact zeros so they contribute nothing.
        logp = torch.where(p > 0, p.clamp_min(eps).log(),
                           torch.zeros_like(p))
        h_tok = -(p * logp).sum(dim=-1)                         # (B, T) per-token H
        # Mean over the LIVE tokens (those with mass); all-zero rows contribute 0.
        # Pure-tensor (clamp_min the count) so there is no hot-path CPU sync.
        n_live = live.sum().clamp_min(1.0)
        return (h_tok * live).sum() / n_live

    def _load_balance(self, w):
        """Switch-Transformer load-balance term ``E * sum_e f_e * P_e``.

        ``P_e`` = mean over tokens of the router weight for expert ``e``
        (differentiable). ``f_e`` = fraction of tokens whose argmax (top) expert
        is ``e`` (DETACHED — argmax is non-differentiable and is used ONLY in this
        loss factor, never in the forward dispatch, so reversibility is intact).
        Minimal (``== 1``) for a perfectly uniform load over ``E`` experts,
        maximal (``== E``) when one expert carries everything. Kept in-graph
        through ``P_e``.
        """
        E = w.shape[-1]
        wf = w.reshape(-1, E)                                   # (N, E) per token
        p_e = wf.mean(dim=0)                                    # (E,) mean mass
        top = wf.argmax(dim=-1)                                 # (N,) top expert
        f_e = torch.zeros(E, device=w.device, dtype=p_e.dtype)
        f_e.scatter_add_(0, top, torch.ones_like(top, dtype=p_e.dtype))
        f_e = (f_e / wf.shape[0]).detach()                     # (E,) top-frac
        return E * (f_e * p_e).sum()

    def _router_weights(self, x):
        """Routing weights ``w`` from input ``x`` (softmax dist, or relu sparse).

        Single source for the router map, shared by :meth:`forward` and the aux
        recompute path. Continuous in ``x`` (no dispatch) so reversibility holds.
        """
        logits = self.router(x)                                  # (B, T, E)
        if self.router_type == "softmax":
            return F.softmax(logits, dim=-1)
        return F.relu(logits)                                    # exact zeros

    def _recompute_router_w(self):
        """Recompute routing weights from the saved (detached) router input.

        The reversible recurrence runs every block forward under ``no_grad`` (it
        recomputes activations in backward for O(1) memory), so the routing
        weights computed during the cached forward are DETACHED — a router aux
        read from them would add NO gradient to the router. To make the router
        regularizers actually train the router, we re-run the (cheap) router map
        on the saved detached block input under the ambient grad mode; the
        gradient then flows to the ROUTER parameters (which is exactly what
        entropy / Switch load-balance / ReMoE-L1 regularizers target — the
        router, not the upstream block). Returns ``None`` if no forward has run.
        """
        if self._route_input is None:
            return None
        return self._router_weights(self._route_input)

    def router_entropy(self):
        """In-graph mean-token router entropy (recomputed -> trains the router)."""
        w = self._recompute_router_w()
        if w is None:
            return None
        return self._route_entropy(w)

    def load_balance_term(self):
        """In-graph Switch load-balance term (recomputed -> trains the router)."""
        w = self._recompute_router_w()
        if w is None:
            return None
        return self._load_balance(w)

    def aux_l1_loadbalanced(self):
        """In-graph ReMoE load-balanced sparsity aux (recomputed -> trains router).

        ``mean_e( f_e * mean_t route_{t,e} )`` with ``f_e`` the detached per-expert
        usage fraction (fraction of tokens with nonzero weight on ``e``) and the
        per-expert mean mass kept live, so the penalty shrinks the actual routing
        weights of over-used experts (prevents a single expert dominating). The
        train loop scales this by the adaptive ``lambda_route``.
        """
        w = self._recompute_router_w()
        if w is None:
            return None
        active = (w > 0).type_as(w)                              # (B, T, E)
        f_e = active.detach().mean(dim=(0, 1))                   # (E,) usage frac
        mean_mass = w.mean(dim=(0, 1))                           # (E,) mean mass
        return (f_e * mean_mass).mean()

    def forward(self, x):
        # --- Smooth router over the input (continuous in x; no dispatch) ---
        w = self._router_weights(x)                             # (B, T, E)

        # Save the (detached) router input so the router aux terms can be
        # recomputed UNDER GRAD by the loss assembly. The recurrence runs this
        # forward under no_grad (O(1)-memory reversibility), so the in-forward
        # routing weights carry no gradient; the recompute path is what makes the
        # entropy / load-balance / ReMoE-L1 regularizers actually train the router.
        self._route_input = x.detach()
        # Detached diagnostics (read at log / controller sites; no hot-path cost).
        self.last_route = w.detach()
        self.last_sparsity = (w.detach() == 0).type_as(w).mean()

        # --- Dense low-rank SwiGLU over ALL routed experts (no skipping) ---
        h = torch.einsum("btd,edr->bter", x, self.w_in)         # (B, T, E, 2*rank)
        gate, up = h.chunk(2, dim=-1)                            # each (B, T, E, rank)
        act = F.silu(gate) * up                                  # (B, T, E, rank)
        expert_out = torch.einsum("bter,erd->bted", act, self.w_out)  # (B, T, E, dim)

        # --- Smooth soft combine of the ROUTED experts: sum_e w[...,e]*expert_e ---
        y = torch.einsum("bte,bted->btd", w.type_as(expert_out), expert_out)

        # --- DeepSeek shared experts: always-on, ungated, SUMMED in ---
        if self.shared_w_in is not None:
            hs = torch.einsum("btd,sdr->btsr", x, self.shared_w_in)   # (B,T,S,2*rank)
            sg, su = hs.chunk(2, dim=-1)
            sa = F.silu(sg) * su                                       # (B,T,S,rank)
            shared_out = torch.einsum("btsr,srd->btd", sa, self.shared_w_out)
            y = y + shared_out.type_as(y)
        return y


# ---------------------------------------------------------------------------
# 8. Mixture-of-Softmaxes output head
# ---------------------------------------------------------------------------
class MoSHead(nn.Module):
    """Mixture-of-Softmaxes language-model head (Yang et al. 2018).

    A single softmax over ``W @ z`` is rank-bounded by ``dim`` ("the softmax
    bottleneck"). MoS instead mixes ``n_mix`` softmaxes, each computed from its
    own nonlinear context vector, yielding a higher-rank log-probability matrix
    at modest extra cost (cheap here since ``vocab`` is small). Output basis is
    ``out_embed``: Task 6's ``M0GPT`` ties ``out_embed.weight`` to the model's
    input token embedding, so MoSHead owns the parameter but does not assume
    untied weights.

    Forward (per mixture component ``k``)::

        h_k      = tanh(ctx_k(z))                  # context, shape (B, T, dim)
        logits_k = h_k @ out_embed.weight.T        # (B, T, vocab)
        p_k      = softmax(logits_k, dim=-1)
        pi       = softmax(gate(z), dim=-1)         # mixture weights (B, T, n_mix)
        p        = sum_k pi[..., k] * p_k           # valid distribution (rows sum to 1)
        return log(p.clamp_min(1e-12))             # log-probabilities (B, T, vocab)

    The mixture is a convex combination of per-component distributions, so ``p``
    is itself a distribution. bf16-friendly: the softmax / mixture arithmetic is
    done in fp32 internally for numerical stability and the fp32 result is
    returned (this casts only the head's tensors, not the whole model).
    """

    def __init__(self, dim, vocab, n_mix):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        self.n_mix = n_mix
        # Output token basis; Task 6 ties this to the input embedding.
        self.out_embed = nn.Embedding(vocab, dim)
        # Per-component context projections (one (dim->dim) map per mixture).
        self.ctx = nn.Linear(dim, n_mix * dim)
        # Mixture gate: dim -> n_mix logits.
        self.gate = nn.Linear(dim, n_mix)

    def forward(self, z):
        B, T, _ = z.shape
        K = self.n_mix
        # Context for every component, in fp32 for a stable softmax mixture.
        h = torch.tanh(self.ctx(z)).float()                  # (B, T, K*dim)
        h = h.view(B, T, K, self.dim)                        # (B, T, K, dim)
        w = self.out_embed.weight.float()                    # (vocab, dim)
        logits = torch.einsum("btkd,vd->btkv", h, w)         # (B, T, K, vocab)
        p_k = F.softmax(logits, dim=-1)                      # (B, T, K, vocab)
        pi = F.softmax(self.gate(z).float(), dim=-1)         # (B, T, K)
        p = torch.einsum("btk,btkv->btv", pi, p_k)           # (B, T, vocab)
        return p.clamp_min(1e-12).log()


# ---------------------------------------------------------------------------
# 9. Test-only delta block
# ---------------------------------------------------------------------------
class _TinyDelta(nn.Module):
    """Minimal delta block for reversibility tests (not a model component)."""

    def __init__(self, d):
        super().__init__()
        self.n = RMSNorm(d)
        self.l = nn.Linear(d, d, bias=False).double()

    def forward(self, x):
        return self.l(self.n(x))


# ---------------------------------------------------------------------------
# 10. LM scaffold
# ---------------------------------------------------------------------------
@dataclass
class Hyperparameters:
    """M0 model configuration (single source of truth for module shapes).

    Only model-architecture fields live here for Task 6; optimizer / data /
    schedule knobs are added by later tasks. ``mlp_mult`` is carried for spec
    parity (expert hidden-width scaling) but is unused while experts are
    parameterized purely by ``expert_rank``.
    """

    model_dim: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    n_experts: int
    expert_rank: int
    n_mix: int
    kv_latent: int
    head_dim: int
    q_latent: int | None = None
    router_type: str = "softmax"
    max_seq_len: int = 2048
    mlp_mult: float = 3.0  # spec-parity (expert hidden-width); unused for now
    # ReMoE adaptive sparsity controller target (relu router): the active
    # fraction the controller holds routing at; sparsity target S* = 1 - this.
    moe_target_active_frac: float = 0.5
    # --- Configurable recurrence-block structure (architecture-search axes) ---
    # All default to the CURRENT M0 behavior so existing runs are unchanged.
    # Each is reversibility-preserving (smooth routing, pure function of input).
    #   block_order: how attn / FFN-MoE compose INSIDE one delta sub-block.
    #     attn_ffn (current): a=attn(norm(inp)); out = a + moe(norm(inp+a))
    #     ffn_attn:           m=moe(norm(inp));  out = m + attn(norm(inp+m))
    #     parallel:           out = attn(norm(inp)) + moe(norm(inp))
    block_order: str = "attn_ffn"
    #   attn_moe: when True the attention is ALSO a (MoEUT-style) MoE over
    #     n_attn_experts low-rank MLA experts; when False it is a single MLA.
    attn_moe: bool = False
    n_attn_experts: int = 4
    #   num_shared_experts: DeepSeek always-on experts (per MoE), summed in
    #     ungated, in ADDITION to the n_experts routed experts.
    num_shared_experts: int = 0
    #   n_sublayers: each F/G delta block is a STACK of n_sublayers UNIQUE
    #     attn+MoE sub-blocks applied in sequence (pure delta). Trades more
    #     experts-in-one-sublayer vs fewer-experts x more-unique-sublayers.
    n_sublayers: int = 1
    #   expert_b_init: routed-expert up-proj (w_out) init. small = non-zero
    #     engagement default; zero = classic LoRA-B (needs shared experts base).
    expert_b_init: str = "small"


class _DeltaSubBlock(nn.Module):
    """One pre-norm attention + MoE *delta* sub-block (a single F/G stage).

    Returns the residual update for ITS input (not input + update); the caller
    (:class:`_PreNormDeltaBlock` or the recurrence) owns the additive coupling.
    The sub-block is a deterministic *pure function of its single input tensor*
    (no dropout / batch-coupled routing / external state) so the reversible
    inverse recomputes the identical update and reconstruction is exact in fp64.

    The attention is a single :class:`MLAttention` (default) or a
    :class:`MLAMoE` (when ``args.attn_moe``). The FFN is a :class:`SwiGLUMoE`.
    The two compose by ``args.block_order``:

      * ``attn_ffn`` (current/default): ``a=attn(norm(inp)); out=a+moe(norm(inp+a))``
      * ``ffn_attn``:                   ``m=moe(norm(inp));  out=m+attn(norm(inp+m))``
      * ``parallel``:                   ``out=attn(norm(inp))+moe(norm(inp))``

    All three are pure functions of ``inp`` (each branch reads only ``inp`` or a
    deterministic function of it), so reversibility is preserved for every order.
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim
        assert args.block_order in ("attn_ffn", "ffn_attn", "parallel"), (
            f"block_order must be attn_ffn/ffn_attn/parallel, got {args.block_order!r}"
        )
        self.block_order = args.block_order
        self.attn_norm = RMSNorm(d)
        self.mlp_norm = RMSNorm(d)
        if args.attn_moe:
            self.attn = MLAMoE(
                dim=d, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                kv_latent=args.kv_latent, head_dim=args.head_dim,
                n_experts=args.n_attn_experts, q_latent=args.q_latent,
                router_type=args.router_type,
                num_shared_experts=args.num_shared_experts,
                expert_b_init=args.expert_b_init,
            )
        else:
            self.attn = MLAttention(
                dim=d, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                kv_latent=args.kv_latent, head_dim=args.head_dim,
                q_latent=args.q_latent,
            )
        self.moe = SwiGLUMoE(
            dim=d, n_experts=args.n_experts, expert_rank=args.expert_rank,
            router_type=args.router_type,
            num_shared_experts=args.num_shared_experts,
            expert_b_init=args.expert_b_init,
        )

    def attn_modules(self):
        """Yield the underlying MLA module(s) (1 for single MLA, N for MLAMoE)."""
        if isinstance(self.attn, MLAMoE):
            yield from self.attn.experts
            yield from self.attn.shared_experts
        else:
            yield self.attn

    def moe_modules(self):
        """Yield the FFN MoE module(s) of this sub-block (one)."""
        yield self.moe

    def forward(self, inp):
        if self.block_order == "attn_ffn":
            a = self.attn(self.attn_norm(inp))
            return a + self.moe(self.mlp_norm(inp + a))
        if self.block_order == "ffn_attn":
            m = self.moe(self.mlp_norm(inp))
            return m + self.attn(self.attn_norm(inp + m))
        # parallel: both branches read the SAME input (norm(inp)).
        return self.attn(self.attn_norm(inp)) + self.moe(self.mlp_norm(inp))


class _PreNormDeltaBlock(nn.Module):
    """One ``F``/``G`` map: a stack of ``n_sublayers`` UNIQUE delta sub-blocks.

    With ``n_sublayers=1`` (default) this is exactly the previous single
    attention+MoE delta block. With ``n_sublayers>1`` it composes ``n_sublayers``
    independent :class:`_DeltaSubBlock` instances (each its own attn+MoE weights)
    into one recurrence-step delta:

        h = inp
        for sub in sublayers: h = h + sub(h)
        return h - inp                       # clean pure delta

    The recurrence owns the outer additive coupling (it adds ``F(...)`` to ``a``),
    so this block returns the residual update for ``inp``. The composition is a
    deterministic pure function of ``inp`` (each ``sub`` is pure and the running
    ``h`` is a deterministic function of ``inp``), so reversibility / exact fp64
    reconstruction is preserved for any ``n_sublayers``. The ``- inp`` makes it a
    clean delta (``inp + F(inp) == h``), matching the recurrence's additive form.
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        assert args.n_sublayers >= 1, "n_sublayers must be >= 1"
        self.sublayers = nn.ModuleList(
            [_DeltaSubBlock(args) for _ in range(args.n_sublayers)]
        )

    def forward(self, inp):
        h = inp
        for sub in self.sublayers:
            h = h + sub(h)
        return h - inp


class M0GPT(nn.Module):
    """M0 reversible recurrent-depth GPT.

    Pipeline::

        x0  = tok_emb(tokens) + pos_emb[:, :T]
        z_K = run_reversible(x0, depth=K)        # midpoint of the reversible pair
        z   = final_norm(z_K)                    # readout norm (OUTSIDE recurrence)
        log p = mos_head(z)                      # LOG-probabilities (MoS)
        loss  = nll_loss(log p, targets)         # MoS emits log-probs -> NLL

    The two recurrence maps ``F``/``G`` are pre-norm MLA+MoE delta blocks
    (:class:`_PreNormDeltaBlock`); each is a deterministic pure function of its
    input, which is what makes the additive-coupling recurrence exactly
    invertible (reconstruction gate, fp64). The MoS output basis is *tied* to
    the input token embedding (shared ``nn.Parameter``).

    ``final_norm`` is applied to the OUTPUT of ``run_reversible`` — strictly
    OUTSIDE the reversible recurrence (``RevRecurrenceFn``/``ReversibleRecurrence``)
    — so reversibility / exact fp64 reconstruction is UNAFFECTED. It is required
    because ``z_K = 0.5*(a_K + b_K)`` accumulates ``depth`` additive updates from
    ``x0`` and so grows with depth; feeding it unnormalized into the MoS head's
    ``tanh(ctx(z_K))`` saturates the tanh, zeroing its gradient and blocking
    learning everywhere upstream. NEVER add a norm INSIDE the recurrence loop —
    that would break the algebraic inverse.
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        d, V = args.model_dim, args.vocab_size

        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, args.max_seq_len, d))

        F_block = _PreNormDeltaBlock(args)
        G_block = _PreNormDeltaBlock(args)
        self.rec = ReversibleRecurrence(F_block, G_block)

        # Readout norm of the reversible midpoint, applied OUTSIDE the recurrence
        # so reconstruction stays exact (see class docstring).
        self.final_norm = RMSNorm(d)
        self.mos_head = MoSHead(d, V, args.n_mix)
        # Tie the MoS output basis to the input token embedding (shared Param).
        self.mos_head.out_embed.weight = self.tok_emb.weight

        # Small, conservative init so the model starts near-uniform (initial NLL
        # ~ ln(vocab)), not ~4x worse. Defaults (~N(0,1) embedding, default Linear
        # init) gave huge logits AND a large z_K that saturated the head's tanh.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)  # also sets tied out_embed
        # MoS gate zero-init -> mixture weights start uniform; ctx small-std ->
        # tanh(ctx(z)) starts near 0 so per-component logits start near-uniform.
        nn.init.zeros_(self.mos_head.gate.weight)
        nn.init.zeros_(self.mos_head.gate.bias)
        nn.init.normal_(self.mos_head.ctx.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mos_head.ctx.bias)
        # Start the recurrence near-identity: zero EVERY delta sub-block's
        # ATTENTION output projection(s) so z_K starts close to x0 (no compounding
        # before training) — this is what keeps the readout stable at init. With
        # attn_moe this covers each routed AND shared MLA expert's o_proj; with a
        # single MLA it is the lone o_proj. Iterates over all sub-blocks of both
        # F and G so the near-identity start holds for any n_sublayers.
        for blk in (self.rec.F, self.rec.G):
            for sub in blk.sublayers:
                for mla in sub.attn_modules():
                    nn.init.zeros_(mla.o_proj.weight)
                # MoE routed-expert up-proj (w_out) init: respect expert_b_init.
                #   small (default): a small non-zero w_out keeps the init delta
                #     tiny (z_K stays near x0, init NLL ~ ln(vocab)) while giving
                #     each expert AND the router a meaningful gradient from step 0.
                #     A zero w_out would make every expert output 0, so the router
                #     weights multiply a zero and receive ZERO task gradient (inert
                #     MoE; bit-identical val_bpb across aux variants).
                #   zero: classic LoRA-B init (routed w_out stays at 0); only
                #     sensible with num_shared_experts>0 providing a live base.
                # Reversibility is unaffected (the reverse pass reconstructs
                # whatever w_out holds; the recurrence inverse is exact for any
                # deterministic F/G). std matches the small-init convention above.
                for moe in sub.moe_modules():
                    if args.expert_b_init != "zero":
                        nn.init.normal_(moe.w_out, std=0.02)

    def forward(self, tokens, targets, depth):
        B, T = tokens.shape
        x0 = self.tok_emb(tokens) + self.pos_emb[:, :T]
        z_K = self.rec.run_reversible(x0, depth)
        z = self.final_norm(z_K)  # readout norm, OUTSIDE the recurrence
        logp = self.mos_head(z)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("non-finite MoS log-probabilities")
        loss = F.nll_loss(logp.reshape(-1, self.args.vocab_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        return loss


# ---------------------------------------------------------------------------
# 11. Muon optimizer (ported verbatim from the a15093a clean scaffold)
# ---------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.

    Muon uses this to normalize matrix-shaped gradients before applying them.
    Ported from ``train_gpt.py`` (a15093a) without the ``torch.compile`` wrap so
    the M0 trainer stays compile-free for dev (CLAUDE.md standing directive).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def zeropower_via_newtonschulz5_batched(G, steps=5, eps=1e-7):
    """Batched Newton-Schulz: orthogonalize a stack of matrices ``(..., M, N)``.

    Vectorized (batched matmul over the last two dims) counterpart of
    :func:`zeropower_via_newtonschulz5`, used by :class:`Muon` for the 3-D MoE
    expert banks (``w_in (E, dim, 2*rank)`` / ``w_out (E, rank, dim)``). Each
    matrix is normalized independently by its own Frobenius norm over the last
    two dims, so the iteration is scale-invariant per matrix. Same coefficients
    and step count as the 2-D path, kept bf16-friendly and compile-free.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    norms = X.flatten(-2).norm(dim=-1)                 # (...,)
    X = X / (norms[..., None, None] + eps)
    # All matrices in a homogeneous stack share the same (M, N); a single
    # transpose decision mirrors the 2-D path's ``size(0) > size(1)`` guard.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    Xt = lambda T: T.transpose(-1, -2)
    for _ in range(steps):
        A = X @ Xt(X)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return Xt(X) if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon (modded-nanogpt). Orthogonalized momentum SGD for matrix params.

    Ported from a15093a. Handles 2-D matrices (attn/MoS projections) via the
    2-D Newton-Schulz and 3-D MoE expert banks ``(E, M, N)`` via the batched NS
    (orthogonalizing each expert's matrix independently). DDP-aware: each rank
    orthogonalizes a disjoint slice of the params, then a single
    ``all_reduce(SUM)`` stitches the flattened updates (the per-param update has
    the same shape as the param, so 3-D banks ride the same stitch logic).
    """

    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if g.ndim == 2:
                        g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    else:
                        # 3-D expert banks (E, M, N): batched NS over last 2 dims.
                        g = zeropower_via_newtonschulz5_batched(g, steps=backend_steps)
                    # RMS-match scale: aspect-ratio factor over the matrix dims
                    # (last two), shared across any leading batch dim.
                    g = g * (max(1, g.size(-2) / g.size(-1)) ** 0.5)
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# ---------------------------------------------------------------------------
# 12. Data loading (DDP-sharded fineweb loader + tiny-synthetic fallback)
# ---------------------------------------------------------------------------
def load_data_shard(file: Path):
    """Read a fineweb ``.bin`` shard (256-int header + uint16 tokens) memmapped.

    Ported from ``train_gpt.py``: memmap so DDP ranks share OS page cache; the
    read-only-buffer UserWarning is suppressed at the wrap site (we never mutate).
    """
    import warnings as _warnings

    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_mmap = np.memmap(
        file, dtype="<u2", mode="r", offset=header_bytes, shape=(num_tokens,)
    )
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        return torch.from_numpy(tokens_mmap.view(np.uint16))


class TokenStream:
    """Sequential, wrap-around reader over a shard glob (no workers/sampling)."""

    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class _SyntheticTokenStream:
    """In-memory random-token stream for CPU smoke tests (no big dataset).

    Emits uniform tokens in ``[0, vocab)`` so the trainer/eval run end-to-end
    without the read-only fineweb shards. Same ``take(n)`` contract as
    :class:`TokenStream`; deterministic via a seeded generator.
    """

    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.gen = torch.Generator().manual_seed(seed)

    def take(self, n: int):
        return torch.randint(
            0, self.vocab_size, (n,), generator=self.gen, dtype=torch.int64
        )


class DistributedTokenLoader:
    """DDP next-token loader: each call slices one disjoint span per rank.

    Ported from a15093a. The extra ``+1`` token per rank-span lets us build
    ``(x, y)`` by shifting. ``stream`` is either a real :class:`TokenStream`
    (fineweb shards) or a :class:`_SyntheticTokenStream` (smoke).
    """

    def __init__(self, stream, rank: int, world_size: int, device):
        self.stream = stream
        self.rank = rank
        self.world_size = world_size
        self.device = device

    @classmethod
    def from_pattern(cls, pattern, rank, world_size, device):
        return cls(TokenStream(pattern), rank, world_size, device)

    @classmethod
    def synthetic(cls, vocab_size, rank, world_size, device, seed=0):
        return cls(_SyntheticTokenStream(vocab_size, seed), rank, world_size, device)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# 13. BPB evaluation (tokenizer-agnostic bits-per-byte)
# ---------------------------------------------------------------------------
def build_sentencepiece_luts(sp, vocab_size, device):
    """Per-token byte-length LUTs for the BPB metric (ported from a15093a)."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def run_validation(model, loader, depth, n_batches, seq_len, global_tokens,
                   grad_accum_steps, device, luts, autocast_enabled):
    """Compute ``(val_loss_nats, val_bpb)`` over ``n_batches`` batches at ``depth``.

    ``model(x, y, depth)`` returns mean per-token NLL in nats (M0GPT forward).
    BPB = (val_loss / ln2) * tokens_per_byte, where bytes come from the
    SentencePiece LUTs. When ``luts is None`` (synthetic smoke, no tokenizer)
    BPB is reported as ``nan`` but the loss path is still exercised.
    """
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        luts if luts is not None else (None, None, None)
    )
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    autocast_dtype = torch.bfloat16
    with torch.inference_mode():
        for _ in range(n_batches):
            x, y = loader.next_batch(global_tokens, seq_len, grad_accum_steps)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_enabled):
                batch_loss = model(x, y, depth).detach()
            n_tok = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n_tok
            tok_count += n_tok
            if base_bytes_lut is not None:
                tgt_ids = y.reshape(-1)
                prev_ids = x.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes = token_bytes + (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    if was_training:
        model.train()
    val_loss = (loss_sum / tok_count).item()
    bits_per_token = val_loss / math.log(2.0)
    if base_bytes_lut is not None and byte_count.item() > 0:
        tokens_per_byte = tok_count.item() / byte_count.item()
        val_bpb = bits_per_token * tokens_per_byte
    else:
        val_bpb = float("nan")
    return float(val_loss), float(val_bpb)


# ---------------------------------------------------------------------------
# 14. int6 artifact codec (per-row SDClip int6 + zstd, 16 MB check)
# ---------------------------------------------------------------------------
# Ported from current train_gpt.py (replaces the a15093a int8 path). Single
# supported export format: per-row SDClip int6 for >8192-elem float tensors,
# fp16 passthrough for the tied embedding, exact passthrough for small/non-float.
# Shared tensors (the tied tok_emb / out_embed weight) are deduped by data_ptr:
# stored once, with the duplicate key recorded as an alias and re-tied on load.
CONTROL_TENSOR_PATTERNS = ("norm.w", "router.bias", "gate.bias")
FP16_KEEP_PATTERNS = ("tok_emb", "out_embed")
SDCLIP_K_MATRIX = 12.85
SDCLIP_K_EMBED = 20.0
INT6_CLIP = 31
INT6_CATEGORIES = {"matrix", "embed"}


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "out_embed" in name or "lm_head" in name:
        return "embed"
    return "matrix"


def _sdclip_scale(t, k):
    if t.ndim >= 2:
        row_std = t.float().std(dim=-1)
        clip_abs = k * row_std
        return (clip_abs / INT6_CLIP).clamp_min(1e-12).to(torch.float16)
    amax = t.float().abs().max().item()
    return torch.tensor(max(amax / INT6_CLIP, 1e-12), dtype=torch.float16)


def quantize_int6_sdclip(t, k=SDCLIP_K_MATRIX):
    t32 = t.float()
    if t32.ndim >= 2:
        s = _sdclip_scale(t32, k).clamp_min(torch.finfo(torch.float16).tiny)
        s_expand = s.float().view(-1, *([1] * (t32.ndim - 1)))
        t_2d = t32.reshape(-1, t32.shape[-1]) if t32.ndim > 2 else t32
        s_2d = s_expand.reshape(-1, 1) if t32.ndim > 2 else s_expand
        q = torch.clamp(torch.round(t_2d / s_2d), -(INT6_CLIP + 1), INT6_CLIP).to(torch.int8)
        if t32.ndim > 2:
            q = q.view(t32.shape)
        return q, s
    s = _sdclip_scale(t32, k)
    q = torch.clamp(torch.round(t32 / s.float()), -(INT6_CLIP + 1), INT6_CLIP).to(torch.int8)
    return q, s


def mixed_quantize_int6(state_dict, int6_cats):
    result = {}
    meta = {}
    # Fix M1: dedup shared tensors (the tied tok_emb / out_embed weight is one
    # physical Parameter). The first key to claim a ``data_ptr`` is serialized;
    # any later key sharing it stores no payload and records an ``alias`` to the
    # owner, resolved on dequant.
    seen_ptrs = {}
    for name, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            meta[name] = {"type": "alias", "to": seen_ptrs[ptr]}
            continue
        seen_ptrs[ptr] = name
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 8192:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(p in name for p in FP16_KEEP_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            k = SDCLIP_K_EMBED if cat == "embed" else SDCLIP_K_MATRIX
            q, s = quantize_int6_sdclip(t, k=k)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_int6_sdclip(t, k=SDCLIP_K_MATRIX)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
    return result, meta


def dequantize_mixed_int6(result, meta, template_sd):
    out = {}
    aliases = {}  # name -> owner name (Fix M1: tied/shared tensors)
    for name, orig in template_sd.items():
        info = meta[name]
        orig_dtype = orig.dtype
        if isinstance(info, dict) and info.get("type") == "alias":
            aliases[name] = info["to"]
            continue
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        orig_shape = orig.shape
        if s.ndim > 0:
            q_2d = q.view(-1, q.shape[-1]) if q.ndim > 2 else q
            deq = q_2d.float() * s.float().view(q_2d.shape[0], *([1] * (q_2d.ndim - 1)))
            out[name] = deq.view(orig_shape).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).view(orig_shape).to(orig_dtype)
    # Resolve aliases against the already-dequantized owner (re-tie shared
    # tensors), casting to each alias's own template dtype.
    for name, owner in aliases.items():
        out[name] = out[owner].to(template_sd[name].dtype)
    return out


def save_int6_artifact(state_dict):
    """Quantize + serialize + compress (per-row int6 + zstd-22 / zlib-9)."""
    qsd, meta = mixed_quantize_int6(state_dict, INT6_CATEGORIES)
    buf = io.BytesIO()
    torch.save({"state_dict": qsd, "meta": meta}, buf)
    raw_bytes = buf.getvalue()
    if _COMPRESSOR == "zstd":
        compressed = zstandard.ZstdCompressor(level=22).compress(raw_bytes)
    else:
        compressed = zlib.compress(raw_bytes, 9)
    return compressed, qsd, meta


def load_int6_artifact(blob, template_state_dict):
    """Decompress + deserialize + dequantize an artifact from save_int6_artifact."""
    if _COMPRESSOR == "zstd":
        decompressed = zstandard.ZstdDecompressor().decompress(blob)
    else:
        decompressed = zlib.decompress(blob)
    payload = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    return dequantize_mixed_int6(payload["state_dict"], payload["meta"], template_state_dict)


# ---------------------------------------------------------------------------
# 15. Finite-horizon loss + optimizer builder
# ---------------------------------------------------------------------------
# Router-bearing MoE module types whose router-aux terms the collectors sum.
# Both the FFN MoE (:class:`SwiGLUMoE`) and the optional attention MoE
# (:class:`MLAMoE`, when ``attn_moe``) expose the same aux contract
# (``router_entropy`` / ``load_balance_term`` / ``aux_l1_loadbalanced`` via
# ``_recompute_router_w``), so both train their routers through these collectors.
_ROUTER_MOE_TYPES = (SwiGLUMoE, MLAMoE)


def _collect_moe_aux(model, method):
    """Sum a router-aux term (named ``method``) over the router-bearing MoE blocks.

    Walks ``model.modules()`` so it works whether or not the model is DDP- or
    compile-wrapped (caller passes the inner module). Each aux is RECOMPUTED in
    the ambient grad mode from the block's saved detached router input (so the
    term trains the router; see :meth:`SwiGLUMoE._recompute_router_w`). Covers
    both the FFN MoE and the optional attention MoE (``_ROUTER_MOE_TYPES``).
    Skips blocks that have not run a forward yet (the method returns ``None``).
    Returns a 0.0 tensor on the model's device when no MoE produced a forward,
    keeping loss assembly type-stable. Single helper drives every collector (DRY).
    """
    total = None
    for m in model.modules():
        if isinstance(m, _ROUTER_MOE_TYPES):
            v = getattr(m, method)()
            if v is not None:
                total = v if total is None else total + v
    if total is None:
        p = next(model.parameters())
        return torch.zeros((), device=p.device, dtype=torch.float32)
    return total


def _collect_aux_lb(model):
    """Sum the in-graph ReMoE load-balanced L1 aux over the MoE blocks."""
    return _collect_moe_aux(model, "aux_l1_loadbalanced")


def _collect_router_entropy(model):
    """Sum the in-graph router entropy over forward-run SwiGLUMoE blocks."""
    return _collect_moe_aux(model, "router_entropy")


def _collect_load_balance(model):
    """Sum the in-graph Switch load-balance term over the MoE blocks."""
    return _collect_moe_aux(model, "load_balance_term")


def recurrence_displacement(model, tokens, depth) -> list[float]:
    """Per-step relative recurrence displacement ``||z_{k+1}-z_k|| / ||z_k||``.

    An EFFECTIVE-DEPTH diagnostic: it measures how much the reversible-recurrence
    midpoint ``z_k = 0.5*(a_k + b_k)`` (already produced by
    :meth:`ReversibleRecurrence.forward_states`) moves at each depth step.

      * A SUSTAINED (non-decaying) displacement => the recurrence keeps doing
        work at depth, i.e. high effective depth.
      * A rapid DECAY to ~0 => the state saturates early and extra depth buys
        nothing (low effective depth).

    Computed under ``no_grad`` from a single forward over the recurrence's
    midpoint sequence, seeded as the model does (``a0 = b0 = x0``), so ``z_0`` is
    the injected input ``x0`` and the returned list has length ``depth`` (one
    relative displacement per step). The Frobenius norm is taken over the full
    ``(B, T, D)`` state. ``||z_k||`` is floored by a tiny epsilon so an all-zero
    state yields ``0.0`` rather than a divide-by-zero. This is a LOG-SITE read
    (it runs an extra forward and CPU-syncs); never call it in the grad-accum
    hot loop.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x0 = model.tok_emb(tokens) + model.pos_emb[:, :tokens.shape[1]]
            # Midpoint sequence: z_0 = x0 (the seed), then one z per depth step.
            _, states = model.rec.forward_states(x0, x0, x0, depth)
            seq = [x0, *states]
            disp = []
            for k in range(len(seq) - 1):
                num = float((seq[k + 1] - seq[k]).norm())
                den = float(seq[k].norm())
                disp.append(num / max(den, 1e-12))
    finally:
        if was_training:
            model.train()
    return disp


def displacement_tail(disp) -> float:
    """``disp_tail`` headline: mean per-step displacement over the LAST HALF of
    the steps. A high tail => the recurrence is still moving deep in the budget
    (high effective depth); a near-zero tail => early saturation. Returns NaN for
    an empty list (no forward yet)."""
    if not disp:
        return float("nan")
    half = len(disp) // 2
    tail = disp[half:]
    return float(sum(tail) / len(tail))


def measured_sparsity(model) -> float:
    """Mean realized routing sparsity over forward-run SwiGLUMoE blocks.

    Sparsity is the mean fraction of route weights ``== 0`` (so dense softmax
    routing reads ~0, ReLU routing reads its realized load-shedding). Reads each
    block's detached ``last_sparsity`` scalar, so this is ONE CPU sync at the
    step boundary — never call it inside the grad-accum micro loop. Returns
    ``0.0`` when no MoE has produced a forward yet (controller no-op).
    """
    vals = [m.last_sparsity for m in model.modules()
            if isinstance(m, SwiGLUMoE) and m.last_sparsity is not None]
    if not vals:
        return 0.0
    return float(torch.stack([v.float() for v in vals]).mean())


def _update_lambda_route(lambda_route, s_measured, s_target, alpha=1.2,
                         lo=1e-8, hi=1e3):
    """ReMoE adaptive sparsity controller (arXiv 2412.14711, Eq. for lambda).

        lambda <- clamp( lambda * alpha ** sign(S_measured - S_target), lo, hi )

    A *fixed* L1 penalty drives every routing weight to zero (router collapse).
    ReMoE instead targets a sparsity level ``S* = 1 - moe_target_active_frac``:
    when routing is TOO sparse (``S_measured > S_target``) it LOWERS lambda so
    experts re-activate; when TOO dense (``S_measured < S_target``) it RAISES
    lambda. The scalar is a plain Python float updated once per optimizer step
    from a single reduced sparsity measurement, and clamped to ``[lo, hi]`` so it
    never runs away or vanishes. Returns the new float.
    """
    if s_measured > s_target:
        lam = lambda_route / alpha
    elif s_measured < s_target:
        lam = lambda_route * alpha
    else:
        lam = lambda_route
    return float(min(max(lam, lo), hi))


def finite_horizon_loss(model, x, y, k_hi, k_lo, lambda_h, margin, lambda_route,
                        use_load_balance=True, entropy_coef=0.0,
                        loadbalance_coef=0.0):
    """Finite-horizon no-degradation loss + composable router auxiliaries.

        L = L_hi + lambda_h * relu(L_hi - sg(L_lo) + margin)
              + lambda_route * aux_lb                 (ReMoE adaptive L1, if on)
              - entropy_coef * H                      (entropy reg: MAXIMIZE H)
              + loadbalance_coef * LB                 (Switch load-balance)

    ``L_hi = model(x, y, k_hi)`` is the deep pass that carries the task gradient.
    ``L_lo = model(x, y, k_lo)`` is a shallow pass; inside the hinge it is
    stop-gradient (``sg``) so the no-degradation pressure pushes the deep state to
    be no worse than the shallow one WITHOUT backpropping into the shallow pass.

    The three collapse-prevention router auxiliaries are an EMPIRICAL bake-off
    (each independently gated, default OFF) — every term is summed over the MoE
    blocks from the L_hi forward (the last forward run):
      * ``aux_lb`` — ReMoE load-balanced sparsity term, scaled by the *adaptive*
        ``lambda_route`` (see :func:`_update_lambda_route`); ``use_load_balance``
        toggles it for explicit ablation (when ``False`` the term is dropped and
        ``lambda_route`` ignored).
      * ``H`` (router entropy) — SUBTRACTED with ``entropy_coef >= 0`` so the loss
        MAXIMIZES entropy (spreads routing mass; standard entropy reg).
      * ``LB`` (Switch load-balance, Fedus et al. 2021) — ADDED with
        ``loadbalance_coef >= 0`` to equalize per-expert usage.

    Returns ``(loss, parts)`` exposing the components for testing/logging.
    """
    l_lo = model(x, y, k_lo).detach()  # stop-grad shallow pass
    l_hi = model(x, y, k_hi)           # deep pass: runs last so aux reflects it
    hinge = torch.relu(l_hi - l_lo + margin)
    loss = l_hi + lambda_h * hinge

    # Each router aux RECOMPUTES the router map (see SwiGLUMoE), so collect a term
    # only when it actually enters the loss; an inactive term reports an in-graph
    # 0.0 in ``parts`` (for logging/tests) without the redundant router matmul.
    def _zero():
        return torch.zeros((), device=l_hi.device, dtype=l_hi.dtype)
    aux_lb = _collect_aux_lb(model) if use_load_balance else _zero()
    router_entropy = _collect_router_entropy(model) if entropy_coef != 0.0 else _zero()
    load_balance = _collect_load_balance(model) if loadbalance_coef != 0.0 else _zero()
    if use_load_balance:
        loss = loss + lambda_route * aux_lb
    if entropy_coef != 0.0:
        loss = loss - entropy_coef * router_entropy   # MAXIMIZE entropy
    if loadbalance_coef != 0.0:
        loss = loss + loadbalance_coef * load_balance
    parts = {
        "l_hi": l_hi, "l_lo_sg": l_lo, "aux_lb": aux_lb, "hinge": hinge,
        "router_entropy": router_entropy, "load_balance": load_balance,
    }
    return loss, parts


def build_optimizers(model, matrix_lr, embed_lr, scalar_lr,
                     beta1=0.9, beta2=0.95, adam_eps=1e-8,
                     muon_momentum=0.95, muon_backend_steps=5):
    """Split trainable params into Muon (matrix banks) + AdamW (embed/scalars/router).

    Coverage contract (CLAUDE.md audit): every trainable parameter lands in
    EXACTLY one group. The tied ``tok_emb.weight`` / ``out_embed.weight`` is one
    physical Parameter, so we deduplicate by ``id`` to avoid double-listing it.

    Group assignment:
      * embedding (``tok_emb.weight``, ``pos_emb``)            -> AdamW @ embed_lr
      * router / gate / 1D scalar params (norm scales, biases) -> AdamW @ scalar_lr
      * remaining matrices, ``ndim >= 2`` (attn/MoS projections AND the 3-D MoE
        expert banks ``w_in``/``w_out`` — the effective-depth basis)
                                                              -> Muon  @ matrix_lr

    The 3-D expert banks are orthogonalized per-expert by Muon's batched
    Newton-Schulz; routing them to Muon (not AdamW) is Fix I1.
    """
    embed_ids = set()
    embed_params = []
    for p in (model.tok_emb.weight, model.pos_emb):
        if id(p) not in embed_ids and p.requires_grad:
            embed_ids.add(id(p))
            embed_params.append(p)

    matrix_params, scalar_params = [], []
    seen = set(embed_ids)
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_router = ("router" in name) or ("gate" in name)
        if p.ndim >= 2 and not is_router:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    optimizers = []
    optimizers.append(torch.optim.AdamW(
        [{"params": embed_params, "lr": embed_lr, "base_lr": embed_lr}],
        betas=(beta1, beta2), eps=adam_eps,
    ))
    if matrix_params:
        opt_muon = Muon(matrix_params, lr=matrix_lr, momentum=muon_momentum,
                        backend_steps=muon_backend_steps)
        for group in opt_muon.param_groups:
            group["base_lr"] = matrix_lr
        optimizers.append(opt_muon)
    optimizers.append(torch.optim.AdamW(
        [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
        betas=(beta1, beta2), eps=adam_eps,
    ))
    return optimizers


def adamw_params(model, optimizers):
    """Trainable params NOT managed by Muon (i.e. the AdamW-group params).

    These are the params whose grads we all-reduce-MEAN across ranks (Muon syncs
    its own matrix *updates*, not grads). Their synced grads have a consistent
    global norm across ranks, so they are the only safe scope for grad clipping
    (Fix I2): clipping over Muon params too would compute a rank-inconsistent
    scale from unsynced per-rank grads. Muon's update is scale-invariant (NS
    re-normalizes), so its params neither need nor should drive the clip.
    """
    muon_ids = {
        id(p) for opt in optimizers if isinstance(opt, Muon)
        for group in opt.param_groups for p in group["params"]
    }
    return [p for p in model.parameters()
            if p.requires_grad and id(p) not in muon_ids]


# ---------------------------------------------------------------------------
# 15b. Two-goal metrics (Resource + Expressiveness)
# ---------------------------------------------------------------------------
# The project tracks two goals with principled, named metrics. These are PURE
# functions (no model state mutation, no hot-path .item()/CPU syncs — call them
# only at log/eval sites). They are the primitives the control-experiment runner
# (Task 9) composes into the recurrence-equivalence exponent ``phi`` over the
# loop count ``r`` and the activation-memory scaling ``R_act`` over the recurrent
# budget ``K``; here we also log the cheap per-step subset — the RESOURCE goal
# (memory-efficiency: ``peak_vram`` primary, plus ``kv_bytes`` / ``params``) and
# the expressiveness ``erank``. ``active_frac`` stays as a MoE-mechanism
# diagnostic (``diag:active_frac``), no longer framed as a resource-goal metric.
def effective_rank(matrix_or_singular_values) -> float:
    """Effective (numerical) rank via spectral entropy — Roy & Vetterli (2007).

    Given a 2D matrix we take its singular values ``s = svdvals(M)``; given a 1D
    tensor we treat it directly as a singular-value spectrum. Normalize to a
    distribution ``p = s / sum(s)`` (the singular values are non-negative), take
    its Shannon entropy ``H = -sum(p log p)`` (zeros contribute 0), and return
    ``erank = exp(H)``. This is the exponential of the spectral entropy:

      * an isotropic spectrum (all ``s`` equal) has the maximal-entropy uniform
        ``p`` over ``n`` atoms, so ``erank == n`` (e.g. ``eye(4)`` -> 4);
      * a rank-1 spectrum concentrates all mass on one atom, so ``H == 0`` and
        ``erank == 1``.

    erank is a smooth, basis-free proxy for "how many directions the matrix
    actually uses", which we read as an expressiveness signal on expert / MoS /
    recurrent-state output spaces. Returns a Python ``float`` (single CPU sync;
    do not call inside the grad-accum hot loop).
    """
    t = matrix_or_singular_values.detach()
    if t.ndim >= 2:
        s = torch.linalg.svdvals(t.float())
    else:
        s = t.float().abs()
    total = s.sum()
    if float(total) <= 0.0:
        return 0.0
    p = s / total
    # Mask exact zeros so 0*log(0) := 0 (the entropy convention).
    nz = p > 0
    h = -(p[nz] * p[nz].log()).sum()
    return float(torch.exp(h))


def fit_phi(losses: dict) -> float:
    """Iso-Depth recurrence-equivalence exponent ``phi`` in ``[0, 1]``.

    Question: looping one shared block ``r`` times — does it buy the capacity of
    ``r`` *unique* blocks? We model the depth scaling law in log-loop space::

        loss(r) ≈ a - b * log(r),     b >= 0  (improvement per log-loop)

    and compare the fitted slope ``b`` against the "one unique block per loop"
    reference slope ``b_ref``. The exponent is the capped ratio::

        phi = clamp(b / b_ref, 0, 1)

    Interpretation: ``phi == 1`` means each extra loop is worth a full unique
    block (perfect recurrence-equivalence); ``phi == 0`` means looping buys
    nothing (loss flat in ``r``); intermediate values quantify partial
    equivalence. We take ``b_ref = 1.0`` so a unit log-loop improvement maps to
    ``phi = 1`` (the test's full-equivalence construction uses unit slope), and
    negative slopes (loss *worsening* with depth) clamp to 0.

    The fit is an ordinary least-squares slope of ``loss`` against ``log(r)``,
    which is deterministic and closed-form (no optimizer, no randomness). With
    fewer than two distinct depths there is no slope to estimate, so ``phi`` is
    defined as ``0.0``.
    """
    pts = sorted((int(r), float(v)) for r, v in losses.items() if int(r) > 0)
    if len(pts) < 2:
        return 0.0
    xs = [math.log(r) for r, _ in pts]
    ys = [v for _, v in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:  # all log(r) identical (cannot happen for distinct r>0, guard anyway)
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx          # dloss/dlog(r); negative when loss improves with depth
    b = -slope                 # improvement per log-loop (>=0 when looping helps)
    b_ref = 1.0                # one unique block per log-loop reference slope
    return float(min(max(b / b_ref, 0.0), 1.0))


def active_expert_fraction(moe_or_last_route) -> float:
    """Mean fraction of routing weights that are strictly positive (MoE sparsity).

    Accepts either a :class:`SwiGLUMoE` (reads its detached ``last_route``) or a
    routing-weight tensor directly. ``softmax`` routing emits strictly positive
    weights everywhere, so the fraction is ~1.0 (dense); ``relu`` routing emits
    exact zeros (load shedding), so the fraction reports the realized sparsity in
    ``[0, 1]``. Returns a Python ``float`` (single CPU sync; log/eval-site only).
    """
    route = moe_or_last_route.last_route if isinstance(moe_or_last_route, SwiGLUMoE) \
        else moe_or_last_route
    if route is None:
        return float("nan")
    return float((route.detach() > 0).float().mean())


def _global_util_entropy(route) -> float:
    """Global expert-utilization entropy ``H(P)`` of the mean routing dist.

    ``P_e = mean_t route_{t,e}`` normalized to a distribution over experts; the
    Shannon entropy ``H = -sum_e P_e log P_e`` (0*log0 := 0) measures how many
    experts carry mass GLOBALLY (utilization), distinct from the per-token router
    entropy (specialization). Returns a Python ``float`` (log/eval-site only);
    ``0.0`` if every weight is zero (collapsed). Diagnostic only — not in-graph.
    """
    if route is None:
        return float("nan")
    r = route.detach().float()
    p = r.reshape(-1, r.shape[-1]).mean(dim=0)        # (E,) mean mass
    total = p.sum()
    if float(total) <= 0.0:
        return 0.0
    p = p / total
    nz = p > 0
    return float(-(p[nz] * p[nz].log()).sum())


def kv_bytes_per_token(args: Hyperparameters) -> int:
    """Autoregressive KV-cache footprint per token, in bytes (MLA cache).

    MLA caches the *compressed* KV latent (``kv_latent`` floats) plus the
    decoupled per-KV-head RoPE key channel (``n_kv_heads * rope_dim`` floats),
    which is what the attention recomputes K/V from at decode time — that is the
    whole point of latent KV compression versus caching full per-head K and V.
    ``rope_dim = head_dim // 2`` mirrors :class:`MLAttention`'s decoupled-RoPE
    split. We bill bf16 (2 bytes/float), the training/inference activation dtype.
    """
    rope_dim = args.head_dim // 2
    floats = args.kv_latent + args.n_kv_heads * rope_dim
    return int(floats * 2)  # bf16 = 2 bytes/element


def _fit_slope(xs, ys):
    """Ordinary-least-squares ``(slope, intercept)`` for ``ys = slope*xs + b``.

    Pure, deterministic, CPU-only (closed-form normal equations) so the slope
    math for :func:`vram_vs_batch_scaling` is unit-testable without a GPU. With
    fewer than two *distinct* ``xs`` the slope is undetermined, so we return
    ``(0.0, mean(ys))`` (a flat fit through the data centroid). Returns Python
    ``float``\\ s.
    """
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:  # all xs identical -> no slope; flat fit at the y-centroid
        return 0.0, float(my)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return float(slope), float(intercept)


def peak_vram_mb(device=None) -> float:
    """Peak CUDA memory allocated since the last reset, in MiB.

    Reads ``torch.cuda.max_memory_allocated`` (the activation+weights high-water
    mark) and converts to MiB. Returns ``0.0`` when CUDA is unavailable so CPU
    log/test sites stay crash-free. This is the headline RESOURCE-goal number
    (memory-efficiency); a flat curve in recurrent depth K and batch size is the
    reversibility win we are after.
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def vram_util_pct(device=None) -> float:
    """Peak VRAM as a percentage of total device memory (OPS/efficiency stat).

    ``100 * peak_vram_mb / total_device_mem_mb``, where the total comes from
    ``torch.cuda.get_device_properties(device).total_memory``. This is an
    ops/efficiency diagnostic (how full the GPU was), NOT a resource-GOAL number
    — the resource goal is the absolute ``peak_vram`` / ``R_act`` scaling. Returns
    ``0.0`` when CUDA is unavailable (CPU log/test sites) so callers stay
    crash-free. A high utilization at grad_accum=1 (whole batch in one forward)
    is the throughput target: reversibility makes a large micro-batch affordable
    even at deep recurrent depth K.
    """
    if not torch.cuda.is_available() or (device is not None
                                         and torch.device(device).type != "cuda"):
        return 0.0
    props = torch.cuda.get_device_properties(device)
    total_mb = float(props.total_memory) / (1024.0 * 1024.0)
    if total_mb <= 0.0:
        return 0.0
    return 100.0 * peak_vram_mb(device) / total_mb


def resolve_grad_accum_steps(grad_accum: int, world_size: int) -> int:
    """Resolve the grad-accum micro-step count from the CLI knob.

    ``grad_accum == 0`` (default) keeps the historical auto value
    ``max(8 // max(world_size, 1), 1)`` (8 global micro-batches per step,
    matching the original scaffold). ``grad_accum > 0`` overrides it verbatim,
    independent of ``world_size`` — ``grad_accum=1`` runs the whole
    ``batch_tokens`` in ONE forward per step (largest micro-batch / best VRAM
    utilization). Correctness is unaffected: only the number of accumulating
    micro-steps changes; the all-reduce + optimizer step still happen once per
    optimizer step.
    """
    if grad_accum > 0:
        return int(grad_accum)
    return max(8 // max(world_size, 1), 1)


def vram_vs_batch_scaling(model, args: Hyperparameters, batch_sizes, device) -> dict:
    """Peak-VRAM-vs-batch-size scaling for the RESOURCE goal (flat == good).

    For each batch size, resets the CUDA peak-memory counter, runs ONE
    train-style forward+backward step (so activation memory is exercised), and
    records ``torch.cuda.max_memory_allocated``. The returned dict carries the
    raw curve plus an OLS fit::

        {available, batch_sizes, peak_vram_mb, slope_mb_per_sample, intercept_mb}

    A LOW/FLAT ``slope_mb_per_sample`` is the memory-efficiency goal: reversible
    recurrence makes activation memory ~constant in K, and ideally sub-linear in
    batch. When CUDA is unavailable the helper returns ``{available: False}`` (no
    crash) — the pure slope math lives in :func:`_fit_slope`, which the CPU tests
    exercise on injected synthetic ``(batch, vram)`` points.
    """
    if not torch.cuda.is_available() or device is None or torch.device(device).type != "cuda":
        return {"available": False}
    model = model.to(device)
    sizes, peaks = [], []
    max_T = getattr(args, "max_seq_len", None) or getattr(args, "seq_len", 16)
    T = min(getattr(args, "seq_len", max_T), max_T)
    k_set = getattr(args, "k_set", None)
    depth = int(k_set[0]) if k_set else 1
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        tokens = torch.randint(0, args.vocab_size, (bs, T), device=device)
        targets = torch.randint(0, args.vocab_size, (bs, T), device=device)
        model.zero_grad(set_to_none=True)
        loss = model(tokens, targets, depth)
        loss.backward()
        sizes.append(int(bs))
        peaks.append(peak_vram_mb(device))
    slope, intercept = _fit_slope(sizes, peaks)
    return {
        "available": True,
        "batch_sizes": sizes,
        "peak_vram_mb": peaks,
        "slope_mb_per_sample": slope,
        "intercept_mb": intercept,
    }


def collect_metrics(model, args: Hyperparameters) -> dict:
    """Assemble the cheap per-step metrics for the ``metrics:`` log line.

    RESOURCE goal (memory-efficiency): ``peak_vram`` is the headline number, with
    ``R_act`` (activation-memory ~constant in recurrent depth K) supplied by the
    control sweep; ``kv_bytes`` (MLA KV-cache footprint) and ``params`` round out
    the static footprint. ``active_frac`` is kept as a MoE *mechanism* diagnostic
    (logged under the ``diag:`` prefix), NOT a resource-goal metric. ``erank`` is
    the expressiveness signal. Reads ``active_frac`` / ``erank`` from the LAST
    forward's routing weights and embedding spectrum (so call after a forward).
    ``R_act`` and ``phi`` come from the multi-K / multi-r control sweep (Task 9);
    the log emitter accepts them when a caller supplies them. Returns a plain
    ``dict[str, float|int]`` (CPU scalars; do not call in the hot loop).
    """
    inner = model.module if hasattr(model, "module") else model
    moes = [m for m in inner.modules()
            if isinstance(m, SwiGLUMoE) and m.last_route is not None]
    routes = [active_expert_fraction(m) for m in moes]
    active_frac = float(sum(routes) / len(routes)) if routes else float("nan")
    # Collapse-prevention diagnostics: router_entropy is the mean per-token router
    # entropy (specialization-vs-uniformity of the routing dist); expert_util is
    # the GLOBAL utilization entropy (entropy of the mean routing dist over all
    # tokens) — how many experts are used at all. Both averaged over MoE blocks.
    # no_grad: this is a log-site read, not a loss term (router_entropy() is
    # otherwise in-graph for the loss path).
    if moes:
        with torch.no_grad():
            router_entropy = float(
                sum(float(m.router_entropy()) for m in moes) / len(moes))
        expert_util = float(
            sum(_global_util_entropy(m.last_route) for m in moes) / len(moes))
    else:
        router_entropy = float("nan")
        expert_util = float("nan")
    erank = effective_rank(inner.tok_emb.weight)
    return {
        "erank": erank,
        "peak_vram": peak_vram_mb(),
        "kv_bytes": kv_bytes_per_token(args),
        "params": int(sum(p.numel() for p in inner.parameters())),
        "active_frac": active_frac,  # MoE mechanism diagnostic (not resource goal)
        "router_entropy": router_entropy,  # mean per-token router entropy
        "expert_util": expert_util,        # global utilization entropy
    }


def format_metrics_line(metrics: dict) -> str:
    """Render the ``metrics:`` log line consumed by ``plot_metrics.parse_log``.

    RESOURCE-goal fields (``erank`` / ``peak_vram`` / ``kv_bytes`` / ``params``)
    are always emitted; ``R_act`` / ``phi`` only when the caller supplies them
    (control-sweep sites). The collapse-prevention diagnostics ``router_entropy``
    (mean per-token router entropy) and ``expert_util`` (global utilization
    entropy) are emitted when supplied, as is the effective-depth diagnostic
    ``disp_tail`` (tail-mean per-step recurrence displacement). The MoE-mechanism
    diagnostic
    ``active_frac`` is emitted with a ``diag:`` prefix so it is clearly NOT framed
    as a resource metric. Field order is fixed for the parser.
    """
    parts = [
        f"erank:{metrics['erank']:.4f}",
        f"peak_vram:{float(metrics.get('peak_vram', 0.0)):.4f}",
        f"kv_bytes:{int(metrics['kv_bytes'])}",
        f"params:{int(metrics['params'])}",
    ]
    if "router_entropy" in metrics and metrics["router_entropy"] is not None:
        parts.append(f"router_entropy:{float(metrics['router_entropy']):.4f}")
    if "expert_util" in metrics and metrics["expert_util"] is not None:
        parts.append(f"expert_util:{float(metrics['expert_util']):.4f}")
    # Effective-depth diagnostic: tail-mean per-step recurrence displacement
    # (sustained => high effective depth; ~0 => early saturation).
    if "disp_tail" in metrics and metrics["disp_tail"] is not None:
        parts.append(f"disp_tail:{float(metrics['disp_tail']):.4f}")
    # OPS/efficiency diagnostics (throughput + VRAM utilization). These are NOT
    # resource-GOAL numbers (the goal is absolute peak_vram / R_act scaling) —
    # the ``ops:`` prefix marks them clearly. tok_per_s = batch_tokens / step
    # wall-time; vram_util_pct = 100 * peak_vram / total device memory.
    if "tok_per_s" in metrics and metrics["tok_per_s"] is not None:
        parts.append(f"ops:tok_per_s:{float(metrics['tok_per_s']):.4f}")
    if "vram_util_pct" in metrics and metrics["vram_util_pct"] is not None:
        parts.append(f"ops:vram_util_pct:{float(metrics['vram_util_pct']):.4f}")
    if "R_act" in metrics and metrics["R_act"] is not None:
        parts.append(f"R_act:{float(metrics['R_act']):.4f}")
    if "phi" in metrics and metrics["phi"] is not None:
        parts.append(f"phi:{float(metrics['phi']):.4f}")
    if "active_frac" in metrics and metrics["active_frac"] is not None:
        parts.append(f"diag:active_frac:{float(metrics['active_frac']):.4f}")
    return "metrics: " + " ".join(parts)


# ---------------------------------------------------------------------------
# 16. CLI trainer
# ---------------------------------------------------------------------------
def _parse_int_set(spec: str):
    """Parse a comma-separated K-set, e.g. ``"32,64,128"`` -> ``(32, 64, 128)``."""
    return tuple(int(v) for v in str(spec).split(",") if v.strip())


def build_arg_parser():
    p = argparse.ArgumentParser(description="M0 reversible recurrent-depth GPT trainer")
    # Model shape.
    p.add_argument("--model-dim", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--n-experts", type=int, default=16)
    p.add_argument("--expert-rank", type=int, default=32)
    p.add_argument("--n-mix", type=int, default=2)
    p.add_argument("--kv-latent", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--q-latent", type=int, default=None)
    p.add_argument("--router-type", type=str, default="softmax", choices=("softmax", "relu"))
    p.add_argument("--mlp-mult", type=float, default=3.0)
    # Configurable recurrence-block structure (architecture-search axes). All
    # default to the current M0 behavior; each is reversibility-preserving.
    p.add_argument("--block-order", type=str, default="attn_ffn",
                   choices=("attn_ffn", "ffn_attn", "parallel"),
                   help="How attn / FFN-MoE compose in one delta sub-block "
                        "(attn_ffn=current; ffn_attn=FFN-first; parallel=both "
                        "from norm(inp)). All pure functions of inp -> reversible.")
    p.add_argument("--attn-moe", action="store_true",
                   help="Make attention a (MoEUT-style) MoE over --n-attn-experts "
                        "low-rank MLA experts (smooth routing). Default: single MLA.")
    p.add_argument("--n-attn-experts", type=int, default=4,
                   help="Number of routed attention experts when --attn-moe is set.")
    p.add_argument("--num-shared-experts", type=int, default=0,
                   help="DeepSeek always-on experts per MoE (FFN, and attention "
                        "if --attn-moe): summed in ungated, in ADDITION to the "
                        "routed experts. 0 (default) disables.")
    p.add_argument("--n-sublayers", type=int, default=1,
                   help="Each F/G delta block is a stack of N UNIQUE attn+MoE "
                        "sub-blocks applied in sequence (pure delta). Trades "
                        "experts-per-sublayer vs unique-sublayers at matched "
                        "param/compute (e.g. 16x1 vs 8x2). Default 1 = current.")
    p.add_argument("--expert-b-init", type=str, default="small",
                   choices=("small", "zero"),
                   help="Routed-expert up-proj (w_out) init: small=non-zero "
                        "engagement default; zero=classic LoRA-B (only sensible "
                        "with --num-shared-experts>0 providing a base).")
    # Training schedule / batch.
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-tokens", type=int, default=None,
                   help="Global tokens per optimizer step (default: 8 * world * seq_len).")
    p.add_argument("--grad-accum", type=int, default=0,
                   help="Grad-accum micro-steps per optimizer step. 0 (default) "
                        "keeps the auto value max(8//world,1). >0 overrides it; "
                        "grad_accum=1 processes the WHOLE batch_tokens in ONE "
                        "forward (largest micro-batch, best VRAM utilization / "
                        "throughput). Reversibility makes large batch x deep K "
                        "affordable (activation memory ~constant in K).")
    p.add_argument("--warmdown-iters", type=int, default=0,
                   help="Linear LR warmdown over the final N iters (0 disables).")
    # Finite-horizon K-set + loss coefficients.
    p.add_argument("--k-set", type=str, default="32,64,128",
                   help="Comma-separated deep budgets; K_hi sampled uniformly per step.")
    p.add_argument("--k-lo", type=int, default=8, help="Shallow budget for the hinge.")
    p.add_argument("--lambda-h", type=float, default=0.1, help="No-degradation hinge weight.")
    p.add_argument("--margin", type=float, default=0.0, help="Hinge margin (nats).")
    p.add_argument("--lambda-route", type=float, default=0.001,
                   help="ReMoE adaptive-sparsity aux weight (initial value; the "
                        "relu-router controller adapts it toward the target).")
    p.add_argument("--moe-target-active-frac", type=float, default=0.5,
                   help="ReMoE target active fraction for the relu router; the "
                        "adaptive controller holds sparsity at S*=1-this.")
    # Composable router collapse-prevention auxiliaries (control-experiment
    # bake-off). Entropy defaults ON at an EFFECTIVE coef (0.1) because it is the
    # chosen default collapse-preventer: the bake-off's ~0.01-style coef was
    # effectively zero against the (then-inert) MoE, leaving the softmax router
    # collapsed to one expert. 0.1 verifiably lifts router entropy well above a
    # no-aux baseline (see test_router_default_entropy_coef_is_effective) while
    # staying a CLI knob (set 0 to disable). Load-balance defaults OFF so the two
    # regularizers are not stacked by default (decouple antagonistic objectives).
    # See SwiGLUMoE / finite_horizon_loss for the term math.
    p.add_argument("--router-entropy-coef", type=float, default=0.1,
                   help="Entropy reg weight (default 0.1, ON): loss += -coef*H "
                        "(router dist), i.e. MAXIMIZE per-token router entropy "
                        "(spreads expert mass; prevents single-expert collapse). "
                        "Set 0 to disable.")
    p.add_argument("--router-loadbalance-coef", type=float, default=0.0,
                   help="Switch-Transformer load-balance weight: "
                        "loss += coef*E*sum_e f_e*P_e (equalizes expert usage).")
    p.add_argument("--router-target-active-frac", type=float, default=0.0,
                   help="ReMoE adaptive-lambda controller (relu router only): "
                        "0 disables; >0 enables and targets sparsity S*=1-this, "
                        "overriding --moe-target-active-frac for the controller.")
    p.add_argument("--k-eval", type=int, default=None,
                   help="Eval depth (default: max of --k-set).")
    # Optimizer.
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--embed-lr", type=float, default=0.1)
    p.add_argument("--scalar-lr", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Eval / IO.
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--val-every", type=int, default=0,
                   help="Validate every N steps (0: only at the end).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--artifact-out", type=str, default="m0_model.int6.bin")
    p.add_argument("--data-path", type=str,
                   default=os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    p.add_argument("--tokenizer-path", type=str,
                   default=os.environ.get("TOKENIZER_PATH",
                                          "./data/tokenizers/fineweb_1024_bpe.model"))
    p.add_argument("--smoke", action="store_true",
                   help="Force the synthetic in-memory shard (no big dataset).")
    return p


def main(argv=None):
    """End-to-end M0 trainer: data -> finite-horizon train loop -> BPB eval -> int6 artifact.

    DDP-aware (reads ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` from the env under
    ``torchrun``; single-process otherwise). Returns the rank-0 ``val_bpb`` (or
    ``nan`` for the synthetic smoke). The int6 artifact is written by rank 0 to
    ``--artifact-out`` and its size is logged as ``artifact_bytes`` for
    information only (no size gate — the RESOURCE goal is memory-efficiency,
    tracked by ``peak_vram_mb``).
    """
    args = build_arg_parser().parse_args(argv)

    # --- DDP / device setup ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master = rank == 0

    requested = torch.device(args.device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; pass --device cpu")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = requested
    autocast_enabled = device.type == "cuda"  # bf16 autocast on GPU; CPU runs fp32

    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    def print0(msg):
        if master:
            print(msg, flush=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # --- grad-accum: 0 keeps the 8-global-micro-batch scaffold value; >0
    # overrides it (grad_accum=1 == whole batch_tokens in ONE forward, the
    # largest micro-batch / best VRAM utilization). See resolve_grad_accum_steps.
    grad_accum_steps = resolve_grad_accum_steps(args.grad_accum, world_size)
    batch_tokens = args.batch_tokens
    if batch_tokens is None:
        batch_tokens = grad_accum_steps * world_size * args.seq_len
    # Ensure each (rank, micro-step) produces at least one full sequence.
    min_global = world_size * grad_accum_steps * args.seq_len
    if batch_tokens < min_global:
        batch_tokens = min_global

    # --- model ---
    model_args = Hyperparameters(
        model_dim=args.model_dim, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size, n_experts=args.n_experts, expert_rank=args.expert_rank,
        n_mix=args.n_mix, kv_latent=args.kv_latent, head_dim=args.head_dim,
        q_latent=args.q_latent, router_type=args.router_type,
        max_seq_len=args.seq_len, mlp_mult=args.mlp_mult,
        moe_target_active_frac=args.moe_target_active_frac,
        block_order=args.block_order, attn_moe=args.attn_moe,
        n_attn_experts=args.n_attn_experts,
        num_shared_experts=args.num_shared_experts,
        n_sublayers=args.n_sublayers, expert_b_init=args.expert_b_init,
    )
    base_model = M0GPT(model_args).to(device)
    model = base_model

    optimizers = build_optimizers(
        base_model, matrix_lr=args.matrix_lr, embed_lr=args.embed_lr,
        scalar_lr=args.scalar_lr,
    )
    # Manual DDP-style gradient sync. We do NOT use the DDP module wrapper because
    # the finite-horizon loss runs TWO forwards (shallow + deep) per micro-step
    # through a custom-autograd recurrence, which the wrapper's single-forward
    # reducer bookkeeping does not support. Instead: Muon all-reduces its own
    # matrix updates internally (per-rank slice then SUM == full update), and we
    # explicitly all-reduce-MEAN the grads of every non-Muon (AdamW) param below.
    adamw_param_list = adamw_params(base_model, optimizers)

    def sync_adamw_grads():
        if not distributed or world_size <= 1:
            return
        for p in adamw_param_list:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    if distributed:
        # Broadcast rank-0 init so every rank starts from identical weights.
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)

    # --- data ---
    use_synthetic = args.smoke or device.type == "cpu"
    train_pattern = os.path.join(args.data_path, "fineweb_train_*.bin")
    val_pattern = os.path.join(args.data_path, "fineweb_val_*.bin")
    if not use_synthetic and not glob.glob(train_pattern):
        print0(f"data:absent pattern={train_pattern} -> synthetic fallback")
        use_synthetic = True

    if use_synthetic:
        train_loader = DistributedTokenLoader.synthetic(
            args.vocab_size, rank, world_size, device, seed=args.seed)
        val_loader = DistributedTokenLoader.synthetic(
            args.vocab_size, rank, world_size, device, seed=args.seed + 7919)
        luts = None
    else:
        train_loader = DistributedTokenLoader.from_pattern(
            train_pattern, rank, world_size, device)
        val_loader = DistributedTokenLoader.from_pattern(
            val_pattern, rank, world_size, device)
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        luts = build_sentencepiece_luts(sp, args.vocab_size, device)

    k_set = _parse_int_set(args.k_set)
    if not k_set:
        raise ValueError("--k-set must contain at least one budget")
    k_eval = args.k_eval if args.k_eval is not None else max(k_set)
    k_gen = torch.Generator().manual_seed(args.seed)

    print0(f"m0_trainer:start device={device} world_size={world_size} "
           f"grad_accum={grad_accum_steps} batch_tokens={batch_tokens} "
           f"synthetic={use_synthetic} k_set={k_set} k_lo={args.k_lo} k_eval={k_eval}")
    print0(f"m0_params:{sum(p.numel() for p in base_model.parameters())}")

    def lr_scale(step):
        if args.warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if step < warmdown_start:
            return 1.0
        return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # --- ReMoE adaptive sparsity controller state (relu router only) ---
    # A fixed L1 penalty collapses the router (active_frac -> 0); the controller
    # adapts a scalar lambda_route per optimizer step to HOLD realized sparsity at
    # the target S* = 1 - target_active_frac. The controller is ON only when
    # BOTH the router is relu AND --router-target-active-frac > 0 (the bake-off
    # gate); >0 overrides --moe-target-active-frac for the controller's target.
    # When OFF (default), lambda_route keeps its fixed initial value and the L1
    # aux is the static "no-fix control" — the entropy / load-balance auxiliaries
    # are the composable alternatives, each gated by its own coef below.
    target_active_frac = (args.router_target_active_frac
                          if args.router_target_active_frac > 0.0
                          else args.moe_target_active_frac)
    adaptive_route = args.router_type == "relu" and args.router_target_active_frac > 0.0
    lambda_route = float(args.lambda_route)
    s_target = 1.0 - target_active_frac

    def reduced_sparsity():
        """Mean realized routing sparsity, all-reduced-MEAN across ranks so the
        controller's lambda update is identical on every rank. One sync at the
        step boundary (mirrors the logging .item() site, not the hot loop)."""
        s = measured_sparsity(base_model)
        if distributed and world_size > 1:
            t = torch.tensor(s, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            s = float(t) / world_size
        return s

    # --- training loop (DDP grad-accum, finite-horizon hinge) ---
    # Reset the CUDA peak-memory counter so the end-of-run ``peak_vram_mb`` (the
    # RESOURCE-goal headline) reflects the training high-water mark, not setup.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(args.iterations):
        # Per-step wall-clock timer for the throughput (tok_per_s) ops metric.
        # Started at the STEP BOUNDARY (not inside the grad-accum micro-loop) so
        # the measured window is exactly one optimizer step.
        step_t0 = time.perf_counter()
        # Sample one deep budget per step (shared across ranks for sync grads).
        k_hi = int(k_set[int(torch.randint(0, len(k_set), (1,), generator=k_gen).item())])
        scale = lr_scale(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        zero_grad_all()
        step_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch(batch_tokens, args.seq_len, grad_accum_steps)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=autocast_enabled):
                loss, _ = finite_horizon_loss(
                    base_model, x, y, k_hi=k_hi, k_lo=args.k_lo,
                    lambda_h=args.lambda_h, margin=args.margin,
                    lambda_route=lambda_route,
                    entropy_coef=args.router_entropy_coef,
                    loadbalance_coef=args.router_loadbalance_coef)
            step_loss = step_loss + loss.detach()
            (loss / grad_accum_steps).backward()
        step_loss = step_loss / grad_accum_steps

        # All-reduce-mean the AdamW (non-Muon) grads across ranks; Muon syncs
        # its own matrix updates internally during opt.step().
        sync_adamw_grads()
        if args.grad_clip > 0:
            # Fix I2: clip over the AdamW group ONLY. Their grads are synced
            # (consistent global norm across ranks), so the clip scale is
            # identical on every rank. Muon's grads are unsynced per-rank (it
            # syncs updates, not grads) and its update is scale-invariant, so
            # including Muon params would give a rank-inconsistent clip.
            torch.nn.utils.clip_grad_norm_(adamw_param_list, args.grad_clip)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # ReMoE adaptive-lambda update (relu router): ONE reduced sparsity read
        # at the step boundary, then a python-float lambda step toward the target
        # sparsity. Keeps active_frac near (1 - S*) instead of collapsing to 0.
        if adaptive_route:
            lambda_route = _update_lambda_route(
                lambda_route, s_measured=reduced_sparsity(), s_target=s_target)

        do_log = args.log_every > 0 and (step < 3 or (step + 1) % args.log_every == 0)
        if do_log:
            # Single rank-0 .item() sync at the log site (no hot-path sync).
            print0(f"step:{step + 1}/{args.iterations} k_hi:{k_hi} "
                   f"train_loss:{step_loss.item():.4f} lambda_route:{lambda_route:.3e}")
            # Two-goal metrics from the just-finished forward (rank-0 log site
            # only; collect_metrics does CPU syncs, never call in the hot loop).
            if master:
                # Ops/efficiency stats measured at the STEP BOUNDARY (not in the
                # micro-loop): throughput = global batch_tokens / step wall-time,
                # and VRAM utilization = peak VRAM / total device mem. One
                # max_memory_allocated read (via vram_util_pct), at the log site.
                step_wall = max(time.perf_counter() - step_t0, 1e-9)
                m = collect_metrics(base_model, model_args)
                m["tok_per_s"] = float(batch_tokens) / step_wall
                m["vram_util_pct"] = vram_util_pct(device)
                # Effective-depth diagnostic: per-step recurrence displacement
                # over the last micro-batch at the deep budget; headline is the
                # tail mean (sustained displacement => high effective depth).
                m["disp_tail"] = displacement_tail(
                    recurrence_displacement(base_model, x, k_hi))
                print0(format_metrics_line(m))

        if args.val_every > 0 and (step + 1) % args.val_every == 0:
            v_loss, v_bpb = run_validation(
                base_model, val_loader, depth=k_eval, n_batches=args.eval_batches,
                seq_len=args.seq_len, global_tokens=batch_tokens,
                grad_accum_steps=grad_accum_steps, device=device, luts=luts,
                autocast_enabled=autocast_enabled)
            print0(f"step:{step + 1}/{args.iterations} val_loss:{v_loss:.4f} "
                   f"val_bpb:{v_bpb:.4f}")
            model.train()

    # --- final validation ---
    val_loss, val_bpb = run_validation(
        base_model, val_loader, depth=k_eval, n_batches=args.eval_batches,
        seq_len=args.seq_len, global_tokens=batch_tokens,
        grad_accum_steps=grad_accum_steps, device=device, luts=luts,
        autocast_enabled=autocast_enabled)
    print0(f"final val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # --- peak VRAM (RESOURCE-goal headline; memory-efficiency, no gate) ---
    # Integer MiB to match the existing ``peak_vram_mb:<int>`` log contract that
    # baselines/train_gpt_comparable_sweep.py and the legacy emitter use.
    if device.type == "cuda":
        print0(f"peak_vram_mb:{int(peak_vram_mb(device))}")

    # --- int6 artifact (rank 0) ---
    # No 16 MB gate: the RESOURCE goal is memory-efficiency, so artifact bytes is
    # logged as an INFORMATIONAL number only (no raise, no all-rank broadcast).
    if master:
        state_dict = base_model.state_dict()
        compressed, _, _ = save_int6_artifact(state_dict)
        artifact_bytes = len(compressed)
        print0(f"artifact_bytes:{artifact_bytes} compressor:{_COMPRESSOR}")
        out_path = Path(args.artifact_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(compressed)
        print0(f"artifact_written:{out_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return val_bpb


if __name__ == "__main__":
    main()
