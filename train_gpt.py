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

# 16 MB hard artifact budget (Parameter Golf challenge constraint).
MAX_ARTIFACT_BYTES = 16_000_000


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

    Diagnostics set after each forward:
      * ``self.last_route`` — detached routing weights, shape ``(B, T, n_experts)``.
      * ``self.aux_l1``     — mean L1 of the (live) routing weights, the ReMoE
        adaptive sparsity / load-balance aux term consumed by the train loop.

    bf16-friendly: no forced fp32 except inside ``RMSNorm`` statistics.
    """

    def __init__(self, dim, n_experts, expert_rank, router_type="softmax"):
        super().__init__()
        assert router_type in ("softmax", "relu"), (
            f"router_type must be 'softmax' or 'relu', got {router_type!r}"
        )
        self.dim = dim
        self.n_experts = n_experts
        self.expert_rank = expert_rank
        self.router_type = router_type

        # Router is full-rank: dim -> n_experts logits.
        self.router = nn.Linear(dim, n_experts)

        # Low-rank (LoRA-style) SwiGLU experts as batched parameter banks.
        # gate/up share a single down-projection to expert_rank, then SwiGLU
        # (silu(gate) * up) at rank, then up-project back to dim.
        #   x:(B,T,dim) @ w_in:(E,dim,2*rank) -> (B,T,E,2*rank) -> SwiGLU(rank)
        #   -> @ w_out:(E,rank,dim) -> (B,T,E,dim)
        self.w_in = nn.Parameter(torch.empty(n_experts, dim, 2 * expert_rank))
        self.w_out = nn.Parameter(torch.empty(n_experts, expert_rank, dim))
        nn.init.normal_(self.w_in, std=dim ** -0.5)
        nn.init.normal_(self.w_out, std=expert_rank ** -0.5)

        self.last_route = None
        self.aux_l1 = None

    def forward(self, x):
        # --- Smooth router over the input (continuous in x; no dispatch) ---
        logits = self.router(x)                                  # (B, T, E)
        if self.router_type == "softmax":
            w = F.softmax(logits, dim=-1)
        else:  # relu: exact zeros => sparsity, still continuous in x
            w = F.relu(logits)

        # ReMoE adaptive sparsity / load-balance aux term (keep w in graph).
        self.aux_l1 = w.abs().mean()
        self.last_route = w.detach()

        # --- Dense low-rank SwiGLU over ALL experts (no skipping) ---
        h = torch.einsum("btd,edr->bter", x, self.w_in)         # (B, T, E, 2*rank)
        gate, up = h.chunk(2, dim=-1)                            # each (B, T, E, rank)
        act = F.silu(gate) * up                                  # (B, T, E, rank)
        expert_out = torch.einsum("bter,erd->bted", act, self.w_out)  # (B, T, E, dim)

        # --- Smooth soft combine: y = sum_e w[...,e] * expert_e(x) ---
        y = torch.einsum("bte,bted->btd", w.type_as(expert_out), expert_out)
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


class _PreNormDeltaBlock(nn.Module):
    """Pre-norm MLA + SwiGLU-MoE *delta* block — one ``F``/``G`` map.

    Returns the residual update the reversible recurrence adds (NOT input +
    update); the recurrence owns the additive coupling. CRITICAL for
    reversibility: the block is a deterministic *pure function of its single
    input tensor* (no dropout, no batch-coupled routing, no external state), so
    the reversible inverse recomputes the identical update and reconstruction is
    exact in fp64.

    Delta form (post-attention-residual; verified in ``p1_synthetic``)::

        h   = attn_norm(inp)
        a   = attn(h)
        h2  = mlp_norm(inp + a)
        out = a + moe(h2)
        return out
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim
        self.attn_norm = RMSNorm(d)
        self.attn = MLAttention(
            dim=d, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
            kv_latent=args.kv_latent, head_dim=args.head_dim, q_latent=args.q_latent,
        )
        self.mlp_norm = RMSNorm(d)
        self.moe = SwiGLUMoE(
            dim=d, n_experts=args.n_experts, expert_rank=args.expert_rank,
            router_type=args.router_type,
        )

    def forward(self, inp):
        a = self.attn(self.attn_norm(inp))
        out = a + self.moe(self.mlp_norm(inp + a))
        return out


class M0GPT(nn.Module):
    """M0 reversible recurrent-depth GPT.

    Pipeline::

        x0  = tok_emb(tokens) + pos_emb[:, :T]
        z_K = run_reversible(x0, depth=K)        # midpoint of the reversible pair
        log p = mos_head(z_K)                    # LOG-probabilities (MoS)
        loss  = nll_loss(log p, targets)         # MoS emits log-probs -> NLL

    The two recurrence maps ``F``/``G`` are pre-norm MLA+MoE delta blocks
    (:class:`_PreNormDeltaBlock`); each is a deterministic pure function of its
    input, which is what makes the additive-coupling recurrence exactly
    invertible (reconstruction gate, fp64). The MoS output basis is *tied* to
    the input token embedding (shared ``nn.Parameter``).
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

        self.mos_head = MoSHead(d, V, args.n_mix)
        # Tie the MoS output basis to the input token embedding (shared Param).
        self.mos_head.out_embed.weight = self.tok_emb.weight

    def forward(self, tokens, targets, depth):
        B, T = tokens.shape
        x0 = self.tok_emb(tokens) + self.pos_emb[:, :T]
        z_K = self.rec.run_reversible(x0, depth)
        logp = self.mos_head(z_K)
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
def _collect_aux_l1(model):
    """Sum ``aux_l1`` over every SwiGLUMoE block that has produced a forward.

    Walks ``model.modules()`` so it works whether or not the model is DDP- or
    compile-wrapped (caller passes the inner module). Skips blocks that have not
    run a forward yet (``aux_l1 is None``).
    """
    total = None
    for m in model.modules():
        if isinstance(m, SwiGLUMoE) and m.aux_l1 is not None:
            total = m.aux_l1 if total is None else total + m.aux_l1
    if total is None:
        # No MoE produced a forward (should not happen on the train path); a 0.0
        # tensor on the right device keeps the loss assembly type-stable.
        p = next(model.parameters())
        return torch.zeros((), device=p.device, dtype=torch.float32)
    return total


def finite_horizon_loss(model, x, y, k_hi, k_lo, lambda_h, margin, lambda_route):
    """Finite-horizon no-degradation loss.

        L = L_hi + lambda_h * relu(L_hi - sg(L_lo) + margin) + lambda_route * aux_l1

    ``L_hi = model(x, y, k_hi)`` is the deep pass that carries the task gradient.
    ``L_lo = model(x, y, k_lo)`` is a shallow pass; inside the hinge it is
    stop-gradient (``sg``) so the no-degradation pressure pushes the deep state to
    be no worse than the shallow one WITHOUT backpropping into the shallow pass.
    ``aux_l1`` is the summed ReMoE routing-mass L1 over the MoE blocks (computed
    from the L_hi forward, which is the last forward run). Returns
    ``(loss, parts)`` where ``parts`` exposes the components for testing/logging.
    """
    l_lo = model(x, y, k_lo).detach()  # stop-grad shallow pass
    l_hi = model(x, y, k_hi)           # deep pass: runs last so aux_l1 reflects it
    aux_l1 = _collect_aux_l1(model)
    hinge = torch.relu(l_hi - l_lo + margin)
    loss = l_hi + lambda_h * hinge + lambda_route * aux_l1
    parts = {"l_hi": l_hi, "l_lo_sg": l_lo, "aux_l1": aux_l1, "hinge": hinge}
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
# budget ``K``; here we also log the cheap per-step subset (``erank`` /
# ``active_frac`` / ``kv_bytes`` / ``params``).
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


def collect_metrics(model, args: Hyperparameters) -> dict:
    """Assemble the cheap per-step two-goal metrics for the ``metrics:`` log line.

    Reads ``active_frac`` / ``erank`` from the LAST forward's routing weights and
    output-embedding spectrum (so call after a forward), ``kv_bytes`` from the
    static head config, and ``params`` from the model. ``R_act`` and ``phi`` are
    NOT computed here — they require the multi-K / multi-r control sweep (Task 9)
    — but the log emitter accepts them when a caller supplies them. Returns a
    plain ``dict[str, float|int]`` (CPU scalars; do not call in the hot loop).
    """
    inner = model.module if hasattr(model, "module") else model
    routes = [active_expert_fraction(m) for m in inner.modules()
              if isinstance(m, SwiGLUMoE) and m.last_route is not None]
    active_frac = float(sum(routes) / len(routes)) if routes else float("nan")
    erank = effective_rank(inner.tok_emb.weight)
    return {
        "active_frac": active_frac,
        "erank": erank,
        "kv_bytes": kv_bytes_per_token(args),
        "params": int(sum(p.numel() for p in inner.parameters())),
    }


def format_metrics_line(metrics: dict) -> str:
    """Render the ``metrics:`` log line consumed by ``plot_metrics.parse_log``.

    Emits ``erank`` / ``active_frac`` / ``kv_bytes`` / ``params`` always, and
    ``R_act`` / ``phi`` only when the caller provides them (control-sweep sites),
    so a plain per-step call stays compact. Field order is fixed for the parser.
    """
    parts = [
        f"erank:{metrics['erank']:.4f}",
        f"active_frac:{metrics['active_frac']:.4f}",
        f"kv_bytes:{int(metrics['kv_bytes'])}",
        f"params:{int(metrics['params'])}",
    ]
    if "R_act" in metrics and metrics["R_act"] is not None:
        parts.append(f"R_act:{float(metrics['R_act']):.4f}")
    if "phi" in metrics and metrics["phi"] is not None:
        parts.append(f"phi:{float(metrics['phi']):.4f}")
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
    # Training schedule / batch.
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-tokens", type=int, default=None,
                   help="Global tokens per optimizer step (default: 8 * world * seq_len).")
    p.add_argument("--warmdown-iters", type=int, default=0,
                   help="Linear LR warmdown over the final N iters (0 disables).")
    # Finite-horizon K-set + loss coefficients.
    p.add_argument("--k-set", type=str, default="32,64,128",
                   help="Comma-separated deep budgets; K_hi sampled uniformly per step.")
    p.add_argument("--k-lo", type=int, default=8, help="Shallow budget for the hinge.")
    p.add_argument("--lambda-h", type=float, default=0.1, help="No-degradation hinge weight.")
    p.add_argument("--margin", type=float, default=0.0, help="Hinge margin (nats).")
    p.add_argument("--lambda-route", type=float, default=0.001, help="ReMoE aux-L1 weight.")
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
    ``--artifact-out``, with the 16 MB budget enforced.
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

    # --- grad-accum: keep 8 "global" micro-batches per step (matches scaffold) ---
    grad_accum_steps = max(8 // max(world_size, 1), 1)
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

    # --- training loop (DDP grad-accum, finite-horizon hinge) ---
    model.train()
    for step in range(args.iterations):
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
                    lambda_route=args.lambda_route)
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

        do_log = args.log_every > 0 and (step < 3 or (step + 1) % args.log_every == 0)
        if do_log:
            # Single rank-0 .item() sync at the log site (no hot-path sync).
            print0(f"step:{step + 1}/{args.iterations} k_hi:{k_hi} "
                   f"train_loss:{step_loss.item():.4f}")
            # Two-goal metrics from the just-finished forward (rank-0 log site
            # only; collect_metrics does CPU syncs, never call in the hot loop).
            if master:
                print0(format_metrics_line(collect_metrics(base_model, model_args)))

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

    # --- int6 artifact (rank 0), with 16 MB budget check ---
    # Budget violation is broadcast from rank 0 so EVERY rank raises collectively
    # (avoids the deadlock where master raises but workers block on the barrier).
    violated = torch.zeros((), device=device)
    artifact_bytes = 0
    if master:
        state_dict = base_model.state_dict()
        compressed, _, _ = save_int6_artifact(state_dict)
        artifact_bytes = len(compressed)
        print0(f"artifact_bytes:{artifact_bytes} compressor:{_COMPRESSOR} "
               f"budget:{MAX_ARTIFACT_BYTES}")
        if artifact_bytes > MAX_ARTIFACT_BYTES:
            violated.fill_(1.0)
        else:
            out_path = Path(args.artifact_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(compressed)
            print0(f"artifact_written:{out_path}")
    if distributed:
        dist.broadcast(violated, src=0)
        dist.barrier()
        dist.destroy_process_group()
    if float(violated.item()) > 0.5:
        raise RuntimeError(
            f"int6 artifact {artifact_bytes} bytes exceeds {MAX_ARTIFACT_BYTES} budget")
    return val_bpb


if __name__ == "__main__":
    main()
