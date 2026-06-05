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
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
