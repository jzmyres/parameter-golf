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
    5. Test-only delta block (_TinyDelta)
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


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
# 5. Test-only delta block
# ---------------------------------------------------------------------------
class _TinyDelta(nn.Module):
    """Minimal delta block for reversibility tests (not a model component)."""

    def __init__(self, d):
        super().__init__()
        self.n = RMSNorm(d)
        self.l = nn.Linear(d, d, bias=False).double()

    def forward(self, x):
        return self.l(self.n(x))
