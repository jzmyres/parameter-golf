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
    4. Test-only delta block (_TinyDelta)
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


# ---------------------------------------------------------------------------
# 4. Test-only delta block
# ---------------------------------------------------------------------------
class _TinyDelta(nn.Module):
    """Minimal delta block for reversibility tests (not a model component)."""

    def __init__(self, d):
        super().__init__()
        self.n = RMSNorm(d)
        self.l = nn.Linear(d, d, bias=False).double()

    def forward(self, x):
        return self.l(self.n(x))
