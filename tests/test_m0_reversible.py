import torch
import torch.nn as nn
from train_gpt import ReversibleRecurrence, _TinyDelta


def test_additive_coupling_reconstructs_initial_state():
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64)
    a0 = b0 = x0
    (aK, bK), _states = rec.forward_states(a0, b0, x0, depth=5)
    a_rec, b_rec = rec.invert(aK, bK, x0, depth=5)
    assert torch.allclose(a_rec, a0, atol=1e-9)
    assert torch.allclose(b_rec, b0, atol=1e-9)


def _clock_rec(d, max_step):
    F, G = _TinyDelta(d), _TinyDelta(d)
    step_emb = nn.Embedding(max_step, d).double()
    nn.init.normal_(step_emb.weight, std=0.5)
    return ReversibleRecurrence(F, G, step_emb=step_emb)


def test_step_conditioning_reconstructs_initial_state():
    """With a per-step clock, the reverse must recompute e_k at each k -> exact
    fp64 reconstruction of the seed."""
    torch.manual_seed(0)
    d = 16
    rec = _clock_rec(d, max_step=8)
    x0 = torch.randn(2, 4, d, dtype=torch.float64)
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=5)
    assert not torch.allclose(aK, x0)
    a_rec, b_rec = rec.invert(aK, bK, x0, depth=5)
    assert torch.allclose(a_rec, x0, atol=1e-9)
    assert torch.allclose(b_rec, x0, atol=1e-9)


def test_step_conditioning_backward_matches_ordinary_autograd():
    """The O(1)-memory custom backward must produce gradients identical to
    ordinary stored-activation BPTT — INCLUDING the step_emb.weight grad slot.
    This is the hard custom-autograd-input gate for the clock parameter."""
    torch.manual_seed(0)
    d = 16
    rec = _clock_rec(d, max_step=8)
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    # Reference: ordinary autograd through forward_states.
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=4)
    (0.5 * (aK + bK)).pow(2).sum().backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    ref_x0 = x0.grad.clone()
    assert "step_emb.weight" in ref_g and ref_g["step_emb.weight"].abs().sum() > 0
    for p in rec.parameters():
        p.grad = None
    # O(1) custom-autograd path.
    x0b = x0.detach().clone().requires_grad_(True)
    rec.run_reversible(x0b, depth=4).pow(2).sum().backward()
    for n, p in rec.named_parameters():
        assert torch.allclose(p.grad, ref_g[n], atol=1e-6), n
    assert torch.allclose(x0b.grad, ref_x0, atol=1e-6)


def test_reversible_backward_matches_ordinary_autograd():
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=4)
    ref = (0.5 * (aK + bK)).pow(2).sum()
    ref.backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    ref_x0 = x0.grad.clone()
    for p in rec.parameters():
        p.grad = None
    x0b = x0.detach().clone().requires_grad_(True)
    zK = rec.run_reversible(x0b, depth=4)
    zK.pow(2).sum().backward()
    for n, p in rec.named_parameters():
        assert torch.allclose(p.grad, ref_g[n], atol=1e-6), n
    assert torch.allclose(x0b.grad, ref_x0, atol=1e-6)


# ---------------------------------------------------------------------------
# init_state=random (Huginn-style path-independent init; the Occam anti-collapse
# fix). a0/b0 are random non-learnable constants independent of x0; x0 is STILL
# injected each step. The seed-grad term (gx0 += ga + gb) is x0-init only and
# MUST be dropped for random init (a0/b0 carry no grad into x0). Reversibility +
# grad-equivalence MUST stay exact for BOTH modes — the hard gate.
# ---------------------------------------------------------------------------
def test_random_init_reconstructs_initial_state():
    """fp64 reconstruction stays EXACT with random (a0, b0) != x0: the algebraic
    inverse recovers whatever seed was used, independent of its value."""
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64)
    a0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    b0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    assert not torch.allclose(a0, x0) and not torch.allclose(b0, x0)
    (aK, bK), _ = rec.forward_states(a0, b0, x0, depth=5)
    a_rec, b_rec = rec.invert(aK, bK, x0, depth=5)
    assert torch.allclose(a_rec, a0, atol=1e-7)
    assert torch.allclose(b_rec, b0, atol=1e-7)


def test_random_init_backward_matches_ordinary_autograd():
    """HARD GATE: with random (a0, b0) independent of x0, the O(1)-memory custom
    backward must produce gradients IDENTICAL to ordinary stored-activation BPTT.

    a0/b0 are non-learnable constants (no grad), so the seed-grad term
    (gx0 += ga + gb) used for the x0-init seed MUST NOT be added: x0 enters ONLY
    via the per-step injection. The reference unrolls forward_states(a0, b0, x0)
    with ordinary autograd; gradients to every param AND to x0 must match."""
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    a0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)  # non-learnable seed
    b0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    # Reference: ordinary autograd through forward_states with the random seed.
    (aK, bK), _ = rec.forward_states(a0, b0, x0, depth=4)
    (0.5 * (aK + bK)).pow(2).sum().backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    ref_x0 = x0.grad.clone()
    for p in rec.parameters():
        p.grad = None
    # O(1) custom-autograd path with the SAME random seed.
    x0b = x0.detach().clone().requires_grad_(True)
    rec.run_reversible(x0b, depth=4, a0=a0, b0=b0).pow(2).sum().backward()
    max_diff = 0.0
    for n, p in rec.named_parameters():
        max_diff = max(max_diff, (p.grad - ref_g[n]).abs().max().item())
        assert torch.allclose(p.grad, ref_g[n], atol=1e-6), n
    max_diff = max(max_diff, (x0b.grad - ref_x0).abs().max().item())
    assert torch.allclose(x0b.grad, ref_x0, atol=1e-6), f"x0 grad max_diff={max_diff}"


def test_random_init_changes_trajectory_vs_x0():
    """The random seed must actually CHANGE the recurrence trajectory: with
    a0/b0 != x0 the terminal state differs from the x0-seeded run (otherwise the
    'fix' is a silent no-op)."""
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64)
    a0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    b0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    z_x0 = rec.run_reversible(x0, depth=5)                 # default a0=b0=x0
    z_rand = rec.run_reversible(x0, depth=5, a0=a0, b0=b0)  # random seed
    assert not torch.allclose(z_x0, z_rand)


def test_step_conditioning_and_random_init_compose_backward():
    """HARD COMBINED GATE: step_conditioning=True AND random (a0, b0) != x0 must
    compose cleanly through the custom backward.

    The two features touch different parts of the backward: random init DROPS the
    x0 seed-grad term (gx0 += ga + gb), while step_conditioning ADDS the
    step_emb.weight gradient via the live e_k on each rebuilt step. Both must
    hold at once: reversible backward == ordinary autograd (atol 1e-6), and
    step_emb.weight.grad is nonzero. This is the M0GPT(init_state="random",
    step_conditioning=True) backward path."""
    torch.manual_seed(0)
    d = 16
    rec = _clock_rec(d, max_step=8)  # has a step_emb clock
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    a0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)  # non-learnable seed
    b0 = 0.02 * torch.randn(2, 4, d, dtype=torch.float64)
    assert not torch.allclose(a0, x0) and not torch.allclose(b0, x0)
    # Reference: ordinary autograd through forward_states with the random seed +
    # the clock (forward_states applies e_k internally).
    (aK, bK), _ = rec.forward_states(a0, b0, x0, depth=4)
    (0.5 * (aK + bK)).pow(2).sum().backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    ref_x0 = x0.grad.clone()
    assert "step_emb.weight" in ref_g and ref_g["step_emb.weight"].abs().sum() > 0
    for p in rec.parameters():
        p.grad = None
    # O(1) custom-autograd path with the SAME random seed + clock.
    x0b = x0.detach().clone().requires_grad_(True)
    rec.run_reversible(x0b, depth=4, a0=a0, b0=b0).pow(2).sum().backward()
    max_diff = 0.0
    for n, p in rec.named_parameters():
        max_diff = max(max_diff, (p.grad - ref_g[n]).abs().max().item())
        assert torch.allclose(p.grad, ref_g[n], atol=1e-6), n
    # step_emb.weight MUST carry a real (nonzero) gradient through the combined path.
    assert rec.step_emb.weight.grad.abs().sum() > 0
    max_diff = max(max_diff, (x0b.grad - ref_x0).abs().max().item())
    assert torch.allclose(x0b.grad, ref_x0, atol=1e-6), f"x0 grad max_diff={max_diff}"


def test_bf16_autocast_backward_matches_ordinary_autograd():
    """bf16 GRAD-EQUIVALENCE: under CPU bf16 autocast the O(1) reversible backward
    must reproduce ordinary stored-activation BPTT grads to a bf16-appropriate
    tolerance. This is the real correctness target: not just small recon, but
    that the autocast-matched, fp64-accumulating backward yields correct
    gradients. The reference unrolls forward_states under the SAME autocast.

    Before the fix the backward recomputed F/G in fp32 (no autocast replayed),
    so forward-F (bf16) != backward-F (fp32) and the algebraic inverse used the
    wrong function -> wrong grads. fp32-param leaves keep all grads fp32.
    """
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d).float(), _TinyDelta(d).float()  # fp32 params
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, requires_grad=True)  # fp32 seed
    # Reference: ordinary autograd through forward_states under bf16 autocast.
    # backward() is called OUTSIDE the autocast region to MATCH the trainer
    # (finite_horizon_loss runs under autocast; backward runs after it closes).
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=8)
        ref_loss = (0.5 * (aK + bK)).float().pow(2).sum()
    ref_loss.backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    ref_x0 = x0.grad.clone()
    for p in rec.parameters():
        p.grad = None
    # O(1) custom-autograd path under the SAME bf16 autocast (backward outside).
    x0b = x0.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        zK = rec.run_reversible(x0b, depth=8)
        loss = zK.float().pow(2).sum()
    loss.backward()

    def relerr(g, ref):
        return float((g - ref).norm() / (ref.norm() + 1e-12))

    for n, p in rec.named_parameters():
        assert p.grad.dtype == p.dtype, (n, p.grad.dtype)  # grads stay fp32
        # bf16 matmuls -> compare with a bf16-appropriate RELATIVE-NORM tolerance
        # (the meaningful BPTT-correctness measure; per-element atol is dominated
        # by bf16 noise on near-zero grad components). < 2% confirms the
        # autocast-matched backward reproduces the true grads.
        assert relerr(p.grad, ref_g[n]) < 2e-2, (n, relerr(p.grad, ref_g[n]))
    assert x0b.grad.dtype == x0.dtype
    assert relerr(x0b.grad, ref_x0) < 2e-2, relerr(x0b.grad, ref_x0)


def test_bf16_backward_under_outer_autocast_keeps_matmul_grads():
    """cache_enabled gate: when ``backward()`` runs WITH an outer autocast still
    active, the backward's no_grad reconstruction would otherwise poison the
    OUTER autocast weight-cast cache with a DETACHED bf16 weight; the enable_grad
    rebuilt step then reuses it and the matmul (Linear.weight) grads come back
    None -> ZERO. The fix enters the backward's autocast with cache_enabled=False
    so each rebuilt step gets a fresh grad-tracked weight cast. Here the Linear
    weight grads MUST be nonzero and match the reference."""
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d).float(), _TinyDelta(d).float()
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, requires_grad=True)
    # Reference (backward outside autocast — always correct).
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=8)
        (0.5 * (aK + bK)).float().pow(2).sum().backward()
    ref_g = {n: p.grad.clone() for n, p in rec.named_parameters()}
    for p in rec.parameters():
        p.grad = None
    # Custom path with backward INSIDE the outer autocast (poisoning condition).
    x0b = x0.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        zK = rec.run_reversible(x0b, depth=8)
        zK.float().pow(2).sum().backward()  # backward UNDER outer autocast
    # The Linear (matmul) weights MUST carry nonzero grads matching the reference.
    for n in ("F.l.weight", "G.l.weight"):
        g = dict(rec.named_parameters())[n].grad
        assert g.norm() > 0, f"{n} grad is zero (autocast cache poisoned)"
        rel = float((g - ref_g[n]).norm() / (ref_g[n].norm() + 1e-12))
        assert rel < 2e-2, (n, rel)


def test_revrecurrence_backward_captures_and_replays_autocast():
    """Part A structural gate: RevRecurrenceFn.forward must CAPTURE the caller's
    autocast (enabled flag, device_type, dtype) onto ctx so backward can REPLAY
    it. We intercept the ctx via a custom Function subclass hook to assert the
    captured metadata matches the active bf16 autocast — a regression that drops
    the capture (reverting Part A, so backward F/G run fp32 != forward bf16)
    fails here."""
    import train_gpt

    captured = {}
    orig_forward = train_gpt.RevRecurrenceFn.forward

    class _SpyFn(train_gpt.RevRecurrenceFn):
        @staticmethod
        def forward(ctx, *a):
            out = orig_forward(ctx, *a)
            captured["enabled"] = ctx.autocast_enabled
            captured["device_type"] = ctx.device_type
            captured["dtype"] = ctx.autocast_dtype
            return out

    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d).float(), _TinyDelta(d).float()
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, requires_grad=True)
    a0 = b0 = x0
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        aK, bK = _SpyFn.apply(rec, x0, a0, b0, 4, True, *rec.parameters())
    assert captured["enabled"] is True
    assert captured["device_type"] == "cpu"
    assert captured["dtype"] == torch.bfloat16
    # And with NO autocast the capture records enabled=False (backward stays fp32).
    captured.clear()
    x0b = x0.detach().clone().requires_grad_(True)
    _SpyFn.apply(rec, x0b, x0b, x0b, 4, True, *rec.parameters())
    assert captured["enabled"] is False


def test_x0_init_byte_identical_to_explicit_x0_seed():
    """Default run_reversible(x0) (a0=b0 implicit x0) is byte-identical to passing
    a0=b0=x0 explicitly — proves the new explicit-seed path keeps the x0 seed
    grad term (gx0 += ga + gb) and changes nothing for the default mode."""
    torch.manual_seed(0)
    d = 16
    F, G = _TinyDelta(d), _TinyDelta(d)
    rec = ReversibleRecurrence(F, G)
    x0 = torch.randn(2, 4, d, dtype=torch.float64, requires_grad=True)
    z_default = rec.run_reversible(x0, depth=4)
    z_default.pow(2).sum().backward()
    g_default = {n: p.grad.clone() for n, p in rec.named_parameters()}
    gx0_default = x0.grad.clone()
    for p in rec.parameters():
        p.grad = None
    x0b = x0.detach().clone().requires_grad_(True)
    # Explicit a0=b0=x0 must reproduce the implicit-default path exactly.
    z_explicit = rec.run_reversible(x0b, depth=4, a0=x0b, b0=x0b)
    z_explicit.pow(2).sum().backward()
    assert torch.equal(z_default.detach(), z_explicit.detach())
    for n, p in rec.named_parameters():
        assert torch.allclose(p.grad, g_default[n], atol=1e-12), n
    assert torch.allclose(x0b.grad, gx0_default, atol=1e-12)
