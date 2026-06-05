import torch
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
