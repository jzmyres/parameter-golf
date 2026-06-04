import torch
from train_gpt_m0 import ReversibleRecurrence, _TinyDelta


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
