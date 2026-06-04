import torch

from train_gpt_m0 import MLAttention


def test_mla_shapes_and_kv_latent():
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8)
    x = torch.randn(2, 6, 32)
    y = m(x)
    assert y.shape == x.shape
    assert m.kv_latent == 8


def test_mla_is_causal():
    """Output at position t must not depend on inputs at positions > t."""
    torch.manual_seed(0)
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8).double().eval()
    T = 7
    x = torch.randn(1, T, 32, dtype=torch.float64)
    with torch.no_grad():
        y_full = m(x)
    # Perturb the LAST token's input; earlier outputs must be unchanged.
    x2 = x.clone()
    x2[:, -1, :] += 1.0
    with torch.no_grad():
        y_pert = m(x2)
    assert torch.allclose(y_full[:, :-1], y_pert[:, :-1], atol=1e-9)
    # Sanity: the last position's output DID change (causality, not a no-op).
    assert not torch.allclose(y_full[:, -1], y_pert[:, -1], atol=1e-6)


def test_mla_q_latent_override():
    m = MLAttention(dim=32, n_heads=4, n_kv_heads=2, kv_latent=8, head_dim=8, q_latent=12)
    x = torch.randn(2, 5, 32)
    assert m(x).shape == x.shape
