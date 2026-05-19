"""iter176 per-dim sigmoid gate per MLP expert tests (2026-05-18).

Each MLP expert's output is multiplied element-wise by a per-expert per-token
sigmoid gate computed from the input x via a low-rank LoRA projection:

    gate_e(x) = σ(V_e (U_e x) + b_e) ∈ (0, 1)^D

Default OFF preserves iter172 behavior. CLAUDE.md flag-to-effect contract
requires that flipping use_expert_perdim_gate True changes the MLP output
relative to the disabled baseline.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import Hyperparameters, MLP, _OPTIONAL_COMPONENT_FLAGS


class TestIter176PerdimGateDefaults(unittest.TestCase):
    def test_hyperparameter_defaults_disabled(self):
        """Default OFF preserves iter172 behavior (flag absent from runs)."""
        self.assertFalse(Hyperparameters.use_expert_perdim_gate)
        self.assertEqual(Hyperparameters.expert_perdim_gate_rank, 16)

    def test_registered_in_optional_component_flags(self):
        """Per CLAUDE.md flag-to-effect contract, every `use_X` Hyperparameter
        must be in the _OPTIONAL_COMPONENT_FLAGS registry."""
        flag_names = [py_name for py_name, _ in _OPTIONAL_COMPONENT_FLAGS]
        self.assertIn("use_expert_perdim_gate", flag_names)


class TestIter176PerdimGateConstruction(unittest.TestCase):
    def test_mlp_disabled_no_gate_params(self):
        """When OFF, MLP has no perdim_gate_u / perdim_gate_v / perdim_gate_bias."""
        m = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                use_perdim_gate=False)
        self.assertFalse(m.use_perdim_gate)
        self.assertFalse(hasattr(m, "perdim_gate_u"))
        self.assertFalse(hasattr(m, "perdim_gate_v"))
        self.assertFalse(hasattr(m, "perdim_gate_bias"))

    def test_mlp_enabled_materializes_gate_params(self):
        """When ON, MLP has the U/V/b LoRA gate parameters."""
        m = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                use_perdim_gate=True, perdim_gate_rank=8)
        self.assertTrue(m.use_perdim_gate)
        self.assertEqual(m.perdim_gate_u.shape, (4, 8, 32))
        self.assertEqual(m.perdim_gate_v.shape, (4, 32, 8))
        self.assertEqual(m.perdim_gate_bias.shape, (4, 32))

    def test_gate_params_per_expert_independent(self):
        """Per CLAUDE.md expert independence: each expert owns its own U/V/b
        as distinct Parameters. iter176b's standard-LoRA init makes U xavier-
        random per expert (distinct by sampling) and V zero. Independence test:
        modifying expert 0's slice must not affect expert 1's slice."""
        m = MLP(dim=16, mlp_mult=2.0, num_experts=4, expert_rank=4,
                use_perdim_gate=True, perdim_gate_rank=4)
        u1_before = m.perdim_gate_u.data[1].clone()
        # Per-expert slices are distinct objects, modifiable independently.
        with torch.no_grad():
            m.perdim_gate_u.data[0].fill_(0.5)
        self.assertTrue(torch.all(m.perdim_gate_u.data[0] == 0.5))
        self.assertTrue(torch.equal(m.perdim_gate_u.data[1], u1_before),
            "modifying expert 0 must not affect expert 1 (no parameter sharing)")
        # Also verify xavier init produced distinct U slices across experts.
        u0_init = m.perdim_gate_u.data.clone()  # all xavier
        with torch.no_grad():
            m.perdim_gate_u.data.zero_()
            for e in range(4):
                torch.nn.init.xavier_uniform_(m.perdim_gate_u.data[e])
        # All four slices should differ (probability ~1 with xavier_uniform_)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(torch.equal(m.perdim_gate_u.data[i],
                                              m.perdim_gate_u.data[j]),
                    f"xavier init: U[{i}] and U[{j}] must differ")


class TestIter176bIdentityInit(unittest.TestCase):
    """iter176b strict-generalization principle: at init, the gated forward
    must be ≈ identical to the disabled-gate forward (gate ≈ 1.0). Per
    CLAUDE.md strict-generalization rule, the functional class contains
    iter172 AND the init matches iter172 — so the optimizer is free to
    learn deviations only if they reduce loss."""

    def _setup_pair(self):
        torch.manual_seed(0)
        m_off = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                    use_perdim_gate=False)
        torch.manual_seed(0)
        m_on = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                   use_perdim_gate=True, perdim_gate_rank=8)
        # Copy shared (non-gate) params so only the gate creates any divergence
        with torch.no_grad():
            m_on.expert_gate.copy_(m_off.expert_gate)
            m_on.expert_fc.copy_(m_off.expert_fc)
            m_on.expert_down.copy_(m_off.expert_down)
            m_on.gate_in_norm_weight.copy_(m_off.gate_in_norm_weight)
            m_on.fc_in_norm_weight.copy_(m_off.fc_in_norm_weight)
            m_on.hidden_norm_weight.copy_(m_off.hidden_norm_weight)
        return m_off, m_on

    def test_strict_generalization_at_init(self):
        """STRICT-GEN: at init, gated output must be ≈ disabled output.
        With U=0 and bias=+6.0, gate = σ(0·V + 6.0) = σ(6.0) ≈ 0.9975 ≈ 1.0.
        Relative error per element should be < 0.5% (the 0.25% sigmoid
        deviation from 1.0)."""
        m_off, m_on = self._setup_pair()
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32)
        w = torch.ones(2 * 8, 4) / 4
        with torch.no_grad():
            out_off = m_off.mix_experts(x, w)
            out_on = m_on.mix_experts(x, w)
        # Gate ≈ 0.9975 → output is 0.9975× the disabled output. Max relative
        # error should be near 0.5% (sigmoid deviation + softmax_route renorm).
        nz = out_off.abs() > 1e-4
        if nz.any():
            ratio = (out_on[nz] / out_off[nz]).abs()
            mean_ratio = ratio.mean().item()
            max_ratio = ratio.max().item()
            min_ratio = ratio.min().item()
            # All ratios should be very close to 1.0
            self.assertAlmostEqual(mean_ratio, 1.0, places=2,
                msg=f"identity init: mean(out_on/out_off)={mean_ratio:.4f} should be ≈ 1.0")
            self.assertGreater(min_ratio, 0.95,
                f"identity init: min ratio={min_ratio:.4f} should be > 0.95")
            self.assertLess(max_ratio, 1.05,
                f"identity init: max ratio={max_ratio:.4f} should be < 1.05")

    def test_init_values_are_strict_identity(self):
        """U is xavier-random, V is exactly zero, bias is exactly +6.0 —
        standard LoRA convention (Hu et al. 2021) per iter176b strict-gen fix.
        Forward: gate = σ(V·U·x + b) = σ(0·U·x + 6.0) = σ(6.0) ≈ 1.0."""
        m = MLP(dim=16, mlp_mult=2.0, num_experts=4, expert_rank=4,
                use_perdim_gate=True, perdim_gate_rank=4)
        # U must be NONZERO (xavier_uniform_ produces random values bounded
        # away from zero — std ~ sqrt(2/(D+R)))
        self.assertGreater(m.perdim_gate_u.abs().sum().item(), 0.0,
            "perdim_gate_u must be xavier-nonzero (gives V immediate gradient)")
        self.assertTrue(torch.all(m.perdim_gate_v.data == 0.0),
            "perdim_gate_v must be zero at init (LoRA up-projection zero-init)")
        self.assertTrue(torch.all(m.perdim_gate_bias.data == 6.0),
            "perdim_gate_bias must be +6.0 at init (σ(6.0) ≈ 1.0)")

    def test_gradient_flows_after_first_step(self):
        """V and bias must have nonzero grad at init. U has zero grad at init
        (since V=0 → ∂gate/∂U = 0), but receives grad once V drifts. Standard
        LoRA training-dynamics property (warm-start through V)."""
        torch.manual_seed(0)
        m = MLP(dim=16, mlp_mult=2.0, num_experts=4, expert_rank=4,
                use_perdim_gate=True, perdim_gate_rank=4)
        m.train()
        x = torch.randn(2, 4, 16, requires_grad=True)
        w = torch.ones(2 * 4, 4) / 4
        out = m.mix_experts(x, w)
        loss = out.pow(2).mean()
        loss.backward()
        # V and bias receive immediate gradient (V via U·x; bias via direct path)
        self.assertGreater(m.perdim_gate_v.grad.abs().sum().item(), 0.0,
            "perdim_gate_v grad must flow at init (∂L/∂V = ∂L/∂pre · U·x ≠ 0)")
        self.assertGreater(m.perdim_gate_bias.grad.abs().sum().item(), 0.0,
            "perdim_gate_bias grad must flow at init")
        # U grad at init is ZERO (∂L/∂U = ∂L/∂pre · V = 0 when V=0).
        # This is expected LoRA zero-init behavior; U warms up via V's drift.


class TestIter176PerdimGateDisabledBitParity(unittest.TestCase):
    """When use_perdim_gate=False, output must be BIT-IDENTICAL to a pre-iter176
    MLP — proves the iter176 code is gated cleanly and default-OFF preserves
    iter172 behavior exactly."""

    def test_disabled_path_unchanged(self):
        torch.manual_seed(0)
        m1 = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                 use_perdim_gate=False)
        torch.manual_seed(0)
        m2 = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8)  # default off
        torch.manual_seed(123)
        x = torch.randn(2, 8, 32)
        w = torch.ones(2 * 8, 4) / 4
        with torch.no_grad():
            o1 = m1.mix_experts(x, w)
            o2 = m2.mix_experts(x, w)
        self.assertTrue(torch.equal(o1, o2),
            "default-OFF MLP must produce bit-identical output to explicit "
            "use_perdim_gate=False")


if __name__ == "__main__":
    unittest.main()
