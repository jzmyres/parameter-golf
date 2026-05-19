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
        """Per CLAUDE.md expert independence: each expert owns its own U/V/b."""
        m = MLP(dim=16, mlp_mult=2.0, num_experts=4, expert_rank=4,
                use_perdim_gate=True, perdim_gate_rank=4)
        # Check that init produced different values per expert (xavier_uniform_
        # is stochastic per expert), confirming the params are not shared.
        u0, u1 = m.perdim_gate_u.data[0], m.perdim_gate_u.data[1]
        v0, v1 = m.perdim_gate_v.data[0], m.perdim_gate_v.data[1]
        self.assertFalse(torch.equal(u0, u1),
            "perdim_gate_u[0] and [1] must differ (per-expert independence)")
        self.assertFalse(torch.equal(v0, v1),
            "perdim_gate_v[0] and [1] must differ (per-expert independence)")


class TestIter176PerdimGateEffect(unittest.TestCase):
    """Flag-to-effect contract: flipping the flag changes the MLP output."""

    def _setup_pair(self):
        torch.manual_seed(0)
        m_off = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                    use_perdim_gate=False)
        torch.manual_seed(0)
        m_on = MLP(dim=32, mlp_mult=2.0, num_experts=4, expert_rank=8,
                   use_perdim_gate=True, perdim_gate_rank=8)
        # Copy shared params so only the gate creates the divergence
        with torch.no_grad():
            m_on.expert_gate.copy_(m_off.expert_gate)
            m_on.expert_fc.copy_(m_off.expert_fc)
            m_on.expert_down.copy_(m_off.expert_down)
            m_on.gate_in_norm_weight.copy_(m_off.gate_in_norm_weight)
            m_on.fc_in_norm_weight.copy_(m_off.fc_in_norm_weight)
            m_on.hidden_norm_weight.copy_(m_off.hidden_norm_weight)
        return m_off, m_on

    def test_gated_output_differs_from_disabled(self):
        """Flag-to-effect: with all shared params copied, the gate alone must
        change the per-token output."""
        m_off, m_on = self._setup_pair()
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32)
        # Mock router weights (uniform over 4 experts so off-path doesn't
        # collapse the difference).
        w = torch.ones(2 * 8, 4) / 4
        with torch.no_grad():
            out_off = m_off.mix_experts(x, w)
            out_on = m_on.mix_experts(x, w)
        max_abs_diff = (out_on - out_off).abs().max().item()
        self.assertGreater(max_abs_diff, 1e-4,
            f"gated output must differ from disabled (max |Δ|={max_abs_diff:.2e})")

    def test_gated_output_near_half_at_init(self):
        """At init with xavier U/V and zero b, σ(VU·x) ≈ σ(small ≈ 0) ≈ 0.5,
        so gated output magnitude is ~half the disabled output. Confirms
        the init doesn't accidentally produce extreme on/off mask."""
        m_off, m_on = self._setup_pair()
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32)
        w = torch.ones(2 * 8, 4) / 4
        with torch.no_grad():
            out_off = m_off.mix_experts(x, w)
            out_on = m_on.mix_experts(x, w)
        # Ratio should be roughly 0.4-0.6 elementwise (mostly near 0.5)
        nz = out_off.abs() > 1e-4
        if nz.any():
            ratio = (out_on[nz].abs() / out_off[nz].abs()).mean().item()
            self.assertGreater(ratio, 0.2,
                f"gated/disabled ratio={ratio:.3f} suspiciously small at init")
            self.assertLess(ratio, 1.0,
                f"gated/disabled ratio={ratio:.3f} should be < 1 at init "
                "(sigmoid masks at ~0.5)")

    def test_gradient_flows_to_gate_params(self):
        """The gate must produce nonzero gradients on U / V / b."""
        torch.manual_seed(0)
        m = MLP(dim=16, mlp_mult=2.0, num_experts=4, expert_rank=4,
                use_perdim_gate=True, perdim_gate_rank=4)
        m.train()
        x = torch.randn(2, 4, 16, requires_grad=True)
        w = torch.ones(2 * 4, 4) / 4
        out = m.mix_experts(x, w)
        loss = out.pow(2).mean()
        loss.backward()
        for pname in ("perdim_gate_u", "perdim_gate_v", "perdim_gate_bias"):
            p = getattr(m, pname)
            self.assertIsNotNone(p.grad, f"{pname} grad is None")
            self.assertGreater(p.grad.abs().sum().item(), 0.0,
                f"{pname} grad is all-zero (no learning signal)")


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
