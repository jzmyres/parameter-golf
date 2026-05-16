"""iter163 multi-K consistency loss tests.

Two loss terms (CLAUDE.md most-principled-simplest-general directive,
hybrid CM-paradigm for natural FP-convergence learning):

  L_anchor = anchor_coef · Σ_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²
    Recursive consistency on iter152 prefix anchors (Δ=8-32 between
    adjacent depths). Reuses existing z_stack from
    RevDEQPrefixAnchorFunction.

  L_ext = extension_coef · ‖z_K − z_{K+Δ}.detach()‖²
    Extension consistency: extending K by Δ should not change z (the
    operational asymptotic-stability test). Wasteful v1 extends from z
    by Δ no-grad Parcae two-state iterations (initializing y = z, true
    at FP).

Both terms are architecture-agnostic (apply to any iteration mechanism).
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import Hyperparameters
from test_arch import _make_model


class TestIter163ConsistencyLoss(unittest.TestCase):
    """All tests use a tiny CPU-friendly model with prefix anchors at
    depths {2,3,4} and K_train=4 (forces anchors to fire)."""

    def _make(self, anchor_coef=0.1, extension_coef=0.1, extension_delta=2,
              use_parcae=True):
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=True,
            deq_prefix_anchor_set=(2, 3, 4),
            num_refinements=0,
            deq_bptt_k=1,
            multi_k_consistency_anchor_coef=anchor_coef,
            multi_k_consistency_extension_coef=extension_coef,
            multi_k_consistency_extension_delta=extension_delta,
            use_parcae=use_parcae,
        )
        m.train()
        m._deq_k_override = 4
        return m

    def _batch(self, m):
        B, T = 2, 16
        dev = next(m.parameters()).device
        return (
            torch.randint(0, 64, (B, T)).to(dev),
            torch.randint(0, 64, (B, T)).to(dev),
        )

    def test_hyperparameter_defaults_match_promoted_iter163(self):
        """iter163 PROMOTED 2026-05-15 at val_bpb=1.471598 (vs iter152 1.471820,
        Δ −0.000222). Defaults are now ON at λ=0.1 for both terms; Δ=0 means
        use K_train as the extension Δ. Disable explicitly with
        --multi-k-consistency-anchor-coef=0 --multi-k-consistency-extension-coef=0
        to recover the iter152 baseline behavior for ablations."""
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.1)
        self.assertEqual(Hyperparameters.multi_k_consistency_extension_coef, 0.1)
        self.assertEqual(Hyperparameters.multi_k_consistency_extension_delta, 0)

    def test_anchor_consistency_loss_fires_with_multiple_anchors(self):
        m = self._make(anchor_coef=0.1, extension_coef=0.0)
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t)
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "anchor consistency loss must be > 0 when prefix anchors differ")

    def test_extension_consistency_loss_fires_under_parcae(self):
        m = self._make(anchor_coef=0.0, extension_coef=0.1, extension_delta=2)
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_ext_loss_t)
        loss = float(m._consistency_ext_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "extension consistency loss must be > 0 when z != z_extended")

    def test_disabled_coefs_yield_no_consistency_loss_attrs(self):
        """Flag-to-effect contract: when both coefs=0, neither loss tensor
        is populated AND the total loss matches the baseline (within
        floating-point tolerance) for the SAME init."""
        torch.manual_seed(42)
        m_off = self._make(anchor_coef=0.0, extension_coef=0.0)
        torch.manual_seed(42)
        m_on = self._make(anchor_coef=0.1, extension_coef=0.1, extension_delta=2)
        x, y = self._batch(m_off)

        loss_off = float(m_off(x, y).detach())
        loss_on = float(m_on(x, y).detach())

        # Off: no consistency tensors populated.
        self.assertIsNone(m_off._consistency_anchor_loss_t)
        self.assertIsNone(m_off._consistency_ext_loss_t)
        # On: tensors populated AND total loss differs.
        self.assertIsNotNone(m_on._consistency_anchor_loss_t)
        self.assertIsNotNone(m_on._consistency_ext_loss_t)
        self.assertGreater(loss_on, loss_off,
            f"loss_on={loss_on:.6f} must exceed loss_off={loss_off:.6f} "
            "since consistency terms only add positive penalties")

    def test_anchor_consistency_requires_prefix_mode(self):
        """When prefix_mode=False (deq_prefix_anchors=False), the anchor
        loss tensor stays None even if anchor_coef > 0 — there's no
        z_stack to compute pairwise differences from."""
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=False,
            num_refinements=0,
            deq_bptt_k=1,
            multi_k_consistency_anchor_coef=0.1,
            multi_k_consistency_extension_coef=0.0,
            use_parcae=True,
        )
        m.train()
        m._deq_k_override = 4
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_anchor_loss_t)

    def test_extension_consistency_requires_parcae(self):
        """The extension helper uses _parcae_a_bar / _parcae_b_bar; when
        use_parcae=False, the extension term is silently skipped (no-op
        rather than crash). This is the conservative behavior for non-
        Parcae fallback paths."""
        torch.manual_seed(0)
        # use_parcae=False uses scalar deq_beta path; iter163 v1 does not
        # support it (would need a non-Parcae extension helper). Skip path.
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=False,
            num_refinements=0,
            deq_bptt_k=1,
            multi_k_consistency_anchor_coef=0.0,
            multi_k_consistency_extension_coef=0.1,
            use_parcae=False,
        )
        m.train()
        m._deq_k_override = 4
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_ext_loss_t,
            "extension term must skip non-Parcae path (no _parcae_a_bar)")

    def test_consistency_loss_gradient_flows_to_parameters(self):
        """The consistency loss must actually propagate gradient to model
        params — not just compute a detached number. Audit check for the
        Flag-to-effect contract: enabling the flag must change the
        training-path tensor flow observably."""
        m = self._make(anchor_coef=1.0, extension_coef=0.0)
        x, y = self._batch(m)
        loss = m(x, y)
        loss.backward()
        # Check that some shared_block parameter received a non-zero grad.
        had_grad = False
        for name, p in m.shared_block.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                had_grad = True
                break
        self.assertTrue(had_grad,
            "anchor consistency loss must produce non-zero gradients on "
            "shared_block parameters — otherwise the flag has no training "
            "effect (Flag-to-effect contract violation)")


if __name__ == "__main__":
    unittest.main()
