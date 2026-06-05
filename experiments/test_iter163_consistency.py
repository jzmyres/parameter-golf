"""Finite-horizon OPG contracts replacing the legacy iter163 consistency path.

The active project document makes finite-horizon training the primary path:
sample deep horizons, optimize the final endpoint, and add a one-sided
no-degradation hinge against a shallow stop-gradient endpoint. The old
fixed-point recursive consistency anchor is legacy and must be rejected at
startup if requested.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy.train_gpt_rich import Hyperparameters, _validate_hyperparameters


class TestFiniteHorizonOPG(unittest.TestCase):
    def _cfg(self, **overrides):
        cfg = Hyperparameters()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_defaults_match_active_finite_horizon_doc(self):
        self.assertIs(Hyperparameters.deq_prefix_anchors, False)
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.0)
        self.assertEqual(Hyperparameters.deq_k_jitter_set, (32, 64, 128))
        self.assertEqual(Hyperparameters.deq_k_jitter_weights, (0.20, 0.40, 0.40))
        self.assertEqual(
            Hyperparameters.finite_horizon_pairs,
            ((16, 64), (16, 128), (32, 128)),
        )
        self.assertGreater(Hyperparameters.finite_horizon_scale_coef, 0.0)
        self.assertEqual(Hyperparameters.finite_horizon_scale_margin, 0.0)

    def test_legacy_consistency_flags_are_rejected(self):
        for cfg in (
            self._cfg(deq_prefix_anchors=True),
            self._cfg(multi_k_consistency_anchor_coef=0.1),
        ):
            with self.assertRaises(SystemExit) as ctx:
                _validate_hyperparameters(cfg)
            msg = str(ctx.exception).lower()
            self.assertIn("legacy", msg)
            self.assertIn("consistency", msg)

    def test_scale_hinge_is_one_sided_and_stops_shallow_gradient(self):
        from legacy.train_gpt_rich import _finite_horizon_scale_hinge

        shallow = torch.tensor(2.0, requires_grad=True)
        deep = torch.tensor(2.3, requires_grad=True)
        hinge = _finite_horizon_scale_hinge(
            shallow, deep, coef=0.5, margin=0.0,
        )
        self.assertAlmostEqual(float(hinge.detach()), 0.15, places=6)
        hinge.backward()
        self.assertIsNone(shallow.grad)
        self.assertAlmostEqual(float(deep.grad), 0.5, places=6)

        improved = _finite_horizon_scale_hinge(
            torch.tensor(2.0), torch.tensor(1.9), coef=0.5, margin=0.0,
        )
        self.assertEqual(float(improved), 0.0)

    def test_paired_depth_metrics_use_same_sequence_losses(self):
        from legacy.train_gpt_rich import _paired_depth_metrics

        shallow = torch.tensor([2.0, 1.0, 3.0])
        deep = torch.tensor([1.5, 1.1, 2.0])
        hard_mask = torch.tensor([False, True, True])
        metrics = _paired_depth_metrics(
            shallow, deep, epsilon=0.05, hard_mask=hard_mask,
        )
        self.assertAlmostEqual(float(metrics["gain"].item()), (0.5 - 0.1 + 1.0) / 3.0)
        self.assertAlmostEqual(float(metrics["ndr"].item()), 1.0 / 3.0)
        self.assertAlmostEqual(float(metrics["hard_gain"].item()), 0.45)

    def test_finite_horizon_training_populates_hinge_not_consistency(self):
        from test_arch import _make_model

        torch.manual_seed(0)
        model = _make_model(
            num_experts=4,
            use_ctp=False,
            deq_prefix_anchors=False,
            num_refinements=0,
            deq_bptt_k=1,
            finite_horizon_scale_coef=0.1,
            finite_horizon_pairs=((2, 4),),
        )
        model.train()
        model._deq_k_override = 4
        x = torch.randint(0, 64, (1, 8), device=next(model.parameters()).device)
        y = torch.randint(0, 64, (1, 8), device=next(model.parameters()).device)

        loss = model(x, y)

        self.assertTrue(loss.requires_grad)
        self.assertEqual(model._finite_horizon_depths_last, (2, 4))
        self.assertIsInstance(model._scale_hinge_loss_t, torch.Tensor)
        self.assertEqual(tuple(model._scale_hinge_loss_t.shape), ())
        self.assertFalse(model._scale_hinge_loss_t.requires_grad)
        self.assertTrue(torch.isfinite(model._scale_hinge_loss_t).item())
        self.assertIsNone(model._consistency_anchor_loss_t)


if __name__ == "__main__":
    unittest.main()
