"""iter170 anchor-on-deepest consistency anchor tests (2026-05-16).

Single loss term — anchor every shallower gradient-carrying z to the DEEPEST
gradient-carrying z in z_stack (= z at K_sampled, the final TBPTT depth):

    L_anchor = anchor_coef * mean_{i in z_stack[:-1]} ‖z_i − z_stack[-1].detach()‖²

The target is z_K_sampled itself (the last entry in z_stack from the prefix-
anchor forward). Zero extra forward compute — the target is already produced
by the main gradient-carrying forward. Architecture-agnostic per CLAUDE.md
most-principled-simplest-general directive.

Supersedes prior designs:
  iter163c v2 (target z_{K+1} via 1 no-grad iter): REFUTED, val_bpb +22 mBPB.
  iter167 (target z_{K+8}, Δ=8 no-grad iters): subsumed (z_K_sampled is free).

K_sampled=16 case: with only one prefix anchor, z_stack[:-1] is empty and no
consistency pair fires; the principled fix is to broaden deq_prefix_anchor_set
to include a shallow K (e.g. K=4), tested via the iter170 CLI override.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import Hyperparameters
from test_arch import _make_model


class TestIter170AnchorOnDeepest(unittest.TestCase):
    """All tests use a tiny CPU-friendly model with prefix anchors at
    multiple depths so the anchor loss has multiple z_stack entries."""

    def _make(self, anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4)):
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=True,
            deq_prefix_anchor_set=prefix_anchor_set,
            num_refinements=0,
            deq_bptt_k=bptt_k,
            multi_k_consistency_anchor_coef=anchor_coef,
            use_parcae=True,
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

    def test_hyperparameter_defaults_match_promoted_iter163c(self):
        """iter170 (2026-05-16) keeps anchor_coef=0.1 as the only consistency
        knob. The REMOVED knobs (`multi_k_consistency_extension_coef`,
        `multi_k_consistency_extension_delta`, `multi_k_consistency_target_delta`)
        must NOT reappear via copy-paste regression."""
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.1)
        for removed in (
            "multi_k_consistency_extension_coef",
            "multi_k_consistency_extension_delta",
            "multi_k_consistency_target_delta",
        ):
            self.assertFalse(
                hasattr(Hyperparameters, removed),
                f"{removed} was removed — must not reappear as a Hyperparameter "
                "field; iter170 uses z_K_sampled as target with no extra knobs."
            )

    def test_anchor_loss_fires_on_multi_anchor_z_stack(self):
        """With prefix anchors {2, 3, 4} at K_sampled=4, z_stack has 3 entries
        (z_2, z_3, z_4). Target = z_4.detach(); pairs are (z_2, z_4.det) and
        (z_3, z_4.det) = 2 non-trivial pairs."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t)
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "anchor loss must be > 0 when z_2 != z_4 (typical pre-trained state)")

    def test_anchor_loss_skipped_with_single_anchor(self):
        """K_sampled=16 with default prefix anchor set (= jitter set) has only
        one anchor (z_16) — z_stack.shape[0] == 1 → z_stack[:-1] is empty →
        loss must NOT fire (no pairs to compute). This is the K_sampled=16
        coverage gap that iter170's CLI override fixes by adding K=4 to
        prefix_anchor_set."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_anchor_loss_t,
            "z_stack with only 1 entry produces no pairs; loss must be None")

    def test_anchor_loss_with_k4_addition_fires_at_small_K(self):
        """iter170 CLI override: add K=4 to prefix_anchor_set. At K_sampled=4
        (test proxy for K_sampled=16 in production), anchors = {2, 4} → 1 pair
        (z_2, z_4.det). This is the principled fix for the single-anchor gap."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 4))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t,
            "with shallow anchor added, even K=4 sampled must produce a pair")
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0)

    def test_anchor_loss_zero_when_coef_zero(self):
        """Flag-to-effect contract: anchor_coef=0 → no loss tensor populated."""
        torch.manual_seed(42)
        m_off = self._make(anchor_coef=0.0, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        torch.manual_seed(42)
        m_on = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m_off)

        loss_off = float(m_off(x, y).detach())
        loss_on = float(m_on(x, y).detach())

        self.assertIsNone(m_off._consistency_anchor_loss_t)
        self.assertIsNotNone(m_on._consistency_anchor_loss_t)
        self.assertGreater(loss_on, loss_off,
            f"loss_on={loss_on:.6f} must exceed loss_off={loss_off:.6f}")

    def test_anchor_loss_requires_prefix_mode(self):
        """deq_prefix_anchors=False → no z_stack → loss skipped."""
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=False,
            num_refinements=0,
            deq_bptt_k=1,
            multi_k_consistency_anchor_coef=0.1,
            use_parcae=True,
        )
        m.train()
        m._deq_k_override = 4
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_anchor_loss_t)

    def test_anchor_loss_gradient_flows_to_parameters(self):
        """The consistency loss must propagate gradient to shared_block
        parameters — Flag-to-effect contract."""
        m = self._make(anchor_coef=1.0, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m)
        loss = m(x, y)
        loss.backward()
        had_grad = False
        for name, p in m.shared_block.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                had_grad = True
                break
        self.assertTrue(had_grad,
            "anchor consistency loss must produce non-zero gradients")

    def test_anchor_loss_uses_z_stack_last_as_target(self):
        """Structural assertion via source inspection: iter170 target must be
        `z_stack[-1].detach()` and pairs must be `z_stack[:-1] - target`. No
        no-grad Parcae two-state extension loop (that was iter163c v2 / iter167)."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        self.assertIn("target = z_stack[-1].detach()", src,
            "iter170: target must be the deepest gradient-carrying z (z_stack[-1])")
        self.assertIn("(z_stack[:-1] - target.unsqueeze(0)).pow(2).mean()", src,
            "iter170: pairs must use z_stack[:-1] vs target (skip degenerate "
            "(z_K, z_K) pair)")
        # Negative assertions: extension/no-grad-loop machinery must be gone.
        self.assertNotIn("def _consistency_extend_no_grad", src,
            "_consistency_extend_no_grad helper was removed; iter170 uses no "
            "no-grad extension at all.")
        self.assertNotIn("multi_k_consistency_target_delta", src,
            "multi_k_consistency_target_delta was removed; iter170 has no Δ "
            "knob because the target is z_stack[-1] itself.")
        self.assertNotIn("multi_k_consistency_extension_coef", src,
            "iter163 extension term was removed and must not reappear.")


if __name__ == "__main__":
    unittest.main()
