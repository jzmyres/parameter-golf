"""iter163c TBPTT-aligned consistency anchor tests (2026-05-15).

Single loss term — anchor every gradient-carrying z to the most-converged
available FP proxy z_{K+1} (computed by one no-grad Parcae iter):

    L_anchor = anchor_coef * mean_i ‖z_{i} − z_{K+1}.detach()‖²

where ``i`` ranges over: (a) the iter152 prefix-anchor depths
``{16, 24, 32, ...}``; AND (b) the last ``deq_bptt_k - 1`` iterations of
the sampled K (TBPTT augmentation). This is strictly stronger than
pair-with-next (`z_i, z_{i+1}.detach()`): the target z_{K+1} is most-
converged so signal magnitude reflects "distance from FP" rather than
just one-step residual, and there is no trivial-zero collapse risk
because the target is input-driven (z_{K+1} = F(z_K, x0), not constant).

This file was rewritten 2026-05-15 to remove iter163's extension term
(`‖z_K − z_{K+Δ}.detach()‖²`, Δ=K_train, ~+60 % step time) which was
superseded by TBPTT augmentation + one extension iter at ~1/30 the cost
while preserving (and strengthening) the FP-condition signal.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import Hyperparameters
from test_arch import _make_model


class TestIter163cTBPTTAlignedConsistency(unittest.TestCase):
    """All tests use a tiny CPU-friendly model with K_train=4 and bptt_k=2
    so the TBPTT augmentation adds the K-1=3 anchor."""

    def _make(self, anchor_coef=0.1, bptt_k=2, prefix_anchor_set=(4,)):
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
        """iter163c (2026-05-15) keeps the iter163 anchor coef at 0.1 and
        removes the extension knobs (`multi_k_consistency_extension_coef`,
        `multi_k_consistency_extension_delta`) entirely. The Hyperparameter
        class must not even expose the removed fields — any consumer that
        still passes them is a stale call site."""
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.1)
        self.assertFalse(
            hasattr(Hyperparameters, "multi_k_consistency_extension_coef"),
            "iter163 extension coef was removed 2026-05-15 — must not reappear "
            "as a Hyperparameter field via copy-paste regression."
        )
        self.assertFalse(
            hasattr(Hyperparameters, "multi_k_consistency_extension_delta"),
            "iter163 extension delta was removed 2026-05-15 — must not reappear "
            "as a Hyperparameter field via copy-paste regression."
        )

    def test_anchor_loss_fires_at_small_K_via_tbptt_augmentation(self):
        """The iter163c TBPTT augmentation is the central improvement over
        iter163: at K_sampled where iter163's coarse jitter anchors give only
        one z_stack entry (zero pairs → zero anchor loss), iter163c adds the
        last `bptt_k - 1` iterations as extra anchors so pairs always exist.
        With prefix_anchor_set=(4,) and K_train=4, iter163's z_stack would
        have length 1 → no pairs → loss=0. iter163c with bptt_k=2 augments
        anchors to {3, 4} → 1 pair → loss > 0."""
        m = self._make(anchor_coef=0.1, bptt_k=2, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t,
            "anchor loss tensor must be populated under TBPTT augmentation")
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "anchor loss must be > 0 when TBPTT augmentation adds extra anchors "
            "even at K_sampled where iter152's coarse anchors give only one entry")

    def test_tbptt_augmentation_records_augmented_anchor_depths(self):
        """The augmented anchor set must include both prefix-anchor depths AND
        the TBPTT-window depths. Verified via `_deq_prefix_anchor_depths_last`."""
        m = self._make(anchor_coef=0.1, bptt_k=3, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        depths = m._deq_prefix_anchor_depths_last
        # K=4, bptt_k=3 → tbptt_extra = {K-1, K-2} = {3, 2}; merged with {4} → {2, 3, 4}.
        self.assertEqual(tuple(depths), (2, 3, 4),
            f"expected augmented anchors (2, 3, 4), got {depths}")

    def test_tbptt_augmentation_off_when_anchor_coef_zero(self):
        """The augmentation must NOT fire when the anchor coef is 0 — adding
        anchors costs backward work and memory, so disabling the loss must
        also disable the augmentation."""
        m = self._make(anchor_coef=0.0, bptt_k=3, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        depths = m._deq_prefix_anchor_depths_last
        # With anchor_coef=0, no augmentation → only the original prefix anchor {4}.
        self.assertEqual(tuple(depths), (4,),
            f"expected un-augmented anchors (4,) when anchor_coef=0, got {depths}")

    def test_anchor_loss_zero_when_coef_zero(self):
        """Flag-to-effect contract: when anchor_coef=0, the loss tensor stays
        None and the consistency term contributes nothing to total loss."""
        torch.manual_seed(42)
        m_off = self._make(anchor_coef=0.0, bptt_k=2, prefix_anchor_set=(2, 3, 4))
        torch.manual_seed(42)
        m_on = self._make(anchor_coef=0.1, bptt_k=2, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m_off)

        loss_off = float(m_off(x, y).detach())
        loss_on = float(m_on(x, y).detach())

        self.assertIsNone(m_off._consistency_anchor_loss_t)
        self.assertIsNotNone(m_on._consistency_anchor_loss_t)
        self.assertGreater(loss_on, loss_off,
            f"loss_on={loss_on:.6f} must exceed loss_off={loss_off:.6f} "
            "since consistency term only adds a positive penalty")

    def test_anchor_loss_requires_prefix_mode(self):
        """When prefix_mode=False (deq_prefix_anchors=False), the augmented
        anchor mechanism does not apply: there's no z_stack to compute pairwise
        differences from."""
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=False,
            num_refinements=0,
            deq_bptt_k=2,
            multi_k_consistency_anchor_coef=0.1,
            use_parcae=True,
        )
        m.train()
        m._deq_k_override = 4
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_anchor_loss_t)

    def test_anchor_loss_gradient_flows_to_parameters(self):
        """The consistency loss must actually propagate gradient to model
        params — Flag-to-effect contract: enabling the flag must change the
        training-path tensor flow observably."""
        m = self._make(anchor_coef=1.0, bptt_k=2, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        loss = m(x, y)
        loss.backward()
        had_grad = False
        for name, p in m.shared_block.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                had_grad = True
                break
        self.assertTrue(had_grad,
            "anchor consistency loss must produce non-zero gradients on "
            "shared_block parameters — otherwise the flag has no training effect")

    def test_anchor_on_deepest_target_structure(self):
        """The iter163c anchor-on-deepest design: every gradient-carrying z in
        z_stack is anchored to z_{K+1}.detach() (the most-converged available
        FP proxy from one no-grad Parcae iter), NOT to its immediate next
        iteration. This gives:
          - Strong directional signal toward the model's best FP estimate
          - Uniform target across all anchors (vs mixed coarse/fine targets)
          - No trivial-zero collapse risk (target is input-driven)

        The anchor-loss tensor must be a scalar (mean over |z_stack| pair
        contributions), and the augmented anchors include both prefix-anchor
        depths AND the TBPTT-window iterations.
        """
        # Use K=4, bptt_k=3, prefix_anchor_set=(4,) so augmented anchors = {2, 3, 4}.
        # z_stack has 3 entries, all targeting z_5 = F(z_4, x0).det → 3 pair
        # contributions in the mean. NOTE: `_deq_prefix_anchor_depths_last` is
        # populated DURING `_run_backbone`, so the depth check must come AFTER
        # the forward call.
        m = self._make(anchor_coef=0.1, bptt_k=3, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        depths = m._deq_prefix_anchor_depths_last
        self.assertEqual(tuple(depths), (2, 3, 4),
            f"expected augmented anchors (2, 3, 4) for bptt_k=3, got {depths}")
        loss_t = m._consistency_anchor_loss_t
        self.assertIsNotNone(loss_t,
            "anchor loss tensor must be populated under anchor-on-deepest")
        self.assertEqual(loss_t.dim(), 0,
            f"anchor loss must be a scalar (mean over pair contributions), "
            f"got tensor of shape {tuple(loss_t.shape)}")
        self.assertGreater(float(loss_t), 0.0,
            "anchor-on-deepest loss must be > 0 when z != z_{K+1} (typical)")

    def test_extension_helper_was_deleted(self):
        """iter163's `_consistency_extend_no_grad` helper was removed 2026-05-15.
        Negative assertion guards against reintroduction via copy-paste.

        Note: iter163c does one INLINE no-grad Parcae iter (~10 lines) inside
        `_run_backbone` to compute z_{K+1} — this is intentional and DOES NOT
        use the deleted helper. The negative assertion targets the helper
        function definition specifically (not the inline code path)."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        self.assertNotIn("def _consistency_extend_no_grad", src,
            "_consistency_extend_no_grad helper was removed 2026-05-15 — must "
            "not reappear; iter163c's inline 1-iter extension supersedes it.")
        self.assertNotIn("multi_k_consistency_extension_coef", src,
            "multi_k_consistency_extension_coef was removed 2026-05-15 — "
            "iter163c uses only the anchor term.")


if __name__ == "__main__":
    unittest.main()
