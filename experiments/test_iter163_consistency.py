"""iter163c anchor-on-deepest consistency anchor tests (2026-05-16).

Single loss term — anchor every gradient-carrying z (the iter152 coarse
prefix-anchor entries in z_stack) to the most-converged available FP
proxy z_{K+1} (computed by one no-grad Parcae iter):

    L_anchor = anchor_coef * mean_i ‖z_{i} − z_{K+1}.detach()‖²

where ``i`` ranges over the iter152 prefix-anchor depths. This is strictly
stronger than pair-with-next (`z_i, z_{i+1}.detach()`): the target is
most-converged so signal magnitude reflects "distance from FP" rather than
just one-step residual, and there is no trivial-zero collapse risk because
the target is input-driven (z_{K+1} = F(z_K, x0), not constant).

A first iter163c attempt augmented anchors with TBPTT-window iterations
{K-bptt_k+1, ..., K-1} for per-iter pairs at every K_sampled, but that
OOM'd on dev L40S at step ~13 (~6 anchors × 3 bptt × T=2048 SDPA backward
activations > 44 GB VRAM). The coarse-only design is the simplest correct
expression of "anchor every gradient-carrying z to the most-converged
proxy" and fits VRAM.

This file was rewritten 2026-05-15 to remove iter163's extension term
(`‖z_K − z_{K+Δ}.detach()‖²`, Δ=K_train, ~+60 % step time) which was
superseded by the cheaper one-iter extension target while preserving
(and strengthening) the FP-condition signal.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import Hyperparameters
from test_arch import _make_model


class TestIter163cAnchorOnDeepestConsistency(unittest.TestCase):
    """All tests use a tiny CPU-friendly model with prefix anchors at
    multiple depths so the anchor loss has multiple terms in z_stack."""

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
        """iter163c (2026-05-16) keeps the iter163 anchor coef at 0.1 and
        removes the extension knobs (`multi_k_consistency_extension_coef`,
        `multi_k_consistency_extension_delta`) entirely. The Hyperparameter
        class must not even expose the removed fields."""
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

    def test_anchor_loss_fires_on_coarse_prefix_anchors(self):
        """With multi-anchor prefix set {2,3,4} at K_train=4, z_stack has 3
        entries (one per anchor depth). The anchor loss pairs each with
        z_{5} = F(z_4, x0).detach() and returns the mean."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t,
            "anchor loss tensor must be populated when coarse anchors > 0")
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "anchor loss must be > 0 when z_i != z_{K+1} (typical pre-trained state)")

    def test_anchor_loss_fires_with_single_anchor(self):
        """Even when only one prefix anchor exists (e.g. K_sampled=16 in real
        runs where only {16} <= 16), the iter163c design still fires: the
        single anchor (z@16) is paired with z_{17}.detach() (one no-grad iter
        beyond K). This is the key improvement over iter163's recursive
        anchor which gave zero pairs in this case."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t,
            "single-anchor z_stack must still produce a non-None loss "
            "(z_{K+1} target gives at least one pair).")
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0)

    def test_anchor_loss_zero_when_coef_zero(self):
        """Flag-to-effect contract: when anchor_coef=0, the loss tensor
        stays None and the consistency term contributes nothing to total
        loss."""
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
            f"loss_on={loss_on:.6f} must exceed loss_off={loss_off:.6f} "
            "since consistency term only adds a positive penalty")

    def test_anchor_loss_requires_prefix_mode(self):
        """When prefix_mode=False (deq_prefix_anchors=False), the
        anchor-on-deepest mechanism does not apply: there's no z_stack
        to compute distances from."""
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
        """The consistency loss must actually propagate gradient to model
        params — Flag-to-effect contract: enabling the flag must change the
        training-path tensor flow observably."""
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
            "anchor consistency loss must produce non-zero gradients on "
            "shared_block parameters — otherwise the flag has no training effect")

    def test_anchor_target_is_one_no_grad_iter_beyond_K(self):
        """The anchor target z_{K+1} is computed via ONE no-grad Parcae iter
        from z_K. This is the key cost/signal trade-off vs iter163's Δ=K_train
        extension (~22 no-grad iters): we get a slightly less converged but
        much cheaper target, which trains the model toward the FP nonetheless
        because rho < 1 means even one iter is contractive at converged points.

        Verified structurally via source inspection: the inline no-grad iter
        in `_run_backbone` must use the Parcae two-state update (a_bar, b_bar)
        rather than a full Δ-iteration helper.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        # The inline no-grad iter must use the Parcae machinery (a_bar, b_bar
        # casts to z.dtype). The detached target must be passed to the loss.
        assert "with torch.no_grad():" in src, (
            "iter163c requires a no_grad block to compute z_{K+1} without "
            "growing the autograd graph"
        )
        assert "y_next = a_bar_one * z_K" in src, (
            "iter163c inline no-grad iter must compute y_next via Parcae "
            "two-state update with a_bar"
        )
        assert "z_next = a_bar_one * z_K" in src, (
            "iter163c inline no-grad iter must compute z_next via Parcae "
            "two-state update with a_bar"
        )
        assert "(z_stack - target.unsqueeze(0)).pow(2).mean()" in src, (
            "iter163c loss must pair EVERY z_stack entry with the target "
            "(anchor-on-deepest) rather than pair-with-next"
        )

    def test_extension_helper_was_deleted(self):
        """iter163's `_consistency_extend_no_grad` helper was removed
        2026-05-15. Negative assertion guards against reintroduction via
        copy-paste.

        Note: iter163c does one INLINE no-grad Parcae iter (~10 lines) inside
        `_run_backbone` to compute z_{K+1} — this is intentional and DOES NOT
        use the deleted helper. The negative assertion targets the helper
        function definition specifically (not the inline code path).
        """
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
