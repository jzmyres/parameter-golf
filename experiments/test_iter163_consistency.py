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

    def _make(self, anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4), target_delta=1):
        torch.manual_seed(0)
        m = _make_model(
            num_experts=4, use_ctp=False,
            deq_prefix_anchors=True,
            deq_prefix_anchor_set=prefix_anchor_set,
            num_refinements=0,
            deq_bptt_k=bptt_k,
            multi_k_consistency_anchor_coef=anchor_coef,
            multi_k_consistency_target_delta=target_delta,
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
        """iter163c v2 (2026-05-16) keeps the iter163 anchor coef at 0.1 and
        introduces `multi_k_consistency_target_delta` (default 1 = z_{K+1}
        target; iter167 promotes 8). The REMOVED iter163 knobs
        (`multi_k_consistency_extension_coef`, `multi_k_consistency_extension_delta`)
        must NOT reappear — they controlled a separate boundary extension
        loss that was superseded by the anchor-on-deepest design."""
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.1)
        self.assertEqual(Hyperparameters.multi_k_consistency_target_delta, 1,
            "iter163c v2 default target Δ is 1 (z_{K+1}); iter167 will A/B "
            "test Δ=8 as a follow-up. Keep default 1 until iter167 promotes.")
        self.assertFalse(
            hasattr(Hyperparameters, "multi_k_consistency_extension_coef"),
            "iter163 extension coef was removed 2026-05-15 — must not reappear "
            "as a Hyperparameter field via copy-paste regression. The new "
            "`multi_k_consistency_target_delta` controls anchor target depth, "
            "NOT a separate boundary extension loss."
        )
        self.assertFalse(
            hasattr(Hyperparameters, "multi_k_consistency_extension_delta"),
            "iter163 extension delta was removed 2026-05-15 and replaced by "
            "`multi_k_consistency_target_delta` with different semantics — "
            "the old name must not reappear."
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

    def test_anchor_target_is_delta_no_grad_iters_beyond_K(self):
        """The anchor target z_{K+Δ} is computed via Δ no-grad Parcae iters
        from z_K. iter163c v2 default Δ=1; iter167 promotes Δ=8. The loop
        starts with y_e = z_e = z_K and iterates the Parcae two-state update.

        Verified structurally via source inspection: the inline no-grad loop
        in `_run_backbone` must use the Parcae machinery (a_bar, b_bar) and
        iterate `multi_k_consistency_target_delta` times.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        assert "with torch.no_grad():" in src, (
            "iter163c requires a no_grad block to compute z_{K+Δ} without "
            "growing the autograd graph"
        )
        assert "delta = max(1, int(self.multi_k_consistency_target_delta))" in src, (
            "iter167: anchor target depth Δ must be controlled by the "
            "`multi_k_consistency_target_delta` Hyperparameter (clamped to >=1)"
        )
        assert "for _ in range(delta):" in src, (
            "iter167: the inline no-grad block must iterate Δ times "
            "(loop required for Δ > 1)"
        )
        assert "y_e = a_bar_one * y_e + one_minus_a_one * sb_inner(z_e" in src, (
            "iter167 inline no-grad iter must use Parcae two-state update "
            "(y_e via current z_e and a_bar damping)"
        )
        assert "z_e = a_bar_one * z_e + one_minus_a_one * sb_inner(y_e" in src, (
            "iter167 inline no-grad iter must update z_e via the freshly-"
            "computed y_e (two-state Parcae cycle)"
        )
        assert "(z_stack - target.unsqueeze(0)).pow(2).mean()" in src, (
            "iter163c loss must pair EVERY z_stack entry with the target "
            "(anchor-on-deepest) rather than pair-with-next"
        )

    def test_iter167_target_delta_changes_loss_observably(self):
        """iter167: with target_delta=8 (vs default 1), the anchor target is
        z_{K+8} instead of z_{K+1}. For a non-trivial model the targets differ
        and the loss values must differ — Flag-to-effect contract.

        We use the same model weights / inputs / seed, varying ONLY the Δ knob.
        At small K_train=4 with bptt_k=1, target_delta=8 reaches z_{K+8}=z_12
        which is substantially more iters away from z_K=z_4 than z_5 is — the
        consistency loss should observably differ.
        """
        torch.manual_seed(42)
        m_d1 = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(4,), target_delta=1)
        torch.manual_seed(42)
        m_d8 = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(4,), target_delta=8)
        x, y = self._batch(m_d1)
        _ = m_d1(x, y)
        _ = m_d8(x, y)
        loss_d1 = float(m_d1._consistency_anchor_loss_t.detach())
        loss_d8 = float(m_d8._consistency_anchor_loss_t.detach())
        # The Δ=8 target is more converged → loss can be EITHER larger or
        # smaller than Δ=1 (depends on whether the iteration map is overshooting
        # or undershooting at K=4 init). The Flag-to-effect requirement is
        # only that the values differ above the noise floor.
        diff = abs(loss_d1 - loss_d8)
        self.assertGreater(diff, 1e-6,
            f"target_delta=1 vs 8 must produce observably different loss "
            f"values (got {loss_d1:.6f} vs {loss_d8:.6f}, diff={diff:.2e}); "
            "otherwise the Δ knob has no training effect.")

    def test_iter167_target_delta_min_clamp(self):
        """target_delta is clamped to at least 1 in the setter (max(1, int(x)))
        so passing 0 or negative values silently becomes Δ=1, preserving the
        iter163c v2 default behavior. Verified via attribute readback."""
        m_zero = self._make(target_delta=0)
        self.assertEqual(m_zero.multi_k_consistency_target_delta, 1,
            "target_delta=0 must clamp to 1 (Δ=1 default behavior)")
        m_neg = self._make(target_delta=-5)
        self.assertEqual(m_neg.multi_k_consistency_target_delta, 1,
            "target_delta<0 must clamp to 1; negative Δ is meaningless")

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
