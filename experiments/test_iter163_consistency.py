"""iter171 recursive nearest-neighbor consistency anchor tests (2026-05-16).

Single loss term — pair each prefix-anchor z_i with its NEXT-DEEPER z_{i+1}:

    L_anchor = anchor_coef * mean_i ‖z_{prefix_i} − z_{prefix_{i+1}}.detach()‖²

This is iter163's PROMOTED recursive anchor formula. Zero extra forward
compute. Architecture-agnostic per CLAUDE.md most-principled-simplest-general.

Supersedes iter170's all-to-deepest design (REFUTED at step 200 val with
val_bpb=2.067 vs baseline 2.003, +65 mBPB regression). iter170's all-to-
deepest pairing had ρ → 0 as its unique global minimum — model satisfied
all pairs simultaneously by making F flat in z (degenerate DEQ, mlp_ortho
exploded +0.40). Recursive pairs only require LOCAL contraction over each
depth range; ρ ≈ 0.85 is a feasible solution.

K_sampled=16 case: with only one prefix anchor, no pair fires; the
principled fix is to add a shallow anchor (K=4) via the CLI override
`--deq-prefix-anchor-set "4,16,32,64,128,256"`.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import Hyperparameters
from test_arch import _make_model


class TestIter171RecursiveAnchor(unittest.TestCase):
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

    def test_hyperparameter_defaults_match_promoted_iter172(self):
        """iter172 (2026-05-17, PROMOTED at val_bpb=1.462898 vs iter163's
        1.471598 = −8.7 mBPB) anchor_coef=0.1 + prefix_anchor_set with K=8
        shallow anchor. The REMOVED knobs (`multi_k_consistency_extension_coef`,
        `multi_k_consistency_extension_delta`, `multi_k_consistency_target_delta`)
        must NOT reappear via copy-paste regression."""
        self.assertEqual(Hyperparameters.multi_k_consistency_anchor_coef, 0.1)
        self.assertEqual(
            Hyperparameters.deq_prefix_anchor_set,
            (8, 16, 24, 32, 64, 128),
            "iter172 promoted with K=8 shallow anchor (gap=8 to next-deeper "
            "K=16, matching iter163's healthy regime). Default must NOT revert "
            "to () fallback (loses the K=8 anchor that gives consistency loss "
            "a non-trivial pair at K_sampled=16) or to iter170's K=4 (gap=12 "
            "collapsed rho_F→0)."
        )
        for removed in (
            "multi_k_consistency_extension_coef",
            "multi_k_consistency_extension_delta",
            "multi_k_consistency_target_delta",
        ):
            self.assertFalse(
                hasattr(Hyperparameters, removed),
                f"{removed} was removed — must not reappear as a Hyperparameter "
                "field; iter172 uses recursive z_{i+1}.detach() target with no "
                "extra knobs."
            )

    def test_anchor_loss_fires_on_multi_anchor_z_stack(self):
        """With prefix anchors {2, 3, 4} at K_sampled=4, z_stack has 3 entries
        (z_2, z_3, z_4). iter172 uses RECURSIVE nearest-neighbor pairing:
        pairs are (z_2, z_3.det) and (z_3, z_4.det) = 2 non-trivial pairs.
        (iter170's all-to-deepest variant pairing both z_2 and z_3 with z_4
        was REFUTED at +65 mBPB regression; do NOT reintroduce.)"""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(2, 3, 4))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNotNone(m._consistency_anchor_loss_t)
        loss = float(m._consistency_anchor_loss_t.detach())
        self.assertGreater(loss, 0.0,
            "anchor loss must be > 0 when consecutive anchors differ")

    def test_anchor_loss_skipped_with_single_anchor(self):
        """K_sampled=16 with a prefix anchor set containing only one match
        (e.g. {16}) has z_stack.shape[0] == 1 → z_stack[:-1] is empty →
        loss must NOT fire (no pairs to compute). This is the K_sampled=16
        coverage gap that iter172's default anchor set fixes by including
        K=8 as a shallow anchor."""
        m = self._make(anchor_coef=0.1, bptt_k=1, prefix_anchor_set=(4,))
        x, y = self._batch(m)
        _ = m(x, y)
        self.assertIsNone(m._consistency_anchor_loss_t,
            "z_stack with only 1 entry produces no pairs; loss must be None")

    def test_anchor_loss_with_shallow_anchor_fires_at_small_K(self):
        """iter172 design: include a shallow anchor (K=8 in production) so
        even small K_sampled produces a pair. At K_sampled=4 (test proxy for
        K_sampled=16 in production), anchors = {2, 4} → 1 pair (z_2, z_4.det).
        This is the principled fix for the single-anchor coverage gap."""
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

    def test_anchor_loss_uses_recursive_pairs(self):
        """Structural assertion via source inspection: iter171 uses recursive
        pairs `(z_stack[:-1] - z_stack[1:].detach()).pow(2).mean()`. The
        all-to-deepest formula `(z_stack[:-1] - z_stack[-1].unsqueeze(0).detach())`
        from iter170 must NOT appear (it caused ρ → 0 degeneracy at s200)."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        # iter171 recursive formula — pairs adjacent z_stack entries
        self.assertIn("(z_stack[:-1] - z_stack[1:].detach()).pow(2).mean()", src,
            "iter171: anchor loss must use recursive nearest-neighbor pairs "
            "(z_stack[:-1] vs z_stack[1:].detach()); the all-to-deepest "
            "formula from iter170 caused ρ → 0 degeneracy at s200.")
        # Negative assertion: iter170's all-to-deepest pattern must NOT reappear
        self.assertNotIn("target = z_stack[-1].detach()", src,
            "iter170's all-to-deepest target (`z_stack[-1].detach()`) was "
            "REFUTED at s200 (val_bpb +65 mBPB, mlp_ortho +0.40). Must not "
            "reappear via copy-paste regression.")
        self.assertNotIn("(z_stack[:-1] - target.unsqueeze(0)).pow(2).mean()", src,
            "iter170's all-to-deepest pair formula was REFUTED.")
        # Negative assertions for removed extension/no-grad-loop machinery
        self.assertNotIn("def _consistency_extend_no_grad", src,
            "_consistency_extend_no_grad helper was removed.")
        self.assertNotIn("multi_k_consistency_target_delta", src,
            "multi_k_consistency_target_delta was removed.")
        self.assertNotIn("multi_k_consistency_extension_coef", src,
            "iter163 extension term was removed.")


if __name__ == "__main__":
    unittest.main()
