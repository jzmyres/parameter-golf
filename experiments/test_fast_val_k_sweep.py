"""iter171 fast K-sweep at fast-val tests (2026-05-16).

When `--fast-val-k-sweep-set "4,128"` is set, the trainer re-runs validation
on the fast-val subset at each extra K and emits `fast_k_sweep:k4=X,k16=Y,
k128=Z` to the log. Diagnostic only — lets us detect degenerate K-sweep
patterns (e.g. K=4 < K=16 < K=128, the iter170 over-pressure signature)
at every val checkpoint instead of waiting for end-of-training K-sweep.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import Hyperparameters, _CLI_TUNABLE_KNOBS


class TestFastValKSweep(unittest.TestCase):
    def test_hyperparameter_default_is_empty_tuple(self):
        """Default disabled — empty tuple → no extra K-sweep, no cost added
        to existing runs unless the CLI flag is set explicitly."""
        self.assertEqual(Hyperparameters.fast_val_k_sweep_set, ())

    def test_cli_knob_is_registered(self):
        """`--fast-val-k-sweep-set` must be in the CLI knob registry so the
        argparse parser accepts it (and Unknown args check doesn't reject)."""
        self.assertIn("fast-val-k-sweep-set", _CLI_TUNABLE_KNOBS)

    def test_fast_val_block_calls_k_sweep_loop(self):
        """Structural assertion via source inspection: the fast-val block
        must include the loop over `args.fast_val_k_sweep_set` and emit a
        `fast_k_sweep:` log field. Without this the feature would be
        silently disabled even when the CLI flag is set."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        # The loop body
        self.assertIn(
            'extra_ks = tuple(int(k) for k in (getattr(args, "fast_val_k_sweep_set", ()) or ()))',
            src,
            "fast-val block missing the K-sweep loop (`extra_ks = ...`)"
        )
        # The K-override save/restore pattern
        self.assertIn("base_model._deq_k_override = k_extra", src,
            "fast-val K-sweep must set base_model._deq_k_override for each extra K"
        )
        self.assertIn("base_model._deq_k_override = default_k", src,
            "fast-val K-sweep must restore base_model._deq_k_override after the loop"
        )
        # The log emission
        self.assertIn('" fast_k_sweep:"', src,
            "fast-val block must emit `fast_k_sweep:` log field when extra Ks set"
        )

    def test_default_k_excluded_from_sweep(self):
        """If the user includes the default K in fast_val_k_sweep_set, it
        must be SKIPPED (already computed by the main fast-val call). The
        skip condition is `k_extra == default_k or k_extra <= 0`."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        self.assertIn("if k_extra == default_k or k_extra <= 0:", src,
            "fast-val K-sweep must skip default_k and non-positive K values"
        )

    def test_k_override_is_restored_via_finally(self):
        """`base_model._deq_k_override` must be restored in a `finally` block
        so an exception during run_validation does not leave the model in a
        wrong-K state for the next training step."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        # The try block setting the override
        self.assertIn(
            "base_model._deq_k_override = k_extra\n                try:",
            src,
            "fast-val K-sweep must wrap run_validation in try/finally to "
            "restore _deq_k_override on exception"
        )


if __name__ == "__main__":
    unittest.main()
