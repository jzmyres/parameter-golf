"""K-jitter / sampler tests aligned to current defaults.

Current baseline enables weighted forward-K jitter over {16, 24, 32, 64};
TBPTT-k jitter remains disabled with fixed k=3. Tests cover the sampler mechanics and the
Hyperparameters defaults that gate it.
"""
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Hyperparameters, KShuffleBagSampler, _parse_cli_overrides, _prefix_anchor_depths


class TestDeqKJitterDefaults(unittest.TestCase):
    def test_hparams_have_k_jitter_defaults(self):
        # Required fields.
        for f in ("deq_k_jitter", "deq_k_min", "deq_k_max", "deq_k_eval",
                  "deq_k_jitter_set", "deq_k_jitter_weights"):
            self.assertTrue(hasattr(Hyperparameters, f), f"missing field: {f}")
        # Current baseline: weighted forward-K jitter; eval remains K=16.
        self.assertEqual(Hyperparameters.deq_k_jitter, True)
        self.assertEqual(Hyperparameters.deq_k_max, 64)
        self.assertEqual(Hyperparameters.deq_k_eval, 16)
        self.assertEqual(Hyperparameters.deq_k_jitter_set, (16, 24, 32, 64))
        self.assertEqual(Hyperparameters.deq_k_jitter_weights, (0.50, 0.40, 0.07, 0.03))
        self.assertGreaterEqual(Hyperparameters.deq_k_min, 2)

    def test_bptt_k_jitter_defaults(self):
        # TBPTT-k jitter is disabled; fixed k=3 is the promoted default.
        self.assertFalse(Hyperparameters.deq_bptt_k_jitter)
        self.assertEqual(Hyperparameters.deq_bptt_k_jitter_set, (3,))
        self.assertEqual(Hyperparameters.deq_bptt_k, 3)

    def test_cli_override_parses_bool(self):
        ov = _parse_cli_overrides(["--deq-k-jitter", "0"])
        self.assertFalse(ov["deq_k_jitter"])
        ov = _parse_cli_overrides(["--deq-k-jitter", "1"])
        self.assertTrue(ov["deq_k_jitter"])
        ov = _parse_cli_overrides(["--deq-prefix-anchors", "1"])
        self.assertTrue(ov["deq_prefix_anchors"])

    def test_prefix_anchor_depths_are_conditional_prefixes(self):
        values = (16, 24, 32, 64)
        self.assertEqual(_prefix_anchor_depths(16, values), (16,))
        self.assertEqual(_prefix_anchor_depths(24, values), (16, 24))
        self.assertEqual(_prefix_anchor_depths(32, values), (16, 24, 32))
        self.assertEqual(_prefix_anchor_depths(64, values), (16, 24, 32, 64))
        # Non-jitter K still includes the actual sampled endpoint.
        self.assertEqual(_prefix_anchor_depths(20, values), (16, 20))


class TestKShuffleBagSampler(unittest.TestCase):
    """Sampler mechanics — independent of the deq_k_jitter flag.

    These tests document the contract the runtime relies on if jitter is ever
    re-enabled (sampler covers every K once per cycle, reset clears the bag).
    """
    def test_shuffle_bag_covers_all_k_each_cycle(self):
        import random
        rng = random.Random(123)
        s = KShuffleBagSampler(2, 12, rng)
        got1 = [s.sample() for _ in range(11)]
        self.assertEqual(set(got1), set(range(2, 13)))
        got2 = [s.sample() for _ in range(11)]
        self.assertEqual(set(got2), set(range(2, 13)))

    def test_shuffle_bag_reset_clears_bag(self):
        import random
        rng = random.Random(123)
        s = KShuffleBagSampler(2, 4, rng)
        _ = [s.sample() for _ in range(2)]  # partial drain
        s.reset()
        got = [s.sample() for _ in range(3)]
        self.assertEqual(set(got), set(range(2, 5)))


if __name__ == "__main__":
    unittest.main()
