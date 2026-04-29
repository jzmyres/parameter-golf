"""K-jitter / sampler tests aligned to current defaults.

User directive 2026-04-28: K-jitter is disabled, K fixed at 16. The shuffle-bag
sampler is retained for future re-enable but is short-circuited at runtime when
`deq_k_jitter=False`. Tests cover the SAMPLER mechanics (independent of the
flag) and the Hyperparameters defaults that gate it.
"""
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Hyperparameters, KShuffleBagSampler, _parse_cli_overrides


class TestDeqKJitterDefaults(unittest.TestCase):
    def test_hparams_have_k_jitter_defaults(self):
        # Required fields.
        for f in ("deq_k_jitter", "deq_k_min", "deq_k_max", "deq_k_eval", "deq_k_jitter_set"):
            self.assertTrue(hasattr(Hyperparameters, f), f"missing field: {f}")
        # Current baseline: jitter disabled, K fixed at 16 (matches deq_k_eval).
        self.assertEqual(Hyperparameters.deq_k_jitter, False)
        self.assertEqual(Hyperparameters.deq_k_max, 16)
        self.assertEqual(Hyperparameters.deq_k_eval, 16)
        self.assertEqual(Hyperparameters.deq_k_jitter_set, (16,))
        self.assertGreaterEqual(Hyperparameters.deq_k_min, 2)

    def test_bptt_k_jitter_defaults(self):
        # User directive 2026-04-28: TBPTT-k jitter also disabled, k=2 fixed.
        self.assertFalse(Hyperparameters.deq_bptt_k_jitter)
        self.assertEqual(Hyperparameters.deq_bptt_k_jitter_set, (2,))
        self.assertEqual(Hyperparameters.deq_bptt_k, 2)

    def test_cli_override_parses_bool(self):
        ov = _parse_cli_overrides(["--deq-k-jitter", "0"])
        self.assertFalse(ov["deq_k_jitter"])
        ov = _parse_cli_overrides(["--deq-k-jitter", "1"])
        self.assertTrue(ov["deq_k_jitter"])


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
