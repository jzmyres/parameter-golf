import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Hyperparameters, KShuffleBagSampler, _parse_cli_overrides


class TestDeqKJitter(unittest.TestCase):
    def test_hparams_have_k_jitter_defaults(self):
        self.assertTrue(hasattr(Hyperparameters, "deq_k_jitter"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_min"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_max"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_eval"))
        # Defaults: shuffle-bag K-jitter (deq_k_max_start/ramp removed).
        self.assertEqual(Hyperparameters.deq_k_jitter, True)
        self.assertGreaterEqual(Hyperparameters.deq_k_min, 2)
        self.assertGreaterEqual(Hyperparameters.deq_k_max, Hyperparameters.deq_k_min)
        self.assertGreater(Hyperparameters.deq_k_max, 0)
        self.assertGreater(Hyperparameters.deq_k_eval, 0)

    def test_cli_override_parses_bool(self):
        ov = _parse_cli_overrides(["--deq-k-jitter", "0"])
        self.assertEqual(ov["deq_k_jitter"], False)
        ov = _parse_cli_overrides(["--deq-k-jitter", "1"])
        self.assertEqual(ov["deq_k_jitter"], True)

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
        # After reset, next cycle draws a full bag again
        got = [s.sample() for _ in range(3)]
        self.assertEqual(set(got), set(range(2, 5)))


if __name__ == "__main__":
    unittest.main()
