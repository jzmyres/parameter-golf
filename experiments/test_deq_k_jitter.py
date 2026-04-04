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
        # Defaults (experiment protocol): jitter enabled, K sampled in [2,8], eval fixed.
        self.assertEqual(Hyperparameters.deq_k_jitter, True)
        self.assertEqual(Hyperparameters.deq_k_min, 2)
        self.assertEqual(Hyperparameters.deq_k_max, 8)
        self.assertEqual(Hyperparameters.deq_k_eval, 4)

    def test_cli_override_parses_bool(self):
        ov = _parse_cli_overrides(["--deq-k-jitter", "0"])
        self.assertEqual(ov["deq_k_jitter"], False)
        ov = _parse_cli_overrides(["--deq-k-jitter", "1"])
        self.assertEqual(ov["deq_k_jitter"], True)

    def test_shuffle_bag_covers_all_k_each_cycle(self):
        import random
        rng = random.Random(123)
        s = KShuffleBagSampler(2, 8, rng)
        got1 = [s.sample() for _ in range(7)]
        self.assertEqual(set(got1), set(range(2, 9)))
        got2 = [s.sample() for _ in range(7)]
        self.assertEqual(set(got2), set(range(2, 9)))


if __name__ == "__main__":
    unittest.main()
