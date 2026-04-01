import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import Hyperparameters, _parse_cli_overrides


class TestDeqKJitter(unittest.TestCase):
    def test_hparams_have_k_jitter_defaults(self):
        self.assertTrue(hasattr(Hyperparameters, "deq_k_jitter"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_min"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_max"))
        self.assertTrue(hasattr(Hyperparameters, "deq_k_eval"))

    def test_cli_override_parses_bool(self):
        ov = _parse_cli_overrides(["--deq-k-jitter", "0"])
        self.assertEqual(ov["deq_k_jitter"], False)
        ov = _parse_cli_overrides(["--deq-k-jitter", "1"])
        self.assertEqual(ov["deq_k_jitter"], True)


if __name__ == "__main__":
    unittest.main()

