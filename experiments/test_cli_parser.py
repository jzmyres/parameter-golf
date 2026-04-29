"""CLI parser parity + correctness tests (CPU-only).

Item 2 of the Phase 9 cleanup plan: every Hyperparameters field tunable from
the CLI must round-trip through `_parse_cli_overrides` with the documented
default. Booleans accept "0"/"1"; tuples are not yet auto-parsed (left for the
future generator-driven refactor); enums are not used in the current schema.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import Hyperparameters, _parse_cli_overrides


class TestCliParser(unittest.TestCase):
    def test_bool_parsing(self):
        for val, expected in [("0", False), ("1", True)]:
            ov = _parse_cli_overrides(["--use-ctp", val])
            self.assertEqual(ov["use_ctp"], expected)

    def test_float_parsing(self):
        ov = _parse_cli_overrides(["--cv-loss-weight", "3.5"])
        self.assertEqual(ov["cv_loss_weight"], 3.5)

    def test_int_parsing(self):
        ov = _parse_cli_overrides(["--bigram-vocab-size", "4096"])
        self.assertEqual(ov["bigram_vocab_size"], 4096)

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_cli_overrides(["--this-flag-does-not-exist", "1"])

    def test_positional_passthrough(self):
        # Wrapper / profile harnesses inject positionals; parser must not reject.
        ov = _parse_cli_overrides(["my-positional", "--cv-loss-weight", "1.5"])
        self.assertEqual(ov["cv_loss_weight"], 1.5)

    def test_no_deq_backward_field(self):
        # User directive 2026-04-28: only revdeq is supported.
        self.assertFalse(hasattr(Hyperparameters, "deq_backward"))
        with self.assertRaises(SystemExit):
            _parse_cli_overrides(["--deq-backward", "revdeq"])

    def test_default_parity(self):
        # Item 1: every documented routing knob is reachable through the CLI.
        for name, expected in [
            ("min_share_loss_weight", Hyperparameters.min_share_loss_weight),
            ("cv_loss_weight", Hyperparameters.cv_loss_weight),
            ("router_entropy_coef", Hyperparameters.router_entropy_coef),
            ("router_entropy_warmup_delay_frac", Hyperparameters.router_entropy_warmup_delay_frac),
            ("parcae_init_b_bar", Hyperparameters.parcae_init_b_bar),
        ]:
            flag = "--" + name.replace("_", "-")
            ov = _parse_cli_overrides([flag, str(expected)])
            self.assertAlmostEqual(float(ov[name]), float(expected), places=9, msg=name)


if __name__ == "__main__":
    unittest.main()
