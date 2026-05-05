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

from train_gpt import Hyperparameters, _CLI_TUNABLE_KNOBS, _parse_cli_overrides


class TestCliParser(unittest.TestCase):
    def test_bool_parsing(self):
        for val, expected in [("0", False), ("1", True)]:
            ov = _parse_cli_overrides(["--use-ctp", val])
            self.assertEqual(ov["use_ctp"], expected)

    def test_float_parsing(self):
        ov = _parse_cli_overrides(["--router-load-cv-coef", "3.5"])
        self.assertEqual(ov["router_load_cv_coef"], 3.5)

    def test_int_parsing(self):
        ov = _parse_cli_overrides(["--bigram-vocab-size", "4096"])
        self.assertEqual(ov["bigram_vocab_size"], 4096)

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_cli_overrides(["--this-flag-does-not-exist", "1"])

    def test_positional_passthrough(self):
        # Wrapper / profile harnesses inject positionals; parser must not reject.
        ov = _parse_cli_overrides(["my-positional", "--router-load-cv-coef", "1.5"])
        self.assertEqual(ov["router_load_cv_coef"], 1.5)

    def test_no_deq_backward_field(self):
        # User directive 2026-04-28: only revdeq is supported.
        self.assertFalse(hasattr(Hyperparameters, "deq_backward"))
        with self.assertRaises(SystemExit):
            _parse_cli_overrides(["--deq-backward", "revdeq"])

    def test_default_parity(self):
        # Hyperparameter fan-out invariant: every CLI-tunable knob (per the
        # _CLI_TUNABLE_KNOBS single source of truth) must (a) name a real
        # Hyperparameters field and (b) round-trip its default through the
        # parser. This catches knob drift the moment a new entry is added
        # to _CLI_TUNABLE_KNOBS without a matching Hyperparameters field.
        skip = {"data-path", "tokenizer-path", "run-id"}  # path strings — separate validation
        for cli_name in _CLI_TUNABLE_KNOBS:
            if cli_name in skip:
                continue
            py_name = cli_name.replace("-", "_")
            self.assertTrue(
                hasattr(Hyperparameters, py_name),
                f"_CLI_TUNABLE_KNOBS entry {cli_name!r} has no Hyperparameters field {py_name!r}",
            )
            expected = getattr(Hyperparameters, py_name)
            if isinstance(expected, bool):
                continue  # booleans go through the int-flag path tested separately
            flag = "--" + cli_name
            ov = _parse_cli_overrides([flag, str(expected)])
            got = ov[py_name]
            if isinstance(expected, float):
                self.assertAlmostEqual(float(got), float(expected), places=9, msg=cli_name)
            else:
                self.assertEqual(type(expected)(got), expected, msg=cli_name)


if __name__ == "__main__":
    unittest.main()
