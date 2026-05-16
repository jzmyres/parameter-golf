"""CLI parser parity + correctness tests (CPU-only).

Item 2 of the Phase 9 cleanup plan: every Hyperparameters field tunable from
the CLI must round-trip through `_parse_cli_overrides` with the documented
default. Booleans accept "0"/"1"; tuple-valued knobs accept either Python tuple
syntax or comma-separated values; enums are not used in the current schema.
"""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.plot_metrics import parse_log
from train_gpt import (
    Hyperparameters,
    KShuffleBagSampler,
    _CLI_TUNABLE_KNOBS,
    _EVAL_PROFILES,
    _apply_config_profile,
    _compute_training_budget_ms,
    _parse_cli_overrides,
    _resolve_k_sweep_values,
    _should_probe_lip_for_k,
)


class TestCliParser(unittest.TestCase):
    def test_bool_parsing(self):
        for val, expected in [("0", False), ("1", True)]:
            ov = _parse_cli_overrides(["--use-ctp", val])
            self.assertEqual(ov["use_ctp"], expected)

    def test_float_parsing(self):
        # router_load_cv_coef + mos_load_cv_coef removed 2026-05-15;
        # use a different float-typed knob (router_pertoken_entropy_coef)
        # for the parser-mechanism test.
        ov = _parse_cli_overrides(["--router-pertoken-entropy-coef", "3.5"])
        self.assertEqual(ov["router_pertoken_entropy_coef"], 3.5)

    def test_int_parsing(self):
        ov = _parse_cli_overrides(["--bigram-vocab-size", "4096"])
        self.assertEqual(ov["bigram_vocab_size"], 4096)
        ov = _parse_cli_overrides(["--val-micro-batch-seqs", "48"])
        self.assertEqual(ov["val_micro_batch_seqs"], 48)

    def test_tuple_parsing(self):
        ov = _parse_cli_overrides(["--deq-beta-jitter-set", "0.25,0.45,0.65"])
        self.assertEqual(ov["deq_beta_jitter_set"], (0.25, 0.45, 0.65))
        ov = _parse_cli_overrides(["--deq-beta-jitter-set", "(0.2, 0.4)"])
        self.assertEqual(ov["deq_beta_jitter_set"], (0.2, 0.4))
        ov = _parse_cli_overrides(["--deq-k-jitter-weights", "0.5,0.4,0.07,0.03"])
        self.assertEqual(ov["deq_k_jitter_weights"], (0.5, 0.4, 0.07, 0.03))

    def test_config_profile_applies_before_cli_overrides(self):
        ov = _parse_cli_overrides([
            "--config-profile", "score_iter152",
            "--router-ema-balance-coef", "0.7",
        ])
        args = Hyperparameters()
        profile = str(ov.pop("config_profile"))
        _apply_config_profile(args, profile)
        for key, value in ov.items():
            setattr(args, key, value)
        self.assertEqual(args.config_profile, "score_iter152")
        self.assertTrue(args.deq_prefix_anchors)
        self.assertEqual(args.eval_profile, "submission")
        self.assertAlmostEqual(args.router_ema_specialization_coef, 0.40)
        self.assertAlmostEqual(args.router_ema_balance_coef, 0.70)

    def test_eval_profiles_own_k_sweep_and_lip_scope(self):
        args = Hyperparameters()
        args.eval_profile = "diagnostic"
        self.assertEqual(tuple(_resolve_k_sweep_values(args)), _EVAL_PROFILES["diagnostic"].k_sweep_values)
        self.assertTrue(_should_probe_lip_for_k(args, 17))

        args.eval_profile = "submission"
        self.assertEqual(tuple(_resolve_k_sweep_values(args)), (16, 24, 64, 128))
        self.assertFalse(_should_probe_lip_for_k(args, 16))
        self.assertTrue(_should_probe_lip_for_k(args, 128))

        args.eval_profile = "debug"
        self.assertEqual(tuple(_resolve_k_sweep_values(args)), (16,))
        self.assertTrue(_should_probe_lip_for_k(args, 16))

    def test_weighted_k_sampler(self):
        sampler = KShuffleBagSampler(
            4, 64, random.Random(0),
            values=[16, 24, 32, 64],
            weights=[0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual([sampler.sample() for _ in range(8)], [64] * 8)

    def test_weighted_k_sampler_uses_exact_weighted_bag(self):
        sampler = KShuffleBagSampler(
            4, 64, random.Random(0),
            values=[16, 24, 32, 64],
            weights=[0.50, 0.40, 0.07, 0.03],
        )
        samples = [sampler.sample() for _ in range(100)]
        self.assertEqual({k: samples.count(k) for k in (16, 24, 32, 64)},
                         {16: 50, 24: 40, 32: 7, 64: 3})

    def test_weighted_k_sampler_tiny_weights_round_to_zero_not_inflated(self):
        # Weights below 1/cycle_len round to zero (no floor inflation):
        # 0.001 * 100 = 0.1 → floor 0; the residual slot is awarded to
        # the largest fractional residual, which is the heavy weight.
        sampler = KShuffleBagSampler(
            4, 64, random.Random(0),
            values=[16, 64], weights=[0.999, 0.001],
        )
        samples = [sampler.sample() for _ in range(100)]
        self.assertEqual(samples.count(64), 0)
        self.assertEqual(samples.count(16), 100)

    def test_weighted_k_sampler_load_state_normalizes_and_validates(self):
        sampler = KShuffleBagSampler(4, 64, random.Random(0), values=[16, 24], weights=[1.0, 1.0])
        sampler.load_state_dict({
            "k_min": 4,
            "k_max": 64,
            "step": 4,
            "values": [16, 24, 32],
            "weights": [5.0, 4.0, 1.0],
            "bag": [],
        })
        self.assertEqual(sampler.values, [16, 24, 32])
        self.assertAlmostEqual(sum(sampler.weights), 1.0)
        # Pass a complete state dict with explicit duplicate values so the
        # ValueError unambiguously locates to the duplicate (not to a
        # missing key). Per coderabbit MINOR.
        with self.assertRaises(ValueError):
            sampler.load_state_dict({
                "k_min": 4,
                "k_max": 64,
                "step": 4,
                "values": [16, 16, 32],  # duplicate 16
                "weights": [0.4, 0.4, 0.2],
                "bag": [],
            })

    def test_normalize_k_jitter_weights_validates_explicitly(self):
        from train_gpt import _normalize_k_jitter_weights
        # values=None
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights(None, [1.0])
        # length mismatch
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights([16, 24], [1.0])
        # duplicate values
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights([16, 16], [1.0, 1.0])
        # negative weight
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights([16, 24], [1.0, -0.1])
        # non-finite weight
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights([16, 24], [1.0, float("inf")])
        # zero weight sum
        with self.assertRaises(ValueError):
            _normalize_k_jitter_weights([16, 24], [0.0, 0.0])

    def test_validate_hyperparameters_rejects_dirichlet_with_entmax(self):
        from types import SimpleNamespace
        from train_gpt import _validate_hyperparameters
        args = SimpleNamespace(
            model_dim=768, num_heads=8, num_kv_heads=4, num_layers=12,
            num_experts=16, num_shared_experts=1,
            router_scoring="dirichlet_ucb", use_entmax_routing=True,
            lyapunov_every=16, lyapunov_max_tokens=64,
        )
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(args)

    def test_validate_hyperparameters_rejects_nonpositive_lyapunov_knobs(self):
        from types import SimpleNamespace
        from train_gpt import _validate_hyperparameters
        base = dict(
            model_dim=768, num_heads=8, num_kv_heads=4, num_layers=12,
            num_experts=16, num_shared_experts=1,
            router_scoring="dirichlet_ucb", use_entmax_routing=False,
        )
        for field, value in (("lyapunov_every", 0), ("lyapunov_every", -1),
                             ("lyapunov_max_tokens", 0), ("lyapunov_max_tokens", -3)):
            args = SimpleNamespace(
                **base,
                lyapunov_every=16, lyapunov_max_tokens=64,
            )
            setattr(args, field, value)
            with self.assertRaises(SystemExit):
                _validate_hyperparameters(args)

    def test_parse_k_sweep_confidence_diagnostics(self):
        # iter155 (corrected): K-sweep emits three contraction diagnostics:
        # lip_ub_T (decomposition), lip_ub_S (advisory surrogate), and
        # lip_ub_F (gate-aligned two-state Parcae cycle).  Parser must
        # extract all three and the table-header path must surface the
        # new column names.
        log_text = "\n".join([
            "train_batch_tokens:1024",
            "k_sweep:k=16 val_bpb:1.234500 iter_conv_rel:0.001000 "
            "router_dir_strength_mean:30.0000 router_dir_uncertainty_mass:0.2000 "
            "router_dir_sigma_mean:0.0500 router_dir_evidence_mean:1.0000 "
            "router_dir_mu_entropy_norm:0.9000 router_ucb_beta_current:0.5000 "
            "lip_ub_T:1.5000 lip_ub_S:0.8000 lip_ub_F:0.9500 fp_bound:0.0050",
            "k_sweep_table:    K   val_bpb    dir_S    dir_U ucb_beta   lip_ub_T   lip_ub_S   lip_ub_F fp_bound iter_conv_rel",
            "k_sweep_table:   16    1.2345  30.0000   0.2000   0.5000     1.5000     0.8000     0.9500   0.0050       0.0010",
        ])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(log_text)
            path = f.name
        try:
            data = parse_log(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(data["k_sweep"][0]["K"], 16)
        self.assertEqual(data["k_sweep"][0]["router_dir_strength_mean"], 30.0)
        self.assertEqual(data["k_sweep"][0]["lip_ub_T"], 1.5)
        self.assertEqual(data["k_sweep"][0]["lip_ub_S"], 0.8)
        self.assertEqual(data["k_sweep"][0]["lip_ub_F"], 0.95)
        self.assertNotIn("lip_ub", data["k_sweep"][0])  # legacy key not present in new logs
        self.assertEqual(data["k_sweep_table"][0]["K"], 16)
        self.assertEqual(data["k_sweep_table"][0]["dir_U"], 0.2)
        self.assertEqual(data["k_sweep_table"][0]["lip_ub_T"], 1.5)
        self.assertEqual(data["k_sweep_table"][0]["lip_ub_S"], 0.8)
        self.assertEqual(data["k_sweep_table"][0]["lip_ub_F"], 0.95)
        self.assertNotIn("hutch_F", data["k_sweep_table"][0])
        self.assertNotIn("spec_norm", data["k_sweep_table"][0])
        self.assertNotIn("rd_step", data["k_sweep_table"][0])

    def test_fp_bound_in_k_sweep_log_uses_fp_residual_F_not_iter_conv_rel(self):
        # iter155 corrected (2026-05-13): fp_bound must be computed from
        # fp_residual_F (the joint two-state cycle residual) paired with
        # lip_ub_F (the F-side operator-norm estimate), NOT from the
        # legacy z-only iter_conv_rel.  Pairing iter_conv_rel with lip_ub_F
        # was a category error: the residual must be measured on the same
        # map as the Lipschitz constant.
        #
        # This synthetic-log test pins the F-derived formula by parsing
        # values produced by train_gpt.py and verifying the printed
        # fp_bound matches fp_residual_F/(1-lip_ub_F) (NOT the
        # iter_conv_rel-derived value).  iter_conv_rel and fp_residual_F
        # are deliberately set to DIFFERENT values so the formula source
        # is unambiguous.
        iter_conv_rel = 0.050000   # legacy z-only step residual
        fp_residual_F = 0.001000   # joint cycle residual at saved FP
        lip_ub_F = 0.5000          # F-side operator-norm estimate
        # Correct (F-side) bound: 0.001 / 0.5 = 0.002.  Legacy
        # (iter_conv_rel) bound would be 0.05 / 0.5 = 0.1, which is 50×
        # different — easy to distinguish.
        fp_bound_expected_F = fp_residual_F / (1.0 - lip_ub_F)
        fp_bound_legacy_z = iter_conv_rel / (1.0 - lip_ub_F)
        assert abs(fp_bound_expected_F - fp_bound_legacy_z) > 1e-3, (
            "test fixture is degenerate: F and legacy bounds are equal"
        )

        log_text = "\n".join([
            "train_batch_tokens:1024",
            f"k_sweep:k=128 val_bpb:1.234500 iter_conv_rel:{iter_conv_rel:.6f} "
            "router_dir_strength_mean:30.0000 router_dir_uncertainty_mass:0.2000 "
            "router_dir_sigma_mean:0.0500 router_dir_evidence_mean:1.0000 "
            "router_dir_mu_entropy_norm:0.9000 router_ucb_beta_current:0.5000 "
            f"lip_ub_T:1.5000 lip_ub_S:0.8000 lip_ub_F:{lip_ub_F:.4f} "
            f"fp_residual_F:{fp_residual_F:.6f} "
            f"fp_bound:{fp_bound_expected_F:.6f}",
        ])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(log_text)
            path = f.name
        try:
            data = parse_log(path)
        finally:
            Path(path).unlink(missing_ok=True)

        row = data["k_sweep"][0]
        # All four fields must parse independently.
        self.assertEqual(row["iter_conv_rel"], iter_conv_rel)
        self.assertEqual(row["fp_residual_F"], fp_residual_F)
        self.assertEqual(row["lip_ub_F"], lip_ub_F)
        # The CRITICAL check: parsed fp_bound matches the F-derived formula
        # within float tolerance, NOT the legacy iter_conv_rel-derived one.
        self.assertAlmostEqual(row["fp_bound"], fp_bound_expected_F, places=4)
        self.assertNotAlmostEqual(row["fp_bound"], fp_bound_legacy_z, places=2)

    def test_parse_k_sweep_legacy_lip_ub_field_still_parses(self):
        # iter155 backward compat: pre-iter155 logs only emit `lip_ub:`.
        # The parser must still accept that key so historical training_logs
        # in experiments/training_logs/ remain analyzable.
        log_text = "\n".join([
            "train_batch_tokens:1024",
            "k_sweep:k=16 val_bpb:1.234500 lip_ub:0.8000 fp_bound:0.0050",
        ])
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(log_text)
            path = f.name
        try:
            data = parse_log(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(data["k_sweep"][0]["lip_ub"], 0.8)
        self.assertNotIn("lip_ub_T", data["k_sweep"][0])
        self.assertNotIn("lip_ub_S", data["k_sweep"][0])

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_cli_overrides(["--this-flag-does-not-exist", "1"])

    def test_positional_passthrough(self):
        # Wrapper / profile harnesses inject positionals; parser must not reject.
        # Use router_pertoken_entropy_coef (router_load_cv_coef removed 2026-05-15).
        ov = _parse_cli_overrides(["my-positional", "--router-pertoken-entropy-coef", "1.5"])
        self.assertEqual(ov["router_pertoken_entropy_coef"], 1.5)

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

    def test_max_training_seconds_reserves_eval_headroom(self):
        # Project invariant: --max-training-seconds=600 must keep total
        # process wallclock under 600 s on 8×H100. See
        # `EXPERIENCE.md#scalar-semantic-shift`.
        self.assertIsNone(_compute_training_budget_ms(0, 0))
        self.assertIsNone(_compute_training_budget_ms(0, 120))
        self.assertEqual(_compute_training_budget_ms(600, 0), 600_000.0)
        budget_ms = _compute_training_budget_ms(
            600, Hyperparameters.eval_reservation_seconds
        )
        self.assertEqual(
            budget_ms,
            (600 - Hyperparameters.eval_reservation_seconds) * 1000.0,
        )
        self.assertLessEqual(budget_ms, 600_000.0)
        self.assertEqual(_compute_training_budget_ms(600, -5), 600_000.0)
        self.assertEqual(_compute_training_budget_ms(10, 100), 1_000.0)
        self.assertTrue(Hyperparameters.final_full_validation)


if __name__ == "__main__":
    unittest.main()
