"""Startup invariant validation: head-dim divisibility, GQA ratio, expert
counts, batch divisibility. Live default Hyperparameters() must pass."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests._helpers import mutate_hyperparameters as _mut  # noqa: E402
from train_gpt import (  # noqa: E402
    Hyperparameters,
    _CONFIG_PROFILES,
    _HYPERPARAMETER_FIELDS,
    _apply_config_profile,
    _assert_known_hyperparameter,
    _compute_run_status,
    _validate_hyperparameters,
)


class TestValidateHyperparameters(unittest.TestCase):
    def test_default_config_passes(self) -> None:
        # Sanity: the live default config must not be rejected by its own validator.
        _validate_hyperparameters(Hyperparameters())

    def test_profile_and_policy_names_are_validated(self) -> None:
        for field, value in (
            ("config_profile", "unknown"),
            ("eval_profile", "unknown"),
            ("diagnostic_gate_policy", "strict"),
        ):
            with self.assertRaises(SystemExit) as ctx:
                _validate_hyperparameters(_mut(**{field: value}))
            self.assertIn(field, str(ctx.exception))

    def test_lyapunov_target_enum_is_validated(self) -> None:
        # iter155 (corrected): `lyapunov_target` must be one of
        # {"transition_T", "iteration_S", "iteration_F"}.  An invalid value
        # would silently fall through to the T-side branch in the FD probe,
        # defeating the iter155 hypothesis, so the validator must hard-reject
        # at startup.  iteration_F is the gate-aligned target on the actual
        # two-state Parcae cycle.
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(lyapunov_target="iteration_X"))
        self.assertIn("lyapunov_target", str(ctx.exception))
        # All three valid values must pass.
        for valid in ("transition_T", "iteration_S", "iteration_F"):
            _validate_hyperparameters(_mut(lyapunov_target=valid))

    def test_lyapunov_estimator_enum_is_validated(self) -> None:
        # `lyapunov_estimator` is now random_fd only — `power_jvp_F` was
        # removed 2026-05-15 alongside the lip_ub_* operator-norm probes
        # (both refuted by iter155/iter152 evidence; CLAUDE.md most-
        # principled-simplest-general directive). Validator must reject
        # both unknown values AND the removed `power_jvp_F` legacy value.
        for invalid in ("hutch_xyz", "power_jvp_F"):
            with self.assertRaises(SystemExit) as ctx:
                _validate_hyperparameters(_mut(lyapunov_estimator=invalid))
            self.assertIn("lyapunov_estimator", str(ctx.exception))
        _validate_hyperparameters(_mut(lyapunov_estimator="random_fd"))

    def test_unknown_profile_key_is_rejected(self) -> None:
        # Defends `_CONFIG_PROFILES` against silent typos: a stray attribute on
        # `args` would let the run proceed with the real Hyperparameter at its
        # default while the banner falsely advertised the chosen profile.
        _CONFIG_PROFILES["test_profile_with_typo"] = {"router_ema_balanced_coef": 0.5}
        try:
            with self.assertRaises(SystemExit) as ctx:
                _apply_config_profile(Hyperparameters(), "test_profile_with_typo")
            self.assertIn("router_ema_balanced_coef", str(ctx.exception))
            self.assertIn("unknown Hyperparameter", str(ctx.exception))
        finally:
            _CONFIG_PROFILES.pop("test_profile_with_typo", None)

    def test_unknown_cli_override_key_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _assert_known_hyperparameter("router_ema_balanced_coef", source="cli_override")
        self.assertIn("router_ema_balanced_coef", str(ctx.exception))
        self.assertIn("cli_override", str(ctx.exception))

    def test_diagnostic_gate_policy_hard_flips_run_valid(self) -> None:
        # Truth table over (policy × health_valid × full_val_completed). Pins the
        # scalar-semantic shift introduced when `diagnostic_gate_policy=hard` was
        # added: previously `run_valid` followed full_validation completion only.
        cases = [
            # (policy, health, full_val) -> (run_valid, status, reason)
            ("advisory", True,  True,  (True,  "validated_clean",          None)),
            ("advisory", False, True,  (True,  "validated_with_tech_debt", None)),
            ("hard",     True,  True,  (True,  "validated_clean",          None)),
            ("hard",     False, True,  (False, "health_gate_failed",       "diagnostic_gate_policy_hard")),
            ("advisory", True,  False, (False, "validated_fast_only",      "final_full_validation_disabled")),
            ("hard",     False, False, (False, "validated_fast_only",      "final_full_validation_disabled")),
        ]
        for policy, health, full_val, expected in cases:
            got = _compute_run_status(
                diagnostic_policy=policy,
                health_valid=health,
                full_val_completed=full_val,
            )
            self.assertEqual(
                got, expected,
                f"policy={policy} health={health} full_val={full_val}: got {got}, want {expected}",
            )

    def test_hyperparameter_fields_covers_score_iter152_profile(self) -> None:
        # The whitelist must include every key referenced by any builtin profile,
        # otherwise the new schema check rejects the live default launch path.
        for profile_name, profile_dict in _CONFIG_PROFILES.items():
            for key in profile_dict:
                self.assertIn(
                    key, _HYPERPARAMETER_FIELDS,
                    f"profile {profile_name!r} references unknown Hyperparameter {key!r}",
                )

    def test_model_dim_not_divisible_by_num_heads(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(num_heads=7))  # 768 / 7 → not integer
        self.assertIn("model_dim", str(ctx.exception))
        self.assertIn("num_heads", str(ctx.exception))

    def test_num_heads_not_divisible_by_num_kv_heads(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(num_heads=8, num_kv_heads=3))
        self.assertIn("num_kv_heads", str(ctx.exception))
        self.assertIn("GQA", str(ctx.exception))

    def test_num_layers_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(num_layers=0))

    def test_num_experts_negative_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(num_experts=-1))

    def test_num_shared_experts_negative_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(num_shared_experts=-1))

    def test_zero_total_experts_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(num_experts=0, num_shared_experts=0))
        self.assertIn("num_experts", str(ctx.exception))

    def test_num_shared_experts_cannot_exceed_total_experts(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(num_experts=1, num_shared_experts=2))
        self.assertIn("num_shared_experts", str(ctx.exception))
        self.assertIn("num_experts", str(ctx.exception))
        self.assertIn("routed expert", str(ctx.exception))

    def test_at_least_one_routed_expert_required(self) -> None:
        # nS == nE is the boundary case: zero routed experts after shared.
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(num_experts=2, num_shared_experts=2))
        self.assertIn("routed expert", str(ctx.exception))

    def test_sparse_dispatch_rejected_for_training(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_sparse_dispatch=True))
        self.assertIn("use_sparse_dispatch", str(ctx.exception))
        self.assertIn("RevDEQ-safe", str(ctx.exception))

    def test_prefix_anchors_reject_refinements(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(deq_prefix_anchors=True, num_refinements=1))
        self.assertIn("deq_prefix_anchors", str(ctx.exception))
        self.assertIn("num_refinements=0", str(ctx.exception))

    def test_rr_attention_rejects_nsa_combo(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_rr_attention=True, use_nsa_attention=True))
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_rr_attention_grid_must_divide_sequence(self) -> None:
        # Pin train_seq_len=256 so the T<=512 cap (added by the
        # flag-to-effect contract) cannot intercept these tests — they target
        # the divisibility checks specifically and must remain order-robust.
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_rr_attention=True, rr_stride=7, train_seq_len=256))
        self.assertIn("rr_block_size", str(ctx.exception))
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(
                _mut(use_rr_attention=True, rr_stride=8, rr_block_size=96, train_seq_len=256)
            )
        self.assertIn("rr_block_size", str(ctx.exception))

    def test_component_windows_must_fit_model_dim(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(smear_gate_window=2048))
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(sparse_attn_gate_window=2048))

    # Flag-to-effect contract gates (iter152 review): each of these flags
    # advertises behaviour the current code does not implement, so the
    # validator must reject them to keep the hypothesis-log honest.
    def test_rr_attention_rejects_T_above_silent_fallback_cap(self) -> None:
        # At default train_seq_len=2048 the rr_attention module returns dense
        # SDPA silently; the validator should refuse the no-op combination.
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_rr_attention=True))
        self.assertIn("use_rr_attention", str(ctx.exception))
        self.assertIn("dense SDPA", str(ctx.exception))

    def test_use_gptq_is_scaffold_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_gptq=True))
        self.assertIn("use_gptq", str(ctx.exception))
        self.assertIn("scaffold", str(ctx.exception))

    def test_use_lqer_is_scaffold_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_lqer=True))
        self.assertIn("use_lqer", str(ctx.exception))
        self.assertIn("scaffold", str(ctx.exception))

    def test_use_ttt_eval_rejected_no_consumer(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_ttt_eval=True))
        self.assertIn("use_ttt_eval", str(ctx.exception))
        self.assertIn("consumer", str(ctx.exception))

    def test_use_caseops_rejected_fixture_only(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(use_caseops=True))
        self.assertIn("use_caseops", str(ctx.exception))
        self.assertIn("fixture", str(ctx.exception))

    def test_train_batch_tokens_not_divisible_by_seq_len(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(train_batch_tokens=524288 + 7))
        self.assertIn("train_batch_tokens", str(ctx.exception))
        self.assertIn("train_seq_len", str(ctx.exception))

    def test_grad_accum_multiplier_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_hyperparameters(_mut(grad_accum_multiplier=0))


if __name__ == "__main__":
    unittest.main()
