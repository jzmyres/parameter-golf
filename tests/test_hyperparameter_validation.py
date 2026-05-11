"""Startup invariant validation: head-dim divisibility, GQA ratio, expert
counts, batch divisibility. Live default Hyperparameters() must pass."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_gpt import Hyperparameters, _validate_hyperparameters


def _mut(**overrides) -> Hyperparameters:
    h = Hyperparameters()
    for k, v in overrides.items():
        setattr(h, k, v)
    return h


class TestValidateHyperparameters(unittest.TestCase):
    def test_default_config_passes(self) -> None:
        # Sanity: the live default config must not be rejected by its own validator.
        _validate_hyperparameters(Hyperparameters())

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
