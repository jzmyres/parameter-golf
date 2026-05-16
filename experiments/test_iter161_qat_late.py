"""iter161-QAT-late tests (2026-05-16): deterministic STE int6-SDCLIP
fake-quant on CastedLinear weights for the last ``(total_steps -
qat_late_start_step)`` training steps.

Three-test framework verification:
- Principled: matches the artifact codec EXACTLY (same per-row SDCLIP
  scaling as encode_scored_artifact's quantize_int6_sdclip); the model
  adapts to the actual deployment quantization, not a proxy.
- Simplest: one STE module + one global step counter; no SVD machinery,
  no per-layer calibration, no two-mechanism bundle.
- General: works for any nn.Linear-shaped weight.

RevDEQ-safety: forward is DETERMINISTIC (only weight values matter, no
RNG, no random noise) — distinct from iter20's stochastic noise injection
(refuted 2026-04 because random per-call noise broke reverse reconstruction).
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (
    CastedLinear,
    Hyperparameters,
    SDCLIP_K_MATRIX,
    _FakeQuantInt6SDClipSTE,
    _QAT_LATE_STATE,
    _validate_hyperparameters,
    quantize_int6_sdclip,
)
from tests._helpers import mutate_hyperparameters as _mut


class TestFakeQuantSTE(unittest.TestCase):
    """The STE function must (a) produce the SAME dequantized output as the
    artifact codec, (b) pass gradient through identity, (c) be deterministic."""

    def setUp(self):
        torch.manual_seed(0)
        self.W = torch.randn(64, 128, requires_grad=True)

    def test_forward_matches_artifact_codec_exactly(self):
        """STE forward must give the SAME dequantized output as the artifact
        codec's quantize_int6_sdclip — this is the most-principled property,
        the model trains against the EXACT deployment quantization."""
        q_int, scale = quantize_int6_sdclip(self.W.detach(), k=SDCLIP_K_MATRIX)
        # Dequantize manually: (q_int * scale)
        # scale is per-row (shape [rows]), q_int is [rows, cols]
        scale_expand = scale.float().view(-1, 1)
        deq_artifact = (q_int.float() * scale_expand).to(self.W.dtype)

        deq_ste = _FakeQuantInt6SDClipSTE.apply(self.W.detach(), SDCLIP_K_MATRIX)
        # Allow tiny floating-point diff (both paths use fp32 intermediates).
        diff = (deq_ste - deq_artifact).abs().max().item()
        self.assertLess(diff, 1e-5,
            f"STE forward {diff:.2e} differs from artifact codec — must match "
            "EXACTLY for the model to train against deployment quantization")

    def test_backward_is_identity(self):
        """STE backward: gradient passes through unchanged. This is what makes
        QAT trainable — round() has zero gradient everywhere, so without STE
        the loss would have no gradient on the weight."""
        deq = _FakeQuantInt6SDClipSTE.apply(self.W, SDCLIP_K_MATRIX)
        loss = deq.sum()
        loss.backward()
        # dL/dW = identity matrix * ones (loss is sum) → all 1s.
        self.assertTrue(torch.allclose(self.W.grad, torch.ones_like(self.W)),
            "STE backward must pass gradient through as identity")

    def test_deterministic_forward(self):
        """RevDEQ-safe: same input → same output. Distinct from iter20's
        stochastic noise injection which made reverse reconstruction fail."""
        W = torch.randn(32, 64)
        out1 = _FakeQuantInt6SDClipSTE.apply(W, SDCLIP_K_MATRIX)
        out2 = _FakeQuantInt6SDClipSTE.apply(W, SDCLIP_K_MATRIX)
        self.assertTrue(torch.equal(out1, out2),
            "STE forward must be deterministic; otherwise RevDEQ reverse "
            "reconstruction breaks (iter20 random-noise refutation)")

    def test_small_weight_passthrough(self):
        """Weights with ndim < 2 (1D biases, scalars) get identity — the
        artifact codec doesn't quantize them either (numel <= 8192 threshold
        + per-row scale needs 2D)."""
        W_small = torch.randn(50)  # 1D
        out = _FakeQuantInt6SDClipSTE.apply(W_small, SDCLIP_K_MATRIX)
        self.assertTrue(torch.equal(out, W_small),
            "1D weights must pass through identity (no per-row quant possible)")


class TestQATLateState(unittest.TestCase):
    """The global QAT state controls when CastedLinear applies STE in forward."""

    def setUp(self):
        _QAT_LATE_STATE.reset()

    def tearDown(self):
        _QAT_LATE_STATE.reset()

    def test_default_is_inactive(self):
        """Default state must be OFF — start_step=-1, never activates."""
        self.assertFalse(_QAT_LATE_STATE.active())
        _QAT_LATE_STATE.set_step(1000)
        self.assertFalse(_QAT_LATE_STATE.active(),
            "start_step=-1 (default) must keep QAT inactive at any step")

    def test_activates_at_configured_step(self):
        _QAT_LATE_STATE.configure(start_step=800, sdclip_k=SDCLIP_K_MATRIX)
        _QAT_LATE_STATE.set_step(799)
        self.assertFalse(_QAT_LATE_STATE.active(),
            "QAT must not activate before start_step")
        _QAT_LATE_STATE.set_step(800)
        self.assertTrue(_QAT_LATE_STATE.active(),
            "QAT must activate at exactly start_step")
        _QAT_LATE_STATE.set_step(1000)
        self.assertTrue(_QAT_LATE_STATE.active(),
            "QAT must stay active after start_step")


class TestCastedLinearQATIntegration(unittest.TestCase):
    """When QAT is active, CastedLinear.forward must apply STE; when OFF, it
    must produce the un-quantized baseline output."""

    def setUp(self):
        _QAT_LATE_STATE.reset()
        torch.manual_seed(0)
        # Use 65×128 = 8320 > 8192 threshold so STE fires (matches the
        # CastedLinear.forward gate `w.numel() > 8192` which mirrors the
        # artifact codec's per-row quantization minimum size).
        self.layer = CastedLinear(in_features=65, out_features=128, bias=True)
        self.x = torch.randn(4, 32, 65)

    def tearDown(self):
        _QAT_LATE_STATE.reset()

    def test_qat_off_matches_plain_linear(self):
        """Default (QAT OFF): output equals plain F.linear(x, W, b)."""
        with torch.no_grad():
            baseline = torch.nn.functional.linear(self.x, self.layer.weight, self.layer.bias)
            ste_out = self.layer(self.x)
        self.assertTrue(torch.allclose(baseline, ste_out, atol=1e-6),
            "QAT OFF: CastedLinear output must match plain Linear")

    def test_qat_on_changes_output_observably(self):
        """When QAT activates, the output MUST differ from the un-quantized
        baseline (otherwise the flag has no effect — Flag-to-effect contract)."""
        with torch.no_grad():
            baseline = torch.nn.functional.linear(self.x, self.layer.weight, self.layer.bias)

        _QAT_LATE_STATE.configure(start_step=0, sdclip_k=SDCLIP_K_MATRIX)
        _QAT_LATE_STATE.set_step(0)
        with torch.no_grad():
            qat_out = self.layer(self.x)

        diff = (baseline - qat_out).abs().max().item()
        self.assertGreater(diff, 1e-4,
            "QAT ON: output must differ from baseline (Flag-to-effect contract); "
            f"got diff={diff:.2e} which is below the int6-quant noise floor")


class TestHyperparameterValidator(unittest.TestCase):
    """Validator must reject malformed qat_late_start_step values."""

    def test_default_passes(self):
        _validate_hyperparameters(Hyperparameters())  # qat_late_start_step=-1

    def test_negative_other_than_minus_one_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(qat_late_start_step=-5))
        self.assertIn("qat_late_start_step", str(ctx.exception))

    def test_exceeding_iterations_rejected(self):
        """start_step > iterations would never activate — fail fast."""
        with self.assertRaises(SystemExit) as ctx:
            _validate_hyperparameters(_mut(qat_late_start_step=2000, iterations=1000))
        self.assertIn("qat_late_start_step", str(ctx.exception))
        self.assertIn("exceeds total iterations", str(ctx.exception))

    def test_valid_zero_passes(self):
        """start_step=0 (QAT from step 0) is a valid edge case."""
        _validate_hyperparameters(_mut(qat_late_start_step=0))

    def test_valid_late_start_passes(self):
        _validate_hyperparameters(_mut(qat_late_start_step=800))


class TestQATPropagationToTrainGpt(unittest.TestCase):
    """Negative assertion: ensure the iter20 anti-pattern (stochastic
    quant-noise injection) does NOT reappear in train_gpt.py — iter161-QAT
    is deterministic by design (forward depends only on weight values)."""

    def test_no_random_quant_noise_in_train_gpt(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        # iter20 used `torch.randn_like(weight) * noise_rate`. The deterministic
        # iter161-QAT must not introduce per-step randomness on weights.
        # Allow the phrase in comments (e.g., docstring referencing iter20).
        # Heuristic: no `torch.randn` followed by weight scaling in CastedLinear.
        cl_block_start = src.index("class CastedLinear")
        cl_block_end = src.index("\nclass ", cl_block_start)
        cl_block = src[cl_block_start:cl_block_end]
        self.assertNotIn("torch.randn", cl_block,
            "CastedLinear must not use torch.randn — iter161-QAT is "
            "deterministic; random per-call noise breaks RevDEQ reverse "
            "reconstruction (iter20 refutation 2026-04)")

    def test_fake_quant_ste_is_registered_in_casted_linear(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(repo_root, "train_gpt.py"), "r").read()
        self.assertIn("_FakeQuantInt6SDClipSTE.apply(w", src,
            "CastedLinear.forward must call _FakeQuantInt6SDClipSTE.apply "
            "for the QAT-late training-time effect to fire.")
        self.assertIn("_QAT_LATE_STATE.set_step(next_step)", src,
            "Training loop must call _QAT_LATE_STATE.set_step each iteration "
            "so CastedLinear's active() check responds to step progression.")


if __name__ == "__main__":
    unittest.main()
