import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = ROOT / "train_gpt.py"
UPDATE_RESULTS = ROOT / "experiments" / "update_results.sh"


def _hot_path_body(text: str) -> str:
    body = text.split("for micro_step in range(grad_accum_steps):", 1)[1]
    return body.split("train_loss /= grad_accum_steps", 1)[0]


class TestTrainingContracts(unittest.TestCase):
    def test_default_training_budget_is_step_count_governed(self) -> None:
        """Default runs: 1000 iterations on DDP with all GPUs; wallclock off.

        Submission runs OVERRIDE via `--max-wallclock-seconds=600` (8xH100
        competition hard cap). The source-of-truth default is step-count,
        not wallclock — so Hyperparameters defaults must be:
            iterations = 1000
            max_wallclock_seconds = 0
        """
        text = TRAIN_GPT.read_text()
        # Match the Hyperparameters class attribute assignments anywhere in the file.
        self.assertRegex(text, r"\n\s*iterations\s*=\s*1000\b",
                         "Hyperparameters.iterations default must be 1000")
        self.assertRegex(text, r"\n\s*max_wallclock_seconds\s*=\s*0\b",
                         "Hyperparameters.max_wallclock_seconds default must be 0 "
                         "(disabled; step-count governs). Submission runs set 600 via CLI.")

    def test_gradient_hot_path_has_no_cpu_syncs(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        violations = re.findall(r"\.(?:item|cpu|tolist|numpy)\s*\(", body)
        self.assertEqual(violations, [])

    def test_gradient_hot_path_has_no_tensor_to_python_casts(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        casts = re.findall(r"\b(?:float|int|bool)\s*\([^)\n]*\)", body)
        offenders = [c for c in casts if re.search(r"_t\b|\.detach\b|tensor|Tensor|\.sum\b|\.mean\b|\.norm\b", c)]
        self.assertEqual(offenders, [],
                         "float()/int()/bool() on a tensor in the hot path — use torch ops.")

    def test_lyapunov_surrogate_uses_tensor_gating(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        self.assertIn("scale_t.to(dtype=", body,
                      "Lyapunov surrogate must use scale_t.to(dtype=...) tensor gating, "
                      "not a Python-scalar branch on scale_t.")
        self.assertNotRegex(body, r"\bif\s+scale_[^\n:]*:",
                            "No Python branch on scale_t / scale_val allowed in the hot path.")

    def test_update_results_log_summary_tolerates_missing_val_bpb(self) -> None:
        text = UPDATE_RESULTS.read_text()
        bpb_lines = [line for line in text.splitlines() if "val_bpb:\\K" in line]
        self.assertEqual(len(bpb_lines), 1)
        self.assertIn("|| true", bpb_lines[0])


if __name__ == "__main__":
    unittest.main()
