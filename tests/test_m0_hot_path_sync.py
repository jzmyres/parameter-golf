"""Contract: the M0 grad-accum micro-loop in train_gpt.main() must not call
.item()/.cpu()/.tolist()/.numpy() — those force a GPU->host barrier on every
micro-step and break throughput / dynamo. The per-step timer and the single
``max_memory_allocated`` read for throughput/VRAM-utilization logging must live
at the STEP BOUNDARY / log site, not inside ``for micro in range(...)``.

See CLAUDE.md §9 "Hot-path sync prohibition" / EXPERIENCE.md#hot-path-sync.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _micro_loop_source() -> str:
    """The textual body of the ``for micro in range(grad_accum_steps):`` loop."""
    text = (REPO_ROOT / "train_gpt.py").read_text(encoding="utf-8")
    # The micro-loop opens at this header and the per-step accumulation ends at
    # the divide-by-grad_accum line that closes the loop scope.
    head = "for micro in range(grad_accum_steps):"
    assert head in text, "M0 grad-accum micro-loop header not found"
    body = text.split(head, 1)[1]
    return body.split("step_loss = step_loss / grad_accum_steps", 1)[0]


class TestM0MicroLoopNoSync(unittest.TestCase):
    def test_micro_loop_has_no_host_syncs(self) -> None:
        body = _micro_loop_source()
        # Parse the loop body as a statement suite (indent-normalize first).
        offenders: list[str] = []
        tree = ast.parse("if True:\n" + "\n".join(
            "    " + ln for ln in body.splitlines()))
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in ("item", "cpu", "tolist", "numpy"):
                    offenders.append(f".{sub.func.attr}()")
        self.assertEqual(
            offenders, [],
            f"M0 grad-accum micro-loop must not call .item()/.cpu()/.tolist()/"
            f".numpy() — those force a GPU->host sync per micro-step. "
            f"Offenders: {offenders}",
        )

    def test_per_step_timer_and_vram_read_are_at_step_boundary(self) -> None:
        # The throughput timer + the single max_memory_allocated read for
        # vram_util_pct must NOT live inside the micro-loop.
        body = _micro_loop_source()
        for forbidden in ("perf_counter", "max_memory_allocated", "vram_util_pct",
                          "tok_per_s"):
            self.assertNotIn(
                forbidden, body,
                f"{forbidden} must be measured at the step boundary / log site, "
                f"not inside the grad-accum micro-loop",
            )


if __name__ == "__main__":
    unittest.main()
