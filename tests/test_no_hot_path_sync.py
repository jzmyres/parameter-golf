"""Contract: SoftDenseRouter.forward() runs inside the RevDEQ FP iter on every
micro-step and must not call .item()/.cpu()/.tolist() — those force a host
barrier and break dynamo compilation. Enforced via AST inspection.

See CLAUDE.md §9 "Hot-path sync prohibition".
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class TestSoftDenseRouterForwardNoSync(unittest.TestCase):
    def test_forward_has_no_item_or_cpu_calls(self) -> None:
        src = (REPO_ROOT / "legacy" / "train_gpt_rich.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        forward_fn: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SoftDenseRouter":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        forward_fn = item
                        break
                break
        self.assertIsNotNone(forward_fn, "SoftDenseRouter.forward not found")

        offenders: list[str] = []
        for sub in ast.walk(forward_fn):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in ("item", "cpu", "tolist"):
                    offenders.append(f"line {sub.lineno}: .{sub.func.attr}()")
        self.assertEqual(
            offenders, [],
            f"SoftDenseRouter.forward must not call .item()/.cpu()/.tolist() — "
            f"these force GPU→host sync inside the DEQ hot path. Offenders: {offenders}"
        )


if __name__ == "__main__":
    unittest.main()
