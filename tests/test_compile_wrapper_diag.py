"""Regression test for `_unwrap_compiled_module` (review item 8).

`torch.compile` wraps a module in an `OptimizedModule` whose attribute writes
may or may not propagate to the original module.  Code that toggles
diagnostic flags (e.g., `_diag_track_enabled`) must always land on the underlying
module so log-site readers can observe them.  This test keeps that
contract honest in both eager and compiled modes.
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import _unwrap_compiled_module  # noqa: E402


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._diag_flag: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class TestUnwrapCompiledModule(unittest.TestCase):
    def test_identity_for_eager_module(self):
        m = _TinyModule()
        self.assertIs(_unwrap_compiled_module(m), m)

    def test_unwraps_compiled_module(self):
        m = _TinyModule()
        try:
            mc = torch.compile(m)
        except Exception as exc:
            self.skipTest(f"torch.compile unavailable: {exc}")
        # We expect the compile wrapper to expose `_orig_mod` pointing at `m`.
        self.assertIs(_unwrap_compiled_module(mc), m)

    def test_diagnostic_flag_writes_land_on_original(self):
        """Attribute writes through the unwrap must be observable by readers
        of the original module instance — mirrors how gate trackers work."""
        m = _TinyModule()
        try:
            mc = torch.compile(m)
        except Exception as exc:
            self.skipTest(f"torch.compile unavailable: {exc}")
        underlying = _unwrap_compiled_module(mc)
        underlying._diag_flag = True
        self.assertTrue(m._diag_flag)


if __name__ == "__main__":
    unittest.main()
