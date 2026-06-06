"""Pytest bootstrap: make the repository root importable for all tests.

Tests use ``from train_gpt import ...`` and ``from experiments.<mod> import ...``.
Inserting the repo root here lets those resolve regardless of how pytest is
invoked (plain ``pytest`` vs ``python -m pytest``) or where a test file lives
(``tests/`` by default, or ``legacy/tests/`` when run explicitly during the
Phase-B contract port). This is the single source for repo-root bootstrap; the
per-file ``sys.path`` shims many tests still carry are now redundant with it
(harmless) and are removed opportunistically as those files are next touched.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
