"""Shared test helpers — keep DRY across tests/ and experiments/test_*.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_gpt import Hyperparameters


def mutate_hyperparameters(**overrides) -> Hyperparameters:
    """Return a Hyperparameters with the given field overrides applied. Used
    by validator tests and the flag-to-effect contract gate to construct
    perturbed configs without retyping the default-init dance."""
    h = Hyperparameters()
    for k, v in overrides.items():
        setattr(h, k, v)
    return h
