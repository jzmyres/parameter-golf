"""Archived fixed-point consistency losses.

This module is documentation-only. `train_gpt.py` rejects the legacy
prefix-anchor consistency flags for finite-horizon OPG runs.
"""

from __future__ import annotations

import torch
from torch import Tensor


def recursive_prefix_anchor_loss(z_stack: Tensor) -> Tensor:
    """Historical iter172 nearest-neighbor anchor loss.

    It paired each prefix endpoint with the next deeper endpoint:
    mean_i ||z_i - stopgrad(z_{i+1})||^2.
    """
    if z_stack.shape[0] < 2:
        return z_stack.new_zeros(())
    return (z_stack[:-1] - z_stack[1:].detach()).pow(2).mean()

