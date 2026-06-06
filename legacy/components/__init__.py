"""Archived component modules for the legacy rich model `legacy/train_gpt_rich.py`.

These are NOT imported by the active M0 `train_gpt.py` — they were archived here
in the 2026-06-06 reorg (ADR 0003). Each is a self-contained module exporting:
- nn.Module classes / autograd functions / Triton kernels
- A module-level toggle + setter
- Public helper functions used by the rich model after gating

Earlier prototypes live under `archive/`. See `README.md` for the integration
cookbook (historical).
"""

from . import artifact_compression  # noqa: F401
from . import caseops_tokenizer  # noqa: F401
from . import expert_layout  # noqa: F401
from . import gptq_lqer  # noqa: F401
from . import phased_ttt  # noqa: F401
from . import rr_attention  # noqa: F401
from . import smear_gate  # noqa: F401
from . import sparse_attn_head_gate  # noqa: F401
