"""Active drop-in component modules for `train_gpt.py`.

Each active component is a self-contained Python module exporting:
- nn.Module classes / autograd functions / Triton kernels
- A module-level toggle + setter
- Public helper functions used by `train_gpt.py` after gating

Historical prototypes live under `experiments/components/archive/`. Restore
individual files into this directory before reactivating; see
`experiments/components/README.md` for the integration cookbook.
"""

from . import artifact_compression  # noqa: F401
from . import caseops_tokenizer  # noqa: F401
from . import expert_layout  # noqa: F401
from . import gptq_lqer  # noqa: F401
from . import phased_ttt  # noqa: F401
from . import rr_attention  # noqa: F401
from . import smear_gate  # noqa: F401
from . import sparse_attn_head_gate  # noqa: F401
