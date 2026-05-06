"""Active drop-in component modules for `train_gpt.py`.

Each active component is a self-contained Python module exporting:
- nn.Module classes / autograd functions / Triton kernels
- A module-level toggle + setter
- Public helper functions used by `train_gpt.py` after gating

Historical prototypes live under `experiments/components/archive/`. Restore
individual files into this directory before reactivating; see
`experiments/components/README.md` for the integration cookbook.
"""
