"""Drop-in component modules for `train_gpt.py` (iter 117b-2/3, iter 103+).

Each component is a self-contained Python module exporting:
- nn.Module classes / autograd functions / Triton kernels
- A module-level toggle + setter
- Public helper functions used by `train_gpt.py` after gating

See `experiments/components/README.md` for the integration cookbook.
"""
