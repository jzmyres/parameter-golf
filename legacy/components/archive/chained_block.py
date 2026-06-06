"""Iter 103 / H77: Chained 2-stage pooled routing inside `T_θ`.

Drop-in component for `train_gpt.py::Block` when `use_chained_routing=True`.

# Design (matches experiments/docs/hypotheses.md H77)

Single-stage baseline (iter 117 v5 / iter 117b-1):
    T_θ(z, x_0) = Δ(z, x_0) + B̄ ⊙ RMSNorm(x_0)
    where Δ(z, x_0) = mlp(attn(z))  # both share ONE pooled router

Chained 2-stage (iter 103, H77 spec):
    T_θ(z, x_0) = Δ_2(Δ_1(z, x_0), x_0) + B̄ ⊙ RMSNorm(x_0)
    Δ_1, Δ_2 use INDEPENDENT pooled routers + INDEPENDENT expert sets,
    each sized num_routed/2 attn + num_routed/2 mlp experts.

# Implementation strategy

Two SoftDenseRouter instances `router_1`, `router_2` each pooling
2 × (num_routed/2) outputs (attn slice + mlp slice). Two CausalSelfAttention
modules `attn_1`, `attn_2` each with num_routed/2 attn experts.
Two MLP modules `mlp_1`, `mlp_2` each with num_routed/2 mlp experts.

Per H77 design questions resolution (defaults chosen):
- Iso-expert-count (option a): keep total experts unchanged at
  num_routed; split as (num_routed//2) per stage.
- 1 shared expert per stage per type (so 2 shared attn + 2 shared mlp
  total per Block).
- Skip-connection: out = stage_1_out + stage_2_out (preserves iter
  100b strict-gen path when stage 2 expert outputs zero-init).

# Strict-generalization preservation

Setting all stage-2 expert weights AND stage-2 router output to zero
recovers single-stage iter 117b-1 forward map exactly:
- stage_2_out = 0 (zero-init) ⇒ T_θ = stage_1_out + B̄·RMSNorm(x_0)
- stage_1 with num_routed/2 experts is NOT iso-functional with
  single-stage num_routed experts. Therefore option (a) does NOT
  strictly generalize iter 117b-1; promotion under §11 standard rule.

# Risks (per H77)

- DDP all-reduce bandwidth doubles (router gradient traffic doubles)
- Step-time +30-50% expected (no parallelism between stages)
- Stage 2 expert starvation if stage 1 output near-zero at cold start.
  Mitigation: existing entropy ramp + cv_loss carries through to
  stage 2.
- Routing depth interacts with DEQ: chained router is part of T_θ,
  so FP equation effectively deeper. K-sweep gate critical; verify
  acyclicity primes K∈{17, 37, 113} per iter 97.6.

# Integration into train_gpt.py (after smoke test)

The integration is a Block-level branch. Three touchpoints:

1. `Block.__init__` — when `use_chained_routing=True`, construct
   `attn_1, attn_2, mlp_1, mlp_2, router_1, router_2` instead of the
   single-stage `attn, mlp, router`. Each at num_routed//2 experts.
2. `Block.forward` — when chained, run stage 1 then stage 2 with chain.
3. `Block.ortho_aux`, optimizer coverage in `_build_optimizer_param_lists`,
   and `_collect_routing_losses` need to enumerate both stages.

This file documents the design but does NOT yet land the actual
`Block` subclass — the existing Block has too many cross-cutting
attributes (DEQ residuals, lyapunov hooks, ortho_aux output_norm
caches, etc.) that a derived `ChainedBlock` would either inherit or
duplicate. Cleaner to refactor `Block.__init__` and `Block.forward`
in-place behind the `use_chained_routing` flag.

# Implementation plan (X3 steps 3-5)

## Step 3 — Block.__init__ branch
- After existing single-stage construction, if `use_chained_routing`:
  - Discard single-stage `self.attn`, `self.mlp`, `self.router`
  - Construct `self.attn_1`, `self.mlp_1`, `self.router_1` at
    num_routed//2 experts
  - Construct `self.attn_2`, `self.mlp_2`, `self.router_2` at
    `num_routed - num_routed//2` experts (handles odd num_routed)
  - Construct shared experts per stage (2× the existing shared
    expert allocations)
  - Halve `attn_expert_rank` and `mlp_expert_rank` to preserve
    total per-stage param count if iso-stage-cost variant requested
    (currently iso-expert-count → ranks unchanged, params halve).

## Step 4 — Block.forward chain
```
def forward(self, z, x_0, b_bar=None):
    if self.use_chained_routing:
        # Stage 1: route on z, use stage-1 experts
        attn_1_out = self.attn_1(z, x_0, ..., router=self.router_1)
        mlp_1_out  = self.mlp_1(attn_1_out, x_0, ..., router=self.router_1)
        delta_1 = self.attn_post_mix_norm_1(attn_1_out) \\
                + self.mlp_post_mix_norm_1(mlp_1_out)
        # Stage 2: route on delta_1, use stage-2 experts
        attn_2_out = self.attn_2(delta_1, x_0, ..., router=self.router_2)
        mlp_2_out  = self.mlp_2(attn_2_out, x_0, ..., router=self.router_2)
        delta_2 = self.attn_post_mix_norm_2(attn_2_out) \\
                + self.mlp_post_mix_norm_2(mlp_2_out)
        delta = delta_1 + delta_2  # skip across the chain
    else:
        # Existing single-stage path unchanged
        ...
    return delta + b_bar * RMSNorm(x_0)
```

## Step 5 — auxiliaries
- `Block.ortho_aux`: enumerate stage_1 + stage_2 expert outputs;
  return concatenated/averaged ortho score
- `GPT._build_optimizer_param_lists`: add stage_2 params to the
  same groups as stage_1 (Muon for matrices, AdamW for scalars)
- `GPT._collect_routing_losses`: dedup BOTH router_1 and router_2
  via id(), accumulate cv + entropy + block_ortho into the unified
  `router_reg_loss` group (see iter 117b-1 commit 54473aa)
- `K-sweep diagnostics`: report attn_cv_1, mlp_cv_1, attn_cv_2,
  mlp_cv_2 per stage so the per-stage routing health is visible

## Testing protocol

1. CPU smoke: construct Block with use_chained_routing=True, fwd,
   verify shapes match single-stage Block output.
2. Gradient check: small-D, small-T case; verify gradcheck passes.
3. GPU smoke: launch via experiments/smoke_test.py
   --use-chained-routing=1 with iterations=300; verify loss decreases.
4. Full launch with --use-chained-routing=1 iterations=1000 once
   the existing iter 117b-1 finishes.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level toggle (mirrors train_gpt.py::_USE_ENTMAX_TRITON pattern)
# ---------------------------------------------------------------------------

_USE_CHAINED_ROUTING: bool = False


def set_chained_routing(enabled: bool) -> None:
    """Module-level toggle invoked from train_gpt.py::main() at startup.

    Set ONCE before model construction so dynamo constant-folds the
    Block.forward branch into the compiled graph.
    """
    global _USE_CHAINED_ROUTING
    _USE_CHAINED_ROUTING = bool(enabled)


def is_chained_routing_enabled() -> bool:
    return _USE_CHAINED_ROUTING


# ---------------------------------------------------------------------------
# ChainedBlockMixin — adds chained routing methods that Block subclasses can
# splice into __init__ and forward. NOT yet wired into train_gpt.py::Block;
# this is the design template for X3 step 3+.
# ---------------------------------------------------------------------------


class ChainedBlockMixin:
    """Methods to be merged into Block (or composed via inheritance) when
    `use_chained_routing=True`.

    Usage (planned):
        class Block(ChainedBlockMixin, nn.Module):
            def __init__(self, ...):
                super().__init__()
                if use_chained_routing:
                    self._init_chained(...)
                else:
                    self._init_single_stage(...)
            def forward(self, z, x_0, b_bar=None):
                if self.use_chained_routing:
                    return self._forward_chained(z, x_0, b_bar)
                else:
                    return self._forward_single_stage(z, x_0, b_bar)

    The current `train_gpt.py::Block` would need these methods added
    + the __init__/forward branches. Splitting via mixin keeps the
    chained-specific code logically separated.
    """

    # Methods that are currently NotImplementedError stubs. The smoke test
    # asserts this set matches `dir()` so adding a method without raising
    # cannot pass vacuously.
    _DEFERRED_METHODS: frozenset[str] = frozenset({
        "_init_chained_routers",
        "_forward_chained",
    })

    def _init_chained_routers(
        self,
        dim: int,
        num_routed_total: int,
        # Forwarded SoftDenseRouter kwargs go here in the actual integration
    ) -> None:
        """Stage-1/stage-2 router construction. Call AFTER super().__init__().

        num_routed_per_stage = num_routed_total // 2 (with stage_2 holding
        the +1 if odd). Each stage's router pools 2 × num_routed_per_stage
        outputs (attn slice + mlp slice).
        """
        self.num_routed_per_stage_1 = num_routed_total // 2
        self.num_routed_per_stage_2 = num_routed_total - self.num_routed_per_stage_1
        # NOTE: this is a stub — actual SoftDenseRouter construction lives
        # in train_gpt.py and depends on its full signature. The integration
        # commit will instantiate `self.router_1` and `self.router_2` here
        # using the same SoftDenseRouter constructor as the single-stage path.
        raise NotImplementedError(
            "Step 3 stub: actual router construction defers to train_gpt.py "
            "where SoftDenseRouter has access to its full kwargs context."
        )

    def _forward_chained(
        self,
        z: Tensor,
        x_0: Tensor,
        b_bar: Optional[Tensor] = None,
    ) -> Tensor:
        """Chained forward: T_2(T_1(z, x_0), x_0) + skip + Parcae injection.

        Args:
            z: (B, T, D) DEQ state
            x_0: (B, T, D) input injection
            b_bar: (D,) Parcae per-dim B̄ (None ⇒ ones)

        Returns:
            (B, T, D) — same shape contract as Block.forward.
        """
        raise NotImplementedError(
            "Step 4 stub: requires full Block context (attn_1/2, mlp_1/2, "
            "post_mix_norms, x0_inject_norm_weight). Integration in "
            "train_gpt.py::Block.forward."
        )


# ---------------------------------------------------------------------------
# Smoke test for the toggle (no model construction)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    # 1. Toggle round-trip
    assert not is_chained_routing_enabled()
    set_chained_routing(True)
    assert is_chained_routing_enabled()
    set_chained_routing(False)
    assert not is_chained_routing_enabled()

    # 2. Mixin contract: deferred methods MUST raise NotImplementedError, and
    #    the set of public methods on ChainedBlockMixin must equal the
    #    declared `_DEFERRED_METHODS`. A new public method without a
    #    NotImplementedError stub would otherwise let the smoke pass vacuously.
    actual_methods = frozenset(
        name for name in dir(ChainedBlockMixin)
        if not name.startswith("_") or name in ChainedBlockMixin._DEFERRED_METHODS
    ) - {"_DEFERRED_METHODS"}
    assert actual_methods == ChainedBlockMixin._DEFERRED_METHODS, (
        f"ChainedBlockMixin method set drifted: declared "
        f"{set(ChainedBlockMixin._DEFERRED_METHODS)}, got {set(actual_methods)}. "
        f"Either implement the new method (and remove its NotImplementedError "
        f"stub + drop it from _DEFERRED_METHODS) or add it to _DEFERRED_METHODS."
    )

    class _DummyHost(ChainedBlockMixin):
        pass

    host = _DummyHost()
    try:
        host._init_chained_routers(dim=8, num_routed_total=4)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("_init_chained_routers should raise NotImplementedError")

    z = torch.zeros(1, 4, 8)
    try:
        host._forward_chained(z, z)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("_forward_chained should raise NotImplementedError")

    # 3. Iso-expert-count split is well-defined for both even and odd totals
    #    (stage 2 holds the remainder).
    for total in (2, 3, 4, 5, 8, 16):
        s1 = total // 2
        s2 = total - s1
        assert s1 + s2 == total
        assert s2 >= s1  # stage 2 absorbs the +1 on odd totals

    print("chained_block.py: 3/3 smoke checks PASS")


if __name__ == "__main__":
    _smoke_test()
