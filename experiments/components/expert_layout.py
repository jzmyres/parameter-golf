"""Explicit expert-slot layout for the recurrent OPG block.

The public architecture knobs are:
  - experts_per_slot: routed+shared expert modules in each attention/MLP bank
  - expert_slots: serial expert slots inside one recurrent block
  - expert_slot_order: parallel, attn_mlp, or mlp_attn

This replaces the older named chained-routing presets with one uniform,
layout-shaped implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor


VALID_EXPERT_SLOT_ORDERS = ("parallel", "attn_mlp", "mlp_attn")


@dataclass(frozen=True)
class ExpertSlotSpec:
    attn_experts: int = 0
    mlp_experts: int = 0
    shared_attn: int = 0
    shared_mlp: int = 0

    def __post_init__(self) -> None:
        vals = {
            "attn_experts": self.attn_experts,
            "mlp_experts": self.mlp_experts,
            "shared_attn": self.shared_attn,
            "shared_mlp": self.shared_mlp,
        }
        for name, value in vals.items():
            if int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.attn_total() <= 0 and self.mlp_total() <= 0:
            raise ValueError("an expert slot must own at least one attention or MLP expert")

    def attn_total(self) -> int:
        return int(self.shared_attn) + int(self.attn_experts)

    def mlp_total(self) -> int:
        return int(self.shared_mlp) + int(self.mlp_experts)

    def routed_total(self) -> int:
        return int(self.attn_experts) + int(self.mlp_experts)


def make_uniform_expert_layout(
    *,
    experts_per_slot: int,
    expert_slots: int,
    num_shared_experts: int = 0,
) -> list[ExpertSlotSpec]:
    experts_per_slot = int(experts_per_slot)
    expert_slots = int(expert_slots)
    num_shared_experts = int(num_shared_experts)
    if experts_per_slot <= 0:
        raise ValueError(f"experts_per_slot must be positive, got {experts_per_slot}")
    if expert_slots <= 0:
        raise ValueError(f"expert_slots must be positive, got {expert_slots}")
    if num_shared_experts < 0:
        raise ValueError(f"num_shared_experts must be non-negative, got {num_shared_experts}")
    if num_shared_experts >= experts_per_slot:
        raise ValueError(
            f"num_shared_experts ({num_shared_experts}) must be < experts_per_slot "
            f"({experts_per_slot}) so each slot has at least one routed expert"
        )
    routed = experts_per_slot - num_shared_experts
    return [
        ExpertSlotSpec(
            attn_experts=routed,
            mlp_experts=routed,
            shared_attn=num_shared_experts,
            shared_mlp=num_shared_experts,
        )
        for _ in range(expert_slots)
    ]


class ExpertSlotStack(nn.Module):
    """Serial stack of routed attention/MLP expert slots."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        kv_latent_dim: int,
        attn_expert_rank: int,
        mlp_expert_rank: int,
        slots: Iterable[ExpertSlotSpec],
        slot_order: str = "parallel",
        router_scoring: str = "linear",
        router_pertoken_entropy_coef: float = 0.0,
        router_dirichlet_ucb_beta: float = 0.0,
        use_router_sigmoid_gate: bool = True,
        use_entmax_routing: bool = False,
        entmax_blend_init_logit: float = 5.0,
        use_reverse_kl_balance: bool = True,
        use_nsa_attention: bool = False,
        nsa_compress_block_size: int = 32,
        nsa_compress_block_sliding_stride: int = 16,
        nsa_sliding_window_size: int = 256,
        nsa_branch_gate_init: float = 0.0,
        use_sparse_attn_head_gate: bool = False,
        sparse_attn_gate_window: int = 12,
        sparse_attn_gate_scale: float = 1.0,
        sparse_attn_gate_factor: float = 2.0,
        sparse_attn_gate_init_std: float = 0.0,
        use_rr_attention: bool = False,
        rr_stride: int = 8,
        rr_block_size: int = 64,
        rr_tau: float = 0.95,
        soft_dense_router_cls: type[nn.Module],
        causal_self_attention_cls: type[nn.Module],
        mlp_cls: type[nn.Module],
        rms_norm_cls: type[nn.Module],
    ):
        super().__init__()
        self.slots = list(slots)
        if not self.slots:
            raise ValueError("ExpertSlotStack requires at least one slot")
        self.slot_order = str(slot_order)
        if self.slot_order not in VALID_EXPERT_SLOT_ORDERS:
            raise ValueError(
                f"expert_slot_order={self.slot_order!r} must be one of "
                f"{','.join(VALID_EXPERT_SLOT_ORDERS)}"
            )

        self.routers = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.attn_post_norms = nn.ModuleList()
        self.mlp_post_norms = nn.ModuleList()
        self.shared_gate_attn = nn.ModuleList()
        self.shared_gate_mlp = nn.ModuleList()
        self.shared_gate_norm_weight_attn = nn.ParameterList()
        self.shared_gate_norm_weight_mlp = nn.ParameterList()
        self._slot_router_indices: list[int | None] = []
        self._slot_attn_indices: list[int | None] = []
        self._slot_mlp_indices: list[int | None] = []
        self._slot_attn_norm_indices: list[int | None] = []
        self._slot_mlp_norm_indices: list[int | None] = []
        self._slot_shared_attn_gate_indices: list[int | None] = []
        self._slot_shared_mlp_gate_indices: list[int | None] = []
        self._slot_shared_attn_norm_indices: list[int | None] = []
        self._slot_shared_mlp_norm_indices: list[int | None] = []

        for spec in self.slots:
            router_idx: int | None = None
            if spec.routed_total() > 0:
                router_idx = len(self.routers)
                health_slices = tuple(v for v in (spec.attn_experts, spec.mlp_experts) if v > 0)
                self.routers.append(
                    soft_dense_router_cls(
                        dim,
                        spec.routed_total(),
                        scoring=router_scoring,
                        health_slices=health_slices,
                        entropy_coef=router_pertoken_entropy_coef,
                        dirichlet_ucb_beta=router_dirichlet_ucb_beta,
                        use_router_sigmoid_gate=use_router_sigmoid_gate,
                        use_entmax_routing=use_entmax_routing,
                        entmax_blend_init_logit=entmax_blend_init_logit,
                        use_reverse_kl_balance=use_reverse_kl_balance,
                    )
                )
            self._slot_router_indices.append(router_idx)

            if spec.attn_total() > 0:
                self._slot_attn_indices.append(len(self.attns))
                self.attns.append(
                    causal_self_attention_cls(
                        dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                        kv_latent_dim=kv_latent_dim,
                        num_experts=spec.attn_total(),
                        expert_rank=attn_expert_rank,
                        router=self.routers[router_idx] if router_idx is not None else None,
                        use_nsa_attention=use_nsa_attention,
                        nsa_compress_block_size=nsa_compress_block_size,
                        nsa_compress_block_sliding_stride=nsa_compress_block_sliding_stride,
                        nsa_sliding_window_size=nsa_sliding_window_size,
                        nsa_branch_gate_init=nsa_branch_gate_init,
                        use_sparse_attn_head_gate=use_sparse_attn_head_gate,
                        sparse_attn_gate_window=sparse_attn_gate_window,
                        sparse_attn_gate_scale=sparse_attn_gate_scale,
                        sparse_attn_gate_factor=sparse_attn_gate_factor,
                        sparse_attn_gate_init_std=sparse_attn_gate_init_std,
                        use_rr_attention=use_rr_attention,
                        rr_stride=rr_stride,
                        rr_block_size=rr_block_size,
                        rr_tau=rr_tau,
                    )
                )
                self._slot_attn_norm_indices.append(len(self.attn_post_norms))
                self.attn_post_norms.append(rms_norm_cls(dim))
            else:
                self._slot_attn_indices.append(None)
                self._slot_attn_norm_indices.append(None)

            if spec.mlp_total() > 0:
                self._slot_mlp_indices.append(len(self.mlps))
                self.mlps.append(
                    mlp_cls(
                        dim, mlp_mult,
                        num_experts=spec.mlp_total(),
                        expert_rank=mlp_expert_rank,
                        router=self.routers[router_idx] if router_idx is not None else None,
                    )
                )
                self._slot_mlp_norm_indices.append(len(self.mlp_post_norms))
                self.mlp_post_norms.append(rms_norm_cls(dim))
            else:
                self._slot_mlp_indices.append(None)
                self._slot_mlp_norm_indices.append(None)

            if spec.shared_attn > 0:
                self._slot_shared_attn_gate_indices.append(len(self.shared_gate_attn))
                gate = nn.Linear(dim, spec.shared_attn, bias=True)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 1.0)
                self.shared_gate_attn.append(gate)
                self._slot_shared_attn_norm_indices.append(len(self.shared_gate_norm_weight_attn))
                self.shared_gate_norm_weight_attn.append(nn.Parameter(torch.ones(dim)))
            else:
                self._slot_shared_attn_gate_indices.append(None)
                self._slot_shared_attn_norm_indices.append(None)

            if spec.shared_mlp > 0:
                self._slot_shared_mlp_gate_indices.append(len(self.shared_gate_mlp))
                gate = nn.Linear(dim, spec.shared_mlp, bias=True)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 1.0)
                self.shared_gate_mlp.append(gate)
                self._slot_shared_mlp_norm_indices.append(len(self.shared_gate_norm_weight_mlp))
                self.shared_gate_norm_weight_mlp.append(nn.Parameter(torch.ones(dim)))
            else:
                self._slot_shared_mlp_gate_indices.append(None)
                self._slot_shared_mlp_norm_indices.append(None)

        self._diag_track_enabled = False
        self._attn_gate_call_track: list[Tensor] = []
        self._router_gate_call_track: list[Tensor] = []
        self._attn_router_gate_call_track: list[Tensor] = []
        self._attn_expert_weights_per_iter: list[Tensor] = []
        self._mlp_expert_weights_per_iter: list[Tensor] = []
        self._shared_gate_mean: Tensor | None = None
        self._shared_gate_min: Tensor | None = None
        self._shared_gate_std: Tensor | None = None
        self._shared_gate_diag_step: int | None = None

    def iter_routers(self):
        yield from self.routers

    def iter_attn_modules(self):
        for idx, module in enumerate(self.attns):
            yield f"attn_s{idx}", module

    def iter_mlp_modules(self):
        for idx, module in enumerate(self.mlps):
            yield f"mlp_s{idx}", module

    def _slot_router(self, slot_idx: int):
        idx = self._slot_router_indices[slot_idx]
        return self.routers[idx] if idx is not None else None

    def _route(self, slot_idx: int, h: Tensor) -> tuple[Tensor | None, Tensor | None]:
        spec = self.slots[slot_idx]
        router = self._slot_router(slot_idx)
        if router is None:
            return None, None
        w_all = router(h, pre_normed=True)
        offset = 0
        w_attn = None
        w_mlp = None
        if spec.attn_experts > 0:
            w_attn = w_all[..., offset:offset + spec.attn_experts].contiguous()
            offset += spec.attn_experts
        if spec.mlp_experts > 0:
            w_mlp = w_all[..., offset:offset + spec.mlp_experts].contiguous()
        return w_attn, w_mlp

    def _shared_gate(
        self,
        h: Tensor,
        *,
        gate_list: nn.ModuleList,
        norm_list: nn.ParameterList,
        gate_idx: int | None,
        norm_idx: int | None,
    ) -> Tensor | None:
        if gate_idx is None or norm_idx is None:
            return None
        weight = norm_list[norm_idx].to(dtype=h.dtype)
        return torch.sigmoid(gate_list[gate_idx](h * weight))

    def _capture_shared_gate_diag(self, gates: list[Tensor]) -> None:
        if not gates:
            self._shared_gate_mean = None
            self._shared_gate_min = None
            self._shared_gate_std = None
            self._shared_gate_diag_step = None
            return
        g = torch.cat([x.detach().float().reshape(-1) for x in gates], dim=0)
        self._shared_gate_mean = g.mean().detach()
        self._shared_gate_min = g.min().detach()
        self._shared_gate_std = g.std(unbiased=False).detach()
        self._shared_gate_diag_step = None

    def _maybe_track(self, slot_idx: int, w_attn: Tensor | None, w_mlp: Tensor | None) -> None:
        if not self._diag_track_enabled:
            return
        router = self._slot_router(slot_idx)
        if router is not None:
            rg = getattr(router, "_router_gate_last_mean", None)
            if rg is not None:
                self._router_gate_call_track.append(rg)
                self._attn_router_gate_call_track.append(rg)
        attn_idx = self._slot_attn_indices[slot_idx]
        if attn_idx is not None:
            ag = getattr(self.attns[attn_idx], "_attn_gate_last_mean", None)
            if ag is not None:
                self._attn_gate_call_track.append(ag)
        if w_attn is not None:
            reduce_dims = tuple(range(w_attn.dim() - 1))
            self._attn_expert_weights_per_iter.append(
                w_attn.detach().float().mean(dim=reduce_dims)
            )
        if w_mlp is not None:
            reduce_dims = tuple(range(w_mlp.dim() - 1))
            self._mlp_expert_weights_per_iter.append(
                w_mlp.detach().float().mean(dim=reduce_dims)
            )

    def _attention_delta(
        self,
        slot_idx: int,
        h: Tensor,
        w_attn: Tensor | None,
        shared_gates: list[Tensor],
    ) -> Tensor | None:
        spec = self.slots[slot_idx]
        attn_idx = self._slot_attn_indices[slot_idx]
        if attn_idx is None:
            return None
        attn_out = self.attns[attn_idx].forward_experts(h)
        g_s_attn = self._shared_gate(
            h,
            gate_list=self.shared_gate_attn,
            norm_list=self.shared_gate_norm_weight_attn,
            gate_idx=self._slot_shared_attn_gate_indices[slot_idx],
            norm_idx=self._slot_shared_attn_norm_indices[slot_idx],
        )
        if g_s_attn is not None:
            shared_gates.append(g_s_attn)
        parts = []
        if spec.shared_attn > 0 and g_s_attn is not None:
            parts.append((attn_out[:, :, :spec.shared_attn, :] * g_s_attn.unsqueeze(-1)).sum(dim=2))
        if spec.attn_experts > 0:
            assert w_attn is not None
            parts.append((attn_out[:, :, spec.shared_attn:, :] * w_attn.unsqueeze(-1)).sum(dim=2))
        attn_mix = sum(parts) if len(parts) > 1 else parts[0]
        norm_idx = self._slot_attn_norm_indices[slot_idx]
        assert norm_idx is not None
        return self.attn_post_norms[norm_idx](attn_mix)

    def _mlp_delta(
        self,
        slot_idx: int,
        h: Tensor,
        w_mlp: Tensor | None,
        shared_gates: list[Tensor],
    ) -> Tensor | None:
        spec = self.slots[slot_idx]
        mlp_idx = self._slot_mlp_indices[slot_idx]
        if mlp_idx is None:
            return None
        g_s_mlp = self._shared_gate(
            h,
            gate_list=self.shared_gate_mlp,
            norm_list=self.shared_gate_norm_weight_mlp,
            gate_idx=self._slot_shared_mlp_gate_indices[slot_idx],
            norm_idx=self._slot_shared_mlp_norm_indices[slot_idx],
        )
        if g_s_mlp is not None:
            shared_gates.append(g_s_mlp)
        if w_mlp is None:
            w_mlp = h.new_empty((*h.shape[:-1], 0))
        mlp_mix = self.mlps[mlp_idx].mix_experts(
            h, w_mlp,
            num_shared=spec.shared_mlp,
            shared_gate=g_s_mlp,
        )
        norm_idx = self._slot_mlp_norm_indices[slot_idx]
        assert norm_idx is not None
        return self.mlp_post_norms[norm_idx](mlp_mix)

    def forward(self, z_in: Tensor, x0: Tensor, rms_unit_fn: Callable[[Tensor], Tensor]) -> Tensor:
        cumulative_state = z_in
        delta_total = torch.zeros_like(z_in)
        shared_gates: list[Tensor] = []
        for slot_idx, _ in enumerate(self.slots):
            h = rms_unit_fn(cumulative_state + x0)
            w_attn, w_mlp = self._route(slot_idx, h)

            if self.slot_order == "parallel":
                attn_delta = self._attention_delta(slot_idx, h, w_attn, shared_gates)
                mlp_delta = self._mlp_delta(slot_idx, h, w_mlp, shared_gates)
                stage_delta = torch.zeros_like(z_in)
                if attn_delta is not None:
                    stage_delta = stage_delta + attn_delta
                if mlp_delta is not None:
                    stage_delta = stage_delta + mlp_delta
                cumulative_state = cumulative_state + stage_delta
                delta_total = delta_total + stage_delta
            elif self.slot_order == "attn_mlp":
                attn_delta = self._attention_delta(slot_idx, h, w_attn, shared_gates)
                if attn_delta is not None:
                    cumulative_state = cumulative_state + attn_delta
                    delta_total = delta_total + attn_delta
                h_mlp = rms_unit_fn(cumulative_state + x0)
                mlp_delta = self._mlp_delta(slot_idx, h_mlp, w_mlp, shared_gates)
                if mlp_delta is not None:
                    cumulative_state = cumulative_state + mlp_delta
                    delta_total = delta_total + mlp_delta
            else:
                mlp_delta = self._mlp_delta(slot_idx, h, w_mlp, shared_gates)
                if mlp_delta is not None:
                    cumulative_state = cumulative_state + mlp_delta
                    delta_total = delta_total + mlp_delta
                h_attn = rms_unit_fn(cumulative_state + x0)
                attn_delta = self._attention_delta(slot_idx, h_attn, w_attn, shared_gates)
                if attn_delta is not None:
                    cumulative_state = cumulative_state + attn_delta
                    delta_total = delta_total + attn_delta

            self._maybe_track(slot_idx, w_attn, w_mlp)

        self._capture_shared_gate_diag(shared_gates)
        return delta_total
