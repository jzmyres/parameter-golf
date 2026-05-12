"""Default-off chained expert stack for iter 103 / H77.

The module is standalone-importable: train_gpt.py injects the model classes
needed to build each stage, avoiding a component -> train_gpt circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor


_CHAINED_ROUTING_ENABLED: bool = False


def set_chained_routing_enabled(enabled: bool) -> None:
    global _CHAINED_ROUTING_ENABLED
    _CHAINED_ROUTING_ENABLED = bool(enabled)


def chained_routing_enabled() -> bool:
    return bool(_CHAINED_ROUTING_ENABLED)


@dataclass(frozen=True)
class StageSpec:
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
            raise ValueError("a chained-routing stage must own at least one attn or mlp expert")

    def attn_total(self) -> int:
        return int(self.shared_attn) + int(self.attn_experts)

    def mlp_total(self) -> int:
        return int(self.shared_mlp) + int(self.mlp_experts)

    def routed_total(self) -> int:
        return int(self.attn_experts) + int(self.mlp_experts)


PRESETS: dict[str, tuple[str, ...]] = {
    "unified": ("mixed_all",),
    "split_2stage": ("mixed_first_half", "mixed_second_half"),
    "attn_first_2stage": ("attn_all", "mlp_all"),
    "mlp_first_2stage": ("mlp_all", "attn_all"),
    "split_4stage": ("attn_first_half", "mlp_first_half", "attn_second_half", "mlp_second_half"),
}


def _split_total_slots(total: int) -> tuple[int, int]:
    first = int(total) // 2
    return first, int(total) - first


def _component_spec(total_slots: int, shared: int) -> tuple[int, int]:
    total_slots = int(total_slots)
    shared = int(shared)
    if total_slots <= 0:
        return 0, 0
    if shared <= 0:
        return total_slots, 0
    if total_slots <= shared:
        raise ValueError(
            f"not enough expert slots ({total_slots}) to allocate {shared} shared expert(s)"
        )
    return total_slots - shared, shared


def _stage_from_token(token: str, *, num_experts: int, num_shared_experts: int) -> StageSpec:
    total = int(num_experts)
    shared = int(num_shared_experts)
    if total <= 0:
        raise ValueError(f"num_experts must be positive, got {total}")
    if shared < 0 or shared >= total:
        raise ValueError(
            f"num_shared_experts must be in [0, num_experts), got {shared} for {total}"
        )
    routed = total - shared
    total0, total1 = _split_total_slots(total)
    r0, s0 = _component_spec(total0, shared)
    r1, s1 = _component_spec(total1, shared)

    if token == "mixed_all":
        return StageSpec(
            attn_experts=routed, mlp_experts=routed,
            shared_attn=shared, shared_mlp=shared,
        )
    if token == "mixed_first_half":
        return StageSpec(
            attn_experts=r0, mlp_experts=r0,
            shared_attn=s0, shared_mlp=s0,
        )
    if token == "mixed_second_half":
        return StageSpec(
            attn_experts=r1, mlp_experts=r1,
            shared_attn=s1, shared_mlp=s1,
        )
    if token == "attn_all":
        return StageSpec(attn_experts=routed, shared_attn=shared)
    if token == "mlp_all":
        return StageSpec(mlp_experts=routed, shared_mlp=shared)
    if token == "attn_first_half":
        return StageSpec(attn_experts=r0, shared_attn=s0)
    if token == "attn_second_half":
        return StageSpec(attn_experts=r1, shared_attn=s1)
    if token == "mlp_first_half":
        return StageSpec(mlp_experts=r0, shared_mlp=s0)
    if token == "mlp_second_half":
        return StageSpec(mlp_experts=r1, shared_mlp=s1)
    raise ValueError(f"unknown chained-routing preset token: {token!r}")


def _parse_stage_dict(raw: dict) -> StageSpec:
    allowed = {"attn_experts", "mlp_experts", "shared_attn", "shared_mlp"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown StageSpec keys: {unknown}")
    return StageSpec(**{k: int(raw.get(k, 0)) for k in allowed})


def parse_stages(
    preset_or_json: str | None,
    *,
    num_experts: int = 16,
    num_shared_experts: int = 1,
) -> list[StageSpec] | None:
    """Parse a preset name or JSON list-of-dicts into stage specs.

    ``None``, ``""``, and ``"none"`` return ``None`` so callers can keep the
    existing single-stage Block path.
    """
    if preset_or_json is None:
        return None
    text = str(preset_or_json).strip()
    if text == "" or text.lower() == "none":
        return None
    if text in PRESETS:
        return [
            _stage_from_token(t, num_experts=num_experts, num_shared_experts=num_shared_experts)
            for t in PRESETS[text]
        ]
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"unknown chained-routing preset or invalid JSON: {text!r}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("chained-routing JSON must be a non-empty list of stage objects")
    specs = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each chained-routing JSON stage must be an object")
        specs.append(_parse_stage_dict(item))
    return specs


class ChainedExpertStack(nn.Module):
    """Sequential residual chain of routed attention and MLP expert stages."""

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
        stages: Iterable[StageSpec],
        router_scoring: str = "linear",
        router_pertoken_entropy_coef: float = 0.0,
        router_dirichlet_ucb_beta: float = 0.0,
        use_router_sigmoid_gate: bool = True,
        use_entmax_routing: bool = False,
        entmax_blend_init_logit: float = 5.0,
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
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("ChainedExpertStack requires at least one stage")
        self.routers = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.attn_post_norms = nn.ModuleList()
        self.mlp_post_norms = nn.ModuleList()
        self.shared_gate_attn = nn.ModuleList()
        self.shared_gate_mlp = nn.ModuleList()
        self.shared_gate_norm_weight_attn = nn.ParameterList()
        self.shared_gate_norm_weight_mlp = nn.ParameterList()
        self._stage_router_indices: list[int | None] = []
        self._stage_attn_indices: list[int | None] = []
        self._stage_mlp_indices: list[int | None] = []
        self._stage_attn_norm_indices: list[int | None] = []
        self._stage_mlp_norm_indices: list[int | None] = []
        self._stage_shared_attn_gate_indices: list[int | None] = []
        self._stage_shared_mlp_gate_indices: list[int | None] = []
        self._stage_shared_attn_norm_indices: list[int | None] = []
        self._stage_shared_mlp_norm_indices: list[int | None] = []

        for spec in self.stages:
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
                    )
                )
            self._stage_router_indices.append(router_idx)

            if spec.attn_total() > 0:
                self._stage_attn_indices.append(len(self.attns))
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
                self._stage_attn_norm_indices.append(len(self.attn_post_norms))
                self.attn_post_norms.append(rms_norm_cls(dim))
            else:
                self._stage_attn_indices.append(None)
                self._stage_attn_norm_indices.append(None)

            if spec.mlp_total() > 0:
                self._stage_mlp_indices.append(len(self.mlps))
                self.mlps.append(
                    mlp_cls(
                        dim, mlp_mult,
                        num_experts=spec.mlp_total(),
                        expert_rank=mlp_expert_rank,
                        router=self.routers[router_idx] if router_idx is not None else None,
                    )
                )
                self._stage_mlp_norm_indices.append(len(self.mlp_post_norms))
                self.mlp_post_norms.append(rms_norm_cls(dim))
            else:
                self._stage_mlp_indices.append(None)
                self._stage_mlp_norm_indices.append(None)

            if spec.shared_attn > 0:
                self._stage_shared_attn_gate_indices.append(len(self.shared_gate_attn))
                gate = nn.Linear(dim, spec.shared_attn, bias=True)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 1.0)
                self.shared_gate_attn.append(gate)
                self._stage_shared_attn_norm_indices.append(len(self.shared_gate_norm_weight_attn))
                self.shared_gate_norm_weight_attn.append(nn.Parameter(torch.ones(dim)))
            else:
                self._stage_shared_attn_gate_indices.append(None)
                self._stage_shared_attn_norm_indices.append(None)

            if spec.shared_mlp > 0:
                self._stage_shared_mlp_gate_indices.append(len(self.shared_gate_mlp))
                gate = nn.Linear(dim, spec.shared_mlp, bias=True)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 1.0)
                self.shared_gate_mlp.append(gate)
                self._stage_shared_mlp_norm_indices.append(len(self.shared_gate_norm_weight_mlp))
                self.shared_gate_norm_weight_mlp.append(nn.Parameter(torch.ones(dim)))
            else:
                self._stage_shared_mlp_gate_indices.append(None)
                self._stage_shared_mlp_norm_indices.append(None)

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

    def _stage_router(self, stage_idx: int):
        idx = self._stage_router_indices[stage_idx]
        return self.routers[idx] if idx is not None else None

    def _route(self, stage_idx: int, h: Tensor) -> tuple[Tensor | None, Tensor | None]:
        spec = self.stages[stage_idx]
        router = self._stage_router(stage_idx)
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

    def _maybe_track(self, stage_idx: int, w_attn: Tensor | None, w_mlp: Tensor | None) -> None:
        if not self._diag_track_enabled:
            return
        router = self._stage_router(stage_idx)
        if router is not None:
            rg = getattr(router, "_router_gate_last_mean", None)
            if rg is not None:
                self._router_gate_call_track.append(rg)
                self._attn_router_gate_call_track.append(rg)
        attn_idx = self._stage_attn_indices[stage_idx]
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

    def forward(self, z_in: Tensor, x0: Tensor, rms_unit_fn: Callable[[Tensor], Tensor]) -> Tensor:
        cumulative_state = z_in
        delta_total = torch.zeros_like(z_in)
        shared_gates: list[Tensor] = []
        for stage_idx, spec in enumerate(self.stages):
            h = rms_unit_fn(cumulative_state + x0)
            w_attn, w_mlp = self._route(stage_idx, h)
            stage_delta = torch.zeros_like(z_in)

            attn_idx = self._stage_attn_indices[stage_idx]
            if attn_idx is not None:
                attn_out = self.attns[attn_idx].forward_experts(h)
                g_s_attn = self._shared_gate(
                    h,
                    gate_list=self.shared_gate_attn,
                    norm_list=self.shared_gate_norm_weight_attn,
                    gate_idx=self._stage_shared_attn_gate_indices[stage_idx],
                    norm_idx=self._stage_shared_attn_norm_indices[stage_idx],
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
                norm_idx = self._stage_attn_norm_indices[stage_idx]
                assert norm_idx is not None
                stage_delta = stage_delta + self.attn_post_norms[norm_idx](attn_mix)

            mlp_idx = self._stage_mlp_indices[stage_idx]
            if mlp_idx is not None:
                g_s_mlp = self._shared_gate(
                    h,
                    gate_list=self.shared_gate_mlp,
                    norm_list=self.shared_gate_norm_weight_mlp,
                    gate_idx=self._stage_shared_mlp_gate_indices[stage_idx],
                    norm_idx=self._stage_shared_mlp_norm_indices[stage_idx],
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
                norm_idx = self._stage_mlp_norm_indices[stage_idx]
                assert norm_idx is not None
                stage_delta = stage_delta + self.mlp_post_norms[norm_idx](mlp_mix)

            cumulative_state = cumulative_state + stage_delta
            delta_total = delta_total + stage_delta
            self._maybe_track(stage_idx, w_attn, w_mlp)

        self._capture_shared_gate_diag(shared_gates)
        return delta_total
