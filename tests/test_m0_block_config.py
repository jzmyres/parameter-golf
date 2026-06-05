"""Configurable recurrence-block structure for the M0 architecture search.

Covers the five composable, reversibility-preserving knobs added to the
M0 ``_PreNormDeltaBlock`` / ``Hyperparameters`` / CLI:

  1. ``block_order`` in {attn_ffn, ffn_attn, parallel}
  2. ``attn_moe`` (bool): attention as a per-token MoE over low-rank MLA experts
  3. ``num_shared_experts`` (int): DeepSeek-style always-on experts
  4. ``n_sublayers`` (int): a stack of UNIQUE attn+MoE sub-blocks per F/G map
  5. ``expert_b_init`` in {small, zero}: routed-expert output-proj init

The hard gate is reversibility: for EVERY knob combination the additive-coupling
recurrence must reconstruct ``x0`` exactly in fp64. Each knob test also asserts
(a) forward+backward runs, (b) the knob actually changes the computation
(not a silent no-op), and (c) optimizer coverage stays an exact partition.
"""
import itertools

import pytest
import torch

from train_gpt import (
    Hyperparameters,
    M0GPT,
    Muon,
    _PreNormDeltaBlock,
    build_optimizers,
)


def _args(**over):
    base = dict(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=16,
        n_experts=4, expert_rank=4, n_mix=2, kv_latent=4, head_dim=8,
        max_seq_len=16,
    )
    base.update(over)
    return Hyperparameters(**base)


# ---------------------------------------------------------------------------
# Defaults preserved (existing behavior unchanged unless a knob is set).
# ---------------------------------------------------------------------------
def test_block_config_defaults_match_current_behavior():
    a = _args()
    assert a.block_order == "attn_ffn"
    assert a.attn_moe is False
    assert a.num_shared_experts == 0
    assert a.n_sublayers == 1
    assert a.expert_b_init == "small"


def test_default_block_is_single_mla_plus_moe():
    """Default ``_PreNormDeltaBlock`` keeps one shared MLA + one FFN MoE."""
    from train_gpt import MLAttention, SwiGLUMoE

    blk = _PreNormDeltaBlock(_args())
    assert len(blk.sublayers) == 1
    sub = blk.sublayers[0]
    assert isinstance(sub.attn, MLAttention)
    assert isinstance(sub.moe, SwiGLUMoE)


# ---------------------------------------------------------------------------
# The hard gate: fp64 reconstruction across every knob combination.
# ---------------------------------------------------------------------------
_BLOCK_ORDERS = ("attn_ffn", "ffn_attn", "parallel")
_ATTN_MOE = (False, True)
_SHARED = (0, 1)
_SUBLAYERS = (1, 2)


@pytest.mark.parametrize(
    "block_order,attn_moe,num_shared_experts,n_sublayers",
    list(itertools.product(_BLOCK_ORDERS, _ATTN_MOE, _SHARED, _SUBLAYERS)),
)
def test_recurrence_reconstructs_fp64_for_every_combo(
    block_order, attn_moe, num_shared_experts, n_sublayers
):
    torch.manual_seed(0)
    args = _args(
        block_order=block_order, attn_moe=attn_moe,
        num_shared_experts=num_shared_experts, n_sublayers=n_sublayers,
        n_experts=4,
    )
    m = M0GPT(args).double()
    rec = m.rec
    # Re-randomize the near-identity inits so the recurrence is NON-trivial
    # (otherwise reconstruction is vacuous against an identity map).
    with torch.no_grad():
        for blk in (rec.F, rec.G):
            for sub in blk.sublayers:
                for mod in sub.attn_modules():
                    mod.o_proj.weight.normal_(std=0.3)
                for moe in sub.moe_modules():
                    moe.w_out.normal_(std=0.3)
    x0 = torch.randn(2, 5, 16, dtype=torch.float64)
    (aK, bK), _ = rec.forward_states(x0, x0, x0, depth=4)
    assert not torch.allclose(aK, x0), "recurrence is trivially identity; test vacuous"
    a0, b0 = rec.invert(aK, bK, x0, depth=4)
    assert torch.allclose(a0, x0, atol=1e-7) and torch.allclose(b0, x0, atol=1e-7)


@pytest.mark.parametrize(
    "block_order,attn_moe,num_shared_experts,n_sublayers",
    list(itertools.product(_BLOCK_ORDERS, _ATTN_MOE, _SHARED, _SUBLAYERS)),
)
def test_forward_backward_for_every_combo(
    block_order, attn_moe, num_shared_experts, n_sublayers
):
    torch.manual_seed(0)
    args = _args(
        block_order=block_order, attn_moe=attn_moe,
        num_shared_experts=num_shared_experts, n_sublayers=n_sublayers,
    )
    m = M0GPT(args)
    x = torch.randint(0, 16, (2, 8))
    y = torch.randint(0, 16, (2, 8))
    loss = m(x, y, depth=3)
    loss.backward()
    assert torch.isfinite(loss)
    # at least one expert/attn weight got a finite gradient
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(
    "block_order,attn_moe,num_shared_experts,n_sublayers",
    list(itertools.product(_BLOCK_ORDERS, _ATTN_MOE, _SHARED, _SUBLAYERS)),
)
def test_optimizer_coverage_exact_for_every_combo(
    block_order, attn_moe, num_shared_experts, n_sublayers
):
    args = _args(
        model_dim=32, vocab_size=1024,
        block_order=block_order, attn_moe=attn_moe,
        num_shared_experts=num_shared_experts, n_sublayers=n_sublayers,
    )
    m = M0GPT(args)
    optimizers = build_optimizers(m, matrix_lr=0.02, embed_lr=0.1, scalar_lr=0.02)
    seen = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                assert id(p) not in seen, "param in more than one optimizer group"
                seen[id(p)] = True
    trainable_ids = {id(p) for p in m.parameters() if p.requires_grad}
    assert trainable_ids == set(seen.keys()), "optimizer groups not an exact cover"
    # The (attn-MoE and FFN-MoE) expert banks all go to Muon (ndim>=2, not router/gate).
    muon_ids = {
        id(p) for opt in optimizers if isinstance(opt, Muon)
        for g in opt.param_groups for p in g["params"]
    }
    for blk in (m.rec.F, m.rec.G):
        for sub in blk.sublayers:
            for moe in sub.moe_modules():
                assert id(moe.w_in) in muon_ids and id(moe.w_out) in muon_ids


# ---------------------------------------------------------------------------
# Knob 1: block order changes the computation.
# ---------------------------------------------------------------------------
def test_block_order_changes_output():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    outs = {}
    for order in _BLOCK_ORDERS:
        torch.manual_seed(0)
        blk = _PreNormDeltaBlock(_args(block_order=order))
        outs[order] = blk(x)
    assert not torch.allclose(outs["attn_ffn"], outs["ffn_attn"], atol=1e-6)
    assert not torch.allclose(outs["attn_ffn"], outs["parallel"], atol=1e-6)
    assert not torch.allclose(outs["ffn_attn"], outs["parallel"], atol=1e-6)


def test_parallel_attn_and_moe_share_same_input():
    """In parallel order both branches read norm(inp) (not the attn-updated state)."""
    torch.manual_seed(0)
    blk = _PreNormDeltaBlock(_args(block_order="parallel"))
    sub = blk.sublayers[0]
    seen = {}
    orig_attn = sub.attn.forward
    orig_moe = sub.moe.forward

    def cap_attn(x):
        seen["attn_in"] = x.detach().clone()
        return orig_attn(x)

    def cap_moe(x):
        seen["moe_in"] = x.detach().clone()
        return orig_moe(x)

    sub.attn.forward = cap_attn
    sub.moe.forward = cap_moe
    blk(torch.randn(1, 3, 16))
    assert torch.allclose(seen["attn_in"], seen["moe_in"])


def test_attn_ffn_feeds_moe_the_attn_updated_state():
    torch.manual_seed(0)
    blk = _PreNormDeltaBlock(_args(block_order="attn_ffn"))
    sub = blk.sublayers[0]
    seen = {}
    orig_attn = sub.attn.forward
    orig_moe = sub.moe.forward

    def cap_attn(x):
        seen["attn_in"] = x.detach().clone()
        return orig_attn(x)

    def cap_moe(x):
        seen["moe_in"] = x.detach().clone()
        return orig_moe(x)

    sub.attn.forward = cap_attn
    sub.moe.forward = cap_moe
    blk(torch.randn(1, 3, 16))
    assert not torch.allclose(seen["attn_in"], seen["moe_in"])


# ---------------------------------------------------------------------------
# Knob 2: attn-MoE actually swaps the attention to a routed MoE.
# ---------------------------------------------------------------------------
def test_attn_moe_uses_mla_moe_and_routes():
    from train_gpt import MLAMoE, MLAttention

    blk_off = _PreNormDeltaBlock(_args(attn_moe=False))
    blk_on = _PreNormDeltaBlock(_args(attn_moe=True, n_attn_experts=3))
    assert isinstance(blk_off.sublayers[0].attn, MLAttention)
    assert isinstance(blk_on.sublayers[0].attn, MLAMoE)
    x = torch.randn(2, 5, 16)
    attn = blk_on.sublayers[0].attn
    y = attn(x)
    assert y.shape == x.shape
    assert attn.last_route is not None
    assert attn.last_route.shape[-1] == 3


def test_attn_moe_changes_block_output_vs_single_mla():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    torch.manual_seed(0)
    off = _PreNormDeltaBlock(_args(attn_moe=False))(x)
    torch.manual_seed(0)
    on = _PreNormDeltaBlock(_args(attn_moe=True))(x)
    assert off.shape == on.shape
    assert not torch.allclose(off, on, atol=1e-6)


def test_attn_moe_is_smooth_for_reversibility():
    from train_gpt import MLAMoE

    torch.manual_seed(0)
    attn = MLAMoE(dim=16, n_heads=2, n_kv_heads=1, kv_latent=4, head_dim=8,
                  n_experts=4, router_type="softmax")
    x = torch.randn(2, 5, 16)
    y1 = attn(x)
    y2 = attn(x + 1e-7)
    assert (y1 - y2).abs().max() < 1e-3


# ---------------------------------------------------------------------------
# Knob 3: shared always-on experts.
# ---------------------------------------------------------------------------
def test_shared_experts_allocated_and_change_output():
    from train_gpt import SwiGLUMoE

    moe0 = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, num_shared_experts=0)
    moe1 = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, num_shared_experts=2)
    assert not hasattr(moe0, "shared_w_in") or moe0.shared_w_in is None
    assert moe1.shared_w_in is not None and moe1.shared_w_in.shape[0] == 2
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    # Same routed weights/experts (seeded identically) but shared adds a base.
    torch.manual_seed(1)
    a = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, num_shared_experts=0)
    torch.manual_seed(1)
    b = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, num_shared_experts=2)
    assert not torch.allclose(a(x), b(x), atol=1e-6)


def test_shared_experts_always_on_independent_of_router():
    """Shared experts must contribute even when the router emits zero mass."""
    from train_gpt import SwiGLUMoE

    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, num_shared_experts=2,
                    router_type="relu")
    # Force the router to emit all-negative logits -> relu -> exact-zero route.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.fill_(-100.0)
    x = torch.randn(2, 5, 16)
    y = moe(x)
    assert (moe.last_route == 0).all(), "router should be fully shed here"
    assert y.abs().sum() > 0, "shared experts must keep the MoE alive"


# ---------------------------------------------------------------------------
# Knob 4: n_sublayers stacks unique sub-blocks.
# ---------------------------------------------------------------------------
def test_n_sublayers_allocates_unique_subblocks():
    blk = _PreNormDeltaBlock(_args(n_sublayers=2))
    assert len(blk.sublayers) == 2
    s0, s1 = blk.sublayers
    # distinct parameter tensors (no sharing/aliasing between sublayers)
    ids0 = {id(p) for p in s0.parameters()}
    ids1 = {id(p) for p in s1.parameters()}
    assert ids0.isdisjoint(ids1)


def test_n_sublayers_changes_output_and_is_pure():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    torch.manual_seed(0)
    one = _PreNormDeltaBlock(_args(n_sublayers=1))(x)
    torch.manual_seed(0)
    two = _PreNormDeltaBlock(_args(n_sublayers=2))(x)
    assert not torch.allclose(one, two, atol=1e-6)
    # purity: identical input -> identical output (deterministic, no state)
    blk = _PreNormDeltaBlock(_args(n_sublayers=2))
    assert torch.allclose(blk(x), blk(x))


def test_layout_param_count_tradeoff_16x1_vs_8x2():
    """16 experts x 1 sublayer vs 8 experts x 2 sublayers are both valid layouts."""
    a = M0GPT(_args(n_experts=16, n_sublayers=1))
    b = M0GPT(_args(n_experts=8, n_sublayers=2))
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    assert na > 0 and nb > 0


# ---------------------------------------------------------------------------
# Knob 5: expert B-init.
# ---------------------------------------------------------------------------
def test_expert_b_init_zero_zeroes_routed_w_out():
    from train_gpt import SwiGLUMoE

    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4,
                    num_shared_experts=1, expert_b_init="zero")
    assert torch.count_nonzero(moe.w_out) == 0
    # shared experts keep a non-zero base so the MoE is not inert.
    assert torch.count_nonzero(moe.shared_w_out) > 0


def test_expert_b_init_small_is_nonzero():
    from train_gpt import SwiGLUMoE

    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=4, expert_b_init="small")
    assert torch.count_nonzero(moe.w_out) > 0


def test_m0gpt_expert_b_init_zero_with_shared_base():
    """expert_b_init=zero only sensible with shared experts; model still runs."""
    args = _args(num_shared_experts=1, expert_b_init="zero")
    m = M0GPT(args)
    # routed w_out zeroed at init; shared base present.
    for blk in (m.rec.F, m.rec.G):
        for sub in blk.sublayers:
            for moe in sub.moe_modules():
                assert torch.count_nonzero(moe.w_out) == 0
                assert torch.count_nonzero(moe.shared_w_out) > 0
    x = torch.randint(0, 16, (2, 8))
    y = torch.randint(0, 16, (2, 8))
    loss = m(x, y, depth=3)
    loss.backward()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# CLI parser wires every knob.
# ---------------------------------------------------------------------------
def test_cli_parser_exposes_block_config_knobs():
    from train_gpt import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args([
        "--block-order", "ffn_attn",
        "--attn-moe",
        "--n-attn-experts", "5",
        "--num-shared-experts", "2",
        "--n-sublayers", "3",
        "--expert-b-init", "zero",
    ])
    assert args.block_order == "ffn_attn"
    assert args.attn_moe is True
    assert args.n_attn_experts == 5
    assert args.num_shared_experts == 2
    assert args.n_sublayers == 3
    assert args.expert_b_init == "zero"


def test_cli_parser_block_config_defaults():
    from train_gpt import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.block_order == "attn_ffn"
    assert args.attn_moe is False
    assert args.num_shared_experts == 0
    assert args.n_sublayers == 1
    assert args.expert_b_init == "small"
