"""Task 7 — M0 LM-scaffold trainer smoke + unit tests.

The smoke test runs ``train_gpt.main`` end-to-end on CPU with tiny dims and a
synthetic in-memory shard (no big dataset required) and asserts an int6 artifact
is written. The unit tests cover the pieces the smoke can't isolate cheaply:
optimizer param coverage (every trainable param in exactly one group) and the
finite-horizon hinge loss form.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_m0_trainer_smoke(tmp_path):
    from train_gpt import main

    main(["--iterations", "3", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8", "--head-dim", "8",
          "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--artifact-out", str(tmp_path / "m.bin")])
    assert (tmp_path / "m.bin").exists()
    assert (tmp_path / "m.bin").stat().st_size > 0


def test_m0_learns_and_init_is_sane():
    """Regression: the model must (a) START near-uniform and (b) LEARN.

    Guards two coupled bugs that made a 100-step GPU smoke stall at ~27 nats
    (~4x WORSE than uniform ``ln(vocab)=6.9``) and never descend:

      1. ``z_K`` (the additive-coupling reversible midpoint) grows with depth and
         was fed UNNORMALIZED into the MoS ``tanh(ctx(z_K))`` head, saturating it
         so its gradient -> 0 and blocking learning everywhere upstream. Fixed by
         an RMSNorm of ``z_K`` at readout (OUTSIDE the recurrence, so
         reversibility is untouched).
      2. The tied token embedding used default ``nn.Embedding`` init (~N(0,1)),
         giving huge initial logits and loss ~27 at init. Fixed by small-std init.

    Task: deterministic copy (``targets == tokens``) on a small vocab — learnable
    in a few dozen steps with tied embeddings, so a healthy model drives the loss
    down sharply. CPU, seeded, fast. ``model_dim`` is kept large enough (128) that
    the bad init's oversized ``ctx(z_K)`` actually saturates the head's tanh — the
    failure mode the tiny ``model_dim=32`` config is too small to surface.
    """
    from train_gpt import Hyperparameters, M0GPT

    torch.manual_seed(0)
    vocab = 64
    args = Hyperparameters(
        model_dim=128, n_heads=4, n_kv_heads=2, vocab_size=vocab,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    model = M0GPT(args)

    # Deterministic learnable next-token task: copy the input token.
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, vocab, (4, 16), generator=gen)
    y = x.clone()
    depth = 4

    log_vocab = math.log(vocab)
    with torch.no_grad():
        initial_loss = model(x, y, depth).item()

    # (b-init) Init must be sane: near-uniform, NOT ~4x worse than uniform.
    assert initial_loss < 2 * log_vocab, (
        f"initial loss {initial_loss:.3f} >= 2*ln(vocab)={2*log_vocab:.3f}; "
        "embedding/head init is too large (regression in init)"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = model(x, y, depth)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_loss = model(x, y, depth).item()

    # (a) The model must actually learn: a clear margin of descent.
    assert final_loss < 0.6 * initial_loss, (
        f"loss did not descend: initial={initial_loss:.3f} final={final_loss:.3f} "
        "(gradient is being blocked — likely tanh saturation from unnormalized z_K)"
    )
    # And the start point is sane in absolute terms (near or below ~ln(vocab)).
    assert initial_loss < 3 * log_vocab


def _tiny_args():
    from train_gpt import Hyperparameters
    return Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )


def test_optimizer_param_coverage():
    """Every trainable parameter lands in exactly one optimizer group."""
    from train_gpt import M0GPT, build_optimizers

    model = M0GPT(_tiny_args())
    optimizers = build_optimizers(model, matrix_lr=0.02, embed_lr=0.1, scalar_lr=0.02)

    seen = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in seen, "parameter appears in more than one optimizer group"
                seen[pid] = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    # Account for tied weights (tok_emb.weight IS mos_head.out_embed.weight).
    trainable_ids = {id(p) for p in trainable}
    assert trainable_ids == set(seen.keys()), (
        "optimizer groups do not exactly cover the trainable parameters"
    )


def test_moe_expert_banks_routed_to_muon():
    """The 3-D MoE expert banks (w_in/w_out) belong to Muon, not AdamW.

    These banks are the model's effective-depth basis (~37% of params). They must
    be orthogonalized by the (batched) Newton-Schulz path, so they belong to the
    Muon group and must NOT appear in any AdamW group. Embeddings + scalars +
    router/gate stay in AdamW.
    """
    from train_gpt import M0GPT, Muon, build_optimizers, SwiGLUMoE

    model = M0GPT(_tiny_args())
    optimizers = build_optimizers(model, matrix_lr=0.02, embed_lr=0.1, scalar_lr=0.02)

    muon_ids, adamw_ids = set(), set()
    for opt in optimizers:
        target = muon_ids if isinstance(opt, Muon) else adamw_ids
        for group in opt.param_groups:
            for p in group["params"]:
                target.add(id(p))

    bank_ids = set()
    for m in model.modules():
        if isinstance(m, SwiGLUMoE):
            assert m.w_in.ndim == 3 and m.w_out.ndim == 3
            bank_ids.add(id(m.w_in))
            bank_ids.add(id(m.w_out))
    assert bank_ids, "no SwiGLUMoE expert banks found"

    for pid in bank_ids:
        assert pid in muon_ids, "expert bank not routed to Muon"
        assert pid not in adamw_ids, "expert bank leaked into AdamW"

    # Coverage stays exact: every trainable param in exactly one group.
    assert not (muon_ids & adamw_ids), "param in both Muon and AdamW"
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable_ids == (muon_ids | adamw_ids)


def test_adamw_params_excludes_muon_for_gradclip():
    """Grad-clip scope (Fix I2) = AdamW params only; excludes all Muon params.

    ``adamw_params`` is the clip/sync scope: every trainable param not managed by
    Muon. It must exclude the Muon matrix/expert-bank params (whose grads are
    unsynced per-rank) and include exactly the embeddings + scalars + router/gate.
    """
    from train_gpt import M0GPT, Muon, build_optimizers, adamw_params

    model = M0GPT(_tiny_args())
    optimizers = build_optimizers(model, matrix_lr=0.02, embed_lr=0.1, scalar_lr=0.02)

    muon_ids = {
        id(p) for opt in optimizers if isinstance(opt, Muon)
        for g in opt.param_groups for p in g["params"]
    }
    assert muon_ids, "expected a non-empty Muon group"

    clip_scope = adamw_params(model, optimizers)
    clip_ids = {id(p) for p in clip_scope}

    # No Muon param is in the clip scope.
    assert not (clip_ids & muon_ids)
    # Clip scope + Muon = all trainable params (exact partition).
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert clip_ids | muon_ids == trainable_ids
    # Embeddings are in the clip scope (AdamW), not Muon.
    assert id(model.tok_emb.weight) in clip_ids
    assert id(model.pos_emb) in clip_ids


def test_batched_newtonschulz_orthogonalizes_each_slice():
    """Batched NS on a (E, M, N) stack orthogonalizes each matrix independently.

    Matches the 2-D NS behavior on each slice: after orthogonalization the
    nonzero singular values cluster near 1 (the polar factor of each matrix),
    and the batched output equals the per-slice 2-D NS within bf16 tolerance.
    """
    from train_gpt import (
        zeropower_via_newtonschulz5,
        zeropower_via_newtonschulz5_batched,
    )

    torch.manual_seed(0)
    G = torch.randn(3, 8, 8)
    out = zeropower_via_newtonschulz5_batched(G, steps=5)
    assert out.shape == G.shape
    for i in range(G.shape[0]):
        s = torch.linalg.svdvals(out[i].float())
        # Square full-rank input -> singular values pulled into the Muon NS band
        # around 1 (5 NS iters cluster them in roughly [0.6, 1.4], not exactly 1).
        assert torch.all((s - 1.0).abs() < 0.4), f"slice {i} svals={s}"

    # Each batched slice matches the standalone 2-D NS (same formula/coeffs).
    # Algebraically identical (fp64 diff ~1e-15); the band here is bf16
    # accumulation-order noise on tiny matrices, not an implementation gap.
    ref = torch.stack([zeropower_via_newtonschulz5(G[i], steps=5) for i in range(3)])
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 0.1, f"batched vs per-slice max_diff={diff}"

    # Scale invariance: rescaling one slice must not change another's output.
    G2 = G.clone()
    G2[1] *= 500.0
    out2 = zeropower_via_newtonschulz5_batched(G2, steps=5)
    for i in (0, 2):
        d = (out[i].float() - out2[i].float()).abs().max().item()
        assert d < 0.1, f"slice {i} affected by rescaling slice 1: diff={d}"


def test_finite_horizon_hinge_loss():
    """Loss = L_hi + lambda_h*relu(L_hi - sg(L_lo) + margin) + lambda_route*aux_lb.

    sg(L_lo): the hinge must not backprop into the shallow pass; only L_hi and
    aux carry gradients into the params besides the hinge's L_hi term. ``aux_lb``
    is the ReMoE load-balanced sparsity term (not a plain mean|route| L1).
    """
    from train_gpt import M0GPT, finite_horizon_loss

    torch.manual_seed(0)
    model = M0GPT(_tiny_args())
    x = torch.randint(0, 1024, (2, 16))
    y = torch.randint(0, 1024, (2, 16))

    loss, parts = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.01,
    )
    assert torch.isfinite(loss)
    # Reconstruct the documented form from the reported parts.
    expected = (
        parts["l_hi"]
        + 0.5 * torch.relu(parts["l_hi"] - parts["l_lo_sg"] + 0.0)
        + 0.01 * parts["aux_lb"]
    )
    assert torch.allclose(loss, expected, atol=1e-5)
    # L_lo inside the hinge must be stop-gradient: parts["l_lo_sg"] is detached.
    assert not parts["l_lo_sg"].requires_grad

    # use_load_balance=False drops the aux term entirely (explicit ablation).
    loss_no_lb, parts_no_lb = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.01,
        use_load_balance=False,
    )
    expected_no_lb = (
        parts_no_lb["l_hi"]
        + 0.5 * torch.relu(parts_no_lb["l_hi"] - parts_no_lb["l_lo_sg"] + 0.0)
    )
    assert torch.allclose(loss_no_lb, expected_no_lb, atol=1e-5)


def test_int6_artifact_dedups_tied_embedding():
    """Fix M1: the tied embedding is serialized once, not twice.

    ``tok_emb.weight`` and ``mos_head.out_embed.weight`` are the SAME Parameter.
    The int6 artifact must store the underlying tensor only once (dedup by
    ``data_ptr``); the duplicate key carries no payload tensor of its own. The
    strict ``load_state_dict`` round-trip must still succeed (both keys present
    and equal after load).
    """
    from train_gpt import (
        M0GPT, save_int6_artifact, load_int6_artifact,
    )

    model = M0GPT(_tiny_args())
    sd = model.state_dict()
    assert sd["tok_emb.weight"].data_ptr() == sd["mos_head.out_embed.weight"].data_ptr()

    compressed, qsd, meta = save_int6_artifact(sd)

    # The duplicate key stores no payload of its own (deduped). Whichever of the
    # tied pair is serialized, exactly one carries the data; the other aliases.
    has_payload = lambda k: (k in qsd) or (k + ".q" in qsd)
    payload_keys = [k for k in ("tok_emb.weight", "mos_head.out_embed.weight")
                    if has_payload(k)]
    assert len(payload_keys) == 1, f"tied embed stored {len(payload_keys)} times"

    # The deduped artifact is smaller than a naive double-store would be.
    # Round-trip back to a strict state_dict load.
    deq = load_int6_artifact(compressed, sd)
    assert "tok_emb.weight" in deq and "mos_head.out_embed.weight" in deq
    assert torch.equal(deq["tok_emb.weight"], deq["mos_head.out_embed.weight"])

    fresh = M0GPT(_tiny_args())
    missing, unexpected = fresh.load_state_dict(deq, strict=True)
    assert not missing and not unexpected


def test_control_experiment_script_shape():
    import subprocess
    txt = open("experiments/run_m0_control_experiments.sh").read()
    for s in ("router_type=relu", "router_type=softmax", "for r in 1 2 4 8", "kv_latent"):
        assert s in txt
    subprocess.run(["bash", "-n", "experiments/run_m0_control_experiments.sh"], check=True)


def test_control_experiment_script_has_architecture_search_sweeps():
    """The runner sweeps every configurable recurrence-block axis with a `?`-safe
    summary (so a missing metric line never aborts the run)."""
    txt = open("experiments/run_m0_control_experiments.sh").read()
    for s in (
        "--block-order", "BLOCK_ORDER_SWEEP", "ffn_attn", "parallel",
        "--attn-moe", "ATTN_MOE_SWEEP",
        "--num-shared-experts", "SHARED_EXPERTS_SWEEP",
        "--n-sublayers", "LAYOUT_SWEEP",
        "--expert-b-init", "EXPERT_B_INIT_SWEEP",
    ):
        assert s in txt, f"missing architecture-search sweep token: {s}"
    # `?`-fallback summary present (EXPERIENCE.md#explicit-boundary).
    assert "${bpb:-?}" in txt


# ---------------------------------------------------------------------------
# ReMoE adaptive sparsity controller + load-balanced aux (anti-collapse fix)
# ---------------------------------------------------------------------------
def test_update_lambda_route_rule_and_clamp():
    """Unit-test the ReMoE adaptive-lambda update rule.

        lambda *= alpha ** sign(S_measured - S_target)

    So MORE sparse than target (S_measured > S_target) => lambda DECREASES
    (relax the penalty so experts re-activate); LESS sparse than target
    (S_measured < S_target) => lambda INCREASES (push toward target sparsity).
    The scalar is clamped to a sane range so it never runs away or vanishes.
    """
    from train_gpt import _update_lambda_route

    lam0 = 1e-3
    alpha = 1.2
    # Too sparse (S=0.9 > target 0.5) -> lambda decreases.
    dn = _update_lambda_route(lam0, s_measured=0.9, s_target=0.5, alpha=alpha)
    assert dn < lam0
    assert abs(dn - lam0 / alpha) < 1e-12
    # Too dense (S=0.1 < target 0.5) -> lambda increases.
    up = _update_lambda_route(lam0, s_measured=0.1, s_target=0.5, alpha=alpha)
    assert up > lam0
    assert abs(up - lam0 * alpha) < 1e-12
    # On target (sign 0) -> unchanged.
    same = _update_lambda_route(lam0, s_measured=0.5, s_target=0.5, alpha=alpha)
    assert abs(same - lam0) < 1e-12
    # Clamp high: repeated increases saturate at hi.
    lam = 1.0
    for _ in range(200):
        lam = _update_lambda_route(lam, s_measured=0.0, s_target=0.5, alpha=alpha,
                                   lo=1e-8, hi=1e3)
    assert lam <= 1e3 + 1e-9
    # Clamp low: repeated decreases saturate at lo.
    lam = 1.0
    for _ in range(400):
        lam = _update_lambda_route(lam, s_measured=1.0, s_target=0.5, alpha=alpha,
                                   lo=1e-8, hi=1e3)
    assert lam >= 1e-8 - 1e-12


def test_moe_aux_lb_is_load_balanced_form():
    """SwiGLUMoE exposes aux_lb = mean_e( f_e * mean_t route_{t,e} ).

    f_e is per-expert relative usage (fraction of tokens with nonzero weight
    on expert e). An expert that is ALWAYS used (high f_e) AND carries large
    mass is penalized more than one used rarely, unlike a plain mean|route|.
    """
    from train_gpt import SwiGLUMoE

    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="relu")
    x = torch.randn(2, 8, 16)
    moe(x)
    aux_lb = moe.aux_l1_loadbalanced()  # in-graph, recomputed from saved input
    assert aux_lb is not None
    assert torch.isfinite(aux_lb)
    # last_sparsity is set for the controller: fraction of route weights == 0.
    assert moe.last_sparsity is not None
    s = float(moe.last_sparsity)
    assert 0.0 <= s <= 1.0

    # Recompute the load-balanced form from last_route and check agreement.
    route = moe.last_route  # (B, T, E), detached
    f_e = (route > 0).float().mean(dim=(0, 1))      # per-expert usage fraction
    mean_mass = route.mean(dim=(0, 1))              # per-expert mean mass
    expected = (f_e * mean_mass).mean()
    assert torch.allclose(aux_lb.detach(), expected, atol=1e-5)


def test_controller_drives_sparsity_to_target_and_prevents_collapse():
    """The ReMoE controller is a NEGATIVE-feedback loop that holds sparsity at
    the target and never lets it run away to total collapse (sparsity -> 1).

    Models the controller against a monotone "router responds to lambda" plant:
    higher lambda => more sparsity (the L1 pressure the real router feels). This
    is the dynamical core of the fix — a FIXED lambda (the bug) has NO feedback,
    so any positive pressure ratchets sparsity to 1.0 (active_frac -> 0); the
    adaptive lambda pulls sparsity back toward S* from BOTH sides. We assert the
    closed loop converges near S* and stays bounded away from full collapse.
    """
    from train_gpt import _update_lambda_route

    def plant_sparsity(lam):
        # Monotone increasing in lambda, saturating in [0, 1): a faithful sign
        # for the controller (more penalty -> more zeros). Exact form is
        # irrelevant; only monotonicity + range matter for the feedback proof.
        return 1.0 - 1.0 / (1.0 + lam)

    s_target = 0.5            # target sparsity (active_frac target 0.5)
    # (a) Fixed lambda well above the equilibrium -> NO feedback -> over-sparse,
    #     i.e. the collapse regime the bug lives in.
    s_fixed = plant_sparsity(lam=50.0)
    assert s_fixed > 0.9, "sanity: a large fixed lambda over-sparsifies (collapse)"

    # (b) Adaptive lambda: closed loop converges to the target sparsity and the
    #     realized active_frac is held near 0.5 (NOT collapsed to ~0).
    lam = 1e-3
    sparsity = plant_sparsity(lam)
    for _ in range(200):
        lam = _update_lambda_route(lam, s_measured=sparsity, s_target=s_target,
                                   alpha=1.2)
        sparsity = plant_sparsity(lam)
    assert abs(sparsity - s_target) < 0.1, (
        f"closed loop did not reach target sparsity: {sparsity:.4f}"
    )
    active_frac = 1.0 - sparsity
    assert active_frac > 0.05, f"controller let the MoE collapse: af={active_frac:.4f}"
    assert 0.25 <= active_frac <= 0.75
    assert 1e-8 <= lam <= 1e3


def test_adaptive_controller_keeps_relu_moe_alive():
    """End-to-end on a real tiny ReLU-router M0GPT: training under the adaptive
    controller keeps the MoE alive (active_frac well above 0) and holds it near
    the configured target, while the load-balanced aux is finite throughout.

    This exercises the exact train-loop wiring (finite_horizon_loss with the
    load-balanced aux + per-step measured_sparsity + _update_lambda_route) end to
    end, so a regression that re-collapses the router (or wires the controller in
    backwards) fails here, not just in the synthetic feedback test above.
    """
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss,
        active_expert_fraction, measured_sparsity, _update_lambda_route,
        SwiGLUMoE,
    )

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
        n_experts=8, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, router_type="relu", moe_target_active_frac=0.5,
    )
    model = M0GPT(args)

    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 64, (4, 16), generator=gen)
    y = x.clone()

    def _active_frac():
        fr = [active_expert_fraction(m) for m in model.modules()
              if isinstance(m, SwiGLUMoE) and m.last_route is not None]
        return sum(fr) / len(fr)

    target = args.moe_target_active_frac
    s_target = 1.0 - target
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    lambda_route = 1e-3
    traj = []
    for _ in range(70):
        opt.zero_grad(set_to_none=True)
        loss, parts = finite_horizon_loss(
            model, x, y, k_hi=4, k_lo=2, lambda_h=0.1, margin=0.0,
            lambda_route=lambda_route, use_load_balance=True)
        assert torch.isfinite(parts["aux_lb"])
        loss.backward()
        opt.step()
        s_meas = measured_sparsity(model)
        lambda_route = _update_lambda_route(
            lambda_route, s_measured=s_meas, s_target=s_target, alpha=1.2)
        traj.append(_active_frac())

    final = traj[-1]
    # Crucially: NOT dead (the collapse bug drove this to ~0).
    assert final > 0.05, f"controller failed to keep MoE alive: active_frac={final:.4f}"
    # And it stays near the configured target active fraction.
    assert target - 0.25 <= final <= target + 0.25, (
        f"active_frac {final:.4f} not within [{target-0.25:.2f}, {target+0.25:.2f}]"
    )
    # lambda stayed in the sane clamp range throughout.
    assert 1e-8 <= lambda_route <= 1e3


# ---------------------------------------------------------------------------
# Composable router auxiliaries (entropy / Switch load-balance / ReMoE-adaptive)
# collapse-prevention bake-off. All default OFF / 0 so existing behavior holds.
# ---------------------------------------------------------------------------
def test_router_entropy_uniform_vs_onehot_softmax():
    """SwiGLUMoE.router_entropy() is the mean-token entropy of the router dist.

    A uniform router dist over E experts has entropy ~ ln(E); a one-hot dist has
    entropy ~ 0. We force the two regimes by saturating the softmax logits with
    very large weights into a constant pattern.
    """
    from train_gpt import SwiGLUMoE

    torch.manual_seed(0)
    E = 4
    moe = SwiGLUMoE(dim=16, n_experts=E, expert_rank=8, router_type="softmax")
    x = torch.randn(2, 8, 16)

    # Uniform: zero the router so every logit is the bias (equal) -> softmax uniform.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
    moe(x)
    h_uniform = moe.router_entropy().detach()
    assert torch.isfinite(h_uniform)
    assert abs(float(h_uniform) - math.log(E)) < 1e-4, float(h_uniform)

    # One-hot: a huge bias on expert 0 saturates softmax onto it -> entropy ~ 0.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        moe.router.bias[0] = 1e4
    moe(x)
    h_onehot = moe.router_entropy().detach()
    assert float(h_onehot) < 1e-3, float(h_onehot)
    assert float(h_uniform) > float(h_onehot)


def test_router_entropy_relu_normalized_uniform_vs_onehot():
    """For the relu router the entropy is over the route renormalized to a dist.

    Tokens whose route sums to 0 contribute 0 (skipped), so an all-zero route is
    handled gracefully. Uniform positive route -> ln(E); one expert only -> ~0.
    """
    from train_gpt import SwiGLUMoE

    E = 4
    moe = SwiGLUMoE(dim=8, n_experts=E, expert_rank=4, router_type="relu")

    # Uniform positive route: inject a constant positive route directly.
    route_uniform = torch.ones(2, 8, E)
    h_uniform = moe._route_entropy(route_uniform)
    assert abs(float(h_uniform) - math.log(E)) < 1e-5, float(h_uniform)

    # One-hot route: all mass on expert 0.
    route_onehot = torch.zeros(2, 8, E)
    route_onehot[..., 0] = 1.0
    h_onehot = moe._route_entropy(route_onehot)
    assert float(h_onehot) < 1e-5, float(h_onehot)

    # All-zero route rows are skipped (no NaN), entropy over the (empty) live set
    # is 0 rather than NaN.
    route_zero = torch.zeros(2, 8, E)
    h_zero = moe._route_entropy(route_zero)
    assert torch.isfinite(h_zero)
    assert float(h_zero) == 0.0


def test_load_balance_term_balanced_lower_than_imbalanced():
    """Switch load-balance term: coef*E*sum_e f_e*P_e.

    Balanced usage (every expert equally likely top + equal prob mass) gives the
    minimal value (== 1 before the E scaling for a perfectly uniform dist over E
    experts); a fully imbalanced dist (all tokens to one expert) gives the max
    (== E for one expert carrying everything). We assert the ORDERING.
    """
    from train_gpt import SwiGLUMoE

    E = 4
    moe = SwiGLUMoE(dim=8, n_experts=E, expert_rank=4, router_type="softmax")

    balanced = torch.full((2, 8, E), 1.0 / E)            # uniform dist per token
    lb_balanced = moe._load_balance(balanced)
    # Perfectly balanced: f_e = 1/E (argmax ties broken to index 0 -> all on 0;
    # so use a slightly perturbed near-uniform that spreads argmax). Use a
    # rotation so each token's argmax lands on a different expert.
    rot = torch.zeros(2, 8, E)
    for t in range(8):
        probs = torch.full((E,), (1.0 - 0.4) / (E - 1))
        probs[t % E] = 0.4
        rot[:, t, :] = probs
    lb_spread = moe._load_balance(rot)

    imbalanced = torch.zeros(2, 8, E)
    imbalanced[..., 0] = 1.0                              # all tokens to expert 0
    lb_imbalanced = moe._load_balance(imbalanced)

    assert torch.isfinite(lb_balanced) and torch.isfinite(lb_imbalanced)
    # Fully imbalanced is the maximum (== E); spread/balanced are below it.
    assert float(lb_imbalanced) > float(lb_spread)
    assert abs(float(lb_imbalanced) - E) < 1e-4
    # The Switch term is differentiable through P_e (the prob mass factor).
    w = torch.full((2, 8, E), 1.0 / E, requires_grad=True)
    lb = moe._load_balance(w)
    lb.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_moe_exposes_entropy_and_loadbalance_after_forward():
    """After a forward, the MoE exposes in-graph entropy + load-balance terms.

    router_entropy() and load_balance_term() must be differentiable (in the
    autograd graph) so the train loop can add them as loss terms. They train the
    ROUTER (the regularizers' target): the gradient reaches the router weights.
    """
    from train_gpt import SwiGLUMoE

    torch.manual_seed(0)
    moe = SwiGLUMoE(dim=16, n_experts=4, expert_rank=8, router_type="relu")
    x = torch.randn(2, 8, 16)
    moe(x)
    h = moe.router_entropy()
    lb = moe.load_balance_term()
    assert h.requires_grad and lb.requires_grad
    (h + lb).backward()
    # The router aux trains the router parameters (recomputed from the saved
    # detached block input), so the router weight receives a finite gradient.
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()


def test_collect_router_aux_sums_over_blocks():
    """The collectors sum router entropy / load-balance over MoE blocks."""
    from train_gpt import (
        Hyperparameters, M0GPT, _collect_router_entropy, _collect_load_balance,
        SwiGLUMoE,
    )

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, router_type="relu",
    )
    model = M0GPT(args)
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    model(x, y, 4)
    n_moe = sum(1 for m in model.modules() if isinstance(m, SwiGLUMoE))
    assert n_moe >= 1
    ent = _collect_router_entropy(model)
    lb = _collect_load_balance(model)
    assert ent.requires_grad and lb.requires_grad
    assert torch.isfinite(ent) and torch.isfinite(lb)


def test_finite_horizon_loss_adds_entropy_and_loadbalance_terms():
    """finite_horizon_loss composes the entropy + Switch load-balance auxiliaries.

        L = base + (-entropy_coef * H) + loadbalance_coef * LB
    with H maximized (so we SUBTRACT it) and LB minimized.
    """
    from train_gpt import M0GPT, finite_horizon_loss

    torch.manual_seed(0)
    model = M0GPT(_tiny_args())
    x = torch.randint(0, 1024, (2, 16))
    y = torch.randint(0, 1024, (2, 16))

    # Baseline (all aux off).
    loss0, parts0 = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.0,
        use_load_balance=False, entropy_coef=0.0, loadbalance_coef=0.0,
    )
    # With entropy + load-balance.
    loss1, parts1 = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.0,
        use_load_balance=False, entropy_coef=0.3, loadbalance_coef=0.2,
    )
    assert torch.isfinite(loss1)
    # Reconstruct the documented form from the reported parts.
    base = parts1["l_hi"] + 0.5 * torch.relu(parts1["l_hi"] - parts1["l_lo_sg"] + 0.0)
    expected = base - 0.3 * parts1["router_entropy"] + 0.2 * parts1["load_balance"]
    assert torch.allclose(loss1, expected, atol=1e-5)
    # parts expose the aux quantities for logging.
    for k in ("router_entropy", "load_balance"):
        assert k in parts0 and k in parts1
        assert torch.isfinite(parts1[k])


def test_finite_horizon_loss_adds_router_z_loss_term():
    """finite_horizon_loss composes the ST-MoE z-loss when z_coef>0:

        L = base + z_coef * Z,   Z = mean(logsumexp(router_logits)^2).
    """
    from train_gpt import M0GPT, finite_horizon_loss

    torch.manual_seed(0)
    model = M0GPT(_tiny_args())
    x = torch.randint(0, 1024, (2, 16))
    y = torch.randint(0, 1024, (2, 16))

    # z-loss off -> reported router_z is an in-graph 0.0; loss == base.
    loss0, parts0 = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.0,
        use_load_balance=False, entropy_coef=0.0, loadbalance_coef=0.0, z_coef=0.0,
    )
    assert "router_z" in parts0 and float(parts0["router_z"]) == 0.0
    # z-loss on -> loss == base + z_coef * Z, and Z > 0 (the experts are live).
    loss1, parts1 = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=0.5, margin=0.0, lambda_route=0.0,
        use_load_balance=False, entropy_coef=0.0, loadbalance_coef=0.0, z_coef=0.5,
    )
    base = parts1["l_hi"] + 0.5 * torch.relu(parts1["l_hi"] - parts1["l_lo_sg"] + 0.0)
    expected = base + 0.5 * parts1["router_z"]
    assert torch.allclose(loss1, expected, atol=1e-5)
    assert torch.isfinite(parts1["router_z"]) and float(parts1["router_z"].detach()) > 0.0


def test_router_bias_update_runs_at_step_boundary_in_trainer():
    """The DeepSeek-V3 step-boundary bias update (update_router_biases) moves the
    recurrence MoEs' router_bias away from zero over a few real optimizer steps,
    and the bias stays a buffer (not a Parameter) throughout."""
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss, update_router_biases,
        _RouterBiasMixin,
    )

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=24, n_heads=2, n_kv_heads=1, vocab_size=64, n_experts=8,
        expert_rank=8, n_mix=2, kv_latent=8, head_dim=8, max_seq_len=16,
        router_type="softmax", router_bias_update_rate=0.05,
    )
    model = M0GPT(args)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(1)
    moved = False
    for _ in range(8):
        x = torch.randint(0, 64, (2, 8), generator=gen)
        y = torch.randint(0, 64, (2, 8), generator=gen)
        opt.zero_grad(set_to_none=True)
        loss, _ = finite_horizon_loss(
            model, x, y, k_hi=4, k_lo=2, lambda_h=0.0, margin=0.0,
            lambda_route=0.0, use_load_balance=False, z_coef=1e-3)
        loss.backward()
        opt.step()
        update_router_biases(model)  # the step-boundary update
    param_ids = {id(p) for p in model.parameters()}
    for m in model._recurrence_moes():
        assert id(m.router_bias) not in param_ids  # still a buffer
        if float(m.router_bias.abs().sum()) > 0.0:
            moved = True
    assert moved, "router_bias never moved off zero — step-boundary update inert"


def test_target_active_frac_gates_adaptive_controller():
    """--router-target-active-frac > 0 (relu) enables the controller; 0 disables.

    Exercised through the CLI builder + the small main() loop semantics: with the
    knob at 0 the adaptive controller is OFF; with it > 0 and relu it is ON.
    """
    from train_gpt import build_arg_parser

    p = build_arg_parser()
    # Default: disabled.
    a0 = p.parse_args(["--router-type", "relu"])
    assert a0.router_target_active_frac == 0.0
    # Enabled.
    a1 = p.parse_args(["--router-type", "relu", "--router-target-active-frac", "0.4"])
    assert a1.router_target_active_frac == 0.4
    # The DEFAULT collapse-preventer is now DeepSeek-V3 loss-free balancing
    # (router-bias-update-rate>0) + ST-MoE z-loss (router-z-coef>0). The
    # entropy-MAX term is DEMOTED to OFF-by-default; load-balance stays OFF. All
    # remain CLI knobs.
    assert a0.router_bias_update_rate > 0.0
    assert a0.router_z_coef > 0.0
    assert a0.router_entropy_coef == 0.0
    assert a0.router_loadbalance_coef == 0.0
    a2 = p.parse_args(["--router-entropy-coef", "0.01", "--router-loadbalance-coef", "0.02"])
    assert a2.router_entropy_coef == 0.01
    assert a2.router_loadbalance_coef == 0.02
    # Entropy aux is still ENABLABLE via the CLI knob (ablation path).
    a3 = p.parse_args(["--router-entropy-coef", "0.1"])
    assert a3.router_entropy_coef == 0.1


def test_entropy_aux_keeps_relu_moe_alive_no_collapse():
    """A short relu-router training run with ONLY the entropy auxiliary (no ReMoE
    controller, fixed lambda_route=0) keeps active_frac well above 0 over ~60
    steps. The entropy MAX pressure spreads routing mass back across experts, so
    the ReLU router does not collapse to all-zero (active_frac -> 0).
    """
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss,
        active_expert_fraction, SwiGLUMoE,
    )

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
        n_experts=8, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, router_type="relu",
    )
    model = M0GPT(args)
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 64, (4, 16), generator=gen)
    y = x.clone()

    def _active_frac():
        fr = [active_expert_fraction(m) for m in model.modules()
              if isinstance(m, SwiGLUMoE) and m.last_route is not None]
        return sum(fr) / len(fr)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    traj = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        # NO ReMoE controller (lambda_route=0, use_load_balance=False); entropy only.
        loss, _ = finite_horizon_loss(
            model, x, y, k_hi=4, k_lo=2, lambda_h=0.1, margin=0.0,
            lambda_route=0.0, use_load_balance=False,
            entropy_coef=0.05, loadbalance_coef=0.0)
        loss.backward()
        opt.step()
        traj.append(_active_frac())
    final = traj[-1]
    assert final > 0.05, f"entropy aux failed to keep MoE alive: active_frac={final:.4f}"


def test_softmax_entropy_aux_keeps_router_from_collapsing():
    """With softmax routing + the entropy auxiliary, the router entropy stays
    high (the dist does not collapse onto a single expert). We compare against a
    run with a strong ANTI-entropy pressure (negative coef) which DOES sharpen.
    """
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss, SwiGLUMoE,
    )

    def _run(entropy_coef):
        torch.manual_seed(0)
        args = Hyperparameters(
            model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
            n_experts=8, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            max_seq_len=16, router_type="softmax",
        )
        model = M0GPT(args)
        gen = torch.Generator().manual_seed(1)
        x = torch.randint(0, 64, (4, 16), generator=gen)
        y = x.clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            loss, _ = finite_horizon_loss(
                model, x, y, k_hi=4, k_lo=2, lambda_h=0.1, margin=0.0,
                lambda_route=0.0, use_load_balance=False,
                entropy_coef=entropy_coef, loadbalance_coef=0.0)
            loss.backward()
            opt.step()
        ents = [float(m.router_entropy().detach()) for m in model.modules()
                if isinstance(m, SwiGLUMoE) and m.last_route is not None]
        return sum(ents) / len(ents)

    h_with_entropy = _run(entropy_coef=0.1)
    h_anti_entropy = _run(entropy_coef=-0.1)
    # Maximizing entropy keeps the dist broad; the anti-entropy run sharpens it.
    assert h_with_entropy > h_anti_entropy, (h_with_entropy, h_anti_entropy)


def test_format_metrics_line_emits_router_entropy_and_expert_util():
    """The metrics: line carries router_entropy + expert_util when supplied."""
    from train_gpt import format_metrics_line

    line = format_metrics_line({
        "erank": 1.0, "peak_vram": 0.0, "kv_bytes": 8, "params": 10,
        "active_frac": 0.5, "router_entropy": 1.234, "expert_util": 2.5,
        "disp_tail": 0.42,
    })
    assert "router_entropy:1.2340" in line
    assert "expert_util:2.5000" in line
    assert "disp_tail:0.4200" in line
    assert "diag:active_frac:0.5000" in line


def test_format_metrics_line_emits_and_omits_moe_basis_diagnostics():
    """The metrics: line carries route_step_div + expert_cos_div WHEN supplied and
    OMITS them when absent (the MoE-basis-depth diagnostics)."""
    from train_gpt import format_metrics_line

    line = format_metrics_line({
        "erank": 1.0, "peak_vram": 0.0, "kv_bytes": 8, "params": 10,
        "route_step_div": 0.625, "expert_cos_div": 0.7777,
    })
    assert "route_step_div:0.6250" in line
    assert "expert_cos_div:0.7777" in line
    # Omitted when absent.
    line2 = format_metrics_line({
        "erank": 1.0, "peak_vram": 0.0, "kv_bytes": 8, "params": 10,
    })
    assert "route_step_div" not in line2
    assert "expert_cos_div" not in line2


# ---------------------------------------------------------------------------
# --grad-accum control + throughput (tok/s) & VRAM-utilization ops logging
# ---------------------------------------------------------------------------
def test_grad_accum_cli_default_and_override():
    """--grad-accum default 0 (auto); >0 overrides the per-step micro-count.

    The trainer underutilizes the GPU because the hardcoded auto value
    (max(8//world,1)=8 on 1 GPU) shrinks the per-forward micro-batch. The knob
    lets us set grad_accum=1 (whole batch_tokens in ONE forward) to fill VRAM.
    """
    from train_gpt import build_arg_parser

    p = build_arg_parser()
    assert p.parse_args([]).grad_accum == 0          # default: keep auto
    assert p.parse_args(["--grad-accum", "1"]).grad_accum == 1
    assert p.parse_args(["--grad-accum", "4"]).grad_accum == 4


def test_resolve_grad_accum_steps_auto_vs_override():
    """The resolver honors >0 override and falls back to max(8//world,1)."""
    from train_gpt import resolve_grad_accum_steps

    # Auto (knob == 0): keep the historical scaffold value.
    assert resolve_grad_accum_steps(0, world_size=1) == 8
    assert resolve_grad_accum_steps(0, world_size=2) == 4
    assert resolve_grad_accum_steps(0, world_size=16) == 1
    # Override (knob > 0): use it verbatim, independent of world_size.
    assert resolve_grad_accum_steps(1, world_size=1) == 1
    assert resolve_grad_accum_steps(2, world_size=1) == 2
    assert resolve_grad_accum_steps(3, world_size=8) == 3


def test_m0_trainer_smoke_grad_accum_one(tmp_path):
    """grad_accum=1: the whole batch_tokens is ONE forward per step (largest
    micro-batch / best throughput at a given VRAM). End-to-end CPU smoke."""
    from train_gpt import main

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--grad-accum", "1", "--artifact-out", str(tmp_path / "m1.bin")])
    assert (tmp_path / "m1.bin").exists()
    assert (tmp_path / "m1.bin").stat().st_size > 0


def test_m0_trainer_smoke_grad_accum_two(tmp_path):
    """grad_accum=2: two micro-steps accumulate per optimizer step. CPU smoke."""
    from train_gpt import main

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--grad-accum", "2", "--artifact-out", str(tmp_path / "m2.bin")])
    assert (tmp_path / "m2.bin").exists()
    assert (tmp_path / "m2.bin").stat().st_size > 0


def test_m0_trainer_k_eval_sweep_emits_depth_gain(tmp_path, capsys):
    """--k-eval-sweep prints one depth_sweep line per K plus a depth_gain_GT
    and a phi_eval line at the end of the run (the depth-gain MEASUREMENT)."""
    import re
    from train_gpt import main

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--k-set", "2,4", "--k-eval-sweep", "2,4,8",
          "--artifact-out", str(tmp_path / "ks.bin")])
    out = capsys.readouterr().out
    swept = {}
    for ln in out.splitlines():
        m = re.match(
            r"depth_sweep: K=(\d+) val_bpb:([-+0-9.eEnNaA]+) val_loss:([-+0-9.eEnNaA]+)",
            ln)
        if m:
            swept[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    assert set(swept) == {2, 4, 8}, f"missing depth_sweep lines: {swept}"
    # val_loss is real (finite) even on the synthetic smoke (bpb may be NaN).
    for k, (bpb, loss) in swept.items():
        assert loss == loss, f"K={k} val_loss is NaN"  # noqa: PLR0124
    m_gt = re.search(r"^depth_gain_GT:([-+0-9.eEnNaA]+)", out, re.MULTILINE)
    m_phi = re.search(r"^phi_eval:([-+0-9.eEnNaA]+)", out, re.MULTILINE)
    assert m_gt is not None, "no depth_gain_GT line"
    assert m_phi is not None, "no phi_eval line"
    # phi_eval is a finite proxy in [0, 1] (synthetic bpb is NaN but loss is real,
    # so fit_phi over the {K: val_loss} map is well-defined and clamped).
    phi = float(m_phi.group(1))
    assert 0.0 <= phi <= 1.0, phi


def test_init_state_cli_default_and_choices():
    """--init-state defaults to x0 (current behavior); random is the Huginn fix."""
    from train_gpt import build_arg_parser

    p = build_arg_parser()
    assert p.parse_args([]).init_state == "x0"
    assert p.parse_args(["--init-state", "random"]).init_state == "random"
    assert p.parse_args(["--init-state", "x0"]).init_state == "x0"


def test_init_state_x0_is_byte_identical_default():
    """init_state='x0' (default) reproduces the current forward/grad EXACTLY.

    A no-regression guard: switching the default Hyperparameter on must not move
    a single bit relative to a model built without the field set (the x0 seed
    path is unchanged)."""
    from train_gpt import Hyperparameters, M0GPT

    def _build():
        return M0GPT(Hyperparameters(
            model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
            n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            max_seq_len=16,
        ))

    torch.manual_seed(0)
    m_default = _build()
    torch.manual_seed(0)
    m_x0 = M0GPT(Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, init_state="x0",
    ))
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    l_default = m_default(x, y, depth=4)
    l_x0 = m_x0(x, y, depth=4)
    assert torch.equal(l_default.detach(), l_x0.detach())
    l_default.backward()
    g_default = {n: p.grad.clone() for n, p in m_default.named_parameters()}
    l_x0.backward()
    for n, p in m_x0.named_parameters():
        assert torch.allclose(p.grad, g_default[n], atol=1e-12), n


def test_init_state_random_changes_forward_and_trains():
    """init_state='random' seeds the recurrence with small random a0/b0 (not x0),
    which changes the forward output vs the x0 seed, and the model still trains
    (finite gradients to every param) under the random seed."""
    from train_gpt import Hyperparameters, M0GPT

    def _build(init_state):
        torch.manual_seed(0)
        return M0GPT(Hyperparameters(
            model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=64,
            n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            max_seq_len=16, init_state=init_state,
        ))

    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    m_x0 = _build("x0")
    m_rand = _build("random")
    # Same params (same seed); only the seed-state path differs -> different loss.
    torch.manual_seed(123)
    l_x0 = m_x0(x, y, depth=4)
    torch.manual_seed(123)
    l_rand = m_rand(x, y, depth=4)
    assert not torch.allclose(l_x0.detach(), l_rand.detach())
    # Trains: finite, non-zero gradients reach the params under the random seed.
    l_rand.backward()
    g_norm = sum(p.grad.norm().item() for p in m_rand.parameters() if p.grad is not None)
    assert g_norm > 0.0
    for p in m_rand.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_m0_trainer_init_state_random_smoke(tmp_path):
    """End-to-end CPU smoke with --init-state random: trains and writes an
    artifact (the Huginn path-independent init runs through the custom backward)."""
    from train_gpt import main

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--init-state", "random", "--k-set", "2,4",
          "--artifact-out", str(tmp_path / "ri.bin")])
    assert (tmp_path / "ri.bin").exists()
    assert (tmp_path / "ri.bin").stat().st_size > 0


def test_m0_trainer_step_conditioning_smoke(tmp_path):
    """End-to-end CPU smoke with --step-conditioning ON: trains and writes an
    artifact (the clock weight rides the optimizer + custom backward)."""
    from train_gpt import main

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--step-conditioning", "--max-step-emb", "8", "--k-set", "2,4",
          "--artifact-out", str(tmp_path / "sc.bin")])
    assert (tmp_path / "sc.bin").exists()
    assert (tmp_path / "sc.bin").stat().st_size > 0


def test_format_metrics_line_emits_throughput_and_vram_util_ops_stats():
    """The metrics: line carries tok_per_s + vram_util_pct as OPS/efficiency
    diagnostics (NOT the resource-GOAL peak-VRAM / R_act numbers). On CPU
    vram_util_pct is 0.0; tok_per_s is the per-step throughput."""
    from train_gpt import format_metrics_line

    line = format_metrics_line({
        "erank": 1.0, "peak_vram": 0.0, "kv_bytes": 8, "params": 10,
        "tok_per_s": 12345.0, "vram_util_pct": 0.0,
    })
    assert "ops:tok_per_s:12345.0000" in line
    assert "ops:vram_util_pct:0.0000" in line


def test_vram_util_pct_cpu_is_zero():
    """vram_util_pct is 0.0 when CUDA is unavailable (CPU log/test sites)."""
    from train_gpt import vram_util_pct

    # No device / CPU device -> 0.0, no crash.
    assert vram_util_pct(None) == 0.0
    assert vram_util_pct("cpu") == 0.0


def test_m0_trainer_smoke_emits_throughput_and_util(tmp_path, capsys):
    """A CPU run emits a metrics: line carrying tok_per_s (>0) and
    vram_util_pct (0.0 on CPU); the plot_metrics parser round-trips both."""
    import re
    from train_gpt import main
    from experiments.plot_metrics import parse_log

    main(["--iterations", "2", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8",
          "--head-dim", "8", "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--log-every", "1", "--artifact-out", str(tmp_path / "m.bin")])
    out = capsys.readouterr().out
    metrics_lines = [ln for ln in out.splitlines() if ln.startswith("metrics:")]
    assert metrics_lines, "no metrics: line emitted"
    last = metrics_lines[-1]
    m_tps = re.search(r"\btok_per_s:([-+0-9.eE]+)", last)
    m_util = re.search(r"\bvram_util_pct:([-+0-9.eE]+)", last)
    assert m_tps is not None, last
    assert m_util is not None, last
    assert float(m_tps.group(1)) > 0.0          # tokens/s is positive
    assert float(m_util.group(1)) == 0.0        # CPU -> 0% utilization

    # Parser round-trip.
    log_path = tmp_path / "run.log"
    log_path.write_text(out, encoding="utf-8")
    d = parse_log(str(log_path))
    assert any(v > 0.0 for v in d["tok_per_s"] if v == v), d["tok_per_s"]
    assert all((v == 0.0 or v != v) for v in d["vram_util_pct"]), d["vram_util_pct"]


# ---------------------------------------------------------------------------
# Bug fixes: router auxes must TRAIN the router (was a no-op) + non-inert MoE
# ---------------------------------------------------------------------------
def _train_m0_for_router(entropy_coef=0.0, loadbalance_coef=0.0, steps=60,
                         seed=0, vocab=64, n_experts=8):
    """Train a tiny softmax-router M0GPT and return (mean_router_entropy,
    mean_util_entropy) at the end. Shared driver so the entropy / load-balance
    regression tests differ only in the single coef under test."""
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss, SwiGLUMoE,
        _global_util_entropy,
    )

    torch.manual_seed(seed)
    args = Hyperparameters(
        model_dim=24, n_heads=2, n_kv_heads=1, vocab_size=vocab,
        n_experts=n_experts, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16, router_type="softmax",
    )
    model = M0GPT(args)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        x = torch.randint(0, vocab, (2, 8), generator=gen)
        y = torch.randint(0, vocab, (2, 8), generator=gen)
        opt.zero_grad(set_to_none=True)
        loss, _ = finite_horizon_loss(
            model, x, y, k_hi=4, k_lo=2, lambda_h=0.0, margin=0.0,
            lambda_route=0.0, use_load_balance=False,
            entropy_coef=entropy_coef, loadbalance_coef=loadbalance_coef)
        loss.backward()
        opt.step()
    moes = [m for m in model.modules()
            if isinstance(m, SwiGLUMoE) and m.last_route is not None]
    h = sum(float(m.router_entropy().detach()) for m in moes) / len(moes)
    util = sum(_global_util_entropy(m.last_route) for m in moes) / len(moes)
    return h, util


def test_router_and_w_in_receive_task_gradient_at_init():
    """DECISIVE root-cause guard. With ``w_out`` zero-initialized the MoE output
    is identically 0, so the task loss gradient to the router weights AND to
    ``w_in`` is EXACTLY zero at init: the router never engages and the bake-off's
    variants were bit-identical. After the non-inert init, both must receive a
    finite, non-zero task gradient from step 0 (no aux term involved).
    """
    from train_gpt import (
        Hyperparameters, M0GPT, finite_horizon_loss, SwiGLUMoE,
    )

    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=24, n_heads=2, n_kv_heads=1, vocab_size=64, n_experts=8,
        expert_rank=8, n_mix=2, kv_latent=8, head_dim=8, max_seq_len=16,
        router_type="softmax",
    )
    m = M0GPT(args)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    m.zero_grad(set_to_none=True)
    # Pure task loss — NO router aux (entropy/load-balance/L1 all off).
    loss, _ = finite_horizon_loss(
        m, x, y, k_hi=4, k_lo=2, lambda_h=0.0, margin=0.0, lambda_route=0.0,
        use_load_balance=False, entropy_coef=0.0, loadbalance_coef=0.0)
    loss.backward()
    for mod in m.modules():
        if isinstance(mod, SwiGLUMoE):
            assert mod.router.weight.grad is not None
            assert float(mod.router.weight.grad.norm()) > 0.0, (
                "router gets ZERO task gradient at init — MoE output is dead "
                "(zero-w_out regression)")
            assert float(mod.w_in.grad.norm()) > 0.0, (
                "w_in gets ZERO task gradient at init — experts are dead "
                "(zero-w_out regression)")


def test_entropy_aux_is_not_a_no_op_decisive_end_to_end():
    """DECISIVE regression guard the bake-off lacked: a real M0GPT trained WITH a
    large entropy coef must end with MEASURABLY HIGHER router entropy than the
    SAME-SEED run WITHOUT it.

    Root cause this guards: the MoE ``w_out`` was zero-initialized, so every
    expert output was 0; the router weights multiplied a zero, so the router
    received ZERO task gradient AND the tiny entropy-aux gradient rounded away in
    practice. Three router variants (entropy / load-balance / no-aux) were then
    BIT-IDENTICAL because the router never moved. With the non-inert init the
    entropy aux actually spreads routing mass — assert the gap is real.
    """
    h_with, _ = _train_m0_for_router(entropy_coef=0.1)
    h_without, _ = _train_m0_for_router(entropy_coef=0.0)
    margin = 0.3
    assert h_with > h_without + margin, (
        f"entropy aux is a NO-OP: with={h_with:.4f} without={h_without:.4f} "
        f"(gap {h_with - h_without:.4f} <= margin {margin}); the aux gradient is "
        "not reaching the router weights (regression of the zero-w_out / detach bug)"
    )


def test_loadbalance_aux_changes_expert_utilization():
    """A load-balance run must change realized expert utilization vs no-aux
    (same seed). The Switch term equalizes per-expert mass, so global utilization
    entropy rises measurably — proof the load-balance aux trains the router."""
    _, util_with = _train_m0_for_router(loadbalance_coef=1.0)
    _, util_without = _train_m0_for_router(loadbalance_coef=0.0)
    assert util_with > util_without + 0.15, (
        f"load-balance aux did not change utilization: with={util_with:.4f} "
        f"without={util_without:.4f}"
    )


def test_router_default_collapse_preventer_is_loss_free_balancing():
    """The DEFAULT collapse-preventer is now DeepSeek-V3 loss-free balancing +
    ST-MoE z-loss (NOT the demoted entropy-max). Assert the parser defaults: the
    bias-update-rate and z-coef are ON (>0) and the entropy-MAX coef is OFF (0.0).

    The entropy-max term MAXIMIZES per-token entropy and fights useful
    specialization; it did not prevent the measured 9.7M-scale collapse, so it is
    no longer the default. It stays a CLI ablation knob."""
    from train_gpt import build_arg_parser

    a = build_arg_parser().parse_args([])
    assert a.router_bias_update_rate > 0.0, (
        "DeepSeek loss-free balancing is OFF by default (collapse risk)")
    assert a.router_z_coef > 0.0, "ST-MoE z-loss is OFF by default (collapse risk)"
    assert a.router_entropy_coef == 0.0, (
        "entropy-MAX is still the default — it should be demoted to OFF")


def test_moe_is_non_inert_n_experts_changes_loss():
    """The MoE must DO WORK: two models differing ONLY in ``n_experts`` (with the
    SAME seed) must reach DIFFERENT losses after a few steps. If ``w_out`` were
    zero-initialized the MoE output is identically 0 and ``n_experts`` is inert —
    both models would track the same attention-only trajectory and collide.
    """
    from train_gpt import Hyperparameters, M0GPT

    def _final_loss(n_experts):
        torch.manual_seed(0)
        args = Hyperparameters(
            model_dim=32, n_heads=2, n_kv_heads=1, vocab_size=64,
            n_experts=n_experts, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            max_seq_len=16, router_type="softmax",
        )
        model = M0GPT(args)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        gen = torch.Generator().manual_seed(1)
        x = torch.randint(0, 64, (4, 16), generator=gen)
        y = x.clone()
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            loss = model(x, y, depth=4)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float(model(x, y, depth=4))

    l2 = _final_loss(n_experts=2)
    l8 = _final_loss(n_experts=8)
    assert abs(l2 - l8) > 1e-3, (
        f"MoE is INERT: n_experts=2 -> {l2:.5f}, n_experts=8 -> {l8:.5f} "
        f"(|diff|={abs(l2 - l8):.2e}); experts are not contributing to the output "
        "(regression of the zero-w_out init)"
    )


def test_moe_w_out_init_is_non_zero():
    """Unit guard: ``M0GPT`` must NOT zero-init the MoE ``w_out`` (that made each
    expert output 0 and the whole MoE inert). The attention ``o_proj`` stays
    near-zero for readout stability, but the experts must start with a small
    non-zero contribution so they receive meaningful gradient from step 0."""
    from train_gpt import Hyperparameters, M0GPT

    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )
    m = M0GPT(args)
    # The delta block is a stack of sub-blocks (default 1); each holds the
    # attn + FFN-MoE. Default config: small non-zero MoE w_out, zero attn o_proj.
    for blk in (m.rec.F, m.rec.G):
        for sub in blk.sublayers:
            for moe in sub.moe_modules():
                assert moe.w_out.abs().sum() > 0.0, "MoE w_out is zero-initialized (inert)"
            for mla in sub.attn_modules():
                # Attention o_proj remains zero (readout-stability near-identity).
                assert mla.o_proj.weight.abs().sum() == 0.0


# ---------------------------------------------------------------------------
# Fix 1 — PAIRED --k-eval-sweep depth-gain measurement.
#
# The sweep must hold depth K as the ONLY variable: identical eval batches AND
# (under --init-state random) identical recurrence-init noise across K. The
# instrument must therefore be REPRODUCIBLE: running main() twice on the SAME
# seed must yield the SAME depth_gain_GT, and two identical-K entries must score
# identically (paired data + paired noise).
# ---------------------------------------------------------------------------
def _parse_depth_sweep(stdout: str):
    """Extract (depth_sweep_map, depth_gain_GT, phi_eval) from main() stdout."""
    import math
    import re
    sweep = {}
    gt = None
    phi = None
    for line in stdout.splitlines():
        m = re.search(r"^depth_sweep:\s*K=(\d+)\s+val_bpb:(\S+)\s+val_loss:(\S+)", line)
        if m:
            sweep[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.search(r"^depth_gain_GT:(\S+)", line)
        if m:
            gt = float(m.group(1))
        m = re.search(r"^phi_eval:(\S+)", line)
        if m:
            phi = float(m.group(1))
    return sweep, gt, phi


def _run_main_capture(capsys, extra):
    """Run train_gpt.main on the tiny synthetic CPU path and return stdout."""
    from train_gpt import main
    base = ["--iterations", "2", "--model-dim", "32", "--n-heads", "4",
            "--n-kv-heads", "2", "--n-experts", "4", "--expert-rank", "8",
            "--n-mix", "2", "--kv-latent", "8", "--head-dim", "8",
            "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
            "--k-set", "2,4", "--k-lo", "2"]
    main(base + extra)
    return capsys.readouterr().out


def test_k_eval_sweep_emits_headline_metrics(capsys, tmp_path):
    """The sweep emits depth_sweep / depth_gain_GT / phi_eval, one line per K."""
    out = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4", "--artifact-out", str(tmp_path / "a.bin")])
    sweep, gt, phi = _parse_depth_sweep(out)
    assert set(sweep.keys()) == {2, 4}, sweep
    assert gt is not None and phi is not None


def test_k_eval_sweep_dedups_K_preserving_order(capsys, tmp_path):
    """A repeated K is scored once (dict.fromkeys dedup)."""
    out = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4,2", "--artifact-out", str(tmp_path / "a.bin")])
    n_lines = sum(1 for ln in out.splitlines() if ln.startswith("depth_sweep:"))
    assert n_lines == 2, f"expected 2 deduped depth_sweep lines, got {n_lines}"


def _depth_gain_loss(sweep):
    """Depth-gain on the FINITE val_loss column (val_bpb is NaN on the synthetic
    CPU path with no tokenizer, so the reproducibility instrument keys off the
    per-K val_loss, which exercises the identical paired pipeline)."""
    ks = sorted(sweep)
    return sweep[ks[0]][1] - sweep[ks[-1]][1]


def test_k_eval_sweep_depth_gain_is_reproducible_x0(capsys, tmp_path):
    """SAME seed -> SAME depth-gain (paired data; x0 init draws no noise)."""
    out1 = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4", "--seed", "123",
                 "--artifact-out", str(tmp_path / "a.bin")])
    out2 = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4", "--seed", "123",
                 "--artifact-out", str(tmp_path / "b.bin")])
    s1, _, _ = _parse_depth_sweep(out1)
    s2, _, _ = _parse_depth_sweep(out2)
    g1, g2 = _depth_gain_loss(s1), _depth_gain_loss(s2)
    assert g1 == g2, f"depth-gain not reproducible: {g1} vs {g2}"


def test_k_eval_sweep_depth_gain_is_reproducible_random_init(capsys, tmp_path):
    """HARD: with --init-state random the per-K RNG reset makes the instrument
    reproducible too — SAME seed -> SAME depth-gain — proving the paired noise
    handling holds depth as the only variable across K (val_loss column, since
    val_bpb is NaN on the synthetic no-tokenizer path)."""
    out1 = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4", "--seed", "77", "--init-state", "random",
                 "--artifact-out", str(tmp_path / "a.bin")])
    out2 = _run_main_capture(
        capsys, ["--k-eval-sweep", "2,4", "--seed", "77", "--init-state", "random",
                 "--artifact-out", str(tmp_path / "b.bin")])
    s1, _, _ = _parse_depth_sweep(out1)
    s2, _, _ = _parse_depth_sweep(out2)
    g1, g2 = _depth_gain_loss(s1), _depth_gain_loss(s2)
    assert g1 == g2, f"random-init depth-gain not reproducible: {g1} vs {g2}"


def test_replay_loader_yields_identical_batches_for_each_K():
    """The paired sweep's _ReplayLoader replays the SAME batches per K (so two
    fresh instances over the same cache yield identical (x, y))."""
    from train_gpt import _ReplayLoader
    cache = [(torch.randint(0, 8, (1, 4)), torch.randint(0, 8, (1, 4)))
             for _ in range(3)]
    a = _ReplayLoader(cache)
    b = _ReplayLoader(cache)
    for _ in range(3):
        xa, ya = a.next_batch(8, 4, 1)
        xb, yb = b.next_batch(8, 4, 1)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_replay_loader_under_random_init_gives_paired_noise(capsys, tmp_path):
    """With --init-state random, two identical-K sweep entries score IDENTICALLY
    (paired noise via the per-K RNG reset). We pass K twice via the dedup-exempt
    path by checking a repeated-K run keeps val_bpb consistent against a single-K
    run on the same seed (instrument determinism is the observable)."""
    out = _run_main_capture(
        capsys, ["--k-eval-sweep", "4", "--seed", "5", "--init-state", "random",
                 "--artifact-out", str(tmp_path / "a.bin")])
    sweep, _, _ = _parse_depth_sweep(out)
    assert 4 in sweep
    # Re-run the same single-K sweep: the paired RNG reset must reproduce it.
    out2 = _run_main_capture(
        capsys, ["--k-eval-sweep", "4", "--seed", "5", "--init-state", "random",
                 "--artifact-out", str(tmp_path / "b.bin")])
    sweep2, _, _ = _parse_depth_sweep(out2)
    assert sweep[4][1] == sweep2[4][1], "single-K val_loss not reproducible"


# ---------------------------------------------------------------------------
# Fix 3 — half-specified-seed guard in run_reversible.
# ---------------------------------------------------------------------------
def test_run_reversible_rejects_half_specified_seed():
    import pytest
    from train_gpt import ReversibleRecurrence, _TinyDelta
    rec = ReversibleRecurrence(_TinyDelta(8), _TinyDelta(8))
    x0 = torch.randn(2, 4, 8, dtype=torch.float64)
    a0 = 0.02 * torch.randn(2, 4, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="both be provided or both be None"):
        rec.run_reversible(x0, depth=3, a0=a0, b0=None)
    with pytest.raises(ValueError, match="both be provided or both be None"):
        rec.run_reversible(x0, depth=3, a0=None, b0=a0)


# ---------------------------------------------------------------------------
# Fix 4 — Hyperparameters validates init_state outside the CLI.
# ---------------------------------------------------------------------------
def test_hyperparameters_rejects_bogus_init_state():
    import pytest
    from train_gpt import Hyperparameters
    with pytest.raises(ValueError, match="init_state"):
        Hyperparameters(
            model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
            n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            init_state="bogus",
        )


def test_hyperparameters_accepts_valid_init_states():
    from train_gpt import Hyperparameters
    for mode in ("x0", "random"):
        Hyperparameters(
            model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
            n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
            init_state=mode,
        )


# ---------------------------------------------------------------------------
# Fix 5 — finite_horizon_loss shares recurrence-init noise across the two passes.
# ---------------------------------------------------------------------------
def test_finite_horizon_loss_shares_random_init_noise_across_passes():
    """With --init-state random, the shallow and deep passes must start from the
    SAME init noise (per-pass RNG reset). When k_lo == k_hi the hinge collapses
    to zero only if both passes saw the SAME start, so l_lo == l_hi exactly."""
    from train_gpt import Hyperparameters, M0GPT, finite_horizon_loss
    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=8, init_state="random", init_state_std=0.1,
    )
    model = M0GPT(args)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    _, parts = finite_horizon_loss(
        model, x, y, k_hi=3, k_lo=3, lambda_h=1.0, margin=0.0, lambda_route=0.0,
        use_load_balance=False)
    # Same depth + same shared init noise => identical scalar losses.
    assert torch.allclose(parts["l_hi"].detach(), parts["l_lo_sg"], atol=1e-6), (
        "shallow/deep passes did not share random-init noise (different starts)"
    )


def test_finite_horizon_loss_noiseless_x0_unaffected():
    """For --init-state x0 (no randn draws) the RNG save/restore is a no-op:
    the loss path is unchanged and finite."""
    from train_gpt import Hyperparameters, M0GPT, finite_horizon_loss
    torch.manual_seed(0)
    args = Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=8, init_state="x0",
    )
    model = M0GPT(args)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    loss, _ = finite_horizon_loss(
        model, x, y, k_hi=4, k_lo=2, lambda_h=1.0, margin=0.0, lambda_route=0.0,
        use_load_balance=False)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Fix 6 — fit_phi drops NaN points before the OLS fit.
# ---------------------------------------------------------------------------
def test_fit_phi_drops_nan_points():
    import math
    from train_gpt import fit_phi
    # A clean descending map: loss(r) = 2 - log(r) => slope -1 => phi == 1.
    clean = {1: 2.0, 2: 2.0 - math.log(2), 4: 2.0 - math.log(4)}
    assert abs(fit_phi(clean) - 1.0) < 1e-9
    # Same map with one diverged (NaN) depth must NOT poison the fit.
    poisoned = dict(clean)
    poisoned[8] = float("nan")
    phi = fit_phi(poisoned)
    assert math.isfinite(phi), f"NaN point poisoned phi: {phi}"
    assert abs(phi - 1.0) < 1e-9


def test_fit_phi_all_nan_returns_zero():
    import math
    from train_gpt import fit_phi
    assert fit_phi({2: float("nan"), 4: float("nan")}) == 0.0
    # A single finite point after dropping NaN -> <2 distinct depths -> 0.0.
    assert fit_phi({2: 1.0, 4: float("nan")}) == 0.0


# ---------------------------------------------------------------------------
# Shared-trajectory k-eval-sweep — one max-K pass read off at each K.
#
# The recurrence is deterministic given (a0, b0, x0): running to depth K is
# EXACTLY running to a smaller depth and CONTINUING. So evaluating losses at a
# set of depths can read the midpoint 0.5*(a+b) off ONE depth-max(K) trajectory
# at each requested K. The KEY correctness gate: that single-trajectory readoff
# must be IDENTICAL to independent single-depth model(x, y, K) runs.
# ---------------------------------------------------------------------------
def _build_equiv_model(step_conditioning=False):
    """A tiny deterministic M0 (x0 init, fp32/CPU) for the equivalence gate."""
    from train_gpt import Hyperparameters, M0GPT
    torch.manual_seed(0)
    return M0GPT(Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=8, init_state="x0", step_conditioning=step_conditioning,
        max_step_emb=16,
    ))


def test_eval_losses_multi_depth_matches_single_depth_x0():
    """Reading the midpoint off ONE shared trajectory at each K is bit-for-bit
    equal (to tight tolerance) to independent single-depth forwards. This proves
    K_big = K_small CONTINUED (one noise draw, one trajectory)."""
    model = _build_equiv_model(step_conditioning=False)
    model.eval()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    depths = [3, 5, 8]
    with torch.inference_mode():
        multi = model.eval_losses_multi_depth(x, y, depths)
        for d in depths:
            single = model(x, y, d)
            assert torch.allclose(multi[d], single, atol=1e-5), (
                f"depth {d}: multi {multi[d].item()} != single {single.item()}")


def test_eval_losses_multi_depth_matches_single_depth_step_conditioning():
    """Same equivalence with step_conditioning on: the per-step clock e_k must be
    applied identically along the shared trajectory."""
    model = _build_equiv_model(step_conditioning=True)
    model.eval()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    depths = [3, 5, 8]
    with torch.inference_mode():
        multi = model.eval_losses_multi_depth(x, y, depths)
        for d in depths:
            single = model(x, y, d)
            assert torch.allclose(multi[d], single, atol=1e-5), (
                f"depth {d}: multi {multi[d].item()} != single {single.item()}")


def test_eval_losses_multi_depth_dedups_and_sorts_depths():
    """Requesting a repeated/unsorted depth set still returns one entry per
    distinct depth, each equal to the single-depth forward."""
    model = _build_equiv_model(step_conditioning=False)
    model.eval()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    with torch.inference_mode():
        multi = model.eval_losses_multi_depth(x, y, [5, 3, 5, 3])
        assert set(multi.keys()) == {3, 5}
        for d in (3, 5):
            assert torch.allclose(multi[d], model(x, y, d), atol=1e-5)


def test_eval_losses_multi_depth_shares_one_random_draw():
    """Under init_state='random' the single shared draw is the WHOLE point: every
    depth reads off ONE trajectory (one noise draw). The deepest depth's
    multi-depth loss equals an independent single-depth forward taken under the
    SAME RNG state (so the same draw)."""
    from train_gpt import Hyperparameters, M0GPT, paired_rng
    torch.manual_seed(0)
    model = M0GPT(Hyperparameters(
        model_dim=16, n_heads=2, n_kv_heads=1, vocab_size=32,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=8, init_state="random", init_state_std=0.1,
    ))
    model.eval()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    depths = [3, 5, 8]
    device = next(model.parameters()).device
    with torch.inference_mode():
        with paired_rng(device) as reset_rng:
            reset_rng()
            multi = model.eval_losses_multi_depth(x, y, depths)
            # The shared trajectory's max depth == a single-depth forward from
            # the SAME draw (reset RNG so randn_like draws the identical seed).
            reset_rng()
            single_max = model(x, y, max(depths))
    assert torch.allclose(multi[max(depths)], single_max, atol=1e-5)


def test_run_validation_multi_depth_matches_per_depth_run_validation():
    """The per-depth eval driver must reproduce run_validation's loss/BPB for
    each depth (same accumulation, same byte math) — only sharing one trajectory."""
    from train_gpt import run_validation, run_validation_multi_depth, _ReplayLoader
    model = _build_equiv_model(step_conditioning=False)
    device = torch.device("cpu")
    cache = [(torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
             for _ in range(2)]
    depths = [3, 5, 8]
    multi = run_validation_multi_depth(
        model, _ReplayLoader(cache), depths=depths, n_batches=len(cache),
        seq_len=8, global_tokens=16, grad_accum_steps=1, device=device,
        luts=None, autocast_enabled=False)
    assert set(multi.keys()) == set(depths)
    for d in depths:
        loss, bpb = run_validation(
            model, _ReplayLoader(cache), depth=d, n_batches=len(cache),
            seq_len=8, global_tokens=16, grad_accum_steps=1, device=device,
            luts=None, autocast_enabled=False)
        m_loss, m_bpb = multi[d]
        assert abs(m_loss - loss) < 1e-5, f"depth {d} loss {m_loss} != {loss}"
        # BPB is NaN without LUTs; only assert finite-loss parity above.


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
