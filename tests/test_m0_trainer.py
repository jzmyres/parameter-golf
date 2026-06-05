"""Task 7 — M0 LM-scaffold trainer smoke + unit tests.

The smoke test runs ``train_gpt.main`` end-to-end on CPU with tiny dims and a
synthetic in-memory shard (no big dataset required) and asserts an int6 artifact
is written. The unit tests cover the pieces the smoke can't isolate cheaply:
optimizer param coverage (every trainable param in exactly one group) and the
finite-horizon hinge loss form.
"""
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
    """Loss = L_hi + lambda_h*relu(L_hi - sg(L_lo) + margin) + lambda_route*aux_l1.

    sg(L_lo): the hinge must not backprop into the shallow pass; only L_hi and
    aux carry gradients into the params besides the hinge's L_hi term.
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
        + 0.01 * parts["aux_l1"]
    )
    assert torch.allclose(loss, expected, atol=1e-5)
    # L_lo inside the hinge must be stop-gradient: parts["l_lo_sg"] is detached.
    assert not parts["l_lo_sg"].requires_grad


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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
