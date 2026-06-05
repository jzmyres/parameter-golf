"""Task 7 — M0 LM-scaffold trainer smoke + unit tests.

The smoke test runs ``train_gpt_m0.main`` end-to-end on CPU with tiny dims and a
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
    from train_gpt_m0 import main

    main(["--iterations", "3", "--model-dim", "32", "--n-heads", "4", "--n-kv-heads", "2",
          "--n-experts", "4", "--expert-rank", "8", "--n-mix", "2", "--kv-latent", "8", "--head-dim", "8",
          "--seq-len", "16", "--eval-batches", "2", "--device", "cpu",
          "--artifact-out", str(tmp_path / "m.bin")])
    assert (tmp_path / "m.bin").exists()
    assert (tmp_path / "m.bin").stat().st_size > 0


def _tiny_args():
    from train_gpt_m0 import Hyperparameters
    return Hyperparameters(
        model_dim=32, n_heads=4, n_kv_heads=2, vocab_size=1024,
        n_experts=4, expert_rank=8, n_mix=2, kv_latent=8, head_dim=8,
        max_seq_len=16,
    )


def test_optimizer_param_coverage():
    """Every trainable parameter lands in exactly one optimizer group."""
    from train_gpt_m0 import M0GPT, build_optimizers

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


def test_finite_horizon_hinge_loss():
    """Loss = L_hi + lambda_h*relu(L_hi - sg(L_lo) + margin) + lambda_route*aux_l1.

    sg(L_lo): the hinge must not backprop into the shallow pass; only L_hi and
    aux carry gradients into the params besides the hinge's L_hi term.
    """
    from train_gpt_m0 import M0GPT, finite_horizon_loss

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


def test_control_experiment_script_shape():
    import subprocess
    txt = open("experiments/run_m0_control_experiments.sh").read()
    for s in ("router_type=relu", "router_type=softmax", "for r in 1 2 4 8", "kv_latent"):
        assert s in txt
    subprocess.run(["bash", "-n", "experiments/run_m0_control_experiments.sh"], check=True)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
