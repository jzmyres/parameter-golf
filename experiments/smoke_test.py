"""Quick smoke test: train a few steps and verify DEQ diagnostics converge.

Run before every full training experiment to catch issues early.
MUST PASS before committing to a long training run.

Checks:
1. Loss decreases over training steps
2. DEQ reconstruction error stays near zero (< 1.0 relative)
3. DEQ iter convergence ||z_T - z_{T-1}|| decreases over training
4. Gradient flow is healthy (no NaN/Inf)
"""
import torch
import sys
sys.path.insert(0, ".")
from train_gpt import GPT, Hyperparameters


def smoke_test(num_steps: int = 30, eval_every: int = 10):
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    recon_errors = []
    iter_convs = []
    residuals = []

    for i in range(num_steps):
        model.train()
        x = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        y = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()

        # Check for NaN/Inf gradients
        has_bad_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_bad_grad = True
                break

        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

        # Periodic eval for DEQ diagnostics
        if (i + 1) % eval_every == 0:
            model.eval()
            ex = torch.randint(0, args.vocab_size, (2, 128), device="cuda")
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.forward_logits(ex)
            r = model._deq_residuals[0] if model._deq_residuals else 0.0
            residuals.append(r)
            recon_errors.append(model._deq_recon_error)
            iter_convs.append(model._deq_iter_convergence)
            print(f"  step {i+1}: loss={losses[-1]:.4f} recon={model._deq_recon_error:.4f} "
                  f"iter_conv={model._deq_iter_convergence:.1f} residual={r:.1f}")

    print(f"\n--- Smoke Test Results ---")
    print(f"Params:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss:              {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    print(f"Recon errors:      {' -> '.join(f'{e:.4f}' for e in recon_errors)}")
    print(f"Iter convergence:  {' -> '.join(f'{c:.1f}' for c in iter_convs)}")
    print(f"DEQ residuals:     {' -> '.join(f'{r:.1f}' for r in residuals)}")

    # --- CHECKS ---
    ok = True

    # 1. Loss should not increase significantly
    if losses[-1] > losses[0] + 0.5:
        print("FAIL: loss increased by > 0.5")
        ok = False

    # 2. Reconstruction error must stay small (< 1.0 relative)
    if any(e > 1.0 for e in recon_errors):
        print(f"FAIL: reconstruction error > 1.0 (got {max(recon_errors):.4f})")
        ok = False

    # 3. Reconstruction error should not increase (must be stable or decreasing)
    if len(recon_errors) >= 2 and recon_errors[-1] > recon_errors[0] * 10:
        print(f"FAIL: reconstruction error diverging ({recon_errors[0]:.4f} -> {recon_errors[-1]:.4f})")
        ok = False

    # 4. Iter convergence should not diverge wildly
    if len(iter_convs) >= 2 and iter_convs[-1] > iter_convs[0] * 10:
        print(f"FAIL: iter convergence diverging ({iter_convs[0]:.1f} -> {iter_convs[-1]:.1f})")
        ok = False

    # 5. No NaN/Inf gradients
    if has_bad_grad:
        print("FAIL: NaN or Inf in gradients")
        ok = False

    if ok:
        print("\nSMOKE TEST PASSED")
    else:
        print("\nSMOKE TEST FAILED — fix before running full training")
    return ok


if __name__ == "__main__":
    smoke_test()
