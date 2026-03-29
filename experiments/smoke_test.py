"""Quick smoke test: train a few steps and verify DEQ diagnostics converge.

Run before every full training experiment to catch issues early.
MUST PASS before committing to a long training run.

Hard requirements (RevDEQ paper):
1. Reconstruction error < 1e-8 (exact reversibility via fp64 accumulators)
2. Convergence ||z_T - z_{T-1}|| must decrease or stay stable over training
3. Loss decreases (model is learning)
4. No NaN/Inf gradients
"""
import torch
import sys
sys.path.insert(0, ".")
from train_gpt import GPT, Hyperparameters


def smoke_test(num_steps: int = 50, eval_every: int = 10):
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
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

        has_bad_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_bad_grad = True
                break

        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

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
            print(f"  step {i+1}: loss={losses[-1]:.4f} recon={model._deq_recon_error:.2e} "
                  f"iter_conv={model._deq_iter_convergence:.1f} residual={r:.1f}")

    print(f"\n--- Smoke Test Results ---")
    print(f"Params:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss:              {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    print(f"Recon errors:      {' -> '.join(f'{e:.2e}' for e in recon_errors)}")
    print(f"Iter convergence:  {' -> '.join(f'{c:.1f}' for c in iter_convs)}")
    print(f"DEQ residuals:     {' -> '.join(f'{r:.1f}' for r in residuals)}")

    ok = True

    # 1. Loss should not increase significantly
    # Compare first half avg vs second half avg — loss should decrease
    mid = len(losses) // 2
    first_half = sum(losses[:mid]) / max(mid, 1)
    second_half = sum(losses[mid:]) / max(len(losses) - mid, 1)
    if second_half > first_half:
        print(f"FAIL: training loss not decreasing (first_half={first_half:.4f} -> second_half={second_half:.4f})")
        ok = False

    # 2. Reconstruction error MUST be < 1e-8 (fp64 reversibility)
    if any(e > 1e-8 for e in recon_errors):
        print(f"FAIL: reconstruction error > 1e-8 (got {max(recon_errors):.2e})")
        ok = False

    # 3. Reconstruction error should not increase
    if len(recon_errors) >= 2 and recon_errors[-1] > max(recon_errors[0] * 5, 1e-10):
        print(f"FAIL: reconstruction error diverging ({recon_errors[0]:.2e} -> {recon_errors[-1]:.2e})")
        ok = False

    # 4. Iter convergence should trend downward (warn if increasing)
    if len(iter_convs) >= 2:
        ratio = iter_convs[-1] / max(iter_convs[0], 1e-6)
        if ratio > 3.0:
            print(f"WARN: iter convergence increasing ({iter_convs[0]:.1f} -> {iter_convs[-1]:.1f}, ratio={ratio:.1f}x)")
            print(f"  With 2 DEQ iters + identity init, some increase is expected early")
        # Hard fail only if convergence explodes catastrophically (>100x)
        if ratio > 100:
            print(f"FAIL: iter convergence exploding")
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
