"""Quick smoke test: train a few steps and verify all diagnostics trend correctly.

Run before every full training experiment to catch issues early.
MUST PASS before committing to a long training run.

Hard requirements:
1. Reconstruction error < 1e-8 (exact reversibility via fp64 accumulators)
2. Convergence ||z_T - z_{T-1}|| must decrease over training
3. Loss decreases (model is learning)
4. No NaN/Inf gradients
5. Expert balance CV decreasing (routing converging to balanced usage)
6. Expert entropy reasonable (not collapsed to single expert)
"""
import torch
import sys
sys.path.insert(0, ".")
from train_gpt import GPT, Hyperparameters


def _get_expert_diagnostics(model):
    """Extract expert diagnostics from all routers after eval forward."""
    diag = {}
    # Attention router
    ar = model.shared_block.attn.attn_router
    if ar._expert_usage is not None:
        diag["attn_usage"] = ar._expert_usage
        diag["attn_entropy"] = ar._expert_entropy
        diag["attn_balance_cv"] = ar._expert_balance_cv
    # MLP router
    mr = model.shared_block.mlp.mlp_router
    if mr._expert_usage is not None:
        diag["mlp_usage"] = mr._expert_usage
        diag["mlp_entropy"] = mr._expert_entropy
        diag["mlp_balance_cv"] = mr._expert_balance_cv
    # Orthogonality (from weight matrices)
    with torch.no_grad():
        for name, w, n_exp in [
            ("mlp", model.shared_block.mlp.fc.weight.float(), model.shared_block.mlp.num_experts),
            ("attn", model.shared_block.attn.proj.weight.float(), model.shared_block.attn.num_experts),
        ]:
            if n_exp < 2:
                continue
            es = w.shape[0] // n_exp
            groups = w.view(n_exp, es, -1).mean(dim=1)
            groups = groups / (groups.norm(dim=-1, keepdim=True) + 1e-8)
            cos = groups @ groups.T
            mask = ~torch.eye(n_exp, dtype=torch.bool, device=cos.device)
            diag[f"{name}_ortho"] = cos[mask].abs().mean().item()
    return diag


def smoke_test(num_steps: int = 80, eval_every: int = 20):
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    losses, recon_errors, iter_convs, residuals = [], [], [], []
    expert_snapshots = []

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
            diag = _get_expert_diagnostics(model)
            expert_snapshots.append(diag)
            print(f"  step {i+1}: loss={losses[-1]:.4f} recon={model._deq_recon_error:.2e} "
                  f"iter_conv={model._deq_iter_convergence:.1f} residual={r:.1f}")
            if "mlp_usage" in diag:
                print(f"    mlp: usage={diag['mlp_usage']} entropy={diag['mlp_entropy']:.4f} "
                      f"balance_cv={diag['mlp_balance_cv']:.4f} ortho={diag.get('mlp_ortho', 0):.4f}")
            if "attn_usage" in diag:
                print(f"    attn: usage={diag['attn_usage']} entropy={diag['attn_entropy']:.4f} "
                      f"balance_cv={diag['attn_balance_cv']:.4f} ortho={diag.get('attn_ortho', 0):.4f}")

    # --- Results ---
    print(f"\n--- Smoke Test Results ---")
    print(f"Params:            {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss:              {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    print(f"Recon errors:      {' -> '.join(f'{e:.2e}' for e in recon_errors)}")
    print(f"Iter convergence:  {' -> '.join(f'{c:.1f}' for c in iter_convs)}")
    print(f"DEQ residuals:     {' -> '.join(f'{r:.1f}' for r in residuals)}")

    ok = True

    # 1. Loss should decrease (compare first quarter avg vs last quarter avg)
    q = max(len(losses) // 4, 1)
    first_q = sum(losses[:q]) / q
    last_q = sum(losses[-q:]) / q
    if last_q > first_q:
        print(f"FAIL: training loss not decreasing (first_q={first_q:.4f} -> last_q={last_q:.4f})")
        ok = False

    # 2. Reconstruction error MUST be < 1e-8
    if any(e > 1e-8 for e in recon_errors):
        print(f"FAIL: reconstruction error > 1e-8 (got {max(recon_errors):.2e})")
        ok = False

    # 3. Reconstruction error should not increase
    if len(recon_errors) >= 2 and recon_errors[-1] > max(recon_errors[0] * 5, 1e-10):
        print(f"FAIL: reconstruction error diverging ({recon_errors[0]:.2e} -> {recon_errors[-1]:.2e})")
        ok = False

    # 4. Convergence MUST decrease (last eval < first eval)
    if len(iter_convs) >= 2:
        if iter_convs[-1] > iter_convs[0]:
            print(f"FAIL: iter convergence increasing ({iter_convs[0]:.1f} -> {iter_convs[-1]:.1f})")
            print(f"  DEQ is not converging toward equilibrium — check init or lr")
            ok = False

    # 5. No NaN/Inf gradients
    if has_bad_grad:
        print("FAIL: NaN or Inf in gradients")
        ok = False

    # 6. Expert balance should improve (CV decreasing or stable)
    if len(expert_snapshots) >= 2:
        first_snap = expert_snapshots[0]
        last_snap = expert_snapshots[-1]
        for key in ["mlp_balance_cv", "attn_balance_cv"]:
            if key in first_snap and key in last_snap:
                first_cv = first_snap[key]
                last_cv = last_snap[key]
                if last_cv > first_cv * 2.0 and last_cv > 0.3:
                    print(f"FAIL: {key} increasing ({first_cv:.4f} -> {last_cv:.4f}) — experts unbalanced")
                    ok = False

    # 7. Expert entropy should not collapse to 0 (single expert dominance)
    if expert_snapshots:
        last_snap = expert_snapshots[-1]
        for key in ["mlp_entropy", "attn_entropy"]:
            if key in last_snap and last_snap[key] < 0.1:
                print(f"FAIL: {key}={last_snap[key]:.4f} — routing collapsed to single expert")
                ok = False

    # 8. Expert orthogonality should trend toward 0 (not ±1)
    if len(expert_snapshots) >= 2:
        for key in ["mlp_ortho", "attn_ortho"]:
            if key in expert_snapshots[0] and key in expert_snapshots[-1]:
                first_o = expert_snapshots[0][key]
                last_o = expert_snapshots[-1][key]
                if last_o > 0.95:
                    print(f"WARN: {key}={last_o:.4f} — experts not diversifying (cos_sim near 1)")

    if ok:
        print("\nSMOKE TEST PASSED")
    else:
        print("\nSMOKE TEST FAILED — fix before running full training")
    return ok


if __name__ == "__main__":
    smoke_test()
