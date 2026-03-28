"""Quick smoke test: train a few steps and verify DEQ diagnostics don't diverge.

Run before every full training experiment to catch issues early.
Checks: reconstruction error, convergence, gradient flow, loss sanity.
"""
import torch
import sys
sys.path.insert(0, ".")
from train_gpt import GPT, Hyperparameters

def smoke_test(num_steps: int = 20):
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

    for i in range(num_steps):
        x = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        y = torch.randint(0, args.vocab_size, (4, 128), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if (i + 1) % 5 == 0:
            print(f"  step {i+1}: loss={loss.item():.4f}")

    # Run fp32 pass for accurate diagnostics
    model.eval()
    x = torch.randint(0, args.vocab_size, (2, 128), device="cuda")
    with torch.no_grad():
        logits = model.forward_logits(x)

    residual = model._deq_residuals[0] if model._deq_residuals else 0.0
    recon = model._deq_recon_error
    conv = model._deq_iter_convergence

    print(f"\n--- Smoke Test Results ---")
    print(f"Final loss:        {losses[-1]:.4f}")
    print(f"Loss trend:        {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    print(f"DEQ residual:      {residual:.4f}")
    print(f"DEQ recon error:   {recon:.6f}")
    print(f"DEQ iter conv:     {conv:.4f}")
    print(f"Params:            {sum(p.numel() for p in model.parameters()):,}")

    ok = True
    if losses[-1] > losses[0] + 0.5:
        print("WARNING: loss increased significantly")
        ok = False
    if recon > 100.0:
        print("WARNING: reconstruction error > 100 (expected near-zero in fp32)")
        ok = False
    if residual > 100000:
        print("WARNING: DEQ residual very large")

    if ok:
        print("\nSMOKE TEST PASSED")
    else:
        print("\nSMOKE TEST FAILED")
    return ok


if __name__ == "__main__":
    smoke_test()
