"""Iter 123 / H91: Phased Test-Time Training (TTT) — per-document LoRA adapter.

Drop-in scaffold for the records' phased TTT eval pipeline. Implements the
canonical `BatchedLinearLoRA` primitive verbatim from the SOTA records, plus
a sketched driver showing the per-doc / per-phase loop math.

Verified against records SOTA (2026-04-27, val_bpb=1.0611):
`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py`
L1639-1731 (BatchedLinearLoRA + BatchedTTTLoRA), L1427+ (forward_ttt),
L3021+ (eval_val_ttt_phased). The `BatchedLinearLoRA` here is BIT-EQUIVALENT
to the records implementation.

# Hypothesis (from experiments/hypotheses.md H91)

Records' SOTA stack uses TTT (Test-Time Training): a small LoRA adapter is
fine-tuned on each document's prefix at eval time, then frozen and used to
predict the suffix. Phased TTT splits this into 3 cumulative phases at doc
boundaries (e.g. 833/1666/2500 docs of a 2500-doc max prefix). LoRA per-doc
reset between documents.

This is the LARGEST single feature in the records corpus — every record from
2026-03-23 onward uses TTT. Gain: −0.05 to −0.10 BPB consistently.

# Math

For each Linear `y = x @ W^T` in the LoRA target set, attach:

    A: (bsz, rank, in_features)  — uniform [-1/√in, 1/√in] init
    B: (bsz, out_features, rank) — zero init (so LoRA contribution = 0 at start)
    alpha: 144 (default)
    scale: alpha / rank

Forward (with LoRA contribution):
    y = x @ W^T + ((x @ A^T) @ B^T) * scale

The `bsz` dimension on A, B is a *document* batch — different docs in the
eval batch get different LoRA states (advanced through SGD per-doc).

# Phased eval algorithm (records' eval_val_ttt_phased)

For each doc-boundary phase ∈ {phase_1, phase_2, phase_3}:
    1. Reset LoRA (zero B; optionally re-randomize A — `TTT_WARM_START_A=1` keeps A).
    2. For each document up to the phase boundary:
        a. Run prefix tokens through `forward_ttt(input_ids, target_ids, lora=cur_lora)`
        b. Compute per-token loss; backward; SGD step on LoRA params with
           AdamW (TTT_BETA2=0.99, TTT_WEIGHT_DECAY=0.5, TTT_LORA_LR=1e-4).
    3. After all docs in the phase done, evaluate suffix (token > prefix_end)
       using the adapted LoRA. Suffix loss contributes to val_bpb.

# Integration into train_gpt.py — DESIGN NOTES (not yet implemented)

This component is a SCAFFOLD. Full integration is non-trivial because our
model has per-expert linears (Q/K/V/O/MLP/MoS-A bank) rather than dense
per-block linears the records use. Two paths:

(a) Per-expert LoRA: instantiate one `BatchedLinearLoRA` per (expert, projection,
    layer). For E=16 experts × 6 projection types × 12 layers = 1152 LoRA
    instances. At rank=80 each LoRA is `bsz * (rank * in + out * rank)` params.
    Substantial param count addition during eval; resets per-doc still cheap.

(b) Pooled LoRA: one BatchedLinearLoRA per projection type, applied uniformly
    across all experts. Loses per-expert specialization but ~16× fewer LoRA
    instances. Records-style: one LoRA per layer, not per expert.

Path (b) is the records-faithful design. Path (a) gives finer control but
costs more eval-time memory.

Smoke-test: `python experiments/components/phased_ttt.py`.
"""
import math
import os
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Default constants matching records SOTA hparam stack.
DEFAULT_TTT_LORA_RANK = 80          # SOTA stack hparam (was 96 default; 80 won the greedy search)
DEFAULT_TTT_LORA_LR = 1e-4
DEFAULT_TTT_LORA_ALPHA = 144.0
DEFAULT_TTT_BETA2 = 0.99
DEFAULT_TTT_WEIGHT_DECAY = 0.5
DEFAULT_TTT_WARM_START_A = True
DEFAULT_PHASE_BOUNDARIES = (833, 1666, 2500)


class BatchedLinearLoRA(nn.Module):
    """Per-document batched LoRA adapter for a single Linear layer.

    Records-canonical (records L1639-1660). Drop-in numerically identical to:
        records/.../2026-04-27_.../train_gpt.py::BatchedLinearLoRA

    Args:
        bsz: number of independent documents in the eval batch (each gets its
            own LoRA state).
        in_features: input dim of the underlying Linear.
        out_features: output dim of the underlying Linear.
        rank: LoRA rank.
        alpha: LoRA scaling factor (records: 144 default). scale = alpha / rank.
        warm_start_a: if True, `reset()` keeps A (only B is zeroed). If False,
            re-randomize A on each reset. Records default: True (TTT_WARM_START_A=1).

    Forward:
        x: (bsz, T, in_features) — per-doc input.
        Returns: (bsz, T, out_features) — LoRA correction term.

    Note: this returns the LoRA CORRECTION, not the full base+LoRA output.
    Caller adds: `y = base_linear(x) + lora(x)`.
    """

    def __init__(
        self,
        bsz: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = DEFAULT_TTT_LORA_ALPHA,
        warm_start_a: bool = DEFAULT_TTT_WARM_START_A,
    ):
        super().__init__()
        self.bsz = int(bsz)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self._scale = self.alpha / self.rank
        self._bound = 1.0 / math.sqrt(self.in_features)
        self.warm_start_a = bool(warm_start_a)
        # A: (bsz, rank, in) uniform-init.
        self.A = nn.Parameter(
            torch.empty(self.bsz, self.rank, self.in_features).uniform_(-self._bound, self._bound)
        )
        # B: (bsz, out, rank) zero-init → LoRA contribution = 0 at start.
        self.B = nn.Parameter(torch.zeros(self.bsz, self.out_features, self.rank))

    def reset(self) -> None:
        """Per-document reset. Always zero B; A only re-randomized if not warm-start."""
        with torch.no_grad():
            if not self.warm_start_a:
                self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """LoRA correction term.

        x: (bsz, T, in_features). Returns (bsz, T, out_features) scaled by alpha/rank.
        """
        # Records-canonical: ((x @ A^T) @ B^T) * scale
        return ((x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)) * self._scale


def attach_lora_to_linear(
    base: nn.Linear,
    bsz: int,
    rank: int,
    alpha: float = DEFAULT_TTT_LORA_ALPHA,
    warm_start_a: bool = DEFAULT_TTT_WARM_START_A,
) -> tuple[nn.Linear, BatchedLinearLoRA]:
    """Construct a LoRA adapter for an existing nn.Linear.

    Returns the (frozen-during-TTT) base Linear and the trainable LoRA module.
    Caller composes: y = base(x) + lora(x) at forward time.
    Caller is responsible for setting `base.weight.requires_grad_(False)` during TTT.
    """
    if base.weight.shape != (base.out_features, base.in_features):
        raise ValueError("base must be a standard nn.Linear with shape (out, in)")
    lora = BatchedLinearLoRA(
        bsz=bsz,
        in_features=base.in_features,
        out_features=base.out_features,
        rank=rank,
        alpha=alpha,
        warm_start_a=warm_start_a,
    )
    return base, lora


def phased_ttt_eval_driver(
    forward_ttt_fn: Callable[[Tensor, Tensor, "BatchedLinearLoRA"], Tensor],
    lora_module: nn.Module,
    docs: Sequence[tuple[Tensor, Tensor, int]],
    phase_boundaries: Sequence[int] = DEFAULT_PHASE_BOUNDARIES,
    ttt_lr: float = DEFAULT_TTT_LORA_LR,
    ttt_beta2: float = DEFAULT_TTT_BETA2,
    ttt_weight_decay: float = DEFAULT_TTT_WEIGHT_DECAY,
    eval_suffix_fn: Callable[[Tensor, Tensor, "BatchedLinearLoRA"], tuple[float, int]] | None = None,
) -> dict:
    """Phased TTT eval driver — sketches the records' eval_val_ttt_phased loop.

    Args:
        forward_ttt_fn(input_ids, target_ids, lora) -> per-token loss tensor.
            Caller wires this to the model's `forward_ttt` method.
        lora_module: an nn.Module containing all `BatchedLinearLoRA` adapters
            with a `.reset()` method that calls each adapter's reset.
        docs: iterable of (input_ids, target_ids, prefix_end) tuples per document.
        phase_boundaries: doc-count cutoffs for cumulative phases. Default
            (833, 1666, 2500) matches records' SOTA hparam.
        ttt_lr / ttt_beta2 / ttt_weight_decay: AdamW hyperparameters for
            LoRA SGD updates. Records SOTA: 1e-4 / 0.99 / 0.5.
        eval_suffix_fn(input_ids, target_ids, lora) -> (suffix_loss_sum, n_tokens).
            Computes loss on the suffix (token > prefix_end) for val_bpb.

    Returns dict with cumulative {suffix_loss_sum, n_suffix_tokens, val_bpb} per phase.

    Note: this is a SCAFFOLD demonstrating the algorithmic structure. Full
    integration requires the model's `forward_ttt` to be exposed and the LoRA
    targets selected (records: Q/V/K/O/MLP/lm_head per layer).
    """
    # AdamW over LoRA params only (frozen base weights — caller must ensure).
    optim = torch.optim.AdamW(
        lora_module.parameters(),
        lr=ttt_lr,
        betas=(0.9, ttt_beta2),
        weight_decay=ttt_weight_decay,
    )
    results = {"phases": []}

    for phase_idx, boundary in enumerate(phase_boundaries):
        # Phase fine-tune: SGD on LoRA params over docs[0:boundary] prefixes.
        for doc_idx, (input_ids, target_ids, prefix_end) in enumerate(docs):
            if doc_idx >= boundary:
                break
            lora_module.reset()  # per-doc reset
            # Prefix-only forward + backward. Slice along the TIME axis (T),
            # which is dim -2 for (B, T, D) hidden tensors or dim -1 for (B, T)
            # integer token streams. Records semantics: input_ids has shape (B, T)
            # of int tokens; here we conventionally slice on dim -2 to support
            # both pre-embedded (B, T, D) and raw token (B, T) tensors via the
            # caller's `forward_ttt_fn`.
            time_axis = -2 if input_ids.ndim >= 3 else -1
            prefix_slicer = [slice(None)] * input_ids.ndim
            prefix_slicer[time_axis] = slice(None, prefix_end)
            prefix_input = input_ids[tuple(prefix_slicer)]
            target_slicer = [slice(None)] * target_ids.ndim
            target_slicer[time_axis if target_ids.ndim >= 3 else -1] = slice(None, prefix_end)
            prefix_target = target_ids[tuple(target_slicer)]
            optim.zero_grad()
            loss_per_token = forward_ttt_fn(prefix_input, prefix_target, lora_module)
            loss = loss_per_token.mean()
            loss.backward()
            optim.step()

        # Phase eval: cumulative suffix-loss accumulation across all docs.
        if eval_suffix_fn is not None:
            phase_loss_sum = 0.0
            phase_n_tokens = 0
            for input_ids, target_ids, prefix_end in docs:
                loss_sum, n_tok = eval_suffix_fn(input_ids, target_ids, lora_module)
                phase_loss_sum += loss_sum
                phase_n_tokens += n_tok
            phase_bpb = phase_loss_sum / max(phase_n_tokens, 1) / math.log(2)
            results["phases"].append(
                {"phase": phase_idx, "boundary": boundary,
                 "loss_sum": phase_loss_sum, "n_tokens": phase_n_tokens,
                 "val_bpb": phase_bpb}
            )

    return results


def _smoke_test() -> None:
    print("phased_ttt.py smoke test (records-canonical BatchedLinearLoRA):")
    torch.manual_seed(0)

    # Case 1: zero-init B → LoRA correction is exactly 0 → identity-on-base
    bsz, T, in_f, out_f, r = 4, 16, 64, 32, 8
    lora = BatchedLinearLoRA(bsz=bsz, in_features=in_f, out_features=out_f, rank=r)
    x = torch.randn(bsz, T, in_f)
    correction = lora(x)
    assert correction.shape == (bsz, T, out_f), f"shape mismatch: {correction.shape}"
    assert correction.abs().max().item() < 1e-7, \
        f"zero-init B should give zero correction, got max {correction.abs().max()}"
    print("  zero-init B → zero correction:  PASS (max |corr| < 1e-7)")

    # Case 2: nonzero B → nonzero correction matches manual formula
    with torch.no_grad():
        lora.B.copy_(torch.randn_like(lora.B) * 0.1)
    correction = lora(x)
    expected = ((x @ lora.A.transpose(1, 2)) @ lora.B.transpose(1, 2)) * lora._scale
    assert torch.allclose(correction, expected, atol=1e-6), "manual formula mismatch"
    print("  manual ((x @ A^T) @ B^T) * scale: PASS")

    # Case 3: scale = alpha / rank
    assert abs(lora._scale - DEFAULT_TTT_LORA_ALPHA / r) < 1e-9
    print(f"  scale = alpha/rank check:        PASS ({DEFAULT_TTT_LORA_ALPHA}/{r}={lora._scale:.4f})")

    # Case 4: A init bound = 1/√in
    expected_bound = 1.0 / math.sqrt(in_f)
    assert lora.A.abs().max().item() <= expected_bound + 1e-6
    assert abs(lora._bound - expected_bound) < 1e-9
    print(f"  A init bound = 1/√in:            PASS (bound={expected_bound:.4f})")

    # Case 5: reset zeros B (warm_start_a=True keeps A)
    lora2 = BatchedLinearLoRA(bsz=2, in_features=8, out_features=4, rank=2, warm_start_a=True)
    A_before = lora2.A.detach().clone()
    with torch.no_grad():
        lora2.B.copy_(torch.randn_like(lora2.B))
    lora2.reset()
    assert lora2.B.abs().max().item() < 1e-9, "reset must zero B"
    assert torch.equal(lora2.A, A_before), "warm_start_a=True must preserve A"
    print("  reset zeros B (warm_start A):    PASS")

    # Case 6: reset re-randomizes A when warm_start_a=False
    lora3 = BatchedLinearLoRA(bsz=2, in_features=8, out_features=4, rank=2, warm_start_a=False)
    A_before = lora3.A.detach().clone()
    lora3.reset()
    assert not torch.equal(lora3.A, A_before), "warm_start_a=False must re-randomize A"
    print("  reset re-randomizes A (no warm): PASS")

    # Case 7: gradient flows to A and B
    lora4 = BatchedLinearLoRA(bsz=2, in_features=8, out_features=4, rank=2)
    with torch.no_grad():
        lora4.B.copy_(torch.randn_like(lora4.B) * 0.1)
    x = torch.randn(2, 5, 8)
    out = lora4(x)
    out.sum().backward()
    assert lora4.A.grad is not None and lora4.A.grad.abs().max().item() > 0, "A must receive grad"
    assert lora4.B.grad is not None and lora4.B.grad.abs().max().item() > 0, "B must receive grad"
    print("  gradient flow to A, B:           PASS")

    # Case 8: per-doc batch independence — different LoRAs for different docs
    lora5 = BatchedLinearLoRA(bsz=3, in_features=4, out_features=4, rank=2)
    with torch.no_grad():
        # Set B for doc 0 to nonzero, leave doc 1, 2 at zero.
        lora5.B[0] = torch.randn_like(lora5.B[0])
    x = torch.randn(3, 4, 4)
    out = lora5(x)
    assert out[0].abs().max().item() > 0, "doc 0 should have nonzero correction"
    assert out[1].abs().max().item() < 1e-7, "doc 1 should still have zero correction"
    assert out[2].abs().max().item() < 1e-7, "doc 2 should still have zero correction"
    print("  per-doc batch independence:      PASS")

    # Case 9: attach_lora_to_linear helper
    base = nn.Linear(8, 4, bias=False)
    base, lora6 = attach_lora_to_linear(base, bsz=2, rank=2)
    x = torch.randn(2, 3, 8)
    base_out = base(x)  # uses base.forward, weight is randomly-init nn.Linear
    lora_out = lora6(x)
    full_out = base_out + lora_out
    assert full_out.shape == (2, 3, 4)
    # At init, LoRA contribution is exactly 0 (B=0), so full_out == base_out.
    assert torch.allclose(full_out, base_out, atol=1e-7)
    print("  attach_lora_to_linear helper:    PASS (zero-init LoRA = identity-on-base)")

    # Case 10: phased_ttt_eval_driver scaffold runs end-to-end on toy problem.
    # Use a 1-Linear toy model: y = lora_correction(x). Adapt LoRA per doc to fit a target.
    class ToyModel(nn.Module):
        def __init__(self, in_f, out_f, bsz, rank):
            super().__init__()
            self.lora = BatchedLinearLoRA(bsz, in_f, out_f, rank)

        def forward(self, x):
            return self.lora(x)

    toy = ToyModel(8, 4, bsz=1, rank=2)

    def _fwd_ttt(input_ids, target_ids, lora):
        # Synthetic per-token loss
        out = lora(input_ids.float())
        # MSE to target_ids (treated as a regression target embedding)
        return ((out - target_ids.float()) ** 2).mean(-1)

    def _eval_suffix(input_ids, target_ids, lora):
        out = lora(input_ids.float())
        loss = ((out - target_ids.float()) ** 2).mean(-1).sum().item()
        n_tok = input_ids.shape[-2]
        return loss, n_tok

    docs = [
        (torch.randn(1, 4, 8), torch.randn(1, 4, 4), 2),
        (torch.randn(1, 4, 8), torch.randn(1, 4, 4), 2),
        (torch.randn(1, 4, 8), torch.randn(1, 4, 4), 2),
    ]
    results = phased_ttt_eval_driver(
        forward_ttt_fn=_fwd_ttt,
        lora_module=toy.lora,
        docs=docs,
        phase_boundaries=(2, 3),
        eval_suffix_fn=_eval_suffix,
    )
    assert "phases" in results
    assert len(results["phases"]) == 2
    print(f"  driver scaffold runs:            PASS ({len(results['phases'])} phases)")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke_test()
