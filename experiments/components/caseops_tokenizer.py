"""Default-off CaseOps tokenizer fixture component.

The live training data is already pre-tokenized into FineWeb binary shards, so
this component exposes a principled, removable CaseOps codec and a startup
fixture smoke rather than mutating the training loader. A future raw-text data
pipeline can call `encode_caseops_text()` before SentencePiece tokenization and
`restore_caseops_text()` after decoding.
"""

from __future__ import annotations

from dataclasses import dataclass


_USE_CASEOPS: bool = False


@dataclass(frozen=True)
class CaseOp:
    index: int
    kind: str


@dataclass(frozen=True)
class CaseOpsEncoded:
    normalized: str
    ops: tuple[CaseOp, ...]


def set_caseops_enabled(enabled: bool) -> None:
    global _USE_CASEOPS
    _USE_CASEOPS = bool(enabled)


def is_caseops_enabled() -> bool:
    return _USE_CASEOPS


def encode_caseops_text(text: str) -> CaseOpsEncoded:
    """Lowercase text and record enough case ops to restore ASCII case."""
    ops: list[CaseOp] = []
    lowered_chars: list[str] = []
    for idx, ch in enumerate(text):
        if "A" <= ch <= "Z":
            ops.append(CaseOp(idx, "upper"))
            lowered_chars.append(ch.lower())
        else:
            lowered_chars.append(ch)
    return CaseOpsEncoded("".join(lowered_chars), tuple(ops))


def restore_caseops_text(encoded: CaseOpsEncoded) -> str:
    chars = list(encoded.normalized)
    for op in encoded.ops:
        if op.kind != "upper":
            raise ValueError(f"unknown CaseOp kind {op.kind!r}")
        if 0 <= op.index < len(chars):
            chars[op.index] = chars[op.index].upper()
    return "".join(chars)


def run_caseops_fixture_smoke(text: str = "Parameter Golf Smoke") -> dict[str, int]:
    encoded = encode_caseops_text(text)
    restored = restore_caseops_text(encoded)
    if restored != text:
        raise RuntimeError(f"CaseOps roundtrip failed: {restored!r} != {text!r}")
    if encoded.normalized != text.lower():
        raise RuntimeError("CaseOps normalization did not lowercase fixture text")
    return {
        "input_chars": len(text),
        "case_ops": len(encoded.ops),
    }
