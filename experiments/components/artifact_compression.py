"""Default-off grouped artifact compression repacker.

The production int6 artifact schema remains unchanged:
`{"state_dict": qsd, "meta": meta}` serialized by `torch.save`. This component
only changes dictionary insertion order before compression, grouping related
quantized tensors so zstd/zlib can see more local regularity. If the grouped
payload is not smaller, callers keep the baseline bytes.
"""

from __future__ import annotations

import io
import zlib
from collections import OrderedDict
from typing import Any

import torch

try:
    import zstandard
except ImportError:  # pragma: no cover - depends on environment
    zstandard = None


_USE_GROUPED_ARTIFACT_COMPRESSION: bool = False


def set_grouped_artifact_compression_enabled(enabled: bool) -> None:
    global _USE_GROUPED_ARTIFACT_COMPRESSION
    _USE_GROUPED_ARTIFACT_COMPRESSION = bool(enabled)


def is_grouped_artifact_compression_enabled() -> bool:
    return _USE_GROUPED_ARTIFACT_COMPRESSION


def _base_key(name: str) -> str:
    if name.endswith(".q"):
        return name[:-2]
    if name.endswith(".scale"):
        return name[:-6]
    return name


def _group_key(name: str, meta: dict[str, Any]) -> tuple[int, str]:
    base = _base_key(name)
    entry = meta.get(base, "other")
    if isinstance(entry, dict) and entry.get("type") == "int6":
        # Keep each tensor's q/scale adjacent while placing all quantized
        # payloads first.
        suffix_rank = 0 if name.endswith(".q") else 1 if name.endswith(".scale") else 2
        return (0, f"{base}:{suffix_rank}:{name}")
    order = {
        "passthrough_ctrl": 1,
        "passthrough_fp16": 2,
        "passthrough": 3,
    }
    return (order.get(str(entry), 4), name)


def grouped_compress_int6_payload(
    qsd: dict[str, torch.Tensor],
    meta: dict[str, Any],
    *,
    compressor: str,
) -> tuple[bytes, dict[str, int | str]]:
    """Serialize and compress the same artifact schema with grouped key order."""
    ordered_qsd = OrderedDict((name, qsd[name]) for name in sorted(qsd, key=lambda n: _group_key(n, meta)))
    payload = {"state_dict": ordered_qsd, "meta": meta}
    buf = io.BytesIO()
    torch.save(payload, buf)
    raw = buf.getvalue()
    if compressor == "zstd":
        if zstandard is None:
            raise RuntimeError("zstd compressor requested but zstandard is not importable")
        compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    elif compressor == "zlib":
        compressed = zlib.compress(raw, 9)
    else:
        raise ValueError(f"unknown compressor {compressor!r}")
    return compressed, {
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "compressor": compressor,
    }


def run_grouped_artifact_compression_smoke(*, compressor: str = "zlib") -> dict[str, int | str]:
    """Small deterministic exec witness for grouping + decompression schema."""
    torch.manual_seed(31)
    qsd = {
        "b.scale": torch.ones(4, dtype=torch.float16),
        "a.q": torch.randint(-31, 31, (4, 8), dtype=torch.int8),
        "a.scale": torch.ones(4, dtype=torch.float16),
        "ctrl": torch.randn(3),
        "b.q": torch.randint(-31, 31, (4, 8), dtype=torch.int8),
    }
    meta: dict[str, Any] = {
        "a": {"type": "int6"},
        "b": {"type": "int6"},
        "ctrl": "passthrough_ctrl",
    }
    blob, stats = grouped_compress_int6_payload(qsd, meta, compressor=compressor)
    raw = zstandard.ZstdDecompressor().decompress(blob) if compressor == "zstd" else zlib.decompress(blob)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if set(payload["state_dict"].keys()) != set(qsd.keys()):
        raise RuntimeError("grouped artifact smoke changed state_dict keys")
    return stats
