"""Tiny parser for the restricted baseline manifest format.

The repo avoids a PyYAML dependency, so the manifest intentionally sticks to a
small YAML subset: a top-level `baselines:` list of scalar string mappings.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "kind",
    "title",
    "paper_url",
    "code_url",
    "commit",
    "license",
    "status",
    "replication_command",
    "notes",
    # The DDP runners dispatch on ddp_mode; require it so a missing mode is a loud
    # parse error, not a silent "blocked" misclassification. Mode-specific fields
    # (ddp_env/ddp_imports/ddp_smoke_command/ddp_required_gpus) stay optional.
    "ddp_mode",
}


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    baselines: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    seen_header = False

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "baselines:":
            seen_header = True
            continue
        if not seen_header:
            raise ValueError(f"{path}:{lineno}: expected top-level 'baselines:' header")
        if line.startswith("  - "):
            if current is not None:
                baselines.append(current)
            current = {}
            payload = line[4:]
        elif line.startswith("    "):
            if current is None:
                raise ValueError(f"{path}:{lineno}: field before first baseline item")
            payload = line[4:]
        else:
            raise ValueError(f"{path}:{lineno}: unsupported indentation")

        if ":" not in payload:
            raise ValueError(f"{path}:{lineno}: expected 'key: value'")
        key, value = payload.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key")
        if key in current:
            raise ValueError(f"{path}:{lineno}: duplicate key {key!r}")
        current[key] = _clean_scalar(value)

    if current is not None:
        baselines.append(current)

    for item in baselines:
        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            ident = item.get("id", "<unknown>")
            raise ValueError(f"{path}: baseline {ident!r} missing fields: {sorted(missing)}")
    return baselines


def find_baseline(items: list[dict[str, str]], baseline_id: str) -> dict[str, str]:
    for item in items:
        if item["id"] == baseline_id:
            return item
    raise KeyError(f"unknown baseline id: {baseline_id}")
