"""Safe serialization helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(payload: Any, destination: str | Path) -> Path:
    """Write JSON atomically and preserve Devanagari text."""
    destination = Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Keep both files on the same filesystem for atomic replacement.
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with temporary.open(mode="w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False, default=str)

        handle.flush()

        os.fsync(handle.fileno())

    os.replace(temporary, destination)

    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    path = Path(path)

    with path.open(mode="r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, but received {type(value).__name__}.")

    return value
