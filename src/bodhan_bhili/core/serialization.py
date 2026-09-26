"""
Safe serialization helpers.

Why atomic writing?
-------------------
Google Colab can disconnect unexpectedly. If Python dies while writing
JSON directly to an important manifest file, that file can become
partially written and unusable.

We therefore:

    1. write the complete document to a temporary file;
    2. flush the data;
    3. replace the destination atomically.

This pattern will be reused for experiment state, manifests and
analysis summaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(
    payload: Any,
    destination: str | Path,
) -> Path:
    """
    Write JSON atomically and preserve Devanagari text.

    Parameters
    ----------
    payload:
        Any JSON-serializable Python object.

    destination:
        Final JSON file path.

    Returns
    -------
    Path
        The final path that was written.
    """

    destination = Path(destination)

    # Make sure the parent directory already exists.
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the temporary file beside the final file.
    # os.replace() is safest when both live on the same filesystem.
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with temporary.open(
        mode="w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=str,
        )

        # Push Python's buffered bytes to the OS.
        handle.flush()

        # Ask the OS to flush the file to disk as well.
        os.fsync(
            handle.fileno()
        )

    # Atomic replacement:
    # readers see either the old complete file or new complete file.
    os.replace(
        temporary,
        destination,
    )

    return destination


def read_json(
    path: str | Path,
) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    path = Path(path)

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as handle:

        value = json.load(
            handle
        )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"but received {type(value).__name__}."
        )

    return value