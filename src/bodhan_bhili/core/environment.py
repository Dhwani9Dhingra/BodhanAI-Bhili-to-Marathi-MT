"""Collect environment metadata for reproducibility."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "sacrebleu",
    "huggingface-hub",
    "pydantic",
    "PyYAML",
)


def package_version(distribution_name: str) -> str | None:
    """Return installed package version without importing the package."""
    try:
        return importlib.metadata.version(distribution_name)

    except importlib.metadata.PackageNotFoundError:
        return None


def _run_git_command(arguments: list[str]) -> str | None:
    """Execute one small Git command safely."""
    try:
        result = subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True, timeout=5
        )

        return result.stdout.strip()

    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_git_info() -> dict[str, Any]:
    """Collect commit and working-tree information."""
    commit = _run_git_command(["rev-parse", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])

    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def collect_gpu_info() -> dict[str, Any]:
    """Collect GPU details when CUDA is available."""
    try:
        import torch

    except ImportError:
        return {"cuda_available": False, "reason": "PyTorch is not installed."}

    info: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }

    if not torch.cuda.is_available():
        return info

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)

    info.update(
        {
            "device_index": int(device_index),
            "gpu_name": properties.name,
            "vram_gib": round(properties.total_memory / (1024**3), 3),
            "compute_capability": (int(properties.major), int(properties.minor)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    )

    return info


def recommended_compute_dtype() -> str:
    """Pick the compute dtype based on the ACTUAL GPU."""
    try:
        import torch

    except ImportError:
        return "unknown"

    if not torch.cuda.is_available():
        return "unknown"

    if torch.cuda.is_bf16_supported():
        return "bfloat16"

    return "float16"


def collect_environment() -> dict[str, Any]:
    """Collect environment information without exposing credentials."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {package: package_version(package) for package in _TRACKED_PACKAGES},
        "gpu": collect_gpu_info(),
        "recommended_compute_dtype": recommended_compute_dtype(),
        "git": collect_git_info(),
        "working_directory": str(Path.cwd()),
    }
