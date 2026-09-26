"""Experiment manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bodhan_bhili.core.config import ExperimentConfig
from bodhan_bhili.core.environment import collect_environment
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.serialization import atomic_write_json


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_initial_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Create the first immutable-ish experiment description."""
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "initialized",
        "project": {
            "name": config.project.name,
            "task": config.project.task,
            "direction": config.project.direction,
            "seed": config.project.seed,
        },
        "run": {"run_id": config.run.run_id, "description": config.run.description},
        "model": {"repo_id": config.model.repo_id, "target_language": config.model.target_language},
        "method": {
            "fine_tuning": "QLoRA",
            "quantization_bits": 4,
            "quantization_type": config.quantization.quant_type,
            "lora_rank": config.lora.rank,
            "lora_alpha": config.lora.alpha,
        },
        "environment": collect_environment(),
        "dataset": None,
        "trainable_parameters": None,
        "best_checkpoint": None,
        "final_metrics": None,
    }


def write_manifest(manifest: dict[str, Any], paths: ArtifactPaths) -> None:
    """Persist the manifest atomically."""
    manifest["updated_at_utc"] = utc_now()

    atomic_write_json(manifest, paths.manifest)
