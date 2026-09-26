"""Resumable QLoRA training utilities: checkpoints, run fingerprints, batching."""

from __future__ import annotations

import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bodhan_bhili.core.config import ExperimentConfig
from bodhan_bhili.core.serialization import atomic_write_json, read_json

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")

# Files Hugging Face Trainer writes for a PEFT checkpoint that can be resumed exactly.
REQUIRED_CHECKPOINT_FILES = (
    "trainer_state.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
)

# Settings that may change between resumes without invalidating saved optimizer state.
_RESUME_SAFE_TRAINING_FIELDS = (
    "eval_batch_size",
    "logging_steps",
    "evaluation_steps",
    "save_steps",
    "save_total_limit",
    "eval_examples",
)

BEST_METRIC_FILE = "best_metric.json"


def _utc_now() -> str:
    """Current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def checkpoint_step(path: Path) -> int | None:
    """Return the global step encoded in a `checkpoint-<step>` folder name."""
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)

    return int(match.group(1)) if match else None


def is_complete_checkpoint(path: Path) -> bool:
    """Reject checkpoints cut short by a disconnect or an unfinished Drive sync."""
    if not path.is_dir():
        return False

    for name in REQUIRED_CHECKPOINT_FILES:
        file = path / name

        if not file.is_file() or file.stat().st_size == 0:
            return False

    try:
        state = read_json(path / "trainer_state.json")

    except (OSError, ValueError):
        return False

    return state.get("global_step") == checkpoint_step(path)


def find_resume_checkpoint(trainer_dir: Path) -> tuple[Path | None, list[Path]]:
    """Return the newest complete checkpoint and any incomplete ones newer than it."""
    if not trainer_dir.is_dir():
        return (None, [])

    candidates = sorted(
        (path for path in trainer_dir.iterdir() if checkpoint_step(path) is not None),
        key=checkpoint_step,
        reverse=True,
    )

    incomplete = []

    for candidate in candidates:
        if is_complete_checkpoint(candidate):
            return (candidate, incomplete)

        incomplete.append(candidate)

    return (None, incomplete)


def quarantine_checkpoints(checkpoints: list[Path], destination: Path) -> list[Path]:
    """Move unusable checkpoints aside so Trainer neither resumes from nor rotates them."""
    moved = []

    if not checkpoints:
        return moved

    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for checkpoint in checkpoints:
        target = destination / f"{checkpoint.name}.{stamp}"

        shutil.move(str(checkpoint), str(target))
        moved.append(target)

    return moved


def build_training_fingerprint(
    config: ExperimentConfig, *, train_sha256: str, dev_sha256: str
) -> dict[str, Any]:
    """Describe everything that must stay identical for a checkpoint to be resumable."""
    training = config.training.model_dump(mode="json")

    for field in _RESUME_SAFE_TRAINING_FIELDS:
        training.pop(field, None)

    return {
        "seed": config.project.seed,
        "model_repo_id": config.model.repo_id,
        "target_language": config.model.target_language,
        "max_train_tokens": config.model.max_train_tokens,
        "quantization": config.quantization.model_dump(mode="json"),
        "lora": config.lora.model_dump(mode="json"),
        "training": training,
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
    }


def fingerprint_differences(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """List top-level fingerprint keys whose values differ."""
    keys = sorted(set(saved) | set(current))

    return [key for key in keys if saved.get(key) != current.get(key)]


def verify_resume_fingerprint(fingerprint_file: Path, current: dict[str, Any]) -> None:
    """Refuse to resume a run whose data or hyperparameters have changed."""
    if not fingerprint_file.exists():
        raise RuntimeError(
            "Checkpoints exist but the training fingerprint is missing: "
            f"{fingerprint_file}. Cannot confirm they belong to this configuration. "
            "Use a new run_id to start fresh."
        )

    saved = read_json(fingerprint_file)
    differences = fingerprint_differences(saved, current)

    if differences:
        raise RuntimeError(
            "Refusing to resume: the configuration or prepared data changed since these "
            f"checkpoints were written. Changed: {differences}. "
            "Restore the original settings, or use a new run_id to start fresh."
        )


def warmup_steps(config: ExperimentConfig) -> int:
    """Convert the configured warmup ratio into whole optimizer steps."""
    return math.ceil(config.training.warmup_ratio * config.training.max_steps)


def to_training_feature(encoded: dict[str, Any]) -> dict[str, Any]:
    """Drop the batch dimension from one `encode_training_example` result."""
    length = encoded["input_ids"].shape[-1]
    feature = {}

    for key, value in encoded.items():
        if not hasattr(value, "dim") or value.dim() != 2 or tuple(value.shape) != (1, length):
            raise ValueError(
                f"Unexpected processor output {key!r}: expected a [1, {length}] tensor."
            )

        feature[key] = value[0]

    return feature


def pad_training_batch(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad token features; padded label positions are ignored by the loss."""
    import torch

    width = max(feature["input_ids"].shape[-1] for feature in features)
    batch = {}

    for key in features[0]:
        if key == "input_ids":
            pad_value = pad_token_id

        elif key == "labels":
            pad_value = -100

        else:
            pad_value = 0

        rows = [
            torch.nn.functional.pad(
                feature[key], (0, width - feature[key].shape[-1]), value=pad_value
            )
            for feature in features
        ]

        batch[key] = torch.stack(rows)

    return batch


def read_best_metric(adapter_best: Path) -> dict[str, Any] | None:
    """Return the saved best dev-loss record, if any."""
    metric_file = adapter_best / BEST_METRIC_FILE

    if not metric_file.exists():
        return None

    return read_json(metric_file)


def save_best_adapter(model, adapter_best: Path, record: dict[str, Any]) -> None:
    """Replace the best adapter without leaving a half-written folder behind."""
    staging = adapter_best.with_name(adapter_best.name + ".tmp")
    previous = adapter_best.with_name(adapter_best.name + ".old")

    for leftover in (staging, previous):
        if leftover.exists():
            shutil.rmtree(leftover)

    model.save_pretrained(staging, safe_serialization=True)
    atomic_write_json(record, staging / BEST_METRIC_FILE)

    if adapter_best.exists():
        adapter_best.rename(previous)

    staging.rename(adapter_best)

    if previous.exists():
        shutil.rmtree(previous)


def build_callbacks(*, progress_file: Path, adapter_best: Path, logger) -> list:
    """Create Trainer callbacks for progress tracking and best-adapter export."""
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        """Record every completed checkpoint so progress is visible on Drive."""

        def on_save(self, args, state, control, **kwargs):
            latest_loss = next(
                (entry["loss"] for entry in reversed(state.log_history) if "loss" in entry), None
            )
            atomic_write_json(
                {
                    "global_step": state.global_step,
                    "max_steps": state.max_steps,
                    "checkpoint": f"checkpoint-{state.global_step}",
                    "latest_train_loss": latest_loss,
                    "saved_at_utc": _utc_now(),
                },
                progress_file,
            )
            logger.info("Checkpoint saved: step %d/%d", state.global_step, state.max_steps)

    class BestAdapterCallback(TrainerCallback):
        """Export the adapter with the lowest dev loss seen across all sessions."""

        def __init__(self):
            record = read_best_metric(adapter_best)
            self.best_loss = record["eval_loss"] if record else math.inf

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            eval_loss = (metrics or {}).get("eval_loss")

            if eval_loss is None or model is None or eval_loss >= self.best_loss:
                return

            self.best_loss = eval_loss
            record = {
                "eval_loss": eval_loss,
                "global_step": state.global_step,
                "saved_at_utc": _utc_now(),
            }
            save_best_adapter(model, adapter_best, record)
            logger.info("New best dev loss %.6f at step %d", eval_loss, state.global_step)

    return [ProgressCallback(), BestAdapterCallback()]


def describe_checkpoint(path: Path) -> dict[str, Any]:
    """Summarize a checkpoint's trainer state for logs and reports."""
    state = read_json(path / "trainer_state.json")

    return {
        "path": str(path),
        "global_step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
    }
