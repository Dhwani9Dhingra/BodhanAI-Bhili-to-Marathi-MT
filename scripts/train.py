"""Resumable Bodhan QLoRA training for Bhili -> Marathi.

Re-running the same command after a Colab disconnect resumes from the newest
complete checkpoint in <run>/checkpoints/trainer on Google Drive.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import pandas as pd

from bodhan_bhili.core.config import load_config
from bodhan_bhili.core.constants import STAGE_TRAINING
from bodhan_bhili.core.logging import setup_logging
from bodhan_bhili.core.manifest import build_initial_manifest, write_manifest
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.reproducibility import seed_everything
from bodhan_bhili.core.run_state import RunStateTracker
from bodhan_bhili.core.serialization import atomic_write_json, read_json
from bodhan_bhili.data import sha256_file
from bodhan_bhili.model import (
    attach_qlora,
    encode_training_example,
    load_processor,
    load_quantized_base_model,
)
from bodhan_bhili.training import (
    BEST_METRIC_FILE,
    build_callbacks,
    build_training_fingerprint,
    describe_checkpoint,
    find_resume_checkpoint,
    pad_training_batch,
    quarantine_checkpoints,
    read_best_metric,
    snapshot_adapter,
    to_training_feature,
    verify_resume_fingerprint,
    warmup_steps,
)
from scripts.smoke_model import load_prepared_split, record_gpu_memory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Train (or resume) the Bodhan Bhili -> Marathi QLoRA adapter.")
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/colab_t4.yaml",
        help=("Experiment YAML configuration. Use the SAME config as prepare_data."),
    )

    parser.add_argument(
        "--extend",
        action="store_true",
        help=(
            "Continue a finished run after raising training.max_steps. No other setting may change."
        ),
    )

    return parser.parse_args()


def require_training_data(paths: ArtifactPaths) -> None:
    """Fail clearly when Part 1 has not been run for this config/run ID."""
    missing = [path for path in (paths.train_data, paths.dev_data) if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Prepared Part 1 data is missing for this run. "
            "Run scripts.prepare_data using the SAME config first. "
            f"Missing: {[str(path) for path in missing]}"
        )


def encode_split(processor, frame: pd.DataFrame, config, *, limit: int | None, logger, name: str):
    """Tokenize a prepared split, skipping sequences longer than max_train_tokens."""
    features = []
    skipped: dict[str, int] = {}

    for _, row in frame.iterrows():
        (encoded, metadata) = encode_training_example(
            processor,
            source=(row["source"]),
            target=(row["target"]),
            target_language=(config.model.target_language),
            max_tokens=(config.model.max_train_tokens),
        )

        if encoded is None:
            reason = metadata["skip_reason"]
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        features.append(to_training_feature(encoded))

        if limit is not None and len(features) >= limit:
            break

    if not features:
        raise RuntimeError(f"No {name} examples fit within {config.model.max_train_tokens} tokens.")

    logger.info("Encoded %d %s examples (skipped: %s).", len(features), name, skipped or "none")

    return (features, skipped)


def build_training_arguments(config, paths: ArtifactPaths):
    """Map the experiment config onto Hugging Face TrainingArguments."""
    from transformers import TrainingArguments

    training = config.training
    kwargs = {
        "output_dir": str(paths.trainer_checkpoints),
        "max_steps": training.max_steps,
        "per_device_train_batch_size": training.train_batch_size,
        "per_device_eval_batch_size": training.eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "warmup_steps": warmup_steps(config),
        "weight_decay": training.weight_decay,
        "optim": training.optimizer,
        "logging_strategy": "steps",
        "logging_steps": training.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": training.evaluation_steps,
        "save_strategy": "steps",
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "seed": config.project.seed,
        "data_seed": config.project.seed,
        # attach_qlora already enabled non-reentrant checkpointing and set adapter dtypes;
        # mixed precision stays off to match the verified smoke-test numerics.
        "gradient_checkpointing": False,
        "fp16": False,
        "bf16": False,
        # PEFT wrappers hide `labels` from signature inspection; without this, eval_loss is None.
        "label_names": ["labels"],
        # Accumulating full-vocabulary logits during evaluation would exhaust the T4.
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": ["tensorboard"],
    }

    field_names = {field.name for field in dataclasses.fields(TrainingArguments)}

    if "logging_dir" in field_names:
        kwargs["logging_dir"] = str(paths.tensorboard)

    else:
        os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(paths.tensorboard))

    return TrainingArguments(**kwargs)


def main() -> int:
    """Execute resumable QLoRA training."""
    args = parse_args()
    config = load_config(args.config)
    paths = ArtifactPaths.from_config(config)

    paths.create_directories()

    logger = setup_logging(paths.pipeline_log)
    state = RunStateTracker(state_file=paths.run_state, run_id=config.run.run_id)
    gpu_memory = {}

    if paths.manifest.exists():
        manifest = read_json(paths.manifest)

    else:
        manifest = build_initial_manifest(config)

    logger.info("=" * 72)

    logger.info("BODHAN BHILI MT — QLORA TRAINING")

    logger.info("=" * 72)

    logger.info("Run ID: %s", config.run.run_id)

    try:
        require_training_data(paths)

        fingerprint = build_training_fingerprint(
            config,
            train_sha256=sha256_file(paths.train_data),
            dev_sha256=sha256_file(paths.dev_data),
        )

        (resume_from, incomplete) = find_resume_checkpoint(paths.trainer_checkpoints)

        for moved in quarantine_checkpoints(incomplete, paths.incomplete_checkpoints):
            logger.warning("Moved incomplete checkpoint aside: %s", moved)

        extended_from = None

        if resume_from is not None:
            extended_from = verify_resume_fingerprint(
                paths.training_fingerprint, fingerprint, allow_extension=args.extend
            )
            resume_info = describe_checkpoint(resume_from)

            if resume_info["global_step"] >= config.training.max_steps:
                logger.info(
                    "Training already completed at %s (max_steps=%d). Nothing to do. "
                    "To train longer, raise training.max_steps and pass --extend.",
                    resume_from.name,
                    config.training.max_steps,
                )
                return 0

            if extended_from is not None:
                snapshot = paths.checkpoints / f"adapter_step_{resume_info['global_step']}"

                if snapshot_adapter(resume_from, snapshot):
                    logger.info("Preserved pre-extension adapter: %s", snapshot)

                atomic_write_json(fingerprint, paths.training_fingerprint)
                manifest.setdefault("training_extensions", []).append(
                    {
                        "from_max_steps": extended_from,
                        "to_max_steps": config.training.max_steps,
                        "resumed_from": resume_from.name,
                        "preserved_adapter": str(snapshot),
                    }
                )
                logger.info(
                    "Extending training: max_steps %d -> %d.",
                    extended_from,
                    config.training.max_steps,
                )

            logger.info(
                "Resuming from %s (step %s/%s).",
                resume_from.name,
                resume_info["global_step"],
                config.training.max_steps,
            )
            state.mark_started(STAGE_TRAINING, message=f"Resuming from {resume_from.name}.")

        else:
            atomic_write_json(fingerprint, paths.training_fingerprint)
            stale_best = paths.adapter_best / BEST_METRIC_FILE

            if stale_best.exists():
                # A best score from an abandoned attempt must not block this run's best adapter.
                stale_best.unlink()
                logger.warning("Cleared stale best-adapter record: %s", stale_best)

            logger.info("No complete checkpoint found. Starting training from step 0.")
            state.mark_started(STAGE_TRAINING, message="Starting training from step 0.")

        atomic_write_json(config.model_dump(mode="json"), paths.resolved_config)
        seed_everything(config.project.seed)

        train_frame = load_prepared_split(paths.train_data)
        dev_frame = (
            load_prepared_split(paths.dev_data)
            .sample(frac=1.0, random_state=(config.project.seed))
            .reset_index(drop=True)
        )

        logger.info("Loading Bodhan processor...")

        processor = load_processor(config)
        (train_features, train_skipped) = encode_split(
            processor, train_frame, config, limit=None, logger=logger, name="train"
        )
        (dev_features, dev_skipped) = encode_split(
            processor,
            dev_frame,
            config,
            limit=(config.training.eval_examples),
            logger=logger,
            name="dev",
        )

        logger.info("Loading Bodhan in 4-bit NF4...")

        record_gpu_memory("before_base_load", gpu_memory)
        model = load_quantized_base_model(config)
        record_gpu_memory("after_base_load", gpu_memory)
        (model, trainable_report) = attach_qlora(model, config)
        record_gpu_memory("after_qlora_setup", gpu_memory)

        atomic_write_json(trainable_report, paths.trainable_parameters_report)

        logger.info(
            "Trainable parameters: %d (%.6f%%)",
            trainable_report["trainable_parameters"],
            trainable_report["trainable_percentage"],
        )

        from transformers import Trainer

        tokenizer = getattr(processor, "tokenizer", processor)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        trainer = Trainer(
            model=model,
            args=build_training_arguments(config, paths),
            train_dataset=train_features,
            eval_dataset=dev_features,
            data_collator=(lambda features: pad_training_batch(features, pad_token_id)),
            callbacks=build_callbacks(
                progress_file=paths.training_progress,
                adapter_best=paths.adapter_best,
                logger=logger,
            ),
        )

        record_gpu_memory("before_training", gpu_memory)
        train_output = trainer.train(
            resume_from_checkpoint=(str(resume_from) if resume_from is not None else None)
        )
        record_gpu_memory("after_training", gpu_memory)

        model.save_pretrained(paths.adapter_final, safe_serialization=True)
        processor.save_pretrained(paths.adapter_final)

        best = read_best_metric(paths.adapter_best)

        if best is not None:
            processor.save_pretrained(paths.adapter_best)

        training_report = {
            "status": "completed",
            "resumed_from": (str(resume_from) if resume_from is not None else None),
            "extended_from_max_steps": extended_from,
            "global_step": trainer.state.global_step,
            "max_steps": config.training.max_steps,
            "train_metrics": train_output.metrics,
            "best_dev": best,
            "train_examples": len(train_features),
            "train_skipped": train_skipped,
            "dev_examples": len(dev_features),
            "dev_skipped": dev_skipped,
            "gpu_memory": gpu_memory,
            "log_history": trainer.state.log_history,
            "adapter_final": str(paths.adapter_final),
            "adapter_best": (str(paths.adapter_best) if best is not None else None),
        }

        atomic_write_json(training_report, paths.training_report)

        state.mark_completed(
            STAGE_TRAINING, message=f"Completed {trainer.state.global_step} optimizer steps."
        )

        manifest["status"] = "training_completed"

        manifest["trainable_parameters"] = {
            "count": trainable_report["trainable_parameters"],
            "percentage": trainable_report["trainable_percentage"],
            "target_module_count": trainable_report["target_module_count"],
        }

        manifest["best_checkpoint"] = best

        write_manifest(manifest, paths)

        logger.info("=" * 72)

        logger.info("TRAINING STATUS: COMPLETED")

        logger.info("Optimizer steps: %d", trainer.state.global_step)

        logger.info("Best dev loss: %s", best)

        logger.info("Final adapter: %s", paths.adapter_final)

        logger.info("Training report: %s", paths.training_report)

        logger.info("=" * 72)

        return 0

    except BaseException as exc:
        # BaseException also covers a manual "stop cell" (KeyboardInterrupt).
        try:
            record_gpu_memory("failure", gpu_memory)
        except Exception:
            logger.warning("GPU memory snapshot unavailable after failure.")

        state.mark_failed(STAGE_TRAINING, message=(f"{type(exc).__name__}: {exc}"))

        manifest["status"] = "training_failed"

        write_manifest(manifest, paths)

        logger.exception("Training failed: %s", exc)

        logger.info("Re-run the same command to resume from the last complete checkpoint.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
