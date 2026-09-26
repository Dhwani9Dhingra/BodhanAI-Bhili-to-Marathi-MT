"""
End-to-end Bodhan QLoRA smoke test.

This script proves:

    base inference
        ->
    LoRA attachment
        ->
    real optimizer updates
        ->
    adapter save
        ->
    training model destruction
        ->
    fresh Bodhan reload
        ->
    adapter reload
        ->
    successful inference

It is NOT the final training pipeline.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

from bodhan_bhili.core.config import (
    load_config,
)
from bodhan_bhili.core.logging import (
    setup_logging,
)
from bodhan_bhili.core.manifest import (
    build_initial_manifest,
    write_manifest,
)
from bodhan_bhili.core.paths import (
    ArtifactPaths,
)
from bodhan_bhili.core.serialization import (
    atomic_write_json,
    read_json,
)
from bodhan_bhili.model import (
    attach_qlora,
    encode_training_example,
    load_processor,
    load_quantized_base_model,
    run_smoke_optimizer_steps,
    trainable_parameter_digest,
    translate,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end Bodhan QLoRA "
            "model smoke test."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke.yaml",
        help=(
            "Experiment configuration. "
            "Use configs/smoke.yaml for Part 2."
        ),
    )

    parser.add_argument(
        "--eval-examples",
        type=int,
        default=5,
        help=(
            "Number of frozen-test Bhili sentences "
            "used for before/after generation."
        ),
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        default=32,
        help=(
            "Maximum number of short training "
            "examples prepared for the smoke loop."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of optimizer steps. "
            "Default: min(config max_steps, 3)."
        ),
    )

    return parser.parse_args()


def release_gpu_memory() -> None:
    """Ask Python and PyTorch to release old model allocations."""

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

    except Exception:
        pass


def require_prepared_data(
    paths: ArtifactPaths,
) -> None:
    """Fail clearly when Part 1 has not been run for this config/run ID."""

    missing = [
        path
        for path
        in (
            paths.train_data,
            paths.test_data,
        )
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Prepared Part 1 data is missing for this run. "
            "Run scripts.prepare_data using the SAME config first. "
            f"Missing: {[str(path) for path in missing]}"
        )


def load_prepared_split(
    path: Path,
) -> pd.DataFrame:
    """Read one canonical Part 1 TSV."""

    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "record_id",
        "group_id",
        "source",
        "target",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:

        raise ValueError(
            f"Prepared dataset {path} is missing "
            f"columns: {sorted(missing)}"
        )

    return frame


def save_predictions(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Persist smoke-test predictions."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        destination,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def main() -> int:
    """Execute the complete Part 2 smoke test."""

    args = parse_args()

    config = load_config(
        args.config
    )

    paths = (
        ArtifactPaths
        .from_config(
            config
        )
    )

    paths.create_directories()

    logger = setup_logging(
        paths.pipeline_log
    )

    smoke_report_path = (
        paths.reports
        / "model_smoke_report.json"
    )

    trainable_report_path = (
        paths.reports
        / "trainable_parameters.json"
    )

    predictions_path = (
        paths.evaluation
        / "model_smoke_predictions.csv"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "BODHAN BHILI MT — PART 2 MODEL SMOKE TEST"
    )

    logger.info(
        "=" * 72
    )

    try:

        # --------------------------------------------------------------
        # VERIFY PART 1 OUTPUTS
        # --------------------------------------------------------------

        require_prepared_data(
            paths
        )

        train_frame = (
            load_prepared_split(
                paths.train_data
            )
        )

        test_frame = (
            load_prepared_split(
                paths.test_data
            )
        )

        # --------------------------------------------------------------
        # FIXED SMALL TEST SAMPLE
        # --------------------------------------------------------------

        eval_count = min(
            args.eval_examples,
            len(
                test_frame
            ),
        )

        if eval_count <= 0:
            raise ValueError(
                "No test examples are available."
            )

        eval_frame = (
            test_frame
            .sample(
                n=eval_count,

                random_state=(
                    config
                    .project
                    .seed
                ),
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        # --------------------------------------------------------------
        # LOAD OFFICIAL PROCESSOR + 4-BIT BASE MODEL
        # --------------------------------------------------------------

        logger.info(
            "Loading Bodhan processor..."
        )

        processor = (
            load_processor(
                config
            )
        )

        logger.info(
            "Loading Bodhan in 4-bit NF4..."
        )

        model = (
            load_quantized_base_model(
                config
            )
        )

        # --------------------------------------------------------------
        # BASELINE GENERATION
        # --------------------------------------------------------------

        base_predictions = []

        for (
            index,
            row,
        ) in eval_frame.iterrows():

            logger.info(
                "Base translation %d/%d",
                index + 1,
                eval_count,
            )

            prediction = translate(
                model,
                processor,
                row[
                    "source"
                ],

                target_language=(
                    config
                    .model
                    .target_language
                ),

                max_new_tokens=(
                    config
                    .model
                    .max_new_tokens
                ),
            )

            base_predictions.append(
                prediction
            )

        eval_frame[
            "base_prediction"
        ] = base_predictions

        # --------------------------------------------------------------
        # ATTACH QLORA
        # --------------------------------------------------------------

        logger.info(
            "Discovering safe LoRA target modules..."
        )

        (
            model,
            trainable_report,
        ) = attach_qlora(
            model,
            config,
        )

        atomic_write_json(
            trainable_report,
            trainable_report_path,
        )

        logger.info(
            "Matched LoRA modules: %d",
            trainable_report[
                "target_module_count"
            ],
        )

        logger.info(
            "Trainable parameters: %d (%.6f%%)",
            trainable_report[
                "trainable_parameters"
            ],
            trainable_report[
                "trainable_percentage"
            ],
        )

        # --------------------------------------------------------------
        # SAVE INITIAL ADAPTER
        # --------------------------------------------------------------
        #
        # This snapshot is BEFORE optimizer update #1.

        model.save_pretrained(
            paths.adapter_initial,
            safe_serialization=True,
        )

        processor.save_pretrained(
            paths.adapter_initial
        )

        initial_digest = (
            trainable_parameter_digest(
                model
            )
        )

        # --------------------------------------------------------------
        # BUILD TINY TRAINING SET
        # --------------------------------------------------------------

        # Randomize deterministically so the smoke test does not always
        # inspect the first rows of the original corpus.
        candidate_frame = (
            train_frame
            .sample(
                frac=1.0,

                random_state=(
                    config
                    .project
                    .seed
                ),
            )
            .reset_index(
                drop=True
            )
        )

        encoded_examples = []

        skipped_long = 0
        skipped_other = 0

        for (
            _,
            row,
        ) in candidate_frame.iterrows():

            (
                encoded,
                metadata,
            ) = encode_training_example(
                processor,

                source=(
                    row[
                        "source"
                    ]
                ),

                target=(
                    row[
                        "target"
                    ]
                ),

                target_language=(
                    config
                    .model
                    .target_language
                ),

                max_tokens=(
                    config
                    .model
                    .max_train_tokens
                ),
            )

            if encoded is None:

                if (
                    metadata[
                        "skip_reason"
                    ]
                    == "sequence_too_long"
                ):
                    skipped_long += 1

                else:
                    skipped_other += 1

                continue

            encoded_examples.append(
                encoded
            )

            if (
                len(
                    encoded_examples
                )
                >= args.train_examples
            ):
                break

        if not encoded_examples:

            raise RuntimeError(
                "No training examples fit within "
                f"{config.model.max_train_tokens} tokens."
            )

        logger.info(
            "Prepared %d smoke-training examples.",
            len(
                encoded_examples
            ),
        )

        # --------------------------------------------------------------
        # REAL OPTIMIZER UPDATES
        # --------------------------------------------------------------

        optimizer_steps = (
            args.steps
            if args.steps is not None
            else min(
                config
                .training
                .max_steps,
                3,
            )
        )

        logger.info(
            "Running %d real optimizer step(s)...",
            optimizer_steps,
        )

        training_report = (
            run_smoke_optimizer_steps(
                model,
                encoded_examples,
                config,

                optimizer_steps=(
                    optimizer_steps
                ),
            )
        )

        # --------------------------------------------------------------
        # PROVE ADAPTER WEIGHTS CHANGED
        # --------------------------------------------------------------

        final_digest = (
            trainable_parameter_digest(
                model
            )
        )

        adapters_changed = (
            initial_digest
            != final_digest
        )

        if not adapters_changed:

            raise RuntimeError(
                "Optimizer steps completed but LoRA parameter "
                "digest did not change."
            )

        logger.info(
            "LoRA weight digest changed successfully."
        )

        # --------------------------------------------------------------
        # SAVE TRAINED SMOKE ADAPTER
        # --------------------------------------------------------------

        model.save_pretrained(
            paths.adapter_final,
            safe_serialization=True,
        )

        processor.save_pretrained(
            paths.adapter_final
        )

        # --------------------------------------------------------------
        # DESTROY TRAINING MODEL
        # --------------------------------------------------------------
        #
        # This step is mandatory.
        #
        # Otherwise a successful inference test could accidentally be
        # using the adapter already resident in memory.

        del model
        del processor

        release_gpu_memory()

        logger.info(
            "Training model destroyed. "
            "Loading fresh Bodhan base model..."
        )

        # --------------------------------------------------------------
        # FRESH BASE + SAVED ADAPTER
        # --------------------------------------------------------------

        processor = (
            load_processor(
                config
            )
        )

        fresh_base = (
            load_quantized_base_model(
                config
            )
        )

        from peft import (
            PeftModel,
        )

        reloaded_model = (
            PeftModel
            .from_pretrained(
                fresh_base,

                str(
                    paths.adapter_final
                ),

                is_trainable=False,
            )
        )

        reloaded_model.eval()

        # --------------------------------------------------------------
        # AFTER-RELOAD GENERATION
        # --------------------------------------------------------------

        tuned_predictions = []

        for (
            index,
            row,
        ) in eval_frame.iterrows():

            logger.info(
                "Reloaded-adapter translation %d/%d",
                index + 1,
                eval_count,
            )

            prediction = translate(
                reloaded_model,
                processor,
                row[
                    "source"
                ],

                target_language=(
                    config
                    .model
                    .target_language
                ),

                max_new_tokens=(
                    config
                    .model
                    .max_new_tokens
                ),
            )

            tuned_predictions.append(
                prediction
            )

        eval_frame[
            "tuned_prediction"
        ] = tuned_predictions

        changed_predictions = int(
            (
                eval_frame[
                    "base_prediction"
                ]
                !=
                eval_frame[
                    "tuned_prediction"
                ]
            )
            .sum()
        )

        save_predictions(
            eval_frame[
                [
                    "record_id",
                    "group_id",
                    "source",
                    "target",
                    "base_prediction",
                    "tuned_prediction",
                ]
            ],
            predictions_path,
        )

        # --------------------------------------------------------------
        # FINAL SMOKE REPORT
        # --------------------------------------------------------------

        smoke_report = {
            "status":
                "passed",

            "model":
                config
                .model
                .repo_id,

            "direction":
                config
                .project
                .direction,

            "quantization": {
                "bits":
                    4,

                "quant_type":
                    config
                    .quantization
                    .quant_type,

                "double_quant":
                    config
                    .quantization
                    .double_quant,
            },

            "lora": {
                "rank":
                    config
                    .lora
                    .rank,

                "alpha":
                    config
                    .lora
                    .alpha,

                "dropout":
                    config
                    .lora
                    .dropout,

                "target_module_count":
                    trainable_report[
                        "target_module_count"
                    ],

                "trainable_parameters":
                    trainable_report[
                        "trainable_parameters"
                    ],

                "trainable_percentage":
                    trainable_report[
                        "trainable_percentage"
                    ],
            },

            "training":
                training_report,

            "training_examples_used":
                len(
                    encoded_examples
                ),

            "skipped_too_long":
                skipped_long,

            "skipped_other":
                skipped_other,

            "adapter_initial_digest":
                initial_digest,

            "adapter_final_digest":
                final_digest,

            "adapter_weights_changed":
                adapters_changed,

            "fresh_reload_verified":
                True,

            "evaluation_examples":
                eval_count,

            # This is informational only.
            # We intentionally do NOT require outputs to change after
            # only a few optimizer steps.
            "predictions_changed_after_smoke_training":
                changed_predictions,

            "adapter_initial":
                str(
                    paths.adapter_initial
                ),

            "adapter_final":
                str(
                    paths.adapter_final
                ),

            "predictions_file":
                str(
                    predictions_path
                ),
        }

        atomic_write_json(
            smoke_report,
            smoke_report_path,
        )

        # --------------------------------------------------------------
        # UPDATE EXPERIMENT MANIFEST
        # --------------------------------------------------------------

        if paths.manifest.exists():

            manifest = (
                read_json(
                    paths.manifest
                )
            )

        else:

            manifest = (
                build_initial_manifest(
                    config
                )
            )

        manifest[
            "status"
        ] = "model_smoke_passed"

        manifest[
            "trainable_parameters"
        ] = {
            "count":
                trainable_report[
                    "trainable_parameters"
                ],

            "percentage":
                trainable_report[
                    "trainable_percentage"
                ],

            "target_module_count":
                trainable_report[
                    "target_module_count"
                ],
        }

        manifest[
            "model_smoke"
        ] = {
            "status":
                "passed",

            "optimizer_steps":
                training_report[
                    "optimizer_steps"
                ],

            "adapter_weights_changed":
                adapters_changed,

            "fresh_reload_verified":
                True,
        }

        write_manifest(
            manifest,
            paths,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "PART 2 SMOKE STATUS: PASSED"
        )

        logger.info(
            "Optimizer steps: %d",
            training_report[
                "optimizer_steps"
            ],
        )

        logger.info(
            "Adapter reload: VERIFIED"
        )

        logger.info(
            "Predictions changed: %d/%d",
            changed_predictions,
            eval_count,
        )

        logger.info(
            "Smoke report: %s",
            smoke_report_path,
        )

        logger.info(
            "=" * 72
        )

        return 0

    except Exception as exc:

        failure_report = {
            "status":
                "failed",

            "error_type":
                type(
                    exc
                ).__name__,

            "error":
                str(
                    exc
                ),
        }

        atomic_write_json(
            failure_report,
            smoke_report_path,
        )

        logger.exception(
            "Part 2 smoke test failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )