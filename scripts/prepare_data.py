"""Prepare the Project Astitva Dehwali Bhili -> Marathi dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bodhan_bhili.core.config import load_config
from bodhan_bhili.core.constants import STAGE_DATA_PREPARATION
from bodhan_bhili.core.logging import setup_logging
from bodhan_bhili.core.manifest import build_initial_manifest, write_manifest
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.run_state import RunStateTracker
from bodhan_bhili.core.serialization import atomic_write_json, read_json
from bodhan_bhili.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Clean, audit and split the Dehwali Bhili -> Marathi TSV.")
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/colab_t4.yaml",
        help=("Experiment YAML configuration."),
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Optional raw TSV path. "
            "Overrides paths.raw_data_file and "
            "BODHAN_DATA_FILE for this invocation."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Execute Part 1 data preparation."""
    args = parse_args()
    config = load_config(args.config)

    if args.data:
        config.paths.raw_data_file = Path(args.data).expanduser().resolve()
    paths = ArtifactPaths.from_config(config)

    paths.create_directories()

    logger = setup_logging(paths.pipeline_log)

    atomic_write_json(config.model_dump(mode="json"), paths.resolved_config)

    state = RunStateTracker(state_file=paths.run_state, run_id=config.run.run_id)

    state.mark_started(
        STAGE_DATA_PREPARATION, message=("Cleaning, auditing and splitting the Bhili-Marathi TSV.")
    )

    if paths.manifest.exists():
        manifest = read_json(paths.manifest)

    else:
        manifest = build_initial_manifest(config)

    logger.info("=" * 72)

    logger.info("BODHAN BHILI MT — DATA PREPARATION")

    logger.info("=" * 72)

    logger.info("Run ID: %s", config.run.run_id)

    logger.info("Raw dataset: %s", config.paths.raw_data_file)

    try:
        report = prepare_dataset(config, paths)

        state.mark_completed(
            STAGE_DATA_PREPARATION,
            message=(
                "Prepared "
                f"{report['cleaning_result']['rows_kept']} rows "
                "with leakage-safe train/dev/test splits."
            ),
        )

        manifest["status"] = "data_prepared"

        manifest["dataset"] = {
            "raw_file": report["raw_dataset"]["file"],
            "raw_sha256": report["raw_dataset"]["sha256"],
            "raw_rows": report["raw_dataset"]["rows"],
            "clean_rows": report["cleaning_result"]["rows_kept"],
            "rows_dropped": report["cleaning_result"]["rows_dropped"],
            "clean_groups": report["cleaning_result"]["unique_groups_kept"],
            "splits": report["splits"],
        }

        write_manifest(manifest, paths)

        logger.info(
            "Rows: %d raw -> %d clean",
            report["raw_dataset"]["rows"],
            report["cleaning_result"]["rows_kept"],
        )

        logger.info("Dropped rows: %d", report["cleaning_result"]["rows_dropped"])

        logger.info("Drop reasons: %s", report["cleaning_result"]["drop_counts"])

        logger.info(
            "Split rows: train=%d | dev=%d | test=%d",
            report["splits"]["train"]["rows"],
            report["splits"]["dev"]["rows"],
            report["splits"]["test"]["rows"],
        )

        logger.info("Frozen test SHA-256: %s", report["splits"]["test"]["sha256"])

        logger.info("Data report: %s", paths.data_report)

        logger.info("=" * 72)

        return 0

    except Exception as exc:
        state.mark_failed(STAGE_DATA_PREPARATION, message=(f"{type(exc).__name__}: {exc}"))

        manifest["status"] = "data_preparation_failed"

        write_manifest(manifest, paths)

        logger.exception("Data preparation failed: %s", exc)

        return 1


if __name__ == "__main__":
    sys.exit(main())
