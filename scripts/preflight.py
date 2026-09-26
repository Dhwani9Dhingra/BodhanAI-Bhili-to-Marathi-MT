"""Run project preflight."""

from __future__ import annotations

import argparse
import sys

from bodhan_bhili.core.config import load_config
from bodhan_bhili.core.constants import STAGE_PREFLIGHT
from bodhan_bhili.core.logging import setup_logging
from bodhan_bhili.core.manifest import build_initial_manifest, write_manifest
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.preflight import PreflightRunner
from bodhan_bhili.core.reproducibility import seed_everything
from bodhan_bhili.core.run_state import RunStateTracker
from bodhan_bhili.core.serialization import atomic_write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Validate the environment before Bodhan QLoRA fine-tuning.")
    )

    parser.add_argument(
        "--config", type=str, default="configs/smoke.yaml", help=("Path to experiment YAML.")
    )

    parser.add_argument(
        "--require-data",
        action="store_true",
        help=("Treat a missing AIKosh dataset as a fatal error."),
    )

    return parser.parse_args()


def main() -> int:
    """Execute preflight and return a shell exit code."""
    args = parse_args()
    config = load_config(args.config)
    paths = ArtifactPaths.from_config(config)

    paths.create_directories()

    logger = setup_logging(paths.pipeline_log)

    logger.info("=" * 72)

    logger.info("BODHAN BHILI MT — PREFLIGHT")

    logger.info("=" * 72)

    logger.info("Run ID: %s", config.run.run_id)

    logger.info("Direction: Dehwali Bhili -> Marathi")

    seed_everything(config.project.seed)

    atomic_write_json(config.model_dump(mode="json"), paths.resolved_config)

    state = RunStateTracker(state_file=paths.run_state, run_id=config.run.run_id)

    state.mark_started(STAGE_PREFLIGHT, message="Running environment checks.")

    manifest = build_initial_manifest(config)

    write_manifest(manifest, paths)

    runner = PreflightRunner(config=config, paths=paths, require_data=args.require_data)
    report = runner.run()

    if report["ready"]:
        state.mark_completed(
            STAGE_PREFLIGHT, message=("Environment passed mandatory preflight checks.")
        )

        manifest["status"] = "preflight_passed"

        write_manifest(manifest, paths)

        logger.info("=" * 72)

        logger.info("PREFLIGHT STATUS: READY")

        logger.info("Warnings: %d", report["warning_count"])

        logger.info("Report: %s", paths.preflight_report)

        logger.info("=" * 72)

        return 0

    state.mark_failed(
        STAGE_PREFLIGHT, message=(f"{report['failure_count']} mandatory preflight check(s) failed.")
    )

    manifest["status"] = "preflight_failed"

    write_manifest(manifest, paths)

    logger.error("=" * 72)

    logger.error("PREFLIGHT STATUS: NOT READY")

    logger.error("Failures: %d", report["failure_count"])

    logger.error("Fix the failed checks before downloading/training Bodhan.")

    logger.error("=" * 72)

    return 1


if __name__ == "__main__":
    sys.exit(main())
