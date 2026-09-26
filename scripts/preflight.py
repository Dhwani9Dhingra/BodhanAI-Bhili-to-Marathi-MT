"""
Run project preflight.

Usage
-----

Part 1, before AIKosh is downloaded:

    python -m scripts.preflight \
        --config configs/smoke.yaml

Part 2+, when the dataset must exist:

    python -m scripts.preflight \
        --config configs/smoke.yaml \
        --require-data
"""

from __future__ import annotations

import argparse
import sys

from bodhan_bhili.core.config import (
    load_config,
)
from bodhan_bhili.core.constants import (
    STAGE_PREFLIGHT,
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
from bodhan_bhili.core.preflight import (
    PreflightRunner,
)
from bodhan_bhili.core.reproducibility import (
    seed_everything,
)
from bodhan_bhili.core.run_state import (
    RunStateTracker,
)
from bodhan_bhili.core.serialization import (
    atomic_write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the environment before "
            "Bodhan QLoRA fine-tuning."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke.yaml",
        help=(
            "Path to experiment YAML."
        ),
    )

    parser.add_argument(
        "--require-data",
        action="store_true",
        help=(
            "Treat a missing AIKosh dataset "
            "as a fatal error."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Execute preflight and return a shell exit code."""

    args = parse_args()

    # --------------------------------------------------------
    # VALIDATE CONFIG BEFORE ANY EXPENSIVE WORK
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    # --------------------------------------------------------
    # STANDARD ARTIFACT TREE
    # --------------------------------------------------------

    paths = ArtifactPaths.from_config(
        config
    )

    paths.create_directories()

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logger = setup_logging(
        paths.pipeline_log
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "BODHAN BHILI MT — PREFLIGHT"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "Run ID: %s",
        config.run.run_id,
    )

    logger.info(
        "Direction: Dehwali Bhili -> Marathi"
    )

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed_everything(
        config.project.seed
    )

    # Save the COMPLETE RESOLVED config.
    #
    # This means later we know exactly which values were used,
    # including environment overrides.
    atomic_write_json(
        config.model_dump(
            mode="json"
        ),
        paths.resolved_config,
    )

    # --------------------------------------------------------
    # RUN STATE
    # --------------------------------------------------------

    state = RunStateTracker(
        state_file=paths.run_state,
        run_id=config.run.run_id,
    )

    state.mark_started(
        STAGE_PREFLIGHT,
        message="Running environment checks.",
    )

    # --------------------------------------------------------
    # EXPERIMENT MANIFEST
    # --------------------------------------------------------

    manifest = build_initial_manifest(
        config
    )

    write_manifest(
        manifest,
        paths,
    )

    # --------------------------------------------------------
    # ACTUAL CHECKS
    # --------------------------------------------------------

    runner = PreflightRunner(
        config=config,
        paths=paths,
        require_data=args.require_data,
    )

    report = runner.run()

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if report[
        "ready"
    ]:

        state.mark_completed(
            STAGE_PREFLIGHT,
            message=(
                "Environment passed mandatory "
                "preflight checks."
            ),
        )

        manifest[
            "status"
        ] = "preflight_passed"

        write_manifest(
            manifest,
            paths,
        )

        logger.info(
            "=" * 72
        )

        logger.info(
            "PREFLIGHT STATUS: READY"
        )

        logger.info(
            "Warnings: %d",
            report[
                "warning_count"
            ],
        )

        logger.info(
            "Report: %s",
            paths.preflight_report,
        )

        logger.info(
            "=" * 72
        )

        return 0

    state.mark_failed(
        STAGE_PREFLIGHT,
        message=(
            f"{report['failure_count']} mandatory "
            f"preflight check(s) failed."
        ),
    )

    manifest[
        "status"
    ] = "preflight_failed"

    write_manifest(
        manifest,
        paths,
    )

    logger.error(
        "=" * 72
    )

    logger.error(
        "PREFLIGHT STATUS: NOT READY"
    )

    logger.error(
        "Failures: %d",
        report[
            "failure_count"
        ],
    )

    logger.error(
        "Fix the failed checks before "
        "downloading/training Bodhan."
    )

    logger.error(
        "=" * 72
    )

    return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )