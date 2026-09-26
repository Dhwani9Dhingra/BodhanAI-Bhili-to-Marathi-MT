"""
Standard experiment artifact layout.

Every stage receives the same ArtifactPaths object rather than inventing
its own output directories. This keeps training, evaluation, dashboards
and documentation aligned on one deterministic run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bodhan_bhili.core.config import ExperimentConfig


@dataclass(frozen=True)
class ArtifactPaths:
    """All filesystem paths belonging to one experiment run."""

    root: Path
    run_root: Path

    data: Path
    reports: Path
    logs: Path
    tensorboard: Path
    checkpoints: Path
    trainer_checkpoints: Path
    adapter_initial: Path
    adapter_best: Path
    adapter_final: Path
    smoke_adapter_initial: Path
    smoke_adapter_final: Path
    evaluation: Path
    weights: Path
    dashboard_exports: Path

    # ------------------------------------------------------------------
    # PREPARED DATASETS
    # ------------------------------------------------------------------
    # These are the frozen train/dev/test files created in Part 1.

    train_data: Path
    dev_data: Path
    test_data: Path

    # ------------------------------------------------------------------
    # DATA AUDIT / REPRODUCIBILITY
    # ------------------------------------------------------------------

    data_report: Path
    data_cleaning_report: Path
    split_manifest: Path
    test_set_sha256: Path

    # ------------------------------------------------------------------
    # MODEL / SMOKE-TEST REPORTS
    # ------------------------------------------------------------------

    trainable_parameters_report: Path
    model_smoke_report: Path
    smoke_predictions: Path
    smoke_weight_changes: Path

    # ------------------------------------------------------------------
    # GENERAL EXPERIMENT METADATA
    # ------------------------------------------------------------------

    resolved_config: Path
    manifest: Path
    run_state: Path
    preflight_report: Path
    pipeline_log: Path

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
    ) -> "ArtifactPaths":
        """Construct all paths deterministically from project name + run ID."""

        root = Path(
            config.paths.artifact_root
        )

        run_root = (
            root
            / config.project.name
            / config.run.run_id
        )

        data = (
            run_root
            / "data"
        )

        reports = (
            run_root
            / "reports"
        )

        logs = (
            run_root
            / "logs"
        )

        checkpoints = (
            run_root
            / "checkpoints"
        )

        return cls(
            root=root,

            run_root=run_root,

            data=data,

            reports=reports,

            logs=logs,

            tensorboard=(
                run_root
                / "tensorboard"
            ),

            checkpoints=checkpoints,

            trainer_checkpoints=(
                checkpoints
                / "trainer"
            ),

            # Snapshot immediately after LoRA attachment,
            # before optimizer update #1.
            adapter_initial=(
                checkpoints
                / "adapter_initial"
            ),

            # Best adapter according to the development MT metric.
            adapter_best=(
                checkpoints
                / "adapter_best"
            ),

            # Final adapter at the end of training.
            adapter_final=(
                checkpoints
                / "adapter_final"
            ),

            smoke_adapter_initial=(
                checkpoints
                / "smoke_adapter_initial"
            ),

            smoke_adapter_final=(
                checkpoints
                / "smoke_adapter_final"
            ),

            evaluation=(
                run_root
                / "evaluation"
            ),

            weights=(
                run_root
                / "weights"
            ),

            dashboard_exports=(
                run_root
                / "dashboard_exports"
            ),

            # ----------------------------------------------------------
            # PREPARED DATA
            # ----------------------------------------------------------

            train_data=(
                data
                / "train.tsv"
            ),

            dev_data=(
                data
                / "dev.tsv"
            ),

            test_data=(
                data
                / "test.tsv"
            ),

            # ----------------------------------------------------------
            # DATA REPORTS
            # ----------------------------------------------------------

            data_report=(
                reports
                / "data_report.json"
            ),

            data_cleaning_report=(
                reports
                / "data_cleaning_report.csv"
            ),

            split_manifest=(
                reports
                / "split_manifest.json"
            ),

            test_set_sha256=(
                reports
                / "test_set.sha256"
            ),

            # ----------------------------------------------------------
            # MODEL / SMOKE-TEST REPORTS
            # ----------------------------------------------------------

            trainable_parameters_report=(
                reports
                / "trainable_parameters.json"
            ),

            model_smoke_report=(
                reports
                / "model_smoke_report.json"
            ),

            smoke_predictions=(
                run_root
                / "evaluation"
                / "smoke_predictions.tsv"
            ),

            smoke_weight_changes=(
                run_root
                / "weights"
                / "smoke_weight_changes.csv"
            ),

            # ----------------------------------------------------------
            # GENERAL METADATA
            # ----------------------------------------------------------

            resolved_config=(
                reports
                / "resolved_config.json"
            ),

            manifest=(
                reports
                / "experiment_manifest.json"
            ),

            run_state=(
                reports
                / "run_state.json"
            ),

            preflight_report=(
                reports
                / "preflight.json"
            ),

            pipeline_log=(
                logs
                / "pipeline.log"
            ),
        )

    def create_directories(
        self,
    ) -> None:
        """Create the complete artifact tree safely and idempotently."""

        directories = (
            self.root,
            self.run_root,
            self.data,
            self.reports,
            self.logs,
            self.tensorboard,
            self.checkpoints,
            self.trainer_checkpoints,
            self.adapter_initial,
            self.adapter_best,
            self.adapter_final,
            self.smoke_adapter_initial,
            self.smoke_adapter_final,
            self.evaluation,
            self.weights,
            self.dashboard_exports,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )