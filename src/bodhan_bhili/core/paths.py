"""Standard experiment artifact layout."""

from __future__ import annotations

import os
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
    evaluation: Path

    train_data: Path
    dev_data: Path
    test_data: Path

    data_report: Path
    data_cleaning_report: Path
    split_manifest: Path
    test_set_sha256: Path

    trainable_parameters_report: Path
    model_smoke_report: Path
    smoke_predictions: Path

    resolved_config: Path
    manifest: Path
    run_state: Path
    preflight_report: Path
    pipeline_log: Path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> ArtifactPaths:
        """Construct all paths deterministically from project name + run ID."""
        root = Path(config.paths.artifact_root)
        run_root = root / config.project.name / config.run.run_id
        data = run_root / "data"
        reports = run_root / "reports"
        logs = run_root / "logs"
        checkpoints = run_root / "checkpoints"

        return cls(
            root=root,
            run_root=run_root,
            data=data,
            reports=reports,
            logs=logs,
            tensorboard=(run_root / "tensorboard"),
            checkpoints=checkpoints,
            trainer_checkpoints=(checkpoints / "trainer"),
            adapter_initial=(checkpoints / "adapter_initial"),
            adapter_best=(checkpoints / "adapter_best"),
            adapter_final=(checkpoints / "adapter_final"),
            evaluation=(run_root / "evaluation"),
            train_data=(data / "train.tsv"),
            dev_data=(data / "dev.tsv"),
            test_data=(data / "test.tsv"),
            data_report=(reports / "data_report.json"),
            data_cleaning_report=(reports / "data_cleaning_report.csv"),
            split_manifest=(reports / "split_manifest.json"),
            test_set_sha256=(reports / "test_set.sha256"),
            trainable_parameters_report=(reports / "trainable_parameters.json"),
            model_smoke_report=(reports / "model_smoke_report.json"),
            smoke_predictions=(run_root / "evaluation" / "model_smoke_predictions.csv"),
            resolved_config=(reports / "resolved_config.json"),
            manifest=(reports / "experiment_manifest.json"),
            run_state=(reports / "run_state.json"),
            preflight_report=(reports / "preflight.json"),
            pipeline_log=(logs / "pipeline.log"),
        )

    def create_directories(self) -> None:
        """Create the complete artifact tree safely and idempotently."""
        # Avoid writing to a temporary Colab folder when Drive is not mounted.
        if self.root.as_posix().startswith("/content/drive/"):
            if not os.path.ismount("/content/drive"):
                raise RuntimeError(
                    "Mount Google Drive at /content/drive before running a stage. "
                    "On a laptop, set BODHAN_ARTIFACT_ROOT to a Google Drive synced folder."
                )

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
            self.evaluation,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
