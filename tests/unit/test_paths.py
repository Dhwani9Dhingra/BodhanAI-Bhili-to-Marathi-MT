"""Tests for deterministic experiment artifact paths."""

import os
from pathlib import Path

import pytest

from bodhan_bhili.core.config import load_config
from bodhan_bhili.core.paths import ArtifactPaths


def test_artifact_tree_uses_run_id(monkeypatch, tmp_path):
    """Every experiment must remain isolated by run ID."""
    monkeypatch.setenv("BODHAN_ARTIFACT_ROOT", str(tmp_path))

    config = load_config("configs/smoke.yaml")
    paths = ArtifactPaths.from_config(config)

    assert paths.run_root == tmp_path / "bodhan-bhili-mt" / "smoke_001"


def test_artifact_tree_can_be_created(monkeypatch, tmp_path):
    """All expected folders should be creatable."""
    monkeypatch.setenv("BODHAN_ARTIFACT_ROOT", str(tmp_path))

    config = load_config("configs/smoke.yaml")
    paths = ArtifactPaths.from_config(config)

    paths.create_directories()

    assert paths.logs.is_dir()

    assert paths.reports.is_dir()

    assert paths.trainer_checkpoints.is_dir()

    assert paths.adapter_initial.is_dir()

    assert paths.adapter_best.is_dir()

    assert paths.adapter_final.is_dir()


def test_unmounted_drive_does_not_create_artifacts(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config.paths.artifact_root = Path("/content/drive/MyDrive/BodhanAI/artifacts")
    paths = ArtifactPaths.from_config(config)
    created = []
    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: created.append(self))

    with pytest.raises(RuntimeError, match="Mount Google Drive"):
        paths.create_directories()

    assert created == []


def test_mounted_drive_keeps_outputs_in_one_run(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config.paths.artifact_root = Path("/content/drive/MyDrive/BodhanAI/artifacts")
    paths = ArtifactPaths.from_config(config)
    created = []
    monkeypatch.setattr(os.path, "ismount", lambda path: True)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: created.append(self))

    paths.create_directories()

    assert paths.data in created
    assert paths.evaluation in created
    assert paths.adapter_final in created
    assert all(path.is_relative_to(paths.root) for path in created)
    assert paths.smoke_predictions == paths.evaluation / "model_smoke_predictions.csv"
