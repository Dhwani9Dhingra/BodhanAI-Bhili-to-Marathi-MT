"""Tests for deterministic experiment artifact paths."""

from bodhan_bhili.core.config import (
    load_config,
)
from bodhan_bhili.core.paths import (
    ArtifactPaths,
)


def test_artifact_tree_uses_run_id(
    monkeypatch,
    tmp_path,
):
    """Every experiment must remain isolated by run ID."""

    monkeypatch.setenv(
        "BODHAN_ARTIFACT_ROOT",
        str(
            tmp_path
        ),
    )

    config = load_config(
        "configs/smoke.yaml"
    )

    paths = ArtifactPaths.from_config(
        config
    )

    assert (
        paths.run_root
        ==
        tmp_path
        / "bodhan-bhili-mt"
        / "smoke_001"
    )


def test_artifact_tree_can_be_created(
    monkeypatch,
    tmp_path,
):
    """All expected folders should be creatable."""

    monkeypatch.setenv(
        "BODHAN_ARTIFACT_ROOT",
        str(
            tmp_path
        ),
    )

    config = load_config(
        "configs/smoke.yaml"
    )

    paths = ArtifactPaths.from_config(
        config
    )

    paths.create_directories()

    assert paths.logs.is_dir()

    assert paths.reports.is_dir()

    assert paths.trainer_checkpoints.is_dir()

    assert paths.adapter_initial.is_dir()

    assert paths.adapter_best.is_dir()

    assert paths.adapter_final.is_dir()