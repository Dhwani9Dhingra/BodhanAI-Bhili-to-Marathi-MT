"""Tests for validated configuration loading."""

from pathlib import Path

from bodhan_bhili.core.config import (
    load_config,
)


def test_environment_can_override_artifact_root(
    monkeypatch,
    tmp_path: Path,
):
    """Colab should be able to redirect artifacts to Drive."""

    monkeypatch.setenv(
        "BODHAN_ARTIFACT_ROOT",
        str(
            tmp_path
            / "drive_artifacts"
        ),
    )

    config = load_config(
        "configs/smoke.yaml"
    )

    assert (
        config.paths.artifact_root
        ==
        tmp_path
        / "drive_artifacts"
    )


def test_main_direction_is_unidirectional():
    """Our primary project must remain Bhili -> Marathi."""

    config = load_config(
        "configs/smoke.yaml"
    )

    assert (
        config.project.direction
        ==
        "bhili_to_marathi"
    )

    assert (
        config.model.target_language
        ==
        "Marathi"
    )


def test_smoke_config_is_qlora():
    """Prevent accidentally switching back to normal LoRA."""

    config = load_config(
        "configs/smoke.yaml"
    )

    assert (
        config.quantization.enabled
        is True
    )

    assert (
        config.quantization.load_in_4bit
        is True
    )

    assert (
        config.quantization.quant_type
        ==
        "nf4"
    )