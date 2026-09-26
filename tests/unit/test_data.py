"""Unit tests for the Part 1 data-preparation pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bodhan_bhili.core.config import DatasetConfig, load_config
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.data import (
    clean_parallel_data,
    devanagari_ratio,
    load_raw_dataset,
    normalize_text,
    prepare_dataset,
    sha256_file,
    split_by_group,
)


def test_normalize_text_is_conservative() -> None:
    """Whitespace and zero-width characters should be normalized."""
    text = "  नमस्कार\u200b   दुनिया  "

    assert normalize_text(text) == "नमस्कार दुनिया"


def test_devanagari_ratio_ignores_digits_and_punctuation() -> None:
    """Numbers and punctuation should not make valid Devanagari fail."""
    assert devanagari_ratio("मारा गाव 2026!") == 1.0

    assert devanagari_ratio("romanized bhili") == 0.0


def test_load_raw_dataset_rejects_schema_mismatch(tmp_path: Path) -> None:
    """The loader must fail clearly rather than guessing column names."""
    bad_path = tmp_path / "bad.tsv"

    pd.DataFrame({"wrong": ["value"]}).to_csv(bad_path, sep="\t", index=False, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required column"):
        load_raw_dataset(bad_path, DatasetConfig())


def test_clean_parallel_data_records_drop_reasons() -> None:
    """Each rejected record should have a deterministic reason."""
    frame = pd.DataFrame(
        [
            {
                "dp_id": "keep-1",
                "id": "g1",
                "marathi": "माझ्या गावात आज पाणी येते.",
                "Dehwali_Bhili": "मारा गावमा आज पाणी आवे हाय.",
            },
            {
                "dp_id": "duplicate-1",
                "id": "g1",
                "marathi": "माझ्या गावात आज पाणी येते.",
                "Dehwali_Bhili": "मारा गावमा आज पाणी आवे हाय.",
            },
            {
                "dp_id": "short-1",
                "id": "g2",
                "marathi": "हा एक पुरेसा मोठा मराठी मजकूर आहे.",
                "Dehwali_Bhili": ".",
            },
            {
                "dp_id": "romanized-1",
                "id": "g3",
                "marathi": "हा आणखी एक मराठी मजकूर आहे.",
                "Dehwali_Bhili": "this is romanized bhili text",
            },
            {
                "dp_id": "conflict-1",
                "id": "g4",
                "marathi": "रस्ता आज बंद आहे.",
                "Dehwali_Bhili": "आ गावनो रस्तो आज बंद हाय.",
            },
            {
                "dp_id": "conflict-2",
                "id": "g5",
                "marathi": "बाजार आज बंद आहे.",
                "Dehwali_Bhili": "आ गावनो रस्तो आज बंद हाय.",
            },
        ]
    )

    (clean, audit) = clean_parallel_data(frame, DatasetConfig())

    assert clean["record_id"].tolist() == ["keep-1"]

    reasons = dict(zip(audit["record_id"], audit["reason"], strict=True))

    assert reasons["duplicate-1"] == "exact_duplicate_pair"

    assert reasons["short-1"] == "source_too_short"

    assert reasons["romanized-1"] == "low_source_devanagari_ratio"

    assert reasons["conflict-1"] == "conflicting_source_translation"

    assert reasons["conflict-2"] == "conflicting_source_translation"


def test_split_by_group_has_no_leakage_and_is_deterministic() -> None:
    """Translation variants from the same group must stay together."""
    rows = []

    for group_index in range(30):
        for variant in range(2):
            rows.append(
                {
                    "record_id": f"r-{group_index}-{variant}",
                    "group_id": f"g-{group_index}",
                    "source": (f"भिली वाक्य {group_index} पर्याय {variant}"),
                    "target": f"मराठी वाक्य {group_index}",
                    "source_language": "dehwali_bhili",
                    "target_language": "marathi",
                    "dataset_source": "project_astitva_aikosh",
                }
            )

    frame = pd.DataFrame(rows)
    first = split_by_group(frame, seed=42, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
    second = split_by_group(frame, seed=42, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)

    for split_name in ("train", "dev", "test"):
        assert first[split_name]["record_id"].tolist() == second[split_name]["record_id"].tolist()

    group_sets = {name: set(split["group_id"]) for (name, split) in first.items()}

    assert group_sets["train"].isdisjoint(group_sets["dev"])

    assert group_sets["train"].isdisjoint(group_sets["test"])

    assert group_sets["dev"].isdisjoint(group_sets["test"])


def test_prepare_dataset_writes_frozen_artifacts(tmp_path: Path) -> None:
    """The full CPU data pipeline should write every required artifact."""
    raw_path = tmp_path / "sample.tsv"
    rows = []

    for index in range(30):
        rows.append(
            {
                "dp_id": f"record-{index}",
                "id": f"group-{index}",
                "marathi": f"हे मराठी वाक्य क्रमांक {index} आहे.",
                "Dehwali_Bhili": f"आ भिली वाक्य क्रमांक {index} हाय.",
            }
        )

    pd.DataFrame(rows).to_csv(raw_path, sep="\t", index=False, encoding="utf-8")

    config = load_config("configs/smoke.yaml")
    config.paths.raw_data_file = raw_path
    config.paths.artifact_root = tmp_path / "artifacts"
    config.run.run_id = "data_test"
    paths = ArtifactPaths.from_config(config)
    report = prepare_dataset(config, paths)

    assert report["raw_dataset"]["rows"] == 30

    assert report["cleaning_result"]["rows_kept"] == 30

    assert paths.train_data.exists()

    assert paths.dev_data.exists()

    assert paths.test_data.exists()

    assert paths.data_report.exists()

    assert paths.data_cleaning_report.exists()

    assert paths.split_manifest.exists()

    assert paths.test_set_sha256.exists()

    sidecar = paths.test_set_sha256.read_text(encoding="utf-8")

    assert sha256_file(paths.test_data) in sidecar
