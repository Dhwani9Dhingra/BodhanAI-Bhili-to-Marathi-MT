"""Data preparation for the Project Astitva Dehwali Bhili -> Marathi corpus."""

from __future__ import annotations

import hashlib
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from bodhan_bhili.core.config import DatasetConfig, ExperimentConfig
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.serialization import atomic_write_json

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

_WHITESPACE_RE = re.compile(r"\s+")


_CANONICAL_COLUMNS = [
    "record_id",
    "group_id",
    "source",
    "target",
    "source_language",
    "target_language",
    "dataset_source",
]


def normalize_text(value: Any) -> str:
    """Apply NFC, remove zero-width characters, and collapse whitespace."""
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)

    return text.strip()


def devanagari_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters that are Devanagari."""
    letters = [character for character in text if character.isalpha()]

    if not letters:
        return 0.0

    def is_devanagari(character: str) -> bool:
        """Check the main and extended Devanagari Unicode blocks."""
        codepoint = ord(character)

        return (
            0x0900 <= codepoint <= 0x097F
            or 0xA8E0 <= codepoint <= 0xA8FF
            or 0x11B00 <= codepoint <= 0x11B5F
        )

    devanagari_letters = sum(is_devanagari(character) for character in letters)

    return devanagari_letters / len(letters)


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 without loading the entire file into memory."""
    digest = hashlib.sha256()

    with Path(path).open(mode="rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_raw_dataset(path: str | Path, dataset_config: DatasetConfig) -> pd.DataFrame:
    """Load the real Project Astitva TSV and validate its schema."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Raw dataset path is not a file: {path}")

    frame = pd.read_csv(
        path,
        sep=dataset_config.delimiter,
        encoding=dataset_config.encoding,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        dataset_config.record_column,
        dataset_config.group_column,
        dataset_config.source_column,
        dataset_config.target_column,
    }

    missing_columns = sorted(required_columns - set(frame.columns))

    if missing_columns:
        raise ValueError(
            "Dataset schema mismatch. "
            "Missing required column(s): "
            f"{missing_columns}. "
            f"Available columns: {list(frame.columns)}"
        )

    if frame.empty:
        raise ValueError("The raw dataset contains zero rows.")

    for column_name, label in (
        (dataset_config.record_column, "record"),
        (dataset_config.group_column, "group"),
    ):
        blank_mask = frame[column_name].astype(str).str.strip().eq("")

        if blank_mask.any():
            raise ValueError(
                f"Dataset contains {int(blank_mask.sum())} "
                f"blank {label} ID(s) in "
                f"column {column_name!r}."
            )

    duplicated_record_ids = frame[dataset_config.record_column].duplicated(keep=False)

    if duplicated_record_ids.any():
        example_ids = (
            frame.loc[duplicated_record_ids, dataset_config.record_column]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        raise ValueError(
            f"Column {dataset_config.record_column!r} "
            "is expected to uniquely identify rows, but "
            f"{int(duplicated_record_ids.sum())} rows "
            "share duplicate IDs. "
            f"Example duplicate IDs: {example_ids}"
        )

    return frame


def _content_character_count(text: str) -> int:
    """Count meaningful letters/digits."""
    return sum(character.isalnum() for character in text)


def _safe_length_ratio(source: str, target: str) -> float:
    """Calculate source-length / target-length."""
    return len(source) / max(len(target), 1)


def clean_parallel_data(
    raw_frame: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned parallel pairs and an audit of every input row."""
    work = raw_frame.copy().reset_index(drop=True)
    record_column = dataset_config.record_column
    group_column = dataset_config.group_column
    source_column = dataset_config.source_column
    target_column = dataset_config.target_column

    work["_raw_row_number"] = work.index + 2

    work["_source"] = work[source_column].map(normalize_text)

    work["_target"] = work[target_column].map(normalize_text)

    work["_source_char_count"] = work["_source"].str.len()

    work["_target_char_count"] = work["_target"].str.len()

    work["_source_content_chars"] = work["_source"].map(_content_character_count)

    work["_target_content_chars"] = work["_target"].map(_content_character_count)

    work["_source_devanagari_ratio"] = work["_source"].map(devanagari_ratio)

    work["_length_ratio"] = [
        _safe_length_ratio(source, target)
        for source, target in zip(work["_source"], work["_target"], strict=True)
    ]

    work["_drop_reason"] = ""

    def apply_reason(mask: pd.Series, reason: str) -> None:
        """Apply only the FIRST matching failure reason."""
        eligible = work["_drop_reason"].eq("") & mask

        work.loc[eligible, "_drop_reason"] = reason

    apply_reason(work["_source"].eq(""), "empty_source")

    apply_reason(work["_target"].eq(""), "empty_target")

    apply_reason(
        (work["_source_content_chars"] < dataset_config.min_text_chars), "source_too_short"
    )

    apply_reason(
        (work["_target_content_chars"] < dataset_config.min_text_chars), "target_too_short"
    )

    apply_reason(
        (work["_source_devanagari_ratio"] < dataset_config.min_source_devanagari_ratio),
        "low_source_devanagari_ratio",
    )

    apply_reason(
        (work["_length_ratio"] < dataset_config.min_length_ratio)
        | (work["_length_ratio"] > dataset_config.max_length_ratio),
        "extreme_length_ratio",
    )

    currently_valid = work["_drop_reason"].eq("")
    valid_pairs = work.loc[currently_valid, ["_source", "_target"]]
    target_counts_per_source = valid_pairs.groupby("_source")["_target"].nunique()
    conflicting_sources = set(target_counts_per_source[target_counts_per_source > 1].index)

    apply_reason(work["_source"].isin(conflicting_sources), "conflicting_source_translation")

    if dataset_config.remove_exact_duplicates:
        currently_valid = work["_drop_reason"].eq("")
        duplicate_flags = work.loc[currently_valid].duplicated(
            subset=["_source", "_target"], keep="first"
        )

        duplicate_row_indices = duplicate_flags[duplicate_flags].index

        work.loc[duplicate_row_indices, "_drop_reason"] = "exact_duplicate_pair"

    kept_mask = work["_drop_reason"].eq("")
    kept = work.loc[kept_mask].copy()
    clean_frame = pd.DataFrame(
        {
            "record_id": kept[record_column].astype(str),
            "group_id": kept[group_column].astype(str),
            "source": kept["_source"],
            "target": kept["_target"],
            "source_language": "dehwali_bhili",
            "target_language": "marathi",
            "dataset_source": "project_astitva_aikosh",
        }
    ).reset_index(drop=True)

    audit_frame = pd.DataFrame(
        {
            "raw_row_number": work["_raw_row_number"].astype(int),
            "record_id": work[record_column].astype(str),
            "group_id": work[group_column].astype(str),
            "status": (work["_drop_reason"].eq("").map({True: "kept", False: "dropped"})),
            "reason": work["_drop_reason"],
            "source_chars": work["_source_char_count"].astype(int),
            "target_chars": work["_target_char_count"].astype(int),
            "length_ratio": work["_length_ratio"].round(6),
            "source_devanagari_ratio": work["_source_devanagari_ratio"].round(6),
        }
    )

    return (clean_frame, audit_frame)


def split_by_group(
    clean_frame: pd.DataFrame, *, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float
) -> dict[str, pd.DataFrame]:
    """Create deterministic and leakage-safe train/dev/test splits."""
    missing_columns = set(_CANONICAL_COLUMNS) - set(clean_frame.columns)

    if missing_columns:
        raise ValueError(f"Canonical dataset is missing columns: {sorted(missing_columns)}")

    group_sizes = clean_frame.groupby("group_id", sort=False).size().to_dict()
    groups = sorted(str(group_id) for group_id in group_sizes)

    if len(groups) < 3:
        raise ValueError(
            "At least 3 unique groups are required to create non-overlapping train/dev/test splits."
        )

    random.Random(seed).shuffle(groups)

    ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    target_rows = {split_name: ratio * len(clean_frame) for split_name, ratio in ratios.items()}
    assigned_rows = {split_name: 0 for split_name in ratios}
    split_groups: dict[str, set[str]] = {split_name: set() for split_name in ratios}
    tie_priority = {"train": 2, "dev": 1, "test": 0}

    for group_id in groups:
        group_size = int(group_sizes[group_id])

        def normalized_deficit(split_name: str) -> tuple[float, int]:
            """Higher score means this split is further below its target size."""
            target = target_rows[split_name]
            deficit = (target - assigned_rows[split_name]) / target

            return (deficit, tie_priority[split_name])

        destination = max(ratios, key=normalized_deficit)

        split_groups[destination].add(group_id)

        assigned_rows[destination] += group_size

    splits: dict[str, pd.DataFrame] = {}

    for split_name, group_ids in split_groups.items():
        split_frame = (
            clean_frame.loc[clean_frame["group_id"].isin(group_ids)].copy().reset_index(drop=True)
        )

        splits[split_name] = split_frame

    assert_no_split_leakage(splits)

    return splits


def assert_no_split_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Verify that no exact records/text/groups cross split boundaries."""
    expected_names = {"train", "dev", "test"}

    if set(splits) != expected_names:
        raise ValueError(f"Expected split keys {expected_names}, received {set(splits)}")

    for split_name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Split {split_name!r} is empty.")

    for column in ("group_id", "source", "target", "record_id"):
        values = {name: set(frame[column].astype(str)) for name, frame in splits.items()}
        split_pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))

        for left, right in split_pairs:
            overlap = values[left] & values[right]

            if overlap:
                preview = sorted(overlap)[:5]

                raise ValueError(
                    "Leakage detected for "
                    f"column {column!r} "
                    f"between {left} and {right}. "
                    f"Example overlap: {preview}"
                )


def _atomic_write_dataframe(
    frame: pd.DataFrame, destination: str | Path, *, separator: str
) -> None:
    """Write a dataframe through a temporary sibling file."""
    destination = Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with temporary.open(mode="w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep=separator, index=False, lineterminator="\n")

        handle.flush()

        os.fsync(handle.fileno())

    os.replace(temporary, destination)


def _write_sha256_sidecar(digest: str, hashed_file: Path, destination: Path) -> None:
    """Write a conventional `HASH  filename` SHA-256 sidecar."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with temporary.open(mode="w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {hashed_file.name}\n")

        handle.flush()

        os.fsync(handle.fileno())

    os.replace(temporary, destination)


def _numeric_summary(series: pd.Series) -> dict[str, float | int]:
    """Create compact JSON-safe descriptive statistics."""
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)

    if numeric.empty:
        return {"count": 0}

    return {
        "count": int(numeric.size),
        "min": round(float(numeric.min()), 4),
        "p25": round(float(numeric.quantile(0.25)), 4),
        "median": round(float(numeric.median()), 4),
        "mean": round(float(numeric.mean()), 4),
        "p75": round(float(numeric.quantile(0.75)), 4),
        "p95": round(float(numeric.quantile(0.95)), 4),
        "max": round(float(numeric.max()), 4),
    }


def prepare_dataset(config: ExperimentConfig, paths: ArtifactPaths) -> dict[str, Any]:
    """Run the complete Part 1 data-preparation pipeline."""
    raw_path = config.paths.raw_data_file

    if raw_path is None:
        raise ValueError(
            "No raw dataset path is configured. "
            "Set paths.raw_data_file in YAML, "
            "set BODHAN_DATA_FILE, or pass "
            "--data to scripts.prepare_data."
        )

    raw_path = Path(raw_path)

    paths.create_directories()

    raw_sha256 = sha256_file(raw_path)
    raw_frame = load_raw_dataset(raw_path, config.dataset)

    (clean_frame, audit) = clean_parallel_data(raw_frame, config.dataset)

    if clean_frame.empty:
        raise ValueError("Cleaning removed every row; refusing to continue.")

    if clean_frame["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values remain after cleaning.")

    if clean_frame.duplicated(["source", "target"]).any():
        raise ValueError("Exact source-target duplicate pairs remain after cleaning.")

    conflicting_sources = clean_frame.groupby("source")["target"].nunique()

    if (conflicting_sources > 1).any():
        raise ValueError("Conflicting source translations remain after cleaning.")

    splits = split_by_group(
        clean_frame,
        seed=(config.project.seed),
        train_ratio=(config.dataset.train_ratio),
        dev_ratio=(config.dataset.dev_ratio),
        test_ratio=(config.dataset.test_ratio),
    )

    split_paths = {"train": paths.train_data, "dev": paths.dev_data, "test": paths.test_data}

    for split_name, split_frame in splits.items():
        _atomic_write_dataframe(
            split_frame[_CANONICAL_COLUMNS], split_paths[split_name], separator="\t"
        )

    record_to_split: dict[str, str] = {}

    for split_name, split_frame in splits.items():
        record_to_split.update(
            {str(record_id): split_name for record_id in split_frame["record_id"]}
        )

    audit["split"] = audit["record_id"].map(record_to_split).fillna("")

    _atomic_write_dataframe(audit, paths.data_cleaning_report, separator=",")

    split_hashes = {
        split_name: sha256_file(split_path) for (split_name, split_path) in split_paths.items()
    }

    _write_sha256_sidecar(split_hashes["test"], paths.test_data, paths.test_set_sha256)

    drop_counts = (
        audit.loc[audit["status"].eq("dropped"), "reason"].value_counts().sort_index().to_dict()
    )

    drop_counts = {str(reason): int(count) for (reason, count) in drop_counts.items()}
    split_summary: dict[str, Any] = {}

    for split_name, split_frame in splits.items():
        split_summary[split_name] = {
            "rows": int(len(split_frame)),
            "groups": int(split_frame["group_id"].nunique()),
            "row_fraction_of_clean_data": round(len(split_frame) / len(clean_frame), 6),
            "sha256": split_hashes[split_name],
            "file": str(split_paths[split_name]),
        }

    kept_audit = audit.loc[audit["status"].eq("kept")].copy()
    data_report: dict[str, Any] = {
        "schema_version": 1,
        "task_direction": "dehwali_bhili_to_marathi",
        "raw_dataset": {
            "file": str(raw_path),
            "sha256": raw_sha256,
            "rows": int(len(raw_frame)),
            "columns": [str(column) for column in raw_frame.columns],
            "unique_record_ids": int(raw_frame[config.dataset.record_column].nunique()),
            "unique_group_ids": int(raw_frame[config.dataset.group_column].nunique()),
        },
        "column_mapping": {
            "record_id": config.dataset.record_column,
            "group_id": config.dataset.group_column,
            "source": config.dataset.source_column,
            "target": config.dataset.target_column,
        },
        "cleaning_configuration": {
            "unicode_normalization": "NFC",
            "zero_width_characters_removed": True,
            "whitespace_collapsed": True,
            "min_text_chars": config.dataset.min_text_chars,
            "min_source_devanagari_ratio": config.dataset.min_source_devanagari_ratio,
            "min_length_ratio": config.dataset.min_length_ratio,
            "max_length_ratio": config.dataset.max_length_ratio,
            "remove_exact_duplicates": config.dataset.remove_exact_duplicates,
            "drop_conflicting_identical_sources": True,
        },
        "cleaning_result": {
            "rows_kept": int(len(clean_frame)),
            "rows_dropped": int(len(raw_frame) - len(clean_frame)),
            "retention_fraction": round(len(clean_frame) / len(raw_frame), 6),
            "unique_groups_kept": int(clean_frame["group_id"].nunique()),
            "drop_counts": drop_counts,
            "identical_source_target_rows_kept": int(
                clean_frame["source"].eq(clean_frame["target"]).sum()
            ),
        },
        "kept_data_statistics": {
            "source_chars": _numeric_summary(kept_audit["source_chars"]),
            "target_chars": _numeric_summary(kept_audit["target_chars"]),
            "length_ratio": _numeric_summary(kept_audit["length_ratio"]),
            "source_devanagari_ratio": _numeric_summary(kept_audit["source_devanagari_ratio"]),
        },
        "splits": split_summary,
        "leakage_checks": {
            "group_overlap": False,
            "exact_source_overlap": False,
            "exact_target_overlap": False,
            "record_id_overlap": False,
        },
    }

    split_manifest = {
        "schema_version": 1,
        "seed": int(config.project.seed),
        "split_method": "deterministic_group_row_balancing",
        "group_key": "group_id",
        "requested_ratios": {
            "train": config.dataset.train_ratio,
            "dev": config.dataset.dev_ratio,
            "test": config.dataset.test_ratio,
        },
        "clean_rows": int(len(clean_frame)),
        "clean_groups": int(clean_frame["group_id"].nunique()),
        "splits": split_summary,
        "test_set_sha256": split_hashes["test"],
    }

    atomic_write_json(data_report, paths.data_report)

    atomic_write_json(split_manifest, paths.split_manifest)

    return data_report
