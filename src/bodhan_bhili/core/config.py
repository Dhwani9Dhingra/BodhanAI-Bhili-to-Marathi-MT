"""
Validated experiment configuration.

Why use Pydantic?
-----------------
Training an ~8B model is expensive even when the GPU itself is free.
A misspelled YAML key should fail immediately rather than after the
model has downloaded.

Every configuration section therefore has an explicit schema.
Unknown keys are rejected.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from bodhan_bhili.core.constants import (
    DEFAULT_MODEL_ID,
    MARATHI_PROMPT_NAME,
    TRANSLATION_DIRECTION,
)


# Allow letters, numbers, underscores, dots and dashes in run IDs.
_SAFE_RUN_ID = re.compile(
    r"^[A-Za-z0-9._-]+$"
)


class StrictModel(BaseModel):
    """
    Base class for all configuration sections.

    `extra="forbid"` is deliberate:
    a typo such as `learning_raet` must raise an error.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class ProjectConfig(StrictModel):
    """Research-task metadata."""

    name: str = "bodhan-bhili-mt"

    task: Literal["machine_translation"] = "machine_translation"

    direction: Literal["bhili_to_marathi"] = TRANSLATION_DIRECTION

    seed: int = Field(
        default=42,
        ge=0,
    )


class RunConfig(StrictModel):
    """Identity of one reproducible experiment."""

    run_id: str

    description: str = ""

    @field_validator("run_id")
    @classmethod
    def validate_run_id(
        cls,
        value: str,
    ) -> str:
        """
        Keep run IDs filesystem-safe.

        Example valid IDs:
            smoke_001
            bhili-marathi-v1
            2026-09-26_t4
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "run_id cannot be empty."
            )

        if not _SAFE_RUN_ID.fullmatch(
            value
        ):
            raise ValueError(
                "run_id may contain only letters, "
                "numbers, '.', '_' and '-'."
            )

        return value


class PathsConfig(StrictModel):
    """Filesystem locations."""

    # Small output files/checkpoints eventually go under this root.
    artifact_root: Path = Path(
        "artifacts"
    )

    # Large HF model files can stay on Colab's temporary disk.
    hf_cache_dir: Path = Path(
        ".cache/huggingface"
    )

    # Unknown until the user downloads AIKosh data.
    raw_data_file: Path | None = None


class DatasetConfig(StrictModel):
    """
    Configuration for the Project Astitva Dehwali Bhili ↔ Marathi TSV.

    These column names come from the real AIKosh dataset that we inspected,
    rather than from assumptions about its schema.
    """

    # ------------------------------------------------------------------
    # RAW TSV FORMAT
    # ------------------------------------------------------------------

    delimiter: str = "\t"

    encoding: Literal[
        "utf-8",
        "utf-8-sig",
    ] = "utf-8"

    # ------------------------------------------------------------------
    # ACTUAL DATASET COLUMNS
    # ------------------------------------------------------------------

    # Source for our primary MT task: Bhili -> Marathi.
    source_column: str = "Dehwali_Bhili"

    # Marathi is the target language.
    target_column: str = "marathi"

    # Rows sharing this ID can represent alternate translations of the
    # same underlying Marathi sentence. They must never be split across
    # train/dev/test.
    group_column: str = "id"

    # Unique row identifier supplied by the dataset.
    record_column: str = "dp_id"

    # ------------------------------------------------------------------
    # DATA SPLIT
    # ------------------------------------------------------------------

    train_ratio: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
    )

    dev_ratio: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
    )

    test_ratio: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
    )

    # ------------------------------------------------------------------
    # CONSERVATIVE CLEANING RULES
    # ------------------------------------------------------------------

    # Reject things such as "." or "0" as translations.
    min_text_chars: int = Field(
        default=2,
        ge=1,
    )

    # The project scope is native-Devanagari Dehwali Bhili.
    #
    # We deliberately use 0.50 rather than an aggressive 0.90+ threshold
    # because otherwise valid sentences containing names, acronyms or
    # occasional Latin text could be removed.
    min_source_devanagari_ratio: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    # Based on the actual 14,737-row dataset distribution.
    #
    # The great majority of valid pairs are close to a 1:1 character-length
    # relationship. Ratios below 0.25 mostly contain incomplete answers such
    # as ".", "0", "होय", "na mudu", etc.
    min_length_ratio: float = Field(
        default=0.25,
        gt=0.0,
    )

    # Values above 2.0 are rare and often contain duplicated/expanded text.
    max_length_ratio: float = Field(
        default=2.00,
        gt=0.0,
    )

    remove_exact_duplicates: bool = True

    @model_validator(mode="after")
    def validate_dataset_settings(
        self,
    ):
        """Validate split proportions and cleaning thresholds."""

        split_total = (
            self.train_ratio
            + self.dev_ratio
            + self.test_ratio
        )

        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio + dev_ratio + test_ratio "
                f"must equal 1.0, received {split_total:.6f}."
            )

        if self.min_length_ratio >= self.max_length_ratio:
            raise ValueError(
                "min_length_ratio must be smaller than "
                "max_length_ratio."
            )

        return self

    
class ModelConfig(StrictModel):
    """Bodhan model settings shared by training and inference."""

    repo_id: str = DEFAULT_MODEL_ID

    # The source language must NOT be put into Bodhan's prompt.
    target_language: str = MARATHI_PROMPT_NAME

    # Conservative sentence-level budget for free Colab.
    max_train_tokens: int = Field(
        default=256,
        ge=32,
        le=4096,
    )

    # Bodhan's own sentence inference examples allow substantial
    # generation headroom.
    max_new_tokens: int = Field(
        default=512,
        ge=16,
        le=4096,
    )

    attention_implementation: Literal[
        "sdpa",
        "eager",
    ] = "sdpa"


class QuantizationConfig(StrictModel):
    """4-bit QLoRA quantization settings."""

    enabled: bool = True

    load_in_4bit: bool = True

    quant_type: Literal[
        "nf4",
        "fp4",
    ] = "nf4"

    double_quant: bool = True

    # `auto` will choose BF16 only when the GPU really supports it.
    compute_dtype: Literal[
        "auto",
        "float16",
        "bfloat16",
    ] = "auto"

    @model_validator(mode="after")
    def validate_qlora_quantization(
        self,
    ):
        """
        Our main experiment is explicitly QLoRA.

        Accidentally setting enabled=true but load_in_4bit=false
        would turn the experiment into ordinary LoRA.
        """

        if (
            self.enabled
            and not self.load_in_4bit
        ):
            raise ValueError(
                "QLoRA is enabled, therefore load_in_4bit "
                "must also be true."
            )

        return self


class LoraConfig(StrictModel):
    """
    Adapter hyperparameters.

    The exact target-module discovery is implemented in Part 3.
    """

    rank: int = Field(
        default=8,
        ge=1,
        le=256,
    )

    alpha: int = Field(
        default=16,
        ge=1,
    )

    dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
    )

    bias: Literal[
        "none"
    ] = "none"

    # We want QLoRA-style broad text-model coverage,
    # but must remove Bodhan's multimodal components.
    target_strategy: Literal[
        "all_linear_filtered"
    ] = "all_linear_filtered"

    exclude_name_fragments: list[str] = Field(
        default_factory=lambda: [
            "vision",
            "visual",
            "image",
            "audio",
            "lm_head",
            "embed_tokens",
        ]
    )


class TrainingConfig(StrictModel):
    """Training settings for later Part 4."""

    max_steps: int = Field(
        default=10,
        ge=1,
    )

    train_batch_size: int = Field(
        default=1,
        ge=1,
    )

    eval_batch_size: int = Field(
        default=1,
        ge=1,
    )

    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
    )

    learning_rate: float = Field(
        default=2e-4,
        gt=0,
    )

    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        lt=1.0,
    )

    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
    )

    gradient_checkpointing: bool = True

    optimizer: Literal[
        "paged_adamw_8bit"
    ] = "paged_adamw_8bit"

    logging_steps: int = Field(
        default=1,
        ge=1,
    )

    evaluation_steps: int = Field(
        default=5,
        ge=1,
    )

    save_steps: int = Field(
        default=5,
        ge=1,
    )

    save_total_limit: int = Field(
        default=2,
        ge=1,
    )


class EvaluationConfig(StrictModel):
    """Evaluation settings used in Part 5."""

    primary_metric: Literal[
        "chrfpp"
    ] = "chrfpp"

    compute_bleu: bool = True

    compute_copy_metrics: bool = True

    bootstrap_samples: int = Field(
        default=200,
        ge=0,
    )

    dev_generation_samples: int = Field(
        default=20,
        ge=1,
    )

    test_generation_samples: int = Field(
        default=20,
        ge=1,
    )


class PreflightConfig(StrictModel):
    """Minimum environment requirements checked before expensive work."""

    minimum_python: str = "3.10"

    minimum_torch: str = "2.6"

    minimum_transformers: str = "5.12"

    # A T4 usually exposes approximately 15 GiB.
    # This threshold is intentionally lower than 16.
    minimum_vram_gib: float = Field(
        default=14.0,
        gt=0,
    )

    # Model downloads plus caches require substantial local storage.
    minimum_disk_free_gib: float = Field(
        default=25.0,
        gt=0,
    )

    require_cuda: bool = True

    check_bodhan_access: bool = True

    check_bitsandbytes_nf4: bool = True


class ExperimentConfig(StrictModel):
    """Complete validated experiment specification."""

    project: ProjectConfig

    run: RunConfig

    paths: PathsConfig

    dataset: DatasetConfig

    model: ModelConfig

    quantization: QuantizationConfig

    lora: LoraConfig

    training: TrainingConfig

    evaluation: EvaluationConfig

    preflight: PreflightConfig


def _expand_environment_variables(
    value,
):
    """Recursively expand `$NAME` and `${NAME}` in YAML values."""

    if isinstance(
        value,
        dict,
    ):
        return {
            key: _expand_environment_variables(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _expand_environment_variables(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        str,
    ):
        return os.path.expandvars(
            value
        )

    return value


def load_config(
    config_path: str | Path,
) -> ExperimentConfig:
    """
    Load YAML, apply environment overrides and validate it.

    Environment overrides are useful in Colab because:
        - code lives in /content,
        - artifacts should live on Google Drive,
        - raw data may also live on Google Drive.
    """

    config_path = Path(
        config_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as handle:

        raw = yaml.safe_load(
            handle
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "The YAML root must be a dictionary."
        )

    raw = _expand_environment_variables(
        raw
    )

    # --------------------------------------------------------
    # ENVIRONMENT OVERRIDES
    # --------------------------------------------------------
    #
    # These let the SAME YAML file run locally and in Colab.

    artifact_override = os.getenv(
        "BODHAN_ARTIFACT_ROOT"
    )

    if artifact_override:
        raw.setdefault(
            "paths",
            {}
        )["artifact_root"] = artifact_override

    data_override = os.getenv(
        "BODHAN_DATA_FILE"
    )

    if data_override:
        raw.setdefault(
            "paths",
            {}
        )["raw_data_file"] = data_override

    cache_override = os.getenv(
        "BODHAN_HF_CACHE"
    )

    if cache_override:
        raw.setdefault(
            "paths",
            {}
        )["hf_cache_dir"] = cache_override

    run_override = os.getenv(
        "BODHAN_RUN_ID"
    )

    if run_override:
        raw.setdefault(
            "run",
            {}
        )["run_id"] = run_override

    # This is where unknown keys, invalid ranges, etc. fail.
    return ExperimentConfig.model_validate(
        raw
    )