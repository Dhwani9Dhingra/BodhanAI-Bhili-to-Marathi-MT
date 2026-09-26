"""Validated experiment configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bodhan_bhili.core.constants import DEFAULT_MODEL_ID, MARATHI_PROMPT_NAME, TRANSLATION_DIRECTION

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class StrictModel(BaseModel):
    """Base class for all configuration sections."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(StrictModel):
    """Research-task metadata."""

    name: str = "bodhan-bhili-mt"
    task: Literal["machine_translation"] = "machine_translation"
    direction: Literal["bhili_to_marathi"] = TRANSLATION_DIRECTION
    seed: int = Field(default=42, ge=0)


class RunConfig(StrictModel):
    """Identity of one reproducible experiment."""

    run_id: str

    description: str = ""

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """Keep run IDs filesystem-safe."""
        value = value.strip()

        if not value:
            raise ValueError("run_id cannot be empty.")

        if not _SAFE_RUN_ID.fullmatch(value):
            raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'.")

        return value


class PathsConfig(StrictModel):
    """Filesystem locations."""

    artifact_root: Path = Path("/content/drive/MyDrive/BodhanAI/artifacts")
    hf_cache_dir: Path = Path(".cache/huggingface")
    raw_data_file: Path | None = None


class DatasetConfig(StrictModel):
    """Configuration for the Project Astitva Dehwali Bhili ↔ Marathi TSV."""

    delimiter: str = "\t"
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8"
    source_column: str = "Dehwali_Bhili"
    target_column: str = "marathi"
    group_column: str = "id"
    record_column: str = "dp_id"
    train_ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    dev_ratio: float = Field(default=0.10, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.10, gt=0.0, lt=1.0)
    min_text_chars: int = Field(default=2, ge=1)
    min_source_devanagari_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    min_length_ratio: float = Field(default=0.25, gt=0.0)
    max_length_ratio: float = Field(default=2.00, gt=0.0)
    remove_exact_duplicates: bool = True

    @model_validator(mode="after")
    def validate_dataset_settings(self):
        """Validate split proportions and cleaning thresholds."""
        split_total = self.train_ratio + self.dev_ratio + self.test_ratio

        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                f"train_ratio + dev_ratio + test_ratio must equal 1.0, received {split_total:.6f}."
            )

        if self.min_length_ratio >= self.max_length_ratio:
            raise ValueError("min_length_ratio must be smaller than max_length_ratio.")

        return self


class ModelConfig(StrictModel):
    """Bodhan model settings shared by training and inference."""

    repo_id: str = DEFAULT_MODEL_ID
    target_language: str = MARATHI_PROMPT_NAME
    max_train_tokens: int = Field(default=256, ge=32, le=4096)
    max_new_tokens: int = Field(default=512, ge=16, le=4096)
    attention_implementation: Literal["sdpa", "eager"] = "sdpa"


class QuantizationConfig(StrictModel):
    """4-bit QLoRA quantization settings."""

    enabled: bool = True
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["auto", "float16", "bfloat16"] = "auto"

    @model_validator(mode="after")
    def validate_qlora_quantization(self):
        """Our main experiment is explicitly QLoRA."""
        if self.enabled and not self.load_in_4bit:
            raise ValueError("QLoRA is enabled, therefore load_in_4bit must also be true.")

        return self


class LoraConfig(StrictModel):
    """Adapter hyperparameters."""

    rank: int = Field(default=8, ge=1, le=256)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: Literal["none"] = "none"
    target_strategy: Literal["all_linear_filtered"] = "all_linear_filtered"
    exclude_name_fragments: list[str] = Field(
        default_factory=lambda: ["vision", "visual", "image", "audio", "lm_head", "embed_tokens"]
    )


class TrainingConfig(StrictModel):
    """Optimizer, checkpoint, and training budget settings."""

    max_steps: int = Field(default=10, ge=1)
    train_batch_size: int = Field(default=1, ge=1)
    eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_checkpointing: bool = True
    optimizer: Literal["paged_adamw_8bit"] = "paged_adamw_8bit"
    logging_steps: int = Field(default=1, ge=1)
    evaluation_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=5, ge=1)
    save_total_limit: int = Field(default=2, ge=1)


class EvaluationConfig(StrictModel):
    """Metrics and generation sample limits."""

    primary_metric: Literal["chrfpp"] = "chrfpp"
    compute_bleu: bool = True
    compute_copy_metrics: bool = True
    bootstrap_samples: int = Field(default=200, ge=0)
    dev_generation_samples: int = Field(default=20, ge=1)
    test_generation_samples: int = Field(default=20, ge=1)


class PreflightConfig(StrictModel):
    """Minimum environment requirements checked before expensive work."""

    minimum_python: str = "3.10"
    minimum_torch: str = "2.6"
    minimum_transformers: str = "5.12"
    minimum_vram_gib: float = Field(default=14.0, gt=0)
    minimum_disk_free_gib: float = Field(default=25.0, gt=0)
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


def _expand_environment_variables(value):
    """Recursively expand `$NAME` and `${NAME}` in YAML values."""
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]

    if isinstance(value, str):
        return os.path.expandvars(value)

    return value


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load YAML, apply environment overrides and validate it."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open(mode="r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a dictionary.")

    raw = _expand_environment_variables(raw)
    artifact_override = os.getenv("BODHAN_ARTIFACT_ROOT")

    if artifact_override:
        raw.setdefault("paths", {})["artifact_root"] = artifact_override

    data_override = os.getenv("BODHAN_DATA_FILE")

    if data_override:
        raw.setdefault("paths", {})["raw_data_file"] = data_override

    cache_override = os.getenv("BODHAN_HF_CACHE")

    if cache_override:
        raw.setdefault("paths", {})["hf_cache_dir"] = cache_override

    run_override = os.getenv("BODHAN_RUN_ID")

    if run_override:
        raw.setdefault("run", {})["run_id"] = run_override

    return ExperimentConfig.model_validate(raw)
