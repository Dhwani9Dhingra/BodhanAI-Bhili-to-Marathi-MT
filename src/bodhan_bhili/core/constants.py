"""
Project-wide constants.

Only values that genuinely should not change between experiments
belong here.

Hyperparameters such as LoRA rank, learning rate, batch size and
sequence length do NOT belong here. Those belong in YAML configs so
that every experiment remains reproducible.
"""

from __future__ import annotations


# ------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------

PACKAGE_NAME = "bodhan-bhili-mt"

PIPELINE_SCHEMA_VERSION = 1


# ------------------------------------------------------------------
# RESEARCH TASK
# ------------------------------------------------------------------

SOURCE_LANGUAGE_DISPLAY_NAME = "Dehwali Bhili"

TARGET_LANGUAGE_DISPLAY_NAME = "Marathi"

TRANSLATION_DIRECTION = "bhili_to_marathi"


# ------------------------------------------------------------------
# BODHAN
# ------------------------------------------------------------------

DEFAULT_MODEL_ID = "bodhan-ai/indic-translate"

# Bodhan's published target-language prompt name.
MARATHI_PROMPT_NAME = "Marathi"


# ------------------------------------------------------------------
# PIPELINE STAGES
# ------------------------------------------------------------------

STAGE_PREFLIGHT = "preflight"
STAGE_DATA_PREPARATION = "data_preparation"
STAGE_BASELINES = "baselines"
STAGE_TRAINING = "training"
STAGE_ADAPTER_VERIFICATION = "adapter_verification"
STAGE_FINAL_EVALUATION = "final_evaluation"
STAGE_ANALYSIS = "analysis"
STAGE_PACKAGING = "packaging"

ALL_PIPELINE_STAGES = (
    STAGE_PREFLIGHT,
    STAGE_DATA_PREPARATION,
    STAGE_BASELINES,
    STAGE_TRAINING,
    STAGE_ADAPTER_VERIFICATION,
    STAGE_FINAL_EVALUATION,
    STAGE_ANALYSIS,
    STAGE_PACKAGING,
)