"""Project-wide constants."""

from __future__ import annotations

TRANSLATION_DIRECTION = "bhili_to_marathi"


DEFAULT_MODEL_ID = "bodhan-ai/indic-translate"

MARATHI_PROMPT_NAME = "Marathi"


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
