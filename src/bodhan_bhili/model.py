"""
Bodhan model utilities for Bhili -> Marathi translation and QLoRA.

Part 2 responsibilities
-----------------------
1. Build Bodhan's exact translation prompt.
2. Load the official AutoProcessor.
3. Load Bodhan Indic-Translate in 4-bit NF4.
4. Generate deterministic Marathi translations.
5. Discover eligible language-model linear layers.
6. Exclude multimodal / output / embedding modules from LoRA.
7. Attach QLoRA safely.
8. Verify which parameters are trainable.
9. Build completion-only training examples.
10. Run a tiny optimizer smoke test.
11. Create a digest proving adapter weights actually changed.

The full Trainer-based experiment belongs to Part 3.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

from bodhan_bhili.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# HARD SAFETY EXCLUSIONS FOR LORA
# ---------------------------------------------------------------------------
#
# Bodhan is based on the multimodal Gemma 4 wrapper.
#
# Our task is TEXT -> TEXT machine translation.
#
# Even if a future config accidentally omits one of these names, we do not
# want the Part 2 smoke test to train vision/audio/projector components.

_MANDATORY_LORA_EXCLUSIONS = (
    "vision",
    "visual",
    "image",
    "audio",
    "multimodal",
    "multi_modal",
    "projector",
    "lm_head",
    "embed_tokens",
)


def build_translation_instruction(
    source_text: str,
    target_language: str = "Marathi",
) -> str:
    """
    Build the exact Bodhan translation instruction.

    Bodhan's model card explicitly says:
        - specify ONLY the target language;
        - do not mention the source language;
        - use one user message.

    Example
    -------
    Translate the following text into Marathi:

    <Bhili source>
    """

    source_text = source_text.strip()

    target_language = (
        target_language
        .strip()
    )

    if not source_text:
        raise ValueError(
            "source_text cannot be empty."
        )

    if not target_language:
        raise ValueError(
            "target_language cannot be empty."
        )

    return (
        "Translate the following text into "
        f"{target_language}:\n\n"
        f"{source_text}"
    )


def build_translation_messages(
    source_text: str,
    target_language: str = "Marathi",
) -> list[dict[str, str]]:
    """
    Create Bodhan's single-user-message conversation.

    There is intentionally NO system message.
    """

    return [
        {
            "role": "user",
            "content": (
                build_translation_instruction(
                    source_text,
                    target_language,
                )
            ),
        }
    ]


def build_completion_labels(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    """
    Build completion-only labels.

    Prompt tokens receive -100 and therefore do not contribute to loss.

    Only the Marathi assistant completion is supervised.

    This function is intentionally pure Python so it can be unit-tested
    without downloading Bodhan or even importing PyTorch.
    """

    prompt_ids = list(
        prompt_ids
    )

    full_ids = list(
        full_ids
    )

    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            "Full training sequence must contain "
            "tokens after the prompt."
        )

    if (
        full_ids[
            : len(prompt_ids)
        ]
        != prompt_ids
    ):
        raise ValueError(
            "The generation prompt is not an exact prefix "
            "of the full training conversation. "
            "Refusing to create incorrect target masking."
        )

    return (
        [-100]
        * len(prompt_ids)
        +
        full_ids[
            len(prompt_ids):
        ]
    )


def _get_hf_token() -> str | None:
    """
    Retrieve the Hugging Face token without logging it.

    First preference:
        HF_TOKEN environment variable.

    Fallback:
        locally cached Hugging Face authentication.
    """

    token = os.getenv(
        "HF_TOKEN"
    )

    if token:
        return token

    try:
        from huggingface_hub import get_token

        return get_token()

    except Exception:
        return None


def resolve_compute_dtype(
    config: ExperimentConfig,
):
    """
    Resolve the actual torch dtype used for 4-bit computation.

    T4:
        float16

    L4 / A100 and newer GPUs:
        usually bfloat16
    """

    import torch

    configured = (
        config
        .quantization
        .compute_dtype
    )

    if configured == "float16":
        return torch.float16

    if configured == "bfloat16":
        return torch.bfloat16

    # "auto"
    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def build_quantization_config(
    config: ExperimentConfig,
):
    """
    Build Hugging Face BitsAndBytesConfig for QLoRA.

    The primary experiment uses:
        - 4 bit
        - NF4
        - double quantization
    """

    from transformers import (
        BitsAndBytesConfig,
    )

    if not (
        config
        .quantization
        .enabled
    ):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=(
            config
            .quantization
            .load_in_4bit
        ),

        bnb_4bit_quant_type=(
            config
            .quantization
            .quant_type
        ),

        bnb_4bit_use_double_quant=(
            config
            .quantization
            .double_quant
        ),

        bnb_4bit_compute_dtype=(
            resolve_compute_dtype(
                config
            )
        ),
    )


def load_processor(
    config: ExperimentConfig,
):
    """
    Load Bodhan's official processor and packaged chat template.
    """

    from transformers import (
        AutoProcessor,
    )

    return AutoProcessor.from_pretrained(
        config.model.repo_id,

        token=(
            _get_hf_token()
        ),

        cache_dir=str(
            config
            .paths
            .hf_cache_dir
        ),
    )


def load_quantized_base_model(
    config: ExperimentConfig,
):
    """
    Load Bodhan Indic-Translate using the current official model class.

    We deliberately use AutoModelForImageTextToText because that is the
    model class used in Bodhan's detailed official inference example.

    A single CUDA device is used for our free-Colab QLoRA workflow.
    """

    import torch

    from transformers import (
        AutoModelForImageTextToText,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Part 2 requires a CUDA GPU. "
            "Use a Google Colab GPU runtime."
        )

    quantization_config = (
        build_quantization_config(
            config
        )
    )

    model = (
        AutoModelForImageTextToText
        .from_pretrained(
            config.model.repo_id,

            token=(
                _get_hf_token()
            ),

            cache_dir=str(
                config
                .paths
                .hf_cache_dir
            ),

            dtype=(
                resolve_compute_dtype(
                    config
                )
            ),

            quantization_config=(
                quantization_config
            ),

            # We want the whole quantized model on the single Colab GPU.
            device_map={
                "": 0
            },

            attn_implementation=(
                config
                .model
                .attention_implementation
            ),

            low_cpu_mem_usage=True,
        )
    )

    return model


def get_model_input_device(
    model,
):
    """
    Determine where token tensors should be placed.

    Using the embedding layer is more robust than assuming `.device`
    behaves identically for every PEFT/multimodal wrapper.
    """

    try:
        embeddings = (
            model
            .get_input_embeddings()
        )

        if (
            embeddings is not None
            and hasattr(
                embeddings,
                "weight",
            )
        ):
            return (
                embeddings
                .weight
                .device
            )

    except Exception:
        pass

    # Safe fallback.
    for parameter in model.parameters():

        if (
            parameter
            .device
            .type
            != "meta"
        ):
            return (
                parameter
                .device
            )

    raise RuntimeError(
        "Could not determine model input device."
    )


def _move_batch_to_device(
    batch: dict[str, Any],
    device,
) -> dict[str, Any]:
    """Move tensor values to the model device."""

    moved = {}

    for key, value in batch.items():

        if hasattr(
            value,
            "to",
        ):
            moved[key] = (
                value.to(
                    device
                )
            )

        else:
            moved[key] = value

    return moved


def translate(
    model,
    processor,
    source_text: str,
    *,
    target_language: str,
    max_new_tokens: int,
) -> str:
    """
    Translate one Bhili string into Marathi.

    Decoding is greedy because Bodhan recommends deterministic greedy
    decoding for translation.
    """

    import torch

    messages = (
        build_translation_messages(
            source_text,
            target_language,
        )
    )

    inputs = (
        processor
        .apply_chat_template(
            messages,

            add_generation_prompt=True,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",
        )
    )

    device = (
        get_model_input_device(
            model
        )
    )

    inputs = (
        _move_batch_to_device(
            dict(inputs),
            device,
        )
    )

    input_length = (
        inputs[
            "input_ids"
        ]
        .shape[-1]
    )

    model.eval()

    with torch.inference_mode():

        generated = (
            model.generate(
                **inputs,

                max_new_tokens=(
                    max_new_tokens
                ),

                do_sample=False,

                use_cache=True,
            )
        )

    completion = (
        generated[
            0,
            input_length:
        ]
    )

    return (
        processor.decode(
            completion,

            skip_special_tokens=True,
        )
        .strip()
    )


def effective_lora_exclusions(
    config: ExperimentConfig,
) -> tuple[str, ...]:
    """
    Combine config exclusions with mandatory multimodal safety exclusions.
    """

    configured = [
        fragment.lower()
        for fragment
        in (
            config
            .lora
            .exclude_name_fragments
        )
    ]

    combined = (
        configured
        +
        list(
            _MANDATORY_LORA_EXCLUSIONS
        )
    )

    # dict preserves order while removing duplicates.
    return tuple(
        dict.fromkeys(
            combined
        )
    )


def is_excluded_module_name(
    module_name: str,
    exclusions: Sequence[str],
) -> bool:
    """Return True when a module path contains an excluded fragment."""

    lower_name = (
        module_name
        .lower()
    )

    return any(
        fragment.lower()
        in lower_name

        for fragment
        in exclusions
    )


def discover_lora_target_modules(
    model,
    exclusions: Sequence[str],
) -> list[str]:
    """
    Discover actual linear modules after 4-bit loading.

    Why runtime discovery?
    ----------------------
    Gemma-family architectures evolve and multimodal wrappers introduce
    additional modules. Hard-coding q_proj/v_proj alone would make our
    implementation architecture-fragile.

    We therefore:
        1. inspect the REAL loaded model;
        2. find linear / 4-bit-linear layers;
        3. explicitly reject multimodal/output/embedding paths.
    """

    import torch

    linear_types: list[type] = [
        torch.nn.Linear,
    ]

    # Quantized linear modules are supplied by bitsandbytes.
    try:
        import bitsandbytes as bnb

        for class_name in (
            "Linear4bit",
            "Linear8bitLt",
        ):

            layer_class = getattr(
                bnb.nn,
                class_name,
                None,
            )

            if (
                layer_class is not None
                and layer_class not in linear_types
            ):
                linear_types.append(
                    layer_class
                )

    except ImportError:
        pass

    target_modules = []

    for (
        module_name,
        module,
    ) in model.named_modules():

        if not module_name:
            continue

        if not isinstance(
            module,
            tuple(
                linear_types
            ),
        ):
            continue

        if is_excluded_module_name(
            module_name,
            exclusions,
        ):
            continue

        target_modules.append(
            module_name
        )

    target_modules = sorted(
        set(
            target_modules
        )
    )

    # An 8B transformer should contain far more than a few linear layers.
    # If this fires, something about the architecture/runtime changed and
    # we should inspect it rather than silently training the wrong modules.
    if len(target_modules) < 8:

        raise RuntimeError(
            "Suspiciously few LoRA target modules were discovered: "
            f"{len(target_modules)}. "
            "Inspect model.named_modules() before training."
        )

    return target_modules


def build_trainable_parameter_report(
    model,
    target_modules: Sequence[str],
    exclusions: Sequence[str],
) -> dict[str, Any]:
    """
    Verify that only LoRA adapter parameters are trainable.
    """

    total_parameters = 0
    trainable_parameters = 0

    trainable_names: list[str] = []
    non_lora_trainable: list[str] = []
    excluded_trainable: list[str] = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        count = (
            parameter.numel()
        )

        total_parameters += count

        if not parameter.requires_grad:
            continue

        trainable_parameters += count

        trainable_names.append(
            name
        )

        lower_name = (
            name.lower()
        )

        if "lora_" not in lower_name:

            non_lora_trainable.append(
                name
            )

        if is_excluded_module_name(
            name,
            exclusions,
        ):

            excluded_trainable.append(
                name
            )

    if trainable_parameters <= 0:

        raise RuntimeError(
            "No trainable LoRA parameters were found."
        )

    if non_lora_trainable:

        raise RuntimeError(
            "Non-LoRA base parameters unexpectedly require gradients. "
            f"Examples: {non_lora_trainable[:10]}"
        )

    if excluded_trainable:

        raise RuntimeError(
            "Excluded multimodal/output parameters unexpectedly "
            "require gradients. "
            f"Examples: {excluded_trainable[:10]}"
        )

    percentage = (
        100.0
        * trainable_parameters
        / total_parameters
    )

    return {
        "total_parameters":
            int(
                total_parameters
            ),

        "trainable_parameters":
            int(
                trainable_parameters
            ),

        "trainable_percentage":
            round(
                percentage,
                6,
            ),

        "target_module_count":
            len(
                target_modules
            ),

        "target_modules":
            list(
                target_modules
            ),

        "effective_exclusions":
            list(
                exclusions
            ),

        "trainable_parameter_count":
            len(
                trainable_names
            ),

        "trainable_parameter_names":
            trainable_names,

        "non_lora_trainable_parameters":
            non_lora_trainable,

        "excluded_trainable_parameters":
            excluded_trainable,
    }


def attach_qlora(
    model,
    config: ExperimentConfig,
):
    """
    Prepare the quantized base model and attach LoRA adapters.
    """

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if not (
        config
        .quantization
        .enabled
        and
        config
        .quantization
        .load_in_4bit
    ):
        raise ValueError(
            "Part 2 smoke testing expects the base model "
            "to be loaded in 4-bit QLoRA mode."
        )

    # Standard PEFT preparation:
    # freezes base weights, prepares low-bit layers and handles
    # gradient-checkpointing requirements.
    model = (
        prepare_model_for_kbit_training(
            model,

            use_gradient_checkpointing=(
                config
                .training
                .gradient_checkpointing
            ),
        )
    )

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = False

    exclusions = (
        effective_lora_exclusions(
            config
        )
    )

    target_modules = (
        discover_lora_target_modules(
            model,
            exclusions,
        )
    )

    lora_config = LoraConfig(
        r=(
            config
            .lora
            .rank
        ),

        lora_alpha=(
            config
            .lora
            .alpha
        ),

        lora_dropout=(
            config
            .lora
            .dropout
        ),

        bias=(
            config
            .lora
            .bias
        ),

        target_modules=(
            target_modules
        ),

        task_type=(
            TaskType.CAUSAL_LM
        ),
    )

    model = (
        get_peft_model(
            model,
            lora_config,
        )
    )

    report = (
        build_trainable_parameter_report(
            model,
            target_modules,
            exclusions,
        )
    )

    return (
        model,
        report,
    )


def trainable_parameter_digest(
    model,
) -> str:
    """
    Hash all trainable adapter values.

    If the digest changes after optimizer steps, the LoRA weights actually
    changed. This is stronger evidence than merely observing a non-zero loss.
    """

    digest = (
        hashlib.sha256()
    )

    trainable_found = False

    for (
        name,
        parameter,
    ) in sorted(
        model.named_parameters(),
        key=lambda item: item[0],
    ):

        if not parameter.requires_grad:
            continue

        trainable_found = True

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        # Convert to float32 before hashing so BF16/FP16 tensors can be
        # represented consistently through NumPy.
        values = (
            parameter
            .detach()
            .float()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

        digest.update(
            values
        )

    if not trainable_found:
        raise RuntimeError(
            "Cannot fingerprint adapters: "
            "no trainable parameters exist."
        )

    return digest.hexdigest()


def encode_training_example(
    processor,
    *,
    source: str,
    target: str,
    target_language: str,
    max_tokens: int,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any],
]:
    """
    Tokenize one Bhili -> Marathi training example.

    The user prompt is masked with -100.
    Only the Marathi assistant completion contributes to loss.

    We do NOT truncate here.

    For the smoke test, examples exceeding max_tokens are simply skipped.
    This avoids silently removing part of the target translation.
    """

    import torch

    source = source.strip()
    target = target.strip()

    prompt_messages = (
        build_translation_messages(
            source,
            target_language,
        )
    )

    # For supervised fine-tuning we append the desired Marathi output as
    # the assistant turn, while still using Bodhan/Gemma's own chat template.
    full_messages = (
        prompt_messages
        +
        [
            {
                "role":
                    "assistant",

                "content":
                    target,
            }
        ]
    )

    prompt_batch = (
        processor
        .apply_chat_template(
            prompt_messages,

            add_generation_prompt=True,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",
        )
    )

    full_batch = (
        processor
        .apply_chat_template(
            full_messages,

            add_generation_prompt=False,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",
        )
    )

    prompt_ids = (
        prompt_batch[
            "input_ids"
        ][0]
        .tolist()
    )

    full_ids = (
        full_batch[
            "input_ids"
        ][0]
        .tolist()
    )

    metadata = {
        "prompt_tokens":
            len(
                prompt_ids
            ),

        "full_tokens":
            len(
                full_ids
            ),

        "max_tokens":
            int(
                max_tokens
            ),
    }

    if len(full_ids) > max_tokens:

        metadata[
            "skip_reason"
        ] = "sequence_too_long"

        return (
            None,
            metadata,
        )

    labels = (
        build_completion_labels(
            prompt_ids,
            full_ids,
        )
    )

    supervised_tokens = sum(
        label != -100
        for label
        in labels
    )

    if supervised_tokens <= 0:

        metadata[
            "skip_reason"
        ] = "no_supervised_tokens"

        return (
            None,
            metadata,
        )

    full_batch[
        "labels"
    ] = torch.tensor(
        [
            labels
        ],
        dtype=torch.long,
    )

    metadata[
        "supervised_tokens"
    ] = supervised_tokens

    metadata[
        "skip_reason"
    ] = None

    return (
        dict(
            full_batch
        ),
        metadata,
    )


def run_smoke_optimizer_steps(
    model,
    encoded_examples: list[
        dict[str, Any]
    ],
    config: ExperimentConfig,
    *,
    optimizer_steps: int,
) -> dict[str, Any]:
    """
    Run a few real optimizer steps.

    This is intentionally a tiny manual loop.

    The production Trainer/checkpoint/evaluation implementation belongs
    to Part 3.
    """

    import math

    import bitsandbytes as bnb
    import torch

    if optimizer_steps < 1:
        raise ValueError(
            "optimizer_steps must be >= 1."
        )

    if not encoded_examples:
        raise ValueError(
            "No encoded smoke-training examples were supplied."
        )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters exist."
        )

    optimizer = (
        bnb.optim.PagedAdamW8bit(
            trainable_parameters,

            lr=(
                config
                .training
                .learning_rate
            ),

            weight_decay=(
                config
                .training
                .weight_decay
            ),
        )
    )

    gradient_accumulation = (
        config
        .training
        .gradient_accumulation_steps
    )

    device = (
        get_model_input_device(
            model
        )
    )

    model.train()

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = False

    optimizer.zero_grad(
        set_to_none=True
    )

    completed_steps = 0
    micro_step = 0

    current_losses: list[float] = []
    optimizer_step_losses: list[float] = []

    while completed_steps < optimizer_steps:

        example = (
            encoded_examples[
                micro_step
                % len(
                    encoded_examples
                )
            ]
        )

        batch = (
            _move_batch_to_device(
                example,
                device,
            )
        )

        output = model(
            **batch
        )

        loss = (
            output.loss
        )

        raw_loss = float(
            loss
            .detach()
            .float()
            .item()
        )

        if not math.isfinite(
            raw_loss
        ):
            raise RuntimeError(
                f"Non-finite training loss encountered: {raw_loss}"
            )

        current_losses.append(
            raw_loss
        )

        (
            loss
            / gradient_accumulation
        ).backward()

        micro_step += 1

        if (
            micro_step
            % gradient_accumulation
            != 0
        ):
            continue

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        completed_steps += 1

        mean_step_loss = (
            sum(
                current_losses
            )
            / len(
                current_losses
            )
        )

        optimizer_step_losses.append(
            mean_step_loss
        )

        current_losses = []

    return {
        "optimizer_steps":
            completed_steps,

        "micro_steps":
            micro_step,

        "gradient_accumulation_steps":
            gradient_accumulation,

        "learning_rate":
            config
            .training
            .learning_rate,

        "loss_per_optimizer_step":
            [
                round(
                    value,
                    6,
                )
                for value
                in optimizer_step_losses
            ],

        "first_step_loss":
            round(
                optimizer_step_losses[0],
                6,
            ),

        "last_step_loss":
            round(
                optimizer_step_losses[-1],
                6,
            ),
    }