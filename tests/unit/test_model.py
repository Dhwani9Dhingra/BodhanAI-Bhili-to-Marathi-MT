"""CPU-only tests for Bodhan prompt/masking/LoRA-selection logic."""

from __future__ import annotations

import pytest

from bodhan_bhili.model import (
    build_completion_labels,
    build_translation_instruction,
    build_translation_messages,
    is_excluded_module_name,
)


def test_translation_prompt_matches_bodhan_format() -> None:
    """The source language must not be named in the instruction."""
    source = "आमरा गावमा पानी आवे हाय."
    prompt = build_translation_instruction(source, "Marathi")

    assert prompt == "Translate the following text into Marathi:\n\nआमरा गावमा पानी आवे हाय."

    assert "from Dehwali Bhili" not in prompt


def test_translation_uses_single_user_message() -> None:
    """Bodhan should receive no system message."""
    messages = build_translation_messages("आमरा गावमा पानी आवे हाय.", "Marathi")

    assert len(messages) == 1

    assert messages[0]["role"] == "user"


def test_empty_source_is_rejected() -> None:
    """Empty translation requests should fail early."""
    with pytest.raises(ValueError, match="source_text cannot be empty"):
        build_translation_instruction("   ", "Marathi")


def test_completion_labels_mask_prompt() -> None:
    """Prompt tokens should be ignored by the training loss."""
    prompt_ids = [2, 10, 11, 12]
    full_ids = [2, 10, 11, 12, 50, 51, 1]
    labels = build_completion_labels(prompt_ids, full_ids)

    assert labels == [-100, -100, -100, -100, 50, 51, 1]


def test_completion_labels_reject_bad_prefix() -> None:
    """We must never silently construct an incorrect loss mask."""
    with pytest.raises(ValueError, match="not an exact prefix"):
        build_completion_labels([2, 10, 11], [2, 99, 11, 50])


@pytest.mark.parametrize(
    "module_name",
    [
        "model.vision_tower.layers.0.q_proj",
        "model.visual.encoder.linear",
        "model.multi_modal_projector.linear",
        "model.image_encoder.proj",
        "model.audio_encoder.proj",
        "model.lm_head",
        "model.embed_tokens",
    ],
)
def test_multimodal_modules_are_excluded(module_name: str) -> None:
    """Non-language components must not receive LoRA adapters."""
    exclusions = (
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

    assert is_excluded_module_name(module_name, exclusions) is True


def test_language_model_projection_is_not_excluded() -> None:
    """Normal text-transformer projection layers must remain eligible."""
    exclusions = (
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

    name = "model.language_model.layers.10.self_attn.q_proj"

    assert is_excluded_module_name(name, exclusions) is False
