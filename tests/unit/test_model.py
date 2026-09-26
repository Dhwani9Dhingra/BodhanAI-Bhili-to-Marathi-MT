"""CPU-only tests for Bodhan prompt/masking/LoRA-selection logic."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from bodhan_bhili.core.config import load_config
from bodhan_bhili.model import (
    attach_qlora,
    build_completion_labels,
    build_translation_instruction,
    build_translation_messages,
    check_adapter_gradients,
    is_excluded_module_name,
    run_smoke_optimizer_steps,
    trainable_parameter_digest,
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


@pytest.mark.parametrize("checkpointing", [True, False])
def test_qlora_preserves_base_weights_and_updates_adapters(monkeypatch, checkpointing):
    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                use_cache=True, text_config=SimpleNamespace(use_cache=True)
            )
            self.embedding = torch.nn.Embedding(8, 4, dtype=torch.float16)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False, dtype=torch.float16) for _ in range(8)]
            )
            self.checkpoint_kwargs = None

        def get_input_embeddings(self):
            return self.embedding

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            self.checkpoint_kwargs = gradient_checkpointing_kwargs

        def forward(self, input_ids):
            hidden = self.embedding(input_ids)

            def block(values):
                delta = values.float() @ self.lora_A.T @ self.lora_B.T
                return self.layers[0](values).float() + delta

            if self.checkpoint_kwargs is not None:
                output = checkpoint(block, hidden, **self.checkpoint_kwargs)
            else:
                output = block(hidden)
            return SimpleNamespace(loss=output.square().mean())

    def attach_test_adapter(model, lora_config, *, autocast_adapter_dtype):
        assert autocast_adapter_dtype is True
        assert lora_config.r == 8
        assert all(not parameter.requires_grad for parameter in model.parameters())
        model.lora_A = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.float16) / 4)
        model.lora_B = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.float16))
        return model

    # Use real autograd; substitute only PEFT attachment and the bitsandbytes optimizer.
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=attach_test_adapter,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes",
        SimpleNamespace(
            nn=SimpleNamespace(), optim=SimpleNamespace(PagedAdamW8bit=torch.optim.AdamW)
        ),
    )
    torch.manual_seed(42)
    model = TinyModel()
    base = {
        name: (parameter, parameter.detach().clone(), parameter.data_ptr())
        for name, parameter in model.named_parameters()
    }
    config = load_config("configs/smoke.yaml")
    config.training.gradient_checkpointing = checkpointing
    config.training.gradient_accumulation_steps = 2
    model, report = attach_qlora(model, config)

    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False
    assert model.checkpoint_kwargs == ({"use_reentrant": False} if checkpointing else None)
    assert report["preparation"] == "freeze_base_preserve_dtype"
    assert model.lora_A.dtype == model.lora_B.dtype == torch.float32

    initial = trainable_parameter_digest(model)
    training = run_smoke_optimizer_steps(
        model, [{"input_ids": torch.tensor([[1, 2, 3]])}], config, optimizer_steps=2
    )
    assert training["optimizer_steps"] == 2
    assert training["parameters_with_gradients_per_step"] == [2, 2]
    assert trainable_parameter_digest(model) != initial
    for parameter, original, pointer in base.values():
        assert parameter.dtype == torch.float16
        assert parameter.data_ptr() == pointer
        assert not parameter.requires_grad
        assert parameter.grad is None
        assert torch.equal(parameter, original)


@pytest.mark.parametrize(
    "gradient, message",
    [
        (None, "No adapter gradients"),
        (0.0, "All adapter gradients are zero"),
        (float("nan"), "Non-finite adapter gradients"),
        (float("inf"), "Non-finite adapter gradients"),
    ],
)
def test_invalid_adapter_gradients_are_rejected(gradient, message):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.ones(1))
    if gradient is not None:
        parameter.grad = torch.tensor([gradient])
    with pytest.raises(RuntimeError, match=message):
        check_adapter_gradients([parameter])


def test_gpu_memory_is_saved_when_adapter_setup_fails(monkeypatch, tmp_path):
    import pandas as pd

    from scripts import smoke_model

    monkeypatch.setenv("BODHAN_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("BODHAN_RUN_ID", "memory_failure")
    monkeypatch.setattr(sys, "argv", ["smoke_model", "--eval-examples", "1"])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                memory_allocated=lambda: 2 * 1024**3,
                memory_reserved=lambda: 3 * 1024**3,
                max_memory_allocated=lambda: 4 * 1024**3,
                max_memory_reserved=lambda: 5 * 1024**3,
            )
        ),
    )
    frame = pd.DataFrame([{"record_id": "1", "group_id": "1", "source": "a", "target": "b"}])
    monkeypatch.setattr(smoke_model, "require_prepared_data", lambda paths: None)
    monkeypatch.setattr(smoke_model, "load_prepared_split", lambda path: frame)
    monkeypatch.setattr(smoke_model, "load_processor", lambda config: object())
    monkeypatch.setattr(smoke_model, "load_quantized_base_model", lambda config: object())
    monkeypatch.setattr(smoke_model, "translate", lambda *args, **kwargs: "b")

    def fail_setup(*args):
        raise RuntimeError("simulated setup OOM")

    monkeypatch.setattr(smoke_model, "attach_qlora", fail_setup)
    assert smoke_model.main() == 1
    report = json.loads(
        (tmp_path / "bodhan-bhili-mt/memory_failure/reports/model_smoke_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["error"] == "simulated setup OOM"
    assert "after_qlora_setup" not in report["gpu_memory"]
    for stage in ("before_base_load", "after_base_load", "before_qlora_setup", "failure"):
        assert report["gpu_memory"][stage] == {
            "allocated_gib": 2.0,
            "reserved_gib": 3.0,
            "peak_allocated_gib": 4.0,
            "peak_reserved_gib": 5.0,
        }
