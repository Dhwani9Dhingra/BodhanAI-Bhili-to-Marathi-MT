"""CPU-only tests for checkpoint selection and resume safety."""

from __future__ import annotations

import json

import pytest

from bodhan_bhili.core.config import load_config
from bodhan_bhili.training import (
    REQUIRED_CHECKPOINT_FILES,
    build_training_fingerprint,
    find_resume_checkpoint,
    is_complete_checkpoint,
    quarantine_checkpoints,
    read_best_metric,
    save_best_adapter,
    snapshot_adapter,
    verify_resume_fingerprint,
    warmup_steps,
)


def make_checkpoint(trainer_dir, step, *, skip=None, state_step=None):
    """Create a fake Trainer checkpoint folder."""
    checkpoint = trainer_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)

    for name in REQUIRED_CHECKPOINT_FILES:
        if name == skip:
            continue

        if name == "trainer_state.json":
            payload = {"global_step": step if state_step is None else state_step, "max_steps": 300}
            (checkpoint / name).write_text(json.dumps(payload), encoding="utf-8")

        else:
            (checkpoint / name).write_bytes(b"x")

    return checkpoint


def make_fingerprint(config):
    """Fingerprint with fixed data hashes."""
    return build_training_fingerprint(config, train_sha256="train", dev_sha256="dev")


def test_complete_checkpoint_is_accepted(tmp_path):
    """A checkpoint with every required file can be resumed."""
    assert is_complete_checkpoint(make_checkpoint(tmp_path, 25)) is True


@pytest.mark.parametrize("missing", REQUIRED_CHECKPOINT_FILES)
def test_checkpoint_missing_any_file_is_rejected(tmp_path, missing):
    """A save interrupted by a disconnect must not be resumed."""
    assert is_complete_checkpoint(make_checkpoint(tmp_path, 25, skip=missing)) is False


def test_empty_or_corrupt_files_are_rejected(tmp_path):
    """Zero-byte files and unreadable trainer state mean an unfinished Drive sync."""
    empty = make_checkpoint(tmp_path, 25)
    (empty / "optimizer.pt").write_bytes(b"")

    corrupt = make_checkpoint(tmp_path, 50)
    (corrupt / "trainer_state.json").write_text("{truncated", encoding="utf-8")

    assert is_complete_checkpoint(empty) is False
    assert is_complete_checkpoint(corrupt) is False


def test_trainer_state_step_must_match_folder(tmp_path):
    """The folder name and saved global step must agree."""
    assert is_complete_checkpoint(make_checkpoint(tmp_path, 50, state_step=25)) is False


def test_newest_complete_checkpoint_is_selected_numerically(tmp_path):
    """checkpoint-100 is newer than checkpoint-25 despite string ordering."""
    make_checkpoint(tmp_path, 25)
    newest = make_checkpoint(tmp_path, 100)
    (tmp_path / "runs").mkdir()

    assert find_resume_checkpoint(tmp_path) == (newest, [])


def test_incomplete_newer_checkpoint_falls_back(tmp_path):
    """An unfinished newest save is reported and the previous one is used."""
    previous = make_checkpoint(tmp_path, 25)
    broken = make_checkpoint(tmp_path, 50, skip="optimizer.pt")

    assert find_resume_checkpoint(tmp_path) == (previous, [broken])


def test_no_checkpoints_means_fresh_start(tmp_path):
    """Missing or empty trainer folders start from step 0."""
    assert find_resume_checkpoint(tmp_path / "missing") == (None, [])
    assert find_resume_checkpoint(tmp_path) == (None, [])


def test_quarantine_moves_checkpoints_out_of_trainer_dir(tmp_path):
    """Incomplete checkpoints are moved aside, not deleted."""
    trainer_dir = tmp_path / "trainer"
    broken = make_checkpoint(trainer_dir, 50, skip="scheduler.pt")
    moved = quarantine_checkpoints([broken], tmp_path / "trainer_incomplete")

    assert not broken.exists()
    assert len(moved) == 1
    assert moved[0].parent == tmp_path / "trainer_incomplete"
    assert (moved[0] / "trainer_state.json").exists()
    assert find_resume_checkpoint(trainer_dir) == (None, [])


def test_matching_fingerprint_allows_resume(tmp_path):
    """Resuming with identical settings is allowed."""
    config = load_config("configs/colab_t4.yaml")
    fingerprint_file = tmp_path / "training_fingerprint.json"
    fingerprint_file.write_text(json.dumps(make_fingerprint(config)), encoding="utf-8")

    verify_resume_fingerprint(fingerprint_file, make_fingerprint(config))


def test_changed_hyperparameter_blocks_resume(tmp_path):
    """Optimizer state from a different learning rate must not be reused."""
    config = load_config("configs/colab_t4.yaml")
    fingerprint_file = tmp_path / "training_fingerprint.json"
    fingerprint_file.write_text(json.dumps(make_fingerprint(config)), encoding="utf-8")
    config.training.learning_rate = 1e-4

    with pytest.raises(RuntimeError, match=r"Changed: \['training'\]"):
        verify_resume_fingerprint(fingerprint_file, make_fingerprint(config))


def test_changed_data_blocks_resume(tmp_path):
    """Re-prepared data must not be mixed into an existing run."""
    config = load_config("configs/colab_t4.yaml")
    fingerprint_file = tmp_path / "training_fingerprint.json"
    fingerprint_file.write_text(json.dumps(make_fingerprint(config)), encoding="utf-8")
    changed = build_training_fingerprint(config, train_sha256="other", dev_sha256="dev")

    with pytest.raises(RuntimeError, match=r"Changed: \['train_sha256'\]"):
        verify_resume_fingerprint(fingerprint_file, changed)


def test_checkpoint_frequency_change_allows_resume(tmp_path):
    """Save/eval/logging cadence does not affect optimizer state."""
    config = load_config("configs/colab_t4.yaml")
    fingerprint_file = tmp_path / "training_fingerprint.json"
    fingerprint_file.write_text(json.dumps(make_fingerprint(config)), encoding="utf-8")
    config.training.save_steps = 10
    config.training.evaluation_steps = 20
    config.training.logging_steps = 1

    verify_resume_fingerprint(fingerprint_file, make_fingerprint(config))


def write_fingerprint(tmp_path, config):
    """Persist the fingerprint of `config` and return its path."""
    fingerprint_file = tmp_path / "training_fingerprint.json"
    fingerprint_file.write_text(json.dumps(make_fingerprint(config)), encoding="utf-8")

    return fingerprint_file


def test_max_steps_change_requires_extend_flag(tmp_path):
    """Raising max_steps without --extend is refused with a hint."""
    config = load_config("configs/colab_t4.yaml")
    config.training.max_steps = 700
    fingerprint_file = write_fingerprint(tmp_path, config)
    config.training.max_steps = 1400

    with pytest.raises(RuntimeError, match="pass --extend"):
        verify_resume_fingerprint(fingerprint_file, make_fingerprint(config))


def test_extend_allows_larger_max_steps(tmp_path):
    """--extend returns the previous max_steps when only max_steps grew."""
    config = load_config("configs/colab_t4.yaml")
    config.training.max_steps = 700
    fingerprint_file = write_fingerprint(tmp_path, config)
    config.training.max_steps = 1400

    extended_from = verify_resume_fingerprint(
        fingerprint_file, make_fingerprint(config), allow_extension=True
    )

    assert extended_from == 700


def test_extend_with_unchanged_max_steps_is_a_normal_resume(tmp_path):
    """Passing --extend again after the extension started is harmless."""
    config = load_config("configs/colab_t4.yaml")
    fingerprint_file = write_fingerprint(tmp_path, config)

    assert (
        verify_resume_fingerprint(fingerprint_file, make_fingerprint(config), allow_extension=True)
        is None
    )


def test_extend_refuses_fewer_steps(tmp_path):
    """Extension cannot shorten a run."""
    config = load_config("configs/colab_t4.yaml")
    config.training.max_steps = 700
    fingerprint_file = write_fingerprint(tmp_path, config)
    config.training.max_steps = 500

    with pytest.raises(RuntimeError, match="can only increase"):
        verify_resume_fingerprint(fingerprint_file, make_fingerprint(config), allow_extension=True)


def test_extend_still_blocks_other_changes(tmp_path):
    """--extend must not smuggle in a different learning rate."""
    config = load_config("configs/colab_t4.yaml")
    config.training.max_steps = 700
    fingerprint_file = write_fingerprint(tmp_path, config)
    config.training.max_steps = 1400
    config.training.learning_rate = 1e-4

    with pytest.raises(RuntimeError, match=r"Changed: \['training'\]"):
        verify_resume_fingerprint(fingerprint_file, make_fingerprint(config), allow_extension=True)


def test_snapshot_adapter_copies_once(tmp_path):
    """The pre-extension adapter is preserved and never overwritten."""
    checkpoint = make_checkpoint(tmp_path / "trainer", 700)
    destination = tmp_path / "adapter_step_700"

    assert snapshot_adapter(checkpoint, destination) is True
    assert sorted(path.name for path in destination.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")

    assert snapshot_adapter(checkpoint, destination) is False
    assert (destination / "adapter_model.safetensors").read_bytes() == b"x"


def test_missing_fingerprint_blocks_resume(tmp_path):
    """Checkpoints of unknown origin must not be resumed."""
    config = load_config("configs/colab_t4.yaml")

    with pytest.raises(RuntimeError, match="fingerprint is missing"):
        verify_resume_fingerprint(tmp_path / "missing.json", make_fingerprint(config))


def test_warmup_ratio_rounds_up_to_whole_steps():
    """3% of 300 steps is 9 warmup steps."""
    config = load_config("configs/colab_t4.yaml")
    config.training.max_steps = 300

    assert warmup_steps(config) == 9


def test_best_adapter_is_replaced_atomically(tmp_path):
    """Replacing the best adapter leaves no staging folders behind."""

    class FakeModel:
        def __init__(self, marker):
            self.marker = marker

        def save_pretrained(self, directory, safe_serialization):
            directory.mkdir(parents=True)
            (directory / "adapter_model.safetensors").write_text(self.marker, encoding="utf-8")

    adapter_best = tmp_path / "adapter_best"
    adapter_best.mkdir()
    save_best_adapter(FakeModel("first"), adapter_best, {"eval_loss": 2.0, "global_step": 50})
    save_best_adapter(FakeModel("second"), adapter_best, {"eval_loss": 1.5, "global_step": 100})

    assert (adapter_best / "adapter_model.safetensors").read_text(encoding="utf-8") == "second"
    assert read_best_metric(adapter_best) == {"eval_loss": 1.5, "global_step": 100}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["adapter_best"]


def test_padding_masks_labels_and_attention():
    """Padded positions are ignored by attention and the loss."""
    torch = pytest.importorskip("torch")
    from bodhan_bhili.training import pad_training_batch, to_training_feature

    short = to_training_feature(
        {
            "input_ids": torch.tensor([[5, 6]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([[-100, 6]]),
        }
    )
    long = to_training_feature(
        {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 6, 7]]),
        }
    )
    batch = pad_training_batch([short, long], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[5, 6, 0], [5, 6, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch["labels"].tolist() == [[-100, 6, -100], [-100, 6, 7]]
