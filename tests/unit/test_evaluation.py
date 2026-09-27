"""CPU-only tests for evaluation storage, metrics, significance and reporting."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from bodhan_bhili.evaluation import (
    append_predictions,
    check_generation_settings,
    length_bucket,
    read_predictions,
    select_test_sample,
    validate_system_name,
)


def test_predictions_resume_skips_truncated_last_line(tmp_path):
    """A disconnect mid-write leaves a partial line that must be ignored."""
    store = tmp_path / "predictions_base.jsonl"
    append_predictions(store, [{"record_id": "1", "prediction": "एक"}])
    append_predictions(store, [{"record_id": "2", "prediction": "दोन"}])

    with store.open("a", encoding="utf-8") as handle:
        handle.write('{"record_id": "3", "predic')

    assert sorted(read_predictions(store)) == ["1", "2"]
    assert read_predictions(store)["1"]["prediction"] == "एक"

    append_predictions(store, [{"record_id": "3", "prediction": "तीन"}])

    assert sorted(read_predictions(store)) == ["1", "2", "3"]


def test_missing_store_is_empty(tmp_path):
    """No file means nothing translated yet."""
    assert read_predictions(tmp_path / "missing.jsonl") == {}


def test_changed_generation_settings_are_refused(tmp_path):
    """Predictions from two different adapters must not share one file."""
    meta = tmp_path / "generation_tuned.json"
    settings = {
        "system": "tuned",
        "model_repo_id": "m",
        "adapter_sha256": "a",
        "max_new_tokens": 256,
        "test_sha256": "t",
    }
    meta.write_text(json.dumps(settings), encoding="utf-8")

    check_generation_settings(meta, settings)

    with pytest.raises(RuntimeError, match="adapter_sha256"):
        check_generation_settings(meta, {**settings, "adapter_sha256": "b"})


def test_test_sample_is_deterministic_and_capped():
    """Every system and every account must see the same sentences."""
    frame = pd.DataFrame({"record_id": [str(i) for i in range(50)]})

    first = select_test_sample(frame, 10, 42)["record_id"].tolist()
    second = select_test_sample(frame, 10, 42)["record_id"].tolist()

    assert first == second
    assert len(select_test_sample(frame, 500, 42)) == 50


@pytest.mark.parametrize(
    ("words", "bucket"), [(1, "1-5"), (5, "1-5"), (6, "6-10"), (20, "11-20"), (40, "21+")]
)
def test_length_buckets(words, bucket):
    assert length_bucket(words) == bucket


def test_system_names_must_be_file_safe():
    assert validate_system_name("tuned_700") == "tuned_700"

    with pytest.raises(ValueError):
        validate_system_name("../base")


def test_copying_the_source_is_detected():
    """Bhili and Marathi share a script, so copying must be visible in the metrics."""
    pytest.importorskip("sacrebleu")
    from bodhan_bhili.evaluation import corpus_metrics

    sources = ["आमरा गावमा पानी आवे हाय", "तु कां जाय रयो"]
    references = ["आमच्या गावात पाणी येते", "तू कुठे जात आहेस"]
    metrics = corpus_metrics(sources, references, sources)

    assert metrics["exact_copy_rate"] == 1.0
    assert metrics["chrfpp_vs_source"] == pytest.approx(100.0)
    assert metrics["chrfpp"] < 100.0


def test_paired_bootstrap_detects_a_better_system():
    """A clearly better candidate gets a positive delta and a small p-value."""
    pytest.importorskip("sacrebleu")
    from bodhan_bhili.evaluation import paired_bootstrap_chrf

    references = [f"हे वाक्य क्रमांक {i} आहे" for i in range(40)]
    baseline = [f"ते {i}" for i in range(40)]

    better = paired_bootstrap_chrf(baseline, references, references, samples=200, seed=1)
    same = paired_bootstrap_chrf(baseline, baseline, references, samples=200, seed=1)

    assert better["observed_delta"] > 0
    assert better["ci95_low"] > 0
    assert better["p_value"] < 0.05
    assert same["observed_delta"] == 0
    assert same["p_value"] == 1.0


def test_left_padding_keeps_prompts_right_aligned():
    """Generation continues from the last real token of every prompt."""
    torch = pytest.importorskip("torch")
    from bodhan_bhili.evaluation import left_pad_prompts

    batch = left_pad_prompts(
        [
            {"input_ids": torch.tensor([5, 6]), "attention_mask": torch.tensor([1, 1])},
            {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.tensor([1, 1, 1])},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[0, 5, 6], [5, 6, 7]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]


def _prepare_run(monkeypatch, tmp_path, sentences):
    """Create a run folder with a hashed test split."""
    from bodhan_bhili.core.config import load_config
    from bodhan_bhili.core.paths import ArtifactPaths
    from bodhan_bhili.data import sha256_file

    monkeypatch.setenv("BODHAN_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("BODHAN_RUN_ID", "eval_test")
    config = load_config("configs/colab_t4.yaml")
    paths = ArtifactPaths.from_config(config)
    paths.create_directories()
    pd.DataFrame(sentences).to_csv(paths.test_data, sep="\t", index=False)
    paths.test_set_sha256.write_text(
        f"{sha256_file(paths.test_data)}  test.tsv\n", encoding="utf-8"
    )

    return paths


def _write_system(paths, system, predictions, adapter_path=None):
    rows = [
        {"record_id": rid, "prediction": text, "hit_token_limit": False}
        for rid, text in predictions.items()
    ]
    append_predictions(paths.evaluation / f"predictions_{system}.jsonl", rows)
    (paths.evaluation / f"generation_{system}.json").write_text(
        json.dumps({"system": system, "max_new_tokens": 256, "adapter_path": adapter_path}),
        encoding="utf-8",
    )


def test_report_scores_compares_and_packages(monkeypatch, tmp_path):
    """End-to-end report: metrics, significance, analysis CSVs and package."""
    pytest.importorskip("sacrebleu")
    from scripts import evaluate

    sentences = [
        {
            "record_id": str(i),
            "group_id": str(i),
            "source": f"आमरा गाव {i} मा पानी",
            "target": f"आमच्या गावात {i} पाणी आहे",
        }
        for i in range(30)
    ]
    paths = _prepare_run(monkeypatch, tmp_path, sentences)
    adapter = tmp_path / "adapter_best"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    _write_system(paths, "base", {s["record_id"]: s["source"] for s in sentences})
    _write_system(paths, "tuned", {s["record_id"]: s["target"] for s in sentences}, str(adapter))

    monkeypatch.setattr(sys, "argv", ["evaluate", "report"])
    assert evaluate.main() == 0

    report = json.loads((paths.reports / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["partial"] is False
    assert report["sentences"] == 30
    assert report["systems"]["tuned"]["chrfpp"] == pytest.approx(100.0)
    assert report["systems"]["base"]["exact_copy_rate"] == 1.0
    assert report["systems"]["copy_source"]["chrfpp"] == report["systems"]["base"]["chrfpp"]
    assert report["significance"]["tuned"]["observed_delta"] > 0
    assert (paths.evaluation / "most_improved.csv").exists()
    assert (paths.evaluation / "test_predictions.csv").exists()
    assert (paths.run_root / "package" / "adapter" / "adapter_model.safetensors").exists()
    assert "chrF++" in (paths.run_root / "package" / "README.md").read_text(encoding="utf-8")


def test_report_refuses_incomplete_systems(monkeypatch, tmp_path):
    """Scoring half-finished predictions requires an explicit --allow-partial."""
    pytest.importorskip("sacrebleu")
    from scripts import evaluate

    sentences = [
        {"record_id": str(i), "group_id": str(i), "source": f"स्रोत {i}", "target": f"लक्ष्य {i}"}
        for i in range(10)
    ]
    paths = _prepare_run(monkeypatch, tmp_path, sentences)
    _write_system(paths, "base", {"0": "लक्ष्य 0", "1": "x"})

    monkeypatch.setattr(sys, "argv", ["evaluate", "report"])
    assert evaluate.main() == 1

    monkeypatch.setattr(sys, "argv", ["evaluate", "report", "--allow-partial"])
    assert evaluate.main() == 0
    report = json.loads((paths.reports / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["partial"] is True
    assert report["sentences"] == 2
    assert not (paths.run_root / "package").exists()
