"""Resumable test-set evaluation for Bhili -> Marathi.

    generate --system base              Untuned Bodhan baseline
    generate --system tuned             Trained adapter (default: checkpoints/adapter_best)
    generate --system NAME --adapter P  Any other adapter, e.g. checkpoints/adapter_step_701
    report                              Metrics, bootstrap significance, analysis, packaging

Predictions are appended to <run>/evaluation/predictions_<system>.jsonl batch by batch,
so re-running the same command after a disconnect continues where it stopped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bodhan_bhili.core.config import load_config
from bodhan_bhili.core.constants import (
    STAGE_ADAPTER_VERIFICATION,
    STAGE_ANALYSIS,
    STAGE_BASELINES,
    STAGE_FINAL_EVALUATION,
    STAGE_PACKAGING,
)
from bodhan_bhili.core.logging import setup_logging
from bodhan_bhili.core.manifest import build_initial_manifest, write_manifest
from bodhan_bhili.core.paths import ArtifactPaths
from bodhan_bhili.core.run_state import RunStateTracker
from bodhan_bhili.core.serialization import atomic_write_json, read_json
from bodhan_bhili.data import sha256_file
from bodhan_bhili.evaluation import (
    append_predictions,
    check_generation_settings,
    corpus_metrics,
    encode_prompt,
    generate_with_backoff,
    length_bucket,
    lora_b_is_trained,
    paired_bootstrap_chrf,
    read_predictions,
    read_sha256_sidecar,
    select_test_sample,
    sentence_chrf,
    validate_system_name,
)
from bodhan_bhili.model import load_processor, load_quantized_base_model
from scripts.smoke_model import load_prepared_split, record_gpu_memory

BASE_SYSTEM = "base"
PRIMARY_SYSTEM = "tuned"
COPY_SYSTEM = "copy_source"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Bhili -> Marathi systems.")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Translate the test sample with one system.")
    generate.add_argument("--config", default="configs/colab_t4.yaml")
    generate.add_argument(
        "--system", required=True, help="'base', 'tuned', or a custom name with --adapter."
    )
    generate.add_argument(
        "--adapter",
        default=None,
        help="Adapter folder (absolute, or relative to the run folder). "
        "Default for non-base systems: checkpoints/adapter_best.",
    )
    generate.add_argument(
        "--limit", type=int, default=None, help="Only the first N sampled sentences (smoke test)."
    )
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--max-new-tokens", type=int, default=256)

    report = commands.add_parser("report", help="Score, analyse and package.")
    report.add_argument("--config", default="configs/colab_t4.yaml")
    report.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score only sentences every system has translated; skips packaging.",
    )

    return parser.parse_args()


def utc_now() -> str:
    """Current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_test_sample(config, paths: ArtifactPaths) -> tuple[pd.DataFrame, str]:
    """Load the frozen test split, verify its hash, and draw the shared sample."""
    if not paths.test_data.exists():
        raise FileNotFoundError(
            f"Test split missing: {paths.test_data}. Run scripts.prepare_data with this config."
        )

    test_sha256 = sha256_file(paths.test_data)
    expected = read_sha256_sidecar(paths.test_set_sha256)

    if test_sha256 != expected:
        raise RuntimeError(
            f"Test split hash {test_sha256} does not match the frozen hash {expected}."
        )

    sample = select_test_sample(
        load_prepared_split(paths.test_data),
        config.evaluation.test_generation_samples,
        config.project.seed,
    )

    return sample, test_sha256


def prediction_files(paths: ArtifactPaths, system: str) -> tuple[Path, Path]:
    """Predictions store and its settings file for one system."""
    return (
        paths.evaluation / f"predictions_{system}.jsonl",
        paths.evaluation / f"generation_{system}.json",
    )


def resolve_adapter(paths: ArtifactPaths, system: str, adapter: str | None) -> Path | None:
    """Pick the adapter folder for a system."""
    if system == BASE_SYSTEM:
        if adapter:
            raise ValueError("The base system must not use an adapter.")

        return None

    path = Path(adapter) if adapter else paths.adapter_best

    if not path.is_absolute():
        path = paths.run_root / path

    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {path}.")

    return path


def run_generate(args, config, paths: ArtifactPaths, logger, state: RunStateTracker) -> int:
    """Translate the remaining test sentences for one system."""
    system = validate_system_name(args.system)
    stage = STAGE_BASELINES if system == BASE_SYSTEM else STAGE_FINAL_EVALUATION
    adapter = resolve_adapter(paths, system, args.adapter)
    (sample, test_sha256) = load_test_sample(config, paths)
    (predictions_file, meta_file) = prediction_files(paths, system)
    settings = {
        "system": system,
        "model_repo_id": config.model.repo_id,
        "adapter_path": (str(adapter) if adapter else None),
        "adapter_sha256": (sha256_file(adapter / "adapter_model.safetensors") if adapter else None),
        "max_new_tokens": args.max_new_tokens,
        "test_sha256": test_sha256,
        "sample_size": len(sample),
        "seed": config.project.seed,
    }

    check_generation_settings(meta_file, settings)

    if not meta_file.exists():
        atomic_write_json({**settings, "created_at_utc": utc_now()}, meta_file)

    target = sample if args.limit is None else sample.iloc[: args.limit]
    done = read_predictions(predictions_file)
    remaining = target[~target["record_id"].astype(str).isin(done)]

    logger.info(
        "System %s: %d/%d sentences already translated, %d to go.",
        system,
        len(target) - len(remaining),
        len(target),
        len(remaining),
    )

    if remaining.empty:
        logger.info("Nothing to do for system %s.", system)

        if args.limit is None:
            state.mark_completed(stage, message=f"{system}: {len(sample)} sentences translated.")

        return 0

    state.mark_started(stage, message=f"Generating test translations for {system}.")

    gpu_memory = {}
    processor = load_processor(config)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    logger.info("Loading Bodhan in 4-bit NF4...")
    model = load_quantized_base_model(config)
    record_gpu_memory("after_base_load", gpu_memory)

    if adapter is not None:
        from peft import PeftModel

        state.mark_started(STAGE_ADAPTER_VERIFICATION, message=f"Loading {adapter}.")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)

        if not lora_b_is_trained(model):
            raise RuntimeError(f"Adapter {adapter} loaded but all LoRA B weights are zero.")

        state.mark_completed(
            STAGE_ADAPTER_VERIFICATION,
            message=f"{adapter.name} loaded on a fresh base model with trained LoRA weights.",
        )
        logger.info("Adapter verified: %s", adapter)

    model.eval()

    encoded = [
        (row, encode_prompt(processor, row["source"], config.model.target_language))
        for _, row in remaining.iterrows()
    ]
    # Similar prompt lengths per batch minimise padding.
    encoded.sort(key=lambda item: item[1]["input_ids"].shape[-1])

    started = time.monotonic()
    translated = 0
    truncated_total = 0

    for offset in range(0, len(encoded), args.batch_size):
        chunk = encoded[offset : offset + args.batch_size]
        (texts, truncated) = generate_with_backoff(
            model,
            processor,
            [prompt for _, prompt in chunk],
            max_new_tokens=args.max_new_tokens,
            pad_token_id=pad_token_id,
            logger=logger,
        )
        append_predictions(
            predictions_file,
            [
                {
                    "record_id": str(row["record_id"]),
                    "group_id": str(row["group_id"]),
                    "source": row["source"],
                    "target": row["target"],
                    "prediction": text,
                    "hit_token_limit": hit,
                    "system": system,
                }
                for (row, _), text, hit in zip(chunk, texts, truncated, strict=True)
            ],
        )

        translated += len(chunk)
        truncated_total += sum(truncated)
        elapsed = time.monotonic() - started
        rate = translated / elapsed
        logger.info(
            "%s: %d/%d translated (%.2f sent/s, ETA %.1f min, hit token limit: %d)",
            system,
            translated,
            len(encoded),
            rate,
            (len(encoded) - translated) / rate / 60,
            truncated_total,
        )

    record_gpu_memory("after_generation", gpu_memory)
    meta = read_json(meta_file)
    meta.update({"updated_at_utc": utc_now(), "gpu_memory": gpu_memory})
    atomic_write_json(meta, meta_file)

    complete = len(read_predictions(predictions_file)) >= len(sample)

    if complete:
        state.mark_completed(stage, message=f"{system}: {len(sample)} sentences translated.")

    logger.info(
        "System %s finished this session (%s). Predictions: %s",
        system,
        "complete" if complete else f"limited to {len(target)}",
        predictions_file,
    )

    return 0


def write_model_card(package: Path, config, report: dict) -> None:
    """Short README describing the packaged adapter and its test results."""
    rows = "\n".join(
        f"| {name} | {m['chrfpp']:.2f} | {m['bleu']:.2f} | {m['exact_copy_rate']:.3f} |"
        for name, m in report["systems"].items()
    )
    significance = report["significance"].get(PRIMARY_SYSTEM)
    significance_text = (
        f"Paired bootstrap ({significance['samples']} samples), tuned minus base chrF++: "
        f"{significance['observed_delta']:+.2f} "
        f"(95% CI {significance['ci95_low']:+.2f} to {significance['ci95_high']:+.2f}, "
        f"p = {significance['p_value']})."
        if significance
        else "No base-model comparison was available."
    )
    card = f"""# Bodhan Indic-Translate QLoRA adapter: Dehwali Bhili → Marathi

Base model: `{config.model.repo_id}`, loaded in 4-bit NF4. LoRA rank {config.lora.rank},
alpha {config.lora.alpha}, dropout {config.lora.dropout}, trained for
{report["training"].get("global_step")} optimizer steps (effective batch
{config.training.train_batch_size * config.training.gradient_accumulation_steps}).
Best dev loss: {report["training"].get("best_dev_loss")}.

## Test results

{report["sentences"]} sentences sampled (seed {config.project.seed}) from the frozen test split
(SHA-256 `{report["test_sha256"]}`). Greedy decoding, at most {report["max_new_tokens"]} new tokens.

| System | chrF++ | BLEU | Exact copy of source |
| --- | --- | --- | --- |
{rows}

`copy_source` outputs the Bhili input unchanged; both languages use Devanagari, so it shows
how much credit copying alone earns.

{significance_text}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("adapter")
base = AutoModelForImageTextToText.from_pretrained("{config.model.repo_id}")
model = PeftModel.from_pretrained(base, "adapter")
```

Prompt: a single user message, `Translate the following text into Marathi:\\n\\n<Bhili text>`.

## Limitations

Scores come from one test sample of {report["sentences"]} sentences with automatic metrics only;
no human evaluation. Retention of the base model's other translation directions was not measured.
"""
    (package / "README.md").write_text(card, encoding="utf-8")


def run_report(args, config, paths: ArtifactPaths, logger, state: RunStateTracker) -> int:
    """Score every generated system, test significance, analyse and package."""
    state.mark_started(STAGE_ANALYSIS, message="Scoring test predictions.")
    (sample, test_sha256) = load_test_sample(config, paths)
    systems = sorted(
        path.stem.removeprefix("predictions_")
        for path in paths.evaluation.glob("predictions_*.jsonl")
    )

    if not systems:
        raise FileNotFoundError(f"No predictions_*.jsonl files in {paths.evaluation}.")

    sample_ids = sample["record_id"].astype(str).tolist()
    predictions = {}
    settings = {}

    for system in systems:
        (predictions_file, meta_file) = prediction_files(paths, system)
        meta = read_json(meta_file) if meta_file.exists() else {}

        if meta.get("test_sha256") not in (None, test_sha256):
            raise RuntimeError(f"System {system} was generated on a different test split.")

        settings[system] = meta
        predictions[system] = read_predictions(predictions_file)
        logger.info("System %s: %d/%d predictions.", system, len(predictions[system]), len(sample))

    incomplete = [s for s in systems if not set(sample_ids) <= set(predictions[s])]

    if incomplete and not args.allow_partial:
        raise RuntimeError(
            f"Incomplete systems: {incomplete}. Finish generation, or pass --allow-partial."
        )

    common = [rid for rid in sample_ids if all(rid in predictions[s] for s in systems)]

    if not common:
        raise RuntimeError("No sentence has been translated by every system yet.")

    frame = sample.assign(record_id=sample["record_id"].astype(str)).set_index("record_id")
    frame = frame.loc[common].reset_index()
    frame[f"prediction_{COPY_SYSTEM}"] = frame["source"]

    for system in systems:
        frame[f"prediction_{system}"] = [predictions[system][rid]["prediction"] for rid in common]
        frame[f"hit_token_limit_{system}"] = [
            bool(predictions[system][rid].get("hit_token_limit")) for rid in common
        ]

    all_systems = [COPY_SYSTEM, *systems]
    references = frame["target"].tolist()
    sources = frame["source"].tolist()
    metrics = {}

    for system in all_systems:
        metrics[system] = corpus_metrics(frame[f"prediction_{system}"], references, sources)
        frame[f"chrfpp_{system}"] = sentence_chrf(frame[f"prediction_{system}"], references)

        if system != COPY_SYSTEM:
            metrics[system]["hit_token_limit_rate"] = round(
                float(frame[f"hit_token_limit_{system}"].mean()), 4
            )

    significance = {}

    if BASE_SYSTEM in systems:
        for system in systems:
            if system == BASE_SYSTEM:
                continue

            significance[system] = paired_bootstrap_chrf(
                frame[f"prediction_{BASE_SYSTEM}"].tolist(),
                frame[f"prediction_{system}"].tolist(),
                references,
                samples=config.evaluation.bootstrap_samples,
                seed=config.project.seed,
            )

    frame["source_words"] = frame["source"].str.split().str.len()
    frame["length_bucket"] = frame["source_words"].map(length_bucket)
    by_length = {}

    for bucket, group in frame.groupby("length_bucket", sort=False):
        by_length[bucket] = {"sentences": len(group)}

        for system in all_systems:
            by_length[bucket][system] = corpus_metrics(
                group[f"prediction_{system}"], group["target"], group["source"]
            )["chrfpp"]

    primary = PRIMARY_SYSTEM if PRIMARY_SYSTEM in systems else None

    if primary and BASE_SYSTEM in systems:
        frame["chrfpp_delta"] = frame[f"chrfpp_{primary}"] - frame[f"chrfpp_{BASE_SYSTEM}"]
        ordered = frame.sort_values("chrfpp_delta")
        columns = [
            "record_id",
            "source",
            "target",
            f"prediction_{BASE_SYSTEM}",
            f"prediction_{primary}",
            f"chrfpp_{BASE_SYSTEM}",
            f"chrfpp_{primary}",
            "chrfpp_delta",
        ]
        ordered[columns].tail(20).iloc[::-1].to_csv(
            paths.evaluation / "most_improved.csv", index=False, encoding="utf-8"
        )
        ordered[columns].head(20).to_csv(
            paths.evaluation / "most_regressed.csv", index=False, encoding="utf-8"
        )

    frame.to_csv(paths.evaluation / "test_predictions.csv", index=False, encoding="utf-8")

    training = read_json(paths.training_report) if paths.training_report.exists() else {}
    report = {
        "created_at_utc": utc_now(),
        "partial": bool(incomplete),
        "sentences": len(common),
        "test_sha256": test_sha256,
        "max_new_tokens": sorted({meta.get("max_new_tokens") for meta in settings.values()}),
        "systems": metrics,
        "significance": significance,
        "by_source_length": by_length,
        "generation_settings": settings,
        "training": {
            "global_step": training.get("global_step"),
            "best_dev_loss": (training.get("best_dev") or {}).get("eval_loss"),
        },
    }
    report["max_new_tokens"] = (
        report["max_new_tokens"][0]
        if len(report["max_new_tokens"]) == 1
        else report["max_new_tokens"]
    )
    report_path = paths.reports / "evaluation_report.json"
    atomic_write_json(report, report_path)
    state.mark_completed(STAGE_ANALYSIS, message=f"Scored {len(systems)} systems.")

    logger.info("=" * 72)
    logger.info("TEST RESULTS (%d sentences%s)", len(common), ", PARTIAL" if incomplete else "")

    for system in all_systems:
        m = metrics[system]
        logger.info(
            "%-14s chrF++ %6.2f | BLEU %6.2f | copy rate %.3f | empty %.3f",
            system,
            m["chrfpp"],
            m["bleu"],
            m["exact_copy_rate"],
            m["empty_rate"],
        )

    for system, result in significance.items():
        logger.info(
            "%s vs base: chrF++ %+.2f (95%% CI %+.2f to %+.2f, p=%s)",
            system,
            result["observed_delta"],
            result["ci95_low"],
            result["ci95_high"],
            result["p_value"],
        )

    logger.info("Report: %s", report_path)

    if incomplete or primary is None:
        logger.info("Packaging skipped (needs complete predictions for the 'tuned' system).")
        logger.info("=" * 72)
        return 0

    state.mark_started(STAGE_PACKAGING, message="Packaging adapter and results.")
    package = paths.run_root / "package"
    adapter = Path(settings[primary]["adapter_path"])
    shutil.copytree(adapter, package / "adapter", dirs_exist_ok=True)
    atomic_write_json(report, package / "evaluation_report.json")
    write_model_card(package, config, report)

    if paths.manifest.exists():
        manifest = read_json(paths.manifest)

    else:
        manifest = build_initial_manifest(config)

    manifest["status"] = "evaluated_and_packaged"
    manifest["final_metrics"] = {
        system: {"chrfpp": m["chrfpp"], "bleu": m["bleu"]} for system, m in metrics.items()
    }
    manifest["significance"] = significance
    write_manifest(manifest, paths)
    state.mark_completed(STAGE_PACKAGING, message=f"Package written to {package}.")
    logger.info("Package: %s", package)
    logger.info("=" * 72)

    return 0


def main() -> int:
    """Dispatch the evaluation command."""
    args = parse_args()
    config = load_config(args.config)
    paths = ArtifactPaths.from_config(config)

    paths.create_directories()

    logger = setup_logging(paths.pipeline_log)
    state = RunStateTracker(state_file=paths.run_state, run_id=config.run.run_id)

    logger.info("=" * 72)
    logger.info("BODHAN BHILI MT — EVALUATION (%s)", args.command)
    logger.info("=" * 72)
    logger.info("Run ID: %s", config.run.run_id)

    try:
        if args.command == "generate":
            return run_generate(args, config, paths, logger, state)

        return run_report(args, config, paths, logger, state)

    except BaseException as exc:
        # BaseException also covers a manual "stop cell" (KeyboardInterrupt).
        logger.exception("Evaluation failed: %s", exc)
        logger.info("Re-run the same command to continue; finished batches are kept.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
