"""Build the results dashboard (one self-contained HTML file) from a run folder.

    python -m scripts.dashboard --run-dir <run folder> --output Dashboard.html

Every number is read or computed from files in the run folder:
reports/evaluation_report.json, reports/training_report.json, reports/resolved_config.json,
reports/trainable_parameters.json, evaluation/test_predictions.csv and
checkpoints/adapter_best/. The page layout lives in scripts/dashboard_template.html.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

TEMPLATE = Path(__file__).parent / "dashboard_template.html"
LORA_KEY = re.compile(r"^(.+)\.lora_([AB])\.weight$")
MODEL_SYSTEMS = ("base", "tuned_700", "tuned")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON report, or return an empty dict if it is missing."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def adapter_statistics(adapter_dir: Path) -> dict[str, Any] | None:
    """Size of the weight change ΔW = (alpha/r)·B·A learned by each LoRA module type."""
    weights = adapter_dir / "adapter_model.safetensors"
    config = read_json(adapter_dir / "adapter_config.json")

    if not weights.exists() or not config:
        return None

    try:
        import numpy as np
        from safetensors.numpy import load_file

    except ImportError:
        return None

    rank = config["r"]
    scale = config["lora_alpha"] / (math.sqrt(rank) if config.get("use_rslora") else rank)
    matrices: dict[str, dict[str, Any]] = {}

    for key, tensor in load_file(str(weights)).items():
        match = LORA_KEY.match(key)

        if match:
            path, which = match.groups()
            matrices.setdefault(path, {})[which] = tensor.astype("float64")

    rows = []

    for path, pair in matrices.items():
        a, b = pair["A"], pair["B"]
        # ||s·B·A||_F² = s²·trace((A·Aᵀ)(Bᵀ·B)); avoids building the full out x in matrix.
        norm = scale * math.sqrt(max(float(np.trace((a @ a.T) @ (b.T @ b))), 0.0))
        rows.append(
            {"module": path.split(".")[-1], "norm": norm, "weights": b.shape[0] * a.shape[1]}
        )

    frame = pd.DataFrame(rows)
    grouped = frame.groupby("module").agg(
        norm_final=("norm", lambda s: math.sqrt((s**2).sum())), weights=("weights", "sum")
    )
    # Root-mean-square change per weight, so module types of different sizes compare fairly.
    grouped["rms_per_weight"] = grouped["norm_final"] / grouped["weights"].pow(0.5)

    return {
        "modules": len(frame),
        "total_norm_final": math.sqrt(float((frame["norm"] ** 2).sum())),
        "by_module": grouped.reset_index().to_dict(orient="records"),
        "adapter_mb": round(weights.stat().st_size / 1e6, 1),
    }


def sentence_rows(predictions_csv: Path, systems: list[str]) -> list[dict[str, Any]]:
    """Source, reference, and each system's translation and sentence chrF++."""
    frame = pd.read_csv(predictions_csv, dtype=str, keep_default_na=False)
    rows = []

    for _, row in frame.iterrows():
        item = {"source": row["source"], "target": row["target"]}

        for system in systems:
            item[system] = row[f"prediction_{system}"]
            item[f"{system}_chrf"] = round(float(row[f"chrfpp_{system}"]), 1)

        rows.append(item)

    return rows


def limit_sentences(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """A small, representative sample: biggest gains, one typical case, and the hardest case."""
    if count >= len(rows):
        return rows

    by_gain = sorted(rows, key=lambda r: r["tuned_chrf"] - r["base_chrf"], reverse=True)
    by_score = sorted(rows, key=lambda r: r["tuned_chrf"])
    picks = [by_score[0], by_score[len(by_score) // 2]]
    chosen = [r for r in by_gain if r not in picks][: max(count - len(picks), 0)] + picks

    return chosen[:count]


def collect(run_dir: Path) -> dict[str, Any]:
    """Gather everything the dashboard shows."""
    reports = run_dir / "reports"
    evaluation = read_json(reports / "evaluation_report.json")

    if not evaluation:
        raise FileNotFoundError(
            f"{reports / 'evaluation_report.json'} is missing; run scripts.evaluate report first."
        )

    training = read_json(reports / "training_report.json")
    config = read_json(reports / "resolved_config.json")
    history = training.get("log_history", [])
    systems = [s for s in MODEL_SYSTEMS if s in evaluation["systems"]]
    adapter = adapter_statistics(run_dir / "checkpoints" / "adapter_best")

    return {
        "model": config.get("model", {}).get("repo_id"),
        "systems": evaluation["systems"],
        "sentences": evaluation["sentences"],
        "significance": evaluation.get("significance", {}),
        "significance_final": evaluation.get("significance_tuned_vs", {}),
        "train_loss": [[e["step"], e["loss"]] for e in history if "loss" in e],
        "dev_loss": [[e["step"], e["eval_loss"]] for e in history if "eval_loss" in e],
        "steps": training.get("global_step"),
        "train_examples": training.get("train_examples"),
        "best_dev": training.get("best_dev"),
        "trainable": {
            key: value
            for key, value in read_json(reports / "trainable_parameters.json").items()
            if not isinstance(value, list)
        },
        "lora": adapter,
        "adapter_mb": adapter["adapter_mb"] if adapter else None,
        "lora_config": config.get("lora"),
        "quantization": config.get("quantization"),
        "explorer": sentence_rows(run_dir / "evaluation" / "test_predictions.csv", systems),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the results dashboard HTML.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run folder (read only).")
    parser.add_argument("--output", required=True, type=Path, help="HTML file to write.")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Include at most this many test sentences (e.g. 5 for a public copy). Default: all.",
    )
    args = parser.parse_args()

    data = collect(args.run_dir)
    data["explorer_total"] = len(data["explorer"])

    if args.max_sentences is not None:
        data["explorer"] = limit_sentences(data["explorer"], args.max_sentences)

    payload = json.dumps(data, ensure_ascii=False, default=str)
    body = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload.replace("</", "<\\/"))
    # The template holds page content only; wrap it so the file is a complete standalone page.
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
