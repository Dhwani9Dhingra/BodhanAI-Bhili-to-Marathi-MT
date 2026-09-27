"""Test-set generation, scoring, significance testing and analysis."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from bodhan_bhili.model import build_translation_messages, get_model_input_device

_SAFE_SYSTEM_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

LENGTH_BUCKETS = ((1, 5), (6, 10), (11, 20), (21, None))


def validate_system_name(name: str) -> str:
    """Keep system names usable as file names."""
    if not _SAFE_SYSTEM_NAME.fullmatch(name):
        raise ValueError("System names may contain only letters, numbers, '_' and '-'.")

    return name


def read_sha256_sidecar(path: Path) -> str:
    """Return the digest from a `HASH  filename` sidecar."""
    return path.read_text(encoding="utf-8").split()[0]


def select_test_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Deterministic test subset shared by every evaluated system."""
    size = min(size, len(frame))

    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Resumable prediction store (JSON lines; a truncated last line is ignored).
# ---------------------------------------------------------------------------


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed predictions keyed by record_id."""
    rows: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return rows

    with path.open(mode="r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                # A disconnect during a write leaves at most one partial line.
                continue

            if isinstance(row, dict) and "record_id" in row and "prediction" in row:
                rows[str(row["record_id"])] = row

    return rows


def append_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows durably so a disconnect loses at most the current batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False

    if path.exists() and path.stat().st_size > 0:
        with path.open(mode="rb") as handle:
            handle.seek(-1, os.SEEK_END)
            # A partial line from a disconnect must not swallow the next row.
            needs_newline = handle.read(1) != b"\n"

    with path.open(mode="a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")

        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        handle.flush()
        os.fsync(handle.fileno())


def check_generation_settings(meta_file: Path, current: dict[str, Any]) -> None:
    """Refuse to mix predictions produced with different settings in one file."""
    if not meta_file.exists():
        return

    saved = json.loads(meta_file.read_text(encoding="utf-8"))
    keys = ("system", "model_repo_id", "adapter_sha256", "max_new_tokens", "test_sha256")
    changed = [key for key in keys if saved.get(key) != current.get(key)]

    if changed:
        raise RuntimeError(
            f"Existing predictions for system {current['system']!r} were generated with "
            f"different settings: {changed}. Use a different --system name, or delete "
            f"{meta_file.parent} predictions for this system to regenerate."
        )


# ---------------------------------------------------------------------------
# Batched greedy generation.
# ---------------------------------------------------------------------------


def encode_prompt(processor, source: str, target_language: str) -> dict[str, Any]:
    """Tokenize one translation prompt, returning 1-D tensors."""
    inputs = processor.apply_chat_template(
        build_translation_messages(source, target_language),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return {key: value[0] for key, value in dict(inputs).items() if hasattr(value, "dim")}


def left_pad_prompts(prompts: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Left-pad prompts so generation continues from each prompt's last token."""
    import torch

    width = max(prompt["input_ids"].shape[-1] for prompt in prompts)
    batch = {}

    for key in prompts[0]:
        pad_value = pad_token_id if key == "input_ids" else 0
        rows = [
            torch.nn.functional.pad(
                prompt[key], (width - prompt[key].shape[-1], 0), value=pad_value
            )
            for prompt in prompts
        ]
        batch[key] = torch.stack(rows)

    return batch


def generate_batch(
    model, processor, prompts: list[dict[str, Any]], *, max_new_tokens: int, pad_token_id: int
) -> tuple[list[str], list[bool]]:
    """Greedy-decode a batch; also report which outputs hit the token limit."""
    import torch

    batch = left_pad_prompts(prompts, pad_token_id)
    device = get_model_input_device(model)
    batch = {key: value.to(device) for key, value in batch.items()}
    width = batch["input_ids"].shape[-1]

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    completions = generated[:, width:]
    texts = [
        processor.decode(completion, skip_special_tokens=True).strip() for completion in completions
    ]
    lengths = (completions != pad_token_id).sum(dim=-1).tolist()
    truncated = [length >= max_new_tokens for length in lengths]

    return texts, truncated


def generate_with_backoff(model, processor, prompts, *, max_new_tokens, pad_token_id, logger):
    """Halve the batch on CUDA out-of-memory instead of failing the run."""
    import torch

    try:
        return generate_batch(
            model, processor, prompts, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id
        )

    except torch.cuda.OutOfMemoryError:
        if len(prompts) == 1:
            raise

        torch.cuda.empty_cache()
        half = len(prompts) // 2
        logger.warning("CUDA OOM at batch size %d; retrying as two halves.", len(prompts))
        first = generate_with_backoff(
            model,
            processor,
            prompts[:half],
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            logger=logger,
        )
        second = generate_with_backoff(
            model,
            processor,
            prompts[half:],
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            logger=logger,
        )

        return first[0] + second[0], first[1] + second[1]


def lora_b_is_trained(model) -> bool:
    """LoRA B matrices start at zero, so any non-zero B proves trained weights loaded."""
    found = False

    for name, parameter in model.named_parameters():
        if "lora_B" not in name:
            continue

        found = True

        if parameter.detach().abs().sum().item() > 0:
            return True

    if not found:
        raise RuntimeError("No LoRA modules were attached; the adapter did not load.")

    return False


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _chrf_metric():
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2)


def corpus_metrics(
    hypotheses: Sequence[str], references: Sequence[str], sources: Sequence[str]
) -> dict[str, Any]:
    """chrF++, BLEU, and source-copying diagnostics for one system."""
    from sacrebleu.metrics import BLEU

    chrf = _chrf_metric()
    bleu = BLEU()
    hypotheses = [_normalize(text) for text in hypotheses]
    references = [_normalize(text) for text in references]
    sources = [_normalize(text) for text in sources]
    count = len(hypotheses)

    return {
        "sentences": count,
        "chrfpp": round(chrf.corpus_score(hypotheses, [references]).score, 4),
        "bleu": round(bleu.corpus_score(hypotheses, [references]).score, 4),
        "chrfpp_vs_source": round(chrf.corpus_score(hypotheses, [sources]).score, 4),
        "exact_copy_rate": round(
            sum(h == s for h, s in zip(hypotheses, sources, strict=True)) / count, 4
        ),
        "exact_match_rate": round(
            sum(h == r for h, r in zip(hypotheses, references, strict=True)) / count, 4
        ),
        "empty_rate": round(sum(not h for h in hypotheses) / count, 4),
        "chrf_signature": str(chrf.get_signature()),
        "bleu_signature": str(bleu.get_signature()),
    }


def sentence_chrf(hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
    """Per-sentence chrF++ for analysis."""
    chrf = _chrf_metric()

    return [
        round(chrf.sentence_score(_normalize(h), [_normalize(r)]).score, 4)
        for h, r in zip(hypotheses, references, strict=True)
    ]


def paired_bootstrap_chrf(
    baseline: Sequence[str],
    candidate: Sequence[str],
    references: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap resampling of the corpus chrF++ difference (candidate - baseline)."""
    import numpy as np

    chrf = _chrf_metric()
    references = [_normalize(text) for text in references]
    stats = {}

    for name, hypotheses in (("baseline", baseline), ("candidate", candidate)):
        hypotheses = [_normalize(text) for text in hypotheses]
        stats[name] = np.array(
            chrf._extract_corpus_statistics(hypotheses, [references]), dtype=np.float64
        )

    def score(matrix, indices):
        return chrf._compute_score_from_stats(matrix[indices].sum(axis=0).tolist()).score

    count = len(references)
    everything = np.arange(count)
    observed = score(stats["candidate"], everything) - score(stats["baseline"], everything)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples)

    for index in range(samples):
        draw = rng.integers(0, count, size=count)
        deltas[index] = score(stats["candidate"], draw) - score(stats["baseline"], draw)

    return {
        "metric": "chrfpp",
        "samples": samples,
        "observed_delta": round(float(observed), 4),
        "ci95_low": round(float(np.percentile(deltas, 2.5)), 4),
        "ci95_high": round(float(np.percentile(deltas, 97.5)), 4),
        # One-sided: how often the candidate fails to beat the baseline under resampling.
        "p_value": round(float((np.sum(deltas <= 0) + 1) / (samples + 1)), 4),
    }


def length_bucket(word_count: int) -> str:
    """Label a source length (in words) with its analysis bucket."""
    for low, high in LENGTH_BUCKETS:
        if word_count >= low and (high is None or word_count <= high):
            return f"{low}+" if high is None else f"{low}-{high}"

    return "0"
