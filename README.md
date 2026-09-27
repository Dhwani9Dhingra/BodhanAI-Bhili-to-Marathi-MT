# Bodhan Indic-Translate: Dehwali Bhili to Marathi

Adapt `bodhan-ai/indic-translate` with 4-bit QLoRA for Dehwali Bhili to Marathi translation. The laptop handles development and data preparation; Google Colab handles GPU work. Google Drive stores experiment artifacts for later analysis and a future dashboard.

## Current status

Implemented:

- Configuration validation, environment checks, logging, and run metadata.
- Quote-aware TSV loading, text cleaning, grouped splits, leakage checks, and dataset hashes.
- Translation prompts, completion-only labels, LoRA module selection, and a GPU smoke test that saves and reloads an adapter.
- CPU unit tests for configuration, paths, data preparation, prompts, masking, and run state.

- Resumable QLoRA training (`scripts.train`) with Drive checkpoints and best-dev-loss adapter export.

- Resumable test evaluation (`scripts.evaluate`): base and tuned translations, chrF++/BLEU/copy metrics, paired bootstrap significance, length analysis, and Drive packaging.

Retention evaluation (other translation directions) and the dashboard are still pending.

## Repository layout

```text
configs/                 Smoke and T4 experiment settings
requirements/            Laptop and Colab installation entry points
scripts/                 One command per implemented stage
src/bodhan_bhili/
    core/                Configuration, paths, logging, state, and preflight
    data.py              CPU data preparation
    model.py             Model loading, translation, and QLoRA utilities
    training.py          Checkpoint discovery, resume guard, batching, callbacks
tests/unit/              CPU unit tests
```

Dependencies are declared in `pyproject.toml`. The requirement files select the appropriate extras. Raw data, credentials, model caches, and generated artifacts are excluded from Git.

## Laptop setup

Use Python 3.10 or newer. Run commands from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src scripts tests
.\.venv\Scripts\ruff.exe format --check src scripts tests
```

For data preparation, set the artifact root to your actual Google Drive synced folder. The drive letter below is an example; replace it with your own location.

```powershell
$env:BODHAN_ARTIFACT_ROOT = 'G:/My Drive/BodhanAI/artifacts'
$env:BODHAN_DATA_FILE = 'C:/path/to/Dehwali_Bhili_Translation_15k_pipeline.tsv'
$env:BODHAN_RUN_ID = 'smoke_001'
.\.venv\Scripts\python.exe -m scripts.prepare_data --config configs/smoke.yaml
```

Wait for Drive to finish syncing before opening the same run in Colab. CPU tests use temporary directories and do not need Drive or a GPU.

## Colab setup

Select a GPU runtime. Clone or upload the repository to Colab's local disk, then change to its root with `%cd`. Accept the model's access terms on Hugging Face and add a read token named `HF_TOKEN` to Colab Secrets.

Run this notebook cell before any pipeline command:

```python
import os
from google.colab import drive, userdata

drive.mount('/content/drive')
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['BODHAN_ARTIFACT_ROOT'] = '/content/drive/MyDrive/BodhanAI/artifacts'
os.environ['BODHAN_HF_CACHE'] = '/content/hf_cache'
os.environ['BODHAN_RUN_ID'] = 'smoke_001'
```

Install the Colab dependencies from the repository root:

```python
%pip install -r requirements/colab.txt
```

The default artifact path in both configs is on Google Drive. Scripts refuse to create that path if Drive is not mounted. Model downloads remain on Colab's temporary disk; only adapters and experiment outputs belong in the artifact directory.

Environment overrides:

| Variable | Purpose |
| --- | --- |
| `BODHAN_ARTIFACT_ROOT` | Drive directory containing all experiment runs |
| `BODHAN_DATA_FILE` | Raw AIKosh TSV location |
| `BODHAN_HF_CACHE` | Temporary model download cache |
| `BODHAN_RUN_ID` | Run folder shared by the stages of one experiment |
| `HF_TOKEN` | Hugging Face access token |

`.env.example` lists these variables. Scripts do not automatically load `.env` files.

## Stage commands

Run preflight in Colab:

```python
!python -m scripts.preflight --config configs/smoke.yaml
```

If you already prepared `smoke_001` on the laptop and synced it to Drive, use those files. Otherwise, upload the raw TSV to Drive, set its path, and prepare it in Colab:

```python
os.environ['BODHAN_DATA_FILE'] = '/content/drive/MyDrive/BodhanAI/data/raw/Dehwali_Bhili_Translation_15k_pipeline.tsv'
!python -m scripts.prepare_data --config configs/smoke.yaml
```

Run the model smoke test:

```python
!python -m scripts.smoke_model --config configs/smoke.yaml --steps 3
```

Use the same config and run ID for stages that share prepared data. Rerunning a stage can replace its outputs, so choose a new run ID for a separate experiment. The smoke script defaults to at most three optimizer steps unless `--steps` is supplied.

Run full training with the same config used for `prepare_data`:

```python
!python -m scripts.train --config configs/colab_t4.yaml
```

Training saves a checkpoint to `checkpoints/trainer/` on Drive every `save_steps` optimizer steps, keeping the newest `save_total_limit`. Each checkpoint holds the LoRA adapter, optimizer, scheduler, RNG state, and step count. If the runtime disconnects, mount Drive, reinstall, and rerun the same command: it resumes from the newest complete checkpoint. Checkpoints missing files (an interrupted save or unfinished Drive sync) are moved to `checkpoints/trainer_incomplete/`. Resuming is refused if the prepared data or training hyperparameters changed since the first checkpoint; save, evaluation, and logging frequencies may change freely. Dev loss is computed on `eval_examples` dev rows every `evaluation_steps`, and the lowest-loss adapter is exported to `adapter_best/`.

Rerunning a finished run does nothing. To train a finished run longer, raise only `training.max_steps` and pass `--extend`: training continues from the newest checkpoint with its optimizer state and data order (so new steps see examples not yet trained on), and the learning rate follows the longer schedule. The adapter being extended is first copied to `checkpoints/adapter_step_<step>/` so checkpoint rotation cannot delete it. The first extended step runs at the old schedule's final learning rate (zero), a Trainer resume quirk.

## Evaluation

Each system translates the same `evaluation.test_generation_samples` sentences drawn (with the project seed) from the frozen test split, whose hash is checked first. Predictions are appended to `evaluation/predictions_<system>.jsonl` batch by batch, so rerunning a command after a disconnect continues where it stopped.

```python
!python -m scripts.evaluate generate --config configs/colab_t4.yaml --system base
!python -m scripts.evaluate generate --config configs/colab_t4.yaml --system tuned
!python -m scripts.evaluate report --config configs/colab_t4.yaml
```

`tuned` uses `checkpoints/adapter_best` by default; any other adapter can be scored under its own name, for example `--system tuned_700 --adapter checkpoints/adapter_step_701`. `--limit 5` translates only the first five sampled sentences as a smoke test; a later full run reuses them. Generation is greedy with `--max-new-tokens 256` and `--batch-size 8` by default (a CUDA out-of-memory error halves the batch). A system's settings are fixed once its predictions exist.

`report` scores every system plus a `copy_source` baseline (the Bhili input unchanged, since both languages use Devanagari), runs a paired bootstrap on chrF++ for each system against `base`, and writes `reports/evaluation_report.json`, `evaluation/test_predictions.csv`, `most_improved.csv`, and `most_regressed.csv`. When `tuned` is complete it also writes `package/` with the adapter, processor, report, and a model card. Prediction files are plain JSON lines, so a system generated in another Colab account can be copied into this run's `evaluation/` folder together with its `generation_<system>.json`.

## Drive artifact layout

```text
BodhanAI/artifacts/bodhan-bhili-mt/<run_id>/
    data/                train.tsv, dev.tsv, test.tsv
    reports/             Config, manifest, state, data audit, hashes, smoke and training reports
    logs/                pipeline.log
    evaluation/          Smoke and test predictions, per-sentence scores
    package/             Final adapter, processor, evaluation report, model card
    checkpoints/
        adapter_initial/ Adapter before smoke updates
        adapter_final/   Final adapter and processor (smoke or training)
        adapter_best/    Lowest dev-loss training adapter and best_metric.json
        trainer/         Resumable checkpoint-<step>/ folders
        trainer_incomplete/  Unusable checkpoints moved aside on resume
    tensorboard/         Training event logs
```

A future dashboard can read the JSON reports and CSV predictions directly from this run directory. Reports include split counts and hashes, cleaning reasons, run metadata, trainable parameter details, and smoke losses. Predictions retain record IDs, source text, references, and before/after outputs. Some folders remain empty until their stage is implemented.

Existing local artifacts are not migrated automatically. Copy any run you want to retain into the same project/run layout on Drive before using it there. The cleanup preserves existing local files.

## Decisions still to resolve

- The saved data report records 11,195 train, 1,400 dev, and 1,400 test rows; the initial plan describes different counts and evaluation caps.
- The smoke test currently samples the test split. Reserving that split for final evaluation requires a separate change.
- Package versions are constrained by ranges, not pinned to a verified Colab environment.

These experiment settings and behaviors were preserved during repository cleanup.

## QLoRA memory fix

QLoRA preparation now freezes the base weights at their loaded precision instead of using
`prepare_model_for_kbit_training`, whose blanket fp32 conversion caused the T4 setup OOM.
Only LoRA parameters are trainable and converted to fp32. Non-reentrant gradient checkpointing
is enabled when configured, and each optimizer step checks for finite, nonzero adapter gradients.
Rank, target modules, optimizer, and dataset settings are unchanged.

The smoke test logs allocated/reserved GPU memory and process-wide peaks around model loading,
adapter setup, training, and reload. These snapshots are also saved in `model_smoke_report.json`,
including on failure. This removes the known upcast allocation; training still needs verification
on Colab. Restart a runtime that has encountered an OOM before retrying with the updated code.
