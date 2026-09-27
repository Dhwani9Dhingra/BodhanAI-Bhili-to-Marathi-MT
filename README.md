# Fine-Tuning Bodhan Indic-Translate for Dehwali Bhili → Marathi

Fine-tuning the Bodhan AI machine-translation model [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate) to translate **Dehwali Bhili into Marathi** with **QLoRA**.

**Results dashboard:** [open the public dashboard](https://htmlpreview.github.io/?https://github.com/Dhwani9Dhingra/BodhanAI-Bhili-to-Marathi-MT/blob/main/Dashboard_public.html) (metrics, training curves, adapter statistics and 5 example sentences). The file is [`Dashboard_public.html`](Dashboard_public.html).

---

## 1. Project overview

Dehwali Bhili is written in the same Devanagari script as Marathi but is a different language, and it has very little parallel data. Out of the box, Bodhan Indic-Translate mostly returns Bhili input unchanged instead of translating it. This project adapts the model to the Bhili → Marathi direction.

| | |
| --- | --- |
| Base model | `bodhan-ai/indic-translate` (5.74B parameters, image-text-to-text architecture) |
| Method | QLoRA: frozen base loaded in 4-bit NF4 with double quantization; LoRA adapters (rank 8, alpha 16, dropout 0.05) trained in fp32 on the language layers only |
| Trainable parameters | 19.4M (0.34%), in 343 adapted modules; the saved adapter is 77.9 MB |
| Data | Dehwali Bhili ↔ Marathi translation corpus (AIKosh, Project Astitva): 14,737 raw pairs → 13,995 after cleaning |
| Split | 11,195 train / 1,400 dev / 1,400 test, grouped by source ID so no group appears in two splits |
| Training | 1,400 optimizer steps (one pass over the training data), effective batch 8, learning rate 2e-4 |
| Hardware | One Tesla T4 (free Colab), about 6.7 hours of training across several resumed sessions |
| Evaluation | 300 held-out test sentences, chrF++ (primary), BLEU, copy rate, paired bootstrap confidence intervals |

The pipeline is built as separate, resumable stages (preflight, data preparation, training, evaluation, reporting) that write all outputs and their status to one run folder on Google Drive, so a Colab disconnect never loses more than the last few minutes of work.

## 2. Results and insights

All systems translate the same 300 held-out test sentences. `Copy input` is a reference point: the Bhili sentence returned unchanged.

| System | chrF++ | BLEU | Output identical to input | Δ chrF++ vs base (95% CI) |
| --- | ---: | ---: | ---: | --- |
| Copy input (reference point) | 35.00 | 8.72 | 100% | — |
| Base model (before fine-tuning) | 37.22 | 11.71 | 59.0% | — |
| Fine-tuned, 700 steps (50% of data) | 58.86 | 33.96 | 0% | +21.63 (+20.23 to +23.22) |
| **Fine-tuned, 1,400 steps (100% of data)** | **60.08** | **35.25** | **0%** | **+22.86 (+21.33 to +24.56)** |

Confidence intervals come from 1,000 paired bootstrap resamples (p = 0.001 for both fine-tuned systems vs base).

**Insights**

- **The base model barely translated Bhili.** It returned the input unchanged for 177 of 300 sentences, scoring only 2.2 chrF++ above plain copying. Because the two languages share script and many words, copying alone already earns 35 chrF++, which is why the copy baseline is reported.
- **Fine-tuning fixed the main failure.** The fine-tuned model copies no inputs and scores +22.9 chrF++ and +23.5 BLEU over the base model. Sentences scoring below 30 chrF++ fell from 121 to 29; sentences scoring 80 or above rose from 5 to 62.
- **Most of the gain came from the first half of the data.** 700 steps gave +21.6 chrF++; the second 700 steps added +1.23 more (95% CI +0.40 to +2.09, p = 0.002): small but statistically reliable. Eval loss fell from 1.063 (step 50) to 0.800 (step 700) to 0.760 (step 1,400) with no sign of overfitting.
- **Many remaining low scores reflect the references.** Where the human translation is much longer than the Bhili sentence (51 test sentences), the final model averages 43.0 chrF++ against 64.2 where lengths match: those references are often paraphrases that add content.
- **The adapter changed the feed-forward layers most in total**, but per weight the largest change is in the model's per-layer projection modules; attention modules changed least.

## 3. Workflow

```mermaid
flowchart TD
    A[Raw AIKosh TSV<br/>14,737 Bhili-Marathi pairs] --> B[Preflight<br/>Python, CUDA, NF4, disk, HF token, model access]
    B --> C[Data preparation<br/>clean, deduplicate, grouped split,<br/>leakage checks, freeze test hash]
    C --> D[Smoke tests<br/>10-step QLoRA run, interrupt + resume check]
    D --> E[QLoRA training on T4<br/>4-bit NF4 base + LoRA adapters<br/>checkpoint to Drive every 25 steps]
    E -->|Colab disconnect| E
    E --> F[Adapters<br/>700 steps and 1,400 steps]
    F --> G[Evaluation<br/>base, 700-step and 1,400-step models<br/>translate 300 test sentences]
    G --> H[Report<br/>chrF++, BLEU, copy rate,<br/>paired bootstrap significance]
    H --> I[Package + dashboard<br/>adapter, model card, Dashboard.html]
```

Every stage records its status in `reports/run_state.json`. Training and evaluation both resume from where they stopped: training from the newest complete checkpoint, evaluation from the last saved batch of translations.

## 4. Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ (Colab ran 3.13) |
| Model and training | PyTorch, Hugging Face Transformers (Trainer), PEFT (LoRA), bitsandbytes (4-bit NF4, paged 8-bit AdamW), Accelerate |
| Data | pandas, NumPy, Hugging Face Hub |
| Evaluation | SacreBLEU (chrF++, BLEU), paired bootstrap resampling |
| Configuration | YAML configs validated with Pydantic |
| Tracking | JSON reports, run-state file, TensorBoard, pipeline log |
| Compute and storage | Google Colab (Tesla T4), Google Drive for all artifacts |
| Dashboard | Self-contained HTML generated by `scripts/dashboard.py` |
| Quality | pytest (CPU unit tests), Ruff (lint and format) |

## 5. Project structure

### Repository (GitHub)

```text
BodhanAI-Bhili-to-Marathi-MT/
├── configs/
│   ├── colab_t4.yaml            Main experiment: data, model, QLoRA, training and evaluation settings
│   └── smoke.yaml               10-step test run used to check the pipeline before the full run
├── notebook/
│   └── Fine_Tuning.ipynb        The Colab workflow, cell by cell (setup → data → train → evaluate)
├── requirements/
│   ├── colab.txt                Colab install: training + development packages
│   └── dev.txt                  Laptop install: development packages only (no GPU libraries)
├── scripts/                     One command per pipeline stage (run as python -m scripts.<name>)
│   ├── preflight.py             Checks Python, CUDA, NF4, disk, Hugging Face token and model access
│   ├── prepare_data.py          Cleans the raw TSV, splits by group, freezes the test-set hash
│   ├── smoke_model.py           Short end-to-end QLoRA check: load, train a few steps, save, reload
│   ├── train.py                 Resumable QLoRA training (--extend to train a finished run longer)
│   ├── evaluate.py              Translates the test sample per system, then scores and packages
│   ├── dashboard.py             Builds the HTML results dashboard from a run folder
│   └── dashboard_template.html  Layout of the dashboard
├── src/bodhan_bhili/            The library the scripts use
│   ├── core/
│   │   ├── config.py            YAML loading and validation (Pydantic), environment overrides
│   │   ├── paths.py             Standard layout of a run folder
│   │   ├── run_state.py         Records each stage as pending / running / completed / failed
│   │   ├── preflight.py         The individual environment checks
│   │   ├── manifest.py          Experiment manifest (what was run, with which settings)
│   │   ├── environment.py       Package and hardware versions for reproducibility
│   │   ├── reproducibility.py   Seeding
│   │   ├── serialization.py     Atomic JSON writes (a crash cannot leave a half-written report)
│   │   ├── logging.py           Shared logger (console + logs/pipeline.log)
│   │   └── constants.py         Stage names and defaults
│   ├── data.py                  Cleaning rules, grouped splits, leakage checks, hashes
│   ├── model.py                 Model loading in 4-bit, prompt format, completion-only labels, LoRA setup
│   ├── training.py              Checkpoint discovery, resume and --extend guards, batching, callbacks
│   └── evaluation.py            Resumable batched generation, chrF++ / BLEU, bootstrap significance
├── tests/unit/                  CPU unit tests (no GPU, Drive or model download needed)
├── Dashboard_public.html        Public results dashboard (5 example sentences)
├── pyproject.toml               Package definition, dependencies, pytest and Ruff settings
├── .env.example                 Environment variables the scripts read (no real values)
└── .gitignore                   Keeps data, checkpoints, tokens and caches out of Git
```

### Run folder (Google Drive artifacts)

Every stage writes into one folder per run: `MyDrive/BodhanAI/artifacts/bodhan-bhili-mt/<run_id>/` (this project: `bhili_marathi_t4_v1`).

```text
<run_id>/
├── data/            train.tsv, dev.tsv, test.tsv (cleaned and split)
├── reports/         data_report.json, training_report.json, evaluation_report.json,
│                    run_state.json, preflight.json, resolved_config.json, test_set.sha256
├── logs/            pipeline.log (every session, resume and failure)
├── evaluation/      predictions per system (*.jsonl), test_predictions.csv with per-sentence scores
├── checkpoints/
│   ├── trainer/            Resumable checkpoints (adapter, optimizer, scheduler, RNG state)
│   ├── adapter_best/       Final adapter: lowest dev loss (step 1,400)
│   ├── adapter_final/      Adapter at the end of training (same step as adapter_best here)
│   └── adapter_step_701/   Mid-training adapter (700 steps), kept for comparison
├── package/         Final adapter + processor + evaluation report + model card
└── tensorboard/     Training curves (tensorboard --logdir <run_id>/tensorboard)
```

Data, checkpoints, adapters, tokens and model caches are never committed to Git; they live only in the Drive run folder.

## 6. How to run

Training and evaluation need a CUDA GPU (the base model is loaded in 4-bit with bitsandbytes), so they run on Colab. A laptop is enough for development, unit tests, data preparation and building the dashboard.

### Before you start

1. Accept the model's terms on its [Hugging Face page](https://huggingface.co/bodhan-ai/indic-translate) and create a read token.
2. Download the Dehwali Bhili ↔ Marathi TSV from AIKosh and upload it to your Google Drive, for example `MyDrive/BodhanAI/data/Dehwali_Bhili_Translation_15k_pipeline.tsv`.

### On Google Colab (full pipeline)

The notebook [`notebook/Fine_Tuning.ipynb`](notebook/Fine_Tuning.ipynb) contains every step in order. In short:

1. Runtime → Change runtime type → **T4 GPU**. Add your token to Colab Secrets (key icon) as `HF_TOKEN`.
2. Mount Drive, clone, install:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   %cd /content
   !git clone https://github.com/Dhwani9Dhingra/BodhanAI-Bhili-to-Marathi-MT.git
   %cd /content/BodhanAI-Bhili-to-Marathi-MT
   !pip install -r requirements/colab.txt
   ```
3. Set the token:
   ```python
   import os
   from google.colab import userdata
   os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
4. Check the environment and prepare the data:
   ```python
   !python -m scripts.preflight --config configs/colab_t4.yaml
   !python -m scripts.prepare_data --config configs/colab_t4.yaml \
       --data "/content/drive/MyDrive/BodhanAI/data/Dehwali_Bhili_Translation_15k_pipeline.tsv"
   ```
5. Optional smoke test of training, including an interrupt and resume: run the same two commands with `configs/smoke.yaml`, then `!python -m scripts.train --config configs/smoke.yaml`.
6. Train (resumable; re-run the same command after a disconnect):
   ```python
   !python -m scripts.train --config configs/colab_t4.yaml
   ```
7. Evaluate and report:
   ```python
   !python -m scripts.evaluate generate --config configs/colab_t4.yaml --system base
   !python -m scripts.evaluate generate --config configs/colab_t4.yaml --system tuned
   !python -m scripts.evaluate report --config configs/colab_t4.yaml
   ```
8. Flush Drive before closing the tab, so unsynced checkpoints are not lost:
   ```python
   drive.flush_and_unmount()
   ```

Results are written to `MyDrive/BodhanAI/artifacts/bodhan-bhili-mt/bhili_marathi_t4_v1/`.

### On a local machine (development, tests, dashboard)

```powershell
git clone https://github.com/Dhwani9Dhingra/BodhanAI-Bhili-to-Marathi-MT.git
cd BodhanAI-Bhili-to-Marathi-MT
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src scripts tests
```

On macOS or Linux use `python3 -m venv .venv` and `.venv/bin/python`.

Prepare data locally (point the artifact root at a Google Drive synced folder so Colab can use it):

```powershell
$env:BODHAN_ARTIFACT_ROOT = "G:/My Drive/BodhanAI/artifacts"
.\.venv\Scripts\python.exe -m scripts.prepare_data --config configs/colab_t4.yaml --data "C:/path/to/Dehwali_Bhili_Translation_15k_pipeline.tsv"
```

Build the dashboard from a downloaded run folder (`safetensors` is needed for the adapter statistics):

```powershell
.\.venv\Scripts\python.exe -m pip install safetensors numpy
.\.venv\Scripts\python.exe -m scripts.dashboard --run-dir "<path to>/bhili_marathi_t4_v1" --output Dashboard.html
# public copy with only 5 example sentences
.\.venv\Scripts\python.exe -m scripts.dashboard --run-dir "<path to>/bhili_marathi_t4_v1" --output Dashboard_public.html --max-sentences 5
```

---

## Reference

### Resuming and extending training

Free Colab sessions can disconnect at any time, so training is built to pick up where it stopped.

**What is saved.** Every `save_steps` steps (25 in this project), `scripts.train` writes a checkpoint to `checkpoints/trainer/checkpoint-<step>/` on Drive. Each checkpoint holds everything needed to continue exactly:

| Saved item | Why it matters |
| --- | --- |
| LoRA adapter weights | The training progress itself |
| Optimizer state | Continues with the same momentum instead of restarting it |
| Learning-rate scheduler | Keeps the learning-rate schedule on track |
| Random-number state and step count | Continues with the same data order, so no example is skipped or repeated |

Only the newest `save_total_limit` checkpoints (2 here) are kept, to save Drive space.

**If the session disconnects.** Reconnect, run the setup cells, and run the same command again:

```python
!python -m scripts.train --config configs/colab_t4.yaml
```

- It resumes from the **newest complete** checkpoint.
- A checkpoint that is only partly written (the save was interrupted, or Drive had not finished uploading) is skipped and moved to `checkpoints/trainer_incomplete/`.
- If the prepared data or training settings have changed since training started, it **refuses to resume**, so two different experiments are never mixed.
- Running it again after training has finished does nothing.

**To train a finished run longer.** Raise only `training.max_steps` in the config, then add `--extend`:

```python
!python -m scripts.train --config configs/colab_t4.yaml --extend
```

- Before continuing, the current adapter is copied to `checkpoints/adapter_step_<step>/`, so it is kept even after old checkpoints are deleted.
- Training continues on examples not yet seen, with the learning rate following the longer schedule.

**In this project:** training first ran for 700 steps (half the data), and that adapter was saved as `adapter_step_701`. It was then extended to 1,400 steps (all the data) with `--extend`, which is why the results compare a mid-training and a final model.


### Environment variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Hugging Face read token (use Colab Secrets; never commit it) |
| `BODHAN_ARTIFACT_ROOT` | Drive folder that holds all runs |
| `BODHAN_DATA_FILE` | Raw TSV location (alternative to `--data`) |
| `BODHAN_HF_CACHE` | Model download cache |
| `BODHAN_RUN_ID` | Overrides the run ID in the config |

`.env.example` lists them; scripts read the environment and do not load `.env` files.


