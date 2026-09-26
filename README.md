# bodhan-bhili-mt
Fine tuning Bodhan AI models

# Bodhan Indic-Translate — Dehwali Bhili → Marathi

This repository contains an end-to-end parameter-efficient fine-tuning
pipeline for adapting `bodhan-ai/indic-translate` to translate
Dehwali Bhili into Marathi.

## Research question

Can 4-bit QLoRA teach Bodhan Indic-Translate to understand an
unsupported low-resource source language (Dehwali Bhili) and translate
it into Marathi, a language the base model already supports?

## Scope

Primary experiment:

    Dehwali Bhili → Marathi

Marathi → Bhili is intentionally excluded from the primary run and is
reserved for future/stretch work.

## Fine-tuning method

    Bodhan Indic-Translate
            +
    4-bit NF4 quantization
            +
         LoRA
            =
         QLoRA

## Current implementation status

Foundation / preflight completed:

- validated experiment configuration
- deterministic artifact layout
- persistent logging
- reproducibility utilities
- environment and Git metadata
- persistent run-state tracking
- experiment manifest
- CUDA / VRAM validation
- bitsandbytes NF4 GPU test
- Hugging Face gated-model access check
- Gemma 4 architecture check
- chrF++ sanity check

The next stage is data preparation using the Project Astitva
Dehwali Bhili–Marathi TSV.

## Planned pipeline

    preflight
       ↓
    data preparation
       ↓
    baselines
       ↓
    QLoRA fine-tuning
       ↓
    adapter reload verification
       ↓
    frozen-test evaluation
       ↓
    analysis
       ↓
    TensorBoard + Streamlit + documentation