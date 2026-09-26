"""
Bodhan Bhili MT
===============

A reproducible research pipeline for parameter-efficient fine-tuning
of Bodhan Indic-Translate on Dehwali Bhili -> Marathi translation.

The package is intentionally organised into small modules:

    core        Experiment infrastructure.
    data        Dataset preparation.       (Part 2)
    model       Bodhan + QLoRA.             (Part 3)
    training    Fine-tuning engine.         (Part 4)
    evaluation  MT metrics and baselines.   (Part 5)
    analysis    Research analysis.          (Part 5)
    pipeline    End-to-end orchestration.   (Parts 4-6)

Keeping these responsibilities separate makes the experiment easier
to debug, reproduce and explain in the assignment documentation.
"""

__version__ = "0.1.0"