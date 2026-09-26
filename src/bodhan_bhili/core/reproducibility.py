"""
Experiment reproducibility utilities.

Perfect bit-for-bit GPU reproducibility is not always possible across
different CUDA/GPU/library versions.

What we CAN do is:
    - seed all major RNGs;
    - use deterministic evaluation;
    - record every environment version;
    - freeze dataset splits;
    - freeze the test-set hash.

That is scientifically more useful than pretending every GPU kernel is
perfectly deterministic.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(
    seed: int,
    strict_determinism: bool = False,
) -> None:
    """Seed Python, NumPy, PyTorch and Transformers."""

    # Python hash seed helps reduce process-level randomness.
    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    # Standard Python random module.
    random.seed(
        seed
    )

    # NumPy random generator.
    np.random.seed(
        seed
    )

    # Import torch here rather than at module-import time.
    import torch

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    # Hugging Face uses this helper for additional internals.
    try:
        from transformers import set_seed

        set_seed(
            seed
        )

    except ImportError:
        # CPU-only development environments may not yet have
        # Transformers installed.
        pass

    # --------------------------------------------------------
    # OPTIONAL STRICT MODE
    # --------------------------------------------------------
    #
    # We do NOT enable this by default because some GPU operations
    # used by large models may not have deterministic implementations.
    #
    # warn_only=True prevents an unsupported deterministic operation
    # from killing an otherwise valid experiment.

    if strict_determinism:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        if torch.backends.cudnn.is_available():

            torch.backends.cudnn.benchmark = False

            torch.backends.cudnn.deterministic = True