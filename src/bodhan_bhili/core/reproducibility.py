"""Experiment reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, strict_determinism: bool = False) -> None:
    """Seed Python, NumPy, PyTorch and Transformers."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    try:
        from transformers import set_seed

        set_seed(seed)

    except ImportError:
        pass

    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
