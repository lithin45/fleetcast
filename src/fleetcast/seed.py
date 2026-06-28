"""Determinism helpers.

A single ``seed_everything`` call seeds Python's ``random``, NumPy, and the
``PYTHONHASHSEED`` env var. LightGBM/statsmodels determinism is additionally
enforced where the models are constructed (fixed ``seed``/``random_state`` and
single-threaded-deterministic options) — see those modules.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed all sources of nondeterminism we control and return the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
