from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed common RNG sources used in this repository.

    Notes:
    - We intentionally do not set any framework-specific seeds (e.g. torch), since this repo is numpy/scipy-only.
    - Entry points should call this once near the beginning if they rely on global RNG.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
