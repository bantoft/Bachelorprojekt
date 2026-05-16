from __future__ import annotations

from dataclasses import dataclass
from time import time

import numpy as np


@dataclass
class EngineConfig:
    # Opløsilighed og domæne.
    nx: int = 32
    ny: int = 32
    tn: float = 10.0
    t0: float = 0.0
    dt: float = 0.001
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 32.0
    y_max: float = 32.0
    dtype: type = np.float32
    n_fields: int = 1

    # Data bevaring
    output: str = f"data/sim_data{time():.3f}.npz"
    save_every: int = 5
    save_dt: float = 0.01
    compression_level: int = 0

    # Gradient specifikationer for adaptiv tidsstyring
    gradient_threshold: float = 5
    sensitivity: float = 2
    min_dt: float = 1e-5
    max_dt: float = 1e-1
    eps: float = 1e-6

