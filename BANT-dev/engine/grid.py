from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .config import EngineConfig



@dataclass
class Grid2D:
    cfg: ClassVar[EngineConfig] = EngineConfig()

    def __post_init__(self) -> None:
        self.x = np.linspace(self.cfg.x_min, self.cfg.x_max, self.cfg.nx, endpoint=False, dtype=self.cfg.dtype)
        self.y = np.linspace(self.cfg.y_min, self.cfg.y_max, self.cfg.ny, endpoint=False, dtype=self.cfg.dtype)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])

    def alloc(self) -> np.ndarray:
        if self.cfg.n_fields == 1:
            return np.zeros((self.cfg.nx + 2, self.cfg.ny + 2), dtype=self.cfg.dtype)
        return np.zeros((self.cfg.n_fields, self.cfg.nx + 2, self.cfg.ny + 2), dtype=self.cfg.dtype)

    def interior(self, u: np.ndarray) -> np.ndarray:
        if u.ndim == 2:
            return u[1:-1, 1:-1]
        return u[:, 1:-1, 1:-1]

    def XY(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="ij")
