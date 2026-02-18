from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable

from setup import Grid1D, ddx_central 


@dataclass
class Transport1D:
    """
    Transport-ligning:
        d_t n + nabla_x(n V) = 0
    dvs.
        d_t n = nabla_x(n V)
    """
    Vfun: Callable[[np.ndarray, float], np.ndarray]
    x0: float = 0.30
    sigma: float = 0.05
    name: str = "Transport 1D (general V)"

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-((x - self.x0) ** 2) / (2.0 * self.sigma ** 2))

    def rhs(self, n: np.ndarray, t: float, grid: Grid1D) -> np.ndarray:
        V = self.Vfun(grid.x, t)
        flux = n * V
        return -ddx_central(flux, grid.dx)
