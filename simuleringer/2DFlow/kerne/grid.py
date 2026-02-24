# Sætter grid op for simuleringen

import numpy as np

from dataclasses import dataclass


@dataclass
class Grid:
    config : object

    def __init__(self, config: object):
        self.config = config

        self.x = np.linspace(self.config.x_min, self.config.x_max, self.config.nx, endpoint=False, dtype=self.config.dtype)
        self.y = np.linspace(self.config.y_min, self.config.y_max, self.config.ny, endpoint=False, dtype=self.config.dtype)

        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
    
    # Allokerer én skalar funktion u(x,y) med ghost cells
    def alloc(self) -> np.ndarray:
        return np.zeros((self.config.nx + 2, self.config.ny + 2), dtype=self.config.dtype)

    # Returnerer fysisk domæne (uden ghost cells)
    def interior(self, u: np.ndarray) -> np.ndarray:
        return u[1:-1, 1:-1]

    # Meshgrid til evaluering af funktioner
    def XY(self)-> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="ij")