# simuleringer/2D_ny/kerne/grid.py

import numpy as np
from dataclasses import dataclass

@dataclass
class Grid2D:
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 50.0
    y_max: float = 50.0
    nx: int = 200
    ny: int = 200
    ghost: int = 1
    dtype: type = np.float32

    def __post_init__(self):
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be >= 3")

        self.x = np.linspace(self.x_min, self.x_max, self.nx, endpoint=False, dtype=self.dtype)
        self.y = np.linspace(self.y_min, self.y_max, self.ny, endpoint=False, dtype=self.dtype)

        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])

    # Allokerer én skalar funktion u(x,y)
    def alloc(self, ghost: int | None = None):
        g = self.ghost if ghost is None else int(ghost)
        return np.zeros((self.nx + 2*g, self.ny + 2*g), dtype=self.dtype)

    # Returnerer fysisk domæne (uden ghost cells)
    def interior(self, u, ghost: int | None = None):
        g = self.ghost if ghost is None else int(ghost)
        return u[g:-g, g:-g]

    # Meshgrid til evaluering af funktioner
    def XY(self):
        return np.meshgrid(self.x, self.y, indexing="ij")
