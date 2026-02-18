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

    def __post_init__(self):
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be >= 3")

        self.x = np.linspace(self.x_min, self.x_max, self.nx, endpoint=False)
        self.y = np.linspace(self.y_min, self.y_max, self.ny, endpoint=False)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

class HelperFunctions:
    @staticmethod
    def dx_central(u: np.ndarray, dx: float) -> np.ndarray:
        return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0*dx)

    @staticmethod
    def dy_central(u: np.ndarray, dy: float) -> np.ndarray:
        return (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0*dy)

    @staticmethod
    def d2x_central(u: np.ndarray, dx: float) -> np.ndarray:
        return (np.roll(u, -1, axis=0) - 2.0*u + np.roll(u, 1, axis=0)) / (dx*dx)

    @staticmethod
    def d2y_central(u: np.ndarray, dy: float) -> np.ndarray:
        return (np.roll(u, -1, axis=1) - 2.0*u + np.roll(u, 1, axis=1)) / (dy*dy)

    @staticmethod
    def dxdy_central(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
        u_pp = np.roll(np.roll(u, -1, axis=0), -1, axis=1)  # i+1, j+1
        u_pm = np.roll(np.roll(u, -1, axis=0),  1, axis=1)  # i+1, j-1
        u_mp = np.roll(np.roll(u,  1, axis=0), -1, axis=1)  # i-1, j+1
        u_mm = np.roll(np.roll(u,  1, axis=0),  1, axis=1)  # i-1, j-1

        return (u_pp - u_pm - u_mp + u_mm) / (4.0*dx*dy)
