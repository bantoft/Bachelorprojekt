import numpy as np
from dataclasses import dataclass
from numba import njit

@dataclass
class Grid2D:
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 50.0
    y_max: float = 50.0
    nx: int = 200
    ny: int = 200

    def __post_init__(self):
        if self.nx < 3 or self.ny < 3: raise ValueError("nx and ny must be >= 3")

        self.x = np.linspace(self.x_min, self.x_max, self.nx, endpoint=False)
        self.y = np.linspace(self.y_min, self.y_max, self.ny, endpoint=False)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
    
    def alloc_field(self, ghost: int = 1, dtype=np.float32):
        """Alloker felt med ghost cells: (nx+2g, ny+2g)."""
        return np.zeros((self.nx + 2*ghost, self.ny + 2*ghost), dtype=dtype)

    def alloc_phys(self, dtype=np.float32):
        """Alloker felt uden ghost cells: (nx, ny)."""
        return np.zeros((self.nx, self.ny), dtype=dtype)

    def interior(self, u, ghost: int = 1):
        """View til det fysiske domæne."""
        return u[ghost:-ghost, ghost:-ghost]
    
    def XY(self): 
        """Lav meshgriddet"""
        return np.meshgrid(self.x, self.y, indexing="ij")


class Dynamics:
    
    @staticmethod
    @njit(fastmath=True)
    def apply_periodic_bc(u):
        # u: (nx+2, ny+2)
        nx, ny = u.shape

        # x-retning
        for j in range(1, ny-1):
            u[0, j]    = u[nx-2, j]
            u[nx-1, j] = u[1, j]

        # y-retning
        for i in range(1, nx-1):
            u[i, 0]    = u[i, ny-2]
            u[i, ny-1] = u[i, 1]

        # hjørner
        u[0, 0]       = u[nx-2, ny-2]
        u[0, ny-1]    = u[nx-2, 1]
        u[nx-1, 0]    = u[1, ny-2]
        u[nx-1, ny-1] = u[1, 1]

    @staticmethod
    @njit(fastmath=True)
    def dx_central(u, dx, out):
        nx, ny = u.shape
        c = 0.5 / dx
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                out[i-1, j-1] = (u[i+1, j] - u[i-1, j]) * c

    @staticmethod
    @njit(fastmath=True)
    def dy_central(u, dy, out):
        nx, ny = u.shape
        c = 0.5 / dy
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                out[i-1, j-1] = (u[i, j+1] - u[i, j-1]) * c

    @staticmethod
    @njit(fastmath=True)
    def d2x_central(u, dx, out):
        nx, ny = u.shape
        c = 1.0 / (dx * dx)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                out[i-1, j-1] = (u[i+1, j] - 2.0*u[i, j] + u[i-1, j]) * c

    @staticmethod
    @njit(fastmath=True)
    def d2y_central(u, dy, out):
        nx, ny = u.shape
        c = 1.0 / (dy * dy)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                out[i-1, j-1] = (u[i, j+1] - 2.0*u[i, j] + u[i, j-1]) * c

    @staticmethod
    @njit(fastmath=True)
    def dxdy_central(u, dx, dy, out):
        nx, ny = u.shape
        c = 0.25 / (dx * dy)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                out[i-1, j-1] = (
                    u[i+1, j+1] - u[i+1, j-1]
                    - u[i-1, j+1] + u[i-1, j-1]
                ) * c

