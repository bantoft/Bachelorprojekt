from __future__ import annotations

import numpy as np

from engine.grid import Grid2D
from engine.num_diff import boundary, laplace


def rhs(
    t: float,
    u: np.ndarray,
    out: np.ndarray,
    grid: Grid2D,
    lap_tmp: np.ndarray,
    alpha: float,
) -> None:
    """Heat equation RHS: du/dt = alpha * Laplace(u)."""
    boundary(u)
    laplace(u=u, hx=grid.dx, hy=grid.dy, out=out, tmp=lap_tmp)
    out *= alpha


def ic(grid: Grid2D) -> np.ndarray:
    """Initial condition: Gaussian blob in the center of the domain."""
    u0 = grid.alloc()
    x0 = 0.5 * (grid.cfg.x_min + grid.cfg.x_max)
    y0 = 0.5 * (grid.cfg.y_min + grid.cfg.y_max)
    x, y = grid.XY()
    u0[1:-1, 1:-1] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 8.0)
    boundary(u0)
    return u0

def create_rhs_kwargs(grid: Grid2D, u: np.ndarray) -> dict[str, np.ndarray | float]:
    return {
        "lap_tmp": grid.alloc(),
        "alpha": 0.01,
    }
