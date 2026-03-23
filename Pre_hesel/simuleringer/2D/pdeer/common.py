# 2D/pdeer/common.py

from __future__ import annotations

import numpy as np


def stable_dt_advection(dx: float, dy: float, vx: float, vy: float, c_adv: float) -> float:
    dt_adv = np.inf
    if abs(vx) > 1e-12:
        dt_adv = min(dt_adv, dx / abs(vx))
    if abs(vy) > 1e-12:
        dt_adv = min(dt_adv, dy / abs(vy))
    return float(dt_adv * c_adv)


def stable_dt_diffusion(dx: float, dy: float, diffusivities: tuple[float, ...], c_diff: float) -> float:
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    base = inv_dx2 + inv_dy2

    dt_diff = np.inf
    for diffusion in diffusivities:
        denom = diffusion * base
        dt_candidate = np.inf if denom <= 1e-30 else c_diff / denom
        dt_diff = min(dt_diff, dt_candidate)

    return float(dt_diff)


def gaussian2d(X: np.ndarray,
               Y: np.ndarray,
               x_center: float,
               y_center: float,
               sigma: float,
               dtype,
               amp: float = 1.0) -> np.ndarray:
    r2 = (X - x_center) ** 2 + (Y - y_center) ** 2
    return (amp * np.exp(-r2 / (2.0 * sigma * sigma))).astype(dtype)


def alloc_xy_scratch(nx: int, ny: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((nx, ny), dtype=dtype),
        np.zeros((nx, ny), dtype=dtype),
    )
