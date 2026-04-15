from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def boundary(u: np.ndarray) -> None:
    nx, ny = u.shape
    for j in prange(1, ny - 1):
        u[0, j] = u[nx - 2, j]
        u[nx - 1, j] = u[1, j]
    for i in prange(1, nx - 1):
        u[i, 0] = u[i, ny - 2]
        u[i, ny - 1] = u[i, 1]
    u[0, 0] = u[nx - 2, ny - 2]
    u[0, ny - 1] = u[nx - 2, 1]
    u[nx - 1, 0] = u[1, ny - 2]
    u[nx - 1, ny - 1] = u[1, 1]


@njit(parallel=True, fastmath=True)
def dx(u: np.ndarray, hx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / hx
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - u[i - 1, j]) * c


@njit(parallel=True, fastmath=True)
def dy(u: np.ndarray, hy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / hy
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - u[i, j - 1]) * c


@njit(parallel=True, fastmath=True)
def d2x(u: np.ndarray, hx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (hx * hx)
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) * c


@njit(parallel=True, fastmath=True)
def d2y(u: np.ndarray, hy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (hy * hy)
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) * c

@njit(parallel=True, fastmath=True)
def nabla(u: np.ndarray, hx: float, hy: float, out: np.ndarray, tmp: np.ndarray) -> None:
    dx(u, hx, out)
    dy(u, hy, tmp)
    out += tmp

@njit(parallel=True, fastmath=True)
def laplace(u: np.ndarray, hx: float, hy: float, out: np.ndarray, tmp: np.ndarray) -> None:
    d2x(u, hx, out)
    d2y(u, hy, tmp)
    out += tmp

@njit(parallel=True, fastmath=True)
def advection(
    u: np.ndarray,
    vx: float,
    vy: float,
    hx: float,
    hy: float,
    out: np.ndarray,
    tmpx: np.ndarray,
    tmpy: np.ndarray,
) -> None:
    dx(u, hx, tmpx)
    dy(u, hy, tmpy)
    out[:] = vx * tmpx + vy * tmpy

