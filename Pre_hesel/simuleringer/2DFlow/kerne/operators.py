# Operators for finite difference methods on a grid

from __future__ import annotations
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def boundery(u: np.ndarray) -> None:
    nx, ny = u.shape
    # x-retning
    for j in prange(1, ny - 1):
        u[0, j]      = u[nx - 2, j]
        u[nx - 1, j] = u[1, j]
    # y-retning
    for i in prange(1, nx - 1):
        u[i, 0]      = u[i, ny - 2]
        u[i, ny - 1] = u[i, 1]
    # hjørner
    u[0, 0]             = u[nx - 2, ny - 2]
    u[0, ny - 1]        = u[nx - 2, 1]
    u[nx - 1, 0]        = u[1, ny - 2]
    u[nx - 1, ny - 1]   = u[1, 1]


@njit(parallel=True, fastmath=True)
def dx(u: np.ndarray, dx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / dx
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - u[i - 1, j]) * c

@njit(parallel=True, fastmath=True)
def dy(u: np.ndarray, dy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / dy
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - u[i, j - 1]) * c

@njit(parallel=True, fastmath=True)
def d2x(u: np.ndarray, dx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (dx * dx)
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) * c

@njit(parallel=True, fastmath=True)
def d2y(u: np.ndarray, dy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (dy * dy)
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) * c

@njit(parallel=True, fastmath=True)
def nabla(u: np.ndarray, hx: float, hy: float, outx: np.ndarray, outy: np.ndarray) -> None:
    """
    Nabla operator: nabla u = dx u i + dy u j
    """
    dx(u, hx, outx)
    dy(u, hy, outy)

@njit(parallel=True, fastmath=True)
def advection(u: np.ndarray, vx: float, vy: float, hx: float, hy: float, out: np.ndarray, tmpx: np.ndarray, tmpy: np.ndarray) -> None:
    """
    Advektion regel:  v nabla u = vx * dx u + vy * dy u
    """
    dx(u, hx, tmpx)
    dy(u, hy, tmpy)
    out[:] = vx * tmpx + vy * tmpy


@njit(parallel=True, fastmath=True)
def laplace(u: np.ndarray, dx: float, dy: float, out: np.ndarray, tmp: np.ndarray) -> None:
    """
    laplace operator: nabla^2 = d2x u + d2y u
    """
    d2x(u, dx, out)
    d2y(u, dy, tmp)
    out += tmp
