# simuleringer/2D_ny/kerne/operators.py

from __future__ import annotations
import numpy as np
import numba
from numba import njit, prange

# ============================================================
# 2D Numba kernels (kun ét felt u(x,y))
# u:   (nx+2g, ny+2g)  (ghosted)
# out: (nx,   ny)      (interior)
# ============================================================


@njit(fastmath=True)
def apply_periodic_bc_2d(u: np.ndarray) -> None:
    nx, ny = u.shape  # inkluderer ghost-lag (typisk g=1)
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
def dx_central_2d(u: np.ndarray, dx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / dx
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - u[i - 1, j]) * c


@njit(parallel=True, fastmath=True)
def dy_central_2d(u: np.ndarray, dy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.5 / dy
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - u[i, j - 1]) * c


@njit(parallel=True, fastmath=True)
def d2x_central_2d(u: np.ndarray, dx: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (dx * dx)
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) * c


@njit(parallel=True, fastmath=True)
def d2y_central_2d(u: np.ndarray, dy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 1.0 / (dy * dy)
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) * c


@njit(parallel=True, fastmath=True)
def dxdy_central_2d(u: np.ndarray, dx: float, dy: float, out: np.ndarray) -> None:
    nx, ny = u.shape
    c = 0.25 / (dx * dy)
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            out[i - 1, j - 1] = (
                u[i + 1, j + 1] - u[i + 1, j - 1]
                - u[i - 1, j + 1] + u[i - 1, j - 1]
            ) * c





def laplace(u: np.ndarray, dx: float, dy: float, out: np.ndarray, tmp: np.ndarray) -> None:
    """
    out = Δu = ∂xx u + ∂yy u   (diskret Laplace)
    Matematik: Δu = u_xx + u_yy
    """
    d2x_central_2d(u, dx, out)
    d2y_central_2d(u, dy, tmp)
    out += tmp


def advection(u: np.ndarray, vx: float, vy: float, dx: float, dy: float, out: np.ndarray, tmpx: np.ndarray, tmpy: np.ndarray) -> None:
    """
    out = -(vx * ∂x u + vy * ∂y u)
    Matematik: - (v_x u_x + v_y u_y) = - v · ∇u
    """
    dx_central_2d(u, dx, tmpx)
    dy_central_2d(u, dy, tmpy)
    out[:] = -(vx * tmpx + vy * tmpy)
