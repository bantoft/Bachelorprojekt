from __future__ import annotations

import numpy as np
from numba import njit, prange

from engine.grid import Grid2D
from engine.num_diff import boundary, dx, dy, laplace
# Four strongly coupled fields:
# 0: density n
# 1: vorticity w
# 2: temperature T
# 3: impurity concentration c
N_FIELDS = 4


@njit(parallel=True, fastmath=True, cache=True)
def _rhs_update(
    n: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    c: np.ndarray,
    out: np.ndarray,
    lap_n: np.ndarray,
    lap_w: np.ndarray,
    lap_T: np.ndarray,
    lap_c: np.ndarray,
    dx_n: np.ndarray,
    dy_n: np.ndarray,
    dx_w: np.ndarray,
    dy_w: np.ndarray,
    dx_T: np.ndarray,
    dy_T: np.ndarray,
    dx_c: np.ndarray,
    dy_c: np.ndarray,
    dx_phi: np.ndarray,
    dy_phi: np.ndarray,
    Dn: float,
    Dw: float,
    DT: float,
    Dc: float,
    mu_w: float,
    beta_nT: float,
    gamma_n: float,
    gamma_w: float,
    gamma_T: float,
    gamma_c: float,
    forcing_ny: float,
    source_T: float,
    source_c: float,
) -> None:
    out[:, :, :] = 0.0
    nx, ny = n.shape

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            vx = -dy_phi[i, j]
            vy = dx_phi[i, j]

            adv_n = vx * dx_n[i, j] + vy * dy_n[i, j]
            adv_w = vx * dx_w[i, j] + vy * dy_w[i, j]
            adv_T = vx * dx_T[i, j] + vy * dy_T[i, j]
            adv_c = vx * dx_c[i, j] + vy * dy_c[i, j]

            n_ij = n[i, j]
            w_ij = w[i, j]
            T_ij = T[i, j]
            c_ij = c[i, j]

            # n-equation: diffusion + advection + logistic drive + cross-loss + forcing.
            out[0, i, j] = (
                Dn * lap_n[i, j]
                - adv_n
                + gamma_n * n_ij * (1.0 - n_ij)
                - beta_nT * n_ij * T_ij
                + forcing_ny * dy_phi[i, j]
            )

            # w-equation: vorticity transport, damping, and coupling to gradients/impurity.
            out[1, i, j] = (
                Dw * lap_w[i, j]
                - adv_w
                - mu_w * w_ij
                + gamma_w * (dy_n[i, j] - dx_T[i, j])
                + 0.35 * c_ij
            )

            # T-equation: transport, production from n, dissipation and nonlinear source.
            out[2, i, j] = (
                DT * lap_T[i, j]
                - adv_T
                + source_T * n_ij
                - gamma_T * T_ij
                + 0.1 * w_ij * w_ij
                - 0.2 * c_ij * T_ij
            )

            # c-equation: slower transport + production from n,T and damping.
            out[3, i, j] = (
                Dc * lap_c[i, j]
                - adv_c
                + source_c * n_ij * T_ij
                - gamma_c * c_ij
                + 0.15 * w_ij
            )


def rhs(
    t: float,
    u: np.ndarray,
    out: np.ndarray,
    grid: Grid2D,
    lap_n: np.ndarray,
    lap_w: np.ndarray,
    lap_T: np.ndarray,
    lap_c: np.ndarray,
    lap_tmp: np.ndarray,
    dx_n: np.ndarray,
    dy_n: np.ndarray,
    dx_w: np.ndarray,
    dy_w: np.ndarray,
    dx_T: np.ndarray,
    dy_T: np.ndarray,
    dx_c: np.ndarray,
    dy_c: np.ndarray,
    dx_phi: np.ndarray,
    dy_phi: np.ndarray,
    phi: np.ndarray,
    Dn: float,
    Dw: float,
    DT: float,
    Dc: float,
    mu_w: float,
    beta_nT: float,
    gamma_n: float,
    gamma_w: float,
    gamma_T: float,
    gamma_c: float,
    forcing_ny: float,
    source_T: float,
    source_c: float,
    a_n: float,
    a_T: float,
    a_c: float,
) -> None:
    """Four-field nonlinear coupled plasma-inspired model in 2D."""
    n = u[0]
    w = u[1]
    T = u[2]
    c = u[3]

    # Enforce periodic BC before derivative evaluation.
    boundary(n)
    boundary(w)
    boundary(T)
    boundary(c)

    # Coupling potential and ExB-like velocity.
    phi[:, :] = a_n * n + a_T * T - a_c * c
    dx(phi, grid.dx, dx_phi)
    dy(phi, grid.dy, dy_phi)

    # Field derivatives.
    laplace(n, grid.dx, grid.dy, lap_n, lap_tmp)
    laplace(w, grid.dx, grid.dy, lap_w, lap_tmp)
    laplace(T, grid.dx, grid.dy, lap_T, lap_tmp)
    laplace(c, grid.dx, grid.dy, lap_c, lap_tmp)

    dx(n, grid.dx, dx_n)
    dy(n, grid.dy, dy_n)
    dx(w, grid.dx, dx_w)
    dy(w, grid.dy, dy_w)
    dx(T, grid.dx, dx_T)
    dy(T, grid.dy, dy_T)
    dx(c, grid.dx, dx_c)
    dy(c, grid.dy, dy_c)

    _rhs_update(
        n=n,
        w=w,
        T=T,
        c=c,
        out=out,
        lap_n=lap_n,
        lap_w=lap_w,
        lap_T=lap_T,
        lap_c=lap_c,
        dx_n=dx_n,
        dy_n=dy_n,
        dx_w=dx_w,
        dy_w=dy_w,
        dx_T=dx_T,
        dy_T=dy_T,
        dx_c=dx_c,
        dy_c=dy_c,
        dx_phi=dx_phi,
        dy_phi=dy_phi,
        Dn=Dn,
        Dw=Dw,
        DT=DT,
        Dc=Dc,
        mu_w=mu_w,
        beta_nT=beta_nT,
        gamma_n=gamma_n,
        gamma_w=gamma_w,
        gamma_T=gamma_T,
        gamma_c=gamma_c,
        forcing_ny=forcing_ny,
        source_T=source_T,
        source_c=source_c,
    )


def ic(grid: Grid2D) -> np.ndarray:
    """Initial condition with coupled perturbations around a radial profile."""
    state = grid.alloc()
    x, y = grid.XY()

    x0 = 0.5 * (grid.cfg.x_min + grid.cfg.x_max)
    y0 = 0.5 * (grid.cfg.y_min + grid.cfg.y_max)
    r2 = (x - x0) ** 2 + (y - y0) ** 2

    base = np.exp(-r2 / 18.0)
    shear = np.sin(0.33 * x) * np.cos(0.27 * y)
    ring = np.exp(-((np.sqrt(r2) - 5.5) ** 2) / 4.0)

    state[0, 1:-1, 1:-1] = 0.45 + 0.30 * base + 0.10 * shear
    state[1, 1:-1, 1:-1] = 0.05 * np.sin(0.7 * x) + 0.08 * ring
    state[2, 1:-1, 1:-1] = 0.40 + 0.25 * base - 0.05 * shear
    state[3, 1:-1, 1:-1] = 0.20 + 0.10 * ring + 0.03 * np.cos(0.6 * x + 0.2 * y)

    boundary(state[0])
    boundary(state[1])
    boundary(state[2])
    boundary(state[3])
    return state


def create_rhs_kwargs(grid: Grid2D, u: np.ndarray) -> dict[str, np.ndarray | float]:
    zeros = np.zeros_like(u[0])

    return {
        "lap_n": zeros.copy(),
        "lap_w": zeros.copy(),
        "lap_T": zeros.copy(),
        "lap_c": zeros.copy(),
        "lap_tmp": zeros.copy(),
        "dx_n": zeros.copy(),
        "dy_n": zeros.copy(),
        "dx_w": zeros.copy(),
        "dy_w": zeros.copy(),
        "dx_T": zeros.copy(),
        "dy_T": zeros.copy(),
        "dx_c": zeros.copy(),
        "dy_c": zeros.copy(),
        "dx_phi": zeros.copy(),
        "dy_phi": zeros.copy(),
        "phi": zeros.copy(),
        # Tuned to keep the model active while still stable for dt around 1e-3 to 1e-2.
        "Dn": 0.018,
        "Dw": 0.014,
        "DT": 0.011,
        "Dc": 0.006,
        "mu_w": 0.28,
        "beta_nT": 0.55,
        "gamma_n": 0.72,
        "gamma_w": 0.48,
        "gamma_T": 0.24,
        "gamma_c": 0.31,
        "forcing_ny": 0.07,
        "source_T": 0.35,
        "source_c": 0.22,
        "a_n": 0.80,
        "a_T": 0.55,
        "a_c": 0.60,
    }
