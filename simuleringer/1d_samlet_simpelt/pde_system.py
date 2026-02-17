# pde_system.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def ddx_central(u: np.ndarray, dx: float) -> np.ndarray:
    """u_x ≈ (u_{i+1} - u_{i-1})/(2 dx), periodisk."""
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def ddxx_central(u: np.ndarray, dx: float) -> np.ndarray:
    """u_xx ≈ (u_{i+1} - 2u_i + u_{i-1})/(dx^2), periodisk."""
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)


@dataclass
class CoupledADR1D:
    """
    Koblet advektion-diffusion-reaktion (1D), periodisk:
        u_t = -a u_x + nu_u u_xx + alpha (v - u)
        v_t = -a v_x + nu_v v_xx + alpha (u - v)
    """
    a: float = 1.0
    nu_u: float = 2e-3
    nu_v: float = 2e-3
    alpha: float = 2.0

    def initial_condition(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u0 = np.exp(-((x - 0.30) ** 2) / (2 * 0.05**2))
        v0 = np.exp(-((x - 0.65) ** 2) / (2 * 0.07**2))
        return u0, v0

    def rhs(self, u: np.ndarray, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        ux = ddx_central(u, dx)
        vx = ddx_central(v, dx)
        uxx = ddxx_central(u, dx)
        vxx = ddxx_central(v, dx)

        du = -self.a * ux + self.nu_u * uxx + self.alpha * (v - u)
        dv = -self.a * vx + self.nu_v * vxx + self.alpha * (u - v)
        return du, dv
