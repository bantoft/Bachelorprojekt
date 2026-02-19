from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable
import numpy as np


@dataclass
class Grid1D:
    x0: float
    x1: float
    nx: int

    def __post_init__(self):
        if self.nx < 3:
            raise ValueError("nx must be >= 3")
        self.x = np.linspace(self.x0, self.x1, self.nx)
        self.dx = self.x[1] - self.x[0]


class PDE1D(Protocol):
    """
    PDE skal kunne levere:
      - initial condition u0(x)
      - rhs(u, t, grid) der returnerer du/dt på grid
    """
    name: str

    def initial_condition(self, x: np.ndarray) -> np.ndarray: ...
    def rhs(self, u: np.ndarray, t: float, grid: Grid1D) -> np.ndarray: ...


def ddx_central(u: np.ndarray, dx: float) -> np.ndarray:
    """2. ordens central differens, periodisk BC."""
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def d2dx2_central(u: np.ndarray, dx: float) -> np.ndarray:
    """2. ordens Laplace i 1D, periodisk BC."""
    return (np.roll(u, -1) - 2.0*u + np.roll(u, 1)) / (dx*dx)


def rk4_step(f: Callable[[np.ndarray, float], np.ndarray],
             u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Klassisk RK4 for du/dt = f(u,t)."""
    k1 = f(u, t)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(u + dt*k3, t + dt)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
