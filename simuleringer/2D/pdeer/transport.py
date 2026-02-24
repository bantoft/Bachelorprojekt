# 2D/transport.py
from __future__ import annotations


import numpy as np

from dataclasses import dataclass

from kerne.grid import Grid2D
from kerne.operators import advection, boundery
from pdeer.common import alloc_xy_scratch, gaussian2d, stable_dt_advection


@dataclass
class Transport:
    """
    Transport
    partial_t n = - nabla(n V)
    r = [x, y]

    V(r, t) = [2, 3]
    n(r, t) = Multivariat gauss fordeling
    mu = [15, 25]
    """
    grid: Grid2D
    vx: float = 2.0
    vy: float = 3.0
    nx: float = 15.0
    ny: float = 25.0
    sigma: float = 2.0

    #CFL stabilitetsfaktor for advektion
    C_adv: float = 0.5
    name: str = "Transport"

    @property
    def m(self) -> int:
        return 1
    
    @property
    def field_names(self):
        return ["n"]

    def __post_init__(self):
        self.dt = stable_dt_advection(self.grid.dx, self.grid.dy, self.vx, self.vy, self.C_adv)
        self._tmpx, self._tmpy = alloc_xy_scratch(self.grid.nx, self.grid.ny, self.grid.dtype)
        self._rhs_adv = np.zeros((self.grid.nx, self.grid.ny), dtype=self.grid.dtype)

    def initial_condition(self) -> np.ndarray:
        U = np.zeros((1, self.grid.nx + 2, self.grid.ny + 2), dtype=self.grid.dtype)
        X, Y = self.grid.XY()

        u0 = gaussian2d(X, Y, self.nx, self.ny, self.sigma, self.grid.dtype)
        U[0, 1:-1, 1:-1] = u0
        boundery(U[0])
        return U


    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        u = U[0]
        boundery(u)
        advection(u, self.vx, self.vy, self.grid.dx, self.grid.dy,
                  out=self._rhs_adv, tmpx=self._tmpx, tmpy=self._tmpy)
        out_rhs[0, :, :] = -self._rhs_adv

