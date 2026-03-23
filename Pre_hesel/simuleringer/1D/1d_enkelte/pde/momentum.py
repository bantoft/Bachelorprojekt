from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional

from setup import Grid1D, ddx_central


Array = np.ndarray


@dataclass
class Momentum1D:
    """
    Momentum-lgningen:
        (d_t V + V nabla_x V)m n = - d_x_alpha p - nabla_x_beta pi + q n (E + VxB) + R
    dvs.
        d_t V = - V_x d_x V_x - 1/(m n) d_x_alpha p - 1//(n m) nabla_x_beta pi q/ms(E + VxB) + R 
        y = [ n (N), Vx (N), Vy (N), Vz (N) ]
    """
    m: float
    q: float
    Tfun:Callable[[Array, float], Array] 
    Efun:Callable[[Array, float], Array]
    Bfun:Callable[[Array, float], Array]
    pi_xj_fun: Optional[Callable[[Array, Array, float], Array]] = None
    Rfun: Optional[Callable[[Array, Array, Array, float], Array]] = None
    n_floor: float = 1e-10

    name: str = "Braginskii Momentum 1D"

    def initial_condition(self, x: Array) -> Array:
        """
        Default: n = 1, V = 0. Du kan override i din runner hvis du vil.
        """
        n0 = np.ones(x.size)
        V0 = np.zeros((x.size, 3))
        return self.pack(n0, V0)

    # Hjælpe funktioner
    def pack(self, n: Array, V: Array) -> Array: 
        return np.concatenate([n, V[:, 0], V[:, 1], V[:, 2]])

    def unpack(self, y: Array, N: int) -> tuple[Array, Array]:
        n = y[:N]
        Vx = y[N:2*N]
        Vy = y[2*N:3*N]
        Vz = y[3*N:4*N]
        V = np.stack([Vx, Vy, Vz], axis=1)
        return n, V

    def pressure(self, n: Array, T: Array) -> Array: return n * T

    def rhs(self, y: Array, t: float, grid: Grid1D) -> Array:
        x = grid.x
        N = x.size

        n, V = self.unpack(y, N)
        n_safe = np.maximum(n, self.n_floor)

        T = self.Tfun(x, t)
        p = self.pressure(n_safe, T)

        flux_n = n_safe * V[:, 0]
        n_t = -ddx_central(flux_n, grid.dx)

        dVdx = np.zeros_like(V)

        for j in range(3): dVdx[:, j] = ddx_central(V[:, j], grid.dx)
        V_xd_xV_x = V[:, [0]] * dVdx

        dpdx = ddx_central(p, grid.dx)
        gradp_term = np.zeros_like(V)
        gradp_term[:, 0] = -(1.0 / (self.m * n_safe)) * dpdx

        B = self.Bfun(x, t)
        E = self.Efun(x, t)
        VxB = np.cross(V, B)
        lorentz_term = (self.q / self.m) * (E + VxB)

        stress_term = np.zeros_like(V)
        if self.pi_xj_fun is not None:
            pi_xj = self.pi_xj_fun(x, V, t)  # (N,3)
            dpi_dx = np.zeros_like(V)

            for j in range(3): dpi_dx[:, j] = ddx_central(pi_xj[:, j], grid.dx)
            stress_term = -(1.0 / (self.m * n_safe)) * dpi_dx

        R_term = np.zeros_like(V)
        if self.Rfun is not None:
            R = self.Rfun(x, n_safe, V, t)   # (N,3)
            R_term = (1.0 / (self.m * n_safe))[:, None] * R

        V_t = -V_xd_xV_x + gradp_term + lorentz_term + stress_term + R_term

        return self.pack(n_t, V_t)
