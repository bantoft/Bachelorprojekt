from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import boundery, advection, laplace

@dataclass
class Temperatur:
    """
    Simpel Braginskii-energi/temperatur model (start-version):

        dT/dt + V·∇T = chi * ∇^2 T + (2/(3n))*(Q0 - lam*(T-Teq) + H)

    Vi antager her n = konstant, V = konstant.
    """
    grid: Grid2D

    # "givne" plasma-parametre (start simpelt)
    n0: float = 1.0          # konstant densitet
    vx: float = 1.0
    vy: float = 0.0

    kappa: float = 0.05      # varmeledning
    lam: float = 0.0         # relaxation
    Teq: float = 0.0         # equilibrium temp
    Q0: float = 0.0          # konstant opvarmning
    H0: float = 0.0          # viskøs opvarmning (her bare konstant)

    # stabilitetsfaktorer (konservativt for eksplicit RK4)
    C_adv: float = 0.4
    C_diff: float = 0.2

    name: str = "Temperatur"

    @property
    def m(self) -> int:
        return 1

    @property
    def field_names(self):
        return ["T"]

    def __post_init__(self):
        # chi = (2/3) kappa / n
        self.chi = (2.0 / 3.0) * (self.kappa / max(self.n0, 1e-12))

        # dt-begrænsning (rough):
        # advektion: dt <= C * min(dx/|vx|, dy/|vy|)
        dt_adv = np.inf
        if abs(self.vx) > 1e-14:
            dt_adv = min(dt_adv, self.grid.dx / abs(self.vx))
        if abs(self.vy) > 1e-14:
            dt_adv = min(dt_adv, self.grid.dy / abs(self.vy))
        dt_adv *= self.C_adv

        # diffusion: dt <= C * min(dx^2, dy^2) / chi
        dt_diff = np.inf
        if self.chi > 0:
            dt_diff = self.C_diff * min(self.grid.dx**2, self.grid.dy**2) / self.chi

        self.dt = float(min(dt_adv, dt_diff))
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"Kunne ikke bestemme stabil dt, fik dt={self.dt}")

        nx, ny = self.grid.nx, self.grid.ny
        self._tmpx = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._tmpy = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._lap  = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._adv  = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._tmp  = np.zeros((nx, ny), dtype=self.grid.dtype)

    def initial_condition(self) -> np.ndarray:
        # U: (m, nx+2, ny+2)
        U = np.zeros((1, self.grid.nx + 2, self.grid.ny + 2), dtype=self.grid.dtype)
        X, Y = self.grid.XY()

        # fx en varm "blob"
        x0, y0, sig = 0.5*(self.grid.x_max-self.grid.x_min), 0.5*(self.grid.y_max-self.grid.y_min), 2.0
        T0 = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2.0*sig**2)).astype(self.grid.dtype)

        U[0, 1:-1, 1:-1] = T0
        boundery(U[0])
        return U

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        T = U[0]
        boundery(T)

        # advektion: V·∇T
        advection(T, self.vx, self.vy, self.grid.dx, self.grid.dy,
                  out=self._adv, tmpx=self._tmpx, tmpy=self._tmpy)

        # diffusion: ∇^2 T
        laplace(T, self.grid.dx, self.grid.dy, out=self._lap, tmp=self._tmp)

        # kildeled: (2/(3n))*(Q0 - lam*(T-Teq) + H0)
        T_in = T[1:-1, 1:-1]
        source = (2.0/(3.0*max(self.n0, 1e-12))) * (self.Q0 - self.lam*(T_in - self.Teq) + self.H0)

        # samlet RHS: dT/dt = - V·∇T + chi ∇^2T + source
        out_rhs[0, :, :] = (-self._adv + self.chi*self._lap + source).astype(self.grid.dtype)