from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import (
    apply_periodic_bc_2d,
    laplace,
    advection,
)

@dataclass
class CoupledRDAdv2D:
    """
    Simpelt dobbelt-koblet PDE-system (u,v):
        u_t = -c·∇u + Du Δu + α (v - u)
        v_t = -c·∇v + Dv Δv + β (u - v)

    Periodiske randbetingelser via ghost cells.

    dt beregnes i __post_init__ ud fra grov stabilitet:
      - advektion: dt <= C_adv * min(dx/|cx|, dy/|cy|)
      - diffusion: dt <= C_diff * 0.5 / (Du*(1/dx^2+1/dy^2)) og samme for Dv
    """
    grid: Grid2D

    # advektion (konstant hastighed)
    cx: float = 1.0
    cy: float = 0.5

    # diffusion
    Du: float = 0.05
    Dv: float = 0.02

    # kobling
    alpha: float = 0.8
    beta: float = 0.6

    # sikkerhedsfaktorer (eksplicit RK4 er stadig eksplicit mht. PDE-stabilitet)
    C_adv: float = 0.4
    C_diff: float = 0.2

    # initial condition parametre
    x0: float = 10.0
    y0: float = 12.0
    sigma: float = 2.0

    name: str = "Coupled RD + Advection (2D)"

    def __post_init__(self):
        g = self.grid

        # --- dt fra advektion (CFL) ---
        dt_adv = np.inf
        if abs(self.cx) > 1e-12:
            dt_adv = min(dt_adv, g.dx / abs(self.cx))
        if abs(self.cy) > 1e-12:
            dt_adv = min(dt_adv, g.dy / abs(self.cy))
        dt_adv *= self.C_adv

        # --- dt fra diffusion ---
        # For 2D eksplicit diffusion er et groft kriterie:
        # dt <= C / (D * (1/dx^2 + 1/dy^2))
        inv_dx2 = 1.0 / (g.dx * g.dx)
        inv_dy2 = 1.0 / (g.dy * g.dy)
        denom_u = self.Du * (inv_dx2 + inv_dy2)
        denom_v = self.Dv * (inv_dx2 + inv_dy2)

        dt_diff_u = np.inf if denom_u <= 1e-30 else self.C_diff / denom_u
        dt_diff_v = np.inf if denom_v <= 1e-30 else self.C_diff / denom_v

        self.dt = float(min(dt_adv, dt_diff_u, dt_diff_v))
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"Kunne ikke bestemme en stabil dt, fik dt={self.dt}")

        # prealloc scratch til rhs (undgår allokeringer i hver step)
        nx, ny = g.nx, g.ny
        self._tmpx = np.zeros((nx, ny), dtype=g.dtype)
        self._tmpy = np.zeros((nx, ny), dtype=g.dtype)
        self._lap_tmp = np.zeros((nx, ny), dtype=g.dtype)   # bruges i laplace
        self._rhs_adv = np.zeros((nx, ny), dtype=g.dtype)   # advektion output
        self._rhs_lap = np.zeros((nx, ny), dtype=g.dtype)   # laplace output

    @property
    def m(self) -> int:
        return 2  # (u,v)

    def initial_condition(self) -> np.ndarray:
        """
        Returnerer U med shape (m, nx+2g, ny+2g).
        U[0]=u, U[1]=v.
        """
        g = self.grid
        U = np.zeros((2, g.nx + 2*g.ghost, g.ny + 2*g.ghost), dtype=g.dtype)

        X, Y = g.XY()  # (nx,ny) på interior-grid
        r2 = (X - self.x0)**2 + (Y - self.y0)**2

        u0 = np.exp(-r2 / (2.0 * self.sigma**2)).astype(g.dtype)
        v0 = (0.5*np.exp(-((X-(self.x0+6.0))**2 + (Y-(self.y0-4.0))**2) / (2.0*(1.4*self.sigma)**2))
              ).astype(g.dtype)

        # put i interior
        U[0, 1:-1, 1:-1] = u0
        U[1, 1:-1, 1:-1] = v0

        # enforce periodic ghosts
        apply_periodic_bc_2d(U[0])
        apply_periodic_bc_2d(U[1])
        return U

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        """
        U:       (2, nx+2, ny+2) ghosted
        out_rhs: (2, nx,   ny)    interior output
        """
        g = self.grid

        u = U[0]
        v = U[1]

        # 1) periodiske BC på ghost cells (krævet før afledte)
        apply_periodic_bc_2d(u)
        apply_periodic_bc_2d(v)

        # views til output
        Fu = out_rhs[0]
        Fv = out_rhs[1]

        # 2) advektion + diffusion for u
        advection(u, self.cx, self.cy, g.dx, g.dy,
                  out=self._rhs_adv, tmpx=self._tmpx, tmpy=self._tmpy)
        laplace(u, g.dx, g.dy,
                out=self._rhs_lap, tmp=self._lap_tmp)

        # Fu = adv + Du*lap + coupling
        Fu[:] = self._rhs_adv + self.Du * self._rhs_lap

        # 3) advektion + diffusion for v
        advection(v, self.cx, self.cy, g.dx, g.dy,
                  out=self._rhs_adv, tmpx=self._tmpx, tmpy=self._tmpy)
        laplace(v, g.dx, g.dy,
                out=self._rhs_lap, tmp=self._lap_tmp)

        Fv[:] = self._rhs_adv + self.Dv * self._rhs_lap

        # 4) kobling (brug interior values)
        u0 = u[1:-1, 1:-1]
        v0 = v[1:-1, 1:-1]
        Fu[:] += self.alpha * (v0 - u0)
        Fv[:] += self.beta  * (u0 - v0)
