# simuleringer/2D/pdeer/tripple_test.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import apply_periodic_bc_2d, laplace, advection

@dataclass
class Coupled3RDAdv2D:
    """
    3-koblet system (u,v,w):
      u_t = -c·∇u + Du Δu + alpha (v - u)
      v_t = -c·∇v + Dv Δv + β (w - v)
      w_t = -c·∇w + Dw Δw + gamma (u - w)

    Periodiske BC via ghost cells.
    dt sættes i __post_init__ ud fra (advektion + diffusion) stabilitetskriterier.
    """
    grid: Grid2D

    # advektion (konstant)
    cx: float = 1.0
    cy: float = 0.6

    # diffusion
    Du: float = 0.05
    Dv: float = 0.03
    Dw: float = 0.02

    # koblingsstyrker (ring)
    alpha: float = 0.8
    beta: float = 0.7
    gamma: float = 0.6

    # sikkerhedsfaktorer
    C_adv: float = 0.4
    C_diff: float = 0.2

    # initial condition parametre
    x0: float = 10.0
    y0: float = 12.0
    sigma: float = 2.0

    name: str = "3-field Coupled RD + Advection (2D)"

    def __post_init__(self):
        g = self.grid

        # --- dt fra advektion (CFL) ---
        dt_adv = np.inf
        if abs(self.cx) > 1e-12:
            dt_adv = min(dt_adv, g.dx / abs(self.cx))
        if abs(self.cy) > 1e-12:
            dt_adv = min(dt_adv, g.dy / abs(self.cy))
        dt_adv *= self.C_adv

        # --- dt fra diffusion (værste D) ---
        inv_dx2 = 1.0 / (g.dx * g.dx)
        inv_dy2 = 1.0 / (g.dy * g.dy)
        base = (inv_dx2 + inv_dy2)

        def dt_diff(D: float) -> float:
            denom = D * base
            return np.inf if denom <= 1e-30 else self.C_diff / denom

        dt = min(dt_adv, dt_diff(self.Du), dt_diff(self.Dv), dt_diff(self.Dw))
        self.dt = float(dt)

        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"Kunne ikke bestemme en stabil dt, fik dt={self.dt}")

        # scratch (genbruges for alle tre felter)
        nx, ny = g.nx, g.ny
        self._tmpx = np.zeros((nx, ny), dtype=g.dtype)
        self._tmpy = np.zeros((nx, ny), dtype=g.dtype)
        self._lap_tmp = np.zeros((nx, ny), dtype=g.dtype)
        self._rhs_adv = np.zeros((nx, ny), dtype=g.dtype)
        self._rhs_lap = np.zeros((nx, ny), dtype=g.dtype)


    def initial_condition(self) -> np.ndarray:
        """
        Returnerer U med shape (3, nx+2g, ny+2g).
        """
        g = self.grid
        U = np.zeros((3, g.nx + 2*g.ghost, g.ny + 2*g.ghost), dtype=g.dtype)

        X, Y = g.XY()

        def gauss(xc, yc, s, amp=1.0):
            r2 = (X - xc)**2 + (Y - yc)**2
            return (amp * np.exp(-r2 / (2.0 * s*s))).astype(g.dtype)

        u0 = gauss(self.x0, self.y0, self.sigma, amp=1.0)
        v0 = gauss(self.x0 + 7.0, self.y0 - 4.0, 1.3*self.sigma, amp=0.7)
        w0 = gauss(self.x0 - 6.0, self.y0 + 5.0, 0.9*self.sigma, amp=0.5)

        U[0, 1:-1, 1:-1] = u0
        U[1, 1:-1, 1:-1] = v0
        U[2, 1:-1, 1:-1] = w0

        apply_periodic_bc_2d(U[0])
        apply_periodic_bc_2d(U[1])
        apply_periodic_bc_2d(U[2])
        return U

    def _adv_diff(self, field_ghosted: np.ndarray, D: float, out: np.ndarray) -> None:
        """
        out (nx,ny) = -c·∇field + D Δfield
        """
        g = self.grid
        advection(field_ghosted, self.cx, self.cy, g.dx, g.dy,
                  out=self._rhs_adv, tmpx=self._tmpx, tmpy=self._tmpy)
        laplace(field_ghosted, g.dx, g.dy,
                out=self._rhs_lap, tmp=self._lap_tmp)
        out[:] = self._rhs_adv + D * self._rhs_lap
    
        
    @property
    def m(self) -> int:
        return 3
    
    @property
    def field_names(self):
        return ["u", "v", "w"]

    def U0(self):
        return self.initial_condition()

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        """
        U:       (3, nx+2, ny+2) ghosted
        out_rhs: (3, nx,   ny)    interior
        """
        u = U[0]
        v = U[1]
        w = U[2]

        # BC før derivativer
        apply_periodic_bc_2d(u)
        apply_periodic_bc_2d(v)
        apply_periodic_bc_2d(w)

        Fu = out_rhs[0]
        Fv = out_rhs[1]
        Fw = out_rhs[2]

        # advektion+diffusion
        self._adv_diff(u, self.Du, Fu)
        self._adv_diff(v, self.Dv, Fv)
        self._adv_diff(w, self.Dw, Fw)

        # kobling (ring): u<-v, v<-w, w<-u
        u0 = u[1:-1, 1:-1]
        v0 = v[1:-1, 1:-1]
        w0 = w[1:-1, 1:-1]

        Fu[:] += self.alpha * (v0 - u0)
        Fv[:] += self.beta  * (w0 - v0)
        Fw[:] += self.gamma * (u0 - w0)
