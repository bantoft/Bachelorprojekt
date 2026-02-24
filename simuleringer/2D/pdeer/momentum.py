# pdeer/momentum.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import boundery, dx, dy, laplace

@dataclass
class Momentum:
    """
    2D 'Braginskii-inspireret' model:
      n_t + div(n V) = 0
      V_t + (V·∇)V = -(1/(m n)) ∇p + nu ∇^2 V + (e/m)(E + (1/c) V×B) - nu_c (V - V0)

    Felter i U:
      U[0] = n   (ghosted)
      U[1] = Vx  (ghosted)
      U[2] = Vy  (ghosted)
    """
    grid: Grid2D

    # fysiske/closure parametre
    m_s: float = 1.0        # masse
    e_s: float = 1.0        # ladning (med fortegn)
    c: float = 1.0          # "lysets hastighed" (skaleringsparameter)
    T0: float = 1.0         # isothermal temperatur => p = n*T0

    # dissipative parametre
    nu: float = 0.05        # kinematisk viskositet (≈ effektiv -div pi /(m n))
    nu_c: float = 0.0       # kollisions-/drag-rate
    V0x: float = 0.0        # mål-hastighed for drag
    V0y: float = 0.0

    # EM felter (start simpelt: konstante)
    Ex: float = 0.0
    Ey: float = 0.0
    Bz: float = 0.0

    # numerik
    C_adv: float = 0.25
    C_diff: float = 0.20
    n_floor: float = 1e-6

    name: str = "Momentum (Braginskii-inspired, 2D)"

    @property
    def m(self) -> int:
        return 3

    @property
    def field_names(self):
        return ["n", "Vx", "Vy"]

    def __post_init__(self):
        nx, ny = self.grid.nx, self.grid.ny
        dt_adv = self._stable_dt_adv()
        dt_diff = self._stable_dt_diff()
        self.dt = min(dt_adv, dt_diff)

        # scratch (interior arrays: (nx, ny))
        self._dxVx = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._dyVx = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._dxVy = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._dyVy = np.zeros((nx, ny), dtype=self.grid.dtype)

        self._dxp  = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._dyp  = np.zeros((nx, ny), dtype=self.grid.dtype)

        self._lapVx = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._lapVy = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._tmp   = np.zeros((nx, ny), dtype=self.grid.dtype)

        # flux arrays (ghosted) til kontinuitet
        self._Fx = self.grid.alloc()  # n*Vx
        self._Fy = self.grid.alloc()  # n*Vy
        self._dxFx = np.zeros((nx, ny), dtype=self.grid.dtype)
        self._dyFy = np.zeros((nx, ny), dtype=self.grid.dtype)

    def _stable_dt_adv(self) -> float:
        # grov CFL: dt <= C * min(dx/|V|max, dy/|V|max)
        # (vi kender ikke |V|max før simulering; brug et konservativt bud)
        Vmax_guess = 1.0
        return self.C_adv * min(self.grid.dx, self.grid.dy) / max(Vmax_guess, 1e-12)

    def _stable_dt_diff(self) -> float:
        # eksplicit diffusion: dt <= C * min(dx^2, dy^2) / nu
        if self.nu <= 0.0:
            return np.inf
        return self.C_diff * min(self.grid.dx*self.grid.dx, self.grid.dy*self.grid.dy) / self.nu

    def initial_condition(self) -> np.ndarray:
        U = np.zeros((3, self.grid.nx + 2, self.grid.ny + 2), dtype=self.grid.dtype)
        X, Y = self.grid.XY()

        # n: en lille gaussian på en baggrund
        n0 = 1.0 + 0.2*np.exp(-((X-15.0)**2 + (Y-25.0)**2)/(2.0*2.0**2))
        U[0, 1:-1, 1:-1] = n0

        # V: start med 0 eller en lille shear
        U[1, 1:-1, 1:-1] = 0.0
        U[2, 1:-1, 1:-1] = 0.0

        # periodiske BC
        boundery(U[0]); boundery(U[1]); boundery(U[2])
        return U

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        n  = U[0]; Vx = U[1]; Vy = U[2]

        # periodiske BC før afledte
        boundery(n); boundery(Vx); boundery(Vy)

        # ---------- n_t = -div(nV) ----------
        self._Fx[1:-1, 1:-1] = n[1:-1, 1:-1] * Vx[1:-1, 1:-1]
        self._Fy[1:-1, 1:-1] = n[1:-1, 1:-1] * Vy[1:-1, 1:-1]
        boundery(self._Fx); boundery(self._Fy)

        dx(self._Fx, self.grid.dx, self._dxFx)
        dy(self._Fy, self.grid.dy, self._dyFy)
        out_rhs[0, :, :] = -(self._dxFx + self._dyFy)

        # ---------- hjælp: tryk p = n*T0 ----------
        # (lav p som ghosted i _Fx for at spare en alloc)
        self._Fx[1:-1, 1:-1] = self.T0 * n[1:-1, 1:-1]
        boundery(self._Fx)
        dx(self._Fx, self.grid.dx, self._dxp)
        dy(self._Fx, self.grid.dy, self._dyp)

        # ---------- konvektion (V·∇)V ----------
        dx(Vx, self.grid.dx, self._dxVx)
        dy(Vx, self.grid.dy, self._dyVx)
        dx(Vy, self.grid.dx, self._dxVy)
        dy(Vy, self.grid.dy, self._dyVy)

        Vx0 = Vx[1:-1, 1:-1]
        Vy0 = Vy[1:-1, 1:-1]
        n0  = n[1:-1, 1:-1]
        inv_mn = 1.0 / (self.m_s * np.maximum(n0, self.n_floor))

        adv_Vx = Vx0 * self._dxVx + Vy0 * self._dyVx
        adv_Vy = Vx0 * self._dxVy + Vy0 * self._dyVy

        # ---------- viskositet: nu * laplace(V) ----------
        laplace(Vx, self.grid.dx, self.grid.dy, self._lapVx, self._tmp)
        laplace(Vy, self.grid.dx, self.grid.dy, self._lapVy, self._tmp)

        # ---------- Lorentz + E ----------
        # V×B med B=(0,0,Bz): (Vy*Bz, -Vx*Bz)
        lor_x = self.Ex + (1.0/self.c) * (Vy0 * self.Bz)
        lor_y = self.Ey + (1.0/self.c) * (-Vx0 * self.Bz)
        EM_pref = self.e_s / self.m_s

        # ---------- drag: -nu_c (V - V0) ----------
        drag_x = -self.nu_c * (Vx0 - self.V0x)
        drag_y = -self.nu_c * (Vy0 - self.V0y)

        # ---------- momentum RHS ----------
        out_rhs[1, :, :] = (
            -adv_Vx
            -inv_mn * self._dxp
            +self.nu * self._lapVx
            +EM_pref * lor_x
            +drag_x
        )

        out_rhs[2, :, :] = (
            -adv_Vy
            -inv_mn * self._dyp
            +self.nu * self._lapVy
            +EM_pref * lor_y
            +drag_y
        )