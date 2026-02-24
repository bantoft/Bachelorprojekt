from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import boundery, dx, dy, laplace

@dataclass
class Braginskii2D:
    """
    Forenklet Braginskii-lignende 2D system for én art s:
      U = [n, Vx, Vy, T]

    PDE (nondimensionaliseret):
      n_t  = - div(n V)
      V_t  = - (V·∇)V - (1/(m n)) ∇p + nu ∆V + (q/m)(E + (1/c) VxB) + R/(m n)
      T_t  = - V·∇T + (kappa/n) ∆T - (2/3) T div V + (2/(3n))( eta ||∇V||^2 + Q )

    Periodiske randbetingelser via ghost-cells og boundery().
    """

    grid: Grid2D

    # fysiske/konstante parametre (du kan gøre dem til funktioner af (x,y,t) senere)
    m_s: float = 1.0          # masse
    q_s: float = 1.0          # ladning (Z*e)
    c: float = 1.0            # lyshastighed (nondim)

    nu: float = 1e-2          # kinematisk viskositet i momentum (≈ η/(m n) hvis du vil)
    kappa: float = 1e-2       # varmeledning
    eta: float = 1e-2         # viskositet til opvarmningsterm (”shear heating”)

    # simple kilder / felter
    Ex: float = 0.0
    Ey: float = 0.0
    Bz: float = 0.0           # antag B = (0,0,Bz)
    nu_drag: float = 0.0      # simpel friktion R = -m n nu_drag V
    Q0: float = 0.0           # konstant varmekilde Q

    # stabilitet
    C_adv: float = 0.25
    C_diff: float = 0.20
    dt: float | None = None

    name: str = "Braginskii2D (simplified)"
    
    @property
    def field_names(self):
        return ["n", "Vx", "Vy", "T"]

    @property
    def m(self) -> int:
        return 4

    def __post_init__(self):
        # vælg en konservativ dt hvis den ikke er givet
        if self.dt is None:
            # estimer Vmax ud fra en grov bound; du kan opdatere dt adaptivt senere
            Vmax = 1.0
            h = min(self.grid.dx, self.grid.dy)
            dt_adv  = self.C_adv  * h / max(Vmax, 1e-12)
            dt_diff = self.C_diff * h*h / max(self.nu, self.kappa, 1e-12)
            self.dt = float(min(dt_adv, dt_diff))

        nx, ny = self.grid.nx, self.grid.ny
        dtype = self.grid.dtype

        # scratch (interior)
        self._tmp1 = np.zeros((nx, ny), dtype=dtype)
        self._tmp2 = np.zeros((nx, ny), dtype=dtype)
        self._tmp3 = np.zeros((nx, ny), dtype=dtype)
        self._tmp4 = np.zeros((nx, ny), dtype=dtype)

        self._dxA = np.zeros((nx, ny), dtype=dtype)
        self._dyA = np.zeros((nx, ny), dtype=dtype)

        self._lap = np.zeros((nx, ny), dtype=dtype)
        self._lap2 = np.zeros((nx, ny), dtype=dtype)

        # scratch (ghosted) til flux/tryk
        self._p = np.zeros((nx + 2, ny + 2), dtype=dtype)
        self._fx = np.zeros((nx + 2, ny + 2), dtype=dtype)
        self._fy = np.zeros((nx + 2, ny + 2), dtype=dtype)

    def initial_condition(self) -> np.ndarray:
        """
        Returnér U med shape (4, nx+2, ny+2).
        """
        nx, ny = self.grid.nx, self.grid.ny
        U = np.zeros((4, nx + 2, ny + 2), dtype=self.grid.dtype)

        X, Y = self.grid.XY()
        # et simpelt bump i densiteten og rolig temperatur
        n0 = 1.0 + 0.1*np.exp(-((X-0.5*(self.grid.x_max-self.grid.x_min))**2 +
                               (Y-0.5*(self.grid.y_max-self.grid.y_min))**2)/(2.0*(2.0**2)))
        U[0, 1:-1, 1:-1] = n0.astype(self.grid.dtype)
        U[1, 1:-1, 1:-1] = 0.0  # Vx
        U[2, 1:-1, 1:-1] = 0.0  # Vy
        U[3, 1:-1, 1:-1] = 1.0  # T

        for k in range(4):
            boundery(U[k])
        return U

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        """
        U: (4, nx+2, ny+2) ghosted
        out_rhs: (4, nx, ny) interior
        """
        g = self.grid
        hx, hy = g.dx, g.dy

        n  = U[0]; Vx = U[1]; Vy = U[2]; T = U[3]

        # periodiske BC
        boundery(n); boundery(Vx); boundery(Vy); boundery(T)

        n0  = n[1:-1, 1:-1]
        Vx0 = Vx[1:-1, 1:-1]
        Vy0 = Vy[1:-1, 1:-1]
        T0  = T[1:-1, 1:-1]

        # ---------- Kontinuitet: n_t = -div(nV) ----------
        # fx = n*Vx, fy = n*Vy (ghosted)
        self._fx[1:-1, 1:-1] = n0 * Vx0
        self._fy[1:-1, 1:-1] = n0 * Vy0
        boundery(self._fx); boundery(self._fy)

        dx(self._fx, hx, self._tmp1)  # tmp1 = d/dx (nVx)
        dy(self._fy, hy, self._tmp2)  # tmp2 = d/dy (nVy)
        out_rhs[0, :, :] = -(self._tmp1 + self._tmp2)

        # ---------- Tryk: p = n*T ----------
        self._p[1:-1, 1:-1] = n0 * T0
        boundery(self._p)
        dx(self._p, hx, self._tmp1)   # dp/dx
        dy(self._p, hy, self._tmp2)   # dp/dy

        inv_mn = 1.0 / (self.m_s * np.maximum(n0, 1e-8))

        # ---------- Konvektion: (V·∇)V ----------
        # dVx/dx,dVx/dy,dVy/dx,dVy/dy
        dx(Vx, hx, self._dxA)   # dVx/dx
        dy(Vx, hy, self._dyA)   # dVx/dy
        convVx = Vx0*self._dxA + Vy0*self._dyA

        dx(Vy, hx, self._tmp3)  # dVy/dx
        dy(Vy, hy, self._tmp4)  # dVy/dy
        convVy = Vx0*self._tmp3 + Vy0*self._tmp4

        # ---------- Viskositet: nu * Laplace(V) ----------
        laplace(Vx, hx, hy, self._lap, self._lap2)
        viscVx = self.nu * self._lap
        laplace(Vy, hx, hy, self._lap, self._lap2)
        viscVy = self.nu * self._lap

        # ---------- Lorentz: (q/m)(E + (1/c) V×B) ----------
        # med B=(0,0,Bz): V×B = (Vy*Bz, -Vx*Bz, 0)
        ax_em = (self.q_s/self.m_s) * (self.Ex + (1.0/self.c)*(Vy0*self.Bz))
        ay_em = (self.q_s/self.m_s) * (self.Ey + (1.0/self.c)*(-Vx0*self.Bz))

        # ---------- Drag / R-term: R = -m n nu_drag V  =>  R/(m n) = -nu_drag V ----------
        Rx = -self.nu_drag * Vx0
        Ry = -self.nu_drag * Vy0

        # ---------- Momentum RHS ----------
        out_rhs[1, :, :] = -convVx - inv_mn*self._tmp1 + viscVx + ax_em + Rx
        out_rhs[2, :, :] = -convVy - inv_mn*self._tmp2 + viscVy + ay_em + Ry

        # ---------- Varme: T_t ----------
        # V·∇T
        dx(T, hx, self._tmp1)   # dT/dx
        dy(T, hy, self._tmp2)   # dT/dy
        advT = Vx0*self._tmp1 + Vy0*self._tmp2

        # div V
        dx(Vx, hx, self._tmp3)  # dVx/dx
        dy(Vy, hy, self._tmp4)  # dVy/dy
        divV = self._tmp3 + self._tmp4

        # diffusion: (kappa/n) Laplace(T)
        laplace(T, hx, hy, self._lap, self._lap2)
        diffT = (self.kappa / np.maximum(n0, 1e-8)) * self._lap

        # shear heating ~ eta * ||∇V||^2  (forenklet)
        # ||∇V||^2 = (dVx/dx)^2 + (dVx/dy)^2 + (dVy/dx)^2 + (dVy/dy)^2
        dx(Vx, hx, self._dxA); dy(Vx, hy, self._dyA)
        dx(Vy, hx, self._tmp1); dy(Vy, hy, self._tmp2)
        gradV2 = self._dxA**2 + self._dyA**2 + self._tmp1**2 + self._tmp2**2
        heat_visc = (2.0/(3.0*np.maximum(n0, 1e-8))) * (self.eta * gradV2)

        heat_Q = (2.0/(3.0*np.maximum(n0, 1e-8))) * (self.Q0 * np.ones_like(n0))

        out_rhs[3, :, :] = -advT + diffT - (2.0/3.0)*T0*divV + heat_visc + heat_Q