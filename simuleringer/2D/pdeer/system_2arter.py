from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid2D
from kerne.operators import boundery, dx, dy, laplace


@dataclass
class Braginskii2Species2D:
    """
    To-arts (e,i) forenklet Braginskii-lignende 2D system.

    State:
      U = [ne, Vex, Vey, Te,  ni, Vix, Viy, Ti]   (m=8)
      ghosted shape (8, nx+2, ny+2)

    Kontinuitet for hver art:
      ∂t n_s = - div(n_s V_s)

    Momentum for hver art (forenklet):
      ∂t V_s = - (V_s·∇)V_s - (1/(m_s n_s)) ∇p_s + nu_s ∆V_s + (q_s/m_s)(E + (1/c) V_s×B) + R_s/(m_s n_s)

      p_s = n_s T_s

    Varme (forenklet temperatur-form):
      ∂t T_s = - V_s·∇T_s + (kappa_s/n_s) ∆T_s - (2/3) T_s div V_s + (2/(3 n_s))(eta_s ||∇V_s||^2 + Q_s) + heat_exchange

    Kobling:
      R_e/(m_e n_e) = nu_ei (V_i - V_e)
      R_i/(m_i n_i) = -(m_e n_e)/(m_i n_i) nu_ei (V_i - V_e)

      varme-udveksling (simpel):
        Te_t += nu_T (Ti - Te)
        Ti_t += nu_T (Te - Ti)
    """

    grid: Grid2D

    # --- masser og ladninger (nondim e=1) ---
    m_e: float = 1.0
    m_i: float = 1836.0
    q_e: float = -1.0      # elektron
    q_i: float = +1.0      # ion (Z=1)
    c: float = 1.0

    # --- transportkoefficienter pr art ---
    nu_e: float = 1e-2
    nu_i: float = 1e-2
    kappa_e: float = 1e-2
    kappa_i: float = 1e-2
    eta_e: float = 1e-2
    eta_i: float = 1e-2

    # --- EM felter (fælles) ---
    Ex: float = 0.0
    Ey: float = 0.0
    Bz: float = 0.0  # antag B=(0,0,Bz)

    # --- inter-species kobling ---
    nu_ei: float = 0.0   # momentum-friktion
    nu_T: float = 0.0    # temperatur-relaxation

    # --- kilder (kan udvides til funktioner af (x,y,t)) ---
    Qe0: float = 0.0
    Qi0: float = 0.0

    # --- partikel-kilder (kan være None eller callable) ---
    Sn_e: float = 0.0   # konstant hvis du vil starte simpelt
    Sn_i: float = 0.0

    # injektionshastighed (til momentum source)
    Vinj_ex: float = 0.0
    Vinj_ey: float = 0.0
    Vinj_ix: float = 0.0
    Vinj_iy: float = 0.0

    # --- tidsstep ---
    C_adv: float = 0.25
    C_diff: float = 0.20
    dt: float | None = None

    name: str = "Braginskii2Species2D (simplified)"
    field_names = ["ne","Vex","Vey","Te","ni","Vix","Viy","Ti"]

    @property
    def m(self) -> int:
        return 8

    def __post_init__(self):
        if self.dt is None:
            h = min(self.grid.dx, self.grid.dy)
            Vmax = 1.0
            dt_adv = self.C_adv * h / max(Vmax, 1e-12)
            dt_diff = self.C_diff * h*h / max(self.nu_e, self.nu_i, self.kappa_e, self.kappa_i, 1e-12)
            self.dt = float(min(dt_adv, dt_diff))

        nx, ny = self.grid.nx, self.grid.ny
        dtype = self.grid.dtype

        # fælles scratch (interior)
        self._a = np.zeros((nx, ny), dtype=dtype)
        self._b = np.zeros((nx, ny), dtype=dtype)
        self._c = np.zeros((nx, ny), dtype=dtype)
        self._d = np.zeros((nx, ny), dtype=dtype)

        self._dxA = np.zeros((nx, ny), dtype=dtype)
        self._dyA = np.zeros((nx, ny), dtype=dtype)

        self._lap = np.zeros((nx, ny), dtype=dtype)
        self._lap2 = np.zeros((nx, ny), dtype=dtype)

        # ghosted scratch til flux/tryk (genbruges for hver art)
        self._p = np.zeros((nx + 2, ny + 2), dtype=dtype)
        self._fx = np.zeros((nx + 2, ny + 2), dtype=dtype)
        self._fy = np.zeros((nx + 2, ny + 2), dtype=dtype)

    def initial_condition(self) -> np.ndarray:
        nx, ny = self.grid.nx, self.grid.ny
        U = np.zeros((8, nx + 2, ny + 2), dtype=self.grid.dtype)

        X, Y = self.grid.XY()
        xmid = 0.5*(self.grid.x_max - self.grid.x_min)
        ymid = 0.5*(self.grid.y_max - self.grid.y_min)

        bump = np.exp(-((X-xmid)**2 + (Y-ymid)**2)/(2.0*(2.0**2))).astype(self.grid.dtype)

        # ne, ni næsten ens
        U[0, 1:-1, 1:-1] = 1.0 + 0.05*bump
        U[4, 1:-1, 1:-1] = 1.0 + 0.05*bump

        # hastigheder
        U[1, 1:-1, 1:-1] = 0.0
        U[2, 1:-1, 1:-1] = 0.0
        U[5, 1:-1, 1:-1] = 0.0
        U[6, 1:-1, 1:-1] = 0.0

        # temperaturer
        U[3, 1:-1, 1:-1] = 1.0
        U[7, 1:-1, 1:-1] = 1.0

        for k in range(8):
            boundery(U[k])
        return U

    # ---------- små hjælpe-funktioner ----------
    def _continuity_rhs(self, n, Vx, Vy, hx, hy, out):
        """out = -div(nV)"""
        n0 = n[1:-1, 1:-1]
        Vx0 = Vx[1:-1, 1:-1]
        Vy0 = Vy[1:-1, 1:-1]
        self._fx[1:-1, 1:-1] = n0 * Vx0
        self._fy[1:-1, 1:-1] = n0 * Vy0
        boundery(self._fx); boundery(self._fy)
        dx(self._fx, hx, self._a)
        dy(self._fy, hy, self._b)
        out[:, :] = -(self._a + self._b)

    def _pressure_grads(self, n, T, hx, hy):
        """return (dpdx, dpdy) i self._a,self._b via p=nT"""
        self._p[1:-1, 1:-1] = (n[1:-1, 1:-1] * T[1:-1, 1:-1])
        boundery(self._p)
        dx(self._p, hx, self._a)  # dp/dx
        dy(self._p, hy, self._b)  # dp/dy
        return self._a, self._b

    def _convection(self, Vx, Vy, hx, hy):
        """return convVx, convVy via (V·∇)V i self._a,self._b"""
        Vx0 = Vx[1:-1, 1:-1]
        Vy0 = Vy[1:-1, 1:-1]

        dx(Vx, hx, self._dxA)   # dVx/dx
        dy(Vx, hy, self._dyA)   # dVx/dy
        convVx = Vx0*self._dxA + Vy0*self._dyA

        dx(Vy, hx, self._c)     # dVy/dx
        dy(Vy, hy, self._d)     # dVy/dy
        convVy = Vx0*self._c + Vy0*self._d

        self._a[:, :] = convVx
        self._b[:, :] = convVy
        return self._a, self._b

    def rhs(self, U: np.ndarray, out_rhs: np.ndarray) -> None:
        """
        U: (8, nx+2, ny+2) ghosted
        out_rhs: (8, nx, ny) interior
        """
        g = self.grid
        hx, hy = g.dx, g.dy

        # unpack
        ne, Vex, Vey, Te = U[0], U[1], U[2], U[3]
        ni, Vix, Viy, Ti = U[4], U[5], U[6], U[7]

        # BC
        for A in (ne, Vex, Vey, Te, ni, Vix, Viy, Ti):
            boundery(A)

        ne0 = ne[1:-1, 1:-1]; ni0 = ni[1:-1, 1:-1]
        Vex0 = Vex[1:-1, 1:-1]; Vey0 = Vey[1:-1, 1:-1]
        Vix0 = Vix[1:-1, 1:-1]; Viy0 = Viy[1:-1, 1:-1]
        Te0 = Te[1:-1, 1:-1];   Ti0 = Ti[1:-1, 1:-1]

        # ---------- kontinuitet ----------
        self._continuity_rhs(ne, Vex, Vey, hx, hy, out_rhs[0])
        self._continuity_rhs(ni, Vix, Viy, hx, hy, out_rhs[4])


        # ---------- trykgradienter ----------
        dpedx, dpedy = self._pressure_grads(ne, Te, hx, hy)
        dpidx, dpidy = self._pressure_grads(ni, Ti, hx, hy)

        inv_me_ne = 1.0 / (self.m_e * np.maximum(ne0, 1e-8))
        inv_mi_ni = 1.0 / (self.m_i * np.maximum(ni0, 1e-8))

        # ---------- konvektion ----------
        conv_ex, conv_ey = self._convection(Vex, Vey, hx, hy)
        conv_ix, conv_iy = self._convection(Vix, Viy, hx, hy)

        # ---------- viskositet (Laplace V) ----------
        laplace(Vex, hx, hy, self._lap, self._lap2); visc_ex = self.nu_e * self._lap.copy()
        laplace(Vey, hx, hy, self._lap, self._lap2); visc_ey = self.nu_e * self._lap.copy()
        laplace(Vix, hx, hy, self._lap, self._lap2); visc_ix = self.nu_i * self._lap.copy()
        laplace(Viy, hx, hy, self._lap, self._lap2); visc_iy = self.nu_i * self._lap.copy()

        # ---------- EM acceleration ----------
        # V×B for B=(0,0,Bz): (Vy*Bz, -Vx*Bz, 0)
        aex_em = (self.q_e/self.m_e) * (self.Ex + (1.0/self.c)*(Vey0*self.Bz))
        aey_em = (self.q_e/self.m_e) * (self.Ey + (1.0/self.c)*(-Vex0*self.Bz))

        aix_em = (self.q_i/self.m_i) * (self.Ex + (1.0/self.c)*(Viy0*self.Bz))
        aiy_em = (self.q_i/self.m_i) * (self.Ey + (1.0/self.c)*(-Vix0*self.Bz))

        # ---------- interspecies friktion ----------
        dVx = (Vix0 - Vex0)
        dVy = (Viy0 - Vey0)

        # e: +nu_ei (Vi - Ve)
        Rex_acc_x = self.nu_ei * dVx
        Rex_acc_y = self.nu_ei * dVy

        # i: -(m_e n_e)/(m_i n_i) nu_ei (Vi - Ve)
        weight = (self.m_e*np.maximum(ne0, 1e-8)) / (self.m_i*np.maximum(ni0, 1e-8))
        Rix_acc_x = -weight * self.nu_ei * dVx
        Rix_acc_y = -weight * self.nu_ei * dVy

        # ---------- momentum RHS ----------
        out_rhs[1][:, :] = -conv_ex - inv_me_ne*dpedx + visc_ex + aex_em + Rex_acc_x
        out_rhs[2][:, :] = -conv_ey - inv_me_ne*dpedy + visc_ey + aey_em + Rex_acc_y
        out_rhs[5][:, :] = -conv_ix - inv_mi_ni*dpidx + visc_ix + aix_em + Rix_acc_x
        out_rhs[6][:, :] = -conv_iy - inv_mi_ni*dpidy + visc_iy + aiy_em + Rix_acc_y

        # ---------- momentum-injektion (SKAL ligge her) ----------
        if self.Sn_e != 0.0:
            fac_e = self.Sn_e / np.maximum(ne0, 1e-8)
            out_rhs[1][:, :] += fac_e * (self.Vinj_ex - Vex0)
            out_rhs[2][:, :] += fac_e * (self.Vinj_ey - Vey0)

        if self.Sn_i != 0.0:
            fac_i = self.Sn_i / np.maximum(ni0, 1e-8)
            out_rhs[5][:, :] += fac_i * (self.Vinj_ix - Vix0)
            out_rhs[6][:, :] += fac_i * (self.Vinj_iy - Viy0)

        # ---------- varme (temperatur) ----------
        # helper: advT = V·∇T, divV = ∂xVx+∂yVy, diff = (kappa/n)∆T, shear heating
        def temperature_rhs(n, Vx, Vy, T, kappa, eta, Q0, out):
            n0 = n[1:-1, 1:-1]
            Vx0 = Vx[1:-1, 1:-1]
            Vy0 = Vy[1:-1, 1:-1]
            T0 = T[1:-1, 1:-1]

            dx(T, hx, self._a)  # dT/dx
            dy(T, hy, self._b)  # dT/dy
            advT = Vx0*self._a + Vy0*self._b

            dx(Vx, hx, self._c)
            dy(Vy, hy, self._d)
            divV = self._c + self._d

            laplace(T, hx, hy, self._lap, self._lap2)
            diffT = (kappa / np.maximum(n0, 1e-8)) * self._lap

            # shear heating ~ eta ||∇V||^2
            dx(Vx, hx, self._dxA); dy(Vx, hy, self._dyA)
            dx(Vy, hx, self._a);   dy(Vy, hy, self._b)
            gradV2 = self._dxA**2 + self._dyA**2 + self._a**2 + self._b**2

            heat_visc = (2.0/(3.0*np.maximum(n0, 1e-8))) * (eta * gradV2)
            heat_Q    = (2.0/(3.0*np.maximum(n0, 1e-8))) * (Q0 * np.ones_like(n0))

            out[:, :] = -advT + diffT - (2.0/3.0)*T0*divV + heat_visc + heat_Q

        temperature_rhs(ne, Vex, Vey, Te, self.kappa_e, self.eta_e, self.Qe0, out_rhs[3])
        temperature_rhs(ni, Vix, Viy, Ti, self.kappa_i, self.eta_i, self.Qi0, out_rhs[7])

        # ---------- temperatur-udveksling ----------
        if self.nu_T != 0.0:
            out_rhs[3][:, :] += self.nu_T * (Ti0 - Te0)
            out_rhs[7][:, :] += self.nu_T * (Te0 - Ti0)