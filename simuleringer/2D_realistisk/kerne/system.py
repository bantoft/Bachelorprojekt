# system_focus.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.grid import Grid
from kerne.operators import boundery, dx, dy, laplace


@dataclass
class TwoSpecies2D_FocusToCenter:
    """
    State U (ghosted): [ne, Vex, Vey, ni, Vix, Viy]  => m=6
    """

    grid: Grid
    config: object

    # tidsstep (fast)
    dt: float = 2e-2

    # --- numeriske floors ---
    n_floor: float = 1e-8

    # --- kilder (små Gauss-cirkler) ---
    Se0: float = 2e-2          # styrke elektron-kilde
    Si0: float = 2e-2          # styrke ion-kilde
    tau_src: float = 0.4       # ramp/tændingstid
    sigma_src: float = 1.2     # radius (lille = skarp)

    # placering: "hver sin side af x-aksen" (y>0 og y<0)
    src_xL: float = 0.25       # relativt i domænet (0..1)
    src_xR: float = 0.75
    src_yOff: float = 10.0     # absolut y-offset (samme enhed som y)

    # --- forcing: fokus i x + y-accel mod +y ved center ---
    k_focus: float = 0.10      # styrke mod x_mid
    k_y: float = 0.50          # styrke mod +y nær midten
    w_y: float = 6.0           # bredde af center-zonen i x

    # --- svag viskositet på V (dæmper blow-ups) ---
    nu_e: float = 2e-2
    nu_i: float = 1e-2

    field_names = ["ne", "Vex", "Vey", "ni", "Vix", "Viy"]

    def __post_init__(self):
        self.x = self.grid.x
        self.y = self.grid.y
        self.dx_ = self.grid.dx
        self.dy_ = self.grid.dy

        X, Y = self.grid.XY()
        self.X = X
        self.Y = Y

        self.x_mid = 0.5 * (self.config.x_min + self.config.x_max)

        # kildernes centre
        xL = self.config.x_min + self.src_xL * (self.config.x_max - self.config.x_min)
        xR = self.config.x_min + self.src_xR * (self.config.x_max - self.config.x_min)
        y0 = 0.5 * (self.config.y_min + self.config.y_max)

        self.src_e = (xL, y0 + self.src_yOff)  # electron source above x-axis
        self.src_i = (xR, y0 - self.src_yOff)  # ion source below x-axis

        # prealloc scratch (interior)
        nx, ny = self.config.nx, self.config.ny
        self._tmp1 = np.zeros((nx, ny), dtype=self.config.dtype)
        self._tmp2 = np.zeros((nx, ny), dtype=self.config.dtype)
        self._tmp3 = np.zeros((nx, ny), dtype=self.config.dtype)

        # prealloc scratch (ghosted scalars)
        self._Fx = self.grid.alloc()
        self._Fy = self.grid.alloc()
        self._tmpg = self.grid.alloc()

    def initial_condition(self) -> np.ndarray:
        # ghosted: (m, nx+2, ny+2)
        U = np.zeros((6, self.config.nx + 2, self.config.ny + 2), dtype=self.config.dtype)

        # start med meget små densiteter (så kilder dominerer visuelt)
        U[0, 1:-1, 1:-1] = 1e-6  # ne
        U[3, 1:-1, 1:-1] = 1e-6  # ni

        # start med nul-hastigheder
        return U

    def _gauss2(self, x0: float, y0: float, sigma: float) -> np.ndarray:
        r2 = (self.X - x0)**2 + (self.Y - y0)**2
        return np.exp(-0.5 * r2 / (sigma*sigma))

    def rhs(self, U: np.ndarray, out: np.ndarray, t: float) -> None:
        """
        Fylder out med shape (m, nx, ny) for RK4. (interior)
        """
        g = self.grid
        nx, ny = self.config.nx, self.config.ny

        # unpack (ghosted)
        ne  = U[0]
        Vex = U[1]
        Vey = U[2]
        ni  = U[3]
        Vix = U[4]
        Viy = U[5]

        # apply periodic BC på alle felter (vigtigt før derivativer)
        boundery(ne);  boundery(Vex); boundery(Vey)
        boundery(ni);  boundery(Vix); boundery(Viy)

        # interior views
        ne0  = ne[1:-1, 1:-1]
        Vex0 = Vex[1:-1, 1:-1]
        Vey0 = Vey[1:-1, 1:-1]
        ni0  = ni[1:-1, 1:-1]
        Vix0 = Vix[1:-1, 1:-1]
        Viy0 = Viy[1:-1, 1:-1]

        # floors (undgå division/NaN senere)
        ne_eff = np.maximum(ne0, self.n_floor)
        ni_eff = np.maximum(ni0, self.n_floor)

        # ---------- kilder ----------
        ramp = 1.0 - np.exp(-t / max(self.tau_src, 1e-12))
        Se = self.Se0 * ramp * self._gauss2(self.src_e[0], self.src_e[1], self.sigma_src)
        Si = self.Si0 * ramp * self._gauss2(self.src_i[0], self.src_i[1], self.sigma_src)

        # ---------- kontinuitet: d_t n = -div(nV) + S ----------
        # electron flux
        self._Fx[1:-1, 1:-1] = ne_eff * Vex0
        self._Fy[1:-1, 1:-1] = ne_eff * Vey0
        boundery(self._Fx); boundery(self._Fy)

        dx(self._Fx, self.dx_, self._tmp1)  # tmp1 = d_x (ne*Vex)
        dy(self._Fy, self.dy_, self._tmp2)  # tmp2 = d_y (ne*Vey)
        div_e = self._tmp1 + self._tmp2

        out[0] = -div_e + Se

        # ion flux
        self._Fx[1:-1, 1:-1] = ni_eff * Vix0
        self._Fy[1:-1, 1:-1] = ni_eff * Viy0
        boundery(self._Fx); boundery(self._Fy)

        dx(self._Fx, self.dx_, self._tmp1)
        dy(self._Fy, self.dy_, self._tmp2)
        div_i = self._tmp1 + self._tmp2

        out[3] = -div_i + Si

        # ---------- forcing: fokus i x + accel mod +y nær center ----------
        # a_focus_x = -k_focus*(x-x_mid)
        ax_focus = -self.k_focus * (self.X - self.x_mid)

        # ay(x) = k_y * exp(-(x-x_mid)^2/(2 w_y^2))
        ay_center = self.k_y * np.exp(-0.5 * ((self.X - self.x_mid)/max(self.w_y, 1e-12))**2)

        # ---------- advektion på V: (V·∇)V ----------
        # helper: compute V·∇Vx og V·∇Vy for en given art
        def advect_velocity(Vx_g, Vy_g, Vx0, Vy0, outVx, outVy):
            # dVx/dx, dVx/dy
            dx(Vx_g, self.dx_, self._tmp1)
            dy(Vx_g, self.dy_, self._tmp2)
            outVx[:] = Vx0 * self._tmp1 + Vy0 * self._tmp2

            # dVy/dx, dVy/dy
            dx(Vy_g, self.dx_, self._tmp1)
            dy(Vy_g, self.dy_, self._tmp2)
            outVy[:] = Vx0 * self._tmp1 + Vy0 * self._tmp2

        # electrons: advection terms
        advVx_e = self._tmp1  # genbrug
        advVy_e = self._tmp2
        advect_velocity(Vex, Vey, Vex0, Vey0, advVx_e, advVy_e)

        # ions: advection terms (brug tmp3 + tmp2)
        advVx_i = self._tmp3
        advVy_i = self._tmp2  # ok at overskrive
        advect_velocity(Vix, Viy, Vix0, Viy0, advVx_i, advVy_i)

        # ---------- diffusion på V: nu * laplace(V) ----------
        laplace(Vex, self.dx_, self.dy_, self._tmp1, self._tmp2)  # tmp1 = ∆Vex
        diffVex = self._tmp1.copy()
        laplace(Vey, self.dx_, self.dy_, self._tmp1, self._tmp2)
        diffVey = self._tmp1.copy()

        laplace(Vix, self.dx_, self.dy_, self._tmp1, self._tmp2)
        diffVix = self._tmp1.copy()
        laplace(Viy, self.dx_, self.dy_, self._tmp1, self._tmp2)
        diffViy = self._tmp1.copy()

        # ---------- momentum RHS ----------
        # d_t V = -(V·∇)V + [ax_focus, ay_center] + nu ∆V
        out[1] = -advVx_e + ax_focus + self.nu_e * diffVex
        out[2] = -advVy_e + ay_center + self.nu_e * diffVey

        out[4] = -advVx_i + ax_focus + self.nu_i * diffVix
        out[5] = -advVy_i + ay_center + self.nu_i * diffViy