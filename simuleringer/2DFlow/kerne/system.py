from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kerne.operators import boundery, dx, dy, laplace


@dataclass
class TwoSpecies2D_FocusToCenter_HeatReaction:
    """
    4-felts system:
      U = [n1, n2, T, r]  (m=4)

    - n1, n2 advectes mod midten i x af et simpelt flow-felt (fokusering)
    - T varmes op hvor n1 og n2 overlapper: Q_heat = alpha_heat * n1*n2
    - r (reaktionsprodukt/blanding) produceres hvor n1 og n2 overlapper: beta_react*n1*n2
      og kan også "forbruge" n1/n2 gennem -beta_react*n1*n2.

    Numeriske stabilitetsgreb:
    - floors på n og T
    - køling/relaksation af T mod T0
    - diffusion på alle felter
    """

    grid: object
    config: object

    # ---------- stabilitet / floors ----------
    n_floor: float = 1e-10
    T_floor: float = 1e-8
    r_floor: float = 0.0

    # ---------- flow (fokusering) ----------
    k_focus: float = 0.15       # styrke mod x-midte
    y_boost: float = 0.8        # acceleration i +y tæt på midten
    y_eps: float = 2.0          # blødgør singularitet i vy

    # ---------- diffusion ----------
    D1: float = 2e-3
    D2: float = 2e-3
    kappaT: float = 5e-3
    Dr: float = 2e-3

    # ---------- varme / reaktion ----------
    alpha_heat: float = 0.08    # hvor meget overlap giver varme
    beta_react: float = 0.02    # hvor hurtigt n1*n2 -> r
    heat_sat: float = 10.0      # (valgfri) mætter Q_heat for store n
    use_saturation: bool = True

    # ---------- temperaturkøling ----------
    T0: float = 0.0             # baggrundstemperatur
    gamma_cool: float = 0.02    # relaksation mod T0

    # ---------- r tab ----------
    lambda_r: float = 0.01      # henfald af r

    # ---------- kilder (valgfrit, små gaussianer) ----------
    S1_0: float = 2e-3
    S2_0: float = 2e-3
    tau_src: float = 30.0

    src1_x: float = 18.0
    src1_y: float = 32.0
    src2_x: float = 46.0
    src2_y: float = 32.0
    src_sigma: float = 2.5

    # dt (fast) – hvis du allerede autovælger dt i dit setup kan du sætte den der
    dt: float = 0.05

    field_names = ("n1", "n2", "T", "r")

    def __post_init__(self):
        g = self.grid
        # scratch arrays (interior størrelse)
        nx, ny = self.config.nx, self.config.ny
        self._tmpx = np.zeros((nx, ny), dtype=self.config.dtype)
        self._tmpy = np.zeros((nx, ny), dtype=self.config.dtype)
        self._lap  = np.zeros((nx, ny), dtype=self.config.dtype)
        self._lap2 = np.zeros((nx, ny), dtype=self.config.dtype)
        self._gx   = np.zeros((nx, ny), dtype=self.config.dtype)
        self._gy   = np.zeros((nx, ny), dtype=self.config.dtype)

        # mesh (interior)
        X, Y = g.XY()  # (nx, ny)
        self._X = X.astype(self.config.dtype, copy=False)
        self._Y = Y.astype(self.config.dtype, copy=False)

        self._xc = 0.5 * (self.config.x_min + self.config.x_max)

    def initial_condition(self) -> np.ndarray:
        nx, ny = self.config.nx, self.config.ny
        U = np.zeros((4, nx + 2, ny + 2), dtype=self.config.dtype)

        # små start-peaks tæt på kilderne så man ser varme hurtigt
        X, Y = self._X, self._Y

        def gauss(x0, y0, s):
            return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * s * s))

        U[0, 1:-1, 1:-1] = 0.2 * gauss(self.src1_x, self.src1_y, self.src_sigma)
        U[1, 1:-1, 1:-1] = 0.2 * gauss(self.src2_x, self.src2_y, self.src_sigma)
        U[2, 1:-1, 1:-1] = self.T0
        U[3, 1:-1, 1:-1] = 0.0

        # periodic BC
        for k in range(4):
            boundery(U[k])
        return U

    def _sources(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Små gauss-kilder der tænder blødt: S(t) = S0*(1-exp(-t/tau))
        """
        X, Y = self._X, self._Y
        amp = (1.0 - np.exp(-t / max(self.tau_src, 1e-12))).astype(self.config.dtype)

        def gauss(x0, y0, s):
            return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * s * s))

        S1 = (self.S1_0 * amp) * gauss(self.src1_x, self.src1_y, self.src_sigma)
        S2 = (self.S2_0 * amp) * gauss(self.src2_x, self.src2_y, self.src_sigma)
        return S1.astype(self.config.dtype, copy=False), S2.astype(self.config.dtype, copy=False)

    def rhs(self, U: np.ndarray, out: np.ndarray, t: float) -> None:
        """
        out shape: (4, nx, ny)  (interior)
        U shape:   (4, nx+2, ny+2) (ghosted)
        """
        g = self.grid
        hx, hy = g.dx, g.dy

        # anvend BC på alle felter
        for k in range(4):
            boundery(U[k])

        n1g, n2g, Tg, rg = U[0], U[1], U[2], U[3]
        n1 = n1g[1:-1, 1:-1]
        n2 = n2g[1:-1, 1:-1]
        T  = Tg[1:-1, 1:-1]
        r  = rg[1:-1, 1:-1]

        # floors (undgå negative/0 i koblinger)
        n1c = np.maximum(n1, self.n_floor)
        n2c = np.maximum(n2, self.n_floor)
        Tc  = np.maximum(T,  self.T_floor)
        rc  = np.maximum(r,  self.r_floor)

        # flowfelt: fokus mod midten i x + vy boost nær midten
        X = self._X
        dxm = (X - self._xc)
        vx = -self.k_focus * dxm
        vy = self.y_boost / (self.y_eps + dxm * dxm)

        # kilder
        S1, S2 = self._sources(t)

        # overlap -> reaktion/varme
        overlap = n1c * n2c
        if self.use_saturation:
            # blød mætning så Q ikke eksploderer ved store n
            overlap_eff = overlap / (1.0 + overlap / max(self.heat_sat, 1e-12))
        else:
            overlap_eff = overlap

        Q_heat = self.alpha_heat * overlap_eff
        R_prod = self.beta_react * overlap_eff

        # ---------------------------
        # n1: ∂t n1 = -v·∇n1 + D1 Δn1 + S1 - R_prod
        # ---------------------------
        dx(n1g, hx, self._tmpx)
        dy(n1g, hy, self._tmpy)
        adv1 = vx * self._tmpx + vy * self._tmpy

        laplace(n1g, hx, hy, self._lap, self._lap2)

        out[0] = (-adv1) + self.D1 * self._lap + S1 - R_prod

        # ---------------------------
        # n2: ∂t n2 = -v·∇n2 + D2 Δn2 + S2 - R_prod
        # ---------------------------
        dx(n2g, hx, self._tmpx)
        dy(n2g, hy, self._tmpy)
        adv2 = vx * self._tmpx + vy * self._tmpy

        laplace(n2g, hx, hy, self._lap, self._lap2)

        out[1] = (-adv2) + self.D2 * self._lap + S2 - R_prod

        # ---------------------------
        # T:  ∂t T  = -v·∇T  + κ ΔT + Q_heat - gamma*(T - T0)
        # ---------------------------
        dx(Tg, hx, self._tmpx)
        dy(Tg, hy, self._tmpy)
        advT = vx * self._tmpx + vy * self._tmpy

        laplace(Tg, hx, hy, self._lap, self._lap2)

        out[2] = (-advT) + self.kappaT * self._lap + Q_heat - self.gamma_cool * (Tc - self.T0)

        # ---------------------------
        # r:  ∂t r  = -v·∇r + Dr Δr + R_prod - lambda_r*r
        # ---------------------------
        dx(rg, hx, self._tmpx)
        dy(rg, hy, self._tmpy)
        advR = vx * self._tmpx + vy * self._tmpy

        laplace(rg, hx, hy, self._lap, self._lap2)

        out[3] = (-advR) + self.Dr * self._lap + R_prod - self.lambda_r * rc