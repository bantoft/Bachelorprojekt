from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional

from setup import Grid1D, ddx_central


Array = np.ndarray


@dataclass
class Varme1D:
    """
    1D Braginskii varme-ligning (for en art s), skrevet som evolution for T:

        (3/2) n T_t + d/dx( (3/2) n T V ) + n T dV/dx + pi_xx dV/dx + d/dx q = Q

    Vi bruger simple lukninger (kan udskiftes):
        q = -kappa * dT/dx
        pi_xx = -eta  * dV/dx

    Bemærk: Dette er "single-field" i den forstand, at vi kun integrerer T.
           n og V kommer udefra (som funktioner af x,t) eller som konstante.
    """
    Vfun: Callable[[Array, float], Array]
    nfun: Optional[Callable[[Array, float], Array]] = None
    Qfun: Optional[Callable[[Array, float], Array]] = None

    # simple closures
    kappa: float = 0.0   # varmeledning
    eta: float = 0.0     # viskositet til pi_xx = -eta dV/dx

    # initial condition for T
    T0: float = 1.0
    x0: float = 0.30
    sigma: float = 0.05
    bump: float = 0.2

    name: str = "Braginskii Heat 1D (single-field T)"

    def initial_condition(self, x: Array) -> Array:
        # baseline + lille gaussian bump
        return self.T0 + self.bump * np.exp(-((x - self.x0) ** 2) / (2.0 * self.sigma ** 2))

    def rhs(self, T: Array, t: float, grid: Grid1D) -> Array:
        x = grid.x
        dx = grid.dx

        V = self.Vfun(x, t)
        dVdx = ddx_central(V, dx)

        # n(x,t): enten udefra eller konstant 1
        if self.nfun is None:
            n = np.ones_like(x)
        else:
            n = self.nfun(x, t)

        # Q(x,t): enten 0 eller udefra
        if self.Qfun is None:
            Q = np.zeros_like(x)
        else:
            Q = self.Qfun(x, t)

        # --- term: d/dx( (3/2) n T V )
        adv_flux = 1.5 * n * T * V
        term_adv = ddx_central(adv_flux, dx)

        # --- term: n T dV/dx (kompressions-/ekspansionsterm)
        term_comp = n * T * dVdx

        # --- pi_xx dV/dx med pi_xx = -eta dV/dx  => term_pi = -(eta)(dVdx)^2
        pi_xx = -self.eta * dVdx
        term_pi = pi_xx * dVdx

        # --- d/dx q med q = -kappa dT/dx
        if self.kappa != 0.0:
            dTdx = ddx_central(T, dx)
            q = -self.kappa * dTdx
            term_q = ddx_central(q, dx)
        else:
            term_q = np.zeros_like(x)

        # undgå division med 0 hvis n kan være meget lille
        n_safe = np.maximum(n, 1e-12)

        # T_t = (2/(3n)) * (Q - term_adv - term_comp - term_pi - term_q)
        return (2.0 / (3.0 * n_safe)) * (Q - term_adv - term_comp - term_pi - term_q)
