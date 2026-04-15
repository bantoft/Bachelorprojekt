from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit, prange


from engine.grid import Grid2D
from engine.types import RHSFn

@njit(parallel=True, fastmath=True)
def lin_update(a: float, x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    """Beregn lineær opdatering in-place: out = y + a * x.

    Parametre:
    a: skalar faktor.
    x: input-array med samme shape som y og out.
    y: input-array med samme shape som x og out.
    out: output-array, overskrives med resultatet.
    """
    x_flat = x.ravel()
    y_flat = y.ravel()
    out_flat = out.ravel()
    for idx in prange(x_flat.size):
        out_flat[idx] = y_flat[idx] + a * x_flat[idx]


@njit(parallel=True, fastmath=True)
def rk4_finalize(
    u: np.ndarray,
    dt: float,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute out = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4) in-place."""
    u_flat = u.ravel()
    k1_flat = k1.ravel()
    k2_flat = k2.ravel()
    k3_flat = k3.ravel()
    k4_flat = k4.ravel()
    out_flat = out.ravel()

    c = dt / 6.0
    for idx in prange(u_flat.size):
        out_flat[idx] = u_flat[idx] + c * (
            k1_flat[idx] + 2.0 * k2_flat[idx] + 2.0 * k3_flat[idx] + k4_flat[idx]
        )


def rk4_step(
    t: float,
    dt: float,
    rhs: RHSFn,
    u: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    tmp: np.ndarray,
    out: np.ndarray,
    *rhs_args: Any,
    **rhs_kwargs: Any,
) -> np.ndarray:
    """Beregn næste RK4 trin for PDE system
    output = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Parametre:
    u: nuværende tilstand (array).
    engine_config: EngineConfig objekt med simuleringsparametre.
    rhs: funktion til at beregne højre side af PDE.
    out: output-array.
    k1, k2, k3, k4: midlertidige arrays til RK4-koefficienter.
    tmp: midlertidig array til beregninger.
    **rhs_kwargs: yderligere keyword-argumenter til rhs-funktionen."""
    rhs(t, u, k1, *rhs_args, **rhs_kwargs)

    half_dt = 0.5 * dt
    lin_update(half_dt, k1, u, tmp)
    rhs(t + 0.5 * dt, tmp, k2, *rhs_args, **rhs_kwargs)

    lin_update(half_dt, k2, u, tmp)
    rhs(t + 0.5 * dt, tmp, k3, *rhs_args, **rhs_kwargs)

    lin_update(dt, k3, u, tmp)
    rhs(t + dt, tmp, k4, *rhs_args, **rhs_kwargs)

    rk4_finalize(u, dt, k1, k2, k3, k4, out)
    return out


@dataclass
class RK4Integrator:
    """Integrationsklasse for RK4 metode.

    Denne klasse indeholder midlertidige arrays og en metode til at udføre RK4 integration.
    """
    grid: Grid2D
    rhs: RHSFn
    _k1: np.ndarray = field(init=False, repr=False)
    _k2: np.ndarray = field(init=False, repr=False)
    _k3: np.ndarray = field(init=False, repr=False)
    _k4: np.ndarray = field(init=False, repr=False)
    _tmp: np.ndarray = field(init=False, repr=False)
    _out: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._k1 = self.grid.alloc()
        self._k2 = self.grid.alloc()
        self._k3 = self.grid.alloc()
        self._k4 = self.grid.alloc()
        self._tmp = self.grid.alloc()
        self._out = self.grid.alloc()

    def step(
        self,
        t: float,
        dt: float,
        u: np.ndarray,
        out: np.ndarray | None = None,
        *rhs_args: Any,
        **rhs_kwargs: Any,
    ) -> np.ndarray:

        target = self._out if out is None else out
        return rk4_step(
            t=t,
            dt=dt,
            rhs=self.rhs,
            u=u,
            k1=self._k1,
            k2=self._k2,
            k3=self._k3,
            k4=self._k4,
            tmp=self._tmp,
            out=target,
            *rhs_args,
            **rhs_kwargs,
        )