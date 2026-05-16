from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit, prange

from .integrator import RK4Integrator
from .grid import Grid2D
from .config import EngineConfig
from .types import PDESystem


@njit(parallel=True, fastmath=True, cache=True)
def _max_abs_parallel(arr: np.ndarray) -> float:
    """Parallel max(abs(arr)) over flattened data."""
    flat = arr.ravel()
    n = flat.size

    n_chunks = min(256, n)
    chunk_size = (n + n_chunks - 1) // n_chunks
    chunk_max = np.zeros(n_chunks, dtype=flat.dtype)

    for chunk_idx in prange(n_chunks):
        start = chunk_idx * chunk_size
        stop = min(start + chunk_size, n)
        local_max = 0.0
        for idx in range(start, stop):
            val = abs(flat[idx])
            if val > local_max:
                local_max = val
        chunk_max[chunk_idx] = local_max

    max_val = 0.0
    for chunk_idx in range(n_chunks):
        if chunk_max[chunk_idx] > max_val:
            max_val = float(chunk_max[chunk_idx])
    return max_val


@njit(fastmath=True, cache=True)
def _next_dt_from_scale(
    current_dt: float,
    scale: float,
    gradient_threshold: float,
    sensitivity: float,
    min_dt: float,
    max_dt: float,
    eps: float,
) -> float:
    """Compute next dt from a precomputed scale."""
    if scale < eps: factor = 1.2
    else: factor = (gradient_threshold / (scale + eps)) ** sensitivity

    if factor < 0.5: factor = 0.5
    elif factor > 1.2: factor = 1.2

    new_dt = current_dt * factor
    if new_dt < min_dt: return min_dt
    if new_dt > max_dt: return max_dt
    return new_dt


@dataclass
class AdaptiveStepController:
    """
    Adaptiv tidsstyring baseret på PDE'ens tidslige hældning rhs = du/dt.

    Klassen:
    1. evaluerer rhs(t, u, ...)
    2. måler størrelsen af rhs
    3. opdaterer dt
    4. udfører et RK4-step med det nye dt
    """
    grid: Grid2D
    system: PDESystem
    cfg: EngineConfig
    integrator: RK4Integrator = field(init=False, repr=False)
    _rhs_buf: np.ndarray = field(init=False, repr=False)
    _out_a: np.ndarray = field(init=False, repr=False)
    _out_b: np.ndarray = field(init=False, repr=False)
    _use_a: bool = field(default=True, init=False, repr=False)
    
    def __post_init__(self) -> None:
        self.integrator = RK4Integrator(grid=self.grid, rhs=self.system.rhs)
        self._rhs_buf = self.grid.alloc()
        self._out_a = self.grid.alloc()
        self._out_b = self.grid.alloc()

    def next_dt(
        self,
        current_dt: float,
        t: float,
        u: np.ndarray,
        *rhs_args: Any,
        **rhs_kwargs: Any,
    ) -> float:
        self.system.rhs(t, u, self._rhs_buf, *rhs_args, **rhs_kwargs)
        scale = _max_abs_parallel(self._rhs_buf)

        return _next_dt_from_scale(
            current_dt=current_dt,
            scale=scale,
            gradient_threshold=self.cfg.gradient_threshold,
            sensitivity=self.cfg.sensitivity,
            min_dt=self.cfg.min_dt,
            max_dt=self.cfg.max_dt,
            eps=self.cfg.eps,
        )

    def step(
        self,
        t: float,
        u: np.ndarray,
        current_dt: float | None = None,
        *rhs_args: Any,
        **rhs_kwargs: Any,
    ) -> tuple[np.ndarray, float]:
        if current_dt is None:
            current_dt = self.cfg.dt
        dt = self.next_dt(current_dt=current_dt, t=t, u=u, *rhs_args, **rhs_kwargs)

        # Ping-pong output buffers so we avoid allocating every time step.
        u_next = self._out_a if self._use_a else self._out_b
        if u_next is u:
            u_next = self._out_b if self._use_a else self._out_a

        self.integrator.step(t=t, dt=dt, u=u, out=u_next, *rhs_args, **rhs_kwargs)
        self._use_a = not self._use_a
        return u_next, dt