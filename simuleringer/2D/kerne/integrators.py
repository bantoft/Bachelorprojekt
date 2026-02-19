# simuleringer/2D_ny/kerne/integrators.py

from __future__ import annotations
import numpy as np

def rk4_step(system, U: np.ndarray, dt: float, work: dict) -> None:
    """
    Generel RK4 for m felter samlet i U.

    U:    (m, nx+2, ny+2) ghosted
    rhs:  system.rhs(U, out_rhs) hvor out_rhs er (m, nx, ny)

    work indeholder:
      k1,k2,k3,k4: (m, nx, ny)
      U_tmp:       (m, nx+2, ny+2)
      rhs_tmp:     (m, nx, ny)  (valgfri, men praktisk)
    """

    # views til interior (fysisk domæne)
    U0 = U[:, 1:-1, 1:-1]              # (m, nx, ny)
    U_tmp = work["U_tmp"]
    U_tmp0 = U_tmp[:, 1:-1, 1:-1]

    k1 = work["k1"]
    k2 = work["k2"]
    k3 = work["k3"]
    k4 = work["k4"]

    # k1
    system.rhs(U, k1)                  # k1 <- F(U)

    # k2: U + dt/2*k1
    U_tmp0[:] = U0 + 0.5*dt*k1
    system.rhs(U_tmp, k2)

    # k3
    U_tmp0[:] = U0 + 0.5*dt*k2
    system.rhs(U_tmp, k3)

    # k4
    U_tmp0[:] = U0 + dt*k3
    system.rhs(U_tmp, k4)

    # opdater interior
    U0[:] = U0 + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate(system,
             U: np.ndarray,
             t0: float,
             t1: float,
             dt: float | None = None,
             save_every: int = 10):
    """
    Simulerer fra t0 til t1.

    U: (m, nx+2, ny+2) ghosted, opdateres in-place
    Returnerer:
      ts: (Nt,)
      Us: (Nt, m, nx, ny)  snapshots af interior
    """
    g = system.grid
    if dt is None:
        dt = float(system.dt)

    nx, ny = g.nx, g.ny
    m = U.shape[0]

    nsteps = int(np.ceil((t1 - t0) / dt))

    work = {
        "k1": np.zeros((m, nx, ny), dtype=U.dtype),
        "k2": np.zeros((m, nx, ny), dtype=U.dtype),
        "k3": np.zeros((m, nx, ny), dtype=U.dtype),
        "k4": np.zeros((m, nx, ny), dtype=U.dtype),
        "U_tmp": np.zeros_like(U),
    }

    ts = []
    Us = []

    t = float(t0)
    for n in range(nsteps + 1):
        if n % save_every == 0:
            ts.append(t)
            Us.append(U[:, 1:-1, 1:-1].copy())

        if t >= t1:
            break

        rk4_step(system, U, dt, work)
        t += dt

    return np.asarray(ts), np.asarray(Us)
