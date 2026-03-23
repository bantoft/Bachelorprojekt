# 2D/kerne/integrators.py

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
    system.rhs(U, k1)

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

def simulate(
    system,
    t0: float,
    t1: float,
    save_every: int = 10,
    progress_percent_every: int | None = 1,
):
    dt = system.dt
    nx, ny = system.grid.nx, system.grid.ny
    m = system.initial_condition().shape[0]
    U = system.initial_condition()

    nsteps = int(np.ceil((t1 - t0) / dt))

    # progress i procentpoint (f.eks. 1 => 1%, 2%, ..., 100%)
    if progress_percent_every is not None:
        progress_percent_every = max(1, int(progress_percent_every))
    next_progress_pct = 0

    work = {
        "k1": np.zeros((m, nx, ny), dtype=system.grid.dtype),
        "k2": np.zeros((m, nx, ny), dtype=system.grid.dtype),
        "k3": np.zeros((m, nx, ny), dtype=system.grid.dtype),
        "k4": np.zeros((m, nx, ny), dtype=system.grid.dtype),
        "U_tmp": np.zeros_like(U),
    }

    ts, Us = [], []
    t = float(t0)

    for n in range(nsteps + 1):
        if n % save_every == 0:
            ts.append(t)
            Us.append(U[:, 1:-1, 1:-1].copy())

        # progress (udskriv ved procent-milepæle)
        if progress_percent_every is not None:
            pct = 100.0 * n / max(nsteps, 1)
            pct_int = int(pct)
            if (pct_int >= next_progress_pct) or (n == nsteps):
                print(f"[simulate] {pct:6.2f}%  t={t:.6g}", flush=True)
                while next_progress_pct <= pct_int:
                    next_progress_pct += progress_percent_every

        if t >= t1:
            break

        rk4_step(system, U, dt, work)
        t += dt

    return np.asarray(ts), np.asarray(Us)