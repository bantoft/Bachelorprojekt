# Integrators for finite difference methods on a grid

from __future__ import annotations
import numpy as np
import os
import time
from pathlib import Path

def rk4_step(system, U: np.ndarray, work: dict, t: float) -> None:
    dt = system.dt
    U0 = U[:, 1:-1, 1:-1]
    U_tmp = work["U_tmp"]
    U_tmp0 = U_tmp[:, 1:-1, 1:-1]
    k1, k2, k3, k4 = work["k1"], work["k2"], work["k3"], work["k4"]

    system.rhs(U,     k1, t)
    U_tmp0[:] = U0 + 0.5*dt*k1
    system.rhs(U_tmp, k2, t + 0.5*dt)

    U_tmp0[:] = U0 + 0.5*dt*k2
    system.rhs(U_tmp, k3, t + 0.5*dt)

    U_tmp0[:] = U0 + dt*k3
    system.rhs(U_tmp, k4, t + dt)

    U0[:] = U0 + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

def simulate(config: object, system: object):
    dt = system.dt
    nx, ny = config.nx, config.ny
    m = system.initial_condition().shape[0]
    U = system.initial_condition()
    save_every = max(int(config.save_every), 1)
    update_every = max(int(config.update_every), 1)
    flush_every_pct = max(int(getattr(config, "flush_every_pct", 10)), 1)

    nsteps = int(np.ceil((config.t1 - config.t0) / dt))
    next_progress_pct = 0
    next_flush_pct = 0
    print(nsteps)


    work = {
        "k1": np.zeros((m, nx, ny), dtype=config.dtype),
        "k2": np.zeros((m, nx, ny), dtype=config.dtype),
        "k3": np.zeros((m, nx, ny), dtype=config.dtype),
        "k4": np.zeros((m, nx, ny), dtype=config.dtype),
        "U_tmp": np.zeros_like(U),
    }

    max_snaps = nsteps // save_every + 2
    if getattr(config, "save_path", None):
        tmp_dir = Path(config.save_path).parent
    else:
        tmp_dir = Path("data")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{int(time.time() * 1e6)}_{os.getpid()}"
    ts_tmp_path = tmp_dir / f"sim_ts_tmp_{run_id}.dat"
    us_tmp_path = tmp_dir / f"sim_Us_tmp_{run_id}.dat"

    ts = np.memmap(ts_tmp_path, mode="w+", dtype=float, shape=(max_snaps,))
    Us = np.memmap(us_tmp_path, mode="w+", dtype=config.dtype, shape=(max_snaps, m, nx, ny))
    saved = 0

    t = float(config.t0)

    for n in range(nsteps + 1):
        if n % save_every == 0:
            ts[saved] = t
            Us[saved] = U[:, 1:-1, 1:-1]
            saved += 1

        pct = int(100.0 * n / max(nsteps, 1))
        if (pct >= next_flush_pct) or (n == nsteps):
            ts.flush()
            Us.flush()
            while next_flush_pct <= pct:
                next_flush_pct += flush_every_pct

        if (pct >= next_progress_pct) or (n == nsteps):
            print(f"[simulate] {pct:6.2f}%  t={t:.6g}", flush=True)
            while next_progress_pct <= pct:
                next_progress_pct += update_every

        if t >= config.t1: break

        rk4_step(system, U, work, t)
        t += dt

    ts.flush()
    Us.flush()
    return ts[:saved], Us[:saved]