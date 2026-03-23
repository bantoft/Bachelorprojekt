import numpy as np
import numba

from kerne.grid import Grid2D
from kerne.integrators import simulate
from pdeer.transport import Transport
from pdeer.momentum import Momentum
from pdeer.temperatur import Temperatur
from pdeer.systemet import Braginskii2D
from pdeer.system_2arter import Braginskii2Species2D

from kerne.run_plot import  plot_fields_with_slider

numba.set_num_threads(15)
if __name__ == "__main__":
    load_saved_data = True
    saved_data_path = "data/min_koersel.npz"

    nx, ny, t1 = 128, 128, 5.0

    g = Grid2D(x_min=0, y_min=0, x_max=32, y_max=32, nx=nx, ny=ny, dtype=np.float32)
    system = Braginskii2Species2D(
        grid=g,
        m_e=1.0, m_i=2.0,
        q_e=-1.0, q_i=+1.0,
        nu_e=10, nu_i=10,
        kappa_e=10, kappa_i=10,
        eta_e=10, eta_i=10,
        Ex=5, Ey=5, Bz=1.0,
        nu_ei=2,
        nu_T=5,
    )

    print(f"[{system.name}] m={system.m}, dt={system.dt:.3e}")
    if load_saved_data:
        data = np.load(saved_data_path)
        ts = data["ts"]
        Us = data["Us"]
        plot_fields_with_slider(ts, Us, system.field_names, save_data=False)
    else:
        ts, Us = simulate(system, t0=0.0, t1=t1, save_every=10, progress_percent_every=5)
        plot_fields_with_slider(
            ts,
            Us,
            system.field_names,
            save_data=True,
            save_path=saved_data_path,
        )
