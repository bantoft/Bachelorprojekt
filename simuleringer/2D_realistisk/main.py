import numpy as np
import numba
from pathlib import Path

from kerne.config import Config
from kerne.grid import Grid
from kerne.system import TwoSpecies2D_FocusToCenter

from kerne.integrators import simulate
from kerne.plotter import save_data, read_data, plot_fields_with_slider




numba.set_num_threads(16)
if __name__ == "__main__":
    c = Config()
    g = Grid(c)
    system = TwoSpecies2D_FocusToCenter(
        grid=g, config=c,
        dt=2e-2,

        # kilder
        Se0=3e-2, Si0=3e-2,
        sigma_src=1.0,
        tau_src=0.3,
        src_yOff=12.0,

        # fokus + y-acceleration
        k_focus=0.12,
        k_y=0.80,
        w_y=7.0,

        # dæmpning
        nu_e=2e-2,
        nu_i=1e-2,
    )

    # save data
    if c.simulate:
        print(f'Simulere system med parametre: dt{system.dt}')
        ts, Us = simulate(c, system)
        save_data(c, ts, Us)
        plot_fields_with_slider(ts, Us, system)
    else:
        ts, Us = read_data(c)
        plot_fields_with_slider(ts, Us, system)
