import numpy as np
import numba
from pathlib import Path

from kerne.config import Config
from kerne.grid import Grid
from kerne.system import TwoSpecies2D_FocusToCenter_HeatReaction

from kerne.integrators import simulate
from kerne.plotter import save_data, read_data, plot_fields_with_slider




numba.set_num_threads(16)
if __name__ == "__main__":
    c = Config()
    g = Grid(c)

    system = TwoSpecies2D_FocusToCenter_HeatReaction(grid=g, config=c)

    if c.simulate:
        print(f'Simulere system med parametre: dt{system.dt}')
        ts, Us = simulate(c, system)
        save_data(c, ts, Us)
        plot_fields_with_slider(ts, Us, system)
    else:
        ts, Us = read_data(c)
        plot_fields_with_slider(ts, Us, system)