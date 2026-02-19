# simuleringer/2D/main.py

import numpy as np
import numba
import matplotlib.pyplot as plt


from matplotlib.widgets import Slider

from kerne.grid import Grid2D
from kerne.integrators import simulate
# vælg ét system ad gangen:
from pdeer.dobbelt_test import CoupledRDAdv2D
from pdeer.tripple_test import Coupled3RDAdv2D
# ... eller et generelt N-system du selv laver



def plot_fields_with_slider(ts: np.ndarray, Us: np.ndarray, field_names=None):
    """
    ts: (Nt,)
    Us: (Nt, m, nx, ny)
    field_names: list[str] længde m
    """
    Nt, m, nx, ny = Us.shape
    if field_names is None:
        field_names = [f"field {k}" for k in range(m)]

    # vælg layout automatisk (pænt grid)
    ncols = int(np.ceil(np.sqrt(m)))
    nrows = int(np.ceil(m / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    plt.subplots_adjust(bottom=0.18)

    # gør axes flad liste
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    ims = []
    for k in range(m):
        ax = axes[k]
        im = ax.imshow(Us[0, k], origin="lower", aspect="auto")
        ax.set_title(field_names[k])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    # slå evt. overskydende subplot-axes fra
    for k in range(m, len(axes)):
        axes[k].axis("off")

    ax_slider = plt.axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(ax_slider, "frame", 0, Nt - 1, valinit=0, valstep=1)

    def update(val):
        i = int(s.val)
        for k in range(m):
            ims[k].set_data(Us[i, k])
        fig.suptitle(f"t = {ts[i]:.4f}  |  frame {i}/{Nt-1}")
        fig.canvas.draw_idle()

    s.on_changed(update)
    update(0)
    plt.show()


def run(system, t0=0.0, t1=20.0, dt=None, save_every=10):
    # system forventes at have: grid, dt, m, field_names, U0(), rhs(...)
    U = system.U0()  # ghosted (m, nx+2, ny+2)
    print(f"[{getattr(system, 'name', 'PDE')}] m={system.m}, dt={system.dt:.4e}")

    ts, Us = simulate(system, U, t0=t0, t1=t1, dt=dt, save_every=save_every)
    plot_fields_with_slider(ts, Us, getattr(system, "field_names", None))

numba.set_num_threads(12)
if __name__ == "__main__":
    nx, ny, t1 = 600, 600, 20.0
    g = Grid2D(x_min=0, y_min=0, x_max=50, y_max=50, nx=nx, ny=ny, ghost=1, dtype=np.float32)

    # --- vælg system ---
    # 2 felter:
    # system = CoupledRDAdv2D(grid=g, cx=1.0, cy=0.6, Du=0.05, Dv=0.02, alpha=0.8, beta=0.6)

    # 3 felter:
    system = Coupled3RDAdv2D(grid=g, cx=1.0, cy=0.6, Du=0.05, Dv=0.03, Dw=0.02)


    run(system, t0=0.0, t1=t1, dt=None, save_every=1)
