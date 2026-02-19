import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from kerne.grid import Grid2D
from kerne.integrators import simulate


if __name__ == "__main__":
    g = Grid2D(nx=200, ny=200, x_max=50, y_max=50, ghost=1, dtype=np.float32)
    system = Coupled3RDAdv2D(grid=g, cx=1.0, cy=0.6, Du=0.05, Dv=0.03, Dw=0.02)

    U = system.initial_condition()
    print("dt =", system.dt)

    ts, Us = simulate(system, U, t0=0.0, t1=20.0, dt=None, save_every=10)  # dt kommer fra system.dt

    # slider-plot: 3 felter side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(bottom=0.22)

    ims = []
    titles = ["u", "v", "w"]
    for k in range(3):
        im = axes[k].imshow(Us[0, k], origin="lower", aspect="auto")
        axes[k].set_title(titles[k])
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
        ims.append(im)

    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04])
    s = Slider(ax_slider, "frame", 0, len(ts)-1, valinit=0, valstep=1)

    def update(val):
        i = int(s.val)
        for k in range(3):
            ims[k].set_data(Us[i, k])
        fig.suptitle(f"t = {ts[i]:.3f}")
        fig.canvas.draw_idle()

    s.on_changed(update)
    update(0)
    plt.show()
