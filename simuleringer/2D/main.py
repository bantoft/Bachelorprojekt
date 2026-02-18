import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

from setup_utils import Grid2D
from funktioner.diffision import SimpleSystem2D
from simulator import simulate


# TODO's
    # dt skal ind i system
grid = Grid2D(nx=400, ny=400)
system = SimpleSystem2D(grid, Da=0.05, Db=0.05, alpha=2.0, vx=0.2, vy=0.0)

a = grid.alloc_field(ghost=1)
b = grid.alloc_field(ghost=1)

X, Y = grid.XY()
grid.interior(a)[:] = np.exp(-((X-25.0)**2 + (Y-25.0)**2)/(2*2.0**2))
grid.interior(b)[:] = 0.0

# stabilitet (meget grov): advektion + diffusion
vmax = max(abs(system.vx), abs(system.vy), 1e-12)
dt_adv  = 0.4 * min(grid.dx, grid.dy) / vmax
dt_diff = 0.2 * min(grid.dx*grid.dx/system.Da, grid.dy*grid.dy/system.Da,
                    grid.dx*grid.dx/system.Db, grid.dy*grid.dy/system.Db)

dt = min(dt_adv, dt_diff)

ts, As, Bs = simulate(system, a, b, t0=0.0, t1=1000, dt=dt, save_every=10)

Nt = len(ts)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(bottom=0.22)  # plads til slider

# start index
i0 = 0
amax = np.max(np.abs(As))
bmax = np.max(np.abs(Bs))

im_a = ax[0].imshow(As[i0].T, origin="lower", vmin=-amax, vmax=amax)
ax[0].set_title(f"a(t={ts[i0]:.3f})")
fig.colorbar(im_a, ax=ax[0], fraction=0.046, pad=0.04)

im_b = ax[1].imshow(Bs[i0].T, origin="lower", vmin=-bmax, vmax=bmax)
ax[1].set_title(f"b(t={ts[i0]:.3f})")
fig.colorbar(im_b, ax=ax[1], fraction=0.046, pad=0.04)

# Slider-akse
ax_slider = plt.axes([0.15, 0.08, 0.70, 0.04])
slider = Slider(
    ax=ax_slider,
    label="t-index",
    valmin=0,
    valmax=Nt-1,
    valinit=i0,
    valstep=1
)

def update(val):
    i = int(slider.val)
    im_a.set_data(As[i].T)
    im_b.set_data(Bs[i].T)
    ax[0].set_title(f"a(t={ts[i]:.3f})")
    ax[1].set_title(f"b(t={ts[i]:.3f})")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
