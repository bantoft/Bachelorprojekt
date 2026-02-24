import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pde_system import Closure1DConst, BragSystem1D
from sim_step import simulate

# grid
Nx = 5000
x = np.linspace(0.0, 10.0, Nx, endpoint=False)
dx = x[1] - x[0]

# closures (konstanter)
closure = Closure1DConst(
    eta=1e-4,
    kappa=1e-2,
    nu=0.1,
    E0=0.0,
    Q0=0.0,
)

pde = BragSystem1D(x=x, closure=closure)

# initial conditions
n0 = 1.0 + 0.1*np.exp(-((x-0.3)**2)/(2*0.03**2))
V0 = 0.2*np.sin(2*np.pi*x)
T0 = 1.0 + 0.05*np.exp(-((x-0.7)**2)/(2*0.05**2))
y0 = pde.pack(n0, V0, T0)

# dt: CFL + diffusion
gamma = 5/3
cs0 = np.sqrt(gamma * np.maximum(T0, 1e-8))
amax0 = np.max(np.abs(V0) + cs0)

dt_adv = 0.2 * dx / max(amax0, 1e-8)

# nu/eta/kappa er nu bare konstanter:
kappa0 = closure.kappa
eta0   = closure.eta

dt_diff_T = 0.2 * dx*dx / max(kappa0, 1e-12)
dt_diff_V = 0.2 * dx*dx / max(eta0,   1e-12)

dt = min(dt_adv, dt_diff_T, dt_diff_V)

# simuler
ts, ys = simulate(pde.rhs, y0, t0=0.0, t1=0.5, dt=dt, save_every=20)

# unpack til arrays [ntimes, Nx]
Ns = np.array([pde.unpack(y)[0] for y in ys])
Vs = np.array([pde.unpack(y)[1] for y in ys])
Ts = np.array([pde.unpack(y)[2] for y in ys])

# plots
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
plt.subplots_adjust(bottom=0.18)

line_n, = axs[0].plot(x, Ns[0])
line_V, = axs[1].plot(x, Vs[0])
line_T, = axs[2].plot(x, Ts[0])

axs[0].set_ylabel("n")
axs[1].set_ylabel("V")
axs[2].set_ylabel("T")
axs[2].set_xlabel("x")

title = fig.suptitle(f"t = {ts[0]:.4f}")

# slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax=ax_slider, label="time index",
                valmin=0, valmax=len(ts)-1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    line_n.set_ydata(Ns[idx])
    line_V.set_ydata(Vs[idx])
    line_T.set_ydata(Ts[idx])
    title.set_text(f"t = {ts[idx]:.4f}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
