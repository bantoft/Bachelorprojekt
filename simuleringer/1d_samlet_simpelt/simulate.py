# simulate.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pde_system import CoupledADR1D


def simulate(pde: CoupledADR1D, x: np.ndarray, t0: float, t1: float, dt: float):
    dx = x[1] - x[0]
    u, v = pde.initial_condition(x)

    ts = [t0]
    us = [u.copy()]
    vs = [v.copy()]

    t = t0
    nsteps = int(np.ceil((t1 - t0) / dt))

    for _ in range(nsteps):
        du, dv = pde.rhs(u, v, dx)

        # Eksplicit Euler
        u = u + dt * du
        v = v + dt * dv

        t += dt

        ts.append(t)
        us.append(u.copy())
        vs.append(v.copy())

    return np.array(ts), np.array(us), np.array(vs)


if __name__ == "__main__":
    # Grid
    nx = 400
    x0, x1 = 0.0, 1.0
    x = np.linspace(x0, x1, nx, endpoint=False)
    dx = x[1] - x[0]

    # PDE parametre
    pde = CoupledADR1D()

    # Stabilitet (eksplicit skema)
    C = 0.2
    dt_adv = C * dx / max(abs(pde.a), 1e-12)
    nu_max = max(pde.nu_u, pde.nu_v)
    dt_diff = C * dx * dx / max(nu_max, 1e-12)
    dt = min(dt_adv, dt_diff)

    t0, t1 = 0.0, 1.0
    ts, us, vs = simulate(pde, x, t0, 10.0, dt)

    # ======================
    # Interaktiv plotting
    # ======================

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.18)

    # Initial plot
    line_u, = ax1.plot(x, us[0])
    line_v, = ax2.plot(x, vs[0])

    ax1.set_title("u(x,t)")
    ax2.set_title("v(x,t)")

    ax1.set_ylabel("u")
    ax2.set_ylabel("v")
    ax2.set_xlabel("x")

    ax1.set_ylim(np.min(us), np.max(us))
    ax2.set_ylim(np.min(vs), np.max(vs))

    # Slider akse
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="t index",
        valmin=0,
        valmax=len(ts) - 1,
        valinit=0,
        valstep=1
    )

    # Update funktion
    def update(val):
        idx = int(slider.val)
        line_u.set_ydata(us[idx])
        line_v.set_ydata(vs[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
