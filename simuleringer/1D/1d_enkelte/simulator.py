# simulate.py
from __future__ import annotations
import numpy as np

from setup import Grid1D, rk4_step

from pde.transport import Transport1D
from pde.momentum import Momentum1D
from pde.varme import Varme1D

def simulate(pde, grid: Grid1D, t0: float, t1: float, dt: float, save_every: int = 10):
    u = pde.initial_condition(grid.x).astype(float)
    t = t0

    ts = [t]
    us = [u.copy()]

    def f(u_, t_): return pde.rhs(u_, t_, grid)

    nsteps = int(np.ceil((t1 - t0) / dt))
    for k in range(nsteps):
        u = rk4_step(f, u, t, dt)
        t = t + dt

        if (k + 1) % save_every == 0 or k == nsteps - 1:
            ts.append(t)
            us.append(u.copy())

    return np.array(ts), np.array(us), grid.x

def unpack_momentum_state(u: np.ndarray, N: int):
    n  = u[:N]
    Vx = u[N:2*N]
    Vy = u[2*N:3*N]
    Vz = u[3*N:4*N]
    return n, Vx, Vy, Vz

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    V0 = 1.1
    omega = 2*np.pi
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.22)
    i0 = 0


    # Transport simulering 

    pde = Transport1D(Vfun=lambda x, t: V0*np.sin(omega*t)*np.ones_like(x))
    grid = Grid1D(x0=0.0, x1=1.0, nx=800)
    dt = 0.2 * grid.dx / max(V0, 1e-12)
    ts, us, x = simulate(pde, grid, t0=0.0, t1=5, dt=dt, save_every=20)
    (line,) = ax.plot(x, us[i0], lw=2)
    def update(val):
        i = int(s.val)
        line.set_ydata(us[i])
        title_time.set_text(f"t = {ts[i]:.4f}")
        fig.canvas.draw_idle()


    # Varme simulering

    # pde = Varme1D(
    #     Vfun=lambda x, t: V0*np.sin(omega*t)*np.ones_like(x),
    #     nfun=lambda x, t: 1.0*np.ones_like(x),
    #     Qfun=lambda x, t: 0.0*np.ones_like(x),
    #     kappa=1e-3, eta=1e-3)
    # grid = Grid1D(x0=0.0, x1=1.0, nx=800)
    # dt_adv = 0.2 * grid.dx / max(abs(V0), 1e-12)
    # dt_diff = 0.2 * grid.dx*grid.dx / max(pde.kappa, 1e-12)
    # dt = min(dt_adv, dt_diff)
    # ts, Ts, x = simulate(pde, grid, t0=0.0, t1=10, dt=dt, save_every=20)
    # (line,) = ax.plot(x, Ts[i0], lw=2)
    # def update(val):
    #     i = int(s.val)
    #     line.set_ydata(Ts[i])
    #     title_time.set_text(f"t = {ts[i]:.4f}")
    #     fig.canvas.draw_idle()


    # Momentum simulering

    # field = "Vx"   # "n", "Vx", "Vy", "Vz"
    # pde = Momentum1D(
    #     m=1.0,
    #     q=1.0,
    #     Tfun=lambda x, t: 1.0 + 0.1*x,
    #     Efun=lambda x, t: np.zeros((x.size, 3)),
    #     Bfun=lambda x, t: np.column_stack([
    #         np.zeros_like(x),
    #         np.zeros_like(x),
    #         np.ones_like(x)])
    # )
    # B0, C = 1.0, 0.2
    # grid = Grid1D(x0=0.0, x1=1.0, nx=400)
    # omega_c = abs(pde.q * B0 / pde.m)
    # dt_adv = C * grid.dx / max(V0, 1e-12)
    # dt_gyro = C / max(omega_c, 1e-12)
    # dt = min(dt_adv, dt_gyro)
    # ts, us, x = simulate(pde, grid, t0=0.0, t1=1.3, dt=dt, save_every=20)
    # N = x.size

    # def get_field(u, N, field):
    #     n, Vx, Vy, Vz = unpack_momentum_state(u, N)
    #     if field == "n":  return n
    #     if field == "Vx": return Vx
    #     if field == "Vy": return Vy
    #     if field == "Vz": return Vz
    #     raise ValueError(f"Unknown field: {field}")
    
    # y0 = get_field(us[i0], N, field)
    # (line,) = ax.plot(x, y0, lw=2)
    # ax.set_ylabel(field)

    # def update(val):
    #     i = int(s.val)
    #     yi = get_field(us[i], N, field)
    #     line.set_ydata(yi)
    #     title_time.set_text(f"t = {ts[i]:.4f}")
    #     fig.canvas.draw_idle()


    
    ax.set_title(pde.name)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    title_time = ax.text(
        0.02, 0.95, f"t = {ts[i0]:.4f}",
        transform=ax.transAxes, va="top"
    )

    # slider-akse
    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    s = Slider(
        ax=ax_slider,
        label="time index",
        valmin=0,
        valmax=len(ts) - 1,
        valinit=i0,
        valstep=1  # gør den “snapper” til heltal
    )

    s.on_changed(update)

    plt.show()
