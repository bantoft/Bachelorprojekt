
    field = "Vx"   # "n", "Vx", "Vy", "Vz"
    pde = Momentum1D(
        m=1.0,
        q=1.0,
        Tfun=lambda x, t: 1.0 + 0.1*x,
        Efun=lambda x, t: np.zeros((x.size, 3)),
        Bfun=lambda x, t: np.column_stack([
            np.zeros_like(x),
            np.zeros_like(x),
            np.ones_like(x)])
    )
    B0, C = 1.0, 0.2
    omega_c = abs(pde.q * B0 / pde.m)
    dt_adv = C * grid.dx / max(V0, 1e-12)
    dt_gyro = C / max(omega_c, 1e-12)
    dt = min(dt_adv, dt_gyro)
    ts, us, x = simulate(pde, grid, t0=0.0, t1=1.3, dt=dt, save_every=20)
    N = x.size

    def get_field(u, N, field):
        n, Vx, Vy, Vz = unpack_momentum_state(u, N)
        if field == "n":  return n
        if field == "Vx": return Vx
        if field == "Vy": return Vy
        if field == "Vz": return Vz
        raise ValueError(f"Unknown field: {field}")
    
    y0 = get_field(us[i0], N, field)
    (line,) = ax.plot(x, y0, lw=2)
    ax.set_ylabel(field)

    def update(val):
        i = int(s.val)
        yi = get_field(us[i], N, field)
        line.set_ydata(yi)
        title_time.set_text(f"t = {ts[i]:.4f}")
        fig.canvas.draw_idle()