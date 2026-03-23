import numpy as np

def rk4_step(rhs, y, t, dt):
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = rhs(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = rhs(y + dt*k3, t + dt)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(rhs, y0, t0, t1, dt, save_every=10):
    t = t0
    y = y0.copy()
    ts = [t]
    ys = [y.copy()]
    k = 0
    while t < t1 - 1e-15:
        y = rk4_step(rhs, y, t, dt)
        t += dt
        k += 1

        if not np.isfinite(y).all():
            print("NaN/Inf ved step", k, "t=", t)
            bad = np.where(~np.isfinite(y))[0][0]
            print("første bad index i y:", bad, "value:", y[bad])
            break

        if k % save_every == 0:
            ts.append(t); ys.append(y.copy())
    return np.array(ts), np.array(ys)
