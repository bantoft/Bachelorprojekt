import numpy as np

def rk4_step(system, a, b, dt, work):
    """
    a, b: arrays med ghost cells, shape (nx+2, ny+2)
    work: dict med preallokerede arrays (for speed)
    """
    g = system.grid

    # Views til fysisk domæne
    a0, b0 = g.interior(a), g.interior(b)

    # unpack work arrays (alle shape (nx, ny) eller (nx+2, ny+2))
    k1a, k1b = work["k1a"], work["k1b"]
    k2a, k2b = work["k2a"], work["k2b"]
    k3a, k3b = work["k3a"], work["k3b"]
    k4a, k4b = work["k4a"], work["k4b"]

    a_tmp, b_tmp = work["a_tmp"], work["b_tmp"]  # ghost arrays

    # k1
    r1a, r1b = system.rhs(a, b)     # phys arrays
    k1a[:] = r1a
    k1b[:] = r1b

    # k2: a + dt/2 k1
    g.interior(a_tmp)[:] = a0 + 0.5*dt*k1a
    g.interior(b_tmp)[:] = b0 + 0.5*dt*k1b
    r2a, r2b = system.rhs(a_tmp, b_tmp)
    k2a[:] = r2a
    k2b[:] = r2b

    # k3
    g.interior(a_tmp)[:] = a0 + 0.5*dt*k2a
    g.interior(b_tmp)[:] = b0 + 0.5*dt*k2b
    r3a, r3b = system.rhs(a_tmp, b_tmp)
    k3a[:] = r3a
    k3b[:] = r3b

    # k4
    g.interior(a_tmp)[:] = a0 + dt*k3a
    g.interior(b_tmp)[:] = b0 + dt*k3b
    r4a, r4b = system.rhs(a_tmp, b_tmp)
    k4a[:] = r4a
    k4b[:] = r4b

    # opdater a,b i fysisk domæne
    a0[:] = a0 + (dt/6.0)*(k1a + 2*k2a + 2*k3a + k4a)
    b0[:] = b0 + (dt/6.0)*(k1b + 2*k2b + 2*k3b + k4b)

def simulate(system, a, b, t0, t1, dt, save_every=10):
    g = system.grid
    nsteps = int(np.ceil((t1 - t0)/dt))

    # preallocate work arrays
    work = {
        "k1a": g.alloc_phys(), "k1b": g.alloc_phys(),
        "k2a": g.alloc_phys(), "k2b": g.alloc_phys(),
        "k3a": g.alloc_phys(), "k3b": g.alloc_phys(),
        "k4a": g.alloc_phys(), "k4b": g.alloc_phys(),
        "a_tmp": g.alloc_field(ghost=1),
        "b_tmp": g.alloc_field(ghost=1),
    }

    ts = []
    As = []
    Bs = []

    t = t0
    for n in range(nsteps + 1):
        if n % save_every == 0:
            ts.append(t)
            As.append(g.interior(a).copy())
            Bs.append(g.interior(b).copy())

        if t >= t1:
            break

        rk4_step(system, a, b, dt, work)
        t += dt

    return np.array(ts), np.array(As), np.array(Bs)
