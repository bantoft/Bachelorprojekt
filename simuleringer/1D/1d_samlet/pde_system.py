from dataclasses import dataclass
import numpy as np

def ddx_central(f, dx):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0*dx)

def ddx_upwind_flux(F, a, dx):
    F_L = F
    F_R = np.roll(F, -1)
    a_half = 0.5*(a + np.roll(a, -1))
    F_half = np.where(a_half >= 0.0, F_L, F_R)
    return (F_half - np.roll(F_half, 1)) / dx

def ddx_rusanov_flux(U, F, amax, dx):
    U_L = U
    U_R = np.roll(U, -1)
    F_L = F
    F_R = np.roll(F, -1)
    a_half = np.maximum(amax, np.roll(amax, -1))
    F_half = 0.5*(F_L + F_R) - 0.5*a_half*(U_R - U_L)
    return (F_half - np.roll(F_half, 1)) / dx


@dataclass
class Closure1DConst:
    # konstanter (skalarer)
    eta: float = 1e-3
    kappa: float = 1e-3
    nu: float = 0.0
    E0: float = 0.0     # konstant elektrisk felt (1D)
    Q0: float = 0.0     # konstant varmekilde

    def pressure(self, x, t, n, V, T):
        return n * T

    def pi_xx(self, x, t, n, V, T, dx):
        # pi_xx = -eta dV/dx
        return -self.eta * ddx_central(V, dx)

    def heat_flux(self, x, t, n, V, T, dx):
        # q = -kappa dT/dx
        return -self.kappa * ddx_central(T, dx)

    def E(self, x, t, n, V, T):
        # returnér et array i samme form som x
        return self.E0 * np.ones_like(x)

    def Q(self, x, t, n, V, T):
        return self.Q0 * np.ones_like(x)

    def nu_arr(self, x, t, n, V, T):
        return self.nu * np.ones_like(x)


@dataclass
class BragSystem1D:
    x: np.ndarray
    closure: Closure1DConst
    m: float = 1.0
    e: float = 1.0

    def __post_init__(self):
        self.dx = self.x[1] - self.x[0]
        self.N = self.x.size

    def pack(self, n, V, T):
        return np.concatenate([n, V, T])

    def unpack(self, y):
        N = self.N
        return y[:N], y[N:2*N], y[2*N:3*N]

    def rhs(self, y, t):
        x, dx = self.x, self.dx
        n, V, T = self.unpack(y)

        # closures (konstante parametre, men arrays ud)
        p  = self.closure.pressure(x, t, n, V, T)
        pi = self.closure.pi_xx(x, t, n, V, T, dx)
        q  = self.closure.heat_flux(x, t, n, V, T, dx)

        nu = self.closure.nu_arr(x, t, n, V, T)
        E  = self.closure.E(x, t, n, V, T)
        Q  = self.closure.Q(x, t, n, V, T)

        inv_n = 1.0 / np.maximum(n, 1e-12)

        # (1) continuity: n_t = - d/dx(nV)
        Fn = n*V
        amax_n = np.abs(V)
        nt = -ddx_rusanov_flux(U=n, F=Fn, amax=amax_n, dx=dx)

        # (2) momentum
        dV2dx = ddx_upwind_flux(V*V, a=V, dx=dx)   # d/dx(V^2)
        dpdx  = ddx_central(p, dx)
        dpidx = ddx_central(pi, dx)
        Vt = -0.5*dV2dx - (dpdx + dpidx)/self.m*inv_n + (self.e/self.m)*E - nu*V

        # (3) temperature
        FT   = 1.5*n*T*V
        advT = ddx_upwind_flux(FT, a=V, dx=dx)
        Vx   = ddx_central(V, dx)
        dqdx = ddx_central(q, dx)
        Tt = (-advT - n*T*Vx - pi*Vx - dqdx + Q) / (1.5*np.maximum(n, 1e-12))

        return self.pack(nt, Vt, Tt)
