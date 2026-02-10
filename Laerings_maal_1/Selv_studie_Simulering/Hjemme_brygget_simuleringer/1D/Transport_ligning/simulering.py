import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sp.init_printing()

# Symboler
x, t = sp.symbols('x t', real=True)
n0, sigma, U, omega = sp.symbols('n0 sigma U omega', positive=True, real=True)

# Oscillerende center og hastighed
x_c = (U/omega) * sp.sin(omega*t)

# Funktioner
V_a = sp.diff(x_c, t)  # = U*cos(omega*t)
n_a = n0 * sp.exp(- (x - x_c)**2 / (2*sigma**2))

# Kontinuitetsligning
dndt = sp.diff(n_a, t)
div_nV = sp.diff(n_a * V_a, x)
residual = dndt + div_nV 

print("Symbolsk residual (skal være 0):")
print(residual)  # bør printe 0

# Parametre (vælg selv)
par = {
    n0: 2.5,
    sigma: 4.5,
    U: 5.0,
    omega: 0.25,
}

# Numeriske funktioner
n_fun   = sp.lambdify((x, t), n_a.subs(par), "numpy")
V_fun_t = sp.lambdify(t, V_a.subs(par), "numpy")
div_fun = sp.lambdify((x, t), div_nV.subs(par), "numpy")
res_fun = sp.lambdify((x, t), residual.subs(par), "numpy")

# Grid
xs = np.linspace(-40, 40, 200)
t_init = 0.0

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# n(x,t)
ln_n, = axs[0].plot(xs, n_fun(xs, t_init))
axs[0].set_ylabel(r"$n_a(x,t)$")
axs[0].set_title("PDE-konsistent frem/tilbage advektion (kontinuitetsligning)")

# V(t) broadcastet til x-aksen
ln_V, = axs[1].plot(xs, np.full_like(xs, V_fun_t(t_init), dtype=float))
axs[1].set_ylabel(r"$V_a(t)$")

# div og residual
ln_div, = axs[2].plot(xs, div_fun(xs, t_init), label=r"$\partial_x(n_aV_a)$")
ln_res, = axs[2].plot(xs, np.zeros_like(xs, dtype=float), "--", label=r"residual: $\partial_t n + \partial_x(nV)$")

axs[2].set_ylabel("Led")
axs[2].set_xlabel("x")
axs[2].legend()

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
t_slider = Slider(ax=ax_slider, label="t", valmin=0.0, valmax=50.0, valinit=t_init)

def update(val):
    tt = t_slider.val
    ln_n.set_ydata(n_fun(xs, tt))
    ln_V.set_ydata(np.full_like(xs, V_fun_t(tt), dtype=float))
    ln_div.set_ydata(div_fun(xs, tt))
    ln_res.set_ydata(np.zeros_like(xs, dtype=float))
    fig.canvas.draw_idle()

t_slider.on_changed(update)
plt.show()
