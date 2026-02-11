import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from funk_ex2 import n_a, V_a, div_nV

# Numeriske funktioner
x, t = sp.symbols('x t')
n_fun   = sp.lambdify((x, t), n_a, "numpy")
V_fun_t = sp.lambdify(t, V_a, "numpy")
div_fun = sp.lambdify((x, t), div_nV, "numpy")

# Grid
xs = np.linspace(-20, 20, 500)
t_init = 0.0

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# n(x,t)
ln_n, = axs[0].plot(xs, n_fun(xs, t_init))
axs[0].set_ylabel(r"$n_a(x,t)$")
axs[0].set_title("PDE-konsistent frem/tilbage advektion (kontinuitetsligning)")

# V(t) broadcastet til x-aksen
ln_V, = axs[1].plot(xs, np.full_like(xs, V_fun_t(t_init), dtype=float))
axs[1].set_ylabel(r"$V_a(t)$")


# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
t_slider = Slider(ax=ax_slider, label="t", valmin=0.0, valmax=20.0, valinit=t_init)

def update(val):
    tt = t_slider.val
    ln_n.set_ydata(n_fun(xs, tt))
    ln_V.set_ydata(np.full_like(xs, V_fun_t(tt), dtype=float))
    fig.canvas.draw_idle()

t_slider.on_changed(update)
plt.show()
