from sympy import sqrt, pi, exp, diff, symbols, sin, cos

x, t = symbols('x t', real = True)

n_a = 1/sqrt(pi)*exp(-(x-sin(t))**2)
V_a = cos(t)

dndt = diff(n_a, t)
div_nV = diff(n_a * V_a, x)