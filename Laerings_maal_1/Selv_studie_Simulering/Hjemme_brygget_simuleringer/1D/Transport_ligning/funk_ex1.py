from sympy import sqrt, pi, exp, diff, symbols,sin

x, t = symbols('x t', real = True)

n_a = 1/sqrt(pi)*exp(-(x-t)**2)
V_a = 1 + t

dndt = diff(n_a, t)
div_nV = diff(n_a * V_a, x)