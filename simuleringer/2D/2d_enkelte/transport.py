import numpy as np
import matplotlib.pyplot as plt



x_min, x_max, Nx = 0, 50, 200
y_min, y_max, Ny = 0, 50, 200

x = np.linspace(start=x_min, stop=x_max, num=Nx)
y = np.linspace(start=y_min, stop=y_max, num=Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")