from math import sqrt, log

pi = 3.141592653589793
# Physical parameters from BOUT.log.0
B = 1.1 # Fra input
me = 9.109382e-31
mi = 3.345243e-27
Rmajor = 0.88
rhos = 8.929325e-04
Te0 = 2.980000e+01
nuei = 4.379143e+06
De = 6.654814e-02
Di = 2.851611e+00

# Relaxation parameters
floor_n = 0.005  # Density profile floor value
floor_pe = 0.000025  # Electron pressure profile floor value
floor_pi = 0.000025  # Ion pressure profile floor value
floor_time = 50  # Profile relaxation time scale
force_time = 50  # Global relaxation time scale

# Profile and domain parameters
x_lcfs = 0.4  # LCFS boundary position
Lc = 15  # Divertor leg length
tau_SH = 50/3.16  # Spitzer-Härm time scale
L_perp = 8  # Perpendicular connection length

# Computed parameters
phi_m = log(sqrt(mi/(2*pi*me)))  # Bohm potential
alpha = 2*Te0/(nuei*me*L_perp**2)  # Relaxation coefficient

