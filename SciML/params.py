from math import sqrt, log, tanh

# TODO: Lav til dictionary for bedre organisering og læsbarhed

#############################
#Parametre til equation loss#
#############################

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


#############################
#Parametre til boundery loss#
#############################
# Alle outer er neuman(0)
x_outer = 0.

#Option init_n:<>
n_inner = 1.5
pe_inner = 9.0
pi_inner = 9.0

# Vort inner er er også dirichlet 0
# 	core region: 	Option vort:bndry_xin = dirichlet_o2(0.) (data/BOUT.inp)
phi_inner = 0.

bundery_val = {
    'n':{
        'x_inner': log(n_inner),
        'x_outer': x_outer
    },
    'p_e':{
        'x_inner': log(pe_inner),
        'x_outer': x_outer
    },
    'p_i':{
        'x_inner': log(pi_inner),
        'x_outer': x_outer
    },
    'phi':{
        'x_inner': phi_inner,
        'x_outer': x_outer
    }
}

######################################
#Parametre til initial condition loss#
######################################
# TODO: Dummy for
# Find ved [] definere en klasse fra_klasse:værdi
n_bg = 0.005 # Hvad er det ?
xr = 0.4 # Hvad er det ?
x_shift_n = 0.05 # Hvad er det ?
step_width = 10
pe_bg = 1
edge_prof = 1

# Funktioner defineret i x-akse
# Option init_<n:function =>
initial_conditions = {
    'n': lambda x: 0.5*(n_inner -n_bg)*(1 - tanh(x)/step_width),
    'pe': lambda x: (0.5 *(sqrt(pe_inner) - pe_bg) * (1 - tanh( (x)/step_width)) + pe_bg) * edge_prof,
    'pe': lambda x: x,
}
