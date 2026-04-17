from math import sqrt, log, pi


# Direkte fra log fil: (copy paste for nu)
Coulomb_Log = 1.328201e+01
vte         = 2.287830e+06	 
vti       = 3.775324e+04	 
cs      = 3.775324e+04
oci         = 4.228007e+07	 
oce       = 1.552653e+11
rhoe        = 1.473497e-05	 
rhoi      = 8.929325e-04	 
rhos    = 8.929325e-04
mi          = 3.345243e-27	 
me        = 9.109382e-31
n0          = 1.850000e+19	 
Te0       = 2.980000e+01	 
Ti0     = 2.980000e+01
B0          = 8.839819e-01	 
q95       = 4.200000e+00	 
Rmajor  = 8.800000e-01
nuei        = 4.379143e+06	 
nuii      = 5.109810e+04	 
nuee    = 3.096522e+06
neoclassical_correction_factor = 6.999200e+01
De          = 6.654814e-02	 
Di        = 2.851611e+00
De_1_tau   = 1.330963e-01
eta         = 3.308956e+11
chi_i_perp  = 1.055096e+20
chi_e_perp  = 5.737116e+18
chi_i_par   = 2.012520e+24
chi_e_par   = 6.987423e+25
taun        = 9.789887e-05	 
en_over_taun    = 1.021462e+04
taudw       = 5.714467e-06	 
en_over_taudw   = 1.749945e+05

# Parametre fra input er fundet i <	Option hesel:>


# Dummy variabler
Lc = 15 # Divertor leg length (Artikel værdi): Option hesel:diag_thermal = 15
tau_SH = 50/3.16 # tau Spitzer-Härm (Artikel sted: Num sim "under eq" (13)): Option hesel:?
L_perp = 8 # L_b = L_perp = q*R  8 ?: and this length should be compared to the connection 
e_skæv = -1 # Blvier brugt til at regne omega_io

# Profile region for phi
connected = 0.8 # Større er den disconected


# værdier i profile region
x_lcfs = 0.4 # LCFS linje (her tror jeg profile region er halvdelen)
x_wall = 0.8 # wall region
# Værdierne
floor_time = 50 # tau_p [aktivt på indersiden]
force_time = 50 # tau TODO: Er Tau over det hele ?
# Profile værdier
floor_n    = 0.005 # n_p
floor_pe   = 0.000025 # p_e,p
floor_pi   = 0.000025 # p_i,p

# udregnet parametre:
phi_m = log(sqrt(mi/(2*pi*me))) # Bohm potential
omega_io = e_skæv*B0/mi # gyro frequency at magnetic
alpha = 2*Te0/(nuei*me*L_perp**2) # Udregnet i (14)

