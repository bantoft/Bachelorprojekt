from math import sqrt

Z_eff   = 1.2
Bt      = 1.11                           # toroidal magnetic field at magnetic axis
q       = 4.2                            # safety factor (95%)
Te0     = 29.8                           # reference electron temperature
Ti0     = 29.8
n0      = 1.85e+19                       # reference electron density
lconn   = 20                             # connection length
lblob   = -1                             # ballooning length, -1 use inner definition: lblob = q*R
R       = 0.88                           # major radius
a       = 0.225                          # minor radius
A       = 2                              # ion mass number. m_i = A*m_p
Z       = 1                              # ion charge
Mach    = 0.5                            # Mach number
e       = 1.60e-19
mp      = 1.67262158e-27

mi      = A*mp
B0      = Bt*R/(R+a)

oci     = e*Z*B0/mi                      # Sigma_{ci} 
cs      = sqrt(e*Te0/mi)
rhos    = cs/oci