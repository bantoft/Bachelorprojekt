from operators import *
from params import *

import torch
from torch import Tensor

def mask_profile(d: Tensor, d_min: float, d_max: float) -> Tensor:
    return ((d>= d_min) & (d < d_max)).to(dtype=d.dtype)


def Lambda_termer(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, x_connected: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Pre define parametre der går igen flere gange
    n_p = floor_n        # profile value: float
    p_e_p = floor_pe     # profile value: float
    p_i_p = floor_pi     # profile value: float
    tau = force_time     # Time scale: float
    tau_p = floor_time   # Time scale profile region: float
    T_e = p_e/(n + 1e-8) # Temp_e: Tensor
    T_i = p_i/(n + 1e-8) # Temp_i: Tensor
    Theta = 3*me*nuei*(p_e - p_i)/mi    # Energy transfer between the electron and ion channels: Tensor
    w_star = laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y) # omega^*: Tenspr
    Relaxationsterm = (
        alpha*(
            tilde(T_e, 0)
            + (bar(T_e, 0) / (bar(n, 0) + 1e-8))
            * tilde(n, 0)
            - tilde(phi, 0)
        )
    )
    # Profile region (aktive)
    profile_n = ((n-n_p)/tau_p)*mask_profile(x, 0, x_lcfs)
    profile_w_connected = (torch.exp(phi_m - phi/(T_e +1e-8)))*mask_profile(x, 0, x_connected)
    profile_w_disconctd = (torch.exp(phi_m - bar(phi, 0)/bar(T_e, 0)))*mask_profile(x, x_connected, x.shape[0])
    profile_pe = (((p_e-p_e_p)/tau_p)*mask_profile(x, 0, x_lcfs))
    profile_pi = (((p_i-p_i_p)/tau_p)*mask_profile(x, 0, x_lcfs))
    
    # Lambda termer
    Lambda_n = (
        De*laplacian_perp(n, x, y)
        - n/tau
        - profile_n
        - Relaxationsterm
    )

    Lambda_w = (
        Di*laplacian_perp(w_star, x, y)
        - w_star/tau
        + rhos/Lc*(1- (profile_w_connected + profile_w_disconctd))
        - Relaxationsterm
    )
    
    Lambda_pe = (
    5/2*De*laplacian_perp(p_e, x, y)
    + (16/6-5/2)*div_perp(n*div_perp(T_e, x, y), x, y)
    - 9*p_e/(2*tau)
    - T_e/tau_SH 
    - Theta
    - profile_pe
    - bar(T_e, 0)*Relaxationsterm
    )

    Lambda_pi = (
    Di*(laplacian_perp(p_i, x, y))
    - Di*T_i*laplacian_perp(n, x, y)
    - 9*p_i/(2*tau)
    + Theta
    - profile_pi
    + p_i*Lambda_w
    )    
    return Lambda_n, Lambda_w, Lambda_pe, Lambda_pi

def RHS_residuals(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    w_star = laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y)
    R_n = Hd_t(n, phi, x, y, t) + n*curvature(phi, y) - curvature(p_e, y)
    R_w = Hd0_t(w_star, phi, x, y, t) + poisson_bracket(phi, p_i, x, y) + curvature(p_e + p_i, y)
    R_pe = 1.5 * Hd_t(p_e, phi, x, y, t) + 2.5 * p_e*curvature(phi, y) - 2.5 * curvature((p_e**2)/(n + 1e-8) ,y)
    R_pi = 1.5 * Hd_t(p_i, phi, x, y, t) + 2.5 * p_i * curvature(phi, y) + 2.5 * curvature((p_i**2) / (n + 1e-8), y) - p_i * curvature(p_e + p_i, y)
    return R_n, R_w, R_pe, R_pi

def EQ_residuals(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor, x_connected: float, weights: dict)-> dict:
    R_n, R_w, R_pe, R_pi = RHS_residuals(n, p_e, p_i, phi, x, y, t)
    Lambda_n, Lambda_w, Lambda_pe, Lambda_pi = Lambda_termer(n, p_e, p_i, phi, x, y, x_connected)

    scale = 1e-20
    mse = torch.nn.MSELoss()
    loss_dict={
        'Eq_n': mse(weights['Eq_n']*(R_n-Lambda_n/scale), torch.zeros_like(R_n)),
        'Eq_w': mse(weights['Eq_w']*(R_w-Lambda_w/scale), torch.zeros_like(R_w)),
        'Eq_pe': mse(weights['Eq_pe']*(R_pe-Lambda_pe/scale), torch.zeros_like(R_pe)),
        'Eq_pi': mse(weights['Eq_pi']*(R_pi-Lambda_pi/scale), torch.zeros_like(R_pi)),
    }
    return loss_dict
