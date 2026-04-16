from operators import *
from params import *

import torch
from torch import Tensor


def Lambda_termer(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    T_e = p_e / (n + 1e-8)
    w_star = laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y)

    safety_term = q95*(tilde(T_e, 0) + (bar(T_e, 0)/bar(n, 0))*tilde(n, 0) - tilde(phi, 0))
    Lambda_n = De*laplacian_perp(n, x, t) - n/force_time - (n - floor_n)/floor_time - safety_term
    Lambda_w = Di*laplacian_perp(w_star, x, y) - w_star/force_time + rhos/Lc
    # TODO: færdig gør lambda termer 


def RHS_residuals(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    w_star = laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y)
    # lambda termer aner det ikk check på det senere
    R_n = Hd_t(n, phi, x, y, t) + n*curvature(phi, y) - curvature(p_e, y) # - lambda_n
    R_w = Hd0_t(w_star, phi, x, y, t) + poisson_bracket(phi, p_i, x, y) + curvature(p_e + p_i, y) # - Lambda_w
    R_pe = 1.5 * Hd_t(p_e, phi, x, y, t) + 2.5 * p_e*curvature(phi, y) - 2.5 * curvature((p_e**2)/(n + 1e-8) ,y) # - Lambda_pe
    R_pi = 1.5 * Hd_t(p_i, phi, x, y, t) + 2.5 * p_i * curvature(phi, y) + 2.5 * curvature((p_i**2) / (n + 1e-8), y) - p_i * curvature(p_e + p_i, y) # - Lambda_pi
    return R_n, R_w, R_pe, R_pi



# def pde_losses(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor):
#     R_n, R_w, R_pe, R_pi = compute_residuals(n=n, p_e=p_e, p_i=p_i, phi=phi, x=x, y=y, t=t)
#     return {
#         "R_n": mse_residual(R_n),
#         "R_w": mse_residual(R_w),
#         "R_pe": mse_residual(R_pe),
#         "R_pi": mse_residual(R_pi),
#     }

