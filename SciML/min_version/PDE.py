from operators import laplacian_perp, Hd_t, Hd0_t, curvature, poisson_bracket

import torch
from torch import Tensor



def mse_residual(r: Tensor) -> Tensor:
    return torch.mean(r**2)

def compute_residuals(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor):
    w_star = laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y)
    # lambda termer aner det ikk check på det senere
    R_n = Hd_t(n, phi, x, y, t) + n*curvature(phi, y) - curvature(p_e, y) # - lambda_n
    R_w = Hd0_t(w_star, phi, x, y, t) + poisson_bracket(phi, p_i, x, y) + curvature(p_e + p_i, y) # - Lambda_w
    R_pe = 1.5 * Hd_t(p_e, phi, x, y, t) + 2.5 * p_e*curvature(phi, y) - 2.5 * curvature((p_e**2)/(n + 1e-8) ,y) # - Lambda_pe
    R_pi = 1.5 * Hd_t(p_i, phi, x, y, t) + 2.5 * p_i * curvature(phi, y) + 2.5 * curvature((p_i**2) / (n + 1e-8), y) - p_i * curvature(p_e + p_i, y) # - Lambda_pi
    return R_n, R_w, R_pe, R_pi

def pde_loss(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor, weights: dict | None = None,):
    if weights is None: weights = {"n": 1.0, "w": 1.0, "pe": 1.0, "pi": 1.0}

    R_n, R_w, R_pe, R_pi = compute_residuals(n=n, p_e=p_e, p_i=p_i, phi=phi, x=x, y=y, t=t)
    loss_n = mse_residual(R_n)
    loss_w = mse_residual(R_w)
    loss_pe = mse_residual(R_pe)
    loss_pi = mse_residual(R_pi)

    total = weights["n"] * loss_n + weights["w"] * loss_w + weights["pe"] * loss_pe + weights["pi"] * loss_pi

    return {"loss_total": total, "loss_n": loss_n, "loss_w": loss_w, "loss_pe": loss_pe, "loss_pi": loss_pi, "R_n": R_n, "R_w": R_w, "R_pe": R_pe, "R_pi": R_pi}
