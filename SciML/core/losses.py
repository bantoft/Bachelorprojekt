import torch
from torch import Tensor

from .operators import curvature, Hd_t, Hd0_t
from .physics import omega_star
from .lambda_terms import lambda_n, lambda_omega, lambda_pe, lambda_pi
from .params import resolve_hesel_params


def mse_residual(r: Tensor) -> Tensor:
    return torch.mean(r**2)


def compute_residuals(
    n: Tensor,
    p_e: Tensor,
    p_i: Tensor,
    phi: Tensor,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    params: dict,
    params_are_resolved: bool = False,
):
    hparams = params if params_are_resolved else resolve_hesel_params(params, phi)
    B = hparams["B"]
    rho_s = hparams["rho_s"]
    R = hparams["R"]
    curvature_coeff = hparams["curvature_coeff"]

    w_star = omega_star(phi, p_i, x, y)

    Lambda_n = lambda_n(n=n, p_e=p_e, phi=phi, x=x, y=y, D_e=hparams["D_e"], tau_n_inv=hparams["tau_n_inv"], alpha=hparams["alpha"], y_dim=hparams["y_dim"])
    Lambda_w = lambda_omega(omega_star=w_star, n=n, p_e=p_e, phi=phi, x=x, y=y, D_i=hparams["D_i"], tau_w_inv=hparams["tau_w_inv"], alpha=hparams["alpha"], rho_s=hparams["rho_s"], L_c=hparams["L_c"], phi_s=hparams["phi_s"], T_e_s=hparams["T_e_s"], y_dim=hparams["y_dim"])
    Lambda_pe = lambda_pe(n=n, p_e=p_e, p_i=p_i, phi=phi, x=x, y=y, D_e=hparams["D_e"], tau_pe_inv=hparams["tau_pe_inv"], tau_SH_e_inv=hparams["tau_SH_e_inv"], alpha=hparams["alpha"], me=hparams["me"], mi=hparams["mi"], nu_ei=hparams["nu_ei"], y_dim=hparams["y_dim"])
    Lambda_pi = lambda_pi(n=n, p_e=p_e, p_i=p_i, omega_star=w_star, x=x, y=y, D_i=hparams["D_i"], tau_pi_inv=hparams["tau_pi_inv"], me=hparams["me"], mi=hparams["mi"], nu_ei=hparams["nu_ei"], lambda_omega_value=Lambda_w)

    R_n = Hd_t(n, phi, x, y, t, B) + n * curvature_coeff * curvature(phi, y, rho_s, R) - curvature_coeff * curvature(p_e, y, rho_s, R) - Lambda_n
    R_w = Hd0_t(w_star, phi, x, y, t, B) - curvature_coeff * curvature(p_e + p_i, y, rho_s, R) - Lambda_w
    R_pe = 1.5 * Hd_t(p_e, phi, x, y, t, B) + 2.5 * p_e * curvature_coeff * curvature(phi, y, rho_s, R) - 2.5 * curvature_coeff * curvature((p_e**2) / (n + 1e-8), y, rho_s, R) - Lambda_pe
    R_pi = 1.5 * Hd_t(p_i, phi, x, y, t, B) + 2.5 * p_i * curvature_coeff * curvature(phi, y, rho_s, R) + 2.5 * curvature_coeff * curvature((p_i**2) / (n + 1e-8), y, rho_s, R) - p_i * curvature_coeff * curvature(p_e + p_i, y, rho_s, R) - Lambda_pi

    return R_n, R_w, R_pe, R_pi


def pde_loss(
    n: Tensor,
    p_e: Tensor,
    p_i: Tensor,
    phi: Tensor,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    params: dict,
    weights: dict | None = None,
    params_are_resolved: bool = False,
):
    if weights is None:
        weights = {"n": 1.0, "w": 1.0, "pe": 1.0, "pi": 1.0}

    R_n, R_w, R_pe, R_pi = compute_residuals(n=n, p_e=p_e, p_i=p_i, phi=phi, x=x, y=y, t=t, params=params, params_are_resolved=params_are_resolved)

    loss_n = mse_residual(R_n)
    loss_w = mse_residual(R_w)
    loss_pe = mse_residual(R_pe)
    loss_pi = mse_residual(R_pi)

    total = weights["n"] * loss_n + weights["w"] * loss_w + weights["pe"] * loss_pe + weights["pi"] * loss_pi

    return {"loss_total": total, "loss_n": loss_n, "loss_w": loss_w, "loss_pe": loss_pe, "loss_pi": loss_pi, "R_n": R_n, "R_w": R_w, "R_pe": R_pe, "R_pi": R_pi}
