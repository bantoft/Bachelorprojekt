import torch
from torch import Tensor

from Operators import curvature, Hd_t, Hd0_t
from funktioner import omega_star
from lambda_termer import lambda_n, lambda_omega, lambda_pe, lambda_pi


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
):
    B = params["B"]
    rho_s = params["rho_s"]
    R = params["R"]

    # omega^*
    w_star = omega_star(phi, p_i, x, y)

    # Lambda-led
    Lambda_n = lambda_n(n=n, p_e=p_e, phi=phi, x=x, y=y, t=t, params=params)

    Lambda_w = lambda_omega(omega_star=w_star, n=n, p_e=p_e, phi=phi, x=x, y=y, t=t, params=params)

    Lambda_pe = lambda_pe(n=n, p_e=p_e, p_i=p_i, phi=phi, x=x, y=y, t=t, params=params)

    Lambda_pi = lambda_pi(n=n, p_e=p_e, p_i=p_i, omega_star=w_star, x=x, y=y, t=t, params=params)

    # Residualer i standard form
    # 1) n-ligningen
    R_n = (
        Hd_t(n, phi, x, y, t, B)
        + n * curvature(phi, y, rho_s, R)
        - curvature(p_e, y, rho_s, R)
        - Lambda_n
    )

    # 2) omega*-ligningen
    R_w = (
        Hd0_t(w_star, phi, x, y, t, B)
        - curvature(p_e + p_i, y, rho_s, R)
        - Lambda_w
    )

    # 3) elektrontryk
    R_pe = (
        1.5 * Hd_t(p_e, phi, x, y, t, B)
        + 2.5 * p_e * curvature(phi, y, rho_s, R)
        - 2.5 * curvature((p_e**2) / (n + 1e-8), y, rho_s, R)
        - Lambda_pe
    )

    # 4) iontryk
    R_pi = (
        1.5 * Hd_t(p_i, phi, x, y, t, B)
        + 2.5 * p_i * curvature(phi, y, rho_s, R)
        + 2.5 * curvature((p_i**2) / (n + 1e-8), y, rho_s, R)
        - p_i * curvature(p_e + p_i, y, rho_s, R)
        - Lambda_pi
    )

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
):
    """
    Returnerer:
        total_loss, loss_dict
    """

    if weights is None:
        weights = {
            "n": 1.0,
            "w": 1.0,
            "pe": 1.0,
            "pi": 1.0,
        }

    R_n, R_w, R_pe, R_pi = compute_residuals(
        n=n, p_e=p_e, p_i=p_i, phi=phi,
        x=x, y=y, t=t,
        params=params,
    )

    loss_n = mse_residual(R_n)
    loss_w = mse_residual(R_w)
    loss_pe = mse_residual(R_pe)
    loss_pi = mse_residual(R_pi)

    total = (
        weights["n"] * loss_n
        + weights["w"] * loss_w
        + weights["pe"] * loss_pe
        + weights["pi"] * loss_pi
    )

    loss_dict = {
        "loss_total": total,
        "loss_n": loss_n,
        "loss_w": loss_w,
        "loss_pe": loss_pe,
        "loss_pi": loss_pi,
        "R_n": R_n,
        "R_w": R_w,
        "R_pe": R_pe,
        "R_pi": R_pi,
    }

    return total, loss_dict