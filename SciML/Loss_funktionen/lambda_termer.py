import torch

from torch import Tensor
from Operators import *
from funktioner import *


def lambda_n(
    n: Tensor,
    p_e: Tensor,
    phi: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    D_e: float,
    tau_n_inv: float | Tensor,
    alpha: float | Tensor,
    y_dim: int,
) -> Tensor:
    """
    Λ_n = D_e ∇_⊥² n - τ_n^{-1} n - alpha( T̃_e + (T̄_e/n̄)ñ - φ̃ )
    """
    diffusion = D_e * laplacian_perp(n, x, y)
    parallel_loss = tau_n_inv * n
    drift = drift_wave_term(n, p_e, phi, y_dim, alpha)

    return diffusion - parallel_loss - drift


def lambda_omega(
    omega_star: Tensor,
    n: Tensor,
    p_e: Tensor,
    phi: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    D_i: float,
    tau_w_inv: float | Tensor,
    alpha: float | Tensor,
    rho_s: float | Tensor,
    L_c: float | Tensor,
    phi_s: Tensor,
    T_e_s: Tensor,
    y_dim: int,
) -> Tensor:
    """
    Λ_ω = D_i ∇_⊥² ω* - τ_ω^{-1} ω*
          + sheath
          - alpha( T̃_e + (T̄_e/n̄)ñ - φ̃ )
    """
    diffusion = D_i * laplacian_perp(omega_star, x, y)
    parallel_loss = tau_w_inv * omega_star
    sheath = sheath_term(phi, phi_s, T_e_s, rho_s, L_c)
    drift = drift_wave_term(n, p_e, phi, y_dim, alpha)

    return diffusion - parallel_loss + sheath - drift


def lambda_pe(
    n: Tensor,
    p_e: Tensor,
    p_i: Tensor,
    phi: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    D_e: float,
    tau_pe_inv: float | Tensor,
    tau_SH_e_inv: float | Tensor,
    alpha: float | Tensor,
    me: float,
    mi: float,
    nu_ei: float | Tensor,
    y_dim: int,
) -> Tensor:
    """
    Λ_pe =
        (5/2) D_e ∇_⊥² p_e
      + (16/6 - 5/2) ∇·(n ∇_⊥ T_e)
      - τ_pe^{-1} p_e
      - τ_SH,e^{-1} p_e
      - Q
      - T̄_e * alpha( T̃_e + (T̄_e/n̄)ñ - φ̃ )
    """
    T_e = electron_temperature(n, p_e)

    diffusion = 2.5 * D_e * laplacian_perp(p_e, x, y)
    conduction = (16.0 / 6.0 - 2.5) * div_n_grad_T(n, T_e, x, y)

    parallel_adv = tau_pe_inv * p_e
    parallel_heat = tau_SH_e_inv * p_e

    Q = theta_ei(p_e, p_i, me=me, mi=mi, nu_ei=nu_ei)

    T_e_bar = bar(T_e, dim=y_dim)
    drift = T_e_bar * drift_wave_term(n, p_e, phi, y_dim, alpha)

    return diffusion + conduction - parallel_adv - parallel_heat - Q - drift


def lambda_pi(
    n: Tensor,
    p_e: Tensor,
    p_i: Tensor,
    omega_star: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    D_i: float,
    tau_pi_inv: float | Tensor,
    me: float,
    mi: float,
    nu_ei: float | Tensor,
    lambda_omega_value: Tensor | None = None,
) -> Tensor:
    """
    Λ_pi =
        D_i ∇_⊥² p_i
      - D_i T_i ∇_⊥² n
      - τ_pi^{-1} p_i
      + Q
      + p_i Λ_ω

    Her skal du være opmærksom på, at sidste led i artiklen står som + p_i Λ_ω.
    """
    T_i = ion_temperature(n, p_i)

    diffusion_pi = D_i * laplacian_perp(p_i, x, y)
    diffusion_n = D_i * T_i * laplacian_perp(n, x, y)

    parallel_loss = tau_pi_inv * p_i
    Q = theta_ei(p_e, p_i, me=me, mi=mi, nu_ei=nu_ei)

    if lambda_omega_value is None:
        lambda_omega_value = torch.zeros_like(p_i)

    return diffusion_pi - diffusion_n - parallel_loss + Q + p_i * lambda_omega_value