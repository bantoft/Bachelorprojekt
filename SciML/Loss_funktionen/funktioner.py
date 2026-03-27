import torch

from torch import Tensor
from Operators import laplacian_perp, bar, tilde, grad

# Sikre x/0
def safe_divide(numerator: Tensor, denominator: Tensor, eps: float = 1e-8) -> Tensor:
    return numerator / (denominator + eps)

# Få T_[e, i] funktionen
def electron_temperature(n: Tensor, p_e: Tensor, eps: float = 1e-8) -> Tensor:
    return safe_divide(p_e, n, eps=eps)

def ion_temperature(n: Tensor, p_i: Tensor, eps: float = 1e-8) -> Tensor:
    return safe_divide(p_i, n, eps=eps)

# phi + p_i
def generalized_potential(phi: Tensor, p_i: Tensor) -> Tensor:
    return phi + p_i

# omega^* = nabla^2 phi + nabla^2 p_i
def omega_star(phi: Tensor, p_i: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return laplacian_perp(phi, x, y) + laplacian_perp(p_i, x, y)


# Theta 3me/mi nu_ei (p_e - p_i)
def theta_ei(p_e: Tensor, p_i: Tensor, me: float, mi: float, nu_ei: Tensor | float) -> Tensor:
    return 3.0 * (me / mi) * nu_ei * (p_e - p_i)

# Alpha(tilde T_e + (tilede T_e_bar/bar n_bar) tilde n - tilde phi)
def drift_wave_term(
    n: Tensor,
    p_e: Tensor,
    phi: Tensor,
    y_dim: int,
    alpha: float | Tensor,
) -> Tensor:
    T_e = electron_temperature(n, p_e)

    T_e_bar = bar(T_e, dim=y_dim)
    n_bar = bar(n, dim=y_dim)

    T_e_tilde = tilde(T_e, dim=y_dim)
    n_tilde = tilde(n, dim=y_dim)
    phi_tilde = tilde(phi, dim=y_dim)

    return alpha * (T_e_tilde + (T_e_bar / (n_bar + 1e-8)) * n_tilde - phi_tilde)

# Sheath-term: (rho_s / L_c) * (1 - exp(phi_s - phi / T_e_s))
def sheath_term(
    phi: Tensor,
    phi_s: Tensor,
    T_e_s: Tensor,
    rho_s: float | Tensor,
    L_c: float | Tensor,
) -> Tensor:
    return (rho_s / L_c) * (1.0 - torch.exp(phi_s - phi / (T_e_s + 1e-8)))