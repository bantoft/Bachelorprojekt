import torch
from torch import Tensor

from ..operators import grad
from ..params import bundery_val


def BC_residuals(n: Tensor, p_e: Tensor, p_i: Tensor, phi: Tensor, x: Tensor, y: Tensor) -> dict:
    BC_n_inner = n[:,0] - bundery_val['n']['x_inner']
    BC_n_outer = grad(n, x)[:, -1] - bundery_val['n']['x_outer']

    BC_p_e = p_e[:,0] - bundery_val['p_e']['x_inner']
    BC_p_e_outer = grad(p_e, x)[:, -1] - bundery_val['p_e']['x_outer']

    BC_p_i = p_i[:,0] - bundery_val['p_i']['x_inner']
    BC_p_i_outer = grad(p_i, x)[:, -1] - bundery_val['p_i']['x_outer']

    BC_phi = phi[:,0] - bundery_val['phi']['x_inner']
    BC_phi_outer = grad(phi, x)[:, -1] - bundery_val['phi']['x_outer']

    return {
        'n': (BC_n_inner, BC_n_outer),
        'p_e': (BC_p_e, BC_p_e_outer),
        'p_i': (BC_p_i, BC_p_i_outer),
        'phi': (BC_phi, BC_phi_outer)
    }