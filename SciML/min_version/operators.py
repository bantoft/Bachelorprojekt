from params2 import *

import torch
from torch import Tensor


def grad(f: Tensor, akse: Tensor, create_graph: bool = True) -> Tensor:
    if not f.requires_grad:
        raise ValueError(f"Husk at sætte requires_grad=True på {f, akse}")
    return torch.autograd.grad(
        f,
        akse,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]

test = 10
def p_xx(f: Tensor, x: Tensor) -> Tensor:
    return grad(grad(f, x), x)


def p_yy(f: Tensor, y: Tensor) -> Tensor:
    return grad(grad(f, y), y)

def curvature(f: Tensor, y: Tensor) -> Tensor:
    return -(rhos/R)*grad(f, y)


def poisson_bracket(f: Tensor, g: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return grad(f, x) * grad(g, y) - grad(f, y) * grad(g, x)


def div_perp(f: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return grad(f, x) + grad(f, y)


def laplacian_perp(f: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return p_xx(f, x) + p_yy(f, y)


def Hd_t(f: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor, ) -> Tensor:
    return grad(f, t) + (1.0 / B0) * poisson_bracket(phi, f, x, y)


def Hd0_t(f: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
    return grad(f, t) + poisson_bracket(phi, f, x, y)


def bar(f: Tensor, dim: int) -> Tensor:
    return torch.mean(f, dim=dim, keepdim=True)


def tilde(f: Tensor, dim: int) -> Tensor:
    return f - bar(f, dim=dim)