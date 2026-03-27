import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Gradient og Partielle afledte
# -----------------------------------------------------------------------------
def grad(f: Tensor, akse: Tensor, create_graph: bool = True) -> Tensor:
    if not f.requires_grad: raise ValueError(f"Husk dog nu at sætte requires_grad=True på'{f, akse}'")
    return torch.autograd.grad(
        f,
        akse,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


# -----------------------------------------------------------------------------
# Anden Partielle afledte i x, y
# -----------------------------------------------------------------------------
def p_xx(f: Tensor, x: Tensor) -> Tensor:
    return grad(grad(f, x), x)

def p_yy(f: Tensor, y: Tensor) -> Tensor:
    return grad(grad(f, y), y)

# -----------------------------------------------------------------------------
# Operatores
# -----------------------------------------------------------------------------
# kappa(f) = - rho_2/R partial_y f
def curvature(f: Tensor, y: Tensor, rho_s: float | Tensor, R: float | Tensor) -> Tensor:
    return -(rho_s / R) * grad(f, y)

# {f, g} = partial_x f partial_y g - partial_y f partial_x g 
def poisson_bracket(f: Tensor, g: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return grad(f, x)*grad(g, y) - grad(f, y)*grad(g, x)

# nabla_perp f = partial_x f + partial_y f
def div_perp(f: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return grad(f, x) + grad(f, y)

# nabla_perp^2 f = partial_xx f + partial_yy f
def laplacian_perp(f: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return p_xx(f, x) + p_yy(f, y)

# d_t f = partial_t f + 1/B {phi, f} 
# Note to self: H er for hårdt d
def Hd_t(f: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor, B: float)-> Tensor:
    return grad(f, t) + (1.0/B)*poisson_bracket(phi, f, x, y)

# d^0_t f = partial_t f + {phi, f} 
# Note to self: H er for hårdt d
def Hd0_t(f: Tensor, phi: Tensor, x: Tensor, y: Tensor, t: Tensor, B: float)-> Tensor:
    return grad(f, t) + poisson_bracket(phi, f, x, y)

# bar f = 1/L_y int_0^L_y f dy
def bar(f: Tensor, dim: int) -> Tensor:
    return torch.mean(f, dim=dim, keepdim=True)

# tilde f = f - bar f
def tilde(f: Tensor, dim: int) -> Tensor:
    return f - bar(f, dim=dim)

def div_n_grad_T(n: Tensor, T: Tensor, x: Tensor, y: Tensor) -> Tensor:
    dTdx = grad(T, x)
    dTdy = grad(T, y)
    return grad(n * dTdx, x) + grad(n * dTdy, y)
