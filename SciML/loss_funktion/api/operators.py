from __future__ import annotations

import torch

from torch import Tensor

def grad(f: Tensor, coord: Tensor, scale: float = 1.0, create_graph: bool = True) -> Tensor:
    if not f.requires_grad:
        raise ValueError("Input tensor to grad must require gradients.")
    raw = torch.autograd.grad(
        f,
        coord,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return raw / scale


def grad_x(f: Tensor, x: Tensor, x_scale: float) -> Tensor:
    return grad(f, x, scale=x_scale)


def grad_z(f: Tensor, z: Tensor, z_scale: float) -> Tensor:
    return grad(f, z, scale=z_scale)


def grad_t(f: Tensor, t: Tensor, t_scale: float) -> Tensor:
    return grad(f, t, scale=t_scale)


def d2dx2(f: Tensor, x: Tensor, x_scale: float) -> Tensor:
    return grad_x(grad_x(f, x, x_scale), x, x_scale)


def d2dz2(f: Tensor, z: Tensor, z_scale: float) -> Tensor:
    return grad_z(grad_z(f, z, z_scale), z, z_scale)


def d2dxdz(f: Tensor, x: Tensor, z: Tensor, x_scale: float, z_scale: float) -> Tensor:
    return grad_z(grad_x(f, x, x_scale), z, z_scale)


def laplacian_perp(f: Tensor, x: Tensor, z: Tensor, x_scale: float, z_scale: float) -> Tensor:
    return d2dx2(f, x, x_scale) + d2dz2(f, z, z_scale)
