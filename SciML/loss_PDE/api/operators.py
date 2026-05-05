from __future__ import annotations

import torch

from torch import Tensor


def grad(f: Tensor, coord: Tensor, create_graph: bool = True) -> Tensor:
    if not f.requires_grad:
        return coord * 0.0
    raw = torch.autograd.grad(
        f,
        coord,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if raw is None:
        raw = coord * 0.0
    return raw


def grad_x(f: Tensor, x: Tensor) -> Tensor:
    return grad(f, x)


def grad_z(f: Tensor, z: Tensor) -> Tensor:
    return grad(f, z)


def grad_t(f: Tensor, t: Tensor) -> Tensor:
    return grad(f, t)


def d2dx2(f: Tensor, x: Tensor) -> Tensor:
    return grad_x(grad_x(f, x), x)


def d2dz2(f: Tensor, z: Tensor) -> Tensor:
    return grad_z(grad_z(f, z), z)


def d2dxdz(f: Tensor, x: Tensor, z: Tensor) -> Tensor:
    return grad_z(grad_x(f, x), z)


def laplacian_perp(f: Tensor, x: Tensor, z: Tensor) -> Tensor:
    return d2dx2(f, x) + d2dz2(f, z)
