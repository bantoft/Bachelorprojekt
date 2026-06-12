from __future__ import annotations

import torch

from torch import Tensor


def grad(f: Tensor,
         coord: Tensor,
         create_graph: bool = True,
         retain_graph: bool | None = None
         ) -> Tensor:
    
    if retain_graph is None:
        retain_graph = create_graph

    return torch.autograd.grad(
        f,
        coord,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
        allow_unused=True,
    )[0]


def grad_components(f: Tensor,
                    coord: Tensor,
                    create_graph: bool = True,
                    retain_graph: bool | None = None
                    ) -> tuple[Tensor, Tensor, Tensor]:
    
    gradients = grad(f,
                     coord,
                     create_graph=create_graph,
                     retain_graph=retain_graph
                     )
    # [batch, [x, z, t]]
    return gradients[:, 0:1], gradients[:, 1:2], gradients[:, 2:3]


def grad_x(f: Tensor, coord: Tensor) -> Tensor:
    return grad(f, coord)[:, 0:1]


def grad_z(f: Tensor, coord: Tensor) -> Tensor:
    return grad(f, coord)[:, 1:2]


def grad_t(f: Tensor, coord: Tensor) -> Tensor:
    return grad(f, coord)[:, 2:3]


def d2dx2(f: Tensor, coord: Tensor) -> Tensor:
    return grad_x(grad_x(f, coord), coord)


def d2dz2(f: Tensor, coord: Tensor) -> Tensor:
    return grad_z(grad_z(f, coord), coord)

def d2dxdz(f: Tensor, coord: Tensor) -> Tensor:
    return grad_x(grad_z(f, coord), coord)


def laplacian_perp(f: Tensor, coord: Tensor) -> Tensor:
    return d2dx2(f, coord) + d2dz2(f, coord)
