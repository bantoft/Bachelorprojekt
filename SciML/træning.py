from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import torch
import torch.nn.functional as F

from loss_funktion.bout_info import BOUTHESELInfo
from loss_funktion.bout_data import BOUTHESELData
from loss_funktion.bout_phys import BOUTHESELPhysics
from utils.model import PINN


NN_STRUCTURE = {
    "input_size": 3,
    "output_size": 4,
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
    ],
}


def train(
    epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 1e-3,
    root: str | Path | None = None,
    w_data: float = 1.0,
    w_eq: float = 1.0,
    w_bc: float = 1.0,
    w_ic: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = BOUTHESELInfo(root) if root is not None else BOUTHESELInfo()
    physics = BOUTHESELPhysics(info)
    dataset = BOUTHESELData(
        info,
        fields=("lnn", "lnpe", "lnpi", "phi"),
    )
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    parameters = torch.tensor(astuple(info.parameters), dtype=torch.float32, device=device)
    model = PINN(NN_STRUCTURE, parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_eq, z_eq, t_eq = info.make_collocation_grid(device=device)
    x_ic = x_eq[:1].clone().detach().requires_grad_(True)
    z_ic = z_eq[:1].clone().detach().requires_grad_(True)
    t_ic = t_eq[:1].clone().detach().requires_grad_(True)

    model.train()
    print(f"Training on device: {device}")
    for epoch in range(1, epochs + 1):
        totals = {"total": 0.0, "data": 0.0, "eq": 0.0, "bc": 0.0, "ic": 0.0}
        steps = 0

        for x_ids, z_ids, t_ids, lnn, lnpe, lnpi, phi in loader:
            x_data, z_data, t_data = dataset.ids_to_inputs(x_ids, z_ids, t_ids)
            x_data = x_data.to(device=device, dtype=torch.float32)
            z_data = z_data.to(device=device, dtype=torch.float32)
            t_data = t_data.to(device=device, dtype=torch.float32)
            targets = {
                "lnn": lnn.to(device=device, dtype=torch.float32).view(-1, 1),
                "lnpe": lnpe.to(device=device, dtype=torch.float32).view(-1, 1),
                "lnpi": lnpi.to(device=device, dtype=torch.float32).view(-1, 1),
                "phi": phi.to(device=device, dtype=torch.float32).view(-1, 1),
            }

            optimizer.zero_grad()
            pred = model(x_data, z_data, t_data)
            loss_data = sum(F.mse_loss(pred[name], targets[name]) for name in targets)
            loss_eq = physics.eq_loss(model, x_eq, z_eq, t_eq)
            loss_bc = physics.bc_loss(model, x_eq, z_eq, t_eq)
            loss_ic = physics.ic_loss(model, x_ic, z_ic, t_ic)
            loss = w_data * loss_data + w_eq * loss_eq + w_bc * loss_bc + w_ic * loss_ic
            loss.backward()
            optimizer.step()

            totals["total"] += float(loss.item())
            totals["data"] += float(loss_data.item())
            totals["eq"] += float(loss_eq.item())
            totals["bc"] += float(loss_bc.item())
            totals["ic"] += float(loss_ic.item())
            steps += 1

        print(
            f"Epoch {epoch}/{epochs} | "
            f"total={totals['total'] / steps:.4e} | "
            f"data={totals['data'] / steps:.4e} | "
            f"eq={totals['eq'] / steps:.4e} | "
            f"bc={totals['bc'] / steps:.4e} | "
            f"ic={totals['ic'] / steps:.4e}"
        )

    return model, info, physics, dataset


if __name__ == "__main__":
    train()
