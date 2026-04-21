from __future__ import annotations

import torch

from utils.custom_types import TrainConfig
from utils.model import PINN
from loss_funktion.loss_from_hesel import BOUTHESELSystem
from utils.data_loader import make_data_loader, ids_to_inputs


nn_structure = {
    "input_size": 3,  # x, z, t
    "output_size": 4,  # lnn, lnpe, lnpi,
    "layers": [
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
    ],
}

def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system = BOUTHESELSystem()

    parameters = torch.tensor(list(system.parameters.values()), dtype=torch.float32, device=device)

    model = PINN(nn_structure, parameters.numel()).to(device)

    epochs = int(getattr(config, "epochs", 5))
    batch_size = int(getattr(config, "batch_size", 128))
    lr = float(getattr(config, "lr", 1e-3))

    # Loss weights
    w_data = float(getattr(config, "w_data", 1.0))
    w_eq = float(getattr(config, "w_eq", 1.0))
    w_bc = float(getattr(config, "w_bc", 1.0))
    w_ic = float(getattr(config, "w_ic", 1.0))

    data_loader = make_data_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        system=system,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    print(f"Training on device: {device}")
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_eq = 0.0
        epoch_bc = 0.0
        epoch_ic = 0.0
        steps = 0

        for batch_idx, batch in enumerate(data_loader, start=1):
            x_ids, z_ids, t_ids, lnn, lnpe, lnpi, vort = batch
            x, z, t = ids_to_inputs(x_ids, z_ids, t_ids, system=system, data=system.dump_data)
            x, z, t = x.to(device), z.to(device), t.to(device)

            targets = {
                "lnn": lnn.to(device).view(-1, 1),
                "lnpe": lnpe.to(device).view(-1, 1),
                "lnpi": lnpi.to(device).view(-1, 1),
                "vort": vort.to(device).view(-1, 1),
            }

            optimizer.zero_grad()
            loss_data_raw = system.data_loss(model, data=targets, x=x, z=z, t=t)
            loss_eq_raw = system.eq_loss(model)
            loss_bc_raw = system.bc_loss(model)
            loss_ic_raw = system.ic_loss(model)

            loss_data = w_data * loss_data_raw
            loss_eq = w_eq * loss_eq_raw
            loss_bc = w_bc * loss_bc_raw
            loss_ic = w_ic * loss_ic_raw

            loss_total = loss_data + loss_eq + loss_bc + loss_ic
            loss_total.backward()
            optimizer.step()

            epoch_total += float(loss_total.item())
            epoch_data += float(loss_data_raw.item())
            epoch_eq += float(loss_eq_raw.item())
            epoch_bc += float(loss_bc_raw.item())
            epoch_ic += float(loss_ic_raw.item())
            steps += 1


    return model


if __name__ == "__main__":
    trained_model = train(TrainConfig())
    print("Training completed")