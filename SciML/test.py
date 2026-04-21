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


from dataclasses import astuple
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

system = BOUTHESELSystem()

param_tensor = torch.tensor(astuple(system.parameters), dtype=torch.float32, device=device)

param_dim = param_tensor.numel()


model = PINN(nn_structure, param_tensor.numel()).to(device)



lnn, lnpe, lnpi, phi = model(torch.randn(1, 3).to(device), param_tensor.unsqueeze(0))[0]
print(lnn, lnpe, lnpi, phi)