from __future__ import annotations

import torch
import torch.nn as nn


nn_mapping = {
    "activation_functions": [nn.ELU(), nn.GELU(), nn.LogSigmoid(), nn.Sigmoid(), nn.SiLU(), nn.Tanh(), nn.Tanhshrink()],
    "layer_sizes": [16, 32, 64, 128, 256, 512],
}


NN_structure = {
    "input_size": 3,
    "output_size": 4,
    "layer1": {"size": 1, "activation": 1},
    "layer2": {"size": 1, "activation": 1},
    "layer3": {"size": 1, "activation": 1},
    "layer4": {"size": 1, "activation": 1},
}


class PINN(nn.Module):
    def __init__(self, nn_structure: dict | None = None):
        super().__init__()

        self.nn_structure = NN_structure if nn_structure is None else nn_structure
        input_size = int(self.nn_structure["input_size"])
        output_size = int(self.nn_structure["output_size"])

        activation_functions = nn_mapping["activation_functions"]
        layer_sizes = nn_mapping["layer_sizes"]

        layers: list[nn.Module] = []
        previous_size = input_size
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = self.nn_structure[layer_name]
            hidden_size = layer_sizes[int(layer["size"])]
            activation = activation_functions[int(layer["activation"])]
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation.__class__())
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((x, z, t), dim=-1))
