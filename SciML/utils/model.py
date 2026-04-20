from __future__ import annotations

import torch
import torch.nn as nn

from .types import nn_mapping


class PINN(nn.Module):
    def __init__(self, nn_structure: dict):
        super().__init__()

        self.nn_structure = nn_structure
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
