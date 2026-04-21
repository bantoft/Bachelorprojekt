import torch
from torch import nn

class PINN(nn.Module):
    def __init__(self, structure: dict, param_dim: int):
        super().__init__()
        self.structure = structure
        self.param_dim = param_dim
        layers = []
        in_features = structure["input_size"] + param_dim
        for layer in structure["layers"]:
            out_features = layer["size"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(layer["non_lin_foo"]())
            in_features = out_features

        layers.append(nn.Linear(in_features, structure["output_size"]))
        self.network = nn.Sequential(*layers)

    def forward(self, x, params):
        x_full = torch.cat([x, params], dim=1)
        return self.network(x_full)
    