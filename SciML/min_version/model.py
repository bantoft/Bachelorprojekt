import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, nn_structure: dict):
        super().__init__()

        input_size = int(nn_structure.get("input_size", 3))
        output_size = int(nn_structure.get("output_size", 1))
        hidden_layers = list(nn_structure.get("hidden_layers", [50, 50, 50]))
        activations = list(nn_structure.get("activation", []))
        dropouts = list(nn_structure.get("dropout", []))
        batch_norms = list(nn_structure.get("batch_norm", []))

        layers = []
        prev_size = input_size

        for hidden_size, activation, dropout, batch_norm in zip(hidden_layers, activations, dropouts, batch_norms):
            hidden_size = int(hidden_size)
            layers.append(nn.Linear(prev_size, hidden_size))

            if activation is not None:
                layers.append(activation)

            if batch_norm is not None:
                if isinstance(batch_norm, str):
                    if batch_norm.lower() == "batch_norm":
                        layers.append(nn.BatchNorm1d(hidden_size))
                    elif batch_norm.lower() == "instance_norm":
                        layers.append(nn.InstanceNorm1d(hidden_size))
                elif isinstance(batch_norm, nn.Module):
                    layers.append(batch_norm)

            if dropout is not None and (isinstance(dropout, nn.Module) or float(dropout) > 0):
                if isinstance(dropout, nn.Module):
                    layers.append(dropout)
                else:
                    layers.append(nn.Dropout(float(dropout)))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, *inputs):
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=1)
        return self.net(x)
