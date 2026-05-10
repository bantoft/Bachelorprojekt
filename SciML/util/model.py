from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor, nn


class PINN(nn.Module):
    def __init__(self, structure: dict):
        super().__init__()

        self.structure = structure
        self.output_names = tuple(self.structure["output_names"])

        layers = []
        in_features = self.structure["input_size"]
        for layer in self.structure["layers"]:
            out_features = layer["size"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(layer["non_lin_foo"]())
            in_features = out_features

        layers.append(nn.Linear(in_features, self.structure["output_size"]))
        self.network = nn.Sequential(*layers)

    def _split_outputs(self, outputs: Tensor) -> dict[str, Tensor]:
        return {
            name: outputs[..., index : index + 1]
            for index, name in enumerate(self.output_names)
        }
    

    def forward(
        self,
        standardization,
        x: Tensor,
        z: Tensor,
        t: Tensor,
        state_inputs: Tensor,
    ):
        mean = standardization.mean
        std = standardization.std
        if mean.device != state_inputs.device:
            mean = mean.to(state_inputs.device)
        if std.device != state_inputs.device:
            std = std.to(state_inputs.device)

        state_inputs_std = (state_inputs - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)

        batch_size = state_inputs_std.shape[0]
        network_input = torch.cat(
            [
                x.reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                t.reshape(batch_size, 1),
                state_inputs_std.reshape(batch_size, -1),
            ],
            dim=-1,
        )

        pred_std = self.network(network_input)
        pred = pred_std * std.view(1, -1) + mean.view(1, -1)
        return self._split_outputs(pred)
