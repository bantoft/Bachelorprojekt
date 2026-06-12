from __future__ import annotations

import torch

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
    

    def forward(self,
                standardization,
                cord_fys: Tensor,
                state_inputs: Tensor
                ):
        
        mean = standardization.mean.to(state_inputs.device)
        std = standardization.std.to(state_inputs.device)

        state_inputs_std = (state_inputs - mean.view(1, 1, -1, 1, 1)) / std.view(1, 1, -1, 1, 1)

        batch_size = state_inputs_std.shape[0]

        network_input = torch.cat([cord_fys,  state_inputs_std.reshape(batch_size, -1),],dim=-1,)

        pred_std = self.network(network_input)

        pred = pred_std * std.view(1, -1) + mean.view(1, -1)
        
        return self._split_outputs(pred)
