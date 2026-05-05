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
    
    def _standardize_field(self, field: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (field - mean) / std

    def _destandardize(self, outputs: Tensor, standardization) -> Tensor:
        mean = standardization.mean.to(device=outputs.device, dtype=outputs.dtype)
        std = standardization.std.to(device=outputs.device, dtype=outputs.dtype)

        view_shape = (1,) * (outputs.ndim - 1) + (-1,)
        mean = mean.view(view_shape)
        std = std.view(view_shape)
        return outputs * std + mean

    def _extract_neighborhood(self, field: Tensor) -> Tensor:
        num_x, num_z = field.shape[-2], field.shape[-1]
        flat = field.reshape(-1, 1, num_x, num_z)
        padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
        neighborhoods = F.unfold(padded, kernel_size=3)
        neighborhoods = neighborhoods.transpose(1, 2)
        return neighborhoods.reshape(*field.shape[:-2], num_x, num_z, 9)

    def _stack_state(self, state_t: dict[str, Tensor], standardization) -> Tensor:
        mean = standardization.mean
        std = standardization.std
        features = []
        for index, name in enumerate(self.output_names):
            field = state_t[name]
            field_mean = mean[index].to(device=field.device, dtype=field.dtype)
            field_std = std[index].to(device=field.device, dtype=field.dtype)
            standardized_field = self._standardize_field(field, field_mean, field_std)
            features.append(self._extract_neighborhood(standardized_field.squeeze(-1)))
        return torch.cat(
            features,
            dim=-1,
        )

    def forward(
        self,
        standardization,
        x: Tensor,
        z: Tensor,
        t: Tensor,
        state_t: dict[str, Tensor],
        state_t_minus_1: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        state_t_features = self._stack_state(state_t, standardization)
        state_t_minus_1_features = self._stack_state(state_t_minus_1, standardization)
        inputs = torch.cat(
            (x, z, t, state_t_features, state_t_minus_1_features),
            dim=-1,
        )
        outputs = self.network(inputs)
        outputs = self._destandardize(outputs, standardization)
        return self._split_outputs(outputs)
