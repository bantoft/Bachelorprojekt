import torch
from torch import nn
from torch import Tensor
from typing import Mapping

class PINN(nn.Module):
    def __init__(self, structure: dict):
        super().__init__()
        self.structure = structure
        self.output_names = tuple(structure["output_names"])
        self.register_buffer("output_mean", torch.zeros(structure["output_size"], dtype=torch.float32))
        self.register_buffer("output_std", torch.ones(structure["output_size"], dtype=torch.float32))
        layers = []
        in_features = structure["input_size"]
        for layer in structure["layers"]:
            out_features = layer["size"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(layer["non_lin_foo"]())
            in_features = out_features

        layers.append(nn.Linear(in_features, structure["output_size"]))
        self.network = nn.Sequential(*layers)

    def set_output_scaling(self, stats: Mapping[str, Mapping[str, float]]) -> None:
        means = self.output_mean.detach().clone()
        stds = self.output_std.detach().clone()
        for i, name in enumerate(self.output_names):
            field_stats = stats.get(name)
            if field_stats is None:
                continue
            means[i] = float(field_stats["mean"])
            std_value = float(field_stats.get("std", 1.0))
            stds[i] = std_value if std_value > 0.0 else 1.0

        self.output_mean.copy_(means.to(device=self.output_mean.device, dtype=self.output_mean.dtype))
        self.output_std.copy_(stds.to(device=self.output_std.device, dtype=self.output_std.dtype))

    def destandardize_outputs(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for i, name in enumerate(self.output_names):
            values = outputs[name]
            mean = self.output_mean[i].to(device=values.device, dtype=values.dtype)
            std = self.output_std[i].to(device=values.device, dtype=values.dtype)
            result[name] = values * std + mean
        return result

    def export_output_scaling(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "mean": float(self.output_mean[i].item()),
                "std": float(self.output_std[i].item()),
                "variance": float(self.output_std[i].item() ** 2),
            }
            for i, name in enumerate(self.output_names)
        }

    def forward(self, x: Tensor, z: Tensor, t: Tensor) -> dict[str, Tensor]:
        inputs = torch.cat([x, z, t], dim=-1)
        out = self.network(inputs)
        return {name: out[..., i : i + 1] for i, name in enumerate(self.output_names)}

    def predict_physical(self, x: Tensor, z: Tensor, t: Tensor) -> dict[str, Tensor]:
        return self.destandardize_outputs(self.forward(x, z, t))
