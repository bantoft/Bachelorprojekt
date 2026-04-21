import torch
from torch import nn
from torch import Tensor



class PINN(nn.Module):
    def __init__(self, structure: dict, parameters: Tensor):
        super().__init__()
        self.structure = structure
        self.output_names = tuple(structure.get("output_names", ("lnn", "lnpe", "lnpi", "phi")))
        params = parameters.detach().reshape(1, -1).float()
        self.register_buffer("param_tensor", params)
        layers = []
        in_features = structure["input_size"] + self.param_tensor.shape[-1]
        for layer in structure["layers"]:
            out_features = layer["size"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(layer["non_lin_foo"]())
            in_features = out_features

        layers.append(nn.Linear(in_features, structure["output_size"]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, z: Tensor, t: Tensor) -> dict[str, Tensor]:
        inputs = torch.cat([x, z, t], dim=-1)
        params = self.param_tensor.to(device=inputs.device, dtype=inputs.dtype)
        params = params.view(*((1,) * (inputs.ndim - 1)), -1).expand(*inputs.shape[:-1], -1)
        out = self.network(torch.cat([inputs, params], dim=-1))
        return {name: out[..., i : i + 1] for i, name in enumerate(self.output_names)}
    
