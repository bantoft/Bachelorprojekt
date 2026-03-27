import torch
import torch.nn as nn
from typing import List


def get_activation(actv):
    if isinstance(actv, nn.Module):
        return actv
    elif actv == "relu":
        return nn.ReLU()
    elif actv == "tanh":
        return nn.Tanh()
    elif actv == "gelu":
        return nn.GELU()
    elif actv == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Ukendt activation: {actv}")


class PINNEnkeltFelt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, struk: List[List]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim, actv, dpot in struk:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(get_activation(actv))
            if dpot is not None and dpot > 0:
                layers.append(nn.Dropout(dpot))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        X = torch.cat((x, y, t), dim=1)   # (N,3)
        return self.net(X)
    

strukture = [
    [128, "tanh", 0.0],
    [128, "tanh", 0.0],
    [128, "tanh", 0.0],
]

model = PINNEnkeltFelt(
    input_dim=3,
    output_dim=1,
    struk=strukture
)