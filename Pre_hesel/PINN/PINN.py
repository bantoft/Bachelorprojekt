import torch
import torch.nn as nn

from PDE_system import device

class PINNWaveSystem(nn.Module):
    def __init__(self, layers):
        super(PINNWaveSystem, self).__init__()
        # layers fx [2, 64, 64, 64, 2]
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < self.depth - 2:
                layer_list.append(nn.Tanh())
        self.net = nn.Sequential(*layer_list)

    def forward(self, x, t):
        # x, t shape: (N,1)
        X = torch.cat((x, t), dim=1)  # (N,2)
        out = self.net(X)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v

# Eksempel: 3 skjulte lag med 64 neuroner
layers = [2, 64, 64, 64, 2]
model = PINNWaveSystem(layers).to(device)