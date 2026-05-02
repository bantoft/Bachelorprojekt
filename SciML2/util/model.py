from torch import nn




class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # (B,4,H,W) -> (B,16,H,W)
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),  # (B,16,H,W) -> (B,4,H,W)
        )

def forward(self, x, standardization):
    mean, std = standardization.mean.to(x.device), standardization.std.to(x.device)
    x_std = (x - mean) / std
    out_std = self.net(x_std)
    out = out_std * std + mean
    return out