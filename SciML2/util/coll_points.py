from ..loss_PDE.bout_dump import BOUTHESELInfo
from pathlib import Path
import torch

root = Path.cwd().parent / 'simulatorer' / 'BOUT' / 'BOUT-HESEL' / 'data'
info = BOUTHESELInfo(root)


def collocation_points(info, num_points):
    nt, nx, nz = info.data.lnn.shape
    t = torch.randint(60, nt, (num_points,))
    x = torch.randint(0, nx, (num_points,))
    # z is normal distributed around z/2 with std 0.5, and clamped to [0, nx]
    z = torch.randn(num_points) * 0.5 + nz / 2
    z = torch.clamp(z, 0, nz - 1)
    return x, z, t
