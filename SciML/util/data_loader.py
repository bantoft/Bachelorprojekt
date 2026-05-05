from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from loss_PDE.bout_dump import BOUTHESELInfo



FIELDS = ("lnn", "lnpe", "lnpi", "phi")


class BOUTDataset(Dataset):
    def __init__(self, info):
        self.info = info

        self.t_line = self.info.t_line
        self.x_line = self.info.x_line
        self.z_line = self.info.z_line

        self.fields = {
            name: torch.as_tensor(getattr(self.info.data, name), dtype=torch.float32)
            for name in FIELDS
        }
        self.x_grid, self.z_grid = self._build_spatial_grids()

    def _build_spatial_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_grid, z_grid = torch.meshgrid(self.x_line, self.z_line, indexing="ij")
        return x_grid.unsqueeze(-1), z_grid.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.t_line)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        t_value = self.t_line[index].view(1, 1, 1).expand_as(self.x_grid)
        fields = {
            name: values[index].unsqueeze(-1)
            for name, values in self.fields.items()
        }
        return {
            "x": self.x_grid.clone(),
            "z": self.z_grid.clone(),
            "t": t_value.clone(),
            "fields": fields,
        }
