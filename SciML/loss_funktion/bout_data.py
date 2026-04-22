from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from .api.read_bout import DEFAULT_BOUT_HESEL_ROOT, DEFAULT_FIELDS, normalize_ids
import xarray as xr


class BOUTHESELData(Dataset):
    def __init__(
        self,
        source: str | Path | Any | None = None,
        fields: tuple[str, ...] = DEFAULT_FIELDS,
    ):
        root = DEFAULT_BOUT_HESEL_ROOT if source is None else getattr(source, "root", source)
        self.root = Path(root).resolve()
        self.dump_path = self.root / "data" / "BOUT.dmp.0.nc"
        self.fields = fields
        with xr.open_dataset(self.dump_path, engine="netcdf4") as ds:
            self.dump_data = {
                name: ((ds[name].squeeze("y", drop=True) if "y" in ds[name].dims else ds[name]).values)
                for name in fields + ("t_array",)
                if name in ds
            }
        self.fields = tuple(name for name in fields if name in self.dump_data)
        self.nt, self.nx, self.nz = self.dump_data[self.fields[0]].shape

    def __len__(self) -> int:
        return self.nt * self.nx * self.nz

    def __getitem__(self, idx: int):
        idx = int(idx)
        t_id, rest = divmod(idx, self.nx * self.nz)
        x_id, z_id = divmod(rest, self.nz)
        return (
            torch.tensor(x_id, dtype=torch.long),
            torch.tensor(z_id, dtype=torch.long),
            torch.tensor(t_id, dtype=torch.long),
            *[
                torch.tensor(self.dump_data[name][t_id, x_id, z_id], dtype=torch.float32)
                for name in self.fields
            ],
        )

    def ids_to_inputs(self, x_ids: torch.Tensor, z_ids: torch.Tensor, t_ids: torch.Tensor):
        return normalize_ids(x_ids, z_ids, t_ids, nt=self.nt, nx=self.nx, nz=self.nz)

    def make_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 1,
        drop_last: bool = True,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

