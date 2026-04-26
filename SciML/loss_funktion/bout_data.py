from __future__ import annotations

import torch
from .bout_info import BOUTHESELInfo
from torch.utils.data import DataLoader, Dataset
from .api.read_bout import DEFAULT_FIELDS, normalize_ids

class BOUTHESELData(Dataset):
    def __init__(self, info: BOUTHESELInfo):
        self.info = info
        self.fields = DEFAULT_FIELDS
        self.data = self.info.data
        self.nt, self.nx, self.nz = self.data[self.fields[0]].shape

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
                torch.as_tensor(self.data[name][t_id, x_id, z_id], dtype=torch.float32)
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
        drop_last: bool = False,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
