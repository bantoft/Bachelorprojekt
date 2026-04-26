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
        return max(self.nt - 1, 0) * self.nx * self.nz

    def __getitem__(self, idx: int):
        idx = int(idx)
        t_id, rest = divmod(idx, self.nx * self.nz)
        x_id, z_id = divmod(rest, self.nz)
        t_next = t_id + 1
        return (
            torch.tensor(x_id, dtype=torch.long),
            torch.tensor(z_id, dtype=torch.long),
            torch.tensor(t_id, dtype=torch.long),
            *[
                self.info.standardize_field(
                    name,
                    torch.as_tensor(self.data[name][t_next, x_id, z_id], dtype=torch.float32),
                ).to(dtype=torch.float32)
                for name in self.fields
            ],
        )


    def ids_to_inputs(self, x_ids: torch.Tensor, z_ids: torch.Tensor, t_ids: torch.Tensor):
        return normalize_ids(x_ids, z_ids, t_ids, nt=self.nt, nx=self.nx, nz=self.nz)

    def _biased_x_coords(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        x_bias_power: float,
    ) -> torch.Tensor:
        if count <= 1:
            return torch.zeros(1, device=device, dtype=dtype)

        if count == 2:
            return torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        interior_count = count - 2
        x_uniform = torch.rand(interior_count, device=device, dtype=dtype)
        x_centered = 2.0 * x_uniform - 1.0
        x_interior = 0.5 * (
            1.0 + torch.sign(x_centered) * torch.abs(x_centered).pow(x_bias_power)
        )
        x_interior, _ = torch.sort(x_interior)
        boundaries = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        return torch.cat((boundaries[:1], x_interior, boundaries[1:]))

    def _random_unit_coords(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        upper: float = 1.0,
        include_endpoints: bool = False,
    ) -> torch.Tensor:
        if count <= 1:
            return torch.zeros(1, device=device, dtype=dtype)

        coords = torch.rand(count, device=device, dtype=dtype) * upper
        coords, _ = torch.sort(coords)
        if include_endpoints:
            coords[0] = 0.0
            coords[-1] = upper
        return coords

    def make_collocation_grid(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_t: int = 8,
        num_x: int = 64,
        num_z: int = 128,
        x_bias_power: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nt = min(num_t, max(self.nt - 1, 1))
        nx = min(num_x, max(self.nx, 2))
        nz = min(num_z, max(self.nz, 2))

        x_coords = self._biased_x_coords(
            count=nx,
            device=device,
            dtype=dtype,
            x_bias_power=x_bias_power,
        )
        t_coords = self._random_unit_coords(
            count=nt,
            device=device,
            dtype=dtype,
            upper=self.info.max_stepper_input_time,
            include_endpoints=True,
        )
        z_coords = self._random_unit_coords(
            count=nz,
            device=device,
            dtype=dtype,
            upper=1.0,
            include_endpoints=True,
        )

        t_grid, x_grid, z_grid = torch.meshgrid(
            t_coords,
            x_coords,
            z_coords,
            indexing="ij",
        )
        return (
            x_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
            z_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
            t_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
        )

    def make_initial_condition_grid(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_x: int = 96,
        num_z: int = 128,
        x_bias_power: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nx = min(num_x, max(self.nx, 2))
        nz = min(num_z, max(self.nz, 2))

        x_coords = self._biased_x_coords(
            count=nx,
            device=device,
            dtype=dtype,
            x_bias_power=x_bias_power,
        )
        z_coords = self._random_unit_coords(
            count=nz,
            device=device,
            dtype=dtype,
            upper=1.0,
            include_endpoints=True,
        )

        x_grid, z_grid = torch.meshgrid(x_coords, z_coords, indexing="ij")
        t_grid = torch.zeros_like(x_grid)
        return (
            x_grid.unsqueeze(0).unsqueeze(-1).clone().detach().requires_grad_(True),
            z_grid.unsqueeze(0).unsqueeze(-1).clone().detach().requires_grad_(True),
            t_grid.unsqueeze(0).unsqueeze(-1).clone().detach().requires_grad_(True),
        )

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
