from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from loss_funktion.loss_from_hesel import BOUTHESELSystem


class HESELIndexDataset(Dataset):
	def __init__(self, system: BOUTHESELSystem):
		super().__init__()
		self.data = dict(system.dump_data)
		self.nt, self.nx, self.nz = self.data["lnn"].shape
		self.total_size = self.nt * self.nx * self.nz

	def __len__(self):
		return self.total_size

	def __getitem__(self, idx):
		idx = int(idx)
		t_id = idx // (self.nx * self.nz)
		rest = idx % (self.nx * self.nz)
		x_id = rest // self.nz
		z_id = rest % self.nz

		return (
			torch.tensor(x_id, dtype=torch.long),
			torch.tensor(z_id, dtype=torch.long),
			torch.tensor(t_id, dtype=torch.long),
			self.data["lnn"][t_id, x_id, z_id],
			self.data["lnpe"][t_id, x_id, z_id],
			self.data["lnpi"][t_id, x_id, z_id],
			self.data["vort"][t_id, x_id, z_id],
		)


def make_data_loader(
	batch_size: int,
	shuffle: bool = True,
	num_workers: int = 0,
	drop_last: bool = True,
	system: BOUTHESELSystem | None = None,
):
	dataset = HESELIndexDataset(system or BOUTHESELSystem())
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		drop_last=drop_last,
	)


def ids_to_inputs(
	x_ids: torch.Tensor,
	z_ids: torch.Tensor,
	t_ids: torch.Tensor,
	system: BOUTHESELSystem | None = None,
	data: Mapping[str, Any] | None = None,
):
	if data is None:
		data = (system or BOUTHESELSystem()).dump_data
	nt, nx, nz = data["lnn"].shape
	x = (x_ids.float() / max(nx - 1, 1)).unsqueeze(1)
	z = (z_ids.float() / max(nz - 1, 1)).unsqueeze(1)
	t = (t_ids.float() / max(nt - 1, 1)).unsqueeze(1)
	return x, z, t
