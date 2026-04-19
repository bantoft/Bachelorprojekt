import torch
from torch.utils.data import DataLoader, Dataset

from SciML.loss.data import data


class HESELIndexDataset(Dataset):
	def __init__(self):
		super().__init__()
		self.nx, self.ny, self.nt = data["lnn"].shape
		self.total_size = self.nx * self.ny * self.nt

	def __len__(self):
		return self.total_size

	def __getitem__(self, idx):
		idx = int(idx)
		x_id = idx // (self.ny * self.nt)
		rest = idx % (self.ny * self.nt)
		y_id = rest // self.nt
		t_id = rest % self.nt

		return (
			torch.tensor(x_id, dtype=torch.long),
			torch.tensor(y_id, dtype=torch.long),
			torch.tensor(t_id, dtype=torch.long),
			data["lnn"][x_id, y_id, t_id],
			data["lnpe"][x_id, y_id, t_id],
			data["lnpi"][x_id, y_id, t_id],
			data["vort"][x_id, y_id, t_id],
		)


def make_data_loader(batch_size: int, shuffle: bool = True, num_workers: int = 0, drop_last: bool = True):
	dataset = HESELIndexDataset()
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		drop_last=drop_last,
	)


def ids_to_inputs(x_ids: torch.Tensor, y_ids: torch.Tensor, t_ids: torch.Tensor):
	nx, ny, nt = data["lnn"].shape
	x = (x_ids.float() / max(nx - 1, 1)).unsqueeze(1)
	y = (y_ids.float() / max(ny - 1, 1)).unsqueeze(1)
	t = (t_ids.float() / max(nt - 1, 1)).unsqueeze(1)
	return x, y, t
