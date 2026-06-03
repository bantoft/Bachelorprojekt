import torch

from torch.utils.data import DataLoader, Dataset, Subset


class HESELOneStepDataset(Dataset):
    def __init__(self, info, m: int = 3):
        if m <= 0 or m % 2 != 1:
            raise ValueError("m skal vaere et positivt ulige tal.")

        self.m = m
        self.info = info
        self.dtype = self.info.dtype

        self.nx = self.info.parameters.num_x
        self.nz = self.info.parameters.num_z
        self.nt = self.info.parameters.num_t

        self.dx = self.info.parameters.dx
        self.dz = self.info.parameters.dz
        self.dt = self.info.parameters.dt

        self.lx = self.info.parameters.Lx
        self.lz = self.info.parameters.Lz
        self.lt = self.info.parameters.Lt

        self.nt_samples = self.nt - 1
        self.t = torch.arange(self.nt, dtype=self.dtype) * self.dt

        # [nt, 4, nx]
        self.avg_z = torch.stack((self.info.avg_in_z["avg_n"].squeeze(-1),
                                  self.info.avg_in_z["avg_te"].squeeze(-1),
                                  self.info.avg_in_z["avg_ti"].squeeze(-1),
                                  self.info.avg_in_z["avg_phi"].squeeze(-1)), dim=1).to(dtype=self.dtype)

        # [nt, 4, nx, nz]
        self.data = torch.stack([self.info.data.lnn,
                                 self.info.data.lnpe,
                                 self.info.data.lnpi,
                                 self.info.data.phi], dim=1).to(dtype=self.dtype)

        half = self.m // 2
        self.z_offsets = torch.arange(-half, half + 1, dtype=torch.long)
        self.x_idx = torch.arange(self.nx, dtype=torch.long).view(1, self.nx, 1)

    def __len__(self):
        return self.nt_samples * self.nz

    def _decode_index(self, idx: int) -> tuple[int, int]:
        time_idx = idx // self.nz
        z_center_idx = idx % self.nz
        return time_idx, z_center_idx

    def __getitem__(self, idx):
        time_idx, z_center_idx = self._decode_index(idx)

        z_idx = (z_center_idx + self.z_offsets) % self.nz
        z_idx_grid = z_idx.view(1, 1, self.m).expand(1, self.nx, self.m)
        
        t_idx = torch.full((1, self.nx, self.m), time_idx + 1, dtype=torch.long)

        avg_z = self.avg_z[time_idx + 1].unsqueeze(-1).expand(-1, -1, self.m)
        u_subset = self.data[time_idx].index_select(-1, z_idx)
        t = self.t[time_idx + 1]
        y_subset = self.data[time_idx + 1].index_select(-1, z_idx)
        cord_num = torch.cat([self.x_idx.expand(1, -1, self.m), z_idx_grid, t_idx], dim=0)

        return avg_z, u_subset, t, y_subset, cord_num


def make_dataloaders(info,
                     batch_size=1,
                     train_split=0.8,
                     val_split=0.1,
                     num_workers=1,
                     prefetch_factor=2,
                     pin_memory=True,
                     shuffle=True,
                     m=3
                     ):
    
    if pin_memory is None:
        pin_memory = info.device.type == "cuda"

    dataset = HESELOneStepDataset(info, m=m)

    n_time_total = dataset.nt_samples
    n_train_time = int(train_split * n_time_total)
    n_val_time = int(val_split * n_time_total)
    n_test_time = n_time_total - n_train_time - n_val_time

    if n_train_time <= 0 or n_val_time < 0 or n_test_time <= 0:
        raise ValueError("train/val/test split gav ugyldige tidsblokke.")

    samples_per_time = dataset.nz

    train_indices = range(0, n_train_time * samples_per_time)
    val_stop = (n_train_time + n_val_time) * samples_per_time
    test_stop = (n_train_time + n_val_time + n_test_time) * samples_per_time

    val_indices = range(n_train_time * samples_per_time, val_stop)
    test_indices = range(val_stop, test_stop)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    return train_loader, val_loader, test_loader
