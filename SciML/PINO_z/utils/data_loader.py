import torch
from torch.utils.data import DataLoader, Dataset, Subset


FIELDS = ("lnn", "lnpe", "lnpi", "phi")
AVG_FIELDS = ("avg_n", "avg_te", "avg_ti", "avg_phi")


class HESEL_z_Dataset(Dataset):
    def __init__(self, info, m: int = 3):
        self.m = m
        self.info = info
        self.dtype = info.dtype

        self.nz = info.parameters.num_z
        self.nt = info.parameters.num_t
        self.dz = info.parameters.dz
        self.dt = info.parameters.dt

        half_width = self.m // 2
        self.z_offsets = torch.arange(-half_width, half_width + 1, dtype=torch.long)
        self.x_idx = torch.arange(info.parameters.num_x, dtype=torch.long).view(1, info.parameters.num_x, 1)

        self.data = torch.stack([getattr(info.data, name) for name in FIELDS], dim=1)
        self.avg_z = torch.stack([info.avg_in_z[name].squeeze(-1) for name in AVG_FIELDS], dim=1)
        self.nt = self.data.shape[0]

    def __len__(self):
        return (self.nt - 2) * self.nz

    def _window(self, t_idx: int, z_idx: int):
        return self.data[t_idx, :, :, (z_idx + self.z_offsets) % self.nz]

    def __getitem__(self, idx):
        t_idx, z_idx = divmod(idx, self.nz)
        z_idx_window = (z_idx + self.z_offsets) % self.nz
        shape = (1, self.info.parameters.num_x, self.m)


        u = torch.cat([self._window(t_idx, z_idx), self._window(t_idx + 1, z_idx)], dim=0)
        y = self._window(t_idx + 2, z_idx)
        
        avg_z = self.avg_z[t_idx + 2].unsqueeze(-1).expand(-1, -1, self.m)

        # Kun center for z og t, wrapper expander selv ud
        coords_fys = torch.tensor([self.dz * z_idx, self.dt * (t_idx + 2)], dtype=self.dtype)

        x_idx_expanded = self.x_idx.expand(*shape)
        z_idx_expanded = z_idx_window.view(1, 1, self.m).expand(*shape)
        t_idx_tensor = torch.full(shape, t_idx + 2, dtype=torch.long)

        coords_num = torch.cat([x_idx_expanded, z_idx_expanded, t_idx_tensor], dim=0)

        return u, y, avg_z, coords_fys, coords_num


def make_dataloaders(info, data_loader_config):
    dataset = HESEL_z_Dataset(info, m=data_loader_config.get("z_width"))
    n_total = len(dataset)
    n_train = int(data_loader_config.get("train_split") * n_total)
    n_val = int(data_loader_config.get("val_split") * n_total)
    splits = ((0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_total))
    kwargs = {
        "batch_size": data_loader_config.get("batch_size"),
        "shuffle": data_loader_config.get("shuffle"),
        "num_workers": data_loader_config.get("num_workers"),
        "pin_memory": data_loader_config.get("pin_memory"),
        "prefetch_factor": data_loader_config.get("prefetch_factor"),
    }
    if data_loader_config.get("TSSplit"):
        return tuple(DataLoader(Subset(dataset, range(start, end)), **kwargs) for start, end in splits)

    elif not data_loader_config.get("TSSplit"): # Random split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_val, n_total - n_train - n_val])
        return tuple(DataLoader(ds, **kwargs) for ds in (train_dataset, val_dataset, test_dataset))
    else:
        raise ValueError("Invalid data_loader_config: TSSplit must be a boolean.")
