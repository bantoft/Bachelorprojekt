import torch
from torch.utils.data import DataLoader, Dataset, Subset


FIELDS = ("lnn", "lnpe", "lnpi", "phi")
AVG_FIELDS = ("avg_n", "avg_te", "avg_ti", "avg_phi")


class HESEL_z_Dataset(Dataset):
    def __init__(self, info, m: int = 3):
        self.info = info
        self.dtype = info.dtype

        self.start_t = 8

        self.m = m
        self.half = m // 2

        self.nz = info.parameters.num_z
        self.nt = info.parameters.num_t
        self.dz = info.parameters.dz
        self.dt = info.parameters.dt

        self.z_offsets = torch.arange(-self.half, self.half + 1, dtype=torch.long)
        self.x_idx = torch.arange(info.parameters.num_x, dtype=torch.long).view(1, info.parameters.num_x, 1)

        self.data = torch.stack([getattr(info.data, name)[self.start_t:] for name in FIELDS], dim=1)
        self.avg_z = torch.stack([info.avg_in_z[name].squeeze(-1)[self.start_t:] for name in AVG_FIELDS], dim=1)
        self.nt = self.data.shape[0]

    def __len__(self):
        return (self.nt - 1) * self.nz

    def _window(self, t_idx: int, z_idx: int):
        return self.data[t_idx, :, :, (z_idx + self.z_offsets) % self.nz]

    def __getitem__(self, idx):
        t_idx, z_idx = divmod(idx, self.nz)
        z_idx_window = (z_idx + self.z_offsets) % self.nz
        shape = (1, self.info.parameters.num_x, self.m)


        u = self._window(t_idx, z_idx)
        y = self._window(t_idx + 1, z_idx)
        
        avg_z = self.avg_z[t_idx + 1].unsqueeze(-1).expand(-1, -1, self.m)

        # Kun center for z og t
        coords_fys = torch.tensor([self.dz * z_idx, self.dt * (t_idx + self.start_t + 1)], dtype=self.dtype)

        x_idx_expanded = self.x_idx.expand(*shape)
        z_idx_expanded = z_idx_window.view(1, 1, self.m).expand(*shape)
        t_idx_tensor = torch.full(shape, t_idx + self.start_t + 1, dtype=torch.long)

        coords_num = torch.cat([x_idx_expanded, z_idx_expanded, t_idx_tensor], dim=0)

        return u, y, avg_z, coords_fys, coords_num


def make_dataloaders(info, train_config):
    dataset = HESEL_z_Dataset(info, m=train_config.get("z_width"))
    n_total = len(dataset)
    n_train = int(train_config.get("train_split") * n_total)
    n_val = int(train_config.get("val_split") * n_total)
    splits = ((0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_total))
    kwargs = {
        "batch_size": train_config.get("batch_size"),
        "shuffle": train_config.get("shuffle"),
        "num_workers": train_config.get("num_workers"),
        "pin_memory": train_config.get("pin_memory"),
        "prefetch_factor": train_config.get("prefetch_factor"),
    }
    return tuple(DataLoader(Subset(dataset, range(start, end)), **kwargs) for start, end in splits)
