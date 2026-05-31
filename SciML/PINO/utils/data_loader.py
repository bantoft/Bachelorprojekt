import torch

from torch.utils.data import Dataset, DataLoader, Subset


class HESELOneStepDataset(Dataset):

    def __init__(self, info):
        self.info = info

        self.data = torch.stack([
            info.data.lnn,
            info.data.lnpe,
            info.data.lnpi,
            info.data.vort,
        ], dim=1).to(dtype=info.dtype)
        # [nt, 4, nx, nz]

        self.t = (
            torch.arange(
                self.data.shape[0],
                dtype=info.dtype,
            )
            * info.parameters.dt
        )

    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx):

        u = self.data[idx]          # [4, nx, nz]
        t = self.t[idx + 1]         # scalar
        y = self.data[idx + 1]      # [4, nx, nz]

        return u, t, y


def make_dataloaders(
    info,
    batch_size=4,
    train_split=0.8,
    val_split=0.1,
):

    dataset = HESELOneStepDataset(info)

    n_total = len(dataset)

    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)

    train_indices = range(0, n_train)

    val_indices = range(
        n_train,
        n_train + n_val,
    )

    test_indices = range(
        n_train + n_val,
        n_total,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
