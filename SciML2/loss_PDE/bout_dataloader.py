from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from loss_PDE.bout_dump import BOUTHESELInfo


FIELDS = ["lnn", "lnpe", "lnpi", "phi"]


class BOUTDataset(Dataset):
    def __init__(self, root):
        info = BOUTHESELInfo(root)
        data = info.data

        self.samples = torch.stack(
            [torch.as_tensor(getattr(data, field), dtype=torch.float32) for field in FIELDS], dim=1,
        )

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]


def make_dataloader(batch_size, shuffle, root):
    dataset = BOUTDataset(root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
