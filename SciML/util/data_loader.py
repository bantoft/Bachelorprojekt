from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset


FIELDS = ("lnn", "lnpe", "lnpi", "phi")

class BOUTDataset(Dataset):
    def __init__(self, info):
        self.info = info
        self.t_line = info.t_line
        
        # Load and pad fields
        self.fields = {
            name: torch.nn.functional.pad(
                torch.as_tensor(getattr(self.info.data, name), dtype=torch.float32),
                (1, 1, 1, 1), mode='replicate'
            )
            for name in FIELDS
        }
        
        # Calculate valid patch center positions
        num_x = self.fields[FIELDS[0]].shape[1]
        num_z = self.fields[FIELDS[0]].shape[2]
        self.patch_centers = [(x, z) for x in range(1, num_x - 1) for z in range(1, num_z - 1)]
        self.num_time_steps = len(self.t_line) - 2

    def __len__(self) -> int:
        return self.num_time_steps * len(self.patch_centers)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        time_idx = index // len(self.patch_centers)
        center_x, center_z = self.patch_centers[index % len(self.patch_centers)]
        slice_x, slice_z = slice(center_x - 1, center_x + 2), slice(center_z - 1, center_z + 2)
        
        inputs = torch.stack([
            torch.stack([
                self.fields[name][time_idx, slice_x, slice_z],
                self.fields[name][time_idx + 1, slice_x, slice_z]
            ], dim=0)
            for name in FIELDS
        ], dim=1)
        
        target = torch.tensor([
            self.fields[name][time_idx + 2, center_x, center_z]
            for name in FIELDS
        ], dtype=torch.float32)
        
        return {
            "inputs": inputs,
            "target": target,
            "t": self.t_line[time_idx + 2].clone().detach(),
            "x": torch.tensor(center_x, dtype=torch.float32),
            "z": torch.tensor(center_z, dtype=torch.float32),
        }


def _collate_fn(batch):
    x = torch.stack([s["x"] for s in batch]).detach().requires_grad_(True)
    z = torch.stack([s["z"] for s in batch]).detach().requires_grad_(True)
    t = torch.stack([s["t"] for s in batch]).detach().requires_grad_(True)
    return {
        "inputs": torch.stack([s["inputs"] for s in batch]),
        "targets": torch.stack([s["target"] for s in batch]),
        "t": t,
        "x": x,
        "z": z,
    }


def make_dataloader(
    info,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    dataset = BOUTDataset(info)
    num_patches = len(dataset.patch_centers)
    num_time_steps = dataset.num_time_steps
    train_end = int(num_time_steps * train_ratio)
    val_end = train_end + int(num_time_steps * val_ratio)

    def time_indices(start: int, end: int) -> list[int]:
        return [time_idx * num_patches + patch_idx for time_idx in range(start, end) for patch_idx in range(num_patches)]

    train_loader = DataLoader(
        Subset(dataset, time_indices(0, train_end)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, time_indices(train_end, val_end)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, time_indices(val_end, num_time_steps)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader, test_loader
