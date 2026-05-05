from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


FIELDS = ("lnn", "lnpe", "lnpi", "phi")


class BOUTDataset(Dataset):
    def __init__(self, info):
        self.info = info

        self.t_line = self.info.t_line
        self.x_line = self.info.x_line
        self.z_line = self.info.z_line

        self.fields = {
            name: torch.as_tensor(getattr(self.info.data, name), dtype=torch.float32)
            for name in FIELDS
        }
        self.x_grid, self.z_grid = self._build_spatial_grids()

    def _build_spatial_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_grid, z_grid = torch.meshgrid(self.x_line, self.z_line, indexing="ij")
        return x_grid.unsqueeze(-1), z_grid.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.t_line)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        t_value = self.t_line[index].view(1, 1, 1).expand_as(self.x_grid)
        fields = {
            name: values[index].unsqueeze(-1)
            for name, values in self.fields.items()
        }
        return {
            "x": self.x_grid.clone(),
            "z": self.z_grid.clone(),
            "t": t_value.clone(),
            "fields": fields,
        }


class BOUTTimeWindowDataset(Dataset):
    def __init__(self, base_dataset: BOUTDataset, step_indices: list[int]):
        self.base_dataset = base_dataset
        self.step_indices = step_indices

    def __len__(self) -> int:
        return len(self.step_indices)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        step_index = self.step_indices[index]
        return {
            "previous": self.base_dataset[step_index - 1],
            "current": self.base_dataset[step_index],
            "next": self.base_dataset[step_index + 1],
        }


def make_time_split_indices(num_time_steps: int) -> tuple[list[int], list[int], list[int], int, int]:
    train_end = max(int(num_time_steps * 0.8), 3)
    val_end = max(int(num_time_steps * 0.9), train_end + 3)
    val_end = min(val_end, num_time_steps)

    train_steps = list(range(1, max(1, train_end - 1)))
    val_steps = list(range(train_end + 1, max(train_end + 1, val_end - 1)))
    test_steps = list(range(val_end + 1, max(val_end + 1, num_time_steps - 1)))
    return train_steps, val_steps, test_steps, train_end, val_end


def make_time_split_loaders(
    dataset: BOUTDataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict[str, object]]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_steps, val_steps, test_steps, train_end, val_end = make_time_split_indices(len(dataset))

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        BOUTTimeWindowDataset(dataset, train_steps),
        shuffle=True,
        **common_kwargs,
    )
    val_loader = None
    test_loader = None
    if val_steps:
        val_loader = DataLoader(
            BOUTTimeWindowDataset(dataset, val_steps),
            shuffle=False,
            **common_kwargs,
        )
    if test_steps:
        test_loader = DataLoader(
            BOUTTimeWindowDataset(dataset, test_steps),
            shuffle=False,
            **common_kwargs,
        )

    split_info = {
        "num_time_steps": len(dataset),
        "train_time_end_exclusive": train_end,
        "val_time_end_exclusive": val_end,
        "num_train_steps": len(train_steps),
        "num_val_steps": len(val_steps),
        "num_test_steps": len(test_steps),
    }
    return train_loader, val_loader, test_loader, split_info
