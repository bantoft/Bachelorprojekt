from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


class WeightedTrainSampler(Sampler[int]):
    def __init__(self, dataset, train_end: int, min_t: int = 50, x_weight: float = 3.0):
        self.dataset = dataset
        self.train_end = train_end
        self.min_t = min_t
        self.x_weight = x_weight
        self.num_samples = train_end * dataset.num_x * dataset.num_z

        # Hvor langt ind i x-dimensionen skal have øget vægt? (50/257) og (200/257) er valgt for at fokusere på midten af x-dimensionen
        self.x_min = int(dataset.num_x * (50 / 257))
        self.x_max = int(dataset.num_x * (200 / 257))

        t_idx = torch.arange(train_end, dtype=torch.long).repeat_interleave(dataset.num_x)
        x_idx = torch.arange(dataset.num_x, dtype=torch.long).repeat(train_end)

        weights = torch.ones(t_idx.numel(), dtype=torch.double)
        weights[t_idx < min_t] = 0.0
        x_window = (x_idx >= self.x_min) & (x_idx <= self.x_max) & (t_idx >= min_t)
        weights[x_window] = x_weight

        self._pair_weights = weights
        self._num_pairs = weights.numel()

    def __iter__(self):
        pair_ids = torch.multinomial(self._pair_weights, self.num_samples, replacement=True)
        t_idx = pair_ids // self.dataset.num_x
        x_idx = pair_ids % self.dataset.num_x
        z_idx = torch.randint(self.dataset.num_z, (self.num_samples,), dtype=torch.long)

        flat_indices = (
            t_idx * self.dataset.num_x * self.dataset.num_z
            + z_idx * self.dataset.num_x
            + x_idx
        )

        return iter(flat_indices.tolist())

    def __len__(self):
        return self.num_samples

def collate_stack(batch):
    """Stack a batch on CPU; device transfer happens in the training loop."""
    avg_z, cord_fys, cord_num, input, target = zip(*batch)
    return (
        torch.stack(avg_z),
        torch.stack(cord_fys),
        torch.stack(cord_num),
        torch.stack(input),
        torch.stack(target),
    )


def move_batch_to_device(batch, device, non_blocking: bool = False):
    """Move a collated batch to the requested device."""
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


class BOUTDataset(Dataset):
    def __init__(self, info, data):
        self.dtype = info.dtype
        self.data = data
        self.num_x = info.parameters.num_x
        self.num_z = info.parameters.num_z
        self.num_t = info.parameters.num_t
        self.dx = info.parameters.dx
        self.dz = info.parameters.dz
        self.dt = info.parameters.dt


        self.avg_z = torch.stack((info.avg_in_z["avg_n"].squeeze(-1),
                                  info.avg_in_z["avg_te"].squeeze(-1),
                                  info.avg_in_z["avg_ti"].squeeze(-1),
                                  info.avg_in_z["avg_phi"].squeeze(-1)), dim=-1).to(dtype=self.dtype)



    def __len__(self):
        return self.num_x * self.num_z * (self.num_t - 2)

    def __getitem__(self, index):
        # Få x, z, t indeks ved x1. z2. t3.
        x_idx = index % self.num_x
        z_idx = (index // self.num_x) % self.num_z
        t_idx = index // (self.num_x * self.num_z)

        # Target avg_z, bliver brugt til phys.eq_res()
        avg_z = self.avg_z[t_idx + 2, x_idx]

        z_cord_fys = torch.rand(1, dtype=self.dtype) * self.num_z * self.dz
        z_cord_fys = z_cord_fys.squeeze(0)

        cord_num = torch.tensor([x_idx, z_idx, t_idx + 2], dtype=self.dtype)

        cord_fys = torch.tensor([x_idx * self.dx, z_cord_fys , (t_idx + 2) * self.dt], dtype=self.dtype)

        # input (batch, 72)
        input = self.data[t_idx:t_idx + 2, :, x_idx:x_idx + 3, z_idx:z_idx + 3]
        # target (batch, 4)
        target = self.data[t_idx + 2, :, x_idx + 1, z_idx + 1]

        return avg_z, cord_fys, cord_num, input, target


def make_dataloader(info, training_config: dict):
    # samler data og padder (x,z) med replicat: (875, 4, 130, 128) -> (875, 4, 132, 130)
    data = F.pad(torch.stack([info.data.lnn,
                              info.data.lnpe,
                              info.data.lnpi,
                              info.data.phi], dim=1).float(), pad=(1, 1, 1, 1), mode="replicate")

    dataset = BOUTDataset(info, data)

    n_total = len(dataset)
    n_train = int(training_config["train_ratio"] * n_total)
    n_val = int(training_config["val_ratio"] * n_total)
    n_test = n_total - n_train - n_val

    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["num_workers"],
        "pin_memory": training_config["pin_memory"],
        "persistent_workers": training_config["persistent_workers"],
        "prefetch_factor": training_config["prefetch_factor"],
        "collate_fn": collate_stack,
    }

    if training_config.get("TSSplit", True):
        train_dataset = Subset(dataset, range(0, n_train))
        val_dataset = Subset(dataset, range(n_train, n_train + n_val))
        test_dataset = Subset(dataset, range(n_train + n_val, n_total))

        train_steps = len(train_dataset) // (dataset.num_x * dataset.num_z)
        train_sampler = WeightedTrainSampler(dataset, train_steps, min_t=50, x_weight=3.0)

        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  **loader_kwargs)
    else:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        
        train_loader = DataLoader(train_dataset,
                                  shuffle=training_config["shuffle"],
                                  **loader_kwargs)

    val_loader = DataLoader(val_dataset,
                            shuffle=training_config["shuffle"],
                            **loader_kwargs)

    test_loader = DataLoader(test_dataset,
                             shuffle=training_config["shuffle"],
                             **loader_kwargs)

    return train_loader, val_loader, test_loader
