from __future__ import annotations

import gc
import json
import torch


from pathlib import Path
from dataclasses import dataclass, field, asdict




@dataclass
class LossGroup:
    total: list[float] = field(default_factory=list)
    da: list[float] = field(default_factory=list)
    eq: list[float] = field(default_factory=list)


@dataclass
class SplitHistory:
    raw: LossGroup = field(default_factory=LossGroup)
    weighted: LossGroup = field(default_factory=LossGroup)

@dataclass
class HistoryData:
    training: SplitHistory = field(default_factory=SplitHistory)
    validation: SplitHistory = field(default_factory=SplitHistory)
    test: SplitHistory = field(default_factory=SplitHistory)


class HistoryBuffer:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.buffer = HistoryData()

        if not self.save_path.exists(): self._save(HistoryData())

    def append(self, split: str, loss_type: str, values: dict[str, float]):
        split = split.lower()
        group: LossGroup = getattr(getattr(self.buffer, split), loss_type)
        for key, value in values.items():
            getattr(group, key).append(float(value))



    def flush(self):
        history = self._load()
        self._extend(history.training, self.buffer.training)
        self._extend(history.validation, self.buffer.validation)
        self._extend(history.test, self.buffer.test)

        self._save(history)
        self.buffer = HistoryData()


    def _extend(self, history_split: SplitHistory, buffer_split: SplitHistory):
        self._extend_group(history_split.raw, buffer_split.raw)
        self._extend_group(history_split.weighted, buffer_split.weighted)


    def _extend_group(self, history_group: LossGroup, buffer_group: LossGroup):
        for key in vars(history_group):
            getattr(history_group, key).extend(getattr(buffer_group, key))


    def _save(self, history: HistoryData):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(asdict(history), f, indent=2)

    def _load(self) -> HistoryData:
        with open(self.save_path, "r") as f:
            data = json.load(f)

        return HistoryData(
            training=SplitHistory(
                raw=LossGroup(**data["training"]["raw"]),
                weighted=LossGroup(**data["training"]["weighted"]),
            ),
            validation=SplitHistory(
                raw=LossGroup(**data["validation"]["raw"]),
                weighted=LossGroup(**data["validation"]["weighted"]),
            ),
            test=SplitHistory(
                raw=LossGroup(**data["test"]["raw"]),
                weighted=LossGroup(**data["test"]["weighted"]),
            ),
        )


def cleanup_phase(model, optimizer=None):
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iterater(
    model,
    dataloader,
    condition,
    optimizer,
    history,
    phys,
    info,
    split,
    status_frequency,
    flush_frequency,
    eq_weight=1.0,
):
    is_training = split == "training"
    was_training = model.training
    param_requires_grad = [param.requires_grad for param in model.parameters()]

    if is_training:
        model.train()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    running_raw = {"total": 0.0, "da": 0.0, "eq": 0.0}
    running_weighted = {"total": 0.0, "da": 0.0, "eq": 0.0}
    window_count = 0

    try:
        for batch_idx, batch in enumerate(dataloader):
            avg_z, u_subset, t, y_subset, cord_num = [x.to(info.device) for x in batch]

            f_theta, cord = model(u_subset, t)
            f_theta_tensor = torch.cat([f_theta[name] for name in ("lnn", "lnpe", "lnpi", "phi")], dim=1)
            data_loss = condition(f_theta_tensor, y_subset)

            eq_residuals = phys.eq_res(avg_z, f_theta, cord, cord_num)
            eq_residuals_stack = torch.stack(list(eq_residuals.values()))
            eq_loss = condition(eq_residuals_stack, torch.zeros_like(eq_residuals_stack))

            raw_total = data_loss + eq_loss
            weighted_eq_loss = eq_weight * eq_loss
            weighted_total = data_loss + weighted_eq_loss

            history.append(
                split=split,
                loss_type="raw",
                values={
                    "total": raw_total.detach().cpu().item(),
                    "da": data_loss.detach().cpu().item(),
                    "eq": eq_loss.detach().cpu().item(),
                },
            )
            history.append(
                split=split,
                loss_type="weighted",
                values={
                    "total": weighted_total.detach().cpu().item(),
                    "da": data_loss.detach().cpu().item(),
                    "eq": weighted_eq_loss.detach().cpu().item(),
                },
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                weighted_total.backward()
                optimizer.step()

            running_raw["total"] += raw_total.detach().cpu().item()
            running_raw["da"] += data_loss.detach().cpu().item()
            running_raw["eq"] += eq_loss.detach().cpu().item()
            running_weighted["total"] += weighted_total.detach().cpu().item()
            running_weighted["da"] += data_loss.detach().cpu().item()
            running_weighted["eq"] += weighted_eq_loss.detach().cpu().item()
            window_count += 1

            if status_frequency and (batch_idx + 1) % status_frequency == 0:
                print(
                    f"{split} batch {batch_idx + 1}/{len(dataloader)} | "
                    f"avg_total={running_raw['total'] / window_count} | "
                    f"avg_total_w={running_weighted['total'] / window_count} | "
                    f"avg_data={running_raw['da'] / window_count} | "
                    f"avg_eq={running_raw['eq'] / window_count} | "
                    f"avg_eq_w={running_weighted['eq'] / window_count}"
                )
                running_raw = {"total": 0.0, "da": 0.0, "eq": 0.0}
                running_weighted = {"total": 0.0, "da": 0.0, "eq": 0.0}
                window_count = 0

            if flush_frequency and (batch_idx + 1) % flush_frequency == 0:
                history.flush()

            del (
                avg_z,
                u_subset,
                t,
                y_subset,
                cord_num,
                f_theta,
                cord,
                f_theta_tensor,
                data_loss,
                eq_residuals,
                eq_residuals_stack,
                eq_loss,
                raw_total,
                weighted_eq_loss,
                weighted_total,
            )
            if batch_idx == 2000: break

        if window_count > 0:
            print(
                f"{split} batch {len(dataloader)}/{len(dataloader)} | "
                f"avg_total={running_raw['total'] / window_count} | "
                f"avg_total_w={running_weighted['total'] / window_count} | "
                f"avg_data={running_raw['da'] / window_count} | "
                f"avg_eq={running_raw['eq'] / window_count} | "
                f"avg_eq_w={running_weighted['eq'] / window_count}"
            )
    finally:
        for param, requires_grad in zip(model.parameters(), param_requires_grad):
            param.requires_grad_(requires_grad)
        model.train(was_training)



def init_experinment(train_cfg, fno):
    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys

    from SciML.PINO_z.utils.model import WrappedFNO
    from SciML.PINO_z.utils.data_loader import make_dataloaders

    info = BOUTHESELInfo(train_cfg["root"])
    phys = BOUTHESELPhys(info)

    data_dir = (train_cfg["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryBuffer(data_dir / "history.json")

    train_loader, val_loader, test_loader = make_dataloaders(info,
                                                             train_split=train_cfg["train_split"],
                                                             val_split=train_cfg["val_split"],
                                                             batch_size=train_cfg["batch_size"],
                                                             shuffle=train_cfg["shuffle"],
                                                             num_workers=train_cfg["num_workers"],
                                                             prefetch_factor=train_cfg["prefetch_factor"],
                                                             pin_memory=train_cfg["pin_memory"],
                                                             m=train_cfg["m"])

    model = WrappedFNO(fno, info, phys, m=train_cfg["m"]).to(info.device)

    condition = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    return model, train_loader, val_loader, test_loader, condition, optimizer, history, phys, info