from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, asdict
import json

import torch


@dataclass
class LossGroup:
    total: list[float] = field(default_factory=list)
    da: list[float] = field(default_factory=list)
    eq: list[float] = field(default_factory=list)
    ic: list[float] = field(default_factory=list)
    bc: list[float] = field(default_factory=list)


@dataclass
class SplitHistory:
    raw: LossGroup = field(default_factory=LossGroup)
    weighted: LossGroup = field(default_factory=LossGroup)


@dataclass
class HistoryData:
    train: SplitHistory = field(default_factory=SplitHistory)
    validation: SplitHistory = field(default_factory=SplitHistory)
    test: SplitHistory = field(default_factory=SplitHistory)

def step(model, info, phys, avg_z, cord_fys, cord_num, input, target, condition, training_config, history, split):
    with torch.enable_grad():
        cord_fys = cord_fys.clone().detach().requires_grad_(True)

        state = model(info.standardized, cord_fys, input)

        pred_da = torch.cat([state["lnn"], state["lnpe"], state["lnpi"], state["phi"]], dim=1)

        res_eq = phys.eq_res(avg_z, state, cord_fys, cord_num)
        res_ic = phys.ic_res(model, cord_fys, cord_num, input, info.standardized)
        res_bc = phys.bc_res(model, cord_fys, input, info.standardized)

        res_ic_stack = torch.stack(list(res_ic.values()))
        res_eq_stack = torch.stack(list(res_eq.values()))
        res_bc_stack = torch.stack(list(res_bc.values()))

        losses = {
            "da": condition(pred_da, target),
            "eq": condition(res_eq_stack, torch.zeros_like(res_eq_stack)),
            "ic": condition(res_ic_stack, torch.zeros_like(res_ic_stack)),
            "bc": condition(res_bc_stack, torch.zeros_like(res_bc_stack)),
        }

        losses["total"] = torch.stack(list(losses.values())).sum()

        weighted_loss = {
            name: training_config[name] * loss
            for name, loss in losses.items()
            if name != "total"
        }

        weighted_loss["total"] = torch.stack(list(weighted_loss.values())).sum()


        history.append(
            split=split,
            loss_type="raw",
            values={
                name: loss.detach().cpu().item()
                for name, loss in losses.items()
            },
        )

        history.append(
            split=split,
            loss_type="weighted",
            values={
                name: loss.detach().cpu().item()
                for name, loss in weighted_loss.items()
            },
        )
        return weighted_loss["total"]



class HistoryBuffer:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.buffer = HistoryData()

        if not self.save_path.exists(): self._save(HistoryData())

    def append(self, split: str, loss_type: str, values: dict[str, float]):
        group: LossGroup = getattr(getattr(self.buffer, split), loss_type)
        for key, value in values.items():
            getattr(group, key).append(float(value))


    def flush(self):
        history = self._load()
        self._extend(history.train, self.buffer.train)
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
            train=SplitHistory(
                raw=LossGroup(**data["train"]["raw"]),
                weighted=LossGroup(**data["train"]["weighted"]),
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


def save_checkpoint(model, optimizer, condition, epoch: int, batch_idx: int, patience_counter: int, best_val_loss: float, training_config: dict, filename: Path):
    ckpt = {
        "model_structure": model.structure,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "condition_state_dict": condition.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "patience_counter": patience_counter,
        "best_val_loss": best_val_loss,
        "training_config": training_config,
        "torch_rng_state": torch.get_rng_state().cpu().numpy().tobytes(),
    }
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filename)