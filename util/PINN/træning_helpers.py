from __future__ import annotations
# Fiks path for imports
from itertools import islice
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn

from util.PINN.model import PINN
from util.PINN.structure import NN_STRUCTURE
from util.PINN.data_loader import make_dataloader, move_batch_to_device

import json


from pathlib import Path
from dataclasses import dataclass, field, asdict




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

def init_trainer(training_config: dict):
    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys

    root = Path(__file__).parents[2] / training_config["root"]
    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info)

    condition = nn.MSELoss().to(info.device)
    model = PINN(NN_STRUCTURE)
    model.to(info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    train_loader, val_loader, test_loader = make_dataloader(info, training_config)

    return (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader)

def save_checkpoint(training_config: dict,
                    model: torch.nn.Module | None = None,
                    condition: torch.nn.Module| None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    epoch: int| None = None,
                    split: str| None = None,
                    batch_idx: int | None = None,
                    best_val_loss: float| None = None,
                    patience_counter: int| None = None,
                    ):
    
    # Load eksisterende checkpoint hvis det findes
    out_folder = Path(training_config["out_folder"])
    if not out_folder.is_absolute():
        out_folder = ROOT_DIR / out_folder
    
    filename = out_folder / "checkpoint.pt"
    
    if filename.exists():
        ckpt = torch.load(filename, weights_only=False)
    else:
        ckpt = {}
    
    ckpt["torch_rng_state"] = torch.get_rng_state().cpu().numpy().tobytes()

    if model is not None:
        ckpt["model_structure"] = model.structure
        ckpt["model_state_dict"] = model.state_dict()

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if condition is not None:
        ckpt["condition_state_dict"] = condition.state_dict()

    if epoch is not None:
        ckpt["epoch"] = epoch

    if split is not None:
        ckpt["split"] = split
    
    if batch_idx is not None:
        ckpt["batch_idx"] = batch_idx

    if best_val_loss is not None:
        ckpt["best_val_loss"] = best_val_loss
    
    if patience_counter is not None:
        ckpt["patience_counter"] = patience_counter


    if training_config is not None:
        ckpt["training_config"] = training_config
    

    
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filename)



def step(info,
         phys,
         model: torch.nn.Module,
         avg_z: torch.Tensor,
         cord_fys: torch.Tensor,
         cord_num: torch.Tensor,
         input: torch.Tensor,
         target: torch.Tensor,
         condition: torch.nn.Module,
         training_config: dict,
         history: HistoryBuffer,
         split: str):
    
    cord_fys = cord_fys.clone().detach().requires_grad_(True)
    state = model.forward(info.standardized, cord_fys, input)
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

    weighted_loss = {
        name: training_config[name] * loss
        for name, loss in losses.items()
        if name != "total"
    }

    losses["total"] = torch.stack(list(losses.values())).sum()
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


def iterate(info,
            phys,
            optimizer: torch.optim.Optimizer,
            model: torch.nn.Module,
            condition: torch.nn.Module,
            batch_iter: islice,
            num_batches: int,
            history: HistoryBuffer,
            training_config: dict,
            split: str,
            first_batch: int
            ):

    if split == "training":
        model.train()
    else:
        model.eval()

    avg_iter_loss = 0.0

    for batch_idx, batch in enumerate(batch_iter, start=first_batch):
        avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])
        loss_total = step(info,
                          phys,
                          model,
                          avg_z,
                          cord_fys,
                          cord_num,
                          input_data,
                          target,
                          condition,
                          training_config,
                          history,
                          split=split
                        )
        
        avg_iter_loss += loss_total.item()

        if split == "training":
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

        if batch_idx % training_config["status_frequency"] == 0:
            print(f"{split} {batch_idx}/{num_batches}\t Loss: {loss_total.item():.6f}")

        if batch_idx % training_config["flush_frequency"] == 0:
            history.flush()
            if split == "training":
                save_checkpoint(training_config,
                                model,
                                optimizer=optimizer,
                                batch_idx=batch_idx)
            else:
                save_checkpoint(training_config,batch_idx=batch_idx)
    
    avg_iter_loss /= num_batches

    return avg_iter_loss