from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from loss_funktion.bout_data import BOUTHESELData
from loss_funktion.bout_phys import BOUTHESELPhysics
from utils.model import PINN

WANDB_PROJECT = "Bachelor_projekt"
WANDB_MODE = "offline"


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    root: Path
    data_folder: str
    incl_data: bool
    w_data: float
    w_eq: float
    w_bc: float
    w_ic: float
    early_stopping_patience: int
    early_stopping_min_delta: float


def _data_loss(
    model: PINN,
    dataset: BOUTHESELData,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor:
    x_ids, z_ids, t_ids, lnn, lnpe, lnpi, phi = batch
    x_data, z_data, t_data = dataset.ids_to_inputs(x_ids, z_ids, t_ids)
    x_data = x_data.to(device=device, dtype=torch.float32)
    z_data = z_data.to(device=device, dtype=torch.float32)
    t_data = t_data.to(device=device, dtype=torch.float32)
    targets = {
        "lnn": lnn.to(device=device, dtype=torch.float32).view(-1, 1),
        "lnpe": lnpe.to(device=device, dtype=torch.float32).view(-1, 1),
        "lnpi": lnpi.to(device=device, dtype=torch.float32).view(-1, 1),
        "phi": phi.to(device=device, dtype=torch.float32).view(-1, 1),
    }
    predictions = model(x_data, z_data, t_data)
    return torch.stack([F.mse_loss(predictions[name], targets[name]) for name in targets]).sum()


def _training_step(
    *,
    model: PINN,
    physics: BOUTHESELPhysics,
    dataset: BOUTHESELData,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    x_eq: torch.Tensor,
    z_eq: torch.Tensor,
    t_eq: torch.Tensor,
    x_ic: torch.Tensor,
    z_ic: torch.Tensor,
    t_ic: torch.Tensor,
    w_data: float,
    w_eq: float,
    w_bc: float,
    w_ic: float,
) -> dict[str, float]:
    loss_data = _data_loss(model, dataset, batch, device)
    loss_eq = physics.eq_loss(model, x_eq, z_eq, t_eq)
    loss_bc = physics.bc_loss(model, x_eq, z_eq, t_eq)
    loss_ic = physics.ic_loss(model, x_ic, z_ic, t_ic)
    loss = w_data * loss_data + w_eq * loss_eq + w_bc * loss_bc + w_ic * loss_ic

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "total": float(loss.item()),
        "data": float(loss_data.item()),
        "eq": float(loss_eq.item()),
        "bc": float(loss_bc.item()),
        "ic": float(loss_ic.item()),
    }


def _physics_only_training_step(
    *,
    model: PINN,
    physics: BOUTHESELPhysics,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    x_eq: torch.Tensor,
    z_eq: torch.Tensor,
    t_eq: torch.Tensor,
    x_ic: torch.Tensor,
    z_ic: torch.Tensor,
    t_ic: torch.Tensor,
    w_eq: float,
    w_bc: float,
    w_ic: float,
) -> dict[str, float]:
    loss_data = torch.zeros((), device=device)
    loss_eq = physics.eq_loss(model, x_eq, z_eq, t_eq)
    loss_bc = physics.bc_loss(model, x_eq, z_eq, t_eq)
    loss_ic = physics.ic_loss(model, x_ic, z_ic, t_ic)
    loss = w_eq * loss_eq + w_bc * loss_bc + w_ic * loss_ic

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "total": float(loss.item()),
        "data": float(loss_data.item()),
        "eq": float(loss_eq.item()),
        "bc": float(loss_bc.item()),
        "ic": float(loss_ic.item()),
    }


def _log_step(epoch: int, batch: int, global_step: int, losses: dict[str, float]) -> None:
    wandb.log(
        {
            "epoch": epoch,
            "batch": batch,
            "train_step/total_loss": losses["total"],
            "train_step/data_loss": losses["data"],
            "train_step/eq_loss": losses["eq"],
            "train_step/bc_loss": losses["bc"],
            "train_step/ic_loss": losses["ic"],
        },
        step=global_step,
    )


def _init_wandb(*, config: dict[str, object]):
    return wandb.init(
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        config=config,
    )


def _serialize_nn_structure(nn_structure: dict[str, object]) -> dict[str, object]:
    return {
        "input_size": nn_structure["input_size"],
        "output_size": nn_structure["output_size"],
        "output_names": nn_structure["output_names"],
        "layers": [
            {
                "size": layer["size"],
                "non_lin_foo": layer["non_lin_foo"].__name__,
            }
            for layer in nn_structure["layers"]
        ],
    }


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    model_path: Path,
    nn_structure: dict[str, object],
    best_epoch: int,
    best_total_loss: float,
    completed_epochs: int,
    stopped_early: bool,
    training_config: dict[str, object],
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "nn_structure": _serialize_nn_structure(nn_structure),
            "training_config": training_config,
            "best_epoch": best_epoch,
            "best_total_loss": best_total_loss,
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
        },
        model_path,
    )
    return model_path


def _default_model_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(__file__).resolve().parents[1] / "data" / f"trained_pinn_{timestamp}.pt"
