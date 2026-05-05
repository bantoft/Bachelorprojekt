from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import wandb


def _load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_paths = [repo_root / ".env", repo_root / "SciML2" / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_local_env()

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Bachelor_projekt")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

_configured_mode = os.getenv("WANDB_MODE")
if _configured_mode is None:
    _configured_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
WANDB_MODE = cast(Literal["online", "offline", "disabled", "shared"], _configured_mode)


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    root: Path
    batch_size: int
    num_workers: int
    eq_weight: float
    early_stopping_patience: int
    early_stopping_min_delta: float


def _init_wandb(*, config: dict[str, object]):
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        config=config,
    )


def _log_step(epoch: int, batch: int, global_step: int, losses: dict[str, float]) -> None:
    wandb.log(
        {
            "epoch": epoch,
            "batch": batch,
            "train_step/total_loss": losses["total"],
            "train_step/data_loss": losses["data"],
            "train_step/ic_loss": losses["ic"],
            "train_step/bc_loss": losses["bc"],
            "train_step/eq_loss": losses["eq"],
        },
        step=global_step,
    )


def _log_epoch_summary(
    *,
    epoch: int,
    global_step: int,
    averages: dict[str, float],
    best_total_loss: float,
    epochs_without_improvement: int,
    val_averages: dict[str, float] | None = None,
    test_averages: dict[str, float] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "epoch_summary/total_loss": averages["total"],
        "epoch_summary/data_loss": averages["data"],
        "epoch_summary/ic_loss": averages["ic"],
        "epoch_summary/bc_loss": averages["bc"],
        "epoch_summary/eq_loss": averages["eq"],
        "epoch_summary/best_total_loss": best_total_loss,
        "epoch_summary/epochs_without_improvement": epochs_without_improvement,
    }
    if val_averages is not None:
        payload.update({
            "epoch_summary/val_total_loss": val_averages["total"],
            "epoch_summary/val_data_loss": val_averages["data"],
            "epoch_summary/val_ic_loss": val_averages["ic"],
            "epoch_summary/val_bc_loss": val_averages["bc"],
            "epoch_summary/val_eq_loss": val_averages["eq"],
        })
    if test_averages is not None:
        payload.update({
            "epoch_summary/test_total_loss": test_averages["total"],
            "epoch_summary/test_data_loss": test_averages["data"],
            "epoch_summary/test_ic_loss": test_averages["ic"],
            "epoch_summary/test_bc_loss": test_averages["bc"],
            "epoch_summary/test_eq_loss": test_averages["eq"],
        })
    wandb.log(payload, step=global_step)


def _standardization_to_scaling(standardization, output_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(standardization.mean[index].item()),
            "std": float(standardization.std[index].item()),
            "variance": float(standardization.std[index].item() ** 2),
        }
        for index, name in enumerate(output_names)
    }


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    model_path: Path,
    output_names: tuple[str, ...],
    standardization,
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "output_scaling": _standardization_to_scaling(standardization, output_names),
        },
        model_path,
    )
    return model_path


def _save_training_history(history: dict[str, list[float]], history_path: Path) -> Path:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(history, history_path)
    return history_path


def _default_model_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(__file__).resolve().parents[1] / "data" / f"trained_pinn_{timestamp}.pt"


def _default_history_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "training_history.pt"
