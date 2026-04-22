from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch
import wandb


def _init_wandb(
    *,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str | None,
    config: dict[str, object],
):
    if not use_wandb:
        return None

    project = os.environ.get("WANDB_PROJECT", wandb_project)
    entity = wandb_entity or os.environ.get("WANDB_ENTITY")
    mode = wandb_mode or os.environ.get("WANDB_MODE") or (
        "online" if os.environ.get("WANDB_API_KEY") else "offline"
    )

    if mode == "offline":
        print(
            "W&B kører i offline mode, fordi WANDB_API_KEY ikke er sat. "
            "Kør `wandb sync wandb/` senere for at uploade runs."
        )

    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
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
