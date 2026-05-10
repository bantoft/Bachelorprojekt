from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import wandb

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Bachelor_projekt")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

_configured_mode = os.getenv("WANDB_MODE")
_configured_mode = "online"
WANDB_MODE = cast(Literal["online", "offline", "disabled", "shared"], _configured_mode)



def _standardization_to_scaling(standardization, output_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(standardization.mean[index].item()),
            "std": float(standardization.std[index].item()),
            "variance": float(standardization.std[index].item() ** 2),
        }
        for index, name in enumerate(output_names)
    }


def _save_training_history(history: dict[str, list[float]], history_path: Path) -> Path:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(history, history_path)
    return history_path


def _default_model_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(__file__).resolve().parents[1] / "data" / f"trained_pinn_{timestamp}.pt"


def _init_wandb(*, config: dict[str, object]):
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        config=config,
    )

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


def _log_split_losses(
    run,
    *,
    split,
    epoch,
    batch_idx,
    step,
    loss_total,
    loss_da,
    loss_ic,
    loss_eq,
    loss_bc,
):
    run.log(
        {
            "epoch": epoch,
            f"{split}/batch": batch_idx,
            f"{split}/losses/total": loss_total,
            f"{split}/losses/da": loss_da,
            f"{split}/losses/ic": loss_ic,
            f"{split}/losses/eq": loss_eq,
            f"{split}/losses/bc": loss_bc,
        },
        step=step,
    )



def _get_loss(model, phys, batch, standardization, condition):

    device = next(model.parameters()).device
    batch = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    pred_state = model.forward(
        standardization,
        batch["x"],
        batch["z"],
        batch["t"],
        batch["inputs"]
    )

    pred_da = torch.cat([pred_state["lnn"], pred_state["lnpe"], pred_state["lnpi"], pred_state["phi"]],dim=1,)

    pred_ic = model.forward(
        standardization,
        batch["x"],
        batch["z"],
        torch.zeros_like(batch["t"]),
        torch.zeros_like(batch["inputs"])
    )


    # Laver ingen prediction
    res_ic = phys.ic_res(pred_ic, batch["x"], batch["z"]) # Dog skal den have en prediction for at kunne køre
    res_eq = phys.eq_res(pred_state, batch["x"], batch["z"], batch["t"])

    # Predicter 4 predictions: xin, xout, zlower, zupper
    res_bc = phys.bc_res(standardization, model, batch["x"], batch["z"], batch["t"], batch["inputs"])
    print(res_bc)

    res_ic_stack = torch.stack(list(res_ic.values()))
    res_eq_stack = torch.stack(list(res_eq.values()))
    res_bc_stack = torch.stack(list(res_bc.values()))

    loss_da = condition(pred_da, batch["targets"])
    loss_ic = condition(res_ic_stack, torch.zeros_like(res_ic_stack))
    loss_eq = condition(res_eq_stack, torch.zeros_like(res_eq_stack))
    loss_bc = condition(res_bc_stack, torch.zeros_like(res_bc_stack))

    return loss_da, loss_ic, loss_eq, loss_bc
