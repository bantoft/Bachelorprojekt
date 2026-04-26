from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import astuple
from pathlib import Path

import torch
import wandb

from utils.model import PINN
from loss_funktion.bout_data import BOUTHESELData
from loss_funktion.bout_info import BOUTHESELInfo
from loss_funktion.bout_phys import BOUTHESELPhysics
from utils.training_helpers import (
    TrainConfig,
    _physics_only_training_step,
    _training_step,
    _log_step,
    _default_model_path,
    _init_wandb,
    _save_checkpoint,
)


def train(config: TrainConfig) -> tuple[PINN, BOUTHESELInfo, BOUTHESELPhysics, BOUTHESELData]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _default_model_path()

    info = BOUTHESELInfo(config.root, config.data_folder)
    physics = BOUTHESELPhysics(info)
    dataset = BOUTHESELData(info)
    loader = dataset.make_loader(batch_size=config.batch_size, shuffle=True) if config.incl_data else None
    physics_steps_per_epoch = len(loader) if loader is not None else max(1, math.ceil(len(dataset) / config.batch_size))

    parameters = torch.tensor(astuple(info.parameters), dtype=torch.float32, device=device)
    model = PINN(NN_STRUCTURE, parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    training_config = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "data_folder": config.data_folder,
        "incl_data": config.incl_data,
        "w_data": config.w_data,
        "w_eq": config.w_eq,
        "w_bc": config.w_bc,
        "w_ic": config.w_ic,
        "device": str(device),
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "model_path": str(model_path),
    }

    run = _init_wandb(config=training_config)

    global_step = 0
    best_total_loss = float("inf")
    best_epoch = 0
    best_model_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0

    x_eq, z_eq, t_eq = info.make_collocation_grid(device=device)
    x_ic = x_eq[:1].clone().detach().requires_grad_(True)
    z_ic = z_eq[:1].clone().detach().requires_grad_(True)
    t_ic = t_eq[:1].clone().detach().requires_grad_(True)

    model.train()
    print(f"Training on device: {device}")

    try:
        for epoch in range(1, config.epochs + 1):
            totals = {"total": 0.0, "data": 0.0, "eq": 0.0, "bc": 0.0, "ic": 0.0}
            steps = 0

            if config.incl_data:
                for batch in loader:
                    losses = _training_step(
                        model=model,
                        physics=physics,
                        dataset=dataset,
                        batch=batch,
                        device=device,
                        optimizer=optimizer,
                        x_eq=x_eq,
                        z_eq=z_eq,
                        t_eq=t_eq,
                        x_ic=x_ic,
                        z_ic=z_ic,
                        t_ic=t_ic,
                        w_data=config.w_data,
                        w_eq=config.w_eq,
                        w_bc=config.w_bc,
                        w_ic=config.w_ic,
                    )

                    for name, value in losses.items():
                        totals[name] += value
                    steps += 1
                    global_step += 1
                    _log_step(epoch, steps, global_step, losses)
            else:
                for _ in range(physics_steps_per_epoch):
                    losses = _physics_only_training_step(
                        model=model,
                        physics=physics,
                        device=device,
                        optimizer=optimizer,
                        x_eq=x_eq,
                        z_eq=z_eq,
                        t_eq=t_eq,
                        x_ic=x_ic,
                        z_ic=z_ic,
                        t_ic=t_ic,
                        w_eq=config.w_eq,
                        w_bc=config.w_bc,
                        w_ic=config.w_ic,
                    )
                    for name, value in losses.items():
                        totals[name] += value
                    steps += 1
                    global_step += 1
                    _log_step(epoch, steps, global_step, losses)

            averages = {name: value / steps for name, value in totals.items()}
            completed_epochs = epoch
            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"total={averages['total']:.4e} | "
                f"data={averages['data']:.4e} | "
                f"eq={averages['eq']:.4e} | "
                f"bc={averages['bc']:.4e} | "
                f"ic={averages['ic']:.4e}"
            )

            if (best_total_loss - averages["total"]) > config.early_stopping_min_delta:
                best_total_loss = averages["total"]
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_summary/total_loss": averages["total"],
                    "epoch_summary/data_loss": averages["data"],
                    "epoch_summary/eq_loss": averages["eq"],
                    "epoch_summary/bc_loss": averages["bc"],
                    "epoch_summary/ic_loss": averages["ic"],
                    "epoch_summary/best_total_loss": best_total_loss,
                    "epoch_summary/epochs_without_improvement": epochs_without_improvement,
                },
                step=global_step,
            )

            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                print(
                    f"Early stopping ved epoch {epoch}. "
                    f"Bedste total_loss var {best_total_loss:.4e} ved epoch {best_epoch}."
                )
                break
    finally:
        model.load_state_dict(best_model_state)

        saved_model_path = _save_checkpoint(
            model=model,
            model_path=model_path,
            nn_structure=NN_STRUCTURE,
            best_epoch=best_epoch,
            best_total_loss=best_total_loss,
            completed_epochs=completed_epochs,
            stopped_early=stopped_early,
            training_config=training_config,
        )
        print(f"Gemte bedste model til {saved_model_path}")

        run.summary["best_epoch"] = best_epoch
        run.summary["best_total_loss"] = best_total_loss
        run.summary["completed_epochs"] = completed_epochs
        run.summary["stopped_early"] = stopped_early
        run.summary["saved_model_path"] = str(saved_model_path)
        wandb.finish()

    return model, info, physics, dataset

NN_STRUCTURE = {
    "input_size": 3,
    "output_size": 4,
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
    ],
}

config = TrainConfig(
    epochs=10,
    batch_size=2048,
    lr=1e-4,
    root=Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL",
    data_folder="data",
    incl_data=False,
    w_data=1.0,
    w_eq=1.0,
    w_bc=1.0,
    w_ic=1.0,
    early_stopping_patience=1,
    early_stopping_min_delta=1e-2
)


if __name__ == "__main__":
    train(config)
