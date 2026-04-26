from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, random_split

from utils.model import PINN
from loss_funktion.bout_data import BOUTHESELData
from loss_funktion.bout_info import BOUTHESELInfo
from loss_funktion.bout_phys import BOUTHESELPhysics
from utils.training_helpers import (
    TrainConfig,
    _physics_only_training_step,
    _training_step,
    _validation_data_loss,
    _log_step,
    _default_model_path,
    _init_wandb,
    _save_checkpoint,
)


def train(config: TrainConfig) -> tuple[PINN, BOUTHESELInfo, BOUTHESELPhysics, BOUTHESELData]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _default_model_path()
    collocation_shape = {
        "num_t": config.collocation_num_t,
        "num_x": config.collocation_num_x,
        "num_z": config.collocation_num_z,
    }
    ic_shape = {
        "num_x": config.ic_num_x,
        "num_z": config.ic_num_z,
    }

    info = BOUTHESELInfo(config.root)
    physics = BOUTHESELPhysics(info)
    dataset = BOUTHESELData(info)
    train_loader = None
    val_loader = None
    train_dataset = dataset
    physics_dataset_len = len(dataset)

    if config.incl_data:
        if not 0.0 <= config.val_split < 1.0:
            raise ValueError(f"val_split skal være i intervallet [0, 1), fik {config.val_split}")

        if len(dataset) > 1 and config.val_split > 0.0:
            val_size = max(1, int(len(dataset) * config.val_split))
            train_size = len(dataset) - val_size
            if train_size <= 0:
                train_size = 1
                val_size = len(dataset) - train_size

            split_seed = 42
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(split_seed),
            )

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, drop_last=False)
            physics_dataset_len = len(train_dataset)
            print(f"Validation aktiv: train={len(train_dataset)} samples, val={len(val_dataset)} samples (val_split={config.val_split:.2f})")
        else:
            train_loader = dataset.make_loader(batch_size=config.batch_size, shuffle=True)

    physics_steps_per_epoch = len(train_loader) if train_loader is not None else max(1, math.ceil(physics_dataset_len / config.batch_size))

    model = PINN(NN_STRUCTURE).to(device)
    model.set_output_scaling(info.standardization_stats)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    training_config = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "incl_data": config.incl_data,
        "w_data": config.w_data,
        "w_eq": config.w_eq,
        "w_bc": config.w_bc,
        "w_ic": config.w_ic,
        "val_split": config.val_split,
        "device": str(device),
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "model_path": str(model_path),
        "standardized_fields": list(info.standardization_stats.keys()),
        "collocation_shape": collocation_shape,
        "ic_shape": ic_shape,
    }

    run = _init_wandb(config=training_config)

    global_step = 0
    best_total_loss = float("inf")
    best_epoch = 0
    best_model_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0

    model.train()
    print(f"Training on device: {device}")

    try:
        for epoch in range(1, config.epochs + 1):
            totals = {"total": 0.0, "data": 0.0, "eq": 0.0, "bc": 0.0, "ic": 0.0}
            steps = 0

            if config.incl_data:
                if train_loader is None:
                    raise RuntimeError("train_loader blev ikke initialiseret selvom incl_data=True")
                for batch in train_loader:
                    x_eq, z_eq, t_eq = dataset.make_collocation_grid(device=device, **collocation_shape)
                    x_ic, z_ic, t_ic = dataset.make_initial_condition_grid(device=device, **ic_shape)
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
                    x_eq, z_eq, t_eq = dataset.make_collocation_grid(device=device, **collocation_shape)
                    x_ic, z_ic, t_ic = dataset.make_initial_condition_grid(device=device, **ic_shape)
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
            val_data_loss = None
            if val_loader is not None:
                val_total = 0.0
                val_steps = 0
                for batch in val_loader:
                    val_total += _validation_data_loss(
                        model=model,
                        dataset=dataset,
                        batch=batch,
                        device=device,
                    )
                    val_steps += 1
                val_data_loss = val_total / max(val_steps, 1)

            completed_epochs = epoch
            epoch_msg = (
                f"Epoch {epoch}/{config.epochs} | "
                f"total={averages['total']:.4e} | "
                f"data={averages['data']:.4e} | "
                f"eq={averages['eq']:.4e} | "
                f"bc={averages['bc']:.4e} | "
                f"ic={averages['ic']:.4e}"
            )
            if val_data_loss is not None:
                epoch_msg += f" | val_data={val_data_loss:.4e}"
            print(epoch_msg)

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
                    **({"epoch_summary/val_data_loss": val_data_loss} if val_data_loss is not None else {}),
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
        {"size": 128, "non_lin_foo": torch.nn.Tanh},
        {"size": 128, "non_lin_foo": torch.nn.SiLU},
        {"size": 128, "non_lin_foo": torch.nn.Tanh},
    ],
}

config = TrainConfig(
    epochs=13,
    batch_size=2048,
    lr=1e-3,
    root=Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL" / "data",
    incl_data=True,
    w_data=1.0,
    w_eq=1.0,
    w_bc=1.0,
    w_ic=1.0,
    val_split=0.1,
    early_stopping_patience=3,
    early_stopping_min_delta=1e-2,
    collocation_num_t=16,
    collocation_num_x=16,
    collocation_num_z=16,
    ic_num_x=32,
    ic_num_z=64,
)


if __name__ == "__main__":
    train(config)
