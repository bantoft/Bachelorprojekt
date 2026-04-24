from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from utils.model import PINN
from utils.training_helpers import _default_model_path, _init_wandb, _save_checkpoint
from loss_funktion.bout_data import BOUTHESELData
from loss_funktion.bout_info import BOUTHESELInfo
from loss_funktion.bout_phys import BOUTHESELPhysics


def train(
    epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 1e-3,
    root: Path = Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL",
    incl_data: bool = True,
    w_data: float = 1.0,
    w_eq: float = 1.0,
    w_bc: float = 1.0,
    w_ic: float = 1.0,
    use_wandb: bool = True,
    wandb_project: str = "Bachelor_projekt",
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 0.0,
    save_model: bool = True,
    model_path: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = _default_model_path()

    info = BOUTHESELInfo(root)
    physics = BOUTHESELPhysics(info)
    dataset = BOUTHESELData(info)

    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    parameters = torch.tensor(astuple(info.parameters), dtype=torch.float32, device=device)
    model = PINN(NN_STRUCTURE, parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "incl_data": incl_data,
        "w_data": w_data,
        "w_eq": w_eq,
        "w_bc": w_bc,
        "w_ic": w_ic,
        "device": str(device),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "model_path": str(model_path),
    }
    run = _init_wandb(
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        config=training_config,
    )
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
        for epoch in range(1, epochs + 1):
            totals = {"total": 0.0, "data": 0.0, "eq": 0.0, "bc": 0.0, "ic": 0.0}
            steps = 0

            for x_ids, z_ids, t_ids, lnn, lnpe, lnpi, phi in loader:
                loss_data = torch.tensor(0.0, device=device)

                if incl_data:
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
                    pred = model(x_data, z_data, t_data)
                    loss_terms = [F.mse_loss(pred[name], targets[name]) for name in targets]
                    loss_data = torch.stack(loss_terms).sum()

                loss_eq = physics.eq_loss(model, x_eq, z_eq, t_eq)
                loss_bc = physics.bc_loss(model, x_eq, z_eq, t_eq)
                loss_ic = physics.ic_loss(model, x_ic, z_ic, t_ic)
                loss = w_data * loss_data + w_eq * loss_eq + w_bc * loss_bc + w_ic * loss_ic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                totals["total"] += float(loss.item())
                totals["data"] += float(loss_data.item())
                totals["eq"] += float(loss_eq.item())
                totals["bc"] += float(loss_bc.item())
                totals["ic"] += float(loss_ic.item())
                steps += 1
                global_step += 1

                if run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "batch": steps,
                            "train_step/total_loss": float(loss.item()),
                            "train_step/data_loss": float(loss_data.item()),
                            "train_step/eq_loss": float(loss_eq.item()),
                            "train_step/bc_loss": float(loss_bc.item()),
                            "train_step/ic_loss": float(loss_ic.item()),
                        },
                        step=global_step,
                    )

            if steps == 0:
                raise ValueError(
                    "Ingen batches blev produceret. Sænk batch_size eller kontroller datasættet."
                )

            averages = {name: value / steps for name, value in totals.items()}
            completed_epochs = epoch
            print(
                f"Epoch {epoch}/{epochs} | "
                f"total={averages['total']:.4e} | "
                f"data={averages['data']:.4e} | "
                f"eq={averages['eq']:.4e} | "
                f"bc={averages['bc']:.4e} | "
                f"ic={averages['ic']:.4e}"
            )

            if (best_total_loss - averages["total"]) > early_stopping_min_delta:
                best_total_loss = averages["total"]
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if run is not None:
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

            if (
                early_stopping_patience is not None
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_early = True
                print(
                    f"Early stopping ved epoch {epoch}. "
                    f"Bedste total_loss var {best_total_loss:.4e} ved epoch {best_epoch}."
                )
                break
    finally:
        model.load_state_dict(best_model_state)

        saved_model_path = None
        if save_model:
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

        if run is not None:
            run.summary["best_epoch"] = best_epoch
            run.summary["best_total_loss"] = best_total_loss
            run.summary["completed_epochs"] = completed_epochs
            run.summary["stopped_early"] = stopped_early
            if saved_model_path is not None:
                run.summary["saved_model_path"] = str(saved_model_path)
            wandb.finish()

    return model, info, physics, dataset

# non-lin-foo at teste: nn.SiLU, nn.Tanh
# Layer_sizes at teste: 16, 32, 64

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


if __name__ == "__main__":
    train(
        epochs=10,
        batch_size=1024,
        lr=1e-4,
        root=Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL",
        incl_data=True,
        w_data=1.0,
        w_eq=1.0,
        w_bc=1.0,
        w_ic=1.0,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-2,
        save_model=True,
    )
