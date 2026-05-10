from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

from util.model import PINN
from util.structure import NN_STRUCTURE
from util.data_loader import make_dataloader
from util.training_helpers import (_default_model_path,
                                   _init_wandb,
                                   _save_checkpoint,
                                   _log_split_losses,
                                   _get_loss)

def init_trainer(
        batch_size = 256,
        num_workers = 2,
        pin_memory = True,
        train_ratio = 0.8,
        val_ratio = 0.1,
        lr = 1e-3,):
    from loss_PDE.bout_phys import BOUTHESELPhys
    from loss_PDE.bout_dump import BOUTHESELInfo
    root = Path(__file__).parents[1] / r"simulatorer/BOUT/BOUT-HESEL/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info).to(device)
    model = PINN(NN_STRUCTURE)
    standardization = type(info.standardized)(
        mean=info.standardized.mean.to(device),
        std=info.standardized.std.to(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    train_loader, val_loader, test_loader = make_dataloader(info,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            pin_memory=pin_memory,
                                                            train_ratio=train_ratio,
                                                            val_ratio=val_ratio)

    return (info,
            phys,
            model,
            train_loader,
            val_loader,
            test_loader,
            standardization,
            optimizer,
            device)



if __name__ == "__main__":

    training_config = {
        "saving_frequency": 10,
        "status_frequency": 500,
        "epochs": 2,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 10,
        "batch_size": 2048,
        "num_workers": 2,
        "pin_memory": True,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "lr": 1e-4,
        "weight_da": 10.0,
        "weight_ic": 1.0,
        "weight_eq": 500.0,
        "weight_bc": 1.0,
        "device": None
    }


    (info,
    phys,
    model,
    train_loader,
    val_loader,
    test_loader,
    standardization,
    optimizer,
    device) = init_trainer(batch_size=training_config["batch_size"],
                           num_workers = training_config["num_workers"],
                           pin_memory = training_config["pin_memory"],
                           train_ratio = training_config["train_ratio"],
                           val_ratio = training_config["val_ratio"],
                           lr = training_config["lr"],)

    mse_loss = nn.MSELoss().to(device)

    training_config["device"] = device
    
    run = _init_wandb(config=training_config)
    model_path = _default_model_path()
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_state = deepcopy(model.state_dict())
    stopped_early = False

    try:
        for epoch in range(training_config["epochs"]+1):

            model.train()
            for batch_idx, batch in enumerate(train_loader):

                loss_da, loss_ic, loss_eq, loss_bc = _get_loss(model, phys, batch, standardization, mse_loss)
                loss_da *= training_config["weight_da"]
                loss_ic *= training_config["weight_ic"]
                loss_eq *= training_config["weight_eq"]
                loss_bc *= training_config["weight_bc"]

                optimizer.zero_grad()
                loss_total = loss_da + loss_ic + loss_eq + loss_bc
                loss_total.backward()
                optimizer.step()

                if batch_idx % training_config["saving_frequency"] == 0:
                    global_step += 1
                    _log_split_losses(
                        run,
                        split="train",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        step=global_step,
                        loss_total=loss_total.item(),
                        loss_da=loss_da.item(),
                        loss_ic=loss_ic.item(),
                        loss_eq=loss_eq.item(),
                        loss_bc=loss_bc.item(),
                    )

                if batch_idx % training_config["status_frequency"] == 0:
                    print(
                        f"Epoch [{epoch+1}/{training_config['epochs']}], "
                        f"Batch [{batch_idx}/{len(train_loader)}], "
                        f"Loss: {loss_total.item():.6e} (DA: {loss_da.item():.6e}, IC: {loss_ic.item():.6e}, EQ: {loss_eq.item():.6e}, BC: {loss_bc.item():.6e})"
                    )

            model.eval()
            val_total_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):

                    loss_da, loss_ic, loss_eq, loss_bc = _get_loss(model, phys, batch, standardization, mse_loss)
                    loss_da *= training_config["weight_da"]
                    loss_ic *= training_config["weight_ic"]
                    loss_eq *= training_config["weight_eq"]
                    loss_bc *= training_config["weight_bc"]
                    val_total = loss_da.item() + loss_ic.item() + loss_eq.item() + loss_bc.item()
                    val_total_loss_sum += val_total
                    val_batches += 1

                    global_step += 1
                    _log_split_losses(
                        run,
                        split="val",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        step=global_step,
                        loss_total=val_total,
                        loss_da=loss_da.item(),
                        loss_ic=loss_ic.item(),
                        loss_eq=loss_eq.item(),
                        loss_bc=loss_bc.item(),
                    )
                    
                    if batch_idx % training_config["status_frequency"] == 0:
                        print(
                            f"Epoch [{epoch+1}/{training_config['epochs']}], "
                            f"Batch [{batch_idx}/{len(train_loader)}], "
                            f"Loss: {val_total:.6e} (DA: {loss_da:.6e}, IC: {loss_ic:.6e}, EQ: {loss_eq:.6e}, BC: {loss_bc:.6e})"
                        )

                mean_val_loss = val_total_loss_sum / val_batches if val_batches > 0 else float("inf")


                for batch_idx, batch in enumerate(test_loader):
                    loss_da, loss_ic, loss_eq, loss_bc = _get_loss(model, phys, batch, standardization, mse_loss)
                    loss_da *= training_config["weight_da"]
                    loss_ic *= training_config["weight_ic"]
                    loss_eq *= training_config["weight_eq"]
                    loss_bc *= training_config["weight_bc"]
                    test_total = loss_da.item() + loss_ic.item() + loss_eq.item() + loss_bc.item()

                    global_step += 1
                    _log_split_losses(
                        run,
                        split="test",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        step=global_step,
                        loss_total=(loss_da.item() + loss_ic.item() + loss_eq.item() + loss_bc.item()),
                        loss_da=loss_da.item(),
                        loss_ic=loss_ic.item(),
                        loss_eq=loss_eq.item(),
                        loss_bc=loss_bc.item(),
                    )

                    if batch_idx % training_config["status_frequency"] == 0:
                        print(
                            f"Epoch [{epoch+1}/{training_config['epochs']}], "
                            f"Batch [{batch_idx}/{len(train_loader)}], "
                            f"Loss: {test_total:.6e} (DA: {loss_da:.6e}, IC: {loss_ic:.6e}, EQ: {loss_eq:.6e}, BC: {loss_bc:.6e})"
                        )



            if mean_val_loss < (best_val_loss - training_config["early_stopping_min_delta"]):
                best_val_loss = mean_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            run.log(
                {
                    "epoch": epoch,
                    "val/epoch_mean_total_loss": mean_val_loss,
                    "early_stopping/best_val_loss": best_val_loss,
                    "early_stopping/best_epoch": best_epoch,
                    "early_stopping/epochs_without_improvement": epochs_without_improvement,
                },
                step=global_step,
            )

            if epochs_without_improvement >= training_config["early_stopping_patience"]:
                stopped_early = True
                print(
                    f"Early stopping ved epoch {epoch}. "
                    f"Bedste val loss var {best_val_loss:.6e} ved epoch {best_epoch}."
                )
                break


    finally:
        model.load_state_dict(best_model_state)
        saved_model_path = _save_checkpoint(
            model=model,
            model_path=model_path,
            output_names=tuple(NN_STRUCTURE["output_names"]),
            standardization=standardization,
        )
        run.summary["saved_model_path"] = str(saved_model_path)
        run.summary["best_val_loss"] = best_val_loss
        run.summary["best_epoch"] = best_epoch
        run.summary["stopped_early"] = stopped_early
        run.finish()

    print(f"Gemte model lokalt til {saved_model_path}")
    print("Done")
