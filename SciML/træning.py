from pathlib import Path
from copy import deepcopy

import torch
from torch import nn

from loss_PDE.bout_dump import BOUTHESELInfo
from loss_PDE.bout_phys import BOUTHESELPhys
from util.data_loader import BOUTDataset
from util.model import PINN
from util.structure import NN_STRUCTURE
from util.training_helpers import (
    TrainConfig,
    _default_history_path,
    _default_model_path,
    _init_wandb,
    _log_epoch_summary,
    _log_step,
    _save_checkpoint,
    _save_training_history,
)


def move_sample_to_device(sample: dict, device: torch.device) -> dict:
    moved_sample = {}
    for key, value in sample.items():
        if isinstance(value, dict):
            moved_sample[key] = {
                name: tensor.to(device)
                for name, tensor in value.items()
            }
            continue

        tensor = value.to(device)
        if key in {"x", "z", "t"}:
            tensor = tensor.detach().requires_grad_(True)
        moved_sample[key] = tensor
    return moved_sample


def print_tensor_shapes(label: str, tensors: dict[str, torch.Tensor]) -> None:
    print(label)
    for name, values in tensors.items():
        print(f"  {name}: {tuple(values.shape)}")


def mean_zero_loss(
    criterion: nn.Module,
    residuals: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.stack([
        criterion(residual, torch.zeros_like(residual))
        for residual in residuals.values()
    ]).mean()


def compute_step_losses(
    model: PINN,
    physics: BOUTHESELPhys,
    standardization,
    criterion: nn.Module,
    previous_sample: dict,
    current_sample: dict,
    next_sample: dict,
    eq_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    initial_state = previous_sample["fields"]
    initial_prediction = model(
        standardization,
        previous_sample["x"],
        previous_sample["z"],
        torch.zeros_like(previous_sample["t"]),
        initial_state,
        initial_state,
    )
    prediction = model(
        standardization,
        next_sample["x"],
        next_sample["z"],
        next_sample["t"],
        current_sample["fields"],
        previous_sample["fields"],
    )

    data_loss = torch.stack([
        criterion(prediction[name], next_sample["fields"][name])
        for name in model.output_names
    ]).mean()
    ic_residuals = physics.ic_res(
        initial_prediction,
        previous_sample["x"],
        previous_sample["z"],
    )
    bc_residuals = physics.bd_res(
        prediction,
        next_sample["x"],
        next_sample["z"],
    )
    eq_residuals = physics.eq_res(
        prediction,
        next_sample["x"],
        next_sample["z"],
        next_sample["t"],
    )

    ic_loss = mean_zero_loss(criterion, ic_residuals)
    bc_loss = mean_zero_loss(criterion, bc_residuals)
    raw_eq_loss = mean_zero_loss(criterion, eq_residuals)
    eq_loss = raw_eq_loss * eq_weight
    total_loss = data_loss + ic_loss + bc_loss + eq_loss

    losses = {
        "total": total_loss,
        "data": data_loss,
        "ic": ic_loss,
        "bc": bc_loss,
        "eq": eq_loss,
        "eq_raw": raw_eq_loss,
    }
    artifacts = {
        "initial_prediction": initial_prediction,
        "prediction": prediction,
        "ic_residuals": ic_residuals,
        "bc_residuals": bc_residuals,
        "eq_residuals": eq_residuals,
    }
    return losses, artifacts


def _make_time_split_ranges(num_time_steps: int) -> tuple[range, range, range, int, int]:
    train_end = max(int(num_time_steps * 0.8), 3)
    val_end = max(int(num_time_steps * 0.9), train_end + 3)
    val_end = min(val_end, num_time_steps)

    train_steps = range(1, max(1, train_end - 1))
    val_steps = range(train_end + 1, max(train_end + 1, val_end - 1))
    test_steps = range(val_end + 1, max(val_end + 1, num_time_steps - 1))
    return train_steps, val_steps, test_steps, train_end, val_end


def _evaluate_split(
    *,
    step_indices: range,
    model: PINN,
    physics: BOUTHESELPhys,
    dataset: BOUTDataset,
    standardization,
    criterion: nn.Module,
    device: torch.device,
    eq_weight: float,
) -> dict[str, float] | None:
    if len(step_indices) == 0:
        return None

    model_was_training = model.training
    model.eval()
    totals = {
        "total": 0.0,
        "data": 0.0,
        "ic": 0.0,
        "bc": 0.0,
        "eq": 0.0,
        "eq_raw": 0.0,
    }

    try:
        for step_index in step_indices:
            previous_sample = move_sample_to_device(dataset[step_index - 1], device)
            current_sample = move_sample_to_device(dataset[step_index], device)
            next_sample = move_sample_to_device(dataset[step_index + 1], device)
            losses, _ = compute_step_losses(
                model=model,
                physics=physics,
                standardization=standardization,
                criterion=criterion,
                previous_sample=previous_sample,
                current_sample=current_sample,
                next_sample=next_sample,
                eq_weight=eq_weight,
            )
            for name in totals:
                totals[name] += float(losses[name].item())
    finally:
        if model_was_training:
            model.train()

    num_steps = len(step_indices)
    return {
        name: value / num_steps
        for name, value in totals.items()
    }


if __name__ == "__main__":
    config = TrainConfig(
        epochs=5,
        lr=1e-3,
        root=Path(__file__).parents[1] / "simulatorer/BOUT/BOUT-HESEL/data",
        eq_weight=500.0,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _default_model_path()
    history_path = _default_history_path()

    info = BOUTHESELInfo(config.root)
    physics = BOUTHESELPhys(info)
    dataset = BOUTDataset(info)
    train_step_indices, val_step_indices, test_step_indices, train_end, val_end = _make_time_split_ranges(len(dataset))

    model = PINN(NN_STRUCTURE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    standardization = info.standardized

    last_artifacts = None
    global_step = 0
    best_total_loss = float("inf")
    best_epoch = 0
    best_model_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    history = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "ic_loss": [],
        "bc_loss": [],
        "eq_loss": [],
        "eq_raw_loss": [],
        "val_total_loss": [],
        "val_data_loss": [],
        "val_ic_loss": [],
        "val_bc_loss": [],
        "val_eq_loss": [],
        "val_eq_raw_loss": [],
        "test_total_loss": [],
        "test_data_loss": [],
        "test_ic_loss": [],
        "test_bc_loss": [],
        "test_eq_loss": [],
        "test_eq_raw_loss": [],
        "split_info": [{
            "num_time_steps": len(dataset),
            "train_time_end_exclusive": train_end,
            "val_time_end_exclusive": val_end,
            "num_train_steps": len(train_step_indices),
            "num_val_steps": len(val_step_indices),
            "num_test_steps": len(test_step_indices),
        }],
    }
    training_config = {
        "epochs": config.epochs,
        "lr": config.lr,
        "device": str(device),
        "eq_weight": config.eq_weight,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "standardized_fields": list(NN_STRUCTURE["output_names"]),
        "input_size": NN_STRUCTURE["input_size"],
        "history_steps": 2,
        "neighborhood_size": 3,
        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "train_time_end_exclusive": train_end,
            "val_time_end_exclusive": val_end,
            "num_train_steps": len(train_step_indices),
            "num_val_steps": len(val_step_indices),
            "num_test_steps": len(test_step_indices),
        },
    }
    run = _init_wandb(config=training_config)

    model.train()
    print(f"Training on device: {device}")
    print(
        "Time split | "
        f"train: [0, {train_end}) | "
        f"val: [{train_end}, {val_end}) | "
        f"test: [{val_end}, {len(dataset)})"
    )
    try:
        for epoch in range(1, config.epochs + 1):
            epoch_totals = {
                "total": 0.0,
                "data": 0.0,
                "ic": 0.0,
                "bc": 0.0,
                "eq": 0.0,
                "eq_raw": 0.0,
            }

            for batch_index, step_index in enumerate(train_step_indices, start=1):
                previous_sample = move_sample_to_device(dataset[step_index - 1], device)
                current_sample = move_sample_to_device(dataset[step_index], device)
                next_sample = move_sample_to_device(dataset[step_index + 1], device)

                optimizer.zero_grad()
                losses, last_artifacts = compute_step_losses(
                    model=model,
                    physics=physics,
                    standardization=standardization,
                    criterion=criterion,
                    previous_sample=previous_sample,
                    current_sample=current_sample,
                    next_sample=next_sample,
                    eq_weight=config.eq_weight,
                )
                losses["total"].backward()
                optimizer.step()

                log_losses = {name: float(value.item()) for name, value in losses.items()}
                for name in epoch_totals:
                    epoch_totals[name] += log_losses[name]

                global_step += 1
                _log_step(epoch, batch_index, global_step, log_losses)

            num_steps = len(train_step_indices)
            averages = {
                name: value / num_steps
                for name, value in epoch_totals.items()
            }
            val_averages = _evaluate_split(
                step_indices=val_step_indices,
                model=model,
                physics=physics,
                dataset=dataset,
                standardization=standardization,
                criterion=criterion,
                device=device,
                eq_weight=config.eq_weight,
            )
            test_averages = _evaluate_split(
                step_indices=test_step_indices,
                model=model,
                physics=physics,
                dataset=dataset,
                standardization=standardization,
                criterion=criterion,
                device=device,
                eq_weight=config.eq_weight,
            )
            completed_epochs = epoch

            print(f"Epoch {epoch}/{config.epochs}")
            print(f"  device: {device}")
            print(f"  train steps: {num_steps}")
            print(f"  train total loss: {averages['total']:.6e}")
            print(f"  train data loss: {averages['data']:.6e}")
            print(f"  train ic loss: {averages['ic']:.6e}")
            print(f"  train bc loss: {averages['bc']:.6e}")
            print(f"  train eq loss: {averages['eq']:.6e}")
            print(f"  train eq raw loss: {averages['eq_raw']:.6e}")
            if val_averages is not None:
                print(f"  val total loss: {val_averages['total']:.6e}")
            if test_averages is not None:
                print(f"  test total loss: {test_averages['total']:.6e}")

            history["epoch"].append(float(epoch))
            history["total_loss"].append(averages["total"])
            history["data_loss"].append(averages["data"])
            history["ic_loss"].append(averages["ic"])
            history["bc_loss"].append(averages["bc"])
            history["eq_loss"].append(averages["eq"])
            history["eq_raw_loss"].append(averages["eq_raw"])
            history["val_total_loss"].append(float("nan") if val_averages is None else val_averages["total"])
            history["val_data_loss"].append(float("nan") if val_averages is None else val_averages["data"])
            history["val_ic_loss"].append(float("nan") if val_averages is None else val_averages["ic"])
            history["val_bc_loss"].append(float("nan") if val_averages is None else val_averages["bc"])
            history["val_eq_loss"].append(float("nan") if val_averages is None else val_averages["eq"])
            history["val_eq_raw_loss"].append(float("nan") if val_averages is None else val_averages["eq_raw"])
            history["test_total_loss"].append(float("nan") if test_averages is None else test_averages["total"])
            history["test_data_loss"].append(float("nan") if test_averages is None else test_averages["data"])
            history["test_ic_loss"].append(float("nan") if test_averages is None else test_averages["ic"])
            history["test_bc_loss"].append(float("nan") if test_averages is None else test_averages["bc"])
            history["test_eq_loss"].append(float("nan") if test_averages is None else test_averages["eq"])
            history["test_eq_raw_loss"].append(float("nan") if test_averages is None else test_averages["eq_raw"])

            monitor_loss = averages["total"] if val_averages is None else val_averages["total"]
            if (best_total_loss - monitor_loss) > config.early_stopping_min_delta:
                best_total_loss = monitor_loss
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            _log_epoch_summary(
                epoch=epoch,
                global_step=global_step,
                averages=averages,
                best_total_loss=best_total_loss,
                epochs_without_improvement=epochs_without_improvement,
                val_averages=val_averages,
                test_averages=test_averages,
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
            output_names=tuple(NN_STRUCTURE["output_names"]),
            standardization=standardization,
        )
        saved_history_path = _save_training_history(history, history_path)
        print(f"Gemte bedste model til {saved_model_path}")
        print(f"Gemte training history til {saved_history_path}")

        run.summary["best_epoch"] = best_epoch
        run.summary["best_total_loss"] = best_total_loss
        run.summary["completed_epochs"] = completed_epochs
        run.summary["stopped_early"] = stopped_early
        run.summary["saved_model_path"] = str(saved_model_path)
        run.summary["saved_history_path"] = str(saved_history_path)
        run.finish()

    if last_artifacts is not None:
        print_tensor_shapes("\nInitial prediction shapes", last_artifacts["initial_prediction"])
        print_tensor_shapes("\nPrediction shapes", last_artifacts["prediction"])
        print_tensor_shapes("\nIC residual shapes", last_artifacts["ic_residuals"])
        print_tensor_shapes("\nBC residual shapes", last_artifacts["bc_residuals"])
        print_tensor_shapes("\nEQ residual shapes", last_artifacts["eq_residuals"])
