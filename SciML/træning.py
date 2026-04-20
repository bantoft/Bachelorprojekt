from __future__ import annotations

from pathlib import Path
import uuid

import torch

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from utils.types import TrainConfig
from utils.model import PINN
from loss_funktion.loss_from_hesel import BOUTHESELSystem
from utils.data_loader import make_data_loader, ids_to_inputs

nn_structure = {
    "input_size": 3,
    "output_size": 4,
    "layer1": {"size": 0, "activation": 0},
    "layer2": {"size": 0, "activation": 0},
    "layer3": {"size": 0, "activation": 0},
    "layer4": {"size": 0, "activation": 0},
}


def _loss_tensor(loss_value: torch.Tensor | dict[str, object]) -> torch.Tensor:
    if torch.is_tensor(loss_value):
        return loss_value
    if isinstance(loss_value, dict) and "total" in loss_value and torch.is_tensor(loss_value["total"]):
        return loss_value["total"]
    raise TypeError("Expected a tensor loss or a dict containing a tensor under 'total'.")


def _save_parquet_batch_logs(
    records: list[dict[str, float | int]],
    batch_log_path: Path,
) -> None:
    if not records:
        return
    if pd is None:
        raise ModuleNotFoundError(
            "Parquet logging requires pandas and pyarrow. Install with: uv add pandas pyarrow"
        )

    batch_log_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if batch_log_path.exists():
        existing_df = pd.read_parquet(batch_log_path)
        log_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        log_df = new_df

    tmp_path = batch_log_path.with_suffix(f".{uuid.uuid4().hex}.tmp.parquet")
    log_df.to_parquet(tmp_path, index=False, compression="zstd")
    tmp_path.replace(batch_log_path)


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN(nn_structure).to(device)
    system = BOUTHESELSystem()

    # Extract training configurations
    epochs = int(getattr(config, "epochs", 5))
    batch_size = int(getattr(config, "batch_size", 128))
    lr = float(getattr(config, "lr", 1e-3))
    print_every = int(getattr(config, "print_every", 1))
    save_data = int(getattr(config, "save_data", 0))
    batch_log_path_cfg = str(getattr(config, "batch_log_path", "SciML/data/training_batch_log.parquet"))

    # Loss weights
    w_data = float(getattr(config, "w_data", 1.0))
    w_eq = float(getattr(config, "w_eq", 1.0))
    w_bc = float(getattr(config, "w_bc", 1.0))
    w_ic = float(getattr(config, "w_ic", 1.0))

    project_root = Path(__file__).resolve().parents[1]
    batch_log_path = Path(batch_log_path_cfg)
    if not batch_log_path.is_absolute():
        batch_log_path = project_root / batch_log_path

    data_loader = make_data_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        system=system,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    batch_log_records: list[dict[str, float | int]] = []

    print(f"Training on device: {device}")
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_eq = 0.0
        epoch_bc = 0.0
        epoch_ic = 0.0
        steps = 0

        for batch_idx, batch in enumerate(data_loader, start=1):
            x_ids, z_ids, t_ids, lnn, lnpe, lnpi, vort = batch
            x, z, t = ids_to_inputs(x_ids, z_ids, t_ids, system=system, data=system.dump_data)
            x, z, t = x.to(device), z.to(device), t.to(device)

            targets = {
                "lnn": lnn.to(device).view(-1, 1),
                "lnpe": lnpe.to(device).view(-1, 1),
                "lnpi": lnpi.to(device).view(-1, 1),
                "vort": vort.to(device).view(-1, 1),
            }

            optimizer.zero_grad()
            loss_data_raw = _loss_tensor(system.data_loss(model, data=targets, x=x, z=z, t=t))
            loss_eq_raw = _loss_tensor(system.eq_loss(model))
            loss_bc_raw = _loss_tensor(system.bc_loss(model))
            loss_ic_raw = _loss_tensor(system.ic_loss(model))

            loss_data = w_data * loss_data_raw
            loss_eq = w_eq * loss_eq_raw
            loss_bc = w_bc * loss_bc_raw
            loss_ic = w_ic * loss_ic_raw

            loss_total = loss_data + loss_eq + loss_bc + loss_ic
            loss_total.backward()
            optimizer.step()

            if save_data > 0 and (batch_idx == 1 or batch_idx % save_data == 0):
                batch_log_records.append(
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss_total": float(loss_total.item()),
                        "loss_data": float(loss_data_raw.item()),
                        "loss_eq": float(loss_eq_raw.item()),
                        "loss_bc": float(loss_bc_raw.item()),
                        "loss_ic": float(loss_ic_raw.item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )

            epoch_total += float(loss_total.item())
            epoch_data += float(loss_data_raw.item())
            epoch_eq += float(loss_eq_raw.item())
            epoch_bc += float(loss_bc_raw.item())
            epoch_ic += float(loss_ic_raw.item())
            steps += 1

        if epoch == 1 or epoch % print_every == 0:
            avg_total = epoch_total / steps
            avg_data = epoch_data / steps
            avg_eq = epoch_eq / steps
            avg_bc = epoch_bc / steps
            avg_ic = epoch_ic / steps
            print(f"Epoch {epoch:5d}/{epochs} | total={avg_total:.4e} | steps={steps}")
            print(f"avg losses -> data={avg_data:.4e} | eq={avg_eq:.4e} | bc={avg_bc:.4e} | ic={avg_ic:.4e}")

    if save_data > 0:
        _save_parquet_batch_logs(batch_log_records, batch_log_path)
        print(f"Saved compressed batch training data to {batch_log_path}")

    return model


if __name__ == "__main__":
    trained_model = train(TrainConfig())
    torch.save(trained_model.state_dict(), r"SciML/data/trained_pinn.pt")
    print("Saved model weights to SciML/data/trained_pinn.pt")
