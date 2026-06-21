from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader



SPLITS = ("training", "validation", "test")
LOSS_TYPES = ("raw", "weighted")
LOSS_KEYS = ("total", "da", "eq")
CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_BACKUP_NAME = "checkpoint.prev.pt"



class HistoryBuffer:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.buffer = self._empty_history()
        if not self.save_path.exists():
            self._save(self._empty_history())

    def append(self, split: str, loss_type: str, values: dict[str, float]):
        target = self.buffer[split.lower()][loss_type]
        for key, value in values.items():
            target[key].append(float(value))

    def _empty_history(self):
        return {
            split: {
                loss_type: {key: [] for key in LOSS_KEYS}
                for loss_type in LOSS_TYPES
            }
            for split in SPLITS
        }

    def flush(self):
        history = self._load()
        for split in SPLITS:
            for loss_type in LOSS_TYPES:
                for key in LOSS_KEYS:
                    history[split][loss_type][key].extend(self.buffer[split][loss_type][key])
        self._save(history)
        self.buffer = self._empty_history()

    def _save(self, history):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(history, f, indent=2)

    def _load(self):
        with open(self.save_path, "r") as f:
            return json.load(f)


def save_checkpoint(training_config: dict,
                    best_val_model: torch.nn.Module | None = None,
                    model_state: torch.nn.Module | None = None,
                    condition: torch.nn.Module| None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    epoch: int| None = None,
                    split: str| None = None,
                    batch_idx: int | None = None,
                    best_val_loss: float| None = None,
                    patience_counter: int| None = None,
                    skipped_batch_info: dict| None = None
                    ):
    filename = Path(training_config["data_dir"]).resolve() / CHECKPOINT_NAME
    backup_filename = filename.with_name(CHECKPOINT_BACKUP_NAME)
    tmp_filename = filename.with_name(f"{filename.name}.tmp")

    checkpoint, checkpoint_source, _ = _load_checkpoint_candidates(filename, backup_filename)
    if checkpoint is None:
        checkpoint = {}

    checkpoint["training_config"] = training_config
    checkpoint["torch_rng_state"] = torch.get_rng_state().cpu().numpy().tobytes()

    save_dict = {"model_state_dict": None if model_state is None else model_state.state_dict(),
                 "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                 "condition_state_dict": None if condition is None else condition.state_dict(),
                 "best_val_model": None if best_val_model is None else best_val_model.state_dict(),
                 "seed": training_config["seed"],
                 "epoch": epoch,
                 "split": split,
                 "batch_idx": batch_idx,
                 "best_val_loss": best_val_loss,
                 "patience_counter": patience_counter,
                 "skipped_batch_info": skipped_batch_info
                 }
    
    for key, value in save_dict.items():
        if value is not None: checkpoint[key] = value

    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, tmp_filename)

    if checkpoint_source == filename and filename.exists():
        os.replace(filename, backup_filename)
    os.replace(tmp_filename, filename)


def load_checkpoint(training_config):
    filename = Path(training_config["data_dir"]).resolve() / CHECKPOINT_NAME
    backup_filename = filename.with_name(CHECKPOINT_BACKUP_NAME)
    checkpoint, _, load_errors = _load_checkpoint_candidates(filename, backup_filename)
    if checkpoint is not None:
        return checkpoint

    if not load_errors:
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    error_text = "\n".join(load_errors)
    raise RuntimeError(
        "Could not load a valid checkpoint. "
        "The primary checkpoint appears corrupted, and no usable backup was found.\n"
        f"{error_text}"
    )


def _load_checkpoint_candidates(*candidates: Path) -> tuple[dict | None, Path | None, list[str]]:
    load_errors = []

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return (
                torch.load(candidate, map_location="cpu", weights_only=False),
                candidate,
                load_errors,
            )
        except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            load_errors.append(f"{candidate}: {exc}")

    return None, None, load_errors


def save_invalid_batch(training_config: dict,
                       state: dict,
                       batch_idx: int,
                       batch: list[torch.Tensor],
                       f_theta: dict[str, torch.Tensor],
                       coord_fys: torch.Tensor,
                       eq_res: torch.Tensor,
                       data_loss_value: float,
                       eq_loss_value: float):
    output_dir = Path(training_config["data_dir"]).resolve() / "skipped_batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{state['split']}_epoch{state['epoch']}_batch{batch_idx:06d}.pt"

    torch.save(
        {
            "epoch": state["epoch"],
            "split": state["split"],
            "batch_idx": batch_idx,
            "data_loss": data_loss_value,
            "eq_loss": eq_loss_value,
            "batch": [tensor.detach().cpu() for tensor in batch],
            "f_theta": {name: tensor.detach().cpu() for name, tensor in f_theta.items()},
            "coord_fys": coord_fys.detach().cpu(),
            "eq_res": eq_res.detach().cpu(),
        },
        output_path,
    )
    return output_path


data_loader_setup1 = {
    "TSSplit": True,
}

data_loader_setup2 = {
    "TSSplit": False,
}

def init_experinment(train_cfg):
    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys
    from SciML.PINN_z.utils.data_loader import make_dataloaders
    from SciML.PINN_z.utils.wrapper import WrappedPINN

    # Laver dobbelt data loader setup
    if len(train_cfg["root"]) == 2:
        info = BOUTHESELInfo(train_cfg["root"][0])
        info2 = BOUTHESELInfo(train_cfg["root"][1])

        phys = BOUTHESELPhys(info)  # Antager at fysikken er den samme for begge datasæt, hvilket er tilfældet for de nuværende data
        model = WrappedPINN(
            info=info,
            phys=phys,
            m=train_cfg["z_width"],
            network_structure=train_cfg["network"],
        ).to(info.device)


        data_loader_dict = {"z_width": train_cfg["z_width"],
                            "batch_size": train_cfg["batch_size"],
                            "val_split": train_cfg["val_split"],
                            "train_split": train_cfg["train_split"],
                            "shuffle": train_cfg["shuffle"],
                            "num_workers": train_cfg["num_workers"],
                            "prefetch_factor": train_cfg["prefetch_factor"],
                            "pin_memory": train_cfg["pin_memory"],
                            "persistent_workers": train_cfg["persistent_workers"]
                            }
        
        for key, value in data_loader_dict.items():
            data_loader_setup1[key] = value
            data_loader_setup2[key] = value

        train_loader1, val_loader1, test_loader1 = make_dataloaders(info, data_loader_setup1)
        train_loader2, val_loader2, test_loader2 = make_dataloaders(info2, data_loader_setup2)

        train_dataset = ConcatDataset([train_loader1.dataset, train_loader2.dataset])
        test_dataset = ConcatDataset([test_loader1.dataset, test_loader2.dataset])
        val_dataset = ConcatDataset([val_loader1.dataset, val_loader2.dataset])


        train_loader = DataLoader(train_dataset,
                                  batch_size=train_cfg["batch_size"],
                                  shuffle=train_cfg["shuffle"],
                                  num_workers=train_cfg["num_workers"],
                                  pin_memory=train_cfg["pin_memory"],
                                  prefetch_factor=train_cfg["prefetch_factor"],
                                  persistent_workers=train_cfg["persistent_workers"]
                                  )
        val_loader = DataLoader(val_dataset,
                                batch_size=train_cfg["batch_size"],
                                shuffle=False,
                                num_workers=train_cfg["num_workers"],
                                pin_memory=train_cfg["pin_memory"],
                                prefetch_factor=train_cfg["prefetch_factor"],
                                persistent_workers=train_cfg["persistent_workers"]
                                )
        test_loader = DataLoader(test_dataset,
                                batch_size=train_cfg["batch_size"],
                                shuffle=False,
                                num_workers=train_cfg["num_workers"],
                                pin_memory=train_cfg["pin_memory"],
                                prefetch_factor=train_cfg["prefetch_factor"],
                                persistent_workers=train_cfg["persistent_workers"]
                                )

    else:
        info = BOUTHESELInfo(train_cfg["root"][0])
        phys = BOUTHESELPhys(info)

        model = WrappedPINN(
            info=info,
            phys=phys,
            m=train_cfg["z_width"],
            network_structure=train_cfg["network"],
        ).to(info.device)
        

        train_loader, val_loader, test_loader = make_dataloaders(info=info, data_loader_config=train_cfg)


    data_dir = Path(train_cfg["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(),
                             lr=train_cfg["lr"],
                             betas= (0.7, 0.95),
                             eps=1e-8,
                             weight_decay=1e-5)
    condition = torch.nn.MSELoss()
    history = HistoryBuffer(data_dir / "history.json")

    return (model,
            train_loader,
            val_loader,
            test_loader,
            condition,
            optimizer,
            history,
            phys,
            info
            )

def iterator(loader,
             start_batch_idx,
             train_config,
             total_loader_batches,
             state,
             history,
             model,
             optimizer,
             condition,
             processed_batches,
             last_batch_idx,
             info,
             phys
             ):

    is_training = state["split"] == "training"
    param_requires_grad = None
    if not is_training:
        optimizer.zero_grad(set_to_none=True)
        param_requires_grad = [param.requires_grad for param in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(False)

    total_loss = 0.0
    num_loss_batches = 0
    status_data = 0.0
    status_eq = 0.0
    status_loss_batches = 0
    try:
        for batch_idx, batch in enumerate(loader, start=start_batch_idx):

            # if batch_idx == 50: break

            if state["batch_idx"] % train_config["flush_frequency"] == 0:
                history.flush()
                save_checkpoint(train_config,
                                model_state=model,
                                optimizer=optimizer,
                                condition=condition,
                                split=state["split"],
                                epoch=state["epoch"],
                                batch_idx=state["batch_idx"],
                                best_val_loss=state["best_val_loss"],
                                patience_counter=state["patience_counter"]
                                )

            batch = [x.to(info.device, non_blocking=True) for x in batch]
            last_batch_idx = batch_idx

            optimizer.zero_grad(set_to_none=True)
            f_theta, coord_fys = model(batch, requires_coord_grad=True)
            _, _, avg_z, _, coords_num = batch

            pred_da = torch.cat([f_theta[name] for name in ("lnn", "lnpe", "lnpi", "phi")], dim=1)
            center_idx = pred_da.shape[-1] // 2
            pred_da_center = pred_da[:, :, :, center_idx : center_idx + 1]
            data_loss = condition(pred_da_center, batch[1])
            data_loss_value = data_loss.detach().cpu().item()

            eq_res = phys.eq_res(avg_z, f_theta, coord_fys, coords_num)
            eq_res = torch.stack(list(eq_res.values()))
            loss_eq = condition(eq_res, torch.zeros_like(eq_res))
            loss_eq = torch.clamp(loss_eq, max=1e5)
            loss_eq_value = loss_eq.detach().cpu().item()

            if not torch.isfinite(data_loss) or not torch.isfinite(loss_eq):
                print(f"Skipping batch {batch_idx + 1}: non-finite loss.")
                invalid_batch_path = save_invalid_batch(
                    training_config=train_config,
                    state=state,
                    batch_idx=batch_idx,
                    batch=batch,
                    f_theta=f_theta,
                    coord_fys=coord_fys,
                    eq_res=eq_res,
                    data_loss_value=data_loss_value,
                    eq_loss_value=loss_eq_value,
                )
                save_checkpoint(train_config,
                                skipped_batch_info={
                                    "reason": "non-finite loss",
                                    "batch_idx": batch_idx,
                                    "data_loss": data_loss_value,
                                    "eq_loss": loss_eq_value,
                                    "saved_batch": str(invalid_batch_path),
                                })
                optimizer.zero_grad(set_to_none=True)
                del f_theta, coord_fys, pred_da, pred_da_center, data_loss, eq_res, loss_eq, batch
                processed_batches += 1
                state["batch_idx"] = batch_idx + 1
                continue


            if is_training:
                (data_loss * train_config["da_weight"] + loss_eq * train_config["eq_weight"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_weighted_loss = data_loss_value * train_config["da_weight"] + loss_eq_value * train_config["eq_weight"]
            num_loss_batches += 1
            
            history.append(split=state["split"], loss_type="raw",
                           values={"total": data_loss_value + loss_eq_value, 
                                   "da": data_loss_value,
                                   "eq": loss_eq_value})

            history.append(split=state["split"], loss_type="weighted",
                           values={"total": batch_weighted_loss,
                                   "da": data_loss_value * train_config["da_weight"],
                                   "eq": loss_eq_value * train_config["eq_weight"]}
                                   )
            
            total_loss += batch_weighted_loss
            status_data += data_loss_value
            status_eq += loss_eq_value
            status_loss_batches += 1

            del f_theta, coord_fys, pred_da, pred_da_center, data_loss, eq_res, loss_eq, batch

            processed_batches += 1
            state["batch_idx"] = batch_idx + 1

            if (train_config["status_frequency"]
                and status_loss_batches
                and processed_batches % train_config["status_frequency"] == 0
                ):

                print(f"{state['split']}, Batch {batch_idx + 1}/{total_loader_batches}\t"
                      f"raw({(status_data + status_eq) / status_loss_batches:.3e}, "
                      f"{status_data / status_loss_batches:.3e}, "
                      f"{status_eq / status_loss_batches:.3e}), "
                      f"weighted({status_data / status_loss_batches * train_config['da_weight'] + status_eq / status_loss_batches * train_config['eq_weight']:.3e}, "
                      f"{status_data / status_loss_batches * train_config['da_weight']:.3e}, "
                      f"{status_eq / status_loss_batches * train_config['eq_weight']:.3e}))"
                      )
                status_data = 0.0
                status_eq = 0.0
                status_loss_batches = 0
    finally:
        if param_requires_grad is not None:
            for param, old_value in zip(model.parameters(), param_requires_grad):
                param.requires_grad_(old_value)

    if train_config["status_frequency"] and status_loss_batches:
        print(f"{state['split']}, Batch {state['batch_idx']}/{total_loader_batches}\t"
              f"raw({(status_data + status_eq) / status_loss_batches:.3e}, "
              f"{status_data / status_loss_batches:.3e}, "
              f"{status_eq / status_loss_batches:.3e}), "
              f"weighted({status_data / status_loss_batches * train_config['da_weight'] + status_eq / status_loss_batches * train_config['eq_weight']:.3e}, "
              f"{status_data / status_loss_batches * train_config['da_weight']:.3e}, "
              f"{status_eq / status_loss_batches * train_config['eq_weight']:.3e}))"
              )

    state["batch_idx"] = max(0, last_batch_idx + 1)
    history.flush()
    save_checkpoint(train_config,
                    model_state=model,
                    optimizer=optimizer,
                    condition=condition,
                    epoch=state["epoch"],
                    split=state["split"],
                    batch_idx=state["batch_idx"],
                    best_val_loss=state["best_val_loss"],
                    patience_counter=state["patience_counter"])
    

    del loader
    avg_loss = total_loss / num_loss_batches if num_loss_batches else float("inf")

    return avg_loss
