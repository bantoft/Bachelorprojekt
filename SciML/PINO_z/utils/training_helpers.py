from __future__ import annotations

import json
from pathlib import Path

import neuralop as nop
import torch
from torch.utils.data import ConcatDataset, DataLoader



SPLITS = ("training", "validation", "test")
LOSS_TYPES = ("raw", "weighted")
LOSS_KEYS = ("total", "da", "eq")

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
                    scheduler: torch.optim.lr_scheduler.CyclicLR | None = None,
                    epoch: int| None = None,
                    split: str| None = None,
                    batch_idx: int | None = None,
                    best_val_loss: float| None = None,
                    patience_counter: int| None = None,
                    skipped_batch_info: dict| None = None
                    ):
    
    filename = Path(training_config["data_dir"]).resolve() / "checkpoint.pt"
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False) if filename.exists() else {}

    checkpoint["training_config"] = training_config
    checkpoint["torch_rng_state"] = torch.get_rng_state().cpu().numpy().tobytes()

    save_dict = {"model_state_dict": None if model_state is None else model_state.state_dict(),
                 "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                 "condition_state_dict": None if condition is None else condition.state_dict(),
                 "best_val_model": None if best_val_model is None else best_val_model.state_dict(),
                 "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
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
    torch.save(checkpoint, filename)


def load_checkpoint(training_config):
    filename = Path(training_config["data_dir"]).resolve() / "checkpoint.pt"

    if not filename.exists():
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    return torch.load(
        filename,
        map_location="cpu",
        weights_only=False,
    )





data_loader_setup1 = {
    "TSSplit": True,
    "train_split": 0.8,
    "val_split": 0.1,
}

data_loader_setup2 = {
    "TSSplit": False,
    "train_split": 0.8,
    "val_split": 0.1,
}

def init_experinment(train_cfg):
    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys
    from SciML.PINO_z.utils.data_loader import make_dataloaders
    from SciML.PINO_z.utils.wrapper import WrappedFNO

    # Laver dobbelt data loader setup
    if len(train_cfg["root"]) == 2:
        info1 = BOUTHESELInfo(train_cfg["root"][0])
        info2 = BOUTHESELInfo(train_cfg["root"][1])

        data_loader_dict = {
            "z_width": train_cfg["z_width"],
            "batch_size": train_cfg["batch_size"],
            "val_split": train_cfg["val_split"],
            "shuffle": train_cfg["shuffle"],
            "num_workers": train_cfg["num_workers"],
            "prefetch_factor": train_cfg["prefetch_factor"],
            "pin_memory": train_cfg["pin_memory"]}
        
        for key, value in data_loader_dict.items():
            data_loader_setup1[key] = value
            data_loader_setup2[key] = value

        train_loader1, val_loader1, test_loader1 = make_dataloaders(info1, data_loader_setup1)
        train_loader2, val_loader2, test_loader2 = make_dataloaders(info2, data_loader_setup2)

        train_dataset = ConcatDataset([train_loader1.dataset, train_loader2.dataset])

        train_loader = DataLoader(
                        train_dataset,
                        batch_size=train_config["batch_size"],
                        shuffle=train_config["shuffle"],
                        num_workers=train_config["num_workers"],
                        pin_memory=train_config["pin_memory"],
                        prefetch_factor=train_config["prefetch_factor"],
                    )




    info = BOUTHESELInfo(train_cfg["root"])
    phys = BOUTHESELPhys(info)
    data_dir = Path(train_cfg["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    model = WrappedFNO(info=info,
                       phys=phys,
                       m=train_cfg["z_width"],
                       fno = nop.models.FNO(**train_cfg["fno"])
                       ).to(info.device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=train_cfg["lr"],
                             betas= (0.7, 0.95),
                             eps=1e-8,
                             weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  max_lr=5e-4,
                                                  base_lr=train_cfg["lr"],
                                                  step_size_up=900,
                                                  step_size_down=900,
                                                  mode="triangular2",
                                                  cycle_momentum=False)

    return (model,
            *make_dataloaders(info=info, train_config=train_cfg),
            torch.nn.MSELoss(),
            optimizer,
            scheduler,
            HistoryBuffer(data_dir / "history.json"),
            phys,
            info
            )

def chunk_info(train_loader, train_config):
    sample_batch = next(iter(train_loader))
    u = sample_batch[0]  # samme som i iterator(...)
    u_chunks = u.chunk(train_config["num_eq_chunk"], dim=2)
    chunk_sizes = [chunk.shape[2] for chunk in u_chunks]

    print(f"Requested num_eq_chunk: {train_config['num_eq_chunk']}")
    print(f"Actual number of chunks: {len(u_chunks)}")
    print(f"Chunk sizes along dim=2: {chunk_sizes}")

def iterator(loader,
             start_batch_idx,
             train_config,
             total_loader_batches,
             state,
             history,
             model,
             optimizer,
             scheduler,
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
    status_total_loss = 0.0
    status_da_loss = 0.0
    status_eq_loss = 0.0
    status_loss_batches = 0
    try:
        for batch_idx, batch in enumerate(loader, start=start_batch_idx):

            if state["batch_idx"] % train_config["flush_frequency"] == 0:
                history.flush()
                save_checkpoint(train_config,
                                model_state=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                condition=condition,
                                split=state["split"],
                                epoch=state["epoch"],
                                batch_idx=state["batch_idx"],
                                best_val_loss=state["best_val_loss"],
                                patience_counter=state["patience_counter"]
                                )

            batch = [x.to(info.device, non_blocking=True) for x in batch]
            last_batch_idx = batch_idx

            # Data loss
            optimizer.zero_grad(set_to_none=True)
            if is_training:
                f_theta, _ = model(batch, requires_coord_grad=False)
            else:
                with torch.no_grad():
                    f_theta, _ = model(batch, requires_coord_grad=False)
            pred_da = torch.cat([f_theta[name] for name in ("lnn", "lnpe", "lnpi", "phi")], dim=1)
            data_loss = condition(pred_da, batch[1])
            data_loss_value = data_loss.detach().cpu().item()

            if not torch.isfinite(data_loss):
                print(f"Skipping batch {batch_idx + 1}: non-finite data loss.")
                save_checkpoint(train_config,
                                skipped_batch_info={
                                    "reason": "non-finite data loss",
                                    "batch_idx": batch_idx,
                                    "data_loss": data_loss_value,
                                })
                del f_theta, pred_da, data_loss, batch
                processed_batches += 1
                state["batch_idx"] = batch_idx + 1
                continue

            if is_training:
                data_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            del f_theta, pred_da, data_loss

            # Equation loss
            optimizer.zero_grad(set_to_none=True)
            skip_eq_step = False

            u, _, avg_z, coords_fys, coords_num = batch
            u_chunks = u.chunk(train_config["num_eq_chunk"], dim=2)
            num_chunks = len(u_chunks)
            avg_chunks = avg_z.chunk(num_chunks, dim=2)
            num_coord_chunks = coords_num.chunk(num_chunks, dim=2)
            loss_total_value_eq_chunk = 0.0

            for chunk_idx, (u_chunk, avg_chunk, num_chunk) in enumerate(zip(u_chunks, avg_chunks, num_coord_chunks)):
                chunk_batch = (u_chunk, None, avg_chunk, coords_fys, num_chunk)
                f_chunk, coord_chunk = model(
                    chunk_batch,
                    chunking=True,
                    requires_coord_grad=True,
                )
                eq_res_chunk = phys.eq_res(avg_chunk, f_chunk, coord_chunk, num_chunk)
                eq_res_chunk = torch.stack(list(eq_res_chunk.values()))
                loss_eq = condition(eq_res_chunk, torch.zeros_like(eq_res_chunk)) / num_chunks

                if not torch.isfinite(loss_eq):
                    print(f"Skipping batch {batch_idx + 1}: non-finite eq loss at chunk {chunk_idx + 1}.")
                    save_checkpoint(train_config,
                                skipped_batch_info={
                                    "reason": "non-finite eq loss",
                                    "batch_idx": batch_idx,
                                    "data_loss": data_loss_value,
                                    "chunk_idx": chunk_idx,
                                })
                    
                    skip_eq_step = True
                    optimizer.zero_grad(set_to_none=True)
                    del f_chunk, coord_chunk, eq_res_chunk, loss_eq
                    break

                loss_total_value_eq_chunk += loss_eq.item()

                if is_training:
                    loss_eq.backward(retain_graph=chunk_idx < num_chunks - 1)

                del f_chunk, coord_chunk, eq_res_chunk, loss_eq

            if not skip_eq_step and is_training:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if not skip_eq_step:
                batch_total_loss = data_loss_value + loss_total_value_eq_chunk
                total_loss += batch_total_loss
                num_loss_batches += 1
                status_total_loss += batch_total_loss
                status_da_loss += data_loss_value
                status_eq_loss += loss_total_value_eq_chunk
                status_loss_batches += 1
                history.append(
                    split=state["split"],
                    loss_type="raw",
                    values={
                        "total": batch_total_loss,
                        "da": data_loss_value,
                        "eq": loss_total_value_eq_chunk},
                    )
                history.append(
                    split=state["split"],
                    loss_type="weighted",
                    values={
                        "total": batch_total_loss,
                        "da": data_loss_value,
                        "eq": loss_total_value_eq_chunk},
                    )

            del u, avg_z, coords_fys, coords_num
            del u_chunks, avg_chunks, num_coord_chunks, batch

            processed_batches += 1
            state["batch_idx"] = batch_idx + 1

            if (
                train_config["status_frequency"]
                and status_loss_batches
                and processed_batches % train_config["status_frequency"] == 0
            ):
                print(
                    f"{state['split']}, Batch {batch_idx + 1}/{total_loader_batches} |\t "
                    f"avg total_loss: {status_total_loss / status_loss_batches:.6e} | "
                    f"avg da_loss: {status_da_loss / status_loss_batches:.6e} | "
                    f"avg eq_loss: {status_eq_loss / status_loss_batches:.6e}"
                )
                status_total_loss = 0.0
                status_da_loss = 0.0
                status_eq_loss = 0.0
                status_loss_batches = 0
    finally:
        if param_requires_grad is not None:
            for param, old_value in zip(model.parameters(), param_requires_grad):
                param.requires_grad_(old_value)

    if train_config["status_frequency"] and status_loss_batches:
        print(
            f"{state['split']}, Batch {state['batch_idx']}/{total_loader_batches} | "
            f"avg total_loss: {status_total_loss / status_loss_batches:.6e} | "
            f"avg da_loss: {status_da_loss / status_loss_batches:.6e} | "
            f"avg eq_loss: {status_eq_loss / status_loss_batches:.6e}"
        )

    state["batch_idx"] = max(0, last_batch_idx + 1)
    history.flush()
    save_checkpoint(train_config,
                    model_state=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    condition=condition,
                    epoch=state["epoch"],
                    split=state["split"],
                    batch_idx=state["batch_idx"],
                    best_val_loss=state["best_val_loss"],
                    patience_counter=state["patience_counter"])
    

    del loader
    avg_loss = total_loss / num_loss_batches if num_loss_batches else float("inf")

    return avg_loss
