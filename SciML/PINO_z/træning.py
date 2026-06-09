import sys
import torch
import shutil
import itertools

import numpy as np

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


from SciML.PINO_z.utils.training_helpers import (
    init_experinment,
    load_checkpoint,
    save_checkpoint,
    iterator,
    chunk_info
)


# Til endelige test
train_config = {
    "resume": False,
    "root": ROOT_DIR / r"sim_data/data_15_512_Alexander_",
    "data_dir": ROOT_DIR / r"SciML/PINO_z/experiments/test_run_4",
    "seed": np.random.randint(0, 2**32 - 1),
    "status_frequency": 1,
    "flush_frequency": 2,
    "z_width": 3,
    "num_eq_chunk": 24,
    "batch_size": 128,
    "train_split": 0.8,
    "val_split": 0.1,
    "shuffle": True,
    "num_workers": 4,
    "prefetch_factor": 1,
    "pin_memory": True,
    "lr": 5e-4,
    "epochs": 10,
    "early_stopping_patience": 2,
    "early_stopping_min_delta": 0.0,
    "fno": {
        "n_modes": (140, 140),
        "in_channels": 8,
        "out_channels": 4,
        "hidden_channels": 25,
        "positional_embedding": None,
    }
}


state = {
    "epoch": 0,
    "split": "training",
    "batch_idx": 0,
    "best_val_loss": float("inf"),
    "patience_counter": 0,
}



# Resume logik
if train_config["resume"] and (train_config["data_dir"] / "checkpoint.pt").exists():
    print(f"Resuming training from checkpoint in {train_config['data_dir']}")
    ckpt = load_checkpoint(train_config)
    train_config = ckpt["training_config"]
    state["epoch"] = ckpt["epoch"]
    state["split"] = ckpt["split"]
    state["batch_idx"] = ckpt["batch_idx"]
    state["best_val_loss"] = ckpt["best_val_loss"]
    state["patience_counter"] = ckpt["patience_counter"]

    torch.set_rng_state(torch.from_numpy(np.frombuffer(ckpt["torch_rng_state"], dtype=np.uint8).copy()))

    (model,
     train_loader,
     val_loader,
     test_loader,
     condition,
     optimizer,
     scheduler,
     history,
     phys,
     info,
    ) = init_experinment(train_config)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    total_test_batches = len(test_loader)

    for name in ("g_x", "H_x", "D_x"):
        if name in model._buffers and model._buffers[name] is not None:
            model._buffers[name] = model._buffers[name].clone()
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    condition.load_state_dict(ckpt["condition_state_dict"])
else:
    print("Starting new training run.")
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config["seed"])

    (model,
     train_loader,
     val_loader,
     test_loader,
     condition,
     optimizer,
     scheduler,
     history,
     phys,
     info,
    ) = init_experinment(train_config)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    total_test_batches = len(test_loader)
    save_checkpoint(
        train_config,
        model_state=model,
        optimizer=optimizer,
        condition=condition,
        scheduler=scheduler,
        epoch=state["epoch"],
        split=state["split"],
        batch_idx=state["batch_idx"],
        best_val_loss=state["best_val_loss"],
        patience_counter=state["patience_counter"],
    )

processed_batches = 0
last_batch_idx = state["batch_idx"] - 1

chunk_info(train_loader, train_config)


while state["epoch"] < train_config["epochs"]:
    width = shutil.get_terminal_size().columns
    print(str(state).center(width)) 

    if state["split"] == "training":
        model.train()
        start_batch_idx = state["batch_idx"]
        loader = train_loader
        if start_batch_idx:
            loader = itertools.islice(train_loader, start_batch_idx, None)

        total_loader_batches = total_train_batches
        _ = iterator(loader,
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
                     phys)
        state["split"] = "validation"
        state["batch_idx"] = 0
        processed_batches = 0
        last_batch_idx = -1
    
    print("")
        
    if state["split"] == "validation":
        model.eval()
        start_batch_idx = state["batch_idx"]
        loader = val_loader
        if start_batch_idx:
            loader = itertools.islice(val_loader, start_batch_idx, None)

        total_loader_batches = total_val_batches
        val_loss = iterator(loader,
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
                            phys)
        
        scheduler.step(val_loss)
        
        if val_loss < state["best_val_loss"] - train_config["early_stopping_min_delta"]:
            state["best_val_loss"] = val_loss
            state["patience_counter"] = 0
            save_checkpoint(train_config,
                            best_val_model=model,
                            model_state=model,
                            optimizer=optimizer,
                            condition=condition,
                            scheduler=scheduler,
                            epoch=state["epoch"],
                            split=state["split"],
                            batch_idx=state["batch_idx"],
                            best_val_loss=state["best_val_loss"],
                            patience_counter=state["patience_counter"]
                            )
        else:
            state["patience_counter"] += 1
            save_checkpoint(train_config,
                            model_state=model,
                            optimizer=optimizer,
                            condition=condition,
                            scheduler=scheduler,
                            epoch=state["epoch"],
                            split=state["split"],
                            batch_idx=state["batch_idx"],
                            best_val_loss=state["best_val_loss"],
                            patience_counter=state["patience_counter"]
                            )

        if state["patience_counter"] >= train_config["early_stopping_patience"]:
            print("Early stopping triggered: Patience limit reached.")
            break
            
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < 1e-65:
            print("Early stopping triggered: Learning rate below threshold.")
            break


        state["epoch"] += 1
        state["split"] = "training"
        state["batch_idx"] = 0
        processed_batches = 0
        last_batch_idx = -1
        save_checkpoint(train_config,
                        model_state=model,
                        optimizer=optimizer,
                        condition=condition,
                        scheduler=scheduler,
                        epoch=state["epoch"],
                        split=state["split"],
                        batch_idx=state["batch_idx"],
                        best_val_loss=state["best_val_loss"],
                        patience_counter=state["patience_counter"]
                        )

if (train_config["data_dir"] / "checkpoint.pt").exists():
    ckpt = load_checkpoint(train_config)
    best_val_model_state = ckpt.get("best_val_model")
    if best_val_model_state is not None:
        model.load_state_dict(best_val_model_state)

state["split"] = "test"
state["batch_idx"] = 0
processed_batches = 0
last_batch_idx = -1

if state["split"] == "test":
    model.eval()
    start_batch_idx = state["batch_idx"]
    loader = test_loader
    if start_batch_idx:
        loader = itertools.islice(test_loader, start_batch_idx, None)

    total_loader_batches = total_test_batches
    _ = iterator(loader,
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
                 phys)
