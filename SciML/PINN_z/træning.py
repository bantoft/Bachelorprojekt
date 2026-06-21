import sys
import torch
import shutil
import itertools

import numpy as np

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


from SciML.PINN_z.utils.training_helpers import (
    init_experinment,
    load_checkpoint,
    save_checkpoint,
    iterator,
)


# Til endelige test
train_config = {
    "resume": False,
    # root er folder med output fra simulering
    "root" : [ROOT_DIR / r"sim_data/Alexander_e", ROOT_DIR / r"sim_data/Alexander_phi_0"],
    # data_dir er folder hvor experiment data og modeller gemmes/læses alt efter resume
    "data_dir": ROOT_DIR / r"Experimenter/PINN_z/Eksperiment_2",
    "seed": np.random.randint(0, 2**32 - 1),
    "lr": 1e-4,
    "da_weight": 1,
    "eq_weight": 1e2,
    "status_frequency": 5000,
    "flush_frequency": 10000, # <- Flush på server er relativt langsom i forhold til lokalt
    "z_width": 3,
    "batch_size": 2**9,
    "train_split": 0.8,
    "val_split": 0.1,
    "shuffle": True,
    "num_workers": 4,
    "prefetch_factor": 2,
    "pin_memory": True,
    "persistent_workers": True,
    "epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0,
    "network": {
        "input_size": 12, # Overskrives i wrapperen til 12 * nx * z_width
        "output_size": 4, # Overskrives i wrapperen til 4 * nx * z_width
        "output_names": ("lnn", "lnpe", "lnpi", "phi"),
        "layers": [
            {"size": 2**3, "non_lin_foo": torch.nn.SiLU},
            {"size": 2**3, "non_lin_foo": torch.nn.SiLU},
        ],
    }
}
# Tanh, SilU

state = {
    "epoch": 0,
    "split": "training",
    "batch_idx": 0,
    "best_val_loss": float("inf"),
    "patience_counter": 0,
}


def _strip_state_dict_metadata(state_dict):
    if state_dict is None:
        raise ValueError("Checkpoint is missing a model state_dict.")
    if "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)
    return state_dict


def _load_model_state(model, state_dict):
    try:
        model.load_state_dict(_strip_state_dict_metadata(state_dict))
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with the current PINN_z network. "
            "Set `resume` to False or choose a new `data_dir` for a fresh run."
        ) from exc



# Resume logik
if train_config["resume"] and (train_config["data_dir"] / "checkpoint.pt").exists():
    print(f"Resuming training from checkpoint in {train_config['data_dir']}")
    ckpt = load_checkpoint(train_config)
    train_config = ckpt["training_config"]
    if "network" not in train_config:
        raise RuntimeError(
            "Checkpoint was created before PINN_z switched from FNO to an MLP. "
            "Set `resume` to False or choose a new `data_dir` for a fresh run."
        )
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
     history,
     phys,
     info,
    ) = init_experinment(train_config)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    total_test_batches = len(test_loader)

    for name in ("g_x", "H_x", "D_x"):
        buffer = model._buffers.get(name)
        if isinstance(buffer, torch.Tensor):
            model._buffers[name] = buffer.clone()
    _load_model_state(model, ckpt["model_state_dict"])
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
        epoch=state["epoch"],
        split=state["split"],
        batch_idx=state["batch_idx"],
        best_val_loss=state["best_val_loss"],
        patience_counter=state["patience_counter"],
    )

width = shutil.get_terminal_size().columns
for name, value in train_config.items():
    print(f"{name}: {value}".center(width))

processed_batches = 0
last_batch_idx = state["batch_idx"] - 1


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
        
    
        
        if val_loss < state["best_val_loss"] - train_config["early_stopping_min_delta"]:
            state["best_val_loss"] = val_loss
            state["patience_counter"] = 0
            save_checkpoint(train_config,
                            best_val_model=model,
                            model_state=model,
                            optimizer=optimizer,
                            condition=condition,
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
                            epoch=state["epoch"],
                            split=state["split"],
                            batch_idx=state["batch_idx"],
                            best_val_loss=state["best_val_loss"],
                            patience_counter=state["patience_counter"]
                            )

        if state["patience_counter"] >= train_config["early_stopping_patience"]:
            print("Early stopping triggered: Patience limit reached.")
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
                        epoch=state["epoch"],
                        split=state["split"],
                        batch_idx=state["batch_idx"],
                        best_val_loss=state["best_val_loss"],
                        patience_counter=state["patience_counter"]
                        )

    print("")


if (train_config["data_dir"] / "checkpoint.pt").exists():
    ckpt = load_checkpoint(train_config)
    best_val_model_state = ckpt.get("best_val_model")
    if best_val_model_state is not None:
        model.load_state_dict(_strip_state_dict_metadata(best_val_model_state))

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
    
    history.flush()
