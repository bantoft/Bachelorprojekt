# Fiks path for imports
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
import itertools
import torch.nn as nn

from util.PINN.model import PINN
from util.PINN.structure import NN_STRUCTURE
from util.PINN.data_loader import make_dataloader, move_batch_to_device
from util.PINN.træning_helpers import step, HistoryBuffer, save_checkpoint


def init_trainer(training_config: dict):
    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys

    root = Path(__file__).parents[2] / training_config["root"]
    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info)

    condition = nn.MSELoss().to(info.device)
    model = PINN(NN_STRUCTURE)
    model.to(info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    train_loader, val_loader, test_loader = make_dataloader(info, training_config)

    return (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader)


if __name__ == "__main__":

    training_config = {
        # !! Paths skal være referet fra projekt root folder !!
        # Folder med simulerings data og settings fil
        "root": r"sim_data/data_15_512_Alexander_",
        # Output folder for checkpoints, history og best model
        # Kald folderen for: history<...> så det ikke flyttes til repo
        "out_folder": r"SciML/PINN/history_test",
        # "out_folder": r"SciML2/PINN/eksperiment_data",
        # Hvis resume er True læses checkpoint fra out_folder ellers startes ny træningen
        # Hvis checkpoint ikke findes, laves ny checkpoint og træningsdata appendes hvis history.json findes
        # Intet slettes
        "resume": True,
        
        # Optimization
        "lr": 1e-4,
        "epochs": 10,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1000,

        # Logging / status
        "status_frequency": 10,
        "flush_frequency": 10,

        # Dataset split
        "train_ratio": 0.8,
        "val_ratio": 0.1,

        # Loss weights
        "da": 500.0,
        "ic": 1.0,
        "eq": 25000.0, # Brug opløsigligheden af (num_x*num_z)
        "bc": 1.0,

        # DataLoader
        "batch_size": 2**10,
        "shuffle": True, # Bruges kun i val/test loader, da sampler bruges i train loader
        "pin_memory": True,
        "persistent_workers": True,
        "num_workers": 4,
        "prefetch_factor": 1,
    }

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    start_batch_idx = 0

    data_dir = (ROOT_DIR / training_config["out_folder"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resume logik
    if training_config["resume"]:
        if not (data_dir/"checkpoint.pt").exists():
            print(f"Could not find checkpoint at {data_dir/'checkpoint.pt'}")
            print("Do you want to start a new training? (y/n)")
            user_input = input().lower().strip()

            if user_input != "y":
                # Stop the program completely
                import os
                import psutil
                psutil.Process(os.getpid()).terminate()

            training_config["resume"] = False
            (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader) = init_trainer(training_config)

        else:
            import numpy as np
            checkpoint = torch.load(data_dir / "checkpoint.pt", weights_only=False)
            training_config = checkpoint["training_config"]
            training_config["resume"] = True
            (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader) = init_trainer(training_config)

            checkpoint = torch.load(
                data_dir / "checkpoint.pt",
                map_location=info.device,
                weights_only=False,
            )
            torch.set_rng_state(
                torch.from_numpy(
                    np.frombuffer(checkpoint["torch_rng_state"], dtype=np.uint8).copy()
                )
            )
            model = PINN(checkpoint["model_structure"])
            model.to(info.device)
            model.load_state_dict(checkpoint["model_state_dict"])

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            condition.load_state_dict(checkpoint["condition_state_dict"])

            patience_counter = checkpoint["patience_counter"]
            start_batch_idx = checkpoint["batch_idx"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            start_epoch = checkpoint["epoch"]
    
    else:
        (info,
        phys,
        condition,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader) = init_trainer(training_config)

    history = HistoryBuffer(data_dir / "training_history.jsonl", )
    best_val_model_path = data_dir / "best_val_model.pt"

    last_epoch = 0

    for epoch in range(start_epoch, training_config["epochs"]):
        print(f"Epoch {epoch+1}/{training_config['epochs']}\nPatience Counter: {patience_counter}/{training_config['early_stopping_patience']}\nBest Val Loss: {best_val_loss:.6f}")

        batch_iter = train_loader
        first_batch = 0
        if epoch == start_epoch:
            first_batch = start_batch_idx
            batch_iter = itertools.islice(train_loader, first_batch, None)
            print(f"Resuming training from batch {first_batch} of epoch {epoch+1}")


        model.train()
        for batch_idx, batch in enumerate(batch_iter, start=first_batch):

            avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])
            loss_total = step(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, split="train")
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            if batch_idx % training_config["status_frequency"] == 0:
                print(f"Training {batch_idx}/{len(train_loader)}\t Loss: {loss_total.item():.6f}")

            if batch_idx % training_config["flush_frequency"] == 0:
                history.flush()
                save_checkpoint(model, optimizer, condition, epoch, batch_idx, patience_counter, best_val_loss, training_config, data_dir / "checkpoint.pt")
            
            if batch_idx == 100: break

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            for batch_idx, batch in enumerate(val_loader):
                avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])
                loss_total = step(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, split="validation")
                val_loss_total += loss_total.item()
            
                if batch_idx % training_config["status_frequency"] == 0:
                    print(f"Validation {batch_idx}/{len(val_loader)}\t Loss: {loss_total.item():.6f}")

                if batch_idx % training_config["flush_frequency"] == 0:
                    history.flush()
                    save_checkpoint(model, optimizer, condition, epoch, batch_idx, patience_counter, best_val_loss, training_config, data_dir / "checkpoint.pt")
                
                if batch_idx == 100: break
            
            avg_val_loss = val_loss_total / len(val_loader)
            if avg_val_loss < best_val_loss - training_config["early_stopping_min_delta"]:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_val_model_path)
                print(f"New best model saved with val loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement in val loss. Patience counter: {patience_counter}/{training_config['early_stopping_patience']}")
                if patience_counter >= training_config["early_stopping_patience"]:
                    last_epoch = epoch
                    print("Early stopping triggered. Ending training.")
                    break
    
    with torch.no_grad():
        test_loss_total = 0.0
        for batch_idx, batch in enumerate(test_loader):
            avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])
            loss_total = step(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, split="test")
            test_loss_total += loss_total.item()
            
            if batch_idx % training_config["status_frequency"] == 0:
                print(f"Test {batch_idx}/{len(test_loader)}\t Loss: {loss_total.item():.6f}")

            if batch_idx % training_config["flush_frequency"] == 0:
                history.flush()
                save_checkpoint(model, optimizer, condition, last_epoch, batch_idx, patience_counter, best_val_loss, training_config, data_dir / "checkpoint.pt")
            
            if batch_idx == 100: break
        
        avg_test_loss = test_loss_total / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss:.6f}")
        