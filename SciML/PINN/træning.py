# Fiks path for imports
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
import itertools

from SciML.PINN.util.model import PINN
from SciML.PINN.util.træning_helpers import HistoryBuffer, save_checkpoint, iterate, init_trainer



if __name__ == "__main__":

    training_config = {
        "resume": False,
        # !! Paths skal være referet fra ROOT_DIR !!
        # Folder med dump filer og settings fil
        "root" : [r"sim_data/Alexander_e"],
        # Bool eller liste med samme længde som root, fx [True, False] for 2 datasæt.
        # True = timeseries split, False = random sample split.
        "TSSplit": [True],
        # Output folder for checkpoints, history og best model
        # Kald folderen for: history<...> så det ikke flyttes til repo
        "out_folder": r"Experimenter/PINN/test_run",
        # Hvis resume er True læses checkpoint fra out_folder ellers startes ny træningen
        # Hvis checkpoint ikke findes, laves ny checkpoint og træningsdata appendes hvis history.json findes
        # Intet slettes
        
        # Optimization
        "lr": 1e-4,
        "epochs": 1,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,

        # Logging / status
        "status_frequency": 100,
        "flush_frequency": 500000,

        # Dataset split
        "train_ratio": 0.8,
        "val_ratio": 0.1,

        # Loss weights
        "da": 500,
        "ic": 1,
        "eq": 1e3,
        "bc": 1,

        # DataLoader
        "batch_size": 3600,
        "shuffle": True, # Bruges kun i val/test loader, da sampler bruges i train loader
        "pin_memory": True,
        "persistent_workers": True,
        "num_workers": 8,
        "prefetch_factor": 2,
    }

    data_dir = (ROOT_DIR / training_config["out_folder"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryBuffer(data_dir / "training_history.jsonl")
    best_val_model_path = data_dir / "best_val_model.pt"
    checkpoint_path = data_dir / "checkpoint.pt"

    split = "training"
    start_epoch = 0
    start_batch_idx = 0
    patience_counter = 0
    best_val_loss = float("inf")

    # Resume logik
    if training_config["resume"]:
        if not checkpoint_path.exists():
            print(f"Could not find checkpoint at {checkpoint_path}")
            print("Do you want to start a new training? (y/n)")
            user_input = input().lower().strip()

            if user_input != "y":
                import os
                import psutil
                psutil.Process(os.getpid()).terminate()

            training_config["resume"] = False
            (
                info,
                phys,
                condition,
                model,
                optimizer,
                train_loader,
                val_loader,
                test_loader,
            ) = init_trainer(training_config)
            save_checkpoint(training_config=training_config,
                            model=model,
                            condition=condition,
                            optimizer=optimizer,
                            epoch=0,
                            split=split,
                            batch_idx=start_batch_idx,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter)
        else:
            import numpy as np

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            training_config = checkpoint["training_config"]
            training_config["resume"] = True

            # Hvis du får 
            # training_config["num_workers"] = 0  # Sæt num_workers til 0 ved resume for at undgå problemer med DataLoader og multiprocessing
            # training_config["pin_memory"] = False  # Sæt pin_memory til False ved resume for at undgå problemer med DataLoader og multiprocessing
            # training_config["persistent_workers"] = None  # Sæt persistent_workers til False ved resume for at undgå problemer med DataLoader og multiprocessing
            # training_config["prefetch_factor"] = None  # Reducer prefetch_factor ved resume for at undgå problemer med DataLoader og multiprocessing

            (
                info,
                phys,
                condition,
                model,
                optimizer,
                train_loader,
                val_loader,
                test_loader,
            ) = init_trainer(training_config)

            checkpoint = torch.load(
                checkpoint_path,
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

            patience_counter = checkpoint.get("patience_counter", 0)
            start_batch_idx = checkpoint.get("batch_idx", -1) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            start_epoch = checkpoint.get("epoch", 0)
            split = checkpoint.get("split", "training")
            print(f"""
                  Resuming from epoch {start_epoch + 1}, split={split}, batch={start_batch_idx}
                  """)
    else:
        (
            info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
        ) = init_trainer(training_config)
        save_checkpoint(training_config=training_config,
                        model=model,
                        condition=condition,
                        optimizer=optimizer,
                        epoch=0,
                        split=split,
                        batch_idx=start_batch_idx,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter)


        training_config["epochs"] = 1


    for key, value in training_config.items():
        print(f"{key}: {value}")
    
    for epoch in range(start_epoch, training_config["epochs"]):
        print(f"""
              Epoch {epoch+1}/{training_config['epochs']}
              Patience Counter: {patience_counter}
              Best Val Loss: {best_val_loss}
              """)
        

        if split == "training":
            batch_iter = itertools.islice(train_loader, start_batch_idx, None)
            num_batches = len(train_loader)
            _ = iterate(info=info,
                    phys=phys,
                    optimizer=optimizer,
                    model=model,
                    condition=condition,
                    batch_iter=batch_iter,
                    num_batches=num_batches,
                    history=history,
                    training_config=training_config,
                    split=split,
                    first_batch=start_batch_idx)
            
            start_batch_idx = 0
            split = "validation"
            history.flush()
            save_checkpoint(training_config=training_config,
                            model=model,
                            optimizer=optimizer,
                            batch_idx=start_batch_idx,
                            split=split)
        
        if split == "validation":
            batch_iter = itertools.islice(val_loader, start_batch_idx, None)
            num_batches = len(val_loader)
            avg_loss = iterate(info=info,
                    phys=phys,
                    optimizer=optimizer,
                    model=model,
                    condition=condition,
                    batch_iter=batch_iter,
                    num_batches=num_batches,
                    history=history,
                    training_config=training_config,
                    split=split,
                    first_batch=start_batch_idx)
            
            split = "training"
            start_batch_idx = 0


            if avg_loss < best_val_loss - training_config["early_stopping_min_delta"]:
                best_val_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_val_model_path)

            else:
                patience_counter += 1
                if patience_counter >= training_config["early_stopping_patience"]:
                    print("Early stopping triggered. Ending training.")
                    split = "test"

            if epoch == training_config["epochs"] - 1:
                print("Last epoch reached. Running test split.")
                split = "test"

        history.flush()
        save_checkpoint(training_config=training_config,
                    batch_idx=start_batch_idx,
                    split=split,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    epoch=epoch)



        if split == "test":
            batch_iter = itertools.islice(test_loader, start_batch_idx, None)
            num_batches = len(test_loader)
            iterate(info=info,
                    phys=phys,
                    optimizer=optimizer,
                    model=model,
                    condition=condition,
                    batch_iter=batch_iter,
                    num_batches=num_batches,
                    history=history,
                    training_config=training_config,
                    split=split,
                    first_batch=start_batch_idx)
            break

        history.flush()
        save_checkpoint(training_config=training_config,
                        model=model,
                        optimizer=optimizer,
                        batch_idx=start_batch_idx,
                        split=split,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter)

        
