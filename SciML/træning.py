import torch

import torch.nn as nn
from pathlib import Path
import itertools

from util.model import PINN
from util.structure import NN_STRUCTURE
from util.data_loader import make_dataloader, move_batch_to_device
from util.træning_helpers import get_loss, HistoryWriter, save_checkpoint

def init_trainer(training_config: dict):
    from loss_PDE.bout_phys import BOUTHESELPhys
    from loss_PDE.bout_dump import BOUTHESELInfo
    root = Path(__file__).parents[1] / r"simulatorer/BOUT/BOUT-HESEL/data_alexander_512_512"
    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info)
    condition = nn.MSELoss().to(info.device)
    model = PINN(NN_STRUCTURE)
    model.to(info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    train_loader, val_loader, test_loader = make_dataloader(info, training_config)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    history = HistoryWriter(data_dir / "training_history.jsonl")
    best_model_path = data_dir / "best_val_model.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    return (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            history,
            best_model_path,
            best_val_loss,
            patience_counter)



if __name__ == "__main__":

    training_config = {
        # Optimization
        "lr": 1e-4,
        "epochs": 10,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 10,

        # Logging / status
        "status_frequency": 50,
        "flush_frequency": 50,

        # Dataset split
        "train_ratio": 0.8,
        "val_ratio": 0.1,

        # Loss weights
        "da": 500.0,
        "ic": 1.0,
        "eq": 25000.0, # Brug opløsigligheden af (num_x*num_z)
        "bc": 1.0,

        # DataLoader
        "batch_size": 2**14,
        "shuffle": True, # Bruges kun i val/test loader, da sampler bruges i train loader
        "pin_memory": True,
        "persistent_workers": True,
        "num_workers": 4,
        "prefetch_factor": 1,
        # Resume training from checkpoint
        "resume": False,
    }


    (info,
    phys,
    condition,
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    history,
    best_model_path,
    best_val_loss,
    patience_counter) = init_trainer(training_config)
    
    checkpoint_path = best_model_path.parent / "checkpoint_latest.pt"
    start_epoch = 0
    start_batch_idx = 0

    # Optionally resume from latest checkpoint
    if training_config.get("resume", False) and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=info.device)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
            saved_epoch = int(ckpt.get("epoch", 0))
            saved_batch_idx = int(ckpt.get("batch_idx", -1))
            
            # If batch_idx == -1, we completed the epoch, so start next epoch
            # Otherwise, we interrupted mid-epoch, so resume in same epoch from next batch
            if saved_batch_idx == -1:
                start_epoch = saved_epoch + 1
                start_batch_idx = 0
            else:
                start_epoch = saved_epoch
                start_batch_idx = saved_batch_idx + 1
            
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            # restore history (buffer and filename) so logging can continue
            if "history" in ckpt and ckpt["history"] is not None:
                try:
                    hist_info = ckpt["history"]
                    if hist_info.get("filename"):
                        history.filename = Path(hist_info["filename"])
                    history.buffer = hist_info.get("buffer", [])
                except Exception:
                    pass
            # restore RNG state (convert bytes back to ByteTensor)
            if "torch_rng_state" in ckpt:
                import numpy as np
                rng_state = ckpt["torch_rng_state"]
                if isinstance(rng_state, bytes):
                    rng_state = torch.from_numpy(np.frombuffer(rng_state, dtype=np.uint8).copy())
                torch.set_rng_state(rng_state)
            if "cuda_rng_state_all" in ckpt and torch.cuda.is_available():
                try:
                    import numpy as np
                    cuda_rng_states = ckpt["cuda_rng_state_all"]
                    # Convert list of bytes back to list of ByteTensors
                    if cuda_rng_states and isinstance(cuda_rng_states[0], bytes):
                        cuda_rng_states = [torch.from_numpy(np.frombuffer(s, dtype=np.uint8).copy()) for s in cuda_rng_states]
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                except Exception:
                    pass
            # reset patience counter on resume
            patience_counter = 0
            print(f"Resumed at epoch {start_epoch}, starting from batch {start_batch_idx}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
    
    

    for epoch in range(start_epoch, training_config["epochs"]):
        print(f"Epoch {epoch+1}/{training_config['epochs']}")

        model.train()
        
        # Skip batches if resuming mid-epoch
        batch_iter = train_loader
        if epoch == start_epoch and start_batch_idx > 0:
            batch_iter = itertools.islice(train_loader, start_batch_idx, None)
            print(f"Skipping first {start_batch_idx} batches to resume from batch {start_batch_idx}")
        
        for batch_idx, batch in enumerate(batch_iter, start=start_batch_idx if epoch == start_epoch else 0):
            avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])
            loss_total = get_loss(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, type="train")
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            if batch_idx % training_config["status_frequency"] == 0:
                print(f"Training {batch_idx}/{len(train_loader)}\t Loss: {loss_total.item():.6f}")

            if (batch_idx + 1) % training_config["flush_frequency"] == 0:
                history.flush() 
                save_checkpoint(model, optimizer, epoch, batch_idx, best_val_loss, training_config, checkpoint_path, history, )
        
        model.eval()
        val_total_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])

                loss_total = get_loss(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, type="val")

                val_total_loss += loss_total.detach().cpu().item()
                val_batches += 1

                if batch_idx % training_config["status_frequency"] == 0:
                    print(f"Validation {batch_idx}/{len(val_loader)}\t Loss: {loss_total.item():.6f}")

                if (batch_idx + 1) % training_config["flush_frequency"] == 0:
                    history.flush() 
                    save_checkpoint(model, optimizer, epoch, batch_idx, best_val_loss, training_config, checkpoint_path, history, )

        val_loss = val_total_loss / max(val_batches, 1)
        print(f"Validation loss: {val_loss:.6f}")

        if val_loss < best_val_loss - training_config["early_stopping_min_delta"]:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement in val loss for {patience_counter} epoch(s) "
                f"(best: {best_val_loss:.6f})"
            )

        history.flush()
        save_checkpoint(model, optimizer, epoch, -1, best_val_loss, training_config, checkpoint_path, history)

        if patience_counter >= training_config["early_stopping_patience"]:
            print("Early stopping triggered")
            break

    with torch.no_grad():
        test_total_loss = 0.0
        test_batches = 0
        for batch_idx, batch in enumerate(test_loader):
            avg_z, cord_fys, cord_num, input_data, target = move_batch_to_device((batch), info.device, training_config["pin_memory"])

            loss_total = get_loss(model, info, phys, avg_z, cord_fys, cord_num, input_data, target, condition, training_config, history, type="test")

            test_total_loss += loss_total.detach().cpu().item()
            test_batches += 1

            if batch_idx % training_config["status_frequency"] == 0:
                print(f"Test {batch_idx}/{len(test_loader)}\t Loss: {loss_total.item():.6f}")

        test_loss = test_total_loss / max(test_batches, 1)
        print(f"Test loss: {test_loss:.6f}")
