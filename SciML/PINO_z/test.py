import sys
import torch
import neuralop as nop

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


from SciML.PINO_z.utils.training_helpers import cleanup_phase, iterater, init_experinment



train_cfg = {
    "root": ROOT_DIR / r"sim_data/data_25_512_Alexander",
    "data_dir": ROOT_DIR / r"SciML/PINO_z/expertiments",
    "m": 3,
    "train_split": 0.8,
    "val_split": 0.1,
    "batch_size": 6,
    "shuffle": True,
    "num_workers": 2,
    "prefetch_factor": 1,
    "pin_memory": True,
    "lr": 1e-4,
    "epochs": 2,
}

fno = nop.models.FNO(n_modes=(160, 160),
                        in_channels=8,
                        out_channels=4,
                        hidden_channels=20,
                        positional_embedding=None)


model, train_loader, val_loader, test_loader, condition, optimizer, history, phys, info = init_experinment(train_cfg, fno)




for epoch in range(train_cfg["epochs"]):
    print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")
    iterater(model, train_loader, condition, optimizer, history, phys, info, split="training", status_frequency=10, flush_frequency=50)
    history.flush()
    cleanup_phase(model, optimizer)
    iterater(model, val_loader, condition, optimizer, history, phys, info, split="validation", status_frequency=10, flush_frequency=50)
    history.flush()
    cleanup_phase(model, optimizer)

iterater(model, test_loader, condition, optimizer, history, phys, info, split="test", status_frequency=10, flush_frequency=50)
history.flush()
cleanup_phase(model, optimizer)
