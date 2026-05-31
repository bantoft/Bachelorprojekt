import sys
import neuralop as nop

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from SciML.PINO.utils.data_loader import make_dataloaders
from SciML.PINO.utils.training_helpers import HistoryBuffer, init_trainer

from hesel_scraper.bout_dump import BOUTHESELInfo
from hesel_scraper.bout_phys import BOUTHESELPhys


root = ROOT_DIR / r"sim_data/data_25_512_Alexander"

info = BOUTHESELInfo(root)
phys = BOUTHESELPhys(info)
train_loader, val_loader, test_loader = make_dataloaders(info, batch_size=4)


fno = nop.models.FNO(
    n_modes=(128, 128),
    in_channels=8,      # 4 fields + t + x + z_1 + z_2 #fourier embedder z
    out_channels=4,
    hidden_channels=64,
    positional_embedding= None, # vigtigt: du embedder selv coords
    domain_padding=[0.1, 0.0], # padding kun i x, ikke z
)


if __name__ == "__main__":

    training_config = {
        # !! Paths skal være referet fra projekt root folder !!
        # Folder med dump filer og settings fil
        "root": r"sim_data/data_15_512_Alexander_",
        # Output folder for checkpoints, history og best model
        # Kald folderen for: history<...> så det ikke flyttes til repo
        "output_dir": r"SciML/PINO/history_tester",
        
        # Optimization
        "lr": 1e-5,
        "epochs": 10,

        # Dataloaer
        "batch_size": 4,
        "train_split": 0.8,
        "val_split": 0.1,

    }

    (info,
    phys,
    condition,
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader) = init_trainer(training_config)


    data_dir = (ROOT_DIR / training_config["out_folder"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryBuffer(data_dir / "training_history.jsonl")
    split = "training"

    for epoch in range(training_config["epochs"]):

        if split == "training":
            model.train()
            for batch_idx, (u, t, y) in enumerate(train_loader):
                u, t, y = u.to(info.device), t.to(info.device), y.to(info.device)
                u_pred = model(u, t)



