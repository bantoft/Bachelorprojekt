from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path





@dataclass
class TrainConfig:
    # hyperparameters
    epochs: int = 2
    batch_size: int = 2048
    lr: float = 1e-3
    incl_data: bool = True
    # logging and saving
    root: Path = Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL",
    # weight distribution
    early_stopping_patience=3,
    early_stopping_min_delta=1e-2,
    save_model=True, # Skal altid være true gennem gå for fallback logik

@dataclass
class WeightDistribution:
    w_data: float = 1.0
    w_eq: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0

@dataclass
class WandBConfig:
    use_wandb: bool = True # Skal altid være true gennem gå for fallback logik
    wandb_project: str = "Bachelor_projekt" # Altid dette gennem gå for fallback logik
    wandb_entity: str | None = None # Undersøg hvad dette er 
    wandb_mode: str | None = None # Undersøg hvad dette er
    save_model: bool = True # Skal altid være true gennem gå for fallback logik
    model_path: Path | None = None # Skal altid samme sted med timestamp gennem gå for fallback logik

