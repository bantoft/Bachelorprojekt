# Indeholde alle konfigurationer til simuleringns setup

from dataclasses import dataclass

@dataclass
class Config:
    # Datatype
    dtype: type = float

    # Fysisk domæne (Grid)
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 32.0
    y_max: float = 32.0
    nx: int = 128
    ny: int = 128

    # Tidsdomæne
    t0: float = 0.0
    t1: float = 200.0
    save_every: int = 10
    update_every: int = 10
    flush_every_pct: int = 10

    # Data
    simulate: bool = True
    save_path: str = "data/sim_data_2.npz"