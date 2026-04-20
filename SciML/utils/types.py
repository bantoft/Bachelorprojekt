from dataclasses import dataclass

from torch import nn


@dataclass
class BoundaryCondition:
    raw: str
    kind: str
    value: float | None
    source_section: str
    source_key: str


@dataclass
class HeselDerivedParameters:
    e: float
    epso: float
    me: float
    mp: float
    pi: float
    bt: float
    q: float
    te0: float
    ti0: float
    n0: float
    lconn: float
    rmajor: float
    rminor: float
    a: float
    z: float
    mach: float
    z_eff: float
    x_lcfs: float
    x_wall: float
    force_time: float
    floor_time: float
    floor_n: float
    floor_pe: float
    floor_pi: float
    n_bck: float
    te_bck: float
    ti_bck: float
    d_lcfs: float
    d_wall: float
    d_force: float
    wall_amp: float
    total_x: float
    total_z: float
    total_t: float
    b0: float
    mi: float
    cs: float
    oci: float
    rhoe: float
    rhoi: float
    rhos: float
    collog: float
    nuei: float
    nuii: float
    nuee: float
    neoclass_correction_factor: float
    lblob: float
    bohm_potential: float
    norm_de: float
    norm_di: float
    norm_eta: float
    norm_taun: float
    norm_taudw: float
    norm_taushe: float
    norm_taushi: float
    norm_lc: float
    norm_lb: float

@dataclass
class TrainConfig:
    # hyperparameters
    epochs: int = 2
    batch_size: int = 2048
    lr: float = 1e-3
    seed: int = 42
    # logging and saving
    print_every: int = 1
    save_data: int = 0
    batch_log_path: str = "SciML/data/training_batch_log.parquet"
    # weight distribution
    w_data: float = 1.0
    w_eq: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0


nn_mapping = {
    "activation_functions": [nn.ELU(), nn.GELU(), nn.LogSigmoid(), nn.Sigmoid(), nn.SiLU(), nn.Tanh(), nn.Tanhshrink()],
    "layer_sizes": [16, 32, 64, 128, 256, 512],
}