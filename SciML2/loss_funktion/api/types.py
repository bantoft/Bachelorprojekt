from dataclasses import dataclass


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