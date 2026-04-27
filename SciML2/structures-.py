from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Mapping
from typing import Any


_STANDARDIZED_FIELD_NAMES = ("lnn", "lnpe", "lnpi", "phi")
_INITIAL_STATE_FIELD_NAMES = _STANDARDIZED_FIELD_NAMES + ("vort",)
_INITIAL_PROFILE_FIELD_NAMES = ("init_n", "init_pe", "init_pi", "sigma_open", "sigma_closed", "sigma_force")
_DUMP_PARAMETER_FIELD_NAMES = (
    "Bt",
    "q",
    "Te0",
    "Ti0",
    "n0",
    "lconn",
    "Rmajor",
    "Rminor",
    "A",
    "Z",
    "Mach",
    "x_lcfs",
    "x_wall",
    "force_time",
    "floor_time",
    "floor_n",
    "floor_pe",
    "floor_pi",
)
_SETTING_PARAMETER_FIELD_NAMES = (
    "z_eff",
    "n_bck",
    "te_bck",
    "ti_bck",
    "d_lcfs",
    "d_wall",
    "d_force",
    "wall_amp",
)
_TRANSPORT_PARAMETER_FIELD_NAMES = (
    "B0",
    "oci",
    "rhoe",
    "rhos",
    "nuei",
    "nuii",
    "nuee",
    "neoclass_correction_factor",
    "lblob",
)
_ACTIVE_SETTING_FIELD_NAMES = (
    "right_handed_coord",
    "interchange_dynamics",
    "parallel_dynamics",
    "perpendicular_dynamics",
    "invert_w_star",
    "force_profiles",
    "floor_profiles",
    "parallel_sheath_damping",
    "parallel_advection_damping",
    "parallel_conduction",
    "parallel_drift_wave",
    "reciprocal_approx",
    "collisional_model",
    "perpend_heat_exchange",
    "perpend_viscous_heating",
    "ti_over_te",
    "diffusion_coeff",
    "qdelta_approx",
    "double_curvature_coeff",
    "h_mode",
    "test_vort_cross_term",
    "ramp_a",
    "ramp_t0",
    "ramp_trans",
    "ramp_peak",
    "not_n_force",
    "not_p_force",
    "power_source",
    "particle_source",
    "parallel_transport",
    "plasma_neutral_interactions",
)


class _MappingDataclass(Mapping[str, Any]):
    def __iter__(self) -> Iterator[str]:
        for field_name in getattr(self, "__dataclass_fields__", {}):
            yield field_name

    def __len__(self) -> int:
        return len(getattr(self, "__dataclass_fields__", {}))

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


@dataclass(frozen=True, slots=True)
class FieldStats(_MappingDataclass):
    mean: float
    variance: float
    std: float


@dataclass(frozen=True, slots=True)
class StandardizationStats(_MappingDataclass):
    lnn: FieldStats
    lnpe: FieldStats
    lnpi: FieldStats
    phi: FieldStats


@dataclass(frozen=True, slots=True)
class BoundaryCondition(_MappingDataclass):
    raw: str
    kind: str
    value: float
    source_section: str
    source_key: str


@dataclass(frozen=True, slots=True)
class BoundaryConditionPair(_MappingDataclass):
    inner: BoundaryCondition
    outer: BoundaryCondition


@dataclass(frozen=True, slots=True)
class BoundaryConditions(_MappingDataclass):
    lnn: BoundaryConditionPair
    lnpe: BoundaryConditionPair
    lnpi: BoundaryConditionPair
    vort: BoundaryConditionPair
    phi: BoundaryConditionPair


@dataclass(frozen=True, slots=True)
class ActiveSettings(_MappingDataclass):
    right_handed_coord: bool | float | int
    interchange_dynamics: bool | float | int
    parallel_dynamics: bool | float | int
    perpendicular_dynamics: bool | float | int
    invert_w_star: bool | float | int
    force_profiles: bool | float | int
    floor_profiles: bool | float | int
    parallel_sheath_damping: bool | float | int
    parallel_advection_damping: bool | float | int
    parallel_conduction: bool | float | int
    parallel_drift_wave: bool | float | int
    reciprocal_approx: bool | float | int
    collisional_model: bool | float | int
    perpend_heat_exchange: bool | float | int
    perpend_viscous_heating: bool | float | int
    ti_over_te: bool | float | int
    diffusion_coeff: bool | float | int
    qdelta_approx: bool | float | int
    double_curvature_coeff: bool | float | int
    h_mode: bool | float | int
    test_vort_cross_term: bool | float | int
    ramp_a: bool | float | int
    ramp_t0: bool | float | int
    ramp_trans: bool | float | int
    ramp_peak: bool | float | int
    not_n_force: bool | float | int
    not_p_force: bool | float | int
    power_source: bool | float | int
    particle_source: bool | float | int
    parallel_transport: bool | float | int
    plasma_neutral_interactions: bool | float | int


@dataclass(frozen=True, slots=True)
class HeselDerivedParameters(_MappingDataclass):
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


@dataclass(frozen=True, slots=True)
class BoutInfoConfig:
    needed_vars: tuple[str, ...] = _STANDARDIZED_FIELD_NAMES + (
        "vort",
        "init_n",
        "init_pe",
        "init_pi",
        "sigma_open",
        "sigma_closed",
        "sigma_force",
        "B",
        "dx",
        "dz",
        "t_array",
    ) + _DUMP_PARAMETER_FIELD_NAMES + ("B0",) + _TRANSPORT_PARAMETER_FIELD_NAMES
    direct_parameter_keys: tuple[str, ...] = _DUMP_PARAMETER_FIELD_NAMES
    setting_parameter_keys: tuple[str, ...] = _SETTING_PARAMETER_FIELD_NAMES
    transport_parameter_keys: tuple[str, ...] = _TRANSPORT_PARAMETER_FIELD_NAMES
    initial_profile_keys: tuple[str, ...] = _INITIAL_PROFILE_FIELD_NAMES
    active_setting_keys: tuple[str, ...] = _ACTIVE_SETTING_FIELD_NAMES
    inner_boundary_targets: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "lnn": ("init_n", "n_inner"),
            "lnpe": ("init_pe", "pe_inner"),
            "lnpi": ("init_pi", "pi_inner"),
        }
    )
    optional_setting_defaults: dict[tuple[str, str], bool | float | int] = field(
        default_factory=lambda: {
            ("hesel", "n_bck"): 0.0,
            ("hesel", "te_bck"): 0.0,
            ("hesel", "ti_bck"): 0.0,
            ("hesel", "double_curvature_coeff"): False,
            ("hesel", "h_mode"): False,
            ("hesel", "invert_w_star"): False,
            ("hesel", "not_n_force"): False,
            ("hesel", "not_p_force"): False,
            ("hesel", "parallel_transport"): False,
            ("hesel", "particle_source"): False,
            ("hesel", "power_source"): False,
            ("hesel", "ramp_a"): 2.0,
            ("hesel", "ramp_peak"): 50000.0,
            ("hesel", "ramp_t0"): 0.0,
            ("hesel", "ramp_trans"): 5000.0,
            ("hesel", "test_vort_cross_term"): False,
        }
    )
    standardized_fields: tuple[str, ...] = _STANDARDIZED_FIELD_NAMES
    initial_state_keys: tuple[str, ...] = _INITIAL_STATE_FIELD_NAMES


BOUT_INFO_CONFIG = BoutInfoConfig()
