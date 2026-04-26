from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import xarray as xr
from torch import Tensor

from .api.bout_struct import BoundaryCondition, HeselDerivedParameters
from .api.read_bout import BOUNDARY_PATTERN, parse_literal, safe_log

DUMP_ALIASES = {
    "Bt": "bt",
    "Te0": "te0",
    "Ti0": "ti0",
    "Rmajor": "rmajor",
    "Rminor": "rminor",
    "Mach": "mach",
    "B0": "b0",
    "A": "a",
    "Z": "z",
}

NEEDED_VARS = (
    "lnn",
    "lnpe",
    "lnpi",
    "vort",
    "phi",
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
    "force_time",
    "floor_time",
    "floor_n",
    "floor_pe",
    "floor_pi",
    "bt",
    "Bt",
    "rmajor",
    "Rmajor",
    "rminor",
    "Rminor",
    "q",
    "te0",
    "Te0",
    "ti0",
    "Ti0",
    "n0",
    "b0",
    "B0",
    "a",
    "A",
    "z",
    "Z",
    "lconn",
    "lblob",
    "mach",
    "Mach",
    "neoclass_correction_factor",
    "x_lcfs",
    "x_wall",
    "oci",
    "nuii",
    "nuei",
    "nuee",
    "rhoe",
    "rhos",
)

DIRECT_PARAMETER_KEYS = (
    "bt",
    "q",
    "te0",
    "ti0",
    "n0",
    "lconn",
    "rmajor",
    "rminor",
    "a",
    "z",
    "mach",
    "x_lcfs",
    "x_wall",
    "force_time",
    "floor_time",
    "floor_n",
    "floor_pe",
    "floor_pi",
)

SETTING_PARAMETER_KEYS = (
    "z_eff",
    "n_bck",
    "te_bck",
    "ti_bck",
    "d_lcfs",
    "d_wall",
    "d_force",
    "wall_amp",
)

TRANSPORT_PARAMETER_KEYS = (
    "b0",
    "oci",
    "rhoe",
    "rhos",
    "nuei",
    "nuii",
    "nuee",
    "neoclass_correction_factor",
    "lblob",
)

INITIAL_PROFILE_KEYS = (
    "init_n",
    "init_pe",
    "init_pi",
    "sigma_open",
    "sigma_closed",
    "sigma_force",
)

INITIAL_STATE_KEYS = ("lnn", "lnpe", "lnpi", "phi", "vort")

ACTIVE_SETTING_KEYS = (
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

INNER_BOUNDARY_TARGETS = {
    "lnn": ("init_n", "n_inner"),
    "lnpe": ("init_pe", "pe_inner"),
    "lnpi": ("init_pi", "pi_inner"),
}

OPTIONAL_SETTING_DEFAULTS: dict[tuple[str, str], bool | float | int] = {
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

STANDARDIZED_FIELDS = ("lnn", "lnpe", "lnpi", "phi")


class BOUTHESELInfo:
    def __init__(self, data_folder_path: Path):
        self.folder_path = data_folder_path

        self.settings = self._read_settings()
        self.data = self._load_data()
        self.standardization_stats = self._build_standardization_stats()
        self.parameters = self._build_parameters()
        self.boundary_conditions = self._build_boundary_conditions()
        self.active_settings = self._build_active_settings()

    def _read_settings(self) -> dict[str, dict[str, str]]:
        settings_path = self.folder_path / "BOUT.settings"
        current_section = "root"
        settings: dict[str, dict[str, str]] = {current_section: {}}

        for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                settings.setdefault(current_section, {})
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            settings[current_section][key.strip()] = value.strip()

        return settings

    def _load_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {}

        for data_path in sorted(self.folder_path.glob("BOUT.dmp.*.nc")):
            with xr.open_dataset(data_path, engine="netcdf4") as dataset:
                for name in NEEDED_VARS:
                    if name not in dataset:
                        continue

                    variable = dataset[name]
                    array = variable.squeeze("y", drop=True) if "y" in variable.dims else variable
                    canonical_name = DUMP_ALIASES.get(name, name)
                    values = torch.as_tensor(array.values)
                    self._merge_dump_values(
                        data=data,
                        name=canonical_name,
                        dims=tuple(array.dims),
                        values=values,
                    )

        return data

    def _merge_dump_values(
        self,
        *,
        data: dict[str, Any],
        name: str,
        dims: tuple[str, ...],
        values: Tensor,
        overlap: int = 2,
    ) -> None:
        if name == "t_array":
            data.setdefault(name, values)
            return

        previous = data.get(name)
        if previous is None:
            data[name] = values
            return

        if "x" not in dims:
            return

        axis = dims.index("x")
        slicer = [slice(None)] * values.ndim
        slicer[axis] = slice(min(overlap, int(values.shape[axis])), None)
        data[name] = torch.cat((previous, values[tuple(slicer)]), dim=axis)

    def _build_standardization_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for name in STANDARDIZED_FIELDS:
            values = torch.as_tensor(self.data[name], dtype=torch.float32)
            mean = float(values.mean().item())
            variance = float(values.var(unbiased=False).item())
            std = math.sqrt(max(variance, 1e-12))
            stats[name] = {"mean": mean, "variance": variance, "std": std}
        return stats

    def standardize_field(self, name: str, values: Any) -> Tensor:
        tensor = torch.as_tensor(values)
        if name not in self.standardization_stats:
            return tensor

        stats = self.standardization_stats[name]
        mean = torch.as_tensor(stats["mean"], device=tensor.device, dtype=tensor.dtype)
        std = torch.as_tensor(stats["std"], device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std

    def destandardize_field(self, name: str, values: Any) -> Tensor:
        tensor = torch.as_tensor(values)
        if name not in self.standardization_stats:
            return tensor

        stats = self.standardization_stats[name]
        mean = torch.as_tensor(stats["mean"], device=tensor.device, dtype=tensor.dtype)
        std = torch.as_tensor(stats["std"], device=tensor.device, dtype=tensor.dtype)
        return tensor * std + mean

    def standardize_state(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: self.standardize_field(name, values) if name in self.standardization_stats else values
            for name, values in state.items()
        }

    def destandardize_state(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: self.destandardize_field(name, values) if name in self.standardization_stats else values
            for name, values in state.items()
        }

    @property
    def num_time_points(self) -> int:
        return int(torch.as_tensor(self.data["t_array"]).shape[0])

    @property
    def normalized_time_step(self) -> float:
        return 1.0 / max(self.num_time_points - 1, 1)

    @property
    def max_stepper_input_time(self) -> float:
        if self.num_time_points <= 1:
            return 0.0
        return (self.num_time_points - 2) / max(self.num_time_points - 1, 1)

    def stepper_target_time(self, t: Tensor) -> Tensor:
        return torch.clamp(t + self.normalized_time_step, max=1.0)

    def _setting_value(self, section: str, key: str) -> Any:
        section_settings = self.settings.get(section, {})
        if key not in section_settings:
            default = OPTIONAL_SETTING_DEFAULTS.get((section, key))
            if default is not None:
                return default
            raise KeyError(key)

        expr = section_settings[key].strip()
        literal = parse_literal(expr)
        if literal is None:
            raise ValueError(
                f"Expected literal BOUT setting for {section}:{key}, got {expr!r}"
            )
        return literal

    def _scalar(self, section: str, key: str) -> float:
        value = self._setting_value(section, key)
        if isinstance(value, bool):
            return float(value)
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    f"Expected scalar for {section}:{key}, got tensor with shape {value.shape}"
                )
            return float(value.detach().cpu().item())
        return float(value)

    def _data_scalar(self, name: str) -> float:
        value = self.data[name]
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return float(value.to(dtype=torch.float64).mean().item())
        return float(value)

    def _scalar_from_dump_or_setting(
        self,
        dump_key: str,
        section: str,
        setting_key: str,
    ) -> float:
        if dump_key in self.data:
            return self._data_scalar(dump_key)
        return self._scalar(section, setting_key)

    def _pick(self, mapping: dict[str, float], *keys: str) -> tuple[float, ...]:
        return tuple(mapping[key] for key in keys)

    def _domain_extents(self) -> tuple[float, float, float]:
        dx = torch.as_tensor(self.data["dx"], dtype=torch.float64).flatten()
        total_x = float(dx[:-1].sum().item()) if dx.numel() > 1 else float(dx.sum().item())
        total_z = self._data_scalar("dz") * max(int(self.data["lnn"].shape[-1]) - 1, 1)
        total_t = float(torch.as_tensor(self.data["t_array"], dtype=torch.float64)[-1].item())
        return total_x, total_z, total_t

    def _build_parameters(self) -> HeselDerivedParameters:
        e = 1.60e-19
        epso = 8.85e-12
        me = 9.1093816e-31
        mp = 1.67262158e-27
        pi_const = math.pi

        values = {
            name: self._scalar_from_dump_or_setting(name, "hesel", name)
            for name in DIRECT_PARAMETER_KEYS
        }
        values.update({name: self._scalar("hesel", name) for name in SETTING_PARAMETER_KEYS})

        total_x, total_z, total_t = self._domain_extents()

        a, z, mach = self._pick(values, "a", "z", "mach")
        x_lcfs, x_wall = self._pick(values, "x_lcfs", "x_wall")
        bt, q, te0, ti0, n0 = self._pick(values, "bt", "q", "te0", "ti0", "n0")
        lconn, rmajor, rminor = self._pick(values, "lconn", "rmajor", "rminor")
        force_time, floor_time = self._pick(values, "force_time", "floor_time")
        floor_n, floor_pe, floor_pi = self._pick(
            values, "floor_n", "floor_pe", "floor_pi"
        )
        z_eff, n_bck, te_bck, ti_bck = self._pick(
            values, "z_eff", "n_bck", "te_bck", "ti_bck"
        )
        d_lcfs, d_wall, d_force, wall_amp = self._pick(
            values, "d_lcfs", "d_wall", "d_force", "wall_amp"
        )

        transport = {name: self._data_scalar(name) for name in TRANSPORT_PARAMETER_KEYS}
        b0 = transport["b0"]
        oci = transport["oci"]
        rhoe = transport["rhoe"]
        rhos = transport["rhos"]
        nuei = transport["nuei"]
        nuii = transport["nuii"]
        nuee = transport["nuee"]
        neoclass_correction_factor = transport["neoclass_correction_factor"]
        lblob = transport["lblob"]

        mi = a * mp
        cs = math.sqrt(e * te0 / mi)
        rhoi = math.sqrt(e * ti0 / mi) / oci
        debye = math.sqrt(epso * e * te0 / (e * e * n0))
        collog = math.log(12.0 * pi_const * n0 * debye**3 / z)

        bohm_potential = math.log(math.sqrt(mi / (2.0 * pi_const * me)))
        de_phys = neoclass_correction_factor * rhoe * rhoe * nuei
        di_phys = neoclass_correction_factor * rhoi * rhoi * nuii
        norm_de = de_phys / (rhos * rhos * oci)
        norm_di = di_phys / (rhos * rhos * oci)
        chi_e_par = 3.16 * n0 * e * te0 / (me * nuei)
        chi_i_par = 3.9 * n0 * e * ti0 / (mi * nuii)
        taun = lblob / (2.0 * mach * cs)
        taudw = lblob * lblob * me * nuei / (2.0 * e * te0)
        taushe = lconn * lconn * n0 / chi_e_par
        taushi = lconn * lconn * n0 / chi_i_par

        return HeselDerivedParameters(
            e=e,
            epso=epso,
            me=me,
            mp=mp,
            pi=pi_const,
            bt=bt,
            q=q,
            te0=te0,
            ti0=ti0,
            n0=n0,
            lconn=lconn,
            rmajor=rmajor,
            rminor=rminor,
            a=a,
            z=z,
            mach=mach,
            z_eff=z_eff,
            x_lcfs=x_lcfs,
            x_wall=x_wall,
            force_time=force_time,
            floor_time=floor_time,
            floor_n=floor_n,
            floor_pe=floor_pe,
            floor_pi=floor_pi,
            n_bck=n_bck,
            te_bck=te_bck,
            ti_bck=ti_bck,
            d_lcfs=d_lcfs,
            d_wall=d_wall,
            d_force=d_force,
            wall_amp=wall_amp,
            total_x=total_x,
            total_z=total_z,
            total_t=total_t,
            b0=b0,
            mi=mi,
            cs=cs,
            oci=oci,
            rhoe=rhoe,
            rhoi=rhoi,
            rhos=rhos,
            collog=collog,
            nuei=nuei,
            nuii=nuii,
            nuee=nuee,
            neoclass_correction_factor=neoclass_correction_factor,
            lblob=lblob,
            bohm_potential=bohm_potential,
            norm_de=norm_de,
            norm_di=norm_di,
            norm_eta=3.0 / 10.0 * norm_di,
            norm_taun=taun * oci,
            norm_taudw=taudw * oci,
            norm_taushe=taushe * oci,
            norm_taushi=taushi * oci,
            norm_lc=lconn / rhos,
            norm_lb=lblob / rhos,
        )

    def _build_boundary_conditions(self) -> dict[str, dict[str, BoundaryCondition]]:
        inner_values = {
            field: safe_log(self._scalar(section, key))
            for field, (section, key) in INNER_BOUNDARY_TARGETS.items()
        }
        inner_values["vort"] = 0.0

        boundary_conditions: dict[str, dict[str, BoundaryCondition]] = {
            field: {
                side: BoundaryCondition(
                    raw=self.settings[field][setting_key].strip(),
                    kind=self._boundary_kind(self.settings[field][setting_key].strip()),
                    value=inner_values[field] if side == "inner" else 0.0,
                    source_section=field,
                    source_key=setting_key,
                )
                for side, setting_key in (
                    ("inner", "bndry_xin"),
                    ("outer", "bndry_xout"),
                )
            }
            for field in ("lnn", "lnpe", "lnpi", "vort")
        }
        boundary_conditions["phi"] = {
            "inner": BoundaryCondition(
                raw=f"laplace:inner_boundary_flags={self.settings['laplace']['inner_boundary_flags']}",
                kind="dirichlet",
                value=0.0,
                source_section="laplace",
                source_key="inner_boundary_flags",
            ),
            "outer": BoundaryCondition(
                raw=f"laplace:outer_boundary_flags={self.settings['laplace']['outer_boundary_flags']}",
                kind="neumann",
                value=0.0,
                source_section="laplace",
                source_key="outer_boundary_flags",
            ),
        }
        return boundary_conditions

    def _boundary_kind(self, raw: str) -> str:
        match = BOUNDARY_PATTERN.match(raw)
        if match is None:
            raise ValueError(f"Could not parse boundary condition: {raw}")
        return match.group("kind")

    def _build_active_settings(self) -> dict[str, Any]:
        return {
            key: self._setting_value("hesel", key)
            for key in ACTIVE_SETTING_KEYS
        }

    def initial_profiles(self, x: Tensor, z: Tensor) -> dict[str, Tensor]:
        del z
        return {name: self._interp_dump_1d(self.data[name], x) for name in INITIAL_PROFILE_KEYS}

    def magnetic_field(self, x: Tensor) -> Tensor:
        return self._interp_dump_1d(self.data["B"], x)

    def _interp_dump_1d(self, values: Any, x: Tensor) -> Tensor:
        field = torch.as_tensor(values, device=x.device, dtype=x.dtype).flatten()
        if field.ndim != 1:
            raise ValueError(
                f"Expected 1D dump field for interpolation, got shape {tuple(field.shape)}"
            )

        x_pos = torch.clamp(x.squeeze(-1), 0.0, 1.0) * (field.shape[0] - 1)
        x0 = x_pos.floor().long().clamp(0, field.shape[0] - 1)
        x1 = (x0 + 1).clamp(0, field.shape[0] - 1)
        wx = (x_pos - x0.to(dtype=x.dtype)).unsqueeze(-1)
        v0 = field[x0].unsqueeze(-1)
        v1 = field[x1].unsqueeze(-1)
        return (1.0 - wx) * v0 + wx * v1

    def _interp_dump_2d(self, values: Any, x: Tensor, z: Tensor) -> Tensor:
        field = torch.as_tensor(values, device=x.device, dtype=x.dtype)
        if field.ndim != 2:
            raise ValueError(
                f"Expected 2D dump field for interpolation, got shape {tuple(field.shape)}"
            )

        x_pos = torch.clamp(x.squeeze(-1), 0.0, 1.0) * (field.shape[0] - 1)
        z_pos = torch.clamp(z.squeeze(-1), 0.0, 1.0) * (field.shape[1] - 1)

        x0 = x_pos.floor().long().clamp(0, field.shape[0] - 1)
        z0 = z_pos.floor().long().clamp(0, field.shape[1] - 1)
        x1 = (x0 + 1).clamp(0, field.shape[0] - 1)
        z1 = (z0 + 1).clamp(0, field.shape[1] - 1)

        wx = (x_pos - x0.to(dtype=x.dtype)).unsqueeze(-1)
        wz = (z_pos - z0.to(dtype=x.dtype)).unsqueeze(-1)

        v00 = field[x0, z0].unsqueeze(-1)
        v10 = field[x1, z0].unsqueeze(-1)
        v01 = field[x0, z1].unsqueeze(-1)
        v11 = field[x1, z1].unsqueeze(-1)

        x_interp_low = (1.0 - wx) * v00 + wx * v10
        x_interp_high = (1.0 - wx) * v01 + wx * v11
        return (1.0 - wz) * x_interp_low + wz * x_interp_high

    def state_targets(
        self,
        x: Tensor,
        z: Tensor,
        time_index: int = 0,
        standardized: bool = False,
    ) -> dict[str, Tensor]:
        time_idx = max(0, min(int(time_index), self.num_time_points - 1))
        targets = {
            name: self._interp_dump_2d(self.data[name][time_idx, :, :], x, z)
            for name in INITIAL_STATE_KEYS
        }
        if not standardized:
            return targets
        return self.standardize_state(targets)

    def initial_state_targets(
        self,
        x: Tensor,
        z: Tensor,
        standardized: bool = False,
    ) -> dict[str, Tensor]:
        return self.state_targets(x=x, z=z, time_index=0, standardized=standardized)
