from __future__ import annotations

import math
from sympy import re
import re
import xarray as xr
import torch
import torch.nn.functional as F

from torch import Tensor
from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping, Callable
from utils.custom_types import BoundaryCondition, HeselDerivedParameters

from .api.read_bout import read_bout_inp

def _parse_literal(value: str):
    text = value.strip()
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return None

def _safe_sqrt(x: float | Tensor) -> float | Tensor:
    if torch.is_tensor(x):
        return torch.sqrt(torch.clamp(x, min=1e-12))
    return math.sqrt(max(float(x), 1e-12))

def _safe_log(x: float | Tensor) -> float | Tensor:
    if torch.is_tensor(x):
        return torch.log(torch.clamp(x, min=1e-12))
    return math.log(max(float(x), 1e-12))

def _torch_or_math_unary(
    tensor_fn: Callable[[Tensor], Tensor],
    math_fn: Callable[[float], float],
) -> Callable[[float | Tensor], float | Tensor]:
    def wrapper(x: float | Tensor) -> float | Tensor:
        if torch.is_tensor(x):
            return tensor_fn(x)
        return math_fn(float(x))

    return wrapper

def _mixmode_seed(seed: float) -> float:
    seed = abs(float(seed))
    niter = 11 + (23 + round(seed)) % 79
    a = 0.01
    b = 1.23456789
    x = (a + math.fmod(seed, b)) / (b + 2.0 * a)
    for _ in range(niter):
        x = 3.99 * x * (1.0 - x)
    return x

def _mixmode(arg: float | Tensor, seed: float = 0.5) -> float | Tensor:
    if torch.is_tensor(arg):
        result = torch.zeros_like(arg)
    else:
        result = 0.0

    for i in range(14):
        phase = math.pi * (2.0 * _mixmode_seed(seed + i) - 1.0)
        weight = 1.0 / (1.0 + abs(i - 4)) ** 2
        result = result + weight * torch.cos(i * arg + phase) if torch.is_tensor(arg) else result + weight * math.cos(i * arg + phase)
    return result

def _as_tensor_like(value: float | Tensor, like: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


DEFAULT_BOUT_HESEL_ROOT = (Path(__file__).resolve().parents[2] / "simulatorer" / "BOUT" / "BOUT-HESEL")
REF_PATTERN = re.compile(r"\b([A-Za-z_]\w*):([A-Za-z_]\w*)\b")
BOUNDARY_PATTERN = re.compile(r"^(?P<kind>[A-Za-z_]\w*)(?:\((?P<expr>.*)\))?$")
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\b")
    


class BOUTHESELInfo:
    def __init__(self, root: str | Path = DEFAULT_BOUT_HESEL_ROOT):
        self.root = Path(root).resolve()
        self.dump_path = self.root / "data" / "BOUT.dmp.0.nc"
        self.settings_path = self.root / "data" / "BOUT.settings"
        self.source_paths = {
            "hesel_cxx": self.root / "hesel.cxx",
            "hesel_hxx": self.root / "hesel.hxx",
            "hesel_parameters_cxx": self.root / "HeselParameters" / "HeselParameters.cxx",
            "hesel_parameters_hxx": self.root / "HeselParameters" / "HeselParameters.hxx",
        }
        self.cfg = read_bout_inp(self.settings_path)
        self.dump_data = self._load_dump()
        self.parameters = self._build_parameters()
        self.boundary_conditions = self._build_boundary_conditions()
        self.active_settings = self._build_active_settings()

    def _load_dump(self) -> dict[str, Any]:
        needed_vars = [
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
            "t_array",
        ]
        with xr.open_dataset(self.dump_path, engine="netcdf4") as ds:
            data: dict[str, Any] = {}
            for name in needed_vars:
                if name not in ds:
                    continue
                arr = ds[name]
                data[name] = (arr.squeeze("y", drop=True) if "y" in arr.dims else arr).values
            return data

    def _resolve_key(
        self,
        section: str,
        key: str,
        variables: Mapping[str, Any] | None = None,
        stack: tuple[tuple[str, str], ...] = (),
    ) -> Any:
        if section not in self.cfg or key not in self.cfg[section]:
            raise KeyError(f"Unknown BOUT setting {section}:{key}")
        node = (section, key)
        if node in stack:
            chain = " -> ".join(f"{s}:{k}" for s, k in stack + (node,))
            raise ValueError(f"Cyclic BOUT expression detected: {chain}")
        return self._eval_expression(
            expr=self.cfg[section][key],
            current_section=section,
            variables=variables,
            stack=stack + (node,),
        )

    def _eval_expression(
        self,
        expr: str,
        current_section: str,
        variables: Mapping[str, Any] | None = None,
        stack: tuple[tuple[str, str], ...] = (),
    ) -> Any:
        variables = dict(variables or {})
        expr = expr.strip().replace("^", "**")
        literal = _parse_literal(expr)
        if literal is not None:
            return literal

        def _ref(section_name: str, key_name: str) -> Any:
            return self._resolve_key(section_name, key_name, variables=variables, stack=stack)

        rewritten = REF_PATTERN.sub(r'__ref("\1", "\2")', expr)
        tokens = {
            token
            for token in IDENTIFIER_PATTERN.findall(REF_PATTERN.sub(" ", expr))
            if token
            not in {
                "__ref",
                "sqrt",
                "tanh",
                "exp",
                "log",
                "sin",
                "cos",
                "abs",
                "mixmode",
                "pi",
                "true",
                "false",
            }
            and token not in variables
        }

        local_symbols: dict[str, Any] = {}
        for section_name in (current_section, "root"):
            if section_name not in self.cfg:
                continue
            for key_name in tokens:
                if key_name not in self.cfg[section_name] or (section_name, key_name) in stack or key_name in local_symbols:
                    continue
                try:
                    local_symbols[key_name] = self._resolve_key(section_name, key_name, variables=variables, stack=stack)
                except Exception:
                    continue

        return eval(
            rewritten,
            {"__builtins__": {}},
            {
                "__ref": _ref,
                "sqrt": _safe_sqrt,
                "tanh": _torch_or_math_unary(torch.tanh, math.tanh),
                "exp": _torch_or_math_unary(torch.exp, math.exp),
                "log": _safe_log,
                "sin": _torch_or_math_unary(torch.sin, math.sin),
                "cos": _torch_or_math_unary(torch.cos, math.cos),
                "abs": _torch_or_math_unary(torch.abs, abs),
                "mixmode": _mixmode,
                "pi": math.pi,
                **local_symbols,
                **variables,
            },
        )

    def _scalar(self, section: str, key: str) -> float:
        value = self._resolve_key(section, key)
        if isinstance(value, bool):
            return float(value)
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar for {section}:{key}, got tensor with shape {value.shape}")
            return float(value.detach().cpu().item())
        return float(value)

    def _build_parameters(self) -> HeselDerivedParameters:
        e, epso, me, mp, pi_const = 1.60e-19, 8.85e-12, 9.1093816e-31, 1.67262158e-27, math.pi
        bt, q, te0, ti0, n0, lconn, rmajor, rminor = (self._scalar("hesel", key) for key in ("bt", "q", "te0", "ti0", "n0", "lconn", "rmajor", "rminor"))
        a, z, mach, z_eff, x_lcfs, x_wall, force_time, floor_time = (self._scalar("hesel", key) for key in ("a", "z", "mach", "z_eff", "x_lcfs", "x_wall", "force_time", "floor_time"))
        floor_n, floor_pe, floor_pi, n_bck, te_bck, ti_bck, d_lcfs, d_wall, d_force, wall_amp = (self._scalar("hesel", key) for key in ("floor_n", "floor_pe", "floor_pi", "n_bck", "te_bck", "ti_bck", "d_lcfs", "d_wall", "d_force", "wall_amp"))
        total_x = float(self._eval_expression("mesh:xl", current_section="root", variables={"x": 1.0}))
        total_z = float(self._eval_expression("mesh:zl", current_section="root", variables={"z": 1.0}))
        total_t = self._scalar("root", "t_end")

        b0 = bt * rmajor / (rmajor + rminor)
        mi = a * mp
        cs = math.sqrt(e * te0 / mi)
        oci = e * z * b0 / mi
        rhoe = math.sqrt(e * te0 / me) / (e * b0 / me)
        rhoi = math.sqrt(e * ti0 / mi) / oci
        rhos = cs / oci
        debye = math.sqrt(epso * e * te0 / (e * e * n0))
        collog = math.log(12.0 * pi_const * n0 * debye**3 / z)
        nuei = math.sqrt(2.0) * n0 * z * z * e**4 * collog / (12.0 * math.sqrt(pi_const**3) * math.sqrt(me) * math.sqrt((e * te0) ** 3) * epso * epso)
        nuii = n0 * z**4 * e**4 * collog / (12.0 * math.sqrt(pi_const**3) * epso * epso * math.sqrt(mi) * math.sqrt((e * ti0) ** 3))
        nuee = nuii / (z**4) * math.sqrt(mi / me) * math.sqrt((ti0 / te0) ** 3)
        neoclass_override = self._scalar("hesel", "neoclass_correction_factor")
        neoclass_correction_factor = 1.0 + neoclass_override if neoclass_override >= 0.0 else 1.0 + rmajor / rminor * q * q
        lblob_setting = self._scalar("hesel", "lblob")
        lblob = lblob_setting if lblob_setting > 0.0 else q * rmajor
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
            e=e, epso=epso, me=me, mp=mp, pi=pi_const, bt=bt, q=q, te0=te0, ti0=ti0, n0=n0, lconn=lconn, rmajor=rmajor, rminor=rminor, a=a, z=z, mach=mach, z_eff=z_eff,
            x_lcfs=x_lcfs, x_wall=x_wall, force_time=force_time, floor_time=floor_time, floor_n=floor_n, floor_pe=floor_pe, floor_pi=floor_pi, n_bck=n_bck, te_bck=te_bck, ti_bck=ti_bck,
            d_lcfs=d_lcfs, d_wall=d_wall, d_force=d_force, wall_amp=wall_amp, total_x=total_x, total_z=total_z, total_t=total_t, b0=b0, mi=mi, cs=cs, oci=oci, rhoe=rhoe, rhoi=rhoi,
            rhos=rhos, collog=collog, nuei=nuei, nuii=nuii, nuee=nuee, neoclass_correction_factor=neoclass_correction_factor, lblob=lblob, bohm_potential=bohm_potential, norm_de=norm_de,
            norm_di=norm_di, norm_eta=3.0 / 10.0 * norm_di, norm_taun=taun * oci, norm_taudw=taudw * oci, norm_taushe=taushe * oci, norm_taushi=taushi * oci, norm_lc=lconn / rhos, norm_lb=lblob / rhos,
        )

    def _parse_boundary(self, section: str, key: str) -> BoundaryCondition:
        raw = self.cfg[section][key].strip()
        match = BOUNDARY_PATTERN.match(raw)
        if match is None:
            return BoundaryCondition(raw=raw, kind=raw, value=None, source_section=section, source_key=key)
        expr = match.group("expr")
        value = None
        if expr:
            try:
                resolved = self._eval_expression(expr, current_section=section)
                value = float(resolved.detach().cpu().item()) if torch.is_tensor(resolved) else float(resolved) if isinstance(resolved, (int, float, bool)) else None
            except Exception:
                value = None
        return BoundaryCondition(raw=raw, kind=match.group("kind"), value=value, source_section=section, source_key=key)

    def _build_boundary_conditions(self) -> dict[str, dict[str, BoundaryCondition]]:
        bc: dict[str, dict[str, BoundaryCondition]] = {}
        for field in ("lnn", "lnpe", "lnpi", "vort"):
            if field not in self.cfg:
                continue
            bc[field] = {
                side: self._parse_boundary(field, key)
                for side, key in (("inner", "bndry_xin"), ("outer", "bndry_xout"))
                if key in self.cfg[field]
            }
        bc["phi"] = {
            "inner": BoundaryCondition(raw=f"laplace:inner_boundary_flags={self.cfg.get('laplace', {}).get('inner_boundary_flags', '0')}", kind="dirichlet", value=0.0, source_section="laplace", source_key="inner_boundary_flags"),
            "outer": BoundaryCondition(raw=f"laplace:outer_boundary_flags={self.cfg.get('laplace', {}).get('outer_boundary_flags', '0')}", kind="neumann", value=0.0, source_section="laplace", source_key="outer_boundary_flags"),
        }
        return bc

    def _build_active_settings(self) -> dict[str, Any]:
        keys = [
            "right_handed_coord", "interchange_dynamics", "parallel_dynamics", "perpendicular_dynamics", "invert_w_star", "force_profiles", "floor_profiles",
            "parallel_sheath_damping", "parallel_advection_damping", "parallel_conduction", "parallel_drift_wave", "reciprocal_approx", "collisional_model",
            "perpend_heat_exchange", "perpend_viscous_heating", "ti_over_te", "diffusion_coeff", "qdelta_approx", "double_curvature_coeff", "h_mode",
            "not_n_force", "not_p_force", "power_source", "particle_source", "parallel_transport", "plasma_neutral_interactions",
        ]
        return {key: self._resolve_key("hesel", key) for key in keys if key in self.cfg.get("hesel", {})}

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "paths": {key: str(path) for key, path in self.source_paths.items()},
            "config": {"settings_path": str(self.settings_path), "dump_path": str(self.dump_path) if self.dump_path.exists() else None},
            "active_settings": self.active_settings,
            "boundary_conditions": {field: {side: asdict(cond) for side, cond in field_bc.items()} for field, field_bc in self.boundary_conditions.items()},
            "derived_parameters": asdict(self.parameters),
        }

    def resolve_profile(self, section: str, x: Tensor, z: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        if section not in self.cfg or "function" not in self.cfg[section]:
            return torch.zeros_like(x)
        value = self._resolve_key(
            section,
            "function",
            variables={"x": x, "z": z if z is not None else torch.zeros_like(x), "y": torch.zeros_like(x), "t": t if t is not None else torch.zeros_like(x)},
        )
        value = value if torch.is_tensor(value) else _as_tensor_like(float(value), x)
        scale = float(self._resolve_key(section, "scale")) if "scale" in self.cfg[section] else 1.0
        return scale * value

    def initial_profiles(self, x: Tensor, z: Tensor | None = None) -> dict[str, Tensor]:
        z = z if z is not None else torch.zeros_like(x)
        profiles = {name: self.resolve_profile(name, x=x, z=z) for name in ("init_n", "init_pe", "init_pi", "sigma_open", "sigma_closed", "sigma_force")}
        profiles["seed_n"] = self.resolve_profile("seed_n", x=x, z=z) if "seed_n" in self.cfg else torch.zeros_like(x)
        profiles["lnn0_from_input"] = _safe_log(profiles["init_n"] + profiles["seed_n"])
        profiles["lnpe0_from_input"] = _safe_log(profiles["init_pe"])
        profiles["lnpi0_from_input"] = _safe_log(profiles["init_pi"])
        return profiles

    def magnetic_field(self, x: Tensor) -> Tensor:
        xr = self._resolve_key("hesel", "xr", variables={"x": x, "z": torch.zeros_like(x), "y": torch.zeros_like(x), "t": torch.zeros_like(x)})
        xr = xr if torch.is_tensor(xr) else _as_tensor_like(float(xr), x)
        numerator = self.parameters.rmajor + self.parameters.rminor
        return numerator / (numerator + self.parameters.rhos * xr)

    def make_collocation_grid(
        self,
        sample_shape: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if sample_shape is None:
            nx, nz, nt = 32, 64, 4
            if self.dump_data:
                ref = self.dump_data.get("lnn", self.dump_data.get("phi"))
                nx, nz = min(nx, int(ref.shape[1])), min(nz, int(ref.shape[-1]))
                nt = min(nt, int(self.dump_data.get("t_array", [0, 1, 2, 3]).shape[0]))
        else:
            nt, nx, nz = sample_shape
        t_grid, x_grid, z_grid = torch.meshgrid(
            torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, nz, device=device, dtype=dtype),
            indexing="ij",
        )
        return (
            x_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
            z_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
            t_grid.unsqueeze(-1).clone().detach().requires_grad_(True),
        )

    def _interp_dump_2d(self, values: Any, x: Tensor, z: Tensor) -> Tensor:
        field = torch.as_tensor(values, device=x.device, dtype=x.dtype)
        if field.ndim != 2:
            raise ValueError(f"Expected 2D dump field for interpolation, got shape {tuple(field.shape)}")
        x_pos = torch.clamp(x.squeeze(-1), 0.0, 1.0) * (field.shape[0] - 1)
        z_pos = torch.clamp(z.squeeze(-1), 0.0, 1.0) * (field.shape[1] - 1)
        x0, z0 = x_pos.floor().long().clamp(0, field.shape[0] - 1), z_pos.floor().long().clamp(0, field.shape[1] - 1)
        x1, z1 = (x0 + 1).clamp(0, field.shape[0] - 1), (z0 + 1).clamp(0, field.shape[1] - 1)
        wx, wz = (x_pos - x0.to(dtype=x.dtype)).unsqueeze(-1), (z_pos - z0.to(dtype=x.dtype)).unsqueeze(-1)
        v00, v10, v01, v11 = field[x0, z0].unsqueeze(-1), field[x1, z0].unsqueeze(-1), field[x0, z1].unsqueeze(-1), field[x1, z1].unsqueeze(-1)
        return (1.0 - wz) * ((1.0 - wx) * v00 + wx * v10) + wz * ((1.0 - wx) * v01 + wx * v11)

    def initial_state_targets(self, x: Tensor, z: Tensor) -> dict[str, Tensor]:
        if self.dump_data and {"lnn", "lnpe", "lnpi", "phi"}.issubset(self.dump_data):
            return {
                "lnn": self._interp_dump_2d(self.dump_data["lnn"][0, :, :], x, z),
                "lnpe": self._interp_dump_2d(self.dump_data["lnpe"][0, :, :], x, z),
                "lnpi": self._interp_dump_2d(self.dump_data["lnpi"][0, :, :], x, z),
                "phi": self._interp_dump_2d(self.dump_data["phi"][0, :, :], x, z),
                "vort": self._interp_dump_2d(self.dump_data["vort"][0, :, :], x, z) if "vort" in self.dump_data else torch.zeros_like(x),
            }
        profiles = self.initial_profiles(x=x, z=z)
        return {"lnn": profiles["lnn0_from_input"], "lnpe": profiles["lnpe0_from_input"], "lnpi": profiles["lnpi0_from_input"], "phi": torch.zeros_like(x), "vort": torch.zeros_like(x)}


__all__ = ["BOUTHESELInfo"]
