from __future__ import annotations

import re
import math
import torch
import xarray as xr
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from pathlib import Path
from dataclasses import asdict
from typing import Any, Callable, Iterable, Mapping


from .api.read_bout import read_bout_inp
from .api.types import BoundaryCondition, HeselDerivedParameters
from .api.operators import *


DEFAULT_BOUT_HESEL_ROOT = (Path(__file__).resolve().parents[2] / "simulatorer" / "BOUT" / "BOUT-HESEL")
REF_PATTERN = re.compile(r"\b([A-Za-z_]\w*):([A-Za-z_]\w*)\b")
BOUNDARY_PATTERN = re.compile(r"^(?P<kind>[A-Za-z_]\w*)(?:\((?P<expr>.*)\))?$")
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\b")


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

def mse_dict(residuals: Mapping[str, Tensor], weights: Mapping[str, float] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    weights = dict(weights or {})
    component_losses: dict[str, Tensor] = {}
    total = None
    for name, residual in residuals.items():
        weight = float(weights.get(name, 1.0))
        loss = F.mse_loss(weight * residual, torch.zeros_like(residual))
        component_losses[name] = loss
        total = loss if total is None else total + loss
    if total is None:
        total = torch.tensor(0.0)
    return total, component_losses

class BOUTHESELSystem:
    default_output_names = ("lnn", "lnpe", "lnpi", "phi")

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
                if "y" in arr.dims:
                    arr = arr.squeeze("y", drop=True)
                data[name] = arr.values
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

        expr = self.cfg[section][key]
        return self._eval_expression(
            expr=expr,
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
        expr_without_refs = REF_PATTERN.sub(" ", expr)
        reserved_names = {
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
        referenced_identifiers = {
            token
            for token in IDENTIFIER_PATTERN.findall(expr_without_refs)
            if token not in reserved_names and token not in variables
        }

        local_symbols: dict[str, Any] = {}
        for section_name in (current_section, "root"):
            if section_name not in self.cfg:
                continue
            for key_name in referenced_identifiers:
                if key_name not in self.cfg[section_name]:
                    continue
                if (section_name, key_name) in stack:
                    continue
                if key_name in local_symbols:
                    continue
                try:
                    local_symbols[key_name] = self._resolve_key(
                        section_name,
                        key_name,
                        variables=variables,
                        stack=stack,
                    )
                except Exception:
                    # Keep the resolver permissive: unsupported symbols are only a
                    # problem if the current expression actually needs them.
                    continue

        scope = {
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
        }

        return eval(rewritten, {"__builtins__": {}}, scope)

    def _scalar(self, section: str, key: str) -> float:
        value = self._resolve_key(section, key)
        if isinstance(value, bool):
            return float(value)
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar for {section}:{key}, got tensor with shape {value.shape}")
            return float(value.detach().cpu().item())
        return float(value)

    def _bool(self, section: str, key: str) -> bool:
        return bool(self._resolve_key(section, key))

    def _build_parameters(self) -> HeselDerivedParameters:
        e = 1.60e-19
        epso = 8.85e-12
        me = 9.1093816e-31
        mp = 1.67262158e-27
        pi_const = math.pi

        bt = self._scalar("hesel", "bt")
        q = self._scalar("hesel", "q")
        te0 = self._scalar("hesel", "te0")
        ti0 = self._scalar("hesel", "ti0")
        n0 = self._scalar("hesel", "n0")
        lconn = self._scalar("hesel", "lconn")
        rmajor = self._scalar("hesel", "rmajor")
        rminor = self._scalar("hesel", "rminor")
        a = self._scalar("hesel", "a")
        z = self._scalar("hesel", "z")
        mach = self._scalar("hesel", "mach")
        z_eff = self._scalar("hesel", "z_eff")
        x_lcfs = self._scalar("hesel", "x_lcfs")
        x_wall = self._scalar("hesel", "x_wall")
        force_time = self._scalar("hesel", "force_time")
        floor_time = self._scalar("hesel", "floor_time")
        floor_n = self._scalar("hesel", "floor_n")
        floor_pe = self._scalar("hesel", "floor_pe")
        floor_pi = self._scalar("hesel", "floor_pi")
        n_bck = self._scalar("hesel", "n_bck")
        te_bck = self._scalar("hesel", "te_bck")
        ti_bck = self._scalar("hesel", "ti_bck")
        d_lcfs = self._scalar("hesel", "d_lcfs")
        d_wall = self._scalar("hesel", "d_wall")
        d_force = self._scalar("hesel", "d_force")
        wall_amp = self._scalar("hesel", "wall_amp")

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
        nuei = (
            math.sqrt(2.0)
            * n0
            * z
            * z
            * e**4
            * collog
            / (12.0 * math.sqrt(pi_const**3) * math.sqrt(me) * math.sqrt((e * te0) ** 3) * epso * epso)
        )
        nuii = (
            n0
            * z**4
            * e**4
            * collog
            / (12.0 * math.sqrt(pi_const**3) * epso * epso * math.sqrt(mi) * math.sqrt((e * ti0) ** 3))
        )
        nuee = nuii / (z**4) * math.sqrt(mi / me) * math.sqrt((ti0 / te0) ** 3)

        neoclass_override = self._scalar("hesel", "neoclass_correction_factor")
        neoclass_correction_factor = (
            1.0 + neoclass_override if neoclass_override >= 0.0 else 1.0 + rmajor / rminor * q * q
        )
        lblob_setting = self._scalar("hesel", "lblob")
        lblob = lblob_setting if lblob_setting > 0.0 else q * rmajor

        bohm_potential = math.log(math.sqrt(mi / (2.0 * pi_const * me)))
        de_phys = neoclass_correction_factor * rhoe * rhoe * nuei
        di_phys = neoclass_correction_factor * rhoi * rhoi * nuii
        norm_de = de_phys / (rhos * rhos * oci)
        norm_di = di_phys / (rhos * rhos * oci)
        norm_eta = 3.0 / 10.0 * norm_di

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
            norm_eta=norm_eta,
            norm_taun=taun * oci,
            norm_taudw=taudw * oci,
            norm_taushe=taushe * oci,
            norm_taushi=taushi * oci,
            norm_lc=lconn / rhos,
            norm_lb=lblob / rhos,
        )

    def _parse_boundary(self, section: str, key: str) -> BoundaryCondition:
        raw = self.cfg[section][key].strip()
        match = BOUNDARY_PATTERN.match(raw)
        if match is None:
            return BoundaryCondition(raw=raw, kind=raw, value=None, source_section=section, source_key=key)

        kind = match.group("kind")
        expr = match.group("expr")
        value = None
        if expr:
            try:
                resolved = self._eval_expression(expr, current_section=section)
                if torch.is_tensor(resolved):
                    value = float(resolved.detach().cpu().item())
                elif isinstance(resolved, (int, float, bool)):
                    value = float(resolved)
            except Exception:
                value = None

        return BoundaryCondition(
            raw=raw,
            kind=kind,
            value=value,
            source_section=section,
            source_key=key,
        )

    def _build_boundary_conditions(self) -> dict[str, dict[str, BoundaryCondition]]:
        bc: dict[str, dict[str, BoundaryCondition]] = {}
        for field in ("lnn", "lnpe", "lnpi", "vort"):
            if field not in self.cfg:
                continue
            field_bc: dict[str, BoundaryCondition] = {}
            for key in ("bndry_xin", "bndry_xout"):
                if key in self.cfg[field]:
                    side = "inner" if key == "bndry_xin" else "outer"
                    field_bc[side] = self._parse_boundary(field, key)
            bc[field] = field_bc

        bc["phi"] = {
            "inner": BoundaryCondition(
                raw=f"laplace:inner_boundary_flags={self.cfg.get('laplace', {}).get('inner_boundary_flags', '0')}",
                kind="dirichlet",
                value=0.0,
                source_section="laplace",
                source_key="inner_boundary_flags",
            ),
            "outer": BoundaryCondition(
                raw=f"laplace:outer_boundary_flags={self.cfg.get('laplace', {}).get('outer_boundary_flags', '0')}",
                kind="neumann",
                value=0.0,
                source_section="laplace",
                source_key="outer_boundary_flags",
            ),
        }
        return bc

    def _build_active_settings(self) -> dict[str, Any]:
        keys = [
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
            "not_n_force",
            "not_p_force",
            "power_source",
            "particle_source",
            "parallel_transport",
            "plasma_neutral_interactions",
        ]

        settings: dict[str, Any] = {}
        for key in keys:
            if key not in self.cfg.get("hesel", {}):
                continue
            settings[key] = self._resolve_key("hesel", key)
        return settings

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "paths": {key: str(path) for key, path in self.source_paths.items()},
            "config": {
                "settings_path": str(self.settings_path),
                "dump_path": str(self.dump_path) if self.dump_path.exists() else None,
            },
            "active_settings": self.active_settings,
            "boundary_conditions": {
                field: {side: asdict(cond) for side, cond in field_bc.items()}
                for field, field_bc in self.boundary_conditions.items()
            },
            "derived_parameters": asdict(self.parameters),
            "equation_layout": {
                "rhs_terms": "RhsInterchangeDynamics in hesel.cxx",
                "lambda_terms": (
                    "RhsPerpendicularDynamics + RhsParallelDynamics + "
                    "RhsPlasmaFieldsForce + RhsPlasmaFieldsFloor in hesel.cxx"
                ),
                "poisson_relation": "vort = Delp2(phi) + Delp2(pi) when invert_w_star is false",
            },
        }

    def resolve_profile(self, section: str, x: Tensor, z: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        if section not in self.cfg:
            return torch.zeros_like(x)
        if "function" not in self.cfg[section]:
            return torch.zeros_like(x)

        variables = {
            "x": x,
            "z": z if z is not None else torch.zeros_like(x),
            "y": torch.zeros_like(x),
            "t": t if t is not None else torch.zeros_like(x),
        }
        value = self._resolve_key(section, "function", variables=variables)
        if not torch.is_tensor(value):
            value = _as_tensor_like(float(value), x)
        scale = float(self._resolve_key(section, "scale")) if "scale" in self.cfg[section] else 1.0
        return scale * value

    def initial_profiles(self, x: Tensor, z: Tensor | None = None) -> dict[str, Tensor]:
        z = z if z is not None else torch.zeros_like(x)
        profiles = {
            "init_n": self.resolve_profile("init_n", x=x, z=z),
            "init_pe": self.resolve_profile("init_pe", x=x, z=z),
            "init_pi": self.resolve_profile("init_pi", x=x, z=z),
            "sigma_open": self.resolve_profile("sigma_open", x=x, z=z),
            "sigma_closed": self.resolve_profile("sigma_closed", x=x, z=z),
            "sigma_force": self.resolve_profile("sigma_force", x=x, z=z),
        }

        if "seed_n" in self.cfg:
            profiles["seed_n"] = self.resolve_profile("seed_n", x=x, z=z)
        else:
            profiles["seed_n"] = torch.zeros_like(x)
        profiles["lnn0_from_input"] = _safe_log(profiles["init_n"] + profiles["seed_n"])
        profiles["lnpe0_from_input"] = _safe_log(profiles["init_pe"])
        profiles["lnpi0_from_input"] = _safe_log(profiles["init_pi"])
        return profiles

    def magnetic_field(self, x: Tensor) -> Tensor:
        xr = self._resolve_key(
            "hesel",
            "xr",
            variables={"x": x, "z": torch.zeros_like(x), "y": torch.zeros_like(x), "t": torch.zeros_like(x)},
        )
        if not torch.is_tensor(xr):
            xr = _as_tensor_like(float(xr), x)
        numerator = self.parameters.rmajor + self.parameters.rminor
        denominator = numerator + self.parameters.rhos * xr
        return numerator / denominator

    def make_collocation_grid(
        self,
        sample_shape: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if sample_shape is None:
            nx = 32
            nz = 64
            nt = 4
            if self.dump_data:
                nx = min(nx, int(self.dump_data.get("lnn", self.dump_data.get("phi")).shape[1]))
                nz = min(nz, int(self.dump_data.get("lnn", self.dump_data.get("phi")).shape[-1]))
                nt = min(nt, int(self.dump_data.get("t_array", [0, 1, 2, 3]).shape[0]))
        else:
            nt, nx, nz = sample_shape

        x_1d = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
        z_1d = torch.linspace(0.0, 1.0, nz, device=device, dtype=dtype)
        t_1d = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
        t_grid, x_grid, z_grid = torch.meshgrid(t_1d, x_1d, z_1d, indexing="ij")

        x_grid = x_grid.unsqueeze(-1).clone().detach().requires_grad_(True)
        z_grid = z_grid.unsqueeze(-1).clone().detach().requires_grad_(True)
        t_grid = t_grid.unsqueeze(-1).clone().detach().requires_grad_(True)
        return x_grid, z_grid, t_grid

    def _interp_dump_2d(self, values: Any, x: Tensor, z: Tensor) -> Tensor:
        field = torch.as_tensor(values, device=x.device, dtype=x.dtype)
        if field.ndim != 2:
            raise ValueError(f"Expected 2D dump field for interpolation, got shape {tuple(field.shape)}")

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

        v0 = (1.0 - wx) * v00 + wx * v10
        v1 = (1.0 - wx) * v01 + wx * v11
        return (1.0 - wz) * v0 + wz * v1

    def initial_state_targets(self, x: Tensor, z: Tensor) -> dict[str, Tensor]:
        if self.dump_data and {"lnn", "lnpe", "lnpi", "phi"}.issubset(self.dump_data):
            return {
                "lnn": self._interp_dump_2d(self.dump_data["lnn"][0, :, :], x, z),
                "lnpe": self._interp_dump_2d(self.dump_data["lnpe"][0, :, :], x, z),
                "lnpi": self._interp_dump_2d(self.dump_data["lnpi"][0, :, :], x, z),
                "phi": self._interp_dump_2d(self.dump_data["phi"][0, :, :], x, z),
                "vort": self._interp_dump_2d(self.dump_data["vort"][0, :, :], x, z)
                if "vort" in self.dump_data
                else torch.zeros_like(x),
            }

        profiles = self.initial_profiles(x=x, z=z)
        return {
            "lnn": profiles["lnn0_from_input"],
            "lnpe": profiles["lnpe0_from_input"],
            "lnpi": profiles["lnpi0_from_input"],
            "phi": torch.zeros_like(x),
            "vort": torch.zeros_like(x),
        }

    def _split_model_outputs(
        self,
        output: Any,
        output_names: Iterable[str] | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(output, Mapping):
            return {str(key): value for key, value in output.items()}

        if isinstance(output, (tuple, list)):
            if output_names is None:
                if len(output) == 4:
                    output_names = self.default_output_names
                elif len(output) == 5:
                    output_names = (*self.default_output_names, "vort")
                else:
                    raise ValueError("Could not infer output names from model output.")
            return {name: tensor for name, tensor in zip(output_names, output)}

        if not torch.is_tensor(output):
            raise TypeError("Model output must be a tensor, tuple/list, or mapping.")

        if output_names is None:
            if output.shape[-1] == 4:
                output_names = self.default_output_names
            elif output.shape[-1] == 5:
                output_names = (*self.default_output_names, "vort")
            else:
                raise ValueError(
                    "Could not infer output names from tensor shape. "
                    "Pass output_names explicitly."
                )

        names = list(output_names)
        if output.shape[-1] != len(names):
            raise ValueError(
                f"Model output last dimension {output.shape[-1]} does not match "
                f"the number of output names {len(names)}."
            )
        return {name: output[..., idx : idx + 1] for idx, name in enumerate(names)}

    def evaluate_model(
        self,
        model: Callable[..., Any],
        x: Tensor,
        z: Tensor,
        t: Tensor,
        output_names: Iterable[str] | None = None,
    ) -> dict[str, Tensor]:
        return self._split_model_outputs(model(x, z, t), output_names=output_names)

    def _zavg(self, value: Tensor) -> Tensor:
        return value.mean(dim=2, keepdim=True)

    def equation_terms(
        self,
        model: Callable[..., Any] | None = None,
        *,
        x: Tensor | None = None,
        z: Tensor | None = None,
        t: Tensor | None = None,
        state: Mapping[str, Tensor] | None = None,
        output_names: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if state is None:
            if model is None:
                raise ValueError("Provide either model or state.")
            if x is None or z is None or t is None:
                x, z, t = self.make_collocation_grid(device=next(model.parameters()).device if hasattr(model, "parameters") else None)
            state = self.evaluate_model(model, x=x, z=z, t=t, output_names=output_names)
        else:
            state = dict(state)
            if x is None or z is None or t is None:
                raise ValueError("When state is given, x, z and t must also be given.")

        if "phi" not in state:
            raise ValueError("The HESEL PINN loss requires `phi` to be part of the state.")

        p = self.parameters
        settings = self.active_settings

        lnn = state["lnn"]
        lnpe = state["lnpe"]
        lnpi = state["lnpi"]
        phi = state["phi"]

        n = torch.exp(lnn)
        pe = torch.exp(lnpe)
        pi = torch.exp(lnpi)
        lnte = lnpe - lnn
        lnti = lnpi - lnn
        te = torch.exp(lnte)
        ti = torch.exp(lnti)
        tau = torch.exp(lnti - lnte)
        cs_hot = torch.sqrt(torch.clamp(ti + te, min=1e-12))
        avg_n = self._zavg(n)
        avg_te = self._zavg(te)
        avg_ti = self._zavg(ti)
        avg_phi = self._zavg(phi)
        avg_tau = self._zavg(tau)
        avg_cs_hot = self._zavg(cs_hot)

        b_field = self.magnetic_field(x)
        inv_b = 1.0 / torch.clamp(b_field, min=1e-12)

        dphi_dx = grad_x(phi, x, p.total_x)
        dphi_dz = grad_z(phi, z, p.total_z)
        dpi_dx = grad_x(pi, x, p.total_x)
        dpi_dz = grad_z(pi, z, p.total_z)
        dlnn_dx = grad_x(lnn, x, p.total_x)
        dlnn_dz = grad_z(lnn, z, p.total_z)
        dlnte_dx = grad_x(lnte, x, p.total_x)
        dlnte_dz = grad_z(lnte, z, p.total_z)
        dlnti_dx = grad_x(lnti, x, p.total_x)
        dlnti_dz = grad_z(lnti, z, p.total_z)

        ddt_lnn = grad_t(lnn, t, p.total_t)
        ddt_lnpe = grad_t(lnpe, t, p.total_t)
        ddt_lnpi = grad_t(lnpi, t, p.total_t)

        vort_from_phi = laplacian_perp(phi + pi, x, z, p.total_x, p.total_z)
        vort = state["vort"] if "vort" in state else vort_from_phi
        ddt_vort = grad_t(vort, t, p.total_t)

        def brackets(f: Tensor, g: Tensor) -> Tensor:
            bracket = grad_x(f, x, p.total_x) * grad_z(g, z, p.total_z) - grad_z(f, z, p.total_z) * grad_x(g, x, p.total_x)
            return -bracket if bool(settings.get("right_handed_coord", False)) else bracket

        def curvature(f: Tensor) -> Tensor:
            coeff = 2.0 if bool(settings.get("double_curvature_coeff", False)) else 1.0
            curv = coeff * p.rhos / (p.rmajor + p.rminor) * grad_z(f, z, p.total_z)
            return -curv if bool(settings.get("right_handed_coord", False)) else curv

        sigma = self.initial_profiles(x=x, z=z)
        sigma_open = sigma["sigma_open"]
        sigma_closed = sigma["sigma_closed"]
        sigma_force = sigma["sigma_force"]
        init_n = sigma["init_n"]
        init_pe = sigma["init_pe"]
        init_pi = sigma["init_pi"]

        interchange = {
            "lnn": torch.zeros_like(lnn),
            "lnpe": torch.zeros_like(lnpe),
            "lnpi": torch.zeros_like(lnpi),
            "vort": torch.zeros_like(vort),
        }
        if bool(settings.get("interchange_dynamics", True)):
            interchange["lnn"] = -inv_b * brackets(phi, lnn) - curvature(phi) + curvature(te) + curvature(lnn) * te
            interchange["lnpe"] = -inv_b * brackets(phi, lnpe) - 5.0 / 3.0 * curvature(phi) + 5.0 / 3.0 * curvature(te) + 5.0 / 3.0 * curvature(lnpe) * te
            interchange["lnpi"] = -inv_b * brackets(phi, lnpi) - 5.0 / 3.0 * curvature(phi) - 5.0 / 3.0 * curvature(ti) - 5.0 / 3.0 * curvature(lnpi) * ti + 2.0 / 3.0 * curvature(pe + pi)
            interchange["vort"] = -brackets(phi, vort) + curvature(pe + pi)
            if not bool(settings.get("test_vort_cross_term", False)):
                interchange["vort"] = interchange["vort"] - brackets(dphi_dx, dpi_dx) - brackets(dphi_dz, dpi_dz)

        if int(settings.get("ti_over_te", 3)) == 1:
            ti_rcpte = torch.full_like(tau, p.ti0 / p.te0)
        elif int(settings.get("ti_over_te", 3)) == 2:
            ti_rcpte = avg_tau
        else:
            ti_rcpte = tau

        if int(settings.get("diffusion_coeff", 1)) == 1:
            de = torch.full_like(n, p.norm_de)
            di = torch.full_like(n, p.norm_di)
        elif int(settings.get("diffusion_coeff", 1)) == 2:
            de = p.norm_de * avg_n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = p.norm_di * avg_n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif int(settings.get("diffusion_coeff", 1)) == 3:
            de = p.norm_de * n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = p.norm_di * n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif int(settings.get("diffusion_coeff", 1)) == 4:
            de = p.norm_de * avg_n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = p.norm_di * avg_n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif int(settings.get("diffusion_coeff", 1)) == 5:
            de = p.norm_de * n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = p.norm_di * n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif int(settings.get("diffusion_coeff", 1)) == 6:
            de = torch.full_like(n, p.norm_de) / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = torch.full_like(n, p.norm_di) / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        else:
            raise ValueError("Unsupported diffusion_coeff option from BOUT-HESEL.")

        de = de * p.z_eff
        di = di * p.z_eff
        dn = de * (1.0 + ti_rcpte)

        if int(settings.get("collisional_model", 2)) == 0:
            collisional = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif int(settings.get("collisional_model", 2)) == 1:
            collisional = {
                "lnn": p.norm_de * (1.0 + p.ti0 / p.te0) * laplacian_perp(lnn, x, z, p.total_x, p.total_z),
                "lnpe": 2.0 / 3.0 * p.norm_de * (1.0 + p.ti0 / p.te0) * laplacian_perp(lnpe, x, z, p.total_x, p.total_z),
                "lnpi": 2.0 / 3.0 * 2.0 * p.norm_di * laplacian_perp(lnti, x, z, p.total_x, p.total_z),
                "vort": p.norm_eta * laplacian_perp(vort, x, z, p.total_x, p.total_z),
            }
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif int(settings.get("collisional_model", 2)) == 2:
            deln_rcpn = laplacian_perp(lnn, x, z, p.total_x, p.total_z) + dlnn_dx * dlnn_dx + dlnn_dz * dlnn_dz
            delte_rcpte = laplacian_perp(lnte, x, z, p.total_x, p.total_z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z, p.total_x, p.total_z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            gradn_gradte_rcppe = dlnn_dx * dlnte_dx + dlnn_dz * dlnte_dz
            gradn_gradti_rcppi = dlnn_dx * dlnti_dx + dlnn_dz * dlnti_dz
            collisional = {
                "lnn": dn * deln_rcpn,
                "lnpe": 2.0 / 3.0 * dn * (deln_rcpn + gradn_gradte_rcppe) + 2.0 / 3.0 * 29.0 / 12.0 * de * (delte_rcpte + gradn_gradte_rcppe),
                "lnpi": 2.0 / 3.0 * 5.0 / 2.0 * dn * (deln_rcpn + gradn_gradti_rcppi) + 2.0 / 3.0 * 2.0 * di * (delti_rcpti + gradn_gradti_rcppi),
                "vort": p.norm_eta * laplacian_perp(vort, x, z, p.total_x, p.total_z),
            }
            u_r_x = -dn * dlnn_dx
            u_r_z = -dn * dlnn_dz
        elif int(settings.get("collisional_model", 2)) == 3:
            delte_rcpte = laplacian_perp(lnte, x, z, p.total_x, p.total_z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z, p.total_x, p.total_z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            u_r_x = -de * ((1.0 + ti_rcpte) * dlnn_dx + dlnti_dx * ti_rcpte - 0.5 * dlnte_dx)
            u_r_z = -de * ((1.0 + ti_rcpte) * dlnn_dz + dlnti_dz * ti_rcpte - 0.5 * dlnte_dz)
            div_gamma_r_rcpn = dlnn_dx * u_r_x + dlnn_dz * u_r_z + grad_x(u_r_x, x, p.total_x) + grad_z(u_r_z, z, p.total_z)
            collisional = {
                "lnn": -div_gamma_r_rcpn,
                "lnpe": -2.0 / 3.0 * (div_gamma_r_rcpn + dlnte_dx * u_r_x + dlnte_dz * u_r_z) + 2.0 / 3.0 * 29.0 / 12.0 * ((grad_x(de, x, p.total_x) + de * dlnn_dx) * dlnte_dx + (grad_z(de, z, p.total_z) + de * dlnn_dz) * dlnte_dz + de * delte_rcpte),
                "lnpi": -2.0 / 3.0 * 5.0 / 2.0 * (div_gamma_r_rcpn + dlnti_dx * u_r_x + dlnti_dz * u_r_z) + 2.0 / 3.0 * 2.0 * ((grad_x(di, x, p.total_x) + di * dlnn_dx) * dlnti_dx + (grad_z(di, z, p.total_z) + di * dlnn_dz) * dlnti_dz + di * delti_rcpti),
                "vort": p.norm_eta * laplacian_perp(vort, x, z, p.total_x, p.total_z),
            }
        else:
            raise ValueError("Unsupported collisional_model option from BOUT-HESEL.")

        heat_exchange = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
        if bool(settings.get("perpend_heat_exchange", True)):
            q_resist_rcppi = u_r_x * (dlnn_dx + dlnti_dx) + u_r_z * (dlnn_dz + dlnti_dz)
            qdelta_approx = int(settings.get("qdelta_approx", 3))
            if qdelta_approx == 0:
                qdelta_rcppe = torch.zeros_like(n)
            elif qdelta_approx == 1:
                qdelta_rcppe = 3.0 * p.me / p.mi * (p.nuei / p.oci) * (1.0 - ti_rcpte)
            elif qdelta_approx == 2:
                qdelta_rcppe = 3.0 * p.me / p.mi * (p.nuei / p.oci) * (1.0 - ti_rcpte) * avg_n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif qdelta_approx == 3:
                qdelta_rcppe = 3.0 * p.me / p.mi * (p.nuei / p.oci) * (1.0 - ti_rcpte) * n / te / torch.sqrt(torch.clamp(te, min=1e-12))
            elif qdelta_approx == 4:
                qdelta_rcppe = 3.0 * p.me / p.mi * (p.nuei / p.oci) * (1.0 - ti_rcpte) * n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            else:
                raise ValueError("Unsupported qdelta_approx option from BOUT-HESEL.")

            heat_exchange["lnpe"] = -2.0 / 3.0 * q_resist_rcppi * ti_rcpte - 2.0 / 3.0 * qdelta_rcppe
            heat_exchange["lnpi"] = 2.0 / 3.0 * q_resist_rcppi + 2.0 / 3.0 * qdelta_rcppe / torch.clamp(ti_rcpte, min=1e-12)

        viscous = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
        if bool(settings.get("perpend_viscous_heating", True)):
            field_sum = phi + pi
            qviscous = (
                3.0
                / 10.0
                * di
                * (
                    (d2dx2(field_sum, x, p.total_x) - d2dz2(field_sum, z, p.total_z)) ** 2
                    + 4.0 * d2dxdz(field_sum, x, z, p.total_x, p.total_z) ** 2
                )
                / torch.clamp(ti, min=1e-12)
            )
            viscous["lnpi"] = 2.0 / 3.0 * qviscous

        perpendicular = {
            name: collisional[name] + heat_exchange[name] + viscous[name]
            for name in interchange
        }
        if not bool(settings.get("perpendicular_dynamics", True)):
            perpendicular = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}

        parallel = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
        if bool(settings.get("parallel_dynamics", True)):
            advection_mode = int(settings.get("parallel_advection_damping", 3))
            if advection_mode == 0:
                damp_advection = torch.zeros_like(n)
            elif advection_mode == 1:
                damp_advection = torch.full_like(n, 1.0 / p.norm_taun)
            elif advection_mode == 2:
                damp_advection = avg_cs_hot / p.norm_taun
            elif advection_mode == 3:
                damp_advection = cs_hot / p.norm_taun
            else:
                raise ValueError("Unsupported parallel_advection_damping option from BOUT-HESEL.")

            parallel["lnn"] = parallel["lnn"] - sigma_open * damp_advection
            parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["vort"] = parallel["vort"] - sigma_open * damp_advection * vort

            sheath_mode = int(settings.get("parallel_sheath_damping", 3))
            if sheath_mode == 0:
                damp_sheath = torch.zeros_like(n)
            elif sheath_mode == 1:
                damp_sheath = 1.0 / p.norm_lc * (1.0 - torch.exp(p.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif sheath_mode == 2:
                damp_sheath = avg_cs_hot / p.norm_lc * (1.0 - torch.exp(p.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif sheath_mode == 3:
                damp_sheath = cs_hot / p.norm_lc * (1.0 - torch.exp(p.bohm_potential - phi / torch.clamp(te, min=1e-12)))
            else:
                raise ValueError("Unsupported parallel_sheath_damping option from BOUT-HESEL.")

            parallel["lnpi"] = parallel["lnpi"] + 2.0 / 3.0 * sigma_open * damp_sheath
            parallel["vort"] = parallel["vort"] + sigma_open * damp_sheath

            conduction_mode = int(settings.get("parallel_conduction", 1))
            if conduction_mode == 0:
                damp_she = torch.zeros_like(n)
                damp_shi = torch.zeros_like(n)
            elif conduction_mode == 1:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / p.norm_taushe
                damp_shi = torch.zeros_like(n)
            elif conduction_mode == 2:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / p.norm_taushe
                damp_shi = ti * ti * torch.sqrt(torch.clamp(ti, min=1e-12)) / p.norm_taushi
            else:
                raise ValueError("Unsupported parallel_conduction option from BOUT-HESEL.")

            pert_n = n - avg_n
            pert_te = te - avg_te
            pert_phi = phi - avg_phi
            driftwave_mode = int(settings.get("parallel_drift_wave", 1))
            if driftwave_mode == 0:
                driftwave = torch.zeros_like(n)
            elif driftwave_mode == 1:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / p.norm_taudw
            elif driftwave_mode == 2:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / p.norm_taudw * avg_te * torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif driftwave_mode == 3:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / p.norm_taudw * te * torch.sqrt(torch.clamp(te, min=1e-12))
            else:
                raise ValueError("Unsupported parallel_drift_wave option from BOUT-HESEL.")

            parallel["vort"] = parallel["vort"] - sigma_closed * driftwave

            reciprocal_approx = int(settings.get("reciprocal_approx", 3))
            if reciprocal_approx == 1:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti
            elif reciprocal_approx == 2:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi / torch.clamp(avg_n, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave
            elif reciprocal_approx == 3:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she * (te - p.te_bck) / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi * (te - p.ti_bck) / torch.clamp(pi, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti / torch.clamp(pi, min=1e-12)
            else:
                raise ValueError("Unsupported reciprocal_approx option from BOUT-HESEL.")

        forcing = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
        if bool(settings.get("force_profiles", True)):
            if not bool(settings.get("not_n_force", False)):
                forcing["lnn"] = forcing["lnn"] + sigma_force * (init_n / torch.clamp(n, min=1e-12) - 1.0) / p.force_time
            if not bool(settings.get("not_p_force", False)):
                forcing_multiplier = torch.ones_like(pi)
                if bool(settings.get("h_mode", False)):
                    ramp_a = float(settings.get("ramp_a", 2.0))
                    ramp_t0 = float(settings.get("ramp_t0", 0.0))
                    ramp_trans = float(settings.get("ramp_trans", 5000.0))
                    ramp_peak = float(settings.get("ramp_peak", 50000.0))
                    t_phys = t * p.total_t
                    forcing_multiplier = forcing_multiplier * (
                        1.0
                        + (ramp_a - 1.0)
                        / 2.0
                        * (
                            torch.tanh((t_phys - ramp_t0) / ramp_trans)
                            - torch.tanh((t_phys - ramp_t0 - ramp_peak) / ramp_trans)
                        )
                    )
                force_pe = sigma_force * (init_pe - pe) / p.force_time
                force_pi = sigma_force * (init_pi * forcing_multiplier - pi) / p.force_time
                forcing["lnpe"] = forcing["lnpe"] + force_pe / torch.clamp(pe, min=1e-12)
                forcing["lnpi"] = forcing["lnpi"] + force_pi / torch.clamp(pi, min=1e-12)

        floor_terms = {name: torch.zeros_like(tensor) for name, tensor in interchange.items()}
        if bool(settings.get("floor_profiles", False)):
            floor_terms["lnn"] = floor_terms["lnn"] + torch.where(
                n < p.floor_n,
                (p.floor_n / torch.clamp(n, min=1e-12) - 1.0) / p.floor_time,
                torch.zeros_like(n),
            )
            floor_terms["lnpe"] = floor_terms["lnpe"] + torch.where(
                pe < p.floor_pe,
                (p.floor_pe / torch.clamp(pe, min=1e-12) - 1.0) / p.floor_time,
                torch.zeros_like(pe),
            )
            floor_terms["lnpi"] = floor_terms["lnpi"] + torch.where(
                pi < p.floor_pi,
                (p.floor_pi / torch.clamp(pi, min=1e-12) - 1.0) / p.floor_time,
                torch.zeros_like(pi),
            )

        rhs_terms = interchange
        lambda_terms = {
            name: perpendicular[name] + parallel[name] + forcing[name] + floor_terms[name]
            for name in interchange
        }

        residuals = {
            "eq_lnn": ddt_lnn - rhs_terms["lnn"] - lambda_terms["lnn"],
            "eq_lnpe": ddt_lnpe - rhs_terms["lnpe"] - lambda_terms["lnpe"],
            "eq_lnpi": ddt_lnpi - rhs_terms["lnpi"] - lambda_terms["lnpi"],
            "eq_vort": ddt_vort - rhs_terms["vort"] - lambda_terms["vort"],
        }

        result = {
            "state": {
                **state,
                "n": n,
                "pe": pe,
                "pi": pi,
                "te": te,
                "ti": ti,
                "tau": tau,
                "vort_from_phi": vort_from_phi,
                "vort": vort,
                "B": b_field,
                "sigma_open": sigma_open,
                "sigma_closed": sigma_closed,
                "sigma_force": sigma_force,
                "init_n": init_n,
                "init_pe": init_pe,
                "init_pi": init_pi,
            },
            "rhs_terms": rhs_terms,
            "lambda_terms": lambda_terms,
            "components": {
                "interchange": interchange,
                "perpendicular": perpendicular,
                "parallel": parallel,
                "forcing": forcing,
                "floor": floor_terms,
            },
            "residuals": residuals,
        }
        if "vort" in state:
            result["poisson_residual"] = state["vort"] - vort_from_phi
        return result

    def eq_loss(
        self,
        model: Callable[..., Any],
        weights: Mapping[str, float] | None = None,
        *,
        x: Tensor | None = None,
        z: Tensor | None = None,
        t: Tensor | None = None,
        output_names: Iterable[str] | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        if x is None or z is None or t is None:
            device = next(model.parameters()).device if hasattr(model, "parameters") else None
            x, z, t = self.make_collocation_grid(device=device)

        terms = self.equation_terms(model, x=x, z=z, t=t, output_names=output_names)
        total, component_losses = mse_dict(terms["residuals"], weights=weights)

        if return_details:
            details = {
                "total": total,
                "component_losses": component_losses,
                **terms,
            }
            return details
        return total

    def bc_loss(
        self,
        model: Callable[..., Any],
        weights: Mapping[str, float] | None = None,
        *,
        x: Tensor | None = None,
        z: Tensor | None = None,
        t: Tensor | None = None,
        output_names: Iterable[str] | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        if x is None or z is None or t is None:
            device = next(model.parameters()).device if hasattr(model, "parameters") else None
            x, z, t = self.make_collocation_grid(device=device)

        state = self.evaluate_model(model, x=x, z=z, t=t, output_names=output_names)
        residuals: dict[str, Tensor] = {}

        for field in ("lnn", "lnpe", "lnpi", "phi"):
            if field not in state or field not in self.boundary_conditions:
                continue
            inner = self.boundary_conditions[field]["inner"]
            outer = self.boundary_conditions[field]["outer"]

            if inner.kind.startswith("dirichlet"):
                target = 0.0 if inner.value is None else inner.value
                residuals[f"bc_{field}_inner"] = state[field][:, 0, :, :] - target
            elif inner.kind.startswith("neumann"):
                residuals[f"bc_{field}_inner"] = grad_x(state[field], x, self.parameters.total_x)[:, 0, :, :]

            if outer.kind.startswith("dirichlet"):
                target = 0.0 if outer.value is None else outer.value
                residuals[f"bc_{field}_outer"] = state[field][:, -1, :, :] - target
            elif outer.kind.startswith("neumann"):
                residuals[f"bc_{field}_outer"] = grad_x(state[field], x, self.parameters.total_x)[:, -1, :, :]

        if "vort" in state and "vort" in self.boundary_conditions:
            vort_inner = self.boundary_conditions["vort"]["inner"]
            vort_outer = self.boundary_conditions["vort"]["outer"]
            if vort_inner.kind.startswith("dirichlet"):
                residuals["bc_vort_inner"] = state["vort"][:, 0, :, :] - (vort_inner.value or 0.0)
            elif vort_inner.kind.startswith("neumann"):
                residuals["bc_vort_inner"] = grad_x(state["vort"], x, self.parameters.total_x)[:, 0, :, :]

            if vort_outer.kind.startswith("dirichlet"):
                residuals["bc_vort_outer"] = state["vort"][:, -1, :, :] - (vort_outer.value or 0.0)
            elif vort_outer.kind.startswith("neumann"):
                residuals["bc_vort_outer"] = grad_x(state["vort"], x, self.parameters.total_x)[:, -1, :, :]

        total, component_losses = mse_dict(residuals, weights=weights)
        if return_details:
            return {
                "total": total,
                "component_losses": component_losses,
                "residuals": residuals,
                "boundary_conditions": {
                    field: {side: asdict(cond) for side, cond in field_bc.items()}
                    for field, field_bc in self.boundary_conditions.items()
                },
            }
        return total

    def ic_loss(
        self,
        model: Callable[..., Any],
        weights: Mapping[str, float] | None = None,
        *,
        x: Tensor | None = None,
        z: Tensor | None = None,
        output_names: Iterable[str] | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        if x is None or z is None:
            device = next(model.parameters()).device if hasattr(model, "parameters") else None
            x_1d = torch.linspace(0.0, 1.0, 32, device=device)
            z_1d = torch.linspace(0.0, 1.0, 64, device=device)
            x_grid, z_grid = torch.meshgrid(x_1d, z_1d, indexing="ij")
            x = x_grid.unsqueeze(0).unsqueeze(-1).clone().detach().requires_grad_(True)
            z = z_grid.unsqueeze(0).unsqueeze(-1).clone().detach().requires_grad_(True)
        t0 = torch.zeros_like(x, requires_grad=True)

        state = self.evaluate_model(model, x=x, z=z, t=t0, output_names=output_names)
        targets = self.initial_state_targets(x.squeeze(0), z.squeeze(0))

        residuals = {
            "ic_lnn": state["lnn"].squeeze(0) - targets["lnn"],
            "ic_lnpe": state["lnpe"].squeeze(0) - targets["lnpe"],
            "ic_lnpi": state["lnpi"].squeeze(0) - targets["lnpi"],
            "ic_phi": state["phi"].squeeze(0) - targets["phi"],
        }
        if "vort" in state:
            residuals["ic_vort"] = state["vort"].squeeze(0) - targets["vort"]

        total, component_losses = mse_dict(residuals, weights=weights)
        if return_details:
            return {
                "total": total,
                "component_losses": component_losses,
                "residuals": residuals,
                "targets": targets,
            }
        return total

    def data_loss(
        self,
        model: Callable[..., Any],
        data: dict[str, Tensor],
        weights: Mapping[str, float] | None = None,
        *,
        x: Tensor | None = None,
        z: Tensor | None = None,
        t: Tensor | None = None,
        output_names: Iterable[str] | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        if x is None or z is None or t is None:
            device = next(model.parameters()).device if hasattr(model, "parameters") else None
            x, z, t = self.make_collocation_grid(device=device)

        state = self.evaluate_model(model, x=x, z=z, t=t, output_names=output_names)
        residuals = {}
        for key in data:
            if key in state:
                residuals[f"data_{key}"] = state[key] - data[key]

        total, component_losses = mse_dict(residuals, weights=weights)
        if return_details:
            return {
                "total": total,
                "component_losses": component_losses,
                "residuals": residuals,
                "data": data,
            }
        return total
    
