import math
from pathlib import Path

import torch
from torch import Tensor

_PROTON_MASS = 1.67262192369e-27
_ELECTRON_MASS = 9.1093837015e-31
_EPS = 1e-12


def _get_param(params: dict, *keys: str, default=None):
    for key in keys:
        if key in params and params[key] is not None:
            return params[key]
    return default


def _get_inverse_param(
    params: dict,
    *,
    inverse_keys: tuple[str, ...],
    tau_keys: tuple[str, ...],
    default: float,
) -> float | Tensor:
    inv = _get_param(params, *inverse_keys, default=None)
    if inv is not None:
        return inv

    tau = _get_param(params, *tau_keys, default=None)
    if tau is None:
        return default

    return 1.0 / (tau + _EPS)


def _evaluate_expressions(params: dict) -> dict:
    max_iterations = 10
    for _ in range(max_iterations):
        changed = False
        for key in list(params.keys()):
            value = params[key]
            if isinstance(value, str):
                try:
                    eval_context = {k: v for k, v in params.items() if isinstance(v, (int, float))}
                    eval_context["sqrt"] = math.sqrt
                    eval_context["abs"] = abs
                    params[key] = float(eval(value, {"__builtins__": {}}, eval_context))
                    changed = True
                except Exception:
                    pass
        if not changed:
            break
    return params


def _parse_bout_format(path: Path) -> dict:
    params = {}
    current_section = None

    with open(path, "r") as file_handle:
        for line in file_handle:
            if "#" in line:
                line = line[: line.index("#")]

            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].lower()
                continue

            if current_section == "hesel" and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

    params = _evaluate_expressions(params)

    param_mappings = {
        "B": ["B", "B0", "Bt"],
        "rho_s": ["rho_s", "rhos"],
        "R": ["R"],
        "Rmajor": ["Rmajor", "R_major", "Rmaj"],
        "Rminor": ["Rminor", "R_minor", "Rmin"],
        "D_e": ["D_e", "De"],
        "D_i": ["D_i", "Di"],
        "tau_n": ["tau_n", "taun"],
        "tau_n_inv": ["tau_n_inv", "taun_inv"],
        "tau_w": ["tau_w", "tau_omega", "tauw"],
        "tau_w_inv": ["tau_w_inv", "tau_omega_inv"],
        "tau_pe": ["tau_pe", "taupe"],
        "tau_pe_inv": ["tau_pe_inv", "taupe_inv"],
        "tau_pi": ["tau_pi", "taupi"],
        "tau_pi_inv": ["tau_pi_inv", "taupi_inv"],
        "tau_SH_e": ["tau_SH_e", "tau_she"],
        "tau_SH_e_inv": ["tau_SH_e_inv", "tau_she_inv"],
        "alpha": ["alpha"],
        "L_c": ["L_c", "lconn"],
        "phi_s": ["phi_s", "bohm_potential"],
        "T_e_s": ["T_e_s", "te_s", "Te0"],
        "nu_ei": ["nu_ei", "nuei"],
        "A": ["A", "mass_ratio"],
        "me": ["me"],
        "mi": ["mi"],
        "y_dim": ["y_dim", "ny"],
        "curvature_coeff": ["curvature_coeff"],
    }

    normalized_params = {}
    for canonical_name, alt_names in param_mappings.items():
        for alt_name in alt_names:
            if alt_name in params:
                normalized_params[canonical_name] = params[alt_name]
                break

    return normalized_params


def load_params_from_bout_inp(bout_inp_path: str | Path) -> dict:
    path = Path(bout_inp_path)
    if not path.exists():
        raise FileNotFoundError(f"BOUT.inp fil ikke fundet: {path}")
    return _parse_bout_format(path)


def resolve_hesel_params(params: dict | None = None, phi: Tensor | None = None) -> dict:
    if params is None:
        params = {}
    if phi is None:
        phi = torch.tensor(0.0)

    B = _get_param(params, "B", "B0", "Bt")
    rho_s = _get_param(params, "rho_s", "rhos")

    R = _get_param(params, "R")
    if R is None:
        Rmajor = _get_param(params, "Rmajor")
        Rminor = _get_param(params, "Rminor")
        if Rmajor is not None and Rminor is not None:
            R = Rmajor + Rminor
        elif Rmajor is not None:
            R = Rmajor

    if B is None or rho_s is None or R is None:
        raise KeyError("Missing required HESEL parameters. Provide B/B0/Bt, rho_s/rhos and R or Rmajor(+Rminor).")

    hparams = {
        "B": B,
        "rho_s": rho_s,
        "R": R,
        "D_e": _get_param(params, "D_e", "De", default=0.0),
        "D_i": _get_param(params, "D_i", "Di", default=0.0),
        "tau_n_inv": _get_inverse_param(params, inverse_keys=("tau_n_inv",), tau_keys=("tau_n",), default=0.0),
        "tau_w_inv": _get_inverse_param(params, inverse_keys=("tau_w_inv", "tau_omega_inv"), tau_keys=("tau_w", "tau_omega"), default=0.0),
        "tau_pe_inv": _get_inverse_param(params, inverse_keys=("tau_pe_inv",), tau_keys=("tau_pe",), default=0.0),
        "tau_pi_inv": _get_inverse_param(params, inverse_keys=("tau_pi_inv",), tau_keys=("tau_pi",), default=0.0),
        "tau_SH_e_inv": _get_inverse_param(params, inverse_keys=("tau_SH_e_inv", "tau_she_inv"), tau_keys=("tau_SH_e", "tau_she"), default=0.0),
        "alpha": _get_param(params, "alpha", default=0.0),
        "L_c": _get_param(params, "L_c", "lconn", default=1.0),
        "phi_s": _get_param(params, "phi_s", "bohm_potential", default=0.0),
        "T_e_s": _get_param(params, "T_e_s", "te_s", "Te0", default=1.0),
        "nu_ei": _get_param(params, "nu_ei", "nuei", default=0.0),
        "me": _get_param(params, "me", default=_ELECTRON_MASS),
        "mi": _get_param(params, "mi", default=_get_param(params, "A", default=2.0) * _PROTON_MASS),
        "y_dim": int(_get_param(params, "y_dim", default=-1)),
        "curvature_coeff": _get_param(params, "curvature_coeff", default=1.0),
    }

    hparams["phi_s"] = torch.as_tensor(hparams["phi_s"], dtype=phi.dtype, device=phi.device)
    hparams["T_e_s"] = torch.as_tensor(hparams["T_e_s"], dtype=phi.dtype, device=phi.device)
    return hparams
