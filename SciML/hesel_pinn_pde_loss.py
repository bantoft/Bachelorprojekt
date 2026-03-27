from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


# -----------------------------
# BOUT.inp parsing
# -----------------------------

_COMMENT_SPLIT_RE = re.compile(r"(?://|#|;).*$")
_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class HeselParameters:
    # Main coefficients used directly in the PDE residuals
    B: float = 1.0
    rho_s: float = 1.0
    R: float = 1.65
    D_e: float = 0.0043
    D_i: float = 0.073
    alpha: float = 2.74e-4
    tau: float = 1.0 / 1.6e-5
    tau_SH: float = 1.0 / 9.8e-4
    tau_p: float = 1e9  # effectively off unless specified
    L_c: float = 15.0
    phi_m: float = math.log(math.sqrt(1836.15267343 / (2.0 * math.pi)))
    m_e_over_m_i: float = 1.0 / 1836.15267343

    # Optional prescribed profiles / reference values
    n_p: float = 0.0
    p_e_p: float = 0.0
    p_i_p: float = 0.0
    n0: float = 1.5e19
    T_e0_eV: float = 25.0
    T_i0_eV: float = 25.0
    q: float = 5.32
    L_b: float = 8.778
    L_parallel: float = 8.778


def _clean_value(raw: str) -> str:
    value = _COMMENT_SPLIT_RE.sub("", raw).strip()
    return value


def _parse_numeric(raw: str) -> Optional[float]:
    value = _clean_value(raw)
    if not value:
        return None
    # If the whole thing is a bare float, use it directly.
    try:
        return float(value)
    except ValueError:
        pass

    # Otherwise, extract the first numeric token. This handles e.g. "25 eV".
    match = _NUMERIC_RE.search(value)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def parse_bout_inp(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Very tolerant parser for BOUT.inp-style files.

    Returns a nested dict: {section_name: {key: parsed_or_raw_value}}.
    The parser accepts both global keys (before a [section]) and sectioned keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find BOUT input file: {path}")

    data: Dict[str, Dict[str, Any]] = {"global": {}}
    section = "global"

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            data.setdefault(section, {})
            continue
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip().lower()
        raw_value = raw_value.strip()
        parsed_value = _parse_numeric(raw_value)
        data.setdefault(section, {})[key] = parsed_value if parsed_value is not None else _clean_value(raw_value)

    return data


def _lookup(parsed: Mapping[str, Mapping[str, Any]], *aliases: str, default: Optional[float] = None) -> Optional[float]:
    aliases = tuple(a.lower() for a in aliases)
    for section_values in parsed.values():
        for alias in aliases:
            if alias in section_values:
                value = section_values[alias]
                if isinstance(value, (int, float)):
                    return float(value)
    return default


def hesel_parameters_from_bout(path: str | Path) -> HeselParameters:
    parsed = parse_bout_inp(path)

    # Pull what we can directly from the file using a generous alias list.
    R = _lookup(parsed, "r", "major_radius", default=1.65)
    a = _lookup(parsed, "a", "minor_radius", default=0.5)
    q = _lookup(parsed, "q", "q95", default=5.32)
    B = _lookup(parsed, "b", "b0", "b_0", default=1.0)
    rho_s = _lookup(parsed, "rho_s", "rhos", default=1.0)
    D_e = _lookup(parsed, "d_e", "de", "diffusion_e", default=0.0043)
    D_i = _lookup(parsed, "d_i", "di", "diffusion_i", default=0.073)
    alpha = _lookup(parsed, "alpha", default=2.74e-4)
    tau = _lookup(parsed, "tau", default=None)
    tau_inv = _lookup(parsed, "tau_inv", "taui", "1/tau", default=None)
    tau_SH = _lookup(parsed, "tau_sh", "tau^sh", default=None)
    tau_SH_inv = _lookup(parsed, "tau_sh_inv", "1/tau_sh", "1/tau^sh", default=None)
    tau_p = _lookup(parsed, "tau_p", default=1e9)
    L_c = _lookup(parsed, "l_c", "lc", "connection_length", default=15.0)
    L_b = _lookup(parsed, "l_b", "lb", default=None)
    L_parallel = _lookup(parsed, "l_parallel", "lpar", "l_||", default=None)
    n0 = _lookup(parsed, "n0", default=1.5e19)
    T_e0_eV = _lookup(parsed, "te0", "t_e0", default=25.0)
    T_i0_eV = _lookup(parsed, "ti0", "t_i0", default=25.0)
    n_p = _lookup(parsed, "n_p", "np", default=0.0)
    p_e_p = _lookup(parsed, "p_e_p", "pep", default=0.0)
    p_i_p = _lookup(parsed, "p_i_p", "pip", default=0.0)

    if tau is None and tau_inv is not None and tau_inv != 0.0:
        tau = 1.0 / tau_inv
    if tau is None:
        tau = 1.0 / 1.6e-5

    if tau_SH is None and tau_SH_inv is not None and tau_SH_inv != 0.0:
        tau_SH = 1.0 / tau_SH_inv
    if tau_SH is None:
        tau_SH = 1.0 / 9.8e-4

    if L_b is None:
        L_b = q * R
    if L_parallel is None:
        L_parallel = L_b

    phi_m = math.log(math.sqrt(1836.15267343 / (2.0 * math.pi)))

    return HeselParameters(
        B=B,
        rho_s=rho_s,
        R=R,
        D_e=D_e,
        D_i=D_i,
        alpha=alpha,
        tau=tau,
        tau_SH=tau_SH,
        tau_p=tau_p,
        L_c=L_c,
        phi_m=phi_m,
        m_e_over_m_i=1.0 / 1836.15267343,
        n_p=n_p,
        p_e_p=p_e_p,
        p_i_p=p_i_p,
        n0=n0,
        T_e0_eV=T_e0_eV,
        T_i0_eV=T_i0_eV,
        q=q,
        L_b=L_b,
        L_parallel=L_parallel,
    )


# -----------------------------
# PINN PDE loss
# -----------------------------

def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def _mean_over_y(field: torch.Tensor) -> torch.Tensor:
    # field shape: [Nt, Nx, Ny]
    return field.mean(dim=2, keepdim=True)


def _laplacian(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dfdx = _grad(field, x)
    dfdy = _grad(field, y)
    d2fdx2 = _grad(dfdx, x)
    d2fdy2 = _grad(dfdy, y)
    return d2fdx2 + d2fdy2


def _poisson_bracket(f: torch.Tensor, g: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dfdx = _grad(f, x)
    dfdy = _grad(f, y)
    dgdx = _grad(g, x)
    dgdy = _grad(g, y)
    return dfdx * dgdy - dfdy * dgdx


def _curvature(f: torch.Tensor, y: torch.Tensor, rho_s: float, R: float) -> torch.Tensor:
    return -(rho_s / R) * _grad(f, y)


class HeselPDELoss(torch.nn.Module):
    """PDE-only loss for the 4-field HESEL system.

    Expected model output order: [n, phi, p_e, p_i].

    Important:
    The drift-wave tilde/bar terms require a poloidal average in y.
    Therefore x, y, t should be provided on a structured tensor-product grid,
    typically with shape [Nt, Nx, Ny].
    """

    def __init__(self, params: HeselParameters, eps: float = 1e-8):
        super().__init__()
        self.params = params
        self.eps = eps

    def forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        return_residuals: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must have identical shapes, e.g. [Nt, Nx, Ny].")
        if x.ndim != 3:
            raise ValueError("Use structured collocation tensors of shape [Nt, Nx, Ny].")

        shape = x.shape
        x_f = x.reshape(-1, 1).clone().detach().requires_grad_(True)
        y_f = y.reshape(-1, 1).clone().detach().requires_grad_(True)
        t_f = t.reshape(-1, 1).clone().detach().requires_grad_(True)
        coords = torch.cat([x_f, y_f, t_f], dim=1)

        pred = model(coords)
        if pred.shape[-1] != 4:
            raise ValueError("Model must output 4 channels in the order [n, phi, p_e, p_i].")

        n = pred[:, 0:1]
        phi = pred[:, 1:2]
        p_e = pred[:, 2:3]
        p_i = pred[:, 3:4]

        T_e = p_e / (n + self.eps)
        T_i = p_i / (n + self.eps)

        # Fluctuating and mean parts along poloidal direction y
        n_g = n.reshape(*shape, 1).squeeze(-1)
        phi_g = phi.reshape(*shape, 1).squeeze(-1)
        T_e_g = T_e.reshape(*shape, 1).squeeze(-1)
        p_e_g = p_e.reshape(*shape, 1).squeeze(-1)
        p_i_g = p_i.reshape(*shape, 1).squeeze(-1)

        n_bar = _mean_over_y(n_g).expand(-1, -1, shape[2]).reshape(-1, 1)
        phi_bar = _mean_over_y(phi_g).expand(-1, -1, shape[2]).reshape(-1, 1)
        T_e_bar = _mean_over_y(T_e_g).expand(-1, -1, shape[2]).reshape(-1, 1)

        n_tilde = n - n_bar
        phi_tilde = phi - phi_bar
        T_e_tilde = T_e - T_e_bar

        # Basic operators
        lap_phi = _laplacian(phi, x_f, y_f)
        lap_p_i = _laplacian(p_i, x_f, y_f)
        omega_star = lap_phi + lap_p_i
        lap_n = _laplacian(n, x_f, y_f)
        lap_pe = _laplacian(p_e, x_f, y_f)
        lap_pi = _laplacian(p_i, x_f, y_f)
        lap_omega_star = _laplacian(omega_star, x_f, y_f)
        lap_Te = _laplacian(T_e, x_f, y_f)
        dndx = _grad(n, x_f)
        dndy = _grad(n, y_f)
        dTedx = _grad(T_e, x_f)
        dTedy = _grad(T_e, y_f)
        div_n_grad_Te = dndx * dTedx + dndy * dTedy + n * lap_Te

        # Time derivatives
        n_t = _grad(n, t_f)
        omega_star_t = _grad(omega_star, t_f)
        p_e_t = _grad(p_e, t_f)
        p_i_t = _grad(p_i, t_f)

        # Advection terms
        bracket_phi_n = _poisson_bracket(phi, n, x_f, y_f)
        bracket_phi_omega = _poisson_bracket(phi, omega_star, x_f, y_f)
        bracket_phi_pe = _poisson_bracket(phi, p_e, x_f, y_f)
        bracket_phi_pi = _poisson_bracket(phi, p_i, x_f, y_f)
        bracket_phi_pi_simple = _poisson_bracket(phi, p_i, x_f, y_f)

        dt_n = n_t + (1.0 / self.params.B) * bracket_phi_n
        dt0_omega = omega_star_t + bracket_phi_omega
        dt_pe = p_e_t + (1.0 / self.params.B) * bracket_phi_pe
        dt_pi = p_i_t + (1.0 / self.params.B) * bracket_phi_pi

        # Curvature operator terms
        K_phi = _curvature(phi, y_f, self.params.rho_s, self.params.R)
        K_pe = _curvature(p_e, y_f, self.params.rho_s, self.params.R)
        K_pe_pi = _curvature(p_e + p_i, y_f, self.params.rho_s, self.params.R)
        K_pe2_over_n = _curvature((p_e ** 2) / (n + self.eps), y_f, self.params.rho_s, self.params.R)
        K_pi2_over_n = _curvature((p_i ** 2) / (n + self.eps), y_f, self.params.rho_s, self.params.R)

        # Auxiliary closures
        Theta = 3.0 * self.params.m_e_over_m_i * (p_e - p_i)
        drift_wave = self.params.alpha * (
            T_e_tilde + (T_e_tilde / (n_bar + self.eps)) * n_tilde - phi_tilde
        )

        # Sheath closure. If your BOUT implementation uses a different formula,
        # adjust this one line.
        sheath = (self.params.rho_s / self.params.L_c) * (
            1.0 - torch.exp(self.params.phi_m - phi / (T_e + self.eps))
        )

        # Lambda terms from your TeX file
        Lambda_n = (
            self.params.D_e * lap_n
            - n / self.params.tau
            - (n - self.params.n_p) / self.params.tau_p
            - drift_wave
        )

        Lambda_omega_star = (
            self.params.D_i * lap_omega_star
            - omega_star / self.params.tau
            + sheath
            - drift_wave
        )

        Lambda_pe = (
            2.5 * self.params.D_e * lap_pe
            + (16.0 / 6.0 - 2.5) * div_n_grad_Te
            - 4.5 * p_e / self.params.tau
            - T_e / self.params.tau_SH
            - Theta
            - (p_e - self.params.p_e_p) / self.params.tau_p
            - T_e_bar * drift_wave
        )

        Lambda_pi = (
            self.params.D_i * lap_pi
            - self.params.D_i * T_i * lap_n
            - 4.5 * p_i / self.params.tau
            + Theta
            - (p_i - self.params.p_i_p) / self.params.tau_p
            + p_i * Lambda_omega_star
        )

        # PDE residuals
        res_n = dt_n + n * K_phi - K_pe - Lambda_n
        res_omega = dt0_omega + bracket_phi_pi_simple - K_pe_pi - Lambda_omega_star
        res_pe = 1.5 * dt_pe + 2.5 * p_e * K_phi - 2.5 * K_pe2_over_n - Lambda_pe
        res_pi = 1.5 * dt_pi + 2.5 * p_i * K_phi + 2.5 * K_pi2_over_n - p_i * K_pe_pi - Lambda_pi

        loss = (
            (res_n ** 2).mean()
            + (res_omega ** 2).mean()
            + (res_pe ** 2).mean()
            + (res_pi ** 2).mean()
        )

        if return_residuals:
            residuals = {
                "res_n": res_n.reshape(shape),
                "res_omega_star": res_omega.reshape(shape),
                "res_p_e": res_pe.reshape(shape),
                "res_p_i": res_pi.reshape(shape),
                "omega_star": omega_star.reshape(shape),
            }
            return loss, residuals
        return loss


# -----------------------------
# Convenience helper
# -----------------------------

def build_hesel_pde_loss(bout_inp_path: str | Path) -> HeselPDELoss:
    params = hesel_parameters_from_bout(bout_inp_path)
    return HeselPDELoss(params)


if __name__ == "__main__":
    # Example usage
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 4),
            )

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return self.net(coords)

    params = HeselParameters()
    loss_fn = HeselPDELoss(params)
    model = DummyNet()

    Nt, Nx, Ny = 2, 4, 8
    x = torch.linspace(0.0, 1.0, Nx)
    y = torch.linspace(0.0, 1.0, Ny)
    t = torch.linspace(0.0, 1.0, Nt)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")

    loss, residuals = loss_fn(model, xx, yy, tt, return_residuals=True)
    print("loss =", float(loss.detach().cpu()))
    print("loaded params =", asdict(params))
    print("residual keys =", sorted(residuals.keys()))
