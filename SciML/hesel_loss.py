from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


mse_loss = nn.MSELoss()


@dataclass
class HeselParams:
	"""Minimal parameter set used in the HESEL loss."""

	B: float
	kappa: float
	Te0: float = 1.0
	Ti0: float = 1.0
	eps: float = 1e-8


@dataclass
class IsothermalAblationParams:
	"""First ablation closure using only Lambda terms (no Sigma terms)."""

	nu_n: float = 0.0
	nu_omega: float = 0.0


def _parse_bout_value(raw: str) -> float:
	"""Parse numeric value from one BOUT.inp assignment line."""
	text = raw.split("#", 1)[0].strip()
	return float(text)


def load_hesel_params_from_bout_inp(path: str) -> HeselParams:
	"""Load B and kappa from BOUT.inp.

	If `kappa` is not explicitly present in the file, use
		kappa = 2 * rhos / Rmajor
	derived from available base parameters in [hesel].
	"""

	keys: Dict[str, float] = {}
	wanted = {
		"B0",
		"kappa",
		"Bt",
		"Rmajor",
		"Rminor",
		"Te0",
		"Ti0",
		"A",
		"Z",
		"e",
		"mp",
	}

	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line_clean = line.strip()
			if not line_clean or line_clean.startswith("#") or "=" not in line_clean:
				continue
			left, right = line_clean.split("=", 1)
			key = left.strip()
			if key in wanted:
				try:
					keys[key] = _parse_bout_value(right)
				except ValueError:
					pass

	# Build B from B0 if available, else from Bt, Rmajor, Rminor.
	if "B0" in keys:
		B = keys["B0"]
	else:
		Bt = keys.get("Bt", 1.11)
		Rmajor = keys.get("Rmajor", 0.88)
		Rminor = keys.get("Rminor", 0.225)
		B = Bt * Rmajor / (Rmajor + Rminor)

	if "kappa" in keys:
		kappa = keys["kappa"]
	else:
		e = keys.get("e", 1.60e-19)
		mp = keys.get("mp", 1.67262158e-27)
		A = keys.get("A", 2.0)
		Z = keys.get("Z", 1.0)
		Te0 = keys.get("Te0", 29.8)
		Rmajor = keys.get("Rmajor", 0.88)
		mi = A * mp
		oci = e * Z * B / mi
		cs = (e * Te0 / mi) ** 0.5
		rhos = cs / oci
		kappa = 2.0 * rhos / Rmajor
	Te0 = keys.get("Te0", 29.8)
	Ti0 = keys.get("Ti0", 29.8)
	return HeselParams(B=B, kappa=kappa, Te0=Te0, Ti0=Ti0)


def _unpack_isothermal_prediction(pred: torch.Tensor | Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Support model output as [N,2], [N,4], or tuple/list with >=2 tensors.

	For isothermal ablation we only need (n, phi). If 4 outputs are given,
	the two extra channels are ignored.
	"""
	if isinstance(pred, torch.Tensor):
		if pred.shape[-1] < 2:
			raise ValueError("Model tensor output must have at least 2 channels: (n, phi)")
		return pred[:, 0:1], pred[:, 1:2]

	pred_tuple = tuple(pred)
	if len(pred_tuple) < 2:
		raise ValueError("Model output iterable must contain at least 2 tensors: (n, phi)")
	return pred_tuple[0], pred_tuple[1]


def _first_derivative(field: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
	return torch.autograd.grad(
		field,
		var,
		grad_outputs=torch.ones_like(field),
		create_graph=True,
		retain_graph=True,
	)[0]


def _laplacian(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	fx = _first_derivative(field, x)
	fy = _first_derivative(field, y)
	fxx = _first_derivative(fx, x)
	fyy = _first_derivative(fy, y)
	return fxx + fyy


def _poisson_bracket(f: torch.Tensor, g: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	fx = _first_derivative(f, x)
	fy = _first_derivative(f, y)
	gx = _first_derivative(g, x)
	gy = _first_derivative(g, y)
	return fx * gy - fy * gx


def _curvature_operator(f: torch.Tensor, y: torch.Tensor, kappa: float) -> torch.Tensor:
	fy = _first_derivative(f, y)
	return -kappa * fy


def _material_derivative(
	f: torch.Tensor,
	phi: torch.Tensor,
	x: torch.Tensor,
	y: torch.Tensor,
	t: torch.Tensor,
	B: float,
) -> torch.Tensor:
	ft = _first_derivative(f, t)
	bracket = _poisson_bracket(phi, f, x, y)
	return ft + bracket / B


def _lambda_rhs(
	n: torch.Tensor,
	omega: torch.Tensor,
	x: torch.Tensor,
	y: torch.Tensor,
	approx: IsothermalAblationParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
	lap_n = _laplacian(n, x, y)
	lap_omega = _laplacian(omega, x, y)

	rhs_n = approx.nu_n * lap_n
	rhs_omega = approx.nu_omega * lap_omega
	return rhs_n, rhs_omega


def hesel_residuals(
	model,
	x_f: torch.Tensor,
	y_f: torch.Tensor,
	t_f: torch.Tensor,
	params: HeselParams,
	approx: IsothermalAblationParams | None = None,
):
	"""Compute PDE residuals for isothermal ablation (n and vorticity only)."""

	approx = approx or IsothermalAblationParams()

	x = x_f.clone().detach().requires_grad_(True)
	y = y_f.clone().detach().requires_grad_(True)
	t = t_f.clone().detach().requires_grad_(True)

	pred = model(x, y, t)
	n, phi = _unpack_isothermal_prediction(pred)

	n_pos = torch.clamp(n, min=params.eps)
	pe = params.Te0 * n_pos
	pi = params.Ti0 * n_pos

	# phi* is approximated as phi in first model.
	phi_star = phi
	omega = _laplacian(phi_star, x, y)

	dt_n = _material_derivative(n, phi, x, y, t, params.B)

	grad_phi_x = _first_derivative(phi_star, x)
	grad_phi_y = _first_derivative(phi_star, y)
	dt0_grad_phi_x = _material_derivative(grad_phi_x, phi, x, y, t, params.B)
	dt0_grad_phi_y = _material_derivative(grad_phi_y, phi, x, y, t, params.B)
	vorticity_lhs_term = _first_derivative(dt0_grad_phi_x, x) + _first_derivative(dt0_grad_phi_y, y)

	K_phi = _curvature_operator(phi, y, params.kappa)
	K_pe = _curvature_operator(pe, y, params.kappa)
	K_pepi = _curvature_operator(pe + pi, y, params.kappa)

	rhs_n, rhs_omega = _lambda_rhs(n, omega, x, y, approx)

	r_n = dt_n + n * K_phi - K_pe - rhs_n
	r_omega = vorticity_lhs_term - K_pepi - rhs_omega

	return r_n, r_omega


def hesel_pinn_loss(
	model,
	x_f,
	y_f,
	t_f,
	x_0,
	y_0,
	t_0,
	n_0,
	phi_0,
	pe_0,
	pi_0,
	x_b1,
	y_b1,
	t_b1,
	n_b1,
	phi_b1,
	pe_b1,
	pi_b1,
	x_b2,
	y_b2,
	t_b2,
	n_b2,
	phi_b2,
	pe_b2,
	pi_b2,
	params: HeselParams,
	approx: IsothermalAblationParams | None = None,
	w_f: float = 1.0,
	w_0: float = 1.0,
	w_b: float = 1.0,
):
	"""HESEL PINN loss with same return style as the existing `loss.py`.

	Note:
		`pe_*` and `pi_*` arguments are kept only for backwards-compatible
		call signatures. In isothermal ablation mode, the loss uses only
		(n, phi) and computes pe, pi from Te0 and Ti0 internally.

	Returns:
		total_loss, loss_pde, loss_ic, loss_bc
	"""

	r_n, r_omega = hesel_residuals(
		model=model,
		x_f=x_f,
		y_f=y_f,
		t_f=t_f,
		params=params,
		approx=approx,
	)

	loss_pde = (
		mse_loss(r_n, torch.zeros_like(r_n))
		+ mse_loss(r_omega, torch.zeros_like(r_omega))
	)

	pred_0 = model(x_0, y_0, t_0)
	n0_pred, phi0_pred = _unpack_isothermal_prediction(pred_0)
	loss_ic = (
		mse_loss(n0_pred, n_0)
		+ mse_loss(phi0_pred, phi_0)
	)

	pred_b1 = model(x_b1, y_b1, t_b1)
	n_b1_pred, phi_b1_pred = _unpack_isothermal_prediction(pred_b1)
	loss_bc1 = (
		mse_loss(n_b1_pred, n_b1)
		+ mse_loss(phi_b1_pred, phi_b1)
	)

	pred_b2 = model(x_b2, y_b2, t_b2)
	n_b2_pred, phi_b2_pred = _unpack_isothermal_prediction(pred_b2)
	loss_bc2 = (
		mse_loss(n_b2_pred, n_b2)
		+ mse_loss(phi_b2_pred, phi_b2)
	)

	loss_bc = loss_bc1 + loss_bc2

	total_loss = w_f * loss_pde + w_0 * loss_ic + w_b * loss_bc
	return total_loss, loss_pde, loss_ic, loss_bc
