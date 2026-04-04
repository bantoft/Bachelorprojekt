from pathlib import Path
import sys

if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from SciML.core.losses import pde_loss
from SciML.core.model import PINN
from SciML.core.params import load_params_from_bout_inp, resolve_hesel_params


BOUT_INP_PATH = (
	Path(__file__).resolve().parent.parent
	/ "Hesel_simuleringer"
	/ "BOUT-HESEL"
	/ "data_isoterm"
	/ "BOUT.inp"
)

NN_STRUCTURE = {
	"input_size": 3,
	"output_size": 1,
	"hidden_layers": [50, 50, 50],
	"activation": [nn.Tanh(), nn.ReLU(), nn.Sigmoid()],
	"dropout": [0.1, 0.2, nn.AlphaDropout(0.2)],
	"batch_norm": ["batch_norm", "batch_norm", None],
}


def make_collocation_points(n_points: int, device: torch.device):
	x = torch.rand(n_points, 1, device=device, requires_grad=True)
	y = torch.rand(n_points, 1, device=device, requires_grad=True)
	t = torch.rand(n_points, 1, device=device, requires_grad=True)
	return x, y, t


def build_model(device: torch.device) -> PINN:
	return PINN(NN_STRUCTURE).to(device)


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	n_model = build_model(device)
	p_e_model = build_model(device)
	p_i_model = build_model(device)
	phi_model = build_model(device)

	params = load_params_from_bout_inp(BOUT_INP_PATH)
	x, y, t = make_collocation_points(n_points=1024, device=device)

	n = n_model(x, y, t)
	p_e = p_e_model(x, y, t)
	p_i = p_i_model(x, y, t)
	phi_tensor = phi_model(x, y, t)

	hparams = resolve_hesel_params(params, phi_tensor)
	loss_dict = pde_loss(n, p_e, p_i, phi_tensor, x, y, t, hparams, params_are_resolved=True)

	print(f"Loaded BOUT.inp from: {BOUT_INP_PATH}")
	print(f"Resolved B-field: {hparams}")
	print(f"Total PDE loss: {loss_dict}")

if __name__ == "__main__":
	main()