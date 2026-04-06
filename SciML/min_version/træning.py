import torch
import torch.nn as nn

from PDE import pde_loss
from model import PINN
from params2 import *



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

	x, y, t = make_collocation_points(n_points=8, device=device)

	n = n_model(x, y, t)
	p_e = p_e_model(x, y, t)
	p_i = p_i_model(x, y, t)
	phi_tensor = phi_model(x, y, t)

	loss_weights = {"n": 1.0, "w": 1.0, "pe": 1.0, "pi": 1.0}
	loss_dict = pde_loss(n, p_e, p_i, phi_tensor, x, y, t, loss_weights)

	for i in loss_dict:
		print(i, loss_dict[i])

if __name__ == "__main__":
	main()