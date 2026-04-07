import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from itertools import chain

from loss_PDE import pde_loss, pde_losses
from data_loader import make_data_loader, ids_to_inputs
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
	results_dir = Path(__file__).with_name("results")
	results_dir.mkdir(parents=True, exist_ok=True)
	history_path = results_dir / "training_history.pt"

	n_model = build_model(device)
	p_e_model = build_model(device)
	p_i_model = build_model(device)
	phi_model = build_model(device)

	optimizer = optim.Adam(
		chain(
			n_model.parameters(),
			p_e_model.parameters(),
			p_i_model.parameters(),
			phi_model.parameters(),
		),
		lr=1e-2,
	)

	w_pde = 1.0
	w_data = 1.0
	batch_size_data = 128
	n_epochs = 30
	data_loader = make_data_loader(batch_size=batch_size_data, shuffle=True)
	history = {
		"epoch": [],
		"total_loss": [],
		"pde_loss": [],
		"data_loss_total": [],
		"models": {
			"n": {"data_loss": []},
			"p_e": {"data_loss": []},
			"p_i": {"data_loss": []},
			"phi": {"data_loss": []},
		},
		"pde_equations": {
			"R_n": [],
			"R_w": [],
			"R_pe": [],
			"R_pi": [],
		},
	}

	for epoch in range(1, n_epochs + 1):
		optimizer.zero_grad()

		x_col, y_col, t_col = make_collocation_points(n_points=256, device=device)
		n_col = n_model(x_col, y_col, t_col)
		p_e_col = p_e_model(x_col, y_col, t_col)
		p_i_col = p_i_model(x_col, y_col, t_col)
		phi_col = phi_model(x_col, y_col, t_col)
		loss_pde = pde_loss(n_col, p_e_col, p_i_col, phi_col, x_col, y_col, t_col)
		losses_pde = pde_losses(n_col, p_e_col, p_i_col, phi_col, x_col, y_col, t_col)

		x_ids, y_ids, t_ids, lnn_target, lnpe_target, lnpi_target, vort_target = next(iter(data_loader))
		x_data, y_data, t_data = ids_to_inputs(x_ids, y_ids, t_ids)

		x_data = x_data.to(device)
		y_data = y_data.to(device)
		t_data = t_data.to(device)
		lnn_target = lnn_target.to(device).unsqueeze(1)
		lnpe_target = lnpe_target.to(device).unsqueeze(1)
		lnpi_target = lnpi_target.to(device).unsqueeze(1)
		vort_target = vort_target.to(device).unsqueeze(1)

		n_data_pred = n_model(x_data, y_data, t_data)
		p_e_data_pred = p_e_model(x_data, y_data, t_data)
		p_i_data_pred = p_i_model(x_data, y_data, t_data)
		phi_data_pred = phi_model(x_data, y_data, t_data)

		loss_n_data = F.mse_loss(n_data_pred, lnn_target)
		loss_p_e_data = F.mse_loss(p_e_data_pred, lnpe_target)
		loss_p_i_data = F.mse_loss(p_i_data_pred, lnpi_target)
		loss_phi_data = F.mse_loss(phi_data_pred, vort_target)

		loss_data_total = (
			loss_n_data
			+ loss_p_e_data
			+ loss_p_i_data
			+ loss_phi_data
		)

		loss_total = w_pde * loss_pde + w_data * loss_data_total
		loss_total.backward()
		optimizer.step()

		history["epoch"].append(epoch)
		history["total_loss"].append(loss_total.detach().cpu().item())
		history["pde_loss"].append(loss_pde.detach().cpu().item())
		history["data_loss_total"].append(loss_data_total.detach().cpu().item())
		history["models"]["n"]["data_loss"].append(loss_n_data.detach().cpu().item())
		history["models"]["p_e"]["data_loss"].append(loss_p_e_data.detach().cpu().item())
		history["models"]["p_i"]["data_loss"].append(loss_p_i_data.detach().cpu().item())
		history["models"]["phi"]["data_loss"].append(loss_phi_data.detach().cpu().item())
		history["pde_equations"]["R_n"].append(losses_pde["R_n"].detach().cpu().item())
		history["pde_equations"]["R_w"].append(losses_pde["R_w"].detach().cpu().item())
		history["pde_equations"]["R_pe"].append(losses_pde["R_pe"].detach().cpu().item())
		history["pde_equations"]["R_pi"].append(losses_pde["R_pi"].detach().cpu().item())

		print(
			f"epoch={epoch:03d} total={loss_total.item():.6e} "
			f"pde={loss_pde.item():.6e} data={loss_data_total.item():.6e} "
			f"Rn={losses_pde['R_n'].item():.6e} Rw={losses_pde['R_w'].item():.6e} "
			f"Rpe={losses_pde['R_pe'].item():.6e} Rpi={losses_pde['R_pi'].item():.6e}"
		)

	torch.save(history, history_path)
	print(f"saved training history to {history_path}")

if __name__ == "__main__":
	main()