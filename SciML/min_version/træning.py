import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from loss_PDE import pde_loss, pde_losses
from data_loader import make_data_loader, ids_to_inputs
from model import PINN
from params2 import *

NN_STRUCTURE = {
    "input_size": 3,
    "output_size": 4,
    "hidden_layers": [64, 64, 64, 64],
    "activation": [nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.Tanh()],
    "dropout": [0.0, 0.0, 0.0, 0.0],
	"batch_norm": [None, None, None, None],
}


def make_collocation_points(n_points: int, device: torch.device):
	x = torch.rand(n_points, 1, device=device, requires_grad=True)
	y = torch.rand(n_points, 1, device=device, requires_grad=True)
	t = torch.rand(n_points, 1, device=device, requires_grad=True)
	return x, y, t


def build_model(device: torch.device) -> PINN:
	return PINN(NN_STRUCTURE).to(device)


def split_outputs(prediction: torch.Tensor):
	return prediction.split(1, dim=1)


def plot_total_loss(history: dict, outdir: Path) -> Path:
	epochs = history.get("epoch", [])
	total_loss = history.get("total_loss", [])

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(epochs, total_loss, linewidth=2.0, label="total_loss")
	ax.set_title("Total training loss")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_yscale("log")
	ax.grid(True, which="both", alpha=0.25)
	ax.legend(frameon=False)
	fig.tight_layout()

	plot_path = outdir / "training_total_loss.png"
	fig.savefig(plot_path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	return plot_path


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	results_dir = Path(__file__).with_name("results")
	results_dir.mkdir(parents=True, exist_ok=True)
	plots_dir = results_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)
	history_path = results_dir / "training_history.pt"

	model = build_model(device)

	optimizer = optim.Adam(
		model.parameters(),
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
		pred_col = model(x_col, y_col, t_col)
		n_col, p_e_col, p_i_col, phi_col = split_outputs(pred_col)
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

		pred_data = model(x_data, y_data, t_data)
		n_data_pred, p_e_data_pred, p_i_data_pred, phi_data_pred = split_outputs(pred_data)

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
	total_loss_plot_path = plot_total_loss(history, plots_dir)
	print(f"saved training history to {history_path}")
	print(f"saved total loss plot to {total_loss_plot_path}")

if __name__ == "__main__":
	main()