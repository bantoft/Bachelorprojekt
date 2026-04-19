from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


MODEL_ORDER = ["n", "p_e", "p_i", "phi"]
PDE_ORDER = ["R_n", "R_w", "R_pe", "R_pi"]
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LEGACY_HISTORY_PATH = BASE_DIR / "training_history.pt"
DEFAULT_HISTORY_PATH = RESULTS_DIR / "training_history.pt"


def load_history(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Could not find training history file: {path}")
	return torch.load(path, map_location="cpu")


def load_history_with_fallback(path: Path) -> dict[str, Any]:
	if path.exists():
		return load_history(path)
	if LEGACY_HISTORY_PATH.exists():
		return load_history(LEGACY_HISTORY_PATH)
	raise FileNotFoundError(f"Could not find training history file: {path}")


def to_float_list(values: Any) -> list[float]:
	if values is None:
		return []
	return [float(value) for value in values]


def build_epoch_axis(history: dict[str, Any]) -> list[int]:
	epochs = history.get("epoch", [])
	if epochs:
		return [int(epoch) for epoch in epochs]
	length = len(history.get("total_loss", []))
	return list(range(1, length + 1))


def plot_model_data_losses(history: dict[str, Any], outdir: Path) -> Path:
	epochs = build_epoch_axis(history)
	models = history.get("models", {})

	fig, ax = plt.subplots(figsize=(10, 6))
	for model_name in MODEL_ORDER:
		model_history = models.get(model_name, {})
		data_loss = to_float_list(model_history.get("data_loss", []))
		if data_loss:
			ax.plot(epochs, data_loss, linewidth=2.0, label=model_name)

	ax.set_title("Model comparison by data loss")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Data loss")
	ax.set_yscale("log")
	ax.grid(True, which="both", alpha=0.25)
	ax.legend(title="Model", frameon=False)
	fig.tight_layout()

	output_path = outdir / "training_models_data_loss.png"
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	return output_path


def plot_pde_equation_losses(history: dict[str, Any], outdir: Path) -> Path:
	epochs = build_epoch_axis(history)
	pde_equations = history.get("pde_equations", {})

	fig, ax = plt.subplots(figsize=(10, 6))
	has_series = False
	for equation_name in PDE_ORDER:
		loss_values = to_float_list(pde_equations.get(equation_name, []))
		if loss_values:
			ax.plot(epochs, loss_values, linewidth=2.0, label=equation_name)
			has_series = True

	if not has_series:
		legacy_pde_loss = to_float_list(history.get("pde_loss", []))
		if legacy_pde_loss:
			ax.plot(epochs, legacy_pde_loss, linewidth=2.0, label="PDE total")
			has_series = True

	title = "PDE equation comparison" if has_series and pde_equations else "PDE loss"
	ax.set_title(title)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Residual loss")
	ax.set_yscale("log")
	ax.grid(True, which="both", alpha=0.25)
	if has_series:
		legend_title = "Equation" if pde_equations else "Loss"
		ax.legend(title=legend_title, frameon=False)
	fig.tight_layout()

	output_path = outdir / "training_pde_losses.png"
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize saved training losses.")
	parser.add_argument(
		"--history",
		type=Path,
		default=DEFAULT_HISTORY_PATH,
		help="Path to the saved training_history.pt file.",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=PLOTS_DIR,
		help="Directory where the plots will be saved.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the figures interactively after saving them.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	outdir = args.outdir
	outdir.mkdir(parents=True, exist_ok=True)

	history = load_history_with_fallback(args.history)
	models_path = plot_model_data_losses(history, outdir)
	pde_path = plot_pde_equation_losses(history, outdir)

	print(f"Saved {models_path}")
	print(f"Saved {pde_path}")

	if args.show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()
