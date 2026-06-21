from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from SciML.PINN_z.utils.training_helpers import init_experinment, load_checkpoint


CHECKPOINT_DIRS = [
    ROOT_DIR / "Experimenter/PINN_z/Eksperiment_1",
    # ROOT_DIR / "Experimenter/PINN_z/Eksperiment_2",
    # ROOT_DIR / "Experimenter/PINN_z/Eksperiment_3",
]

# Tom liste betyder: brug datasættene gemt i hvert checkpoint, ét ad gangen.
# Hvis du vil teste på andre datasæt, så læg dem ind som enkelte mapper herunder.
DATASET_ROOTS: list[Path] = [
    ROOT_DIR / "sim_data/Alexander_pi",
    # ROOT_DIR / "sim_data/Alexander__fys_R_1",
    # ROOT_DIR / "sim_data/Alexander__t_n",
]

SPLIT = "test"  # "training", "validation" eller "test"
STATE_DICT_KEY = "best_val_model"  # eller "model_state_dict"
BATCH_SIZE_OVERRIDE: int | None = None
NUM_WORKERS_OVERRIDE: int | None = 0
MAX_BATCHES: int | None = None
# Vælg selv mellem "total", "da" og "eq".
PLOT_LOSSES = ["da", "eq"]
PLOT_SOURCES = ["raw"]
AVG_VINDUE = 1
SAVE_PLOT = True
SHOW_PLOT = True
OUTPUT_DIR = ROOT_DIR / "billeder" / "PINN_z" / "Eksperiment"


def _strip_state_dict_metadata(state_dict):
    if state_dict is None:
        return None
    if "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)
    return state_dict


def _load_model_state(model: torch.nn.Module, checkpoint: dict, state_dict_key: str):
    state_dict = checkpoint.get(state_dict_key)
    if state_dict is None and state_dict_key != "model_state_dict":
        state_dict = checkpoint.get("model_state_dict")

    state_dict = _strip_state_dict_metadata(state_dict)
    if state_dict is None:
        raise KeyError(
            f"Checkpoint mangler både '{state_dict_key}' og 'model_state_dict'."
        )

    model.load_state_dict(state_dict)


def _normalize_checkpoint_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    return path.resolve()


def _normalize_loader_config(train_cfg: dict) -> dict:
    train_cfg = copy.deepcopy(train_cfg)

    if BATCH_SIZE_OVERRIDE is not None:
        train_cfg["batch_size"] = BATCH_SIZE_OVERRIDE

    if NUM_WORKERS_OVERRIDE is not None:
        train_cfg["num_workers"] = NUM_WORKERS_OVERRIDE

    if train_cfg.get("num_workers", 0) <= 0:
        train_cfg["num_workers"] = 0
        train_cfg["prefetch_factor"] = None
        train_cfg["persistent_workers"] = False

    return train_cfg


def _resolve_root_groups(checkpoint_roots) -> list[list[Path]]:
    if DATASET_ROOTS:
        return [[Path(root).resolve()] for root in DATASET_ROOTS]
    return [[Path(root).resolve()] for root in checkpoint_roots]


def _split_loader(split: str, train_loader, val_loader, test_loader):
    loaders = {
        "training": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }
    try:
        return loaders[split]
    except KeyError as exc:
        raise ValueError(f"Ugyldigt split '{split}'. Vælg mellem {tuple(loaders)}.") from exc


def _evaluate_loader(loader, model, condition, phys, info, train_cfg, max_batches=None):
    was_training = model.training
    model.eval()

    total_data_loss = 0.0
    total_eq_loss = 0.0
    total_weighted_loss = 0.0
    num_batches = 0
    num_samples = 0
    history = {
        "raw": {"total": [], "da": [], "eq": []},
        "weighted": {"total": [], "da": [], "eq": []},
    }
    old_requires_grad = [param.requires_grad for param in model.parameters()]

    try:
        for param in model.parameters():
            param.requires_grad_(False)

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = [tensor.to(info.device, non_blocking=True) for tensor in batch]
            f_theta, coord_fys = model(batch, requires_coord_grad=True)
            _, _, avg_z, _, coords_num = batch

            pred_da = torch.cat([f_theta[name] for name in ("lnn", "lnpe", "lnpi", "phi")], dim=1)
            center_idx = pred_da.shape[-1] // 2
            pred_da_center = pred_da[:, :, :, center_idx : center_idx + 1]

            data_loss = condition(pred_da_center, batch[1])
            eq_res = phys.eq_res(avg_z, f_theta, coord_fys, coords_num)
            eq_res = torch.stack(list(eq_res.values()))
            eq_loss = condition(eq_res, torch.zeros_like(eq_res))

            data_loss_value = float(data_loss.detach().cpu())
            eq_loss_value = float(eq_loss.detach().cpu())
            weighted_loss_value = (
                data_loss_value * train_cfg["da_weight"]
                + eq_loss_value * train_cfg["eq_weight"]
            )

            total_data_loss += data_loss_value
            total_eq_loss += eq_loss_value
            total_weighted_loss += weighted_loss_value
            num_batches += 1
            num_samples += batch[0].shape[0]
            history["raw"]["da"].append(data_loss_value)
            history["raw"]["eq"].append(eq_loss_value)
            history["raw"]["total"].append(data_loss_value + eq_loss_value)
            history["weighted"]["da"].append(data_loss_value * train_cfg["da_weight"])
            history["weighted"]["eq"].append(eq_loss_value * train_cfg["eq_weight"])
            history["weighted"]["total"].append(weighted_loss_value)

            del f_theta, coord_fys, pred_da, pred_da_center, data_loss, eq_res, eq_loss, batch
    finally:
        for param, old_value in zip(model.parameters(), old_requires_grad):
            param.requires_grad_(old_value)
        if was_training:
            model.train()

    if num_batches == 0:
        return {
            "num_batches": 0,
            "num_samples": 0,
            "raw_total": float("inf"),
            "raw_da": float("inf"),
            "raw_eq": float("inf"),
            "weighted_total": float("inf"),
            "weighted_da": float("inf"),
            "weighted_eq": float("inf"),
            "history": history,
        }

    avg_data_loss = total_data_loss / num_batches
    avg_eq_loss = total_eq_loss / num_batches
    return {
        "num_batches": num_batches,
        "num_samples": num_samples,
        "raw_total": avg_data_loss + avg_eq_loss,
        "raw_da": avg_data_loss,
        "raw_eq": avg_eq_loss,
        "weighted_total": total_weighted_loss / num_batches,
        "weighted_da": avg_data_loss * train_cfg["da_weight"],
        "weighted_eq": avg_eq_loss * train_cfg["eq_weight"],
        "history": history,
    }


def _result_label(checkpoint_path: Path, root_group: list[Path], checkpoint_roots) -> str:
    _ = checkpoint_roots
    return f"{checkpoint_path.parent.name} {root_group[0].name}"


def _normalize_losses(losses):
    return [losses] if isinstance(losses, str) else list(losses)


def plot_test_experiment(results, losses, source="raw"):
    losses = _normalize_losses(losses)
    sources = [source] if isinstance(source, str) else list(source)
    kernel = np.ones(AVG_VINDUE, dtype=float) / AVG_VINDUE

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    for result in results:
        history = result["history"]
        for src in sources:
            for loss in losses:
                values = np.asarray(history[src][loss], dtype=float)
                if values.size == 0:
                    continue
                values = np.convolve(values, kernel, mode="same")
                values = np.log10(np.clip(values, 1e-30, None))
                steps = np.arange(1, len(values) + 1)
                label = result["label"] if len(losses) == 1 else f"{result['label']} {loss}"
                ax.plot(steps, values, label=label, alpha=0.8)

    ax.set(title="Test", ylabel="Log_10 Loss")
    ax.set_title("Test", pad=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Batches")
    fig.tight_layout(h_pad=2.0)

    if SAVE_PLOT:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figure_name = f"{'_'.join(losses)}_{'_'.join(sources)}.png"
        fig.savefig(OUTPUT_DIR / figure_name, dpi=300, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def evaluate_checkpoint(checkpoint_path: Path):
    checkpoint_stub = {
        "data_dir": checkpoint_path.parent,
    }
    checkpoint = load_checkpoint(checkpoint_stub)
    base_train_cfg = checkpoint["training_config"]
    root_groups = _resolve_root_groups(base_train_cfg["root"])

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"State dict: {STATE_DICT_KEY}")
    results = []

    for root_group in root_groups:
        train_cfg = _normalize_loader_config(base_train_cfg)
        train_cfg["resume"] = False
        train_cfg["root"] = root_group
        train_cfg["data_dir"] = checkpoint_path.parent

        print("Datasæt:", ", ".join(str(root) for root in root_group))

        (
            model,
            train_loader,
            val_loader,
            test_loader,
            condition,
            _optimizer,
            _history,
            phys,
            info,
        ) = init_experinment(train_cfg)

        _load_model_state(model, checkpoint, STATE_DICT_KEY)
        loader = _split_loader(SPLIT, train_loader, val_loader, test_loader)
        metrics = _evaluate_loader(
            loader=loader,
            model=model,
            condition=condition,
            phys=phys,
            info=info,
            train_cfg=train_cfg,
            max_batches=MAX_BATCHES,
        )

        print(
            f"{SPLIT}: batches={metrics['num_batches']}, samples={metrics['num_samples']}, "
            f"raw(total={metrics['raw_total']:.3e}, da={metrics['raw_da']:.3e}, eq={metrics['raw_eq']:.3e}), "
            f"weighted(total={metrics['weighted_total']:.3e}, da={metrics['weighted_da']:.3e}, eq={metrics['weighted_eq']:.3e})"
        )
        results.append(
            {
                "label": _result_label(checkpoint_path, root_group, base_train_cfg["root"]),
                "history": metrics["history"],
                "metrics": metrics,
            }
        )

    return results


def main():
    plot_results = []
    for checkpoint_dir in CHECKPOINT_DIRS:
        checkpoint_path = _normalize_checkpoint_path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Kunne ikke finde checkpoint: {checkpoint_path}")
        plot_results.extend(evaluate_checkpoint(checkpoint_path))

    if plot_results:
        plot_test_experiment(plot_results, losses=PLOT_LOSSES, source=PLOT_SOURCES)


if __name__ == "__main__":
    main()
