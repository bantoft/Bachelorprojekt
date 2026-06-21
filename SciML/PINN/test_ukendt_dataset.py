from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from SciML.PINN.util.model import PINN
from SciML.PINN.util.træning_helpers import init_trainer


CHECKPOINT_DIRS = [
    ROOT_DIR / "Experimenter/PINN/test_run",
] 

# Tom liste betyder: brug datasættene gemt i hvert checkpoint, ét ad gangen.
DATASET_ROOTS: list[Path] = [
    # ROOT_DIR / "sim_data/Alexander_pi",
    # ROOT_DIR / "sim_data/Alexander__fys_R_1",
    ROOT_DIR / "sim_data/Alexander__grid_c_0",
]
# Brug `None` for at genbruge split-opsætningen fra checkpointet.
# Hvis du tester på egne datasæt, kan du sætte denne til f.eks. `True`
# eller en liste som matcher `DATASET_ROOTS`.
DATASET_TSSPLIT: bool | list[bool] | None = None

SPLIT = "test"  # "training", "validation" eller "test"
STATE_DICT_KEY = "model_state_dict"
BATCH_SIZE_OVERRIDE: int | None = None
NUM_WORKERS_OVERRIDE: int | None = 0
MAX_BATCHES: int | None = None
PLOT_LOSSES = ["total"]
PLOT_SOURCES = ["raw"]
AVG_VINDUE = 1
SAVE_PLOT = True
SHOW_PLOT = True
OUTPUT_DIR = ROOT_DIR / "billeder" / "PINN" / "Eksperiment"

LOSS_KEYS = ("total", "da", "eq", "ic", "bc")


def _normalize_checkpoint_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    return path.resolve()


def _load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_out_folder(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT_DIR / path)


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
        train_cfg["pin_memory"] = False

    return train_cfg


def _resolve_root_groups(checkpoint_roots) -> list[list[Path]]:
    if DATASET_ROOTS:
        return [[Path(root).resolve()] for root in DATASET_ROOTS]
    return [[Path(root).resolve()] for root in checkpoint_roots]


def _resolve_tssplit(base_tssplit, num_roots: int):
    if isinstance(base_tssplit, bool):
        return [base_tssplit] * num_roots
    return list(base_tssplit)


def _resolve_custom_tssplit(index: int, fallback: bool) -> bool:
    if DATASET_TSSPLIT is None:
        return fallback
    if isinstance(DATASET_TSSPLIT, bool):
        return DATASET_TSSPLIT
    return DATASET_TSSPLIT[index]


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


def _load_model_state(model: torch.nn.Module, checkpoint: dict, state_dict_key: str):
    state_dict = checkpoint.get(state_dict_key)
    if state_dict is None:
        raise KeyError(f"Checkpoint mangler '{state_dict_key}'.")
    model.load_state_dict(state_dict)


def _evaluate_loader(loader, model, condition, phys, info, train_cfg, max_batches=None):
    was_training = model.training
    model.eval()

    totals = {key: 0.0 for key in LOSS_KEYS}
    weighted_totals = {key: 0.0 for key in LOSS_KEYS}
    num_batches = 0
    num_samples = 0
    history = {
        "raw": {key: [] for key in LOSS_KEYS},
        "weighted": {key: [] for key in LOSS_KEYS},
    }
    old_requires_grad = [param.requires_grad for param in model.parameters()]

    try:
        for param in model.parameters():
            param.requires_grad_(False)

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            avg_z, cord_fys, cord_num, input_data, target = (
                tensor.to(info.device, non_blocking=False) for tensor in batch
            )
            cord_fys = cord_fys.clone().detach().requires_grad_(True)

            state = model.forward(info.standardized, cord_fys, input_data)
            pred_da = torch.cat(list(state.values()), dim=-1)

            res_eq = phys.eq_res(avg_z, state, cord_fys, cord_num)
            res_ic = phys.ic_res(model, cord_fys, cord_num, input_data, info.standardized)
            res_bc = phys.bc_res(model, cord_fys, input_data, info.standardized)

            res_eq_stack = torch.stack(list(res_eq.values()))
            res_ic_stack = torch.stack(list(res_ic.values()))
            res_bc_stack = torch.stack(list(res_bc.values()))

            losses = {
                "da": float(condition(pred_da, target).detach().cpu()),
                "eq": float(condition(res_eq_stack, torch.zeros_like(res_eq_stack)).detach().cpu()),
                "ic": float(condition(res_ic_stack, torch.zeros_like(res_ic_stack)).detach().cpu()),
                "bc": float(condition(res_bc_stack, torch.zeros_like(res_bc_stack)).detach().cpu()),
            }
            losses["total"] = losses["da"] + losses["eq"] + losses["ic"] + losses["bc"]

            weighted = {
                "da": losses["da"] * train_cfg["da"],
                "eq": losses["eq"] * train_cfg["eq"],
                "ic": losses["ic"] * train_cfg["ic"],
                "bc": losses["bc"] * train_cfg["bc"],
            }
            weighted["total"] = (
                weighted["da"] + weighted["eq"] + weighted["ic"] + weighted["bc"]
            )

            for key in LOSS_KEYS:
                totals[key] += losses[key]
                weighted_totals[key] += weighted[key]
                history["raw"][key].append(losses[key])
                history["weighted"][key].append(weighted[key])

            num_batches += 1
            num_samples += avg_z.shape[0]

            del (
                avg_z,
                cord_fys,
                cord_num,
                input_data,
                target,
                state,
                pred_da,
                res_eq,
                res_ic,
                res_bc,
                res_eq_stack,
                res_ic_stack,
                res_bc_stack,
            )
    finally:
        for param, old_value in zip(model.parameters(), old_requires_grad):
            param.requires_grad_(old_value)
        if was_training:
            model.train()

    if num_batches == 0:
        return {
            "num_batches": 0,
            "num_samples": 0,
            "history": history,
            **{f"raw_{key}": float("inf") for key in LOSS_KEYS},
            **{f"weighted_{key}": float("inf") for key in LOSS_KEYS},
        }

    return {
        "num_batches": num_batches,
        "num_samples": num_samples,
        "history": history,
        **{f"raw_{key}": totals[key] / num_batches for key in LOSS_KEYS},
        **{f"weighted_{key}": weighted_totals[key] / num_batches for key in LOSS_KEYS},
    }


def _result_label(checkpoint_path: Path, root_group: list[Path]) -> str:
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
    checkpoint = _load_checkpoint(checkpoint_path)
    base_train_cfg = checkpoint["training_config"]
    checkpoint_roots = base_train_cfg["root"]
    base_tssplit = _resolve_tssplit(base_train_cfg.get("TSSplit", True), len(checkpoint_roots))
    root_groups = _resolve_root_groups(checkpoint_roots)

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"State dict: {STATE_DICT_KEY}")
    results = []

    for index, root_group in enumerate(root_groups):
        train_cfg = _normalize_loader_config(base_train_cfg)
        train_cfg["resume"] = False
        train_cfg["root"] = root_group

        fallback_split = base_tssplit[min(index, len(base_tssplit) - 1)]
        train_cfg["TSSplit"] = _resolve_custom_tssplit(index, fallback_split)
        train_cfg["out_folder"] = str(_resolve_out_folder(base_train_cfg["out_folder"]))

        (
            info,
            phys,
            condition,
            _model,
            _optimizer,
            train_loader,
            val_loader,
            test_loader,
        ) = init_trainer(train_cfg)

        model = PINN(checkpoint["model_structure"]).to(info.device)
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
            f"raw(total={metrics['raw_total']:.3e}, da={metrics['raw_da']:.3e}, eq={metrics['raw_eq']:.3e}, "
            f"ic={metrics['raw_ic']:.3e}, bc={metrics['raw_bc']:.3e}), "
            f"weighted(total={metrics['weighted_total']:.3e}, da={metrics['weighted_da']:.3e}, "
            f"eq={metrics['weighted_eq']:.3e}, ic={metrics['weighted_ic']:.3e}, bc={metrics['weighted_bc']:.3e})"
        )
        results.append(
            {
                "label": _result_label(checkpoint_path, root_group),
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
