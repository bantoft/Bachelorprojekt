from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import json

import torch


def get_loss(model, info, phys, avg_z, cord_fys, cord_num, input, target, condition, training_config, history, type):
    with torch.enable_grad():
        cord_fys = cord_fys.clone().detach().requires_grad_(True)

        state = model(info.standardized, cord_fys, input)

        pred_da = torch.cat([state["lnn"], state["lnpe"], state["lnpi"], state["phi"]], dim=1)

        res_eq = phys.eq_res(avg_z, state, cord_fys, cord_num)
        res_ic = phys.ic_res(model, cord_fys, cord_num, input, info.standardized)
        res_bc = phys.bc_res(model, cord_fys, input, info.standardized)

        res_ic_stack = torch.stack(list(res_ic.values()))
        res_eq_stack = torch.stack(list(res_eq.values()))
        res_bc_stack = torch.stack(list(res_bc.values()))

        losses = {
            "da": condition(pred_da, target),
            "ic": condition(res_ic_stack, torch.zeros_like(res_ic_stack)),
            "eq": condition(res_eq_stack, torch.zeros_like(res_eq_stack)),
            "bc": condition(res_bc_stack, torch.zeros_like(res_bc_stack)),
        }

        losses["total"] = sum(losses.values())

        weighted_loss = {name: training_config[name] * loss for name, loss in losses.items() if name != "total" }
        weighted_loss["total"] = torch.stack(list(weighted_loss.values())).sum()
        weighted_loss = {name: training_config[name] * loss for name, loss in losses.items() if name != "total" }
        weighted_loss["total"] = torch.stack(list(weighted_loss.values())).sum()

        history.append(losses, "raw", type)
        history.append(weighted_loss, "weighted", type)

        return weighted_loss["total"]



@dataclass
class HistoryWriter:
    filename: Path
    buffer: list[dict[str, object]] | None = None

    def append(self, losses: dict[str, torch.Tensor], kind: str, split: str) -> None:
        if self.buffer is None:
            self.buffer = []

        record = {
            "kind": kind,
            "split": split,
            "values": {
                "total": losses["total"].detach().cpu().item(),
                "da": losses["da"].detach().cpu().item(),
                "ic": losses["ic"].detach().cpu().item(),
                "eq": losses["eq"].detach().cpu().item(),
                "bc": losses["bc"].detach().cpu().item(),
            },
        }

        self.buffer.append(record)

    def flush(self) -> None:
        if not self.buffer:
            return

        with open(self.filename, "a") as file_handle:
            for record in self.buffer:
                json.dump(record, file_handle)
                file_handle.write("\n")

        self.buffer.clear()


def save_checkpoint(model, optimizer, epoch: int, batch_idx: int, best_val_loss: float, training_config: dict, filename: Path, history=None, ):
    import numpy as np
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "training_config": training_config,
        # Convert RNG state ByteTensor to bytes to preserve it during serialization
        "torch_rng_state": torch.get_rng_state().cpu().numpy().tobytes(),
    }

    if history is not None:
        ckpt["history"] = {
            "filename": str(history.filename) if hasattr(history, "filename") else None,
        }

    if torch.cuda.is_available():
        try:
            # Convert CUDA RNG states to list of bytes to preserve during serialization
            ckpt["cuda_rng_state_all"] = [s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            pass

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filename)