from __future__ import annotations

import math
import re
import torch.nn.functional as F
import torch

from torch import Tensor
from typing import Mapping

BOUNDARY_PATTERN = re.compile(r"^(?P<kind>[A-Za-z_]\w*)(?:\((?P<expr>.*)\))?$")
DEFAULT_FIELDS = ("lnn", "lnpe", "lnpi", "phi")

def parse_literal(value: str):
    text = value.strip()
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return None

def safe_log(x: float | Tensor) -> float | Tensor:
    if torch.is_tensor(x):
        return torch.log(torch.clamp(x, min=1e-12))
    return math.log(max(float(x), 1e-12))

def normalize_ids(
    x_ids: torch.Tensor,
    z_ids: torch.Tensor,
    t_ids: torch.Tensor,
    nt: int,
    nx: int,
    nz: int,
):
    return (
        (x_ids.float() / max(nx - 1, 1)).unsqueeze(1),
        (z_ids.float() / max(nz - 1, 1)).unsqueeze(1),
        (t_ids.float() / max(nt - 1, 1)).unsqueeze(1),
    )


def mse_dict(residuals: Mapping[str, Tensor]) -> Tensor:
    first_residual = next(iter(residuals.values()))
    total = torch.zeros((), device=first_residual.device, dtype=first_residual.dtype)
    for residual in residuals.values():
        loss = F.mse_loss(residual, torch.zeros_like(residual))
        total = total + loss
    return total
