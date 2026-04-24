from __future__ import annotations


import math
import re
import torch

import torch.nn.functional as F

from torch import Tensor
from pathlib import Path
from typing import Callable, Mapping



DEFAULT_BOUT_HESEL_ROOT = (Path(__file__).resolve().parents[2] / "simulatorer" / "BOUT" / "BOUT-HESEL")
REF_PATTERN = re.compile(r"\b([A-Za-z_]\w*):([A-Za-z_]\w*)\b")
BOUNDARY_PATTERN = re.compile(r"^(?P<kind>[A-Za-z_]\w*)(?:\((?P<expr>.*)\))?$")
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\b")
DEFAULT_FIELDS = ("lnn", "lnpe", "lnpi", "vort")


def read_bout_inp(path):
    section = "root"
    data = {section: {}}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            data[section][k.strip()] = v.strip()
    return data



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

def safe_sqrt(x: float | Tensor) -> float | Tensor:
    if torch.is_tensor(x):
        return torch.sqrt(torch.clamp(x, min=1e-12))
    return math.sqrt(max(float(x), 1e-12))

def safe_log(x: float | Tensor) -> float | Tensor:
    if torch.is_tensor(x):
        return torch.log(torch.clamp(x, min=1e-12))
    return math.log(max(float(x), 1e-12))

def torch_or_math_unary(
    tensor_fn: Callable[[Tensor], Tensor],
    math_fn: Callable[[float], float],
) -> Callable[[float | Tensor], float | Tensor]:
    def wrapper(x: float | Tensor) -> float | Tensor:
        if torch.is_tensor(x):
            return tensor_fn(x)
        return math_fn(float(x))

    return wrapper

def mixmode_seed(seed: float) -> float:
    seed = abs(float(seed))
    niter = 11 + (23 + round(seed)) % 79
    a = 0.01
    b = 1.23456789
    x = (a + math.fmod(seed, b)) / (b + 2.0 * a)
    for _ in range(niter):
        x = 3.99 * x * (1.0 - x)
    return x

def mixmode(arg: float | Tensor, seed: float = 0.5) -> float | Tensor:
    if torch.is_tensor(arg):
        result = torch.zeros_like(arg)
    else:
        result = 0.0

    for i in range(14):
        phase = math.pi * (2.0 * mixmode_seed(seed + i) - 1.0)
        weight = 1.0 / (1.0 + abs(i - 4)) ** 2
        result = result + weight * torch.cos(i * arg + phase) if torch.is_tensor(arg) else result + weight * math.cos(i * arg + phase)
    return result

def as_tensor_like(value: float | Tensor, like: Tensor) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


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

def mse_dict(residuals: Mapping[str, Tensor], weights: Mapping[str, float] | None = None) -> Tensor:
    weights = dict(weights or {})
    total = torch.tensor(0.0)
    for name, residual in residuals.items():
        weight = float(weights.get(name, 1.0))
        loss = F.mse_loss(weight * residual, torch.zeros_like(residual))
        total = total + loss
    return total