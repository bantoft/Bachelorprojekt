import xarray as xr
import torch
import torch.nn.functional as F
from pathlib import Path

from torch import Tensor

DATA_PATH = Path(__file__).with_name("BOUT.dmp.0.nc")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find dataset: {DATA_PATH}")

ds = xr.open_dataset(DATA_PATH, engine="netcdf4")

vars_needed = ["lnn", "lnpe", "lnpi", "vort"]
data = {}

for name in vars_needed:
    arr = ds[name].squeeze("y", drop=True)
    arr = arr.rename({"z": "y"})
    arr = arr.transpose("x", "y", "t")
    data[name] = torch.tensor(arr.values, dtype=torch.float32)

def data_loss(pred: Tensor, name: str, x_ids: Tensor, y_ids: Tensor, t_ids: Tensor):
    target = data[name][x_ids, y_ids, t_ids]
    target = target.reshape_as(pred)
    return F.mse_loss(pred, target)

def initial_condition_loss(pred: Tensor, name: str, x_ids: Tensor, y_ids: Tensor):
    target = data[name][x_ids, y_ids, 0]
    target = target.reshape_as(pred)
    return F.mse_loss(pred, target)

def left_dirichlet_loss(pred: Tensor, name: str, y_ids: Tensor, t_ids: Tensor):
    target = data[name][0, y_ids, t_ids]
    target = target.reshape_as(pred)
    return F.mse_loss(pred, target)

def right_dirichlet_loss(pred: Tensor, name: str, y_ids: Tensor, t_ids: Tensor):
    target = data[name][-1, y_ids, t_ids]
    target = target.reshape_as(pred)
    return F.mse_loss(pred, target)