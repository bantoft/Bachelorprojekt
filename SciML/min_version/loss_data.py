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


def to_xyt_tensor(da: xr.DataArray) -> torch.Tensor:
    # Keep one perpendicular plane and map to the training convention (x, y, t).
    arr = da
    if "y" in arr.dims:
        arr = arr.squeeze("y", drop=True)
    if "z" in arr.dims:
        arr = arr.rename({"z": "y"})

    required_dims = {"x", "y", "t"}
    if not required_dims.issubset(set(arr.dims)):
        raise ValueError(f"Expected dims including x, y, t but got {arr.dims} for variable {da.name}")

    arr = arr.transpose("x", "y", "t")
    return torch.tensor(arr.values, dtype=torch.float32)


for name in vars_needed:
    data[name] = to_xyt_tensor(ds[name])
    print(f"loaded {name} with shape {data[name].shape}")

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