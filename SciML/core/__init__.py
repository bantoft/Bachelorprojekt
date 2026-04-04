"""Core SciML package with models, operators and losses."""

from .model import PINN
from .params import load_params_from_bout_inp, resolve_hesel_params
from .losses import pde_loss
