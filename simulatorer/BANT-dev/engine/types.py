from typing import Callable, Any
from dataclasses import dataclass

import numpy as np

from .grid import Grid2D

RHSFn = Callable[[float, np.ndarray, np.ndarray], None]
CreateGridFn = Callable[[], Grid2D]
InitialConditionFn = Callable[[Grid2D], np.ndarray]
CreateRHSKwargsFn = Callable[[Grid2D, np.ndarray], dict[str, Any]]

@dataclass
class PDESystem:
    """Container for a user-defined PDE system."""
    rhs: RHSFn
    create_grid: CreateGridFn
    initial_condition: InitialConditionFn
    create_rhs_kwargs: CreateRHSKwargsFn | None = None