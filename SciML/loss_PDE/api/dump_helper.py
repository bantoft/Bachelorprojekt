from __future__ import annotations

import ast
import re
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional

def _tensor_aware(math_func, torch_func):
    def wrapped(*args):
        if any(torch.is_tensor(arg) for arg in args):
            tensor_args = tuple(arg if torch.is_tensor(arg) else torch.as_tensor(arg) for arg in args)
            return torch_func(*tensor_args)
        return math_func(*args)
    return wrapped


def _heaviside(x):
    if torch.is_tensor(x):
        one = torch.ones((), device=x.device, dtype=x.dtype)
        return torch.heaviside(x, one)
    return 1.0 if x >= 0 else 0.0


def _gauss(x, width=1.0, center=0.0):
    shifted = (x - center) / width
    if torch.is_tensor(shifted):
        return torch.exp(-(shifted ** 2))
    return math.exp(-(shifted ** 2))


TOKEN = re.compile(r'\b[A-Za-z_]\w*(?::[A-Za-z_]\w*)*\b')
SIMPLE = re.compile(r'^[A-Za-z_]\w*$')
FUNCTION_ARGS = ('x', 'y', 'z', 't')
FUNCTION_ARG_SET = set(FUNCTION_ARGS)
MATH_NAMESPACE = {name: value for name, value in vars(math).items() if not name.startswith('_')}
SKIP = {
    'pi',
    'true',
    'false',
    'none',
    'neumann',
    'mixmode',
    'gauss',
    'heaviside',
    'ballooning',
    'dirichlet_o2',
    'neumann_o2',
    *(name for name, value in MATH_NAMESPACE.items() if callable(value)),
}
EVAL_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)
FUNCTION_ALLOWED = EVAL_ALLOWED + (
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
DEFAULT_FACTORY = {
            'n_bck': 0.0,
            'te_bck': 0.0,
            'ti_bck': 0.0,
            'double_curvature_coeff': False,
            'h_mode': False,
            'invert_w_star': False,
            'not_n_force': False,
            'not_p_force': False,
            'parallel_transport': False,
            'particle_source': False,
            'power_source': False,
            'ramp_a': 2.0,
            'ramp_peak': 50000.0,
            'ramp_t0': 0.0,
            'ramp_trans': 5000.0,
            'test_vort_cross_term': False,
        }

needed_params = [
'a',
'b0',
'bohm_potential',
'bt',
'collog',
'cs',
'd_force',
'd_lcfs',
'd_wall',
'e',
'epso',
'floor_n',
'floor_pe',
'floor_pi',
'floor_time',
'force_time',
'lblob',
'lconn',
'mach',
'me',
'mi',
'mp',
'n0',
'n_bck',
'neoclass_correction_factor',
'norm_de',
'norm_di',
'norm_eta',
'norm_lb',
'norm_lc',
'norm_taudw',
'norm_taun',
'norm_taushe',
'norm_taushi',
'nuee',
'nuei',
'nuii',
'oci',
'pi',
'q',
'rhoe',
'rhoi',
'rhos',
'rmajor',
'rminor',
'te0',
'te_bck',
'ti0',
'ti_bck',
'wall_amp',
'x_lcfs',
'x_wall',
'z',
'z_eff',
]


FUNCTION_NAMESPACE = dict(MATH_NAMESPACE)
FUNCTION_NAMESPACE.update(
    {
        name: _tensor_aware(value, getattr(torch, name))
        for name, value in MATH_NAMESPACE.items()
        if callable(value) and hasattr(torch, name)
    }
)
FUNCTION_NAMESPACE.update({'heaviside': _heaviside, 'gauss': _gauss})
FUNCTION_GLOBALS = {'__builtins__': {}, **FUNCTION_NAMESPACE}


@dataclass
class param:
    e: float
    epso: float
    me: float
    mp: float
    pi: float
    num_t: int
    num_x: int
    num_z: int
    dt: float
    dx: float
    dz: float
    total_t: float
    total_z: float
    total_x: float
    a: int
    b0: float
    bt: float
    cs: float
    d_force: int
    d_lcfs: int
    d_wall: int
    floor_n: float
    floor_pe: float
    floor_pi: float
    floor_time: float
    force_time: float
    lblob: float
    lconn: int
    mach: float
    mi: float
    n0: float
    n_bck: float
    neoclass_correction_factor: float
    nuee: float
    nuei: float
    nuii: float
    oci: float
    q: float
    rhoe: float
    rhos: float
    rmajor: float
    rminor: float
    te0: float
    te_bck: float
    ti0: float
    ti_bck: float
    wall_amp: int
    x_lcfs: float
    x_wall: float
    z: int
    z_eff: float
    rhoi: float
    collog: float
    bohm_potential: float
    norm_de: float
    norm_di: float
    norm_eta: float
    norm_taun: float
    norm_lc: float
    norm_lb: float
    norm_taudw: float
    norm_taushe: float
    norm_taushi: float


@dataclass
class foo:
    hesel_xr: Callable
    init_n_function: Callable
    init_neutrals_ncold_init: Callable
    init_pe_edge_prof: Callable
    init_pe_function: Callable
    init_pi_edge_prof: Callable
    init_pi_function: Callable
    lnncold_function: Callable
    lnpe_function: Callable
    lnpi_function: Callable
    mesh_xl: Callable
    mesh_zl: Callable
    seed_n_xaxis: Callable
    seed_n_zaxis: Callable
    sigma_closed_function: Callable
    sigma_force_function: Callable
    sigma_open_function: Callable


@dataclass
class data:
    lnn: Optional[torch.Tensor]
    lnpe: Optional[torch.Tensor]
    lnpi: Optional[torch.Tensor]
    phi: Optional[torch.Tensor]
    vort: Optional[torch.Tensor]
    init_n: Optional[torch.Tensor]
    init_pe: Optional[torch.Tensor]
    init_pi: Optional[torch.Tensor]
    sigma_open: Optional[torch.Tensor]
    sigma_closed: Optional[torch.Tensor]
    sigma_force: Optional[torch.Tensor]
    B: Optional[torch.Tensor]
    dx: Optional[torch.Tensor]
    dz: Optional[torch.Tensor]
    t_array: Optional[torch.Tensor]
    Bt: Optional[torch.Tensor]
    q: Optional[torch.Tensor]
    Te0: Optional[torch.Tensor]
    Ti0: Optional[torch.Tensor]
    n0: Optional[torch.Tensor]
    lconn: Optional[torch.Tensor]
    Rmajor: Optional[torch.Tensor]
    Rminor: Optional[torch.Tensor]
    A: Optional[torch.Tensor]
    Z: Optional[torch.Tensor]
    Mach: Optional[torch.Tensor]
    x_lcfs: Optional[torch.Tensor]
    x_wall: Optional[torch.Tensor]
    force_time: Optional[torch.Tensor]
    floor_time: Optional[torch.Tensor]
    floor_n: Optional[torch.Tensor]
    floor_pe: Optional[torch.Tensor]
    floor_pi: Optional[torch.Tensor]
    B0: Optional[torch.Tensor]
    oci: Optional[torch.Tensor]
    rhoe: Optional[torch.Tensor]
    rhos: Optional[torch.Tensor]
    nuei: Optional[torch.Tensor]
    nuii: Optional[torch.Tensor]
    nuee: Optional[torch.Tensor]
    neoclass_correction_factor: Optional[torch.Tensor]
    lblob: Optional[torch.Tensor]


@dataclass
class standardization:
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

        