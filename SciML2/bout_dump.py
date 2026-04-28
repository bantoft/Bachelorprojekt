from __future__ import annotations

import ast
import re
import math
import torch
import xarray as xr
from pathlib import Path


_TOKEN = re.compile(r'\b[A-Za-z_]\w*(?::[A-Za-z_]\w*)*\b')
_SIMPLE = re.compile(r'^[A-Za-z_]\w*$')
_FUNCTION_ARGS = ('x', 'y', 'z', 't')
_FUNCTION_ARG_SET = set(_FUNCTION_ARGS)
_MATH_NAMESPACE = {name: value for name, value in vars(math).items() if not name.startswith('_')}
_SKIP = {
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
    *(name for name, value in _MATH_NAMESPACE.items() if callable(value)),
}
_EVAL_ALLOWED = (
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
_FUNCTION_ALLOWED = _EVAL_ALLOWED + (
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
_DEFAULT_FACTORY = {
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

_needed_params = [
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


_FUNCTION_NAMESPACE = dict(_MATH_NAMESPACE)
_FUNCTION_NAMESPACE.update(
    {
        name: _tensor_aware(value, getattr(torch, name))
        for name, value in _MATH_NAMESPACE.items()
        if callable(value) and hasattr(torch, name)
    }
)
_FUNCTION_NAMESPACE.update({'heaviside': _heaviside, 'gauss': _gauss})
_FUNCTION_GLOBALS = {'__builtins__': {}, **_FUNCTION_NAMESPACE}


class BOUTHESELInfo:
    def __init__(self, root: Path):
        self._raw = self._read(Path(root) / 'BOUT.settings')
        self.settings = {ref: self._resolve(*ref, ()) for ref in self._raw}

        for key, value in _DEFAULT_FACTORY.items():
            if ('hesel', key) in self.settings: continue
            self.settings.setdefault(('hesel', key), value)

        self.functions = self._parse_functions()
        self.data = self._load_data(Path(root))
        self.parameters = self._build_param()

    def _read(self, path: Path) -> dict[tuple[str, str], str]:
        settings, section = {}, 'root'
        for raw in path.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].strip()
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                settings[(section, key.strip())] = value.strip()
        return settings
    
    def _resolve(self, section: str, key: str, stack: tuple[tuple[str, str], ...]) -> object:
        ref = (section, key)

        value = self._raw[ref]
        literal = self._literal(value)
        if literal is not None:
            return literal

        value = _TOKEN.sub(
            lambda match: self._replace(section, match.group(0), stack + (ref,)),
            value,
        )
        try:
            return self._eval(value)
        except Exception:
            return value

    def _replace(self, section: str, token: str, stack: tuple[tuple[str, str], ...]) -> str:

        ref = self._find(section, token)
        if ref is None and token.lower() in _SKIP | {'x', 'y', 'z', 't'}: return token
        if ref is None: return token

        value = str(self._resolve(*ref, stack))
        return value if self._literal(value) is not None or _SIMPLE.fullmatch(value) else f'({value})'

    def _find(self, section: str, token: str) -> tuple[str, str] | None:
        if ':' in token:
            *parts, key = token.split(':')
            ref = (':'.join(parts) or 'root', key)
            return ref if ref in self._raw else None

        parts = section.split(':')
        chain = ['root'] if section == 'root' else [':'.join(parts[:i]) for i in range(len(parts), 0, -1)] + ['root']
        for current in chain:
            ref = (current, token)
            if ref in self._raw:
                return ref
        return None

    def _literal(self, value: str) -> bool | int | float | None:
        value = value.strip()
        if value == 'true':
            return True
        if value == 'false':
            return False
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return None

    def _eval(self, value: str) -> bool | int | float:
        tree = ast.parse(value, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, _EVAL_ALLOWED) or isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise ValueError

        return eval(
            compile(tree, '<bout-settings>', 'eval'),
            {'__builtins__': {}},
            _MATH_NAMESPACE,
        )

    def _parse_functions(self) -> dict[tuple[str, str], object]:
        functions = {}
        for ref, value in self.settings.items():
            if not isinstance(value, str):
                continue

            try:
                tree = ast.parse(value, mode='eval')
            except SyntaxError:
                continue

            if not {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & _FUNCTION_ARG_SET:
                continue

            if not self._valid_function_tree(tree):
                continue

            lambda_node = ast.Expression(
                body=ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=name) for name in _FUNCTION_ARGS],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[ast.Constant(None) for _ in _FUNCTION_ARGS],
                    ),
                    body=tree.body,
                )
            )
            ast.fix_missing_locations(lambda_node)
            functions[ref] = eval(compile(lambda_node, '<bout-function>', 'eval'), _FUNCTION_GLOBALS)
        return functions

    def _valid_function_tree(self, tree: ast.Expression) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, _FUNCTION_ALLOWED):
                return False
            if isinstance(node, ast.Name) and node.id not in _FUNCTION_NAMESPACE and node.id not in _FUNCTION_ARG_SET:
                return False
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_NAMESPACE):
                return False
        return True

    def _load_data(self, path: Path):
        needed_vars = ['lnn', 'lnpe', 'lnpi', 'phi', 'vort', 'init_n', 'init_pe', 'init_pi', 'sigma_open', 'sigma_closed', 'sigma_force', 'B', 'dx', 'dz', 't_array', 'Bt', 'q', 'Te0', 'Ti0', 'n0', 'lconn', 'Rmajor', 'Rminor', 'A', 'Z', 'Mach', 'x_lcfs', 'x_wall', 'force_time', 'floor_time', 'floor_n', 'floor_pe', 'floor_pi', 'B0', 'oci', 'rhoe', 'rhos', 'nuei', 'nuii', 'nuee', 'neoclass_correction_factor', 'lblob']
        data = {}
        overlap = 2
        for data_path in sorted(path.glob('BOUT.dmp.*.nc')):
            with xr.open_dataset(data_path, engine='netcdf4') as dataset:
                for name in needed_vars:
                    if name not in dataset: continue

                    variable = dataset[name]
                    array = variable.squeeze('y', drop=True) if 'y' in variable.dims else variable
                    values = torch.as_tensor(array.values)

                    if name == 't_array':
                        data.setdefault(name, values)
                        continue

                    previous = data.get(name)
                    if previous is None:
                        data[name] = values
                        continue

                    if 'x' not in array.dims: continue

                    axis = array.dims.index('x')
                    slicer = [slice(None)] * values.ndim
                    slicer[axis] = slice(min(overlap, int(values.shape[axis])), None)
                    data[name.lower()] = torch.cat((previous, values[tuple(slicer)]), dim=axis)
        return data

    def _build_param(self):
        data_key_low = [key for key in self.data.keys()]
        params = {
        'e': 1.60e-19,
        'epso': 8.85e-12,
        'me': 9.1093816e-31,
        'mp': 1.67262158e-27,
        'pi': math.pi
        }
        for param in _needed_params:
            if param.lower() in data_key_low and self.data[param.lower()].ndim == 0:
                params[param] = self.data[param.lower()].item()
            elif ('hesel', param.lower()) in self.settings:
                params[param] = self.settings[('hesel', param.lower())]
        params['rhoi'] = math.sqrt(params['e'] * params['ti0'] / params['mi']) / params['oci']
        debye = math.sqrt(params['epso'] * params['e'] * params['te0'] / (params['e'] * params['e'] * params['n0']))
        params['collog'] = math.log(12.0 * params['pi'] * params['n0'] * debye**3 / params['z'])
        params['bohm_potential'] = math.log(math.sqrt(params['mi'] / (2.0 * params['pi'] * params['me'])))


        params['norm_de'] = params['neoclass_correction_factor'] * params['rhoe'] * params['rhoe'] * params['nuei'] / (params['rhos'] * params['rhos'] * params['oci'])
        params['norm_di'] = (params['neoclass_correction_factor'] * params['rhoi'] * params['rhoi'] * params['nuii']) / (params['rhos'] * params['rhos'] * params['oci'])
        params['norm_eta']=3.0 / 10.0 * params['norm_di']
        params['norm_taun']=(params['lblob'] / (2.0 * params['mach'] * params['cs'])) * params['oci']
        params['norm_lc']=params['lconn'] / params['rhos']
        params['norm_lb']=params['lblob'] / params['rhos']
        params['norm_taudw']=(params['lblob'] * params['lblob'] * params['me'] * params['nuei'] / (2.0 * params['e'] * params['te0'])) * params['oci']
        params['norm_taushe']=(params['lconn'] * params['lconn'] * params['n0'] / (3.16 * params['n0'] * params['e'] * params['te0'] / (params['me'] * params['nuei']))) * params['oci']
        params['norm_taushi']=(params['lconn'] * params['lconn'] * params['n0'] / (3.9 * params['n0'] * params['e'] * params['ti0'] / (params['mi'] * params['nuii']))) * params['oci']
        return params



# dx = torch.as_tensor(self.data["dx"], dtype=torch.float64).flatten()
# total_x = float(dx[:-1].sum().item()) if dx.numel() > 1 else float(dx.sum().item())
# total_z = self._data_scalar("dz") * max(int(self.data["lnn"].shape[-1]) - 1, 1)
# total_t = float(torch.as_tensor(self.data["t_array"], dtype=torch.float64)[-1].item())


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'simulatorer' / 'BOUT' / 'BOUT-HESEL' / 'data'
    info = BOUTHESELInfo(root)
    print(sorted(info.parameters.items()))
    # How do i see keys in info.data and but them in lower case


    # params = {
    #     'e': 1.60e-19,
    #     'epso': 8.85e-12,
    #     'me': 9.1093816e-31,
    #     'mp': 1.67262158e-27,
    #     'pi': math.pi
    # }

    # for name, source_name in direct_params.items():
    #     if source_name in info.data and info.data[source_name].ndim == 0:
    #         params[name] = info.data[source_name].item()
    #     elif ('hesel', source_name) in info.settings:
    #         params[name] = info.settings[('hesel', source_name)]

    # params['rhoi'] = math.sqrt(params['e'] * params['ti0'] / params['mi']) / params['oci']
    # debye = math.sqrt(params['epso'] * params['e'] * params['te0'] / (params['e'] * params['e'] * params['n0']))
    # params['collog'] = math.log(12.0 * params['pi'] * params['n0'] * debye**3 / params['z'])
    # params['bohm_potential'] = math.log(math.sqrt(params['mi'] / (2.0 * params['pi'] * params['me'])))


    # params['norm_de'] = params['neoclass_correction_factor'] * params['rhoe'] * params['rhoe'] * params['nuei'] / (params['rhos'] * params['rhos'] * params['oci'])
    # params['norm_di'] = (params['neoclass_correction_factor'] * params['rhoi'] * params['rhoi'] * params['nuii']) / (params['rhos'] * params['rhos'] * params['oci'])
    # params['norm_eta']=3.0 / 10.0 * params['norm_di']
    # params['norm_taun']=(params['lblob'] / (2.0 * params['mach'] * params['cs'])) * params['oci']
    # params['norm_lc']=params['lconn'] / params['rhos']
    # params['norm_lb']=params['lblob'] / params['rhos']
    # params['norm_taudw']=(params['lblob'] * params['lblob'] * params['me'] * params['nuei'] / (2.0 * params['e'] * params['te0'])) * params['oci']
    # params['norm_taushe']=(params['lconn'] * params['lconn'] * params['n0'] / (3.16 * params['n0'] * params['e'] * params['te0'] / (params['me'] * params['nuei']))) * params['oci']
    # params['norm_taushi']=(params['lconn'] * params['lconn'] * params['n0'] / (3.9 * params['n0'] * params['e'] * params['ti0'] / (params['mi'] * params['nuii']))) * params['oci']
