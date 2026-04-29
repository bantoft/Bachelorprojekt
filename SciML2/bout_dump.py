from __future__ import annotations

import re
import ast
import math
import torch
import xarray

from pathlib import Path
from types import SimpleNamespace


from api.dump_helper import *



class BOUTHESELInfo:
    def __init__(self, root: Path):
        self.root = root
        self._raw = self._read(Path(root) / 'BOUT.settings')
        self.settings = {ref: self._resolve(*ref, ()) for ref in self._raw}

        for key, value in DEFAULT_FACTORY.items():
            if ('hesel', key) in self.settings: continue
            self.settings.setdefault(('hesel', key), value)

        self.functions = self._parse_functions()
        self.data = self._load_data()
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

        value = TOKEN.sub(
            lambda match: self._replace(section, match.group(0), stack + (ref,)),
            value,
        )
        try:
            return self._eval(value)
        except Exception:
            return value

    def _replace(self, section: str, token: str, stack: tuple[tuple[str, str], ...]) -> str:

        ref = self._find(section, token)
        if ref is None and token.lower() in SKIP | {'x', 'y', 'z', 't'}: return token
        if ref is None: return token

        value = str(self._resolve(*ref, stack))
        return value if self._literal(value) is not None or SIMPLE.fullmatch(value) else f'({value})'

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
            if not isinstance(node, EVAL_ALLOWED) or isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise ValueError

        return eval(
            compile(tree, '<bout-settings>', 'eval'),
            {'__builtins__': {}},
            MATH_NAMESPACE,
        )

    def _parse_functions(self):
        foo = SimpleNamespace()
        for ref, value in self.settings.items():
            if not isinstance(value, str): continue

            try:
                tree = ast.parse(value, mode='eval')
            except SyntaxError:
                continue

            if not {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & FUNCTION_ARG_SET:
                continue

            if not self._valid_function_tree(tree): continue

            lambda_node = ast.Expression(
                body=ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=name) for name in FUNCTION_ARGS],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[ast.Constant(None) for _ in FUNCTION_ARGS],
                    ),
                    body=tree.body,
                )
            )
            ast.fix_missing_locations(lambda_node)
            func = eval(compile(lambda_node, '<bout-function>', 'eval'), FUNCTION_GLOBALS)
            func.__name__ = f"{ref[0]}_{ref[1]}"
            setattr(foo, f"{ref[0]}_{ref[1]}", func)
        return foo

    def _valid_function_tree(self, tree: ast.Expression) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, FUNCTION_ALLOWED):
                return False
            if isinstance(node, ast.Name) and node.id not in FUNCTION_NAMESPACE and node.id not in FUNCTION_ARG_SET:
                return False
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCTION_NAMESPACE):
                return False
        return True

    def _load_data(self):
        path = self.root
        needed_vars = ['lnn', 'lnpe', 'lnpi', 'phi', 'vort', 'init_n', 'init_pe', 'init_pi', 'sigma_open', 'sigma_closed', 'sigma_force', 'B', 'dx', 'dz', 't_array', 'Bt', 'q', 'Te0', 'Ti0', 'n0', 'lconn', 'Rmajor', 'Rminor', 'A', 'Z', 'Mach', 'x_lcfs', 'x_wall', 'force_time', 'floor_time', 'floor_n', 'floor_pe', 'floor_pi', 'B0', 'oci', 'rhoe', 'rhos', 'nuei', 'nuii', 'nuee', 'neoclass_correction_factor', 'lblob']
        data_obj = SimpleNamespace(**{name: None for name in needed_vars})
        overlap = 2
        for data_path in sorted(path.glob('BOUT.dmp.*.nc')):
            with xarray.open_dataset(data_path, engine='netcdf4') as dataset:
                for name in needed_vars:
                    if name not in dataset: continue

                    variable = dataset[name]
                    array = variable.squeeze('y', drop=True) if 'y' in variable.dims else variable
                    values = torch.as_tensor(array.values)

                    if name == 't_array':
                        if getattr(data_obj, name) is None:
                            setattr(data_obj, name, values)
                        continue

                    previous = getattr(data_obj, name)
                    if previous is None:
                        setattr(data_obj, name, values)
                        continue

                    if 'x' not in array.dims: continue

                    axis = array.dims.index('x')
                    slicer = [slice(None)] * values.ndim
                    slicer[axis] = slice(min(overlap, int(values.shape[axis])), None)
                    concatenated = torch.cat((previous, values[tuple(slicer)]), dim=axis)
                    setattr(data_obj, name, concatenated)
        return data_obj

    def _build_param(self):
        from types import SimpleNamespace
        
        param = SimpleNamespace()
        param.e = 1.60e-19
        param.epso = 8.85e-12
        param.me = 9.1093816e-31
        param.mp = 1.67262158e-27
        param.pi = math.pi
        param.totalt_t = self.data.lnn.shape[0]  # type: ignore
        param.total_z = self.data.lnn.shape[1]  # type: ignore
        param.total_x = self.data.lnn.shape[2]  # type: ignore

        def get_param(name: str):
            key = name.lower()
            data_attr = getattr(self.data, key, None)
            if data_attr is not None and data_attr.ndim == 0:
                return data_attr.item()
            if ("hesel", key) in self.settings:
                return self.settings["hesel", key]
            return None

        for name in needed_params:
            if (value := get_param(name)) is not None:
                setattr(param, name, value)
        
        debye = math.sqrt(param.epso*param.e*param.te0/(param.e**2*param.n0))
        param.collog = math.log(12.0 * param.pi * param.n0 * debye**3 / param.z)
        param.rhoi = math.sqrt(param.e * param.ti0 / param.mi) / param.oci
        param.bohm_potential = math.log(math.sqrt(param.mi / (2.0 * param.pi * param.me)))
        
        param.norm_de = (param.neoclass_correction_factor
                         * param.rhoe**2
                         * param.nuei
                         / (param.rhos**2 * param.oci))
        
        param.norm_di = (param.neoclass_correction_factor
                         * param.rhoi**2
                         * param.nuii
                         / (param.rhos**2 * param.oci))
        
        param.norm_eta = 3.0 / 10.0 * param.norm_di
        param.norm_taun = param.lblob/ (2.0 * param.mach * param.cs)* param.oci
        param.norm_lc = param.lconn / param.rhos
        param.norm_lb = param.lblob / param.rhos
        param.norm_taudw = (param.lblob**2
                            * param.me
                            * param.nuei
                            / (2.0 * param.e * param.te0)
                            * param.oci)
        
        param.norm_taushe = (param.lconn**2
                             * param.n0
                             / (3.16 * param.n0 * param.e * param.te0/ (param.me * param.nuei))
                             * param.oci)
        
        param.norm_taushi = (param.lconn**2
                             * param.n0
                             / (3.9 * param.n0 * param.e * param.ti0 / (param.mi * param.nuii))
                             * param.oci)
        return param



if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'simulatorer' / 'BOUT' / 'BOUT-HESEL' / 'data'
    info = BOUTHESELInfo(root)
    # for name, foon in  info.functions.__dict__.items():
    #     print(name, foon)
    for name, arr in info.data.__dict__.items():
        print(name, arr.shape)
    # for (name, key), value in info.settings.items():
    #     print(f"{key} ({name}): {value}")
    
#     "(rel, abs) error of data"
#     for (name, array) in data.__dict__.items():
#         x32 = array.float()
#         abs_err = (array - x32.double()).abs().max()

#         rel_err = (
#             (array - x32.double()).abs()
#             / array.abs().clamp(min=1e-12)
#         ).max()
#         print(f"({rel_err:.3e}, {abs_err:.3e}) \t {name}")


#     param = info.parameters
#     "(rel, abs) error of parameters"
#     for (name, value) in param.__dict__.items():
#         array = torch.as_tensor(value)
#         x32 = torch.tensor(array, dtype=torch.float32)
#         abs_err = (array - x32.double()).abs().max()

#         rel_err = (
#             (array - x32.double()).abs()
#             / array.abs().clamp(min=1e-12)
#         ).max()
#         print(f"({rel_err:.3e}, {abs_err:.3e}) \t {name}")
