from __future__ import annotations


import math
import torch
import xarray as xr

from pathlib import Path




class BOUTHESELInfo:
    def __init__(self, data_folder_path: Path):
        self.folder_path = data_folder_path

        self.settings = self._load_settings()
        self.data = self._load_data()
        self.parameters = self._build_parameters()
    
    def _load_settings(self):
        settings_path = self.folder_path / "BOUT.settings"
        current_section = "root"
        settings: dict[str, dict[str, str]] = {current_section: {}}

        for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                settings.setdefault(current_section, {})
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            settings[current_section][key.strip()] = value.strip()

        return settings


    def _load_data(self):
        needed_vars = ['lnn', 'lnpe', 'lnpi', 'phi', 'vort', 'init_n', 'init_pe', 'init_pi', 'sigma_open', 'sigma_closed', 'sigma_force', 'B', 'dx', 'dz', 't_array', 'Bt', 'q', 'Te0', 'Ti0', 'n0', 'lconn', 'Rmajor', 'Rminor', 'A', 'Z', 'Mach', 'x_lcfs', 'x_wall', 'force_time', 'floor_time', 'floor_n', 'floor_pe', 'floor_pi', 'B0', 'oci', 'rhoe', 'rhos', 'nuei', 'nuii', 'nuee', 'neoclass_correction_factor', 'lblob']
        data = {}
        overlap = 2
        for data_path in sorted(self.folder_path.glob("BOUT.dmp.*.nc")):
            with xr.open_dataset(data_path, engine="netcdf4") as dataset:
                for name in needed_vars:
                    if name not in dataset: continue

                    variable = dataset[name]
                    array = variable.squeeze("y", drop=True) if "y" in variable.dims else variable
                    values = torch.as_tensor(array.values)

                    if name == "t_array":
                        data.setdefault(name, values)
                        continue

                    previous = data.get(name)
                    if previous is None:
                        data[name] = values
                        continue

                    if "x" not in array.dims: continue

                    axis = array.dims.index("x")
                    slicer = [slice(None)] * values.ndim
                    slicer[axis] = slice(min(overlap, int(values.shape[axis])), None)
                    data[name] = torch.cat((previous, values[tuple(slicer)]), dim=axis)

        return data

    
    def _build_parameters(self):
        params = {
            "e": 1.60e-19,
            "epso": 8.85e-12,
            "me": 9.1093816e-31,
            "mp": 1.67262158e-27,
            "pi_const": math.pi
        }
        for name, value in self.settings["hesel"].items():
            # Convert to float if possible, otherwise keep as string
            try:                
                value = float(value)
                params[name] = value
            except ValueError:
                continue
        # mi = a * mp
        # cs = math.sqrt(e * te0 / mi)
        # rhoi = math.sqrt(e * ti0 / mi) / oci
        # debye = math.sqrt(epso * e * te0 / (e * e * n0))
        # collog = math.log(12.0 * pi_const * n0 * debye**3 / z)

        # bohm_potential = math.log(math.sqrt(mi / (2.0 * pi_const * me)))
        # de_phys = neoclass_correction_factor * rhoe * rhoe * nuei
        # di_phys = neoclass_correction_factor * rhoi * rhoi * nuii
        # norm_de = de_phys / (rhos * rhos * oci)
        # norm_di = di_phys / (rhos * rhos * oci)
        # chi_e_par = 3.16 * n0 * e * te0 / (me * nuei)
        # chi_i_par = 3.9 * n0 * e * ti0 / (mi * nuii)
        # taun = lblob / (2.0 * mach * cs)
        # taudw = lblob * lblob * me * nuei / (2.0 * e * te0)
        # taushe = lconn * lconn * n0 / chi_e_par
        # taushi = lconn * lconn * n0 / chi_i_par


        return params


root = Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL" / "data"

info = BOUTHESELInfo(root)

sorted_parameters = sorted(info.parameters.items())

for navn, værdi in sorted_parameters:
    print(f"{navn}: {værdi}")