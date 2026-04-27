from __future__ import annotations


import math
import torch
import xarray as xr
import re
from pathlib import Path


BOUNDARY_PATTERN = re.compile(r"^(?P<kind>[A-Za-z_]\w*)(?:\((?P<expr>.*)\))?$")


class BOUTHESELInfo:
    def __init__(self, data_folder_path: Path):
        self.folder_path = data_folder_path

        self.settings = self._load_settings()
#        self.data = self._load_data()
    
    def _load_settings(self):
        settings_path = self.folder_path / "BOUT.settings"
        current_section = "root"
        settings = {current_section: {}}

        # Læser rå fil input og laver: dict[str, dict[str, str]]
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

        # Sætter værdier til bool/int/float hvis muligt, ellers beholder som string
        settings_ref = {}
        for section, value in settings.items():
            for ref_key, ref_value in value.items():                
                for cast in (int, float):
                    try:
                        settings_ref[section, ref_key] = cast(ref_value)
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    if ref_value.lower() in ("true", "false"):
                        settings_ref[section, ref_key] = ref_value == "true"
                        continue
                    settings_ref[section, ref_key] = ref_value
        
        

        return settings_ref


"""
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

"""


root = Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL" / "data"

info = BOUTHESELInfo(root)
for (section, key), value in info.settings.items():
    if type(value) != str: print(f"{section}, {key}\t {value}")