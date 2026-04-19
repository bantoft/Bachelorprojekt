from pathlib import Path
import re

import numpy as np

REF_PATTERN = re.compile(r"\b([A-Za-z_]\w*):([A-Za-z_]\w*)\b")

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



