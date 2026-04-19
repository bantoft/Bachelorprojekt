from pathlib import Path
import re

import numpy as np

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


REF_PATTERN = re.compile(r"\b([A-Za-z_]\w*):([A-Za-z_]\w*)\b")


def _parse_literal(value: str):
    text = value.strip()
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return None


def _resolve(cfg, section, key, variables, stack):
    node = (section, key)
    if node in stack:
        chain = " -> ".join(f"{s}:{k}" for s, k in stack + [node])
        raise ValueError(f"Cyclic reference detected: {chain}")

    raw = cfg[section][key].strip()
    literal = _parse_literal(raw)
    if literal is not None:
        return literal

    def _ref(sec, opt):
        if sec not in cfg or opt not in cfg[sec]:
            raise KeyError(f"Unknown reference {sec}:{opt}")
        return _resolve(cfg, sec, opt, variables, stack + [node])

    expr = REF_PATTERN.sub(r'__ref("\1", "\2")', raw)

    local_symbols = {}
    for opt_name in cfg[section].keys():
        if opt_name.isidentifier() and opt_name != key:
            try:
                local_symbols[opt_name] = _resolve(cfg, section, opt_name, variables, stack + [node])
            except Exception:
                # Ignore symbols that cannot be resolved yet; they might be irrelevant for this expression.
                pass

    # BOUT options often use top-level symbols (outside sections), e.g. mxg.
    if "root" in cfg:
        for opt_name in cfg["root"].keys():
            if opt_name.isidentifier() and opt_name != key and opt_name not in local_symbols:
                try:
                    local_symbols[opt_name] = _resolve(cfg, "root", opt_name, variables, stack + [node])
                except Exception:
                    pass

    scope = {
        "__ref": _ref,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "abs": np.abs,
        "pi": np.pi,
        **local_symbols,
        **variables,
    }
    return eval(expr, {"__builtins__": {}}, scope)


def make_bout_function(cfg, section, key):
    def f(**variables):
        return _resolve(cfg, section, key, variables=variables, stack=[])

    return f
