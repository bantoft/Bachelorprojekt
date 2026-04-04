try:
    from SciML.core.params import load_params_from_bout_inp, resolve_hesel_params
except ImportError:
    from ..core.params import load_params_from_bout_inp, resolve_hesel_params


_resolve_hesel_params = resolve_hesel_params

