"""Injector registry: name -> factory."""

_INJECTORS = {}


def register_injector(name, factory):
    _INJECTORS[name] = factory


def available_injectors():
    _bootstrap()
    return sorted(_INJECTORS.keys())


def get_injector(name, lab):
    if name not in _INJECTORS:
        _bootstrap()
    if name not in _INJECTORS:
        raise KeyError("unknown injector '%s' (available: %s)" % (name, available_injectors()))
    return _INJECTORS[name](lab)


def _bootstrap():
    if "nvbitfi" not in _INJECTORS:
        from core.injection.nvbitfi import NVBitFIInjector
        register_injector("nvbitfi", NVBitFIInjector)
