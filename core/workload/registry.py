"""Workload registry: name -> factory. New workloads register here."""

_REGISTRY = {}


def register_workload(name, factory):
    _REGISTRY[name] = factory


def available_workloads():
    _bootstrap()
    return sorted(_REGISTRY.keys())


def get_workload(name, params, lab):
    if name not in _REGISTRY:
        _bootstrap()
    if name not in _REGISTRY:
        raise KeyError("unknown workload '%s' (available: %s)" % (name, available_workloads()))
    return _REGISTRY[name](params, lab)


def _bootstrap():
    from core.workload.gemm import GEMMWorkload
    from core.workload.reduce import ReduceWorkload
    from core.workload.conv2d import Conv2dWorkload
    from core.workload.softmax import SoftmaxWorkload
    from core.workload.attention import AttentionWorkload
    from core.workload.mlp import MLPWorkload
    for name, cls in (("gemm", GEMMWorkload), ("reduce", ReduceWorkload),
                      ("conv2d", Conv2dWorkload), ("softmax", SoftmaxWorkload),
                      ("attention", AttentionWorkload), ("mlp", MLPWorkload)):
        if name not in _REGISTRY:
            register_workload(name, cls)
