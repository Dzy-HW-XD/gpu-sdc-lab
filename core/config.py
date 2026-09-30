"""Lab configuration loading and path resolution."""

import os

from core import util
from core import yamlmini

# Resolved through realpath so the project works from any location and through
# symlinks.
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DEFAULTS = {
    "gpu": 1,
    "cuda_home": None,
    "nvbit_home": "third_party/nvbit_release_x86_64",
    "nvbitfi_home": "third_party/nvbit_release_x86_64/tools/nvbitfi",
    "injector_lib": "third_party/nvbit_release_x86_64/tools/nvbitfi/injector/injector.so",
    "profiler_lib": "third_party/nvbit_release_x86_64/tools/nvbitfi/profiler/profiler.so",
    "results_dir": "results",
    "oracle": {
        "abs_tol": 0.0,
        "rel_tol": 0.0,
        "nan_inf_dominant_ratio": 0.5,
    },
    "runtime": {
        "default_timeout": 300,
        "profile_timeout": 1800,
        "nvbit_arch": "sm_80",
    },
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class LabConfig(object):
    def __init__(self, data=None, root=LAB_ROOT):
        self.root = root
        self.raw = _deep_merge(DEFAULTS, data or {})

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, key):
        return self.raw[key]

    def path(self, p):
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    @property
    def gpu(self):
        return int(self.raw["gpu"])

    @property
    def cuda_home(self):
        raw = self.raw.get("cuda_home")
        if raw and os.path.exists(os.path.join(raw, "bin", "nvcc")):
            return raw
        return util.detect_cuda_home(raw)

    @property
    def nvbit_home(self):
        return self.path(self.raw["nvbit_home"])

    @property
    def nvbitfi_home(self):
        return self.path(self.raw["nvbitfi_home"])

    @property
    def injector_lib(self):
        return self.path(self.raw["injector_lib"])

    @property
    def profiler_lib(self):
        return self.path(self.raw["profiler_lib"])

    @property
    def results_dir(self):
        return self.path(self.raw["results_dir"])

    @property
    def oracle(self):
        return self.raw["oracle"]

    @property
    def runtime(self):
        return self.raw["runtime"]

    def to_dict(self):
        return dict(self.raw)


def load_lab_config(path=None, overrides=None):
    data = {}
    if path and os.path.exists(path):
        data = yamlmini.load(path) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return LabConfig(data)
