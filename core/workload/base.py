"""Unified Workload interface.

Every workload (GEMM, reduce, conv2d, softmax, attention, training, ...) exposes
the same lifecycle:  prepare() -> run() -> collect_output() -> cleanup().

Workloads may declare:
  * `input_flags`  : {input_name: cli_flag}  e.g. {"A": "--inputA"}
  * `input_size(name)` : number of float elements for that input
  * `output_specs` : [{"name","suffix","format","tol"(opt)}]

When `params["input_spec"]` is present (a dict of {input_name: generator spec}),
prepare() materialises raw float32 files via core.inputgen and command() passes
them to the binary. Otherwise the kernel uses its built-in deterministic input.
"""

import os

from core import inputgen
from core import util


class Workload(object):
    name = "base"
    kernel_token = None
    input_flags = {}
    output_specs = [{"name": "out", "suffix": ".bin", "format": "f32"}]

    def __init__(self, params, lab):
        self.params = params or {}
        self.lab = lab
        self.input_paths = {}

    # ---- lifecycle ------------------------------------------------------
    def prepare(self):
        self.input_paths = self._materialize_inputs()
        return None

    def command(self, out_prefix, device=0):
        raise NotImplementedError

    def run(self, out_prefix, device=0, cwd=None, extra_env=None, timeout=None):
        cmd = self.command(out_prefix, device=device)
        if timeout is None:
            timeout = self.lab.runtime.get("default_timeout", 300)
        cuda_bin = os.path.join(self.lab.cuda_home, "bin")
        cuda_lib = os.path.join(self.lab.cuda_home, "lib64")
        base_env = {
            "PATH": cuda_bin + ":" + os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        }
        if extra_env:
            base_env.update(extra_env)
        return util.run_cmd(cmd, cwd=cwd, extra_env=base_env, timeout=timeout)

    def collect_output(self, out_prefix):
        outputs = []
        for s in self.output_specs:
            outputs.append({"name": s["name"], "path": out_prefix + s["suffix"]})
        meta = {}
        mpath = out_prefix + ".json"
        if os.path.exists(mpath):
            try:
                meta = util.read_json(mpath)
            except (ValueError, OSError):
                meta = {}
        return {"prefix": out_prefix, "outputs": outputs, "meta": meta}

    def cleanup(self):
        return None

    # ---- input materialisation -----------------------------------------
    def input_size(self, name):
        raise NotImplementedError("workload %s does not declare input sizes" % self.name)

    def _materialize_inputs(self):
        spec = self.params.get("input_spec") or {}
        if not spec:
            return {}
        cdir = os.path.join(self.lab.results_dir, "_inputs",
                            util.short_hash(self.signature()))
        util.ensure_dir(cdir)
        paths = {}
        for name, gen_spec in spec.items():
            p = os.path.join(cdir, "%s.bin" % name)
            if not os.path.exists(p):
                inputgen.materialize(gen_spec, self.input_size(name), p)
            paths[name] = p
        return paths

    def _input_args(self):
        args = []
        for name, flag in self.input_flags.items():
            if name in self.input_paths:
                args += [flag, self.input_paths[name]]
        return args

    # ---- helpers --------------------------------------------------------
    def signature(self):
        return "%s:%s" % (self.name, sorted(_hashable(self.params).items()))

    @property
    def output_bin(self):
        return None


def _hashable(obj):
    """Make nested params JSON-ish so signature() is stable."""
    if isinstance(obj, dict):
        return {k: _hashable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_hashable(v) for v in obj]
    return obj
