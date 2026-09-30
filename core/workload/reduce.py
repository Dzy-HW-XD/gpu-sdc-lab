"""Reduce workload: deterministic FP32 block reduction with atomics.

Observable = per-block partial sums (bit-exact); the atomic accumulator is
recorded as metadata only (order dependent). See workloads/reduce/reduce.cu.
"""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {
    "N": 1048576,
    "blocks": 256,
    "threads": 256,
    "dtype": "fp32",
    "seed": 1234,
}


class ReduceWorkload(Workload):
    name = "reduce"
    kernel_token = "reduce"
    input_flags = {"X": "--input"}
    output_specs = [{"name": "partials", "suffix": ".bin", "format": "f32"}]

    def input_size(self, name):
        if name == "X":
            return int(self.params["N"])
        raise KeyError(name)

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(ReduceWorkload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "reduce")
        self.src = os.path.join(self.dir, "reduce.cu")
        self.bin = os.path.join(self.dir, "reduce")

    def prepare(self):
        if not os.path.exists(self.bin) or (
            os.path.getmtime(self.src) > os.path.getmtime(self.bin)
        ):
            self.build()
        return self.bin

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            raise RuntimeError("nvcc not found at %s" % nvcc)
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        cmd = [nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
               "-o", self.bin, self.src]
        res = util.run_cmd(cmd, cwd=self.dir, timeout=600)
        if res["exit_code"] != 0:
            raise RuntimeError("reduce build failed:\n%s\n%s"
                               % (res["stdout"], res["stderr"]))
        return self.bin

    def command(self, out_prefix, device=0):
        p = self.params
        return [
            self.bin,
            "--N", str(p["N"]),
            "--blocks", str(p.get("blocks", 256)),
            "--threads", str(p.get("threads", 256)),
            "--dtype", str(p["dtype"]),
            "--seed", str(p["seed"]),
            "--device", str(device),
            "--out", out_prefix,
        ] + self._input_args()

    def collect_output(self, out_prefix):
        binpath = out_prefix + ".bin"
        metapath = out_prefix + ".json"
        meta = util.read_json(metapath) if os.path.exists(metapath) else {}
        return {"bin": binpath, "meta": meta, "exists": os.path.exists(binpath)}
