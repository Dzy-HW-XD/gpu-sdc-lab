"""Softmax workload: deterministic FP32 row-wise softmax."""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {"R": 1024, "C": 1024, "dtype": "fp32", "seed": 1234}


class SoftmaxWorkload(Workload):
    name = "softmax"
    kernel_token = "softmax"
    input_flags = {"X": "--input"}
    output_specs = [{"name": "Y", "suffix": ".bin", "format": "f32"}]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(SoftmaxWorkload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "softmax")
        self.src = os.path.join(self.dir, "softmax.cu")
        self.bin = os.path.join(self.dir, "softmax")

    def prepare(self):
        if not os.path.exists(self.bin) or os.path.getmtime(self.src) > os.path.getmtime(self.bin):
            self.build()
        return super(SoftmaxWorkload, self).prepare()

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        res = util.run_cmd([nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
                            "-o", self.bin, self.src], cwd=self.dir, timeout=600)
        if res["exit_code"] != 0:
            raise RuntimeError("softmax build failed:\n%s\n%s" % (res["stdout"], res["stderr"]))

    def input_size(self, name):
        if name == "X":
            return int(self.params["R"]) * int(self.params["C"])
        raise KeyError(name)

    def command(self, out_prefix, device=0):
        p = self.params
        return [self.bin, "--R", str(p["R"]), "--C", str(p["C"]),
                "--seed", str(p["seed"]), "--device", str(device),
                "--out", out_prefix] + self._input_args()
