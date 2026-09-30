"""Conv2d workload: deterministic FP32 direct 2-D convolution."""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {"Cin": 8, "Cout": 16, "H": 32, "W": 32, "kernel": 3,
                  "dtype": "fp32", "seed": 1234}


class Conv2dWorkload(Workload):
    name = "conv2d"
    kernel_token = "conv2d"
    input_flags = {"X": "--input", "Wt": "--weight"}
    output_specs = [{"name": "Y", "suffix": ".bin", "format": "f32"}]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(Conv2dWorkload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "conv2d")
        self.src = os.path.join(self.dir, "conv2d.cu")
        self.bin = os.path.join(self.dir, "conv2d")

    def prepare(self):
        if not os.path.exists(self.bin) or os.path.getmtime(self.src) > os.path.getmtime(self.bin):
            self.build()
        return super(Conv2dWorkload, self).prepare()

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        res = util.run_cmd([nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
                            "-o", self.bin, self.src], cwd=self.dir, timeout=600)
        if res["exit_code"] != 0:
            raise RuntimeError("conv2d build failed:\n%s\n%s" % (res["stdout"], res["stderr"]))

    def input_size(self, name):
        p = self.params
        if name == "X":
            return int(p["Cin"]) * int(p["H"]) * int(p["W"])
        if name == "Wt":
            return int(p["Cout"]) * int(p["Cin"]) * int(p["kernel"]) * int(p["kernel"])
        raise KeyError(name)

    def command(self, out_prefix, device=0):
        p = self.params
        return [self.bin,
                "--Cin", str(p["Cin"]), "--Cout", str(p["Cout"]),
                "--H", str(p["H"]), "--W", str(p["W"]),
                "--kernel", str(p["kernel"]), "--seed", str(p["seed"]),
                "--device", str(device), "--out", out_prefix] + self._input_args()
