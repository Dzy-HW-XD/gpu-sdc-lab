"""GEMM workload: deterministic FP32 C = A x B."""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {
    "M": 1024,
    "N": 1024,
    "K": 1024,
    "dtype": "fp32",
    "seed": 1234,
    "pattern": "normal",
    "block": [16, 16],
}


class GEMMWorkload(Workload):
    name = "gemm"
    kernel_token = "gemm"
    input_flags = {"A": "--inputA", "B": "--inputB"}
    output_specs = [{"name": "C", "suffix": ".bin", "format": "f32"}]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(GEMMWorkload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "gemm")
        self.src = os.path.join(self.dir, "gemm.cu")
        self.bin = os.path.join(self.dir, "gemm")

    def prepare(self):
        if not os.path.exists(self.bin) or (
            os.path.getmtime(self.src) > os.path.getmtime(self.bin)
        ):
            self.build()
        super(GEMMWorkload, self).prepare()
        return self.bin

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            raise RuntimeError("nvcc not found at %s" % nvcc)
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        cmd = [
            nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
            "-o", self.bin, self.src,
        ]
        res = util.run_cmd(cmd, cwd=self.dir, timeout=600)
        if res["exit_code"] != 0:
            raise RuntimeError("GEMM build failed:\n%s\n%s" % (res["stdout"], res["stderr"]))
        return self.bin

    def input_size(self, name):
        p = self.params
        if name == "A":
            return int(p["M"]) * int(p["K"])
        if name == "B":
            return int(p["K"]) * int(p["N"])
        raise KeyError(name)

    def command(self, out_prefix, device=0):
        p = self.params
        cmd = [
            self.bin,
            "--M", str(p["M"]),
            "--N", str(p["N"]),
            "--K", str(p["K"]),
            "--dtype", str(p["dtype"]),
            "--seed", str(p["seed"]),
            "--pattern", str(p["pattern"]),
            "--device", str(device),
            "--out", out_prefix,
        ]
        op_mode = str(p.get("op_mode", "fma"))
        if op_mode and op_mode != "fma":
            cmd += ["--op-mode", op_mode]
        return cmd + self._input_args()

    def collect_output(self, out_prefix):
        binpath = out_prefix + ".bin"
        metapath = out_prefix + ".json"
        meta = {}
        if os.path.exists(metapath):
            try:
                meta = util.read_json(metapath)
            except (ValueError, OSError):
                meta = {}
        return {"prefix": out_prefix, "bin": binpath, "meta": meta,
                "exists": os.path.exists(binpath)}

    @property
    def output_bin(self):
        return None
