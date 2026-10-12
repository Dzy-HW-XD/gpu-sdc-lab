"""BF16 x BF16 Tensor Core GEMM workload (cuBLAS, FP32 accumulation).

C = A x B with bf16 A (MxK), B (KxN) and fp32 C (MxN). The multiply is executed
by cuBLAS (HMMA / Tensor Core on sm_80) with FP32 accumulators -- the fault
target of the TC experiments.
"""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {
    "M": 1024,
    "N": 1024,
    "K": 1024,
    "dtype": "bf16",
    "seed": 1234,
    "pattern": "normal",
}


class GEMMBF16Workload(Workload):
    name = "gemm_bf16"
    # cuBLAS BF16 kernel names look like
    #   "ampere_s16816gemm_bf16_64x64_..._nn"; match on "gemm" (falls back to
    #   the largest kernel if there are several).
    kernel_token = "gemm"
    input_flags = {"A": "--inputA", "B": "--inputB"}
    input_dtype = "bf16"
    output_specs = [{"name": "C", "suffix": ".bin", "format": "f32"}]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(GEMMBF16Workload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "gemm_bf16")
        self.src = os.path.join(self.dir, "gemm_bf16.cu")
        self.bin = os.path.join(self.dir, "gemm_bf16")

    def prepare(self):
        if not os.path.exists(self.bin) or (
            os.path.getmtime(self.src) > os.path.getmtime(self.bin)
        ):
            self.build()
        super(GEMMBF16Workload, self).prepare()
        return self.bin

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            raise RuntimeError("nvcc not found at %s" % nvcc)
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        cuda_lib = os.path.join(self.lab.cuda_home, "lib64")
        cmd = [
            nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
            "-I%s" % os.path.join(self.lab.cuda_home, "include"),
            "-L%s" % cuda_lib, "-lcublas",
            "-o", self.bin, self.src,
        ]
        res = util.run_cmd(cmd, cwd=self.dir, timeout=600,
                           extra_env={"LD_LIBRARY_PATH": cuda_lib + ":" +
                                      os.environ.get("LD_LIBRARY_PATH", "")})
        if res["exit_code"] != 0:
            raise RuntimeError("gemm_bf16 build failed:\n%s\n%s"
                               % (res["stdout"], res["stderr"]))
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
        return [
            self.bin,
            "--M", str(p["M"]),
            "--N", str(p["N"]),
            "--K", str(p["K"]),
            "--dtype", "bf16",
            "--seed", str(p["seed"]),
            "--device", str(device),
            "--out", out_prefix,
        ] + self._input_args()

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
