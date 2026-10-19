"""Causal self-attention forward workload (BF16 GEMM, FP32 accumulation).

Single-head (H=1, dh=D) and multi-head (H>1, dh=D/H) standard self-attention:

    Q = X Wq, K = X Wk, V = X Wv            (cuBLAS BF16 x BF16 -> FP32)
    Sh = Qh Kh^T / sqrt(dh)                 (per-head cuBLAS launch)
    Ph = causal_softmax(Sh)                 (explicit FP32)
    Oh = Ph Vh                              (per-head cuBLAS launch)
    Y  = concat(Oh) Wo

Score and PV are launched once per head so a fault can be attributed to a
specific head by kernel-invocation selection (see docs/ATTENTION.md).
"""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {
    "S": 128,
    "D": 2048,
    "H": 1,
    "dtype": "bf16",
    "mask": "causal",
    "seed": 1234,
    "dump": 1,
}


class AttentionForwardWorkload(Workload):
    name = "attention_forward"
    # cuBLAS kernels are named "...gemm..."; falls back to the largest kernel.
    kernel_token = "gemm"
    input_flags = {"X": "--inputX"}
    input_dtype = "bf16"
    output_specs = [{"name": "Y", "suffix": ".Y.bin", "format": "f32"}]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(AttentionForwardWorkload, self).__init__(merged, lab)
        if int(merged["D"]) % int(merged["H"]) != 0:
            raise ValueError("attention_forward: D (%s) must be divisible by H (%s)"
                             % (merged["D"], merged["H"]))
        self.dir = os.path.join(lab.root, "workloads", "attention_forward")
        self.src = os.path.join(self.dir, "attention_forward.cu")
        self.bin = os.path.join(self.dir, "attention_forward")

    @property
    def head_dim(self):
        return int(self.params["D"]) // int(self.params["H"])

    def prepare(self):
        if not os.path.exists(self.bin) or (
            os.path.getmtime(self.src) > os.path.getmtime(self.bin)
        ):
            self.build()
        super(AttentionForwardWorkload, self).prepare()
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
            raise RuntimeError("attention_forward build failed:\n%s\n%s"
                               % (res["stdout"], res["stderr"]))
        return self.bin

    def input_size(self, name):
        p = self.params
        if name == "X":
            return int(p["S"]) * int(p["D"])
        raise KeyError(name)

    def command(self, out_prefix, device=0):
        p = self.params
        return [
            self.bin,
            "--S", str(p["S"]),
            "--D", str(p["D"]),
            "--H", str(p["H"]),
            "--dtype", "bf16",
            "--seed", str(p["seed"]),
            "--device", str(device),
            "--out", out_prefix,
            "--dump", str(p.get("dump", 1)),
        ] + self._input_args()

    def select_kernel(self, prof, case):
        """Map a (stage, head) case to its kernel launch.

        The workload launches, in order, per-head GEMM kernels:
            [Q, K, V] projections, then H score GEMMs, then H PV GEMMs, then Wo.
        Only kernels that actually execute tensor-core instructions are
        considered (mma > 0), which skips the cast/softmax/concat kernels that
        sit between the GEMM launches.
        """
        runs = [r for r in (prof.get("runs") or [])
                if int((r.get("groups") or {}).get("mma", 0)) > 0]
        if not runs:
            return None
        H = int(self.params["H"])
        labels = case.get("labels") or {}
        stage = labels.get("stage")
        head = labels.get("head")
        if stage == "qkv_q":
            return runs[0] if len(runs) > 0 else None
        if stage == "qkv_k":
            return runs[1] if len(runs) > 1 else None
        if stage == "qkv_v":
            return runs[2] if len(runs) > 2 else None
        if stage == "qk":
            i = 3 + int(head)
            return runs[i] if i < 3 + H else None
        if stage == "pv":
            i = 3 + H + int(head)
            return runs[i] if i < 3 + 2 * H else None
        if stage == "wo":
            i = 3 + 2 * H
            return runs[i] if i < len(runs) else None
        return None

    def collect_output(self, out_prefix):
        binpath = out_prefix + ".Y.bin"
        metapath = out_prefix + ".json"
        meta = {}
        if os.path.exists(metapath):
            try:
                meta = util.read_json(metapath)
            except (ValueError, OSError):
                meta = {}
        return {"prefix": out_prefix, "bin": binpath, "meta": meta,
                "exists": os.path.exists(binpath)}
