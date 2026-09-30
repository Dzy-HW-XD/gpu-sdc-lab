"""Mini-training workload: deterministic 2-layer MLP forward + one backward step.

Emits several named tensors so error propagation can be observed at every layer:
A1 (activation), logits (Z2), probs (P), loss (per-batch), dW1, dW2.
"""

import os

from core import util
from core.workload.base import Workload

DEFAULT_PARAMS = {"B": 64, "D": 64, "H": 128, "O": 10, "dtype": "fp32", "seed": 1234}


class MLPWorkload(Workload):
    name = "mlp"
    kernel_token = "lin1"
    input_flags = {"X": "--input", "W1": "--weight1", "W2": "--weight2"}
    output_specs = [
        {"name": "A1", "suffix": "_a1.bin", "format": "f32"},
        {"name": "logits", "suffix": "_logits.bin", "format": "f32"},
        {"name": "probs", "suffix": "_probs.bin", "format": "f32"},
        {"name": "loss", "suffix": "_loss.bin", "format": "f32"},
        {"name": "dW1", "suffix": "_dW1.bin", "format": "f32"},
        {"name": "dW2", "suffix": "_dW2.bin", "format": "f32"},
    ]

    def __init__(self, params, lab):
        merged = dict(DEFAULT_PARAMS)
        merged.update(params or {})
        super(MLPWorkload, self).__init__(merged, lab)
        self.dir = os.path.join(lab.root, "workloads", "mlp")
        self.src = os.path.join(self.dir, "mlp.cu")
        self.bin = os.path.join(self.dir, "mlp")

    def prepare(self):
        if not os.path.exists(self.bin) or os.path.getmtime(self.src) > os.path.getmtime(self.bin):
            self.build()
        return super(MLPWorkload, self).prepare()

    def build(self):
        nvcc = os.path.join(self.lab.cuda_home, "bin", "nvcc")
        arch = self.lab.runtime.get("nvbit_arch", "sm_80")
        res = util.run_cmd([nvcc, "-O3", "-std=c++11", "-arch=%s" % arch,
                            "-o", self.bin, self.src], cwd=self.dir, timeout=600)
        if res["exit_code"] != 0:
            raise RuntimeError("mlp build failed:\n%s\n%s" % (res["stdout"], res["stderr"]))

    def input_size(self, name):
        p = self.params
        if name == "X":
            return int(p["B"]) * int(p["D"])
        if name == "W1":
            return int(p["D"]) * int(p["H"])
        if name == "W2":
            return int(p["H"]) * int(p["O"])
        raise KeyError(name)

    def command(self, out_prefix, device=0):
        p = self.params
        return [self.bin, "--B", str(p["B"]), "--D", str(p["D"]),
                "--H", str(p["H"]), "--O", str(p["O"]),
                "--seed", str(p["seed"]), "--device", str(device),
                "--out", out_prefix] + self._input_args()
