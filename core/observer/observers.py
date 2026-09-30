"""Phase-1 observers: Output, Runtime, GPU."""

import os

from core import util
from core.observer.base import Observer
from core.oracle.comparator import compare, compare_outputs

CRASH_PATTERNS = [
    "ERROR: CUDA error",
    "ERROR FAIL",
    "an illegal memory access",
    "misaligned address",
    "CUDA_ERROR_INVALID",
    "device-side assert",
    "unspecified launch failure",
    "Segmentation fault",
    "core dumped",
    "Bus error",
]


class RuntimeObserver(Observer):
    name = "runtime"

    def __init__(self):
        pass

    def observe(self, run):
        stdout = run.get("stdout", "") or ""
        stderr = run.get("stderr", "") or ""
        blob = stdout + "\n" + stderr
        crash = False
        reason = None

        if run.get("timeout"):
            crash, reason = True, "timeout"
        elif run.get("exit_code", 0) != 0:
            crash, reason = True, "non-zero exit code %s" % run.get("exit_code")
        else:
            for pat in CRASH_PATTERNS:
                if pat in blob:
                    crash, reason = True, "crash signature: %s" % pat
                    break

        return {
            "exit_code": run.get("exit_code"),
            "runtime_sec": round(run.get("runtime_sec", 0.0), 6),
            "timeout": bool(run.get("timeout")),
            "crash": crash,
            "crash_reason": reason,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "stdout_sha256": util.sha256_text(stdout),
            "stderr_sha256": util.sha256_text(stderr),
        }


class OutputObserver(Observer):
    name = "output"

    def __init__(self, abs_tol=0.0, rel_tol=0.0):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def observe(self, golden_bin, fault_bin):
        return compare(golden_bin, fault_bin, self.abs_tol, self.rel_tol)

    def observe_outputs(self, golden_prefix, fault_prefix, specs):
        return compare_outputs(golden_prefix, fault_prefix, specs,
                               self.abs_tol, self.rel_tol)


class GPUObserver(Observer):
    """Best-effort nvidia-smi telemetry. Never fails the experiment."""

    name = "gpu"

    def __init__(self, gpu_index):
        self.gpu_index = gpu_index

    def observe(self, context=None):
        try:
            res = util.run_cmd(
                [
                    "nvidia-smi", "-i", str(self.gpu_index),
                    "--query-gpu=temperature.gpu,power.draw,utilization.gpu,"
                    "memory.used,memory.total,name",
                    "--format=csv,noheader,nounits",
                ],
                timeout=15,
            )
            if res["exit_code"] != 0:
                return {"available": False}
            parts = [p.strip() for p in res["stdout"].strip().split(",")]
            return {
                "available": True,
                "temperature_c": parts[0],
                "power_w": parts[1],
                "utilization_pct": parts[2],
                "memory_used_mb": parts[3],
                "memory_total_mb": parts[4],
                "name": parts[5] if len(parts) > 5 else "",
            }
        except Exception as e:  # pragma: no cover - telemetry must never fail runs
            return {"available": False, "error": str(e)}


class Observers(object):
    """Aggregate of the Phase-1 observers."""

    def __init__(self, lab):
        self.lab = lab
        self.runtime = RuntimeObserver()
        self.output = OutputObserver(
            abs_tol=lab.oracle.get("abs_tol", 0.0),
            rel_tol=lab.oracle.get("rel_tol", 0.0),
        )
        self.gpu = GPUObserver(lab.gpu)
