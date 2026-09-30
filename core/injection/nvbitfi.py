"""NVBitFI backend (NVBit dynamic binary instrumentation fault injection).

Mechanism: the NVBitFI injector tool is an NVBit tool (.so) that is LD_PRELOADed
into the workload process. One injection info file selects a single dynamic
thread-instruction and a single destination register / bit. The resulting
`nvbitfi-injection-log-temp.txt` records the fault metadata (register, opcode,
pcOffset, thread id, mask, before/after values).
"""

import os
import re

from core import util
from core.injection.base import FaultInjector

PROFILE_LOG = "nvbitfi-injection-log-temp.txt"
INJECTION_INFO = "nvbitfi-injection-info.txt"

GROUP_KEYS = ("fp64", "fp32", "ld", "pr", "nodest", "others", "atom", "gppr", "gp")


def parse_profile_line(line):
    if not line.startswith("NVBit-igprofile"):
        return None
    body = line.split(";", 1)[1]
    rec = {}

    def _int(pat):
        m = re.search(pat, body)
        return int(m.group(1)) if m else None

    rec["index"] = _int(r"index:\s*(\d+)")
    rec["ctas"] = _int(r"ctas:\s*(\d+)")
    rec["instrs"] = _int(r"instrs:\s*(\d+)")
    m = re.search(r"kernel_name:\s*(.*?);", body)
    rec["name"] = m.group(1).strip() if m else None

    groups = {}
    tail = line.rsplit(";", 1)[-1]
    for kv in tail.split(","):
        kv = kv.strip()
        if ":" in kv:
            k, _, v = kv.partition(":")
            k, v = k.strip(), v.strip()
            if k in GROUP_KEYS:
                try:
                    groups[k] = int(v)
                except ValueError:
                    groups[k] = 0
    rec["groups"] = groups
    return rec


def parse_injection_log(text):
    meta = {
        "injected": False,
        "log_index": None,
        "kernel_name": None,
        "ctas": None,
        "instrs": None,
        "group_counts": {},
        "mask": None,
        "before_val": None,
        "after_val": None,
        "reg_no": None,
        "opcode": None,
        "pc_offset": None,
        "tid": None,
        "error_not_injected": False,
        "kernel_error": False,
    }
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Injection data"):
            meta["injected"] = True
        elif s.startswith("kernel_name:"):
            meta["kernel_name"] = s.split(":", 1)[1].strip()
        elif s.startswith("index:"):
            try:
                meta["log_index"] = int(s.split(":", 1)[1])
            except ValueError:
                pass
        elif s.startswith("ctas:"):
            try:
                meta["ctas"] = int(s.split(":", 1)[1])
            except ValueError:
                pass
        elif s.startswith("instrs:"):
            try:
                meta["instrs"] = int(s.split(":", 1)[1])
            except ValueError:
                pass
        elif s.startswith("mask:"):
            meta["mask"] = s.split(":", 1)[1].strip()
            meta["injected"] = True
        elif s.startswith("beforeVal:"):
            try:
                b, a = s.split(";")
                meta["before_val"] = b.split(":", 1)[1].strip()
                meta["after_val"] = a.split(":", 1)[1].strip()
            except ValueError:
                pass
        elif s.startswith("regNo:"):
            try:
                meta["reg_no"] = int(s.split(":", 1)[1])
            except ValueError:
                pass
        elif s.startswith("opcode:"):
            meta["opcode"] = s.split(":", 1)[1].strip()
        elif s.startswith("pcOffset:"):
            meta["pc_offset"] = s.split(":", 1)[1].strip()
        elif s.startswith("tid:"):
            try:
                meta["tid"] = int(s.split(":", 1)[1])
            except ValueError:
                pass
        elif s.startswith("grp "):
            for m in re.finditer(r"grp\s+(\d+):\s*(\d+)", s):
                meta["group_counts"][int(m.group(1))] = int(m.group(2))
        if "Error not injected" in s:
            meta["error_not_injected"] = True
        if "ERROR FAIL in kernel execution" in s:
            meta["kernel_error"] = True
    return meta


class NVBitFIInjector(FaultInjector):
    name = "nvbitfi"

    def available(self):
        return os.path.exists(self.lab.injector_lib) and os.path.exists(self.lab.profiler_lib)

    def describe(self):
        d = super(NVBitFIInjector, self).describe()
        d.update({
            "injector_lib": self.lab.injector_lib,
            "profiler_lib": self.lab.profiler_lib,
        })
        return d

    def _env(self, lib):
        cuda_bin = os.path.join(self.lab.cuda_home, "bin")
        cuda_lib = os.path.join(self.lab.cuda_home, "lib64")
        return {
            "CUDA_VISIBLE_DEVICES": str(self.lab.gpu),
            "LD_PRELOAD": lib,
            "NOBANNER": "1",
            "TOOL_VERBOSE": "0",
            "PATH": cuda_bin + ":" + os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        }

    def profile(self, workload, workdir):
        util.ensure_dir(workdir)
        out_prefix = os.path.join(workdir, "profile")
        res = workload.run(
            out_prefix, device=0, cwd=workdir,
            extra_env=self._env(self.lab.profiler_lib),
            timeout=self.lab.runtime.get("profile_timeout", 1800),
        )
        log_path = os.path.join(workdir, PROFILE_LOG)
        raw = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        runs = []
        for line in raw.splitlines():
            rec = parse_profile_line(line)
            if rec:
                runs.append(rec)
        return {"runs": runs, "raw": raw, "run": res, "log_path": log_path}

    def inject(self, workload, spec, run_dir, device=0, timeout=None):
        util.ensure_dir(run_dir)
        info_path = os.path.join(run_dir, INJECTION_INFO)
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(spec.injection_info_text())
        out_prefix = os.path.join(run_dir, "out")
        res = workload.run(
            out_prefix, device=device, cwd=run_dir,
            extra_env=self._env(self.lab.injector_lib),
            timeout=timeout or self.lab.runtime.get("default_timeout", 300),
        )
        log_path = os.path.join(run_dir, PROFILE_LOG)
        raw = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        meta = parse_injection_log(raw)
        output = workload.collect_output(out_prefix)
        return {
            "run": res,
            "fault_meta": meta,
            "output": output,
            "log": raw,
            "info_path": info_path,
            "out_prefix": out_prefix,
        }
