#!/usr/bin/env python3
"""Profile the four CUTLASS qkv_v kernels with NVBitFI."""
import json
import os
import re
import subprocess

ROOT = "/home/dzy/gpu-sdc-lab"
PROFILER = ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/profiler/profiler.so"
BIN = ROOT + "/results/qkv_v_kernels/bin"
DATA = ROOT + "/results/qkv_v_kernels/data"
OUT = ROOT + "/results/qkv_v_kernels/profile"
variants = [("K1", "k1"), ("K2a", "k2"), ("K2b", "k3"), ("K3", "k4")]

env = os.environ.copy()
env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
env["LD_PRELOAD"] = PROFILER
env["NOBANNER"] = "1"
env["TOOL_VERBOSE"] = "0"
env["CUDA_VISIBLE_DEVICES"] = "0"

for label, exe in variants:
    d = os.path.join(OUT, exe)
    os.makedirs(d, exist_ok=True)
    cmd = [os.path.join(BIN, exe), "--M", "2264", "--N", "256", "--K", "1536",
           "--inputA", os.path.join(DATA, "A.bin"),
           "--inputB", os.path.join(DATA, "W.bin"),
           "--out", os.path.join(d, "V.bin")]
    with open(os.path.join(d, "profiler.out"), "w") as f:
        p = subprocess.run(cmd, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
    print(label, "exit", p.returncode)
    text = open(os.path.join(d, "nvbitfi-injection-log-temp.txt"), encoding="utf-8", errors="ignore").read()
    rec = None
    for line in text.splitlines():
        if not line.startswith("NVBit-igprofile"):
            continue
        m = re.search(r"index:\s*(\d+); kernel_name:\s*(.*?);", line)
        g = re.search(r"(?:^|[;,])\s*mma:\s*(\d+)", line)
        if m and g and int(g.group(1)) > 0:
            rec = {"label": label, "binary": exe, "index": int(m.group(1)),
                   "name": m.group(2), "mma": int(g.group(1))}
            break
    if rec is None:
        raise SystemExit("no MMA kernel for " + label)
    json.dump(rec, open(os.path.join(d, "kernel.json"), "w", encoding="utf-8"), indent=2)
    print(rec["label"], rec["index"], rec["mma"], len(rec["name"]))
