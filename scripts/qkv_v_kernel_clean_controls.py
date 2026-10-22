#!/usr/bin/env python3
"""10 clean downstream runs per CUTLASS variant."""
import json
import os
import subprocess
import sys
import time

ROOT = "/home/dzy/gpu-sdc-lab"
RUNNER = ROOT + "/scripts/qkv_v_from_v_run.py"
BASE = ROOT + "/results/qkv_v_kernels"
variants = [("K1", "k1"), ("K2a", "k2"), ("K2b", "k3"), ("K3", "k4")]
rows = []
t0 = time.time()
for label, binary in variants:
    d = os.path.join(BASE, "clean_controls", binary)
    os.makedirs(d, exist_ok=True)
    for i in range(10):
        out = os.path.join(d, "clean_%02d.pt" % i)
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [sys.executable, RUNNER, "--v", os.path.join(BASE, "clean", binary, "V.bin"),
               "--out", out, "--mode", "clean"]
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)
        rows.append({"variant": label, "rep": i, "exit": p.returncode, "out": out})
        print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(BASE, "clean_controls", "summary.json"), "w", encoding="utf-8"), indent=2)
print("elapsed", round(time.time() - t0, 1))
