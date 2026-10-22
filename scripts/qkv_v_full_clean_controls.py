#!/usr/bin/env python3
"""30 clean full-stack controls for layer14 -> layer27."""
import json
import os
import subprocess
import sys
import time

ROOT = "/home/dzy/gpu-sdc-lab"
RUNNER = ROOT + "/scripts/qkv_v_fullstack_run.py"
INPUT = ROOT + "/results/qkv_v_block/inputs/layer14.pt"
OUT = ROOT + "/results/qkv_v_full/clean_controls"
os.makedirs(OUT, exist_ok=True)
rows = []
t0 = time.time()
for i in range(30):
    out = os.path.join(OUT, "clean_%02d.json" % i)
    cmd = [sys.executable, RUNNER, "--input", INPUT, "--out", out, "--mode", "clean"]
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120, text=True)
    rec = {"rep": i, "exit": p.returncode, "out": out}
    if os.path.exists(out):
        d = json.load(open(out, encoding="utf-8"))
        rec.update({k: d.get(k) for k in
                   ("v_bitwise_equal", "attn_bitwise_equal",
                    "block_bitwise_equal", "final_bitwise_equal",
                    "nan_inf_any", "elapsed_s")})
    rows.append(rec)
    print(rec, flush=True)
json.dump(rows, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=2)
print("elapsed", round(time.time() - t0, 1))
