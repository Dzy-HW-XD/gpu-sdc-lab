#!/usr/bin/env python3
"""Run repeated clean controls for the qkv_v block replay."""
import json
import os
import subprocess
import sys
import time

ROOT = "/home/dzy/gpu-sdc-lab"
RUNNER = ROOT + "/scripts/qkv_v_block_run.py"
INPUT = ROOT + "/results/qkv_v_block/inputs"
OUT = ROOT + "/results/qkv_v_block/clean_controls"
os.makedirs(OUT, exist_ok=True)
rows = []
t0 = time.time()
for layer in (0, 14, 27):
    for i in range(10):
        out = os.path.join(OUT, "layer%02d_%02d.json" % (layer, i))
        cmd = [sys.executable, RUNNER, "--input", os.path.join(INPUT, "layer%02d.pt" % layer),
               "--out", out, "--mode", "clean"]
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=120, text=True)
        rec = {"layer": layer, "rep": i, "exit": p.returncode, "out": out}
        if os.path.exists(out):
            d = json.load(open(out, encoding="utf-8"))
            rec["block_equal"] = d.get("block_bitwise_equal")
            rec["attn_equal"] = d.get("attn_bitwise_equal")
            rec["v_equal"] = d.get("v_bitwise_equal")
            rec["elapsed"] = d.get("elapsed_s")
        rows.append(rec)
        print(rec, flush=True)
json.dump(rows, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=2)
print("elapsed", round(time.time() - t0, 1))
