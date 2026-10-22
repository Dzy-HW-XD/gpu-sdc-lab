#!/usr/bin/env python3
"""Run 27 no-flip sham controls with the DUMMY injector."""
import csv
import json
import os

import qkv_v_block_campaign as c

c.OUT_DIR = c.ROOT + "/results/qkv_v_block/sham"
c.INJECTOR = c.ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/injector_dummy/injector.so"
os.makedirs(c.OUT_DIR, exist_ok=True)
rows = list(csv.DictReader(open(c.MANIFEST, encoding="utf-8")))
sites = []
for layer in c.LAYERS:
    for bit in c.BITS:
        rs = [r for r in rows if int(r["layer"]) == layer and int(r["bit"]) == bit and int(r["reserve"]) == 0]
        sites.extend(rs[:3])
out = []
for s in sites:
    rec = c.run_one(s, gpu=0, resume=False)
    out.append(rec)
    print(s["site_id"], "injected", rec["injected"], "V", rec["outcome_v"],
          "attn", rec["outcome_attn"], "block", rec["outcome_block"], flush=True)
path = c.ROOT + "/results/qkv_v_block/sham.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print("wrote", path, len(out))
