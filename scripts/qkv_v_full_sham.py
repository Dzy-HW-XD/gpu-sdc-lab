#!/usr/bin/env python3
"""27 no-flip sham controls for the full-stack layer14 campaign."""
import csv
import os
import qkv_v_full_campaign as c

c.OUT_DIR = c.ROOT + "/results/qkv_v_full/sham"
c.INJECTOR = c.ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/injector_dummy/injector.so"
os.makedirs(c.OUT_DIR, exist_ok=True)
rows = list(csv.DictReader(open(c.MANIFEST, encoding="utf-8")))
sites = []
for bit in c.BITS:
    rs = [r for r in rows if int(r["bit"]) == bit and int(r["reserve"]) == 0]
    sites.extend(rs[:9])
out = []
for s in sites:
    rec = c.run_one(s, gpu=0, resume=False)
    out.append(rec)
    print(s["site_id"], rec["injected"], rec["run_d0"], rec["run_d1"], flush=True)
import json
path = c.ROOT + "/results/qkv_v_full/sham.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print("wrote", path, len(out))
