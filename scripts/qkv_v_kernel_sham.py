#!/usr/bin/env python3
"""9 sham runs per CUTLASS variant with a DUMMY long-name injector."""
import csv
import json
import os
import qkv_v_kernel_campaign as c

c.INJECTOR = c.ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/injector_dummy_long/injector.so"
c.RUN_ROOT = c.BASE + "/sham"
os.makedirs(c.RUN_ROOT, exist_ok=True)
rows = list(csv.DictReader(open(c.MANIFEST, encoding="utf-8")))
sites = []
for label, _ in c.VARIANTS:
    for bit in c.BITS:
        rs = [r for r in rows if r["variant"] == label and int(r["bit"]) == bit
              and int(r["reserve"]) == 0]
        sites.extend(rs[:3])
out = []
for s in sites:
    rec = c.run_one(s, gpu=0, resume=False, injector=c.INJECTOR, run_root=c.RUN_ROOT)
    out.append(rec)
    print(s["site_id"], rec["injected"], rec["run_d0"], rec["run_d1"], flush=True)
path = c.BASE + "/sham.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print("wrote", path, len(out))
