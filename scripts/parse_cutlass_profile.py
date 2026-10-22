import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--profile", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--inst-id", type=int, default=0)
ap.add_argument("--op-seed", default="0.5")
ap.add_argument("--bit-seed", default="0.953125")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rec = None
for line in open(args.profile, encoding="utf-8", errors="ignore"):
    if not line.startswith("NVBit-igprofile"):
        continue
    m = re.search(r"index:\s*(\d+); kernel_name:\s*(.*?);", line)
    g = re.search(r"(?:^|[;,])\s*mma:\s*(\d+)", line)
    if m and g and int(g.group(1)) > 0:
        rec = {"index": int(m.group(1)), "name": m.group(2), "mma": int(g.group(1))}
        break
if rec is None:
    raise SystemExit("no MMA kernel")
json.dump(rec, open(os.path.join(args.outdir, "kernel.json"), "w", encoding="utf-8"), indent=2)
open(os.path.join(args.outdir, "name.txt"), "w", encoding="utf-8").write(rec["name"])
open(os.path.join(args.outdir, "nvbitfi-injection-info.txt"), "w", encoding="utf-8").write(
    "7\n0\n%s\n%d\n%d\n%s\n%s\n" % (rec["name"], rec["index"], args.inst_id,
                                      args.op_seed, args.bit_seed))
print(rec["index"], rec["mma"], len(rec["name"]))
