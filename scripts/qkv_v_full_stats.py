import csv
import math
import statistics

rows = list(csv.DictReader(open("results/qkv_v_full/runs_all.csv", encoding="utf-8")))
cps = ["C0", "C1", "C2", "C3"]
for cp in cps:
    vals = []
    rel = []
    mis = []
    huge = 0
    for r in rows:
        if r["run_d0"] != "SDC":
            continue
        try:
            v = float(r[cp + "_max_abs"])
            q = float(r[cp + "_rel_l2"])
            m = int(float(r[cp + "_mismatch"]))
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
            huge += v > 1e10
        if math.isfinite(q):
            rel.append(q)
        mis.append(m)
    print(cp, "n", len(mis), "mismatch median", statistics.median(mis),
          "mismatch min/max", min(mis), max(mis),
          "maxabs median", statistics.median(vals) if vals else None,
          "maxabs max", max(vals) if vals else None,
          "huge", huge,
          "rel_l2 median", statistics.median(rel) if rel else None,
          "rel_l2 max", max(rel) if rel else None)
print("by bit")
for b in ["28", "29", "30"]:
    rs = [r for r in rows if r["bit"] == b]
    print("bit", b, "D0", {k: sum(r["run_d0"] == k for r in rs) for k in ("SDC", "Masked")},
          "D1", {k: sum(r["run_d1"] == k for r in rs) for k in ("SDC", "Masked", "DUE")})
    for cp in cps:
        vals = [float(r[cp + "_max_abs"]) for r in rs if r["run_d0"] == "SDC" and math.isfinite(float(r[cp + "_max_abs"]))]
        if vals:
            print("  ", cp, "median_maxabs", statistics.median(vals), "max", max(vals))
