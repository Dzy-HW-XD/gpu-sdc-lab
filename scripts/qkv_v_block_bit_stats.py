import csv, math, statistics
rows = list(csv.DictReader(open("results/qkv_v_block/runs_all.csv", encoding="utf-8")))
for b in ["28", "29", "30"]:
    rs = [r for r in rows if r["bit"] == b]
    vals, non, huge = [], 0, 0
    for r in rs:
        v = -1.0
        try:
            x = float(r["max_abs_block"])
            if math.isfinite(x):
                v = x
                vals.append(x)
                huge += x > 1e10
            else:
                non += 1
        except Exception:
            non += 1
    print("bit", b, "n", len(rs), "finite", len(vals), "nonfinite", non,
          "huge", huge, "median", statistics.median(vals) if vals else None,
          "max", max(vals) if vals else None,
          "mask", sum(r["outcome_block"] == "Mask" for r in rs))
    print("top", [round(math.log10(x), 1) for x in sorted(vals)[-5:]])
