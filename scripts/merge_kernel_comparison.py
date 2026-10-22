import csv
import glob

rows = []
for r in csv.DictReader(open("results/qkv_v_full/runs_all.csv", encoding="utf-8")):
    r["variant"] = "K0"
    r["binary"] = "cublas"
    rows.append(r)
for f in sorted(glob.glob("results/qkv_v_kernels/runs_s*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        rows.append(r)
fields = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with open("results/qkv_v_kernels/comparison_all.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(len(rows), len(fields), sum(r.get("variant") == "K0" for r in rows))
