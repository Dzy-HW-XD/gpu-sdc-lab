#!/usr/bin/env python3
"""Pick one representative SDC run per (stage, bit, position).

Reads <exp>/observations.jsonl and chooses the SDC run with the largest Y
relative-L2 in each group (heads pooled). Writes
<exp>/propagation_manifest.csv with golden/faulty tensor prefixes.

Usage: attention_select_runs.py --exp results/046_attn_mha_bit_sweep
"""
import argparse
import csv
import json
import os
import sys

SUFFIX = ".Y.bin"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args(argv)

    obs_path = os.path.join(args.exp, "observations.jsonl")
    if not os.path.exists(obs_path):
        raise SystemExit("missing %s" % obs_path)

    groups = {}
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("classification") != "SDC":
                continue
            o = rec.get("observation") or {}
            rel = o.get("relative_l2_error")
            if rel in (None, ""):
                continue
            labels = rec.get("labels") or {}
            key = (labels.get("stage"), labels.get("bit"),
                   labels.get("position", "all"))
            if key not in groups or float(rel) > float(groups[key]["rel"]):
                groups[key] = {"rec": rec, "rel": rel}

    rows = []
    for key in sorted(groups, key=lambda k: (str(k[0]), int(k[1]), str(k[2]))):
        g = groups[key]
        rec = g["rec"]
        labels = rec.get("labels") or {}
        gbin = rec.get("golden_bin")
        fbin = rec.get("raw_out_bin")
        if not (gbin and fbin and gbin.endswith(SUFFIX) and fbin.endswith(SUFFIX)):
            continue
        rows.append({
            "stage": labels.get("stage"), "head": labels.get("head", ""),
            "position": labels.get("position", "all"), "bit": labels.get("bit"),
            "seq": rec.get("seq"), "classification": "SDC",
            "y_rel_l2": g["rel"],
            "golden": gbin[:-len(SUFFIX)], "faulty": fbin[:-len(SUFFIX)],
        })

    out = os.path.join(args.exp, "propagation_manifest.csv")
    cols = ["stage", "head", "position", "bit", "seq", "classification",
            "y_rel_l2", "golden", "faulty"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("wrote %s (%d representative runs)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
