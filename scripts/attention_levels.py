#!/usr/bin/env python3
"""Fast L0-L5 reclassification of every run (final Y only).

Uses tolerance_sweep.corrupted_at_levels, which has a bitwise-equality fast
path and early-exits once all levels are exceeded, so it stays fast even for
thousands of runs.

Writes <exp>/attention_levels.csv and <exp>/attention_by_level.csv.

Usage: attention_levels.py --exp results/046_attn_mha_bit_sweep
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import tolerance_sweep as TS  # noqa: E402

LEVELS = [("L0", 0.0, 0.0),
          ("L1", 1e-7, 1e-6),
          ("L2", 1e-6, 1e-5),
          ("L3", 1e-5, 1e-4),
          ("L4", 1e-4, 1e-3),
          ("L5", 1e-3, 1e-2)]
ORDER = {n: i for i, (n, _, _) in enumerate(LEVELS)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args(argv)

    obs_path = os.path.join(args.exp, "observations.jsonl")
    if not os.path.exists(obs_path):
        raise SystemExit("missing %s" % obs_path)

    rows = []
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = rec.get("labels") or {}
            stage = labels.get("stage")
            head = labels.get("head")
            position = labels.get("position", "all")
            bit = labels.get("bit")
            seq = rec.get("seq")
            cls = rec.get("classification")
            obs = rec.get("observation") or {}
            rel = obs.get("relative_l2_error", "")
            mx = obs.get("max_abs_error", "")
            g = rec.get("golden_bin")
            fo = rec.get("raw_out_bin")

            if (cls in ("MASKED", "SDC") and g and fo
                    and os.path.exists(g) and os.path.exists(fo)):
                try:
                    counts = TS.corrupted_at_levels(
                        TS.read_floats(g), TS.read_floats(fo), LEVELS)
                except Exception:
                    counts = [cls == "SDC"] * len(LEVELS)
                for (name, _, _), corrupt in zip(LEVELS, counts):
                    rows.append({
                        "seq": seq, "stage": stage, "head": head,
                        "position": position, "bit": bit, "level": name,
                        "classification": "SDC" if corrupt else "MASKED",
                        "y_rel_l2": rel, "y_max_abs": mx})
            else:
                for name, _, _ in LEVELS:
                    rows.append({
                        "seq": seq, "stage": stage, "head": head,
                        "position": position, "bit": bit, "level": name,
                        "classification": cls, "y_rel_l2": "", "y_max_abs": ""})

    cols = ["seq", "stage", "head", "position", "bit", "level",
            "classification", "y_rel_l2", "y_max_abs"]
    with open(os.path.join(args.exp, "attention_levels.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg = {}
    for r in rows:
        key = (r["stage"], str(r["head"]), str(r["position"]), r["level"])
        a = agg.setdefault(key, {"n": 0, "MASKED": 0, "SDC": 0, "CRASH": 0,
                                 "NOT_INJECTED": 0, "NOT_TARGETED": 0})
        a["n"] += 1
        c = r["classification"]
        if c in a:
            a[c] += 1
    with open(os.path.join(args.exp, "attention_by_level.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "head", "position", "level", "n", "MASKED", "SDC",
                    "CRASH", "NOT_INJECTED", "NOT_TARGETED", "sdc_rate"])
        for key in sorted(agg, key=lambda k: (k[0], str(k[1]), str(k[2]),
                                               ORDER.get(k[3], 99))):
            a = agg[key]
            denom = a["MASKED"] + a["SDC"] + a["CRASH"]
            w.writerow(list(key) + [a["n"], a["MASKED"], a["SDC"], a["CRASH"],
                                    a["NOT_INJECTED"], a["NOT_TARGETED"],
                                    "%.4f" % (a["SDC"] / denom) if denom else ""])
    print("wrote %s/attention_levels.csv (%d rows)" % (args.exp, len(rows)))
    print("wrote %s/attention_by_level.csv" % args.exp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
