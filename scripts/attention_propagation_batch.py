#!/usr/bin/env python3
"""Batch-generate propagation plots from a propagation_manifest.csv.

The manifest holds local golden/faulty prefixes (see
scripts/attention_select_runs.py; download + rewrite paths before running this
on a machine with matplotlib).

For each representative run it calls attention_propagation_plots.py and writes
<outroot>/propagation_index.csv listing the generated figures.

Usage:
  attention_propagation_batch.py --manifest manifest_local.csv --outroot plots
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PLOTS = os.path.join(HERE, "attention_propagation_plots.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outroot", required=True)
    args = ap.parse_args()
    os.makedirs(args.outroot, exist_ok=True)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    idx = []
    for r in rows:
        tag = "prop_%s_h%s_%s_bit%s_seq%s" % (
            r["stage"], r.get("head") or "x", r["position"], r["bit"], r["seq"])
        outdir = os.path.join(args.outroot, tag)
        title = "%s h%s %s bit%s seq%s  relL2=%s" % (
            r["stage"], r.get("head") or "-", r["position"], r["bit"],
            r["seq"], r.get("y_rel_l2", ""))
        cmd = [sys.executable, PLOTS, "--golden", r["golden"],
               "--faulty", r["faulty"], "--outdir", outdir, "--title", title]
        print("== %s ==" % tag)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("  FAILED (rc=%d)" % res.returncode)
            continue
        idx.append({
            "tag": tag, "stage": r["stage"], "head": r.get("head", ""),
            "position": r["position"], "bit": r["bit"], "seq": r["seq"],
            "y_rel_l2": r.get("y_rel_l2", ""),
            "stage_grid": os.path.join(outdir, "stage_grid.png"),
            "head_token": os.path.join(outdir, "head_token.png"),
            "support_curve": os.path.join(outdir, "support_curve.png"),
        })

    out = os.path.join(args.outroot, "propagation_index.csv")
    cols = ["tag", "stage", "head", "position", "bit", "seq", "y_rel_l2",
            "stage_grid", "head_token", "support_curve"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in idx:
            w.writerow(r)
    print("wrote %s (%d runs)" % (out, len(idx)))


if __name__ == "__main__":
    main()
