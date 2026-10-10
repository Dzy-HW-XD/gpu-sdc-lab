#!/usr/bin/env python3
"""Offline tolerance sensitivity (decoupled from injection).

Re-reads the preserved raw faulty outputs (and golden outputs) of an already
completed run and re-classifies every sample under a set of (abs_tol, rel_tol)
levels. No GPU / no re-injection is needed, so tolerances can be changed freely.

Default levels ("synchronous scaling" from the base point (1e-7, 1e-6)):
  L1 (1e-7, 1e-6)  L2 (1e-6, 1e-5)  L3 (1e-5, 1e-4)  L4 (1e-4, 1e-3)  L5 (1e-3, 1e-2)

An element is "corrupted" at a level iff |fault-golden| > abs_tol + rel_tol*|golden|.
Non-finite elements (fault or golden NaN/Inf) are corrupted at every level.
Classification: CRASH stays CRASH; corrupted==0 -> MASKED; else SDC.

Usage:
  python3 scripts/tolerance_sweep.py 015_input_bit_gemm 016_input_instruction_gemm \
      --out results/_tolerance
"""

import argparse
import array
import csv
import json
import math
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)

from core.config import load_lab_config  # noqa: E402
from core import util                    # noqa: E402

DEFAULT_LEVELS = [("L1_1e-7", 1e-7, 1e-6), ("L2_1e-6", 1e-6, 1e-5),
                  ("L3_1e-5", 1e-5, 1e-4), ("L4_1e-4", 1e-4, 1e-3),
                  ("L5_1e-3", 1e-3, 1e-2)]


def read_floats(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def corrupted_at_levels(golden, fault, levels):
    counts = [0] * len(levels)
    n = min(len(golden), len(fault))
    for i in range(n):
        gv = golden[i]
        fv = fault[i]
        if not (fv == fv) or fv == math.inf or fv == -math.inf:
            for j in range(len(levels)):
                counts[j] += 1
            continue
        if not (gv == gv) or gv == math.inf or gv == -math.inf:
            for j in range(len(levels)):
                counts[j] += 1
            continue
        d = fv - gv
        ad = d if d >= 0.0 else -d
        ag = gv if gv >= 0.0 else -gv
        for j, (_name, a, r) in enumerate(levels):
            if ad > (a + r * ag):
                counts[j] += 1
    if len(fault) != len(golden):
        extra = abs(len(fault) - len(golden))
        for j in range(len(levels)):
            counts[j] += extra
    return counts


def load_obs_dirs(lab, eids):
    rbase = lab.results_dir
    all_dirs = sorted(d for d in os.listdir(rbase)
                      if os.path.isdir(os.path.join(rbase, d)))
    for eid in eids:
        groups = [d for d in all_dirs if d == eid or d.startswith(eid + "__")]
        for d in groups:
            p = os.path.join(rbase, d, "observations.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield eid, json.loads(line)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+")
    p.add_argument("--lab-config", default=os.path.join(LAB_ROOT, "config.yaml"))
    p.add_argument("--out", default=None, help="output dir (default results/_tolerance)")
    p.add_argument("--levels", default=None,
                   help="comma list of abs:rel, e.g. 1e-7:1e-6,1e-5:1e-4")
    args = p.parse_args(argv)

    if args.levels:
        levels = []
        for k, pair in enumerate(args.levels.split(",")):
            a, _, r = pair.partition(":")
            levels.append(("L%d_%s" % (k + 1, pair), float(a), float(r)))
    else:
        levels = DEFAULT_LEVELS

    lab = load_lab_config(args.lab_config)
    outdir = util.ensure_dir(args.out or os.path.join(lab.results_dir, "_tolerance"))

    golden_cache = {}
    overall = {name: {"masked": 0, "sdc": 0, "crash": 0} for name, _, _ in levels}
    by_input = {}
    by_cell = {}

    for eid, rec in load_obs_dirs(lab, args.experiments):
        cls = rec.get("classification", "NOT_INJECTED")
        if cls in ("NOT_INJECTED", "NOT_TARGETED"):
            continue
        labels = rec.get("labels", {})
        inp = labels.get("input", "all")
        bit = rec.get("fault", {}).get("bit_index")
        cell = "%s|bit%s" % (inp, bit) if bit is not None else inp

        if cls == "CRASH":
            for name, _, _ in levels:
                overall[name]["crash"] += 1
                by_input.setdefault((eid, inp), {n: [0, 0, 0] for n, _, _ in levels})
                by_cell.setdefault((eid, cell), {n: [0, 0, 0] for n, _, _ in levels})
                by_input[(eid, inp)][name][2] += 1
                by_cell[(eid, cell)][name][2] += 1
            continue

        gpath = rec.get("golden_bin")
        fpath = rec.get("raw_out_bin")
        if not gpath or not fpath or not os.path.exists(gpath) or not os.path.exists(fpath):
            continue
        if gpath not in golden_cache:
            golden_cache[gpath] = read_floats(gpath)
        golden = golden_cache[gpath]
        fault = read_floats(fpath)
        counts = corrupted_at_levels(golden, fault, levels)

        for (name, _, _), c in zip(levels, counts):
            b = overall[name]
            mk = "MASKED" if c == 0 else "SDC"
            if mk == "MASKED":
                b["masked"] += 1
            else:
                b["sdc"] += 1
            bi = by_input.setdefault((eid, inp), {n: [0, 0, 0] for n, _, _ in levels})[name]
            bc = by_cell.setdefault((eid, cell), {n: [0, 0, 0] for n, _, _ in levels})[name]
            if mk == "MASKED":
                bi[0] += 1
                bc[0] += 1
            else:
                bi[1] += 1
                bc[1] += 1

    # overall table
    with open(os.path.join(outdir, "tolerance_overall.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "abs_tol", "rel_tol", "total", "masked", "sdc",
                    "crash", "sdc_rate"])
        for (name, a, r) in levels:
            b = overall[name]
            n = b["masked"] + b["sdc"] + b["crash"]
            rate = (b["sdc"] / n) if n else 0.0
            w.writerow([name, a, r, n, b["masked"], b["sdc"], b["crash"],
                        "%.6f" % rate])

    def write_table(path, agg, keyname):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["experiment_id", keyname, "level", "masked", "sdc",
                        "crash", "total", "sdc_rate"])
            for (eid, key) in sorted(agg, key=lambda k: (k[0], str(k[1]))):
                for (name, _, _) in levels:
                    m, s, c = agg[(eid, key)][name]
                    n = m + s + c
                    w.writerow([eid, key, name, m, s, c, n,
                                "%.6f" % ((s / n) if n else 0.0)])

    write_table(os.path.join(outdir, "tolerance_by_input.csv"), by_input, "input")
    write_table(os.path.join(outdir, "tolerance_by_cell.csv"), by_cell, "cell")

    print("wrote tolerance_overall.csv / tolerance_by_input.csv / tolerance_by_cell.csv -> %s"
          % outdir)
    for (name, a, r) in levels:
        b = overall[name]
        n = b["masked"] + b["sdc"] + b["crash"]
        print("  %-10s abs=%.0e rel=%.0e  SDC %d/%d = %.4f"
              % (name, a, r, b["sdc"], n, (b["sdc"] / n) if n else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
