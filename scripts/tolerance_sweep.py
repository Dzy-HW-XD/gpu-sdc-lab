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


def _same_value(a, b):
    if a == b:
        return True
    if a != a and b != b:  # both NaN
        return True
    return False


def corrupted_at_levels(golden, fault, levels):
    """Return a list of booleans: True if the run is corrupted (SDC) at that
    level, i.e. some element differs from golden beyond abs_tol + rel_tol*|g|.

    A run whose output is bitwise identical to golden is MASKED at every level
    (this matches the run-time oracle's sha256 fast path, and is essential for
    overflow-prone inputs where golden is legitimately Inf/NaN: an unchanged
    Inf output is NOT corruption). Non-finite elements that are *equal* to the
    golden value (same Inf, or both NaN) are not corruption; a non-finite value
    that differs corrupts at every level (it can never be within tolerance).

    We only need the MASKED-vs-SDC decision, so the scan stops as soon as every
    level has already been seen to exceed (early exit).
    """
    n = len(levels)
    if len(fault) != len(golden):
        return [True] * n
    if golden.tobytes() == fault.tobytes():
        return [False] * n
    done = [False] * n
    for i in range(len(golden)):
        gv = golden[i]
        fv = fault[i]
        if (fv != fv or fv == math.inf or fv == -math.inf
                or gv != gv or gv == math.inf or gv == -math.inf):
            if _same_value(gv, fv):
                continue
            return [True] * n
        d = fv - gv
        ad = d if d >= 0.0 else -d
        ag = gv if gv >= 0.0 else -gv
        for j in range(n):
            if not done[j]:
                _name, a, r = levels[j]
                if ad > (a + r * ag):
                    done[j] = True
        if all(done):
            return done
    return done


def load_obs_dirs(lab, eids):
    rbase = lab.results_dir
    all_dirs = sorted(d for d in os.listdir(rbase)
                      if os.path.isdir(os.path.join(rbase, d)))
    for eid in eids:
        # Prefer shard dirs; only fall back to the merged base dir if no shards
        # exist (the base dir is itself the concatenation of the shards).
        shards = [d for d in all_dirs if d.startswith(eid + "__")]
        if shards:
            groups = shards
        elif eid in all_dirs:
            groups = [eid]
        else:
            groups = []
        for d in groups:
            p = os.path.join(rbase, d, "observations.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield eid, json.loads(line)


def cell_keys(rec):
    """Yield (axis, cell) pairs for one record, mirroring campaign._cell_keys."""
    labels = rec.get("labels", {}) or {}
    pr = rec.get("params", {}) or {}
    fault = rec.get("fault", {}) or {}
    inp = labels.get("input", "all")
    if fault.get("bit_index") is not None:
        yield "bit", "%s|bit%s" % (inp, fault["bit_index"])
    if fault.get("opcode"):
        yield "instruction", "%s|%s" % (inp, fault["opcode"])
    if all(k in pr for k in ("M", "N", "K")):
        yield "size", "%s|%sx%sx%s" % (inp, pr["M"], pr["N"], pr["K"])
    if labels.get("position"):
        yield "position", "%s|%s" % (inp, labels["position"])
    if fault.get("register_class"):
        yield "register", "%s|%s" % (inp, fault["register_class"])
    yield "input", inp


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
    overall = {name: [0, 0, 0] for name, _, _ in levels}   # masked, sdc, crash
    by_input = {}
    by_exp = {}
    by_cell = {}

    def bump(store, key, name, kind):
        d = store.setdefault(key, {n: [0, 0, 0] for n, _, _ in levels})
        d[name][{"masked": 0, "sdc": 1, "crash": 2}[kind]] += 1

    for eid, rec in load_obs_dirs(lab, args.experiments):
        cls = rec.get("classification", "NOT_INJECTED")
        if cls in ("NOT_INJECTED", "NOT_TARGETED"):
            continue
        labels = rec.get("labels", {}) or {}
        inp = labels.get("input", "all")
        keys = list(cell_keys(rec))

        if cls == "CRASH":
            for name, _, _ in levels:
                overall[name][2] += 1
                bump(by_input, (eid, inp), name, "crash")
                bump(by_exp, (eid,), name, "crash")
                for axis, cell in keys:
                    bump(by_cell, (eid, axis, cell), name, "crash")
            continue

        gpath = rec.get("golden_bin")
        fpath = rec.get("raw_out_bin")
        if not gpath or not fpath or not os.path.exists(gpath) or not os.path.exists(fpath):
            continue
        golden = golden_cache.get(gpath)
        if golden is None:
            golden = read_floats(gpath)
            golden_cache[gpath] = golden
        fault = read_floats(fpath)
        counts = corrupted_at_levels(golden, fault, levels)

        for (name, _, _), corrupted in zip(levels, counts):
            kind = "sdc" if corrupted else "masked"
            overall[name][0 if kind == "masked" else 1] += 1
            bump(by_input, (eid, inp), name, kind)
            bump(by_exp, (eid,), name, kind)
            for axis, cell in keys:
                bump(by_cell, (eid, axis, cell), name, kind)

    # overall table
    with open(os.path.join(outdir, "tolerance_overall.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "abs_tol", "rel_tol", "total", "masked", "sdc",
                    "crash", "sdc_rate"])
        for (name, a, r) in levels:
            m, s, c = overall[name]
            n = m + s + c
            w.writerow([name, a, r, n, m, s, c, "%.6f" % ((s / n) if n else 0.0)])

    # per-experiment table
    with open(os.path.join(outdir, "tolerance_by_experiment.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment_id", "level", "masked", "sdc", "crash", "total",
                    "sdc_rate"])
        for eid in sorted(by_exp):
            for (name, _, _) in levels:
                m, s, c = by_exp[eid][name]
                n = m + s + c
                w.writerow([eid[0], name, m, s, c, n,
                            "%.6f" % ((s / n) if n else 0.0)])

    def write_table(path, agg, keynames, keyorder):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keynames + ["level", "masked", "sdc", "crash", "total",
                                   "sdc_rate"])
            for key in sorted(agg, key=lambda k: keyorder(k)):
                for (name, _, _) in levels:
                    m, s, c = agg[key][name]
                    n = m + s + c
                    w.writerow(list(key) + [name, m, s, c, n,
                                            "%.6f" % ((s / n) if n else 0.0)])

    write_table(os.path.join(outdir, "tolerance_by_input.csv"), by_input,
                ["experiment_id", "input"], lambda k: (k[0], str(k[1])))
    write_table(os.path.join(outdir, "tolerance_by_cell.csv"), by_cell,
                ["experiment_id", "axis", "cell"],
                lambda k: (k[0], k[1], str(k[2])))

    print("wrote tolerance_overall.csv / tolerance_by_experiment.csv / "
          "tolerance_by_input.csv / tolerance_by_cell.csv -> %s" % outdir)
    for (name, a, r) in levels:
        m, s, c = overall[name]
        n = m + s + c
        print("  %-10s abs=%.0e rel=%.0e  SDC %d/%d = %.4f"
              % (name, a, r, s, n, (s / n) if n else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
