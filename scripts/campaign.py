#!/usr/bin/env python3
"""Batch campaign driver + collector for input-dependent SDC studies.

Runs a list of experiments in-process (same Runner as `sdc-lab run`), then
merges every result's observations.jsonl into a single master CSV plus a
per-cell summary with Wilson confidence intervals, ready for plotting.

Usage:
  python3 scripts/campaign.py run 015_input_bit_gemm 016_input_instruction_gemm ...
  python3 scripts/campaign.py collect [--master results/_campaign/master.csv]
  python3 scripts/campaign.py all <experiments...>

This script is intentionally read/write of results only; it never touches the
injector. Each experiment already generates its own golden + profile (cached).
"""

import argparse
import csv
import json
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from core.config import load_lab_config       # noqa: E402
from core.runner import Runner                # noqa: E402
from core import yamlmini                     # noqa: E402
import stats                                  # noqa: E402

MASTER_COLUMNS = [
    "experiment_id", "input", "bit", "opcode", "target", "register_class",
    "size", "position", "classification", "valid", "max_abs_error",
    "relative_l2_error", "corrupted_elements", "finite_pairs", "nan_count",
    "inf_count", "golden_inf_count", "golden_nan_count", "injected",
    "target_matched",
]


def load_experiment(lab, eid):
    path = os.path.join(lab.root, "experiments", eid, "experiment.yaml")
    data = yamlmini.load(path) or {}
    data.setdefault("id", eid)
    return data


def run(lab, eids, shard=0, nshards=1, tag=None):
    runner = Runner(lab)
    for eid in eids:
        expt = load_experiment(lab, eid)
        rid = eid if not tag else "%s__%s" % (eid, tag)
        expt["id"] = rid
        expt["shard"] = shard
        expt["nshards"] = nshards
        if expt.get("kind") == "baseline":
            print("== baseline %s ==" % rid)
            runner.run_baseline(expt)
            continue
        print("== run %s (kind=%s, axis=%s, shard=%d/%d) ==" %
              (rid, expt.get("kind"), expt.get("axis"), shard, nshards))
        result = runner.run_fault(expt)
        c = result["counts"]
        print("[%s] total=%d masked=%d sdc=%d crash=%d rate=%.4f" %
              (eid, c["MASKED"] + c["SDC"] + c["CRASH"], c["MASKED"],
               c["SDC"], c["CRASH"], c["sdc_rate"]))
    return 0


def _size(rec):
    p = rec.get("params", {})
    if all(k in p for k in ("M", "N", "K")):
        return "%sx%sx%s" % (p["M"], p["N"], p["K"])
    return ""


def _row(eid, rec):
    labels = rec.get("labels", {})
    fault = rec.get("fault", {})
    obs = rec.get("observation", {})
    cls = rec.get("classification", "NOT_INJECTED")
    return {
        "experiment_id": eid,
        "input": labels.get("input", ""),
        "bit": fault.get("bit_index", ""),
        "opcode": fault.get("opcode", ""),
        "target": fault.get("target", ""),
        "register_class": fault.get("register_class", ""),
        "size": _size(rec),
        "position": labels.get("position", labels.get("position_decile", "")),
        "classification": cls,
        "valid": "1" if cls in ("MASKED", "SDC", "CRASH") else "0",
        "max_abs_error": obs.get("max_abs_error", ""),
        "relative_l2_error": obs.get("relative_l2_error", ""),
        "corrupted_elements": obs.get("corrupted_elements", ""),
        "finite_pairs": obs.get("finite_pairs", ""),
        "nan_count": obs.get("nan_count", ""),
        "inf_count": obs.get("inf_count", ""),
        "golden_inf_count": obs.get("golden_inf_count", ""),
        "golden_nan_count": obs.get("golden_nan_count", ""),
        "injected": fault.get("injected", ""),
        "target_matched": fault.get("target_matched", ""),
    }


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def collect(lab, eids, master_path):
    campaign_dir = os.path.dirname(master_path)
    os.makedirs(campaign_dir, exist_ok=True)
    rows = []
    rbase = lab.results_dir
    all_dirs = sorted(d for d in os.listdir(rbase)
                      if os.path.isdir(os.path.join(rbase, d)))
    for eid in eids:
        # Prefer shard dirs; fall back to the merged base dir only if no shards
        # (the base dir is the concatenation of the shards -> would duplicate).
        shards = [d for d in all_dirs if d.startswith(eid + "__")]
        groups = shards if shards else ([eid] if eid in all_dirs else [])
        if not groups:
            print("  [skip] no observations for %s" % eid)
            continue
        for d in groups:
            p = os.path.join(rbase, d, "observations.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(_row(eid, json.loads(line)))

    with open(master_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (master_path, len(rows)))

    _summarize_cells(campaign_dir, rows)
    return rows


def _present(v):
    return v not in ("", None)


def _cell_keys(r):
    """Yield (axis, cell) pairs applicable to one record. cell = input|value."""
    inp = r["input"] or "all"
    if _present(r["bit"]):
        yield "bit", "%s|bit%s" % (inp, r["bit"])
    if _present(r["opcode"]):
        yield "instruction", "%s|%s" % (inp, r["opcode"])
    if _present(r["size"]):
        yield "size", "%s|%s" % (inp, r["size"])
    if _present(r["position"]):
        yield "position", "%s|%s" % (inp, r["position"])
    if _present(r["register_class"]):
        yield "register", "%s|%s" % (inp, r["register_class"])
    yield "input", inp


def _summarize_cells(campaign_dir, rows):
    agg = {}
    for r in rows:
        if r["valid"] != "1":
            continue
        for axis, cell in _cell_keys(r):
            key = (r["experiment_id"], axis, cell)
            b = agg.setdefault(key, {"masked": 0, "sdc": 0, "crash": 0,
                                     "abs_err_sum": 0.0, "abs_err_max": 0.0,
                                     "finite": 0, "nonfinite": 0})
            cls = r["classification"]
            if cls == "MASKED":
                b["masked"] += 1
            elif cls == "SDC":
                b["sdc"] += 1
                nonfinite = (_int(r.get("nan_count")) + _int(r.get("inf_count"))
                             + _int(r.get("golden_inf_count"))
                             + _int(r.get("golden_nan_count"))) > 0
                if nonfinite:
                    b["nonfinite"] += 1
                if _int(r.get("finite_pairs")) > 0:
                    try:
                        e = float(r["max_abs_error"])
                        b["abs_err_sum"] += e
                        b["abs_err_max"] = max(b["abs_err_max"], e)
                        b["finite"] += 1
                    except (TypeError, ValueError):
                        pass
            elif cls == "CRASH":
                b["crash"] += 1

    out = os.path.join(campaign_dir, "cells.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment_id", "axis", "cell", "total", "masked", "sdc",
                    "crash", "sdc_rate", "ci_low", "ci_high", "mean_abs_err",
                    "max_abs_err", "finite", "nonfinite"])
        for (eid, axis, cell) in sorted(agg, key=lambda k: (k[0], k[1], str(k[2]))):
            b = agg[(eid, axis, cell)]
            n = b["masked"] + b["sdc"] + b["crash"]
            rate = (b["sdc"] / n) if n else 0.0
            lo, hi = stats.wilson_ci(b["sdc"], n)
            mean_err = (b["abs_err_sum"] / b["finite"]) if b["finite"] else 0.0
            w.writerow([eid, axis, cell, n, b["masked"], b["sdc"], b["crash"],
                        "%.6f" % rate, "%.6f" % lo, "%.6f" % hi,
                        "%.6g" % mean_err, "%.6g" % b["abs_err_max"],
                        b["finite"], b["nonfinite"]])
    print("wrote %s (%d cells)" % (out, len(agg)))


def main(argv=None):
    p = argparse.ArgumentParser(description="gpu-sdc-lab batch campaign")
    p.add_argument("--lab-config",
                   default=os.path.join(LAB_ROOT, "config.yaml"))
    p.add_argument("--master",
                   default=None, help="master CSV path")
    sub = p.add_subparsers(dest="cmd")
    for cmd in ("run", "collect", "all"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--master", default=None, help="master CSV path")
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--nshards", type=int, default=1)
        sp.add_argument("--tag", default=None,
                        help="suffix for the result dir (e.g. s0of8)")
        sp.add_argument("experiments", nargs="*")
    args = p.parse_args(argv)

    lab = load_lab_config(args.lab_config)
    master = args.master or os.path.join(lab.results_dir, "_campaign", "master.csv")
    eids = args.experiments

    if args.cmd in ("run", "all") and eids:
        run(lab, eids, shard=args.shard, nshards=args.nshards, tag=args.tag)
    if args.cmd in ("collect", "all"):
        if not eids:
            eids = sorted(d for d in os.listdir(
                os.path.join(lab.root, "experiments"))
                if os.path.isdir(os.path.join(lab.root, "experiments", d)))
        collect(lab, eids, master)
    if not args.cmd:
        p.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
