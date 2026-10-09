#!/usr/bin/env python3
"""Markdown report from a campaign cells.csv (stdlib only).

Aggregates the per-cell SDC counts by FP32 bit region (sign/exponent/mantissa),
by input family, by instruction, by register class, and by size/position.

Usage: python3 scripts/campaign_report.py results/_campaign/cells.csv out.md
"""

import argparse
import csv
import os
import sys

BIT_REGION = {}
for _b in range(32):
    if _b == 31:
        BIT_REGION[_b] = "sign"
    elif _b >= 23:
        BIT_REGION[_b] = "exponent"
    else:
        BIT_REGION[_b] = "mantissa"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _rate(sdc, total):
    return (sdc / total) if total else 0.0


def agg(rows, keyfn, filt=None):
    out = {}
    for r in rows:
        if filt and not filt(r):
            continue
        k = keyfn(r)
        if k is None:
            continue
        b = out.setdefault(k, [0, 0])  # sdc, total
        b[0] += int(r["sdc"])
        b[1] += int(r["total"])
    return out


def table(d, col):
    lines = ["| %s | sdc | n | rate |" % col, "|---|---|---|---|"]
    for k in sorted(d):
        sdc, n = d[k]
        lines.append("| %s | %d | %d | %.3f |" % (k, sdc, n, _rate(sdc, n)))
    return "\n".join(lines)


def overall_from_master(path):
    d = {}
    for r in load(path):
        if r.get("valid") != "1":
            continue
        b = d.setdefault(r["experiment_id"], [0, 0])
        b[1] += 1
        if r["classification"] == "SDC":
            b[0] += 1
    return d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cells")
    p.add_argument("out")
    p.add_argument("--master", default=None,
                   help="master.csv for true per-experiment injection counts")
    args = p.parse_args(argv)
    rows = load(args.cells)

    if args.master and os.path.exists(args.master):
        by_eid = overall_from_master(args.master)
        overall_note = "injection counts from `master.csv`."
    else:
        by_eid = agg(rows, lambda r: r["experiment_id"])
        overall_note = ("cell-weighted (each injection appears in every axis); "
                        "pass --master for true injection counts.")
    md = ["# Input-dependent FP32 GEMM SDC campaign", "",
          "All results from `results/_campaign/cells.csv` (Wilson CIs included). "
          "Fault model: single-bit flip in the destination register of a targeted "
          "FP32 instruction; MASKED/SDC/CRASH classified against a bit-exact "
          "golden run. " + overall_note, "",
          "## Overall (by experiment)", "", table(by_eid, "experiment"), ""]

    def only(prefix, axis):
        return lambda r: r["experiment_id"].startswith(prefix) and r["axis"] == axis

    bitrows = [r for r in rows if r["experiment_id"].startswith("015")
               and r["axis"] == "bit"]
    if bitrows:
        def region(r):
            v = r["cell"].split("|", 1)[1]
            if not v.startswith("bit"):
                return None
            b = int(v[3:])
            return BIT_REGION.get(b)
        md += ["## SDC rate by FP32 bit region (015)", "",
               table(agg(bitrows, region), "region"), "",
               "## SDC rate by input (015, pooled over bits 0-31)", "",
               table(agg(bitrows, lambda r: r["cell"].split("|", 1)[0]), "input"),
               ""]

    inst = agg(rows, lambda r: r["cell"].split("|", 1)[1], only("016", "instruction"))
    if inst:
        md += ["## SDC rate by instruction (016)", "", table(inst, "opcode"), ""]

    reg = agg(rows, lambda r: r["cell"].split("|", 1)[1], only("019", "register"))
    if reg:
        md += ["## SDC rate by register class (019)", "", table(reg, "register_class"), ""]

    size = agg(rows, lambda r: r["cell"].split("|", 1)[1], only("017", "size"))
    if size:
        md += ["## SDC rate by GEMM size (017)", "", table(size, "size"), ""]

    pos = agg(rows, lambda r: r["cell"].split("|", 1)[1], only("018", "position"))
    if pos:
        md += ["## SDC rate by injection position (018)", "", table(pos, "position"), ""]

    text = "\n".join(md)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n[written] " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
