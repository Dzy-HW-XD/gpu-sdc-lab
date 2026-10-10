#!/usr/bin/env python3
"""Render one input x <second-variable> SDC-rate heatmap per tolerance level
(L1..L5) for each experiment.

Consumes results/_tolerance/tolerance_by_cell.csv (produced by
scripts/tolerance_sweep.py) and, for every experiment, draws one heatmap per
tolerance level so the effect of tolerance can be compared across the grid:

  015 input x bit | 016 input x instruction | 017 input x size
  018 input x position | 019 input x register

Usage:
  python3 scripts/tolerance_heatmaps.py \
      --cells results/_tolerance/tolerance_by_cell.csv \
      --outdir results/_tolerance/figs
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import heatmap  # noqa: E402

DEFAULT_AXIS = {
    "015_input_bit_gemm": "bit",
    "016_input_instruction_gemm": "instruction",
    "017_input_size_gemm": "size",
    "018_input_position_gemm": "position",
    "019_register_class_gemm": "register",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cells",
                   default=os.path.join("results", "_tolerance", "tolerance_by_cell.csv"))
    p.add_argument("--experiments", nargs="*", default=list(DEFAULT_AXIS))
    p.add_argument("--axis", default=None,
                   help="force a second-variable axis for all experiments")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    rows = list(csv.DictReader(open(args.cells, encoding="utf-8")))
    outdir = args.outdir or os.path.join(os.path.dirname(args.cells), "figs")
    os.makedirs(outdir, exist_ok=True)

    levels = sorted({r["level"] for r in rows})
    written = 0
    for eid in args.experiments:
        axis = args.axis or DEFAULT_AXIS.get(eid)
        if not axis:
            continue
        erows = [r for r in rows
                 if r["experiment_id"] == eid and r["axis"] == axis]
        if not erows:
            print("skip %s (no axis=%s cells)" % (eid, axis))
            continue
        for lvl in levels:
            data = []
            for r in erows:
                if r["level"] != lvl:
                    continue
                if "|" not in r["cell"]:
                    continue
                inp, val = r["cell"].split("|", 1)
                data.append({"_input": inp, "_value": val,
                             "sdc_rate": float(r["sdc_rate"])})
            if not data:
                continue
            title = "%s  [%s]" % (eid, lvl)
            tag = "%s_%s_%s_sdc_rate" % (eid, axis, lvl)
            out = os.path.join(outdir, "heatmap_%s.svg" % tag)
            piv = os.path.join(outdir, "pivot_%s.csv" % tag)
            heatmap.render(title, data, "sdc_rate", axis, out, piv,
                           log_color=False, annotate=False)
            written += 1
            print("wrote %s" % out)
    print("total %d heatmaps" % written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
