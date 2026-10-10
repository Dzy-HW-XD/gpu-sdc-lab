#!/usr/bin/env python3
"""Render one input x bit SDC-rate heatmap per tolerance level (L1..L5).

Consumes results/_tolerance/tolerance_by_cell.csv (produced by
scripts/tolerance_sweep.py) and draws, for the chosen experiment, one heatmap
per level so the effect of tolerance on the SDC map can be compared directly.

Usage:
  python3 scripts/tolerance_heatmaps.py \
      --cells results/_tolerance/tolerance_by_cell.csv \
      --experiment 015_input_bit_gemm \
      --outdir results/_tolerance/figs
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import heatmap  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cells",
                   default=os.path.join("results", "_tolerance", "tolerance_by_cell.csv"))
    p.add_argument("--experiment", default="015_input_bit_gemm")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    rows = list(csv.DictReader(open(args.cells, encoding="utf-8")))
    rows = [r for r in rows if r["experiment_id"] == args.experiment and "|" in r["cell"]]
    if not rows:
        print("no cells for %s" % args.experiment)
        return 1

    levels = []
    for r in rows:
        if r["level"] not in levels:
            levels.append(r["level"])
    levels.sort()

    outdir = args.outdir or os.path.join(os.path.dirname(args.cells), "figs")
    os.makedirs(outdir, exist_ok=True)

    for lvl in levels:
        erows = []
        for r in rows:
            if r["level"] != lvl:
                continue
            inp, val = r["cell"].split("|", 1)
            erows.append({"_input": inp, "_value": val,
                          "sdc_rate": float(r["sdc_rate"])})
        tag = "%s_%s_sdc_rate" % (args.experiment, lvl)
        out = os.path.join(outdir, "heatmap_%s.svg" % tag)
        piv = os.path.join(outdir, "pivot_%s.csv" % tag)
        heatmap.render(args.experiment + "  [" + lvl + "]", erows, "sdc_rate",
                       "bit", out, piv, log_color=False, annotate=False)
        print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
