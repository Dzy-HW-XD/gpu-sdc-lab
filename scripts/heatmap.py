#!/usr/bin/env python3
"""Render SDC heatmaps from campaign/experiment cells as SVG (stdlib only).

Reads a cells.csv (produced by scripts/campaign.py) or any CSV with columns
`experiment_id,cell,...` and draws an input x <axis> grid, where the cell label
is `input|value`. Works for the input x bit / instruction / size / position
tables. Also writes a pivot CSV next to each SVG.

Usage:
  python3 scripts/heatmap.py results/_campaign/cells.csv --metric sdc_rate
  python3 scripts/heatmap.py results/_campaign/cells.csv --metric mean_abs_err \
      --axis bit --outdir results/_campaign/figs
"""

import argparse
import csv
import os
import sys

INPUT_ORDER = ["uniform", "normal", "lognormal", "sparse", "cancellation",
               "near_overflow", "correlated", "adversarial",
               "ones", "near_zero", "near_one", "extreme", "small", "large"]


def _input_rank(name):
    try:
        return INPUT_ORDER.index(name)
    except ValueError:
        return len(INPUT_ORDER)


def _value_sort(v):
    s = str(v)
    if s.startswith("bit") and s[3:].isdigit():
        return (0, int(s[3:]))
    return (1, s)


def load_cells(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cell = r.get("cell", "")
            if "|" not in cell:
                continue
            inp, val = cell.split("|", 1)
            r["_input"] = inp
            r["_value"] = val
            r["_axis"] = r.get("axis", "")
            rows.append(r)
    return rows


def color(v, vmin, vmax):
    if vmax <= vmin:
        t = 0.0
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # light grey (low) -> dark red (high)
    lo = (242, 242, 242)
    hi = (178, 24, 43)
    rgb = tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % rgb, (t > 0.6)


def render(eid, rows, metric, axis, outpath, pivot_path):
    inputs = sorted({r["_input"] for r in rows}, key=_input_rank)
    values = sorted({r["_value"] for r in rows}, key=_value_sort)
    if axis:
        values = [v for v in values if str(v).startswith(axis) or axis == "register"]
    if not inputs or not values:
        return None

    data = {}
    for r in rows:
        if r["_value"] not in values:
            continue
        try:
            data[(r["_input"], r["_value"])] = float(r.get(metric) or 0.0)
        except ValueError:
            data[(r["_input"], r["_value"])] = 0.0
    allv = list(data.values())
    vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)

    cw, ch = 30, 24
    left, top, right, bottom = 150, 70, 40, 60
    w = left + cw * len(values) + right
    h = top + ch * len(inputs) + bottom
    svg = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w, h),
           '<rect width="100%%" height="100%%" fill="white"/>',
           '<text x="%d" y="30" font-family="sans-serif" font-size="18" '
           'font-weight="bold">%s - %s</text>' % (left, eid, metric)]
    for j, v in enumerate(values):
        x = left + j * cw + cw / 2
        svg.append('<text x="%d" y="%d" font-family="sans-serif" font-size="10" '
                   'text-anchor="end" transform="rotate(-90 %d %d)">%s</text>'
                   % (x, top - 8, x, top - 8, v))
    for i, inp in enumerate(inputs):
        y = top + i * ch
        svg.append('<text x="%d" y="%d" font-family="sans-serif" font-size="12" '
                   'text-anchor="end">%s</text>' % (left - 6, y + ch * 0.7, inp))
        for j, v in enumerate(values):
            x = left + j * cw
            val = data.get((inp, v), 0.0)
            fill, dark = color(val, vmin, vmax)
            svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" '
                       'stroke="#dddddd"/>' % (x, y, cw, ch, fill))
            if metric == "sdc_rate" and val > 0:
                svg.append('<text x="%d" y="%d" font-family="sans-serif" '
                           'font-size="9" fill="%s" text-anchor="middle">%.2f</text>'
                           % (x + cw / 2, y + ch * 0.7,
                              "#ffffff" if dark else "#333333", val))
    # legend
    lx, ly = left, top + ch * len(inputs) + 30
    svg.append('<text x="%d" y="%d" font-family="sans-serif" font-size="10">%.3g</text>'
               % (lx, ly - 4, vmin))
    for k in range(50):
        c, _ = color(vmin + (vmax - vmin) * k / 49.0, vmin, vmax)
        svg.append('<rect x="%d" y="%d" width="4" height="10" fill="%s"/>' % (lx, ly, c))
        lx += 4
    svg.append('<text x="%d" y="%d" font-family="sans-serif" font-size="10">%.3g</text>'
               % (lx + 4, ly - 4, vmax))
    svg.append('</svg>')
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    with open(pivot_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["input"] + values)
        for inp in inputs:
            wtr.writerow([inp] + ["%.6f" % data.get((inp, v), 0.0) for v in values])
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cells", help="cells.csv path")
    p.add_argument("--metric", default="sdc_rate",
                   help="column to colour (sdc_rate, mean_abs_err, max_abs_err)")
    p.add_argument("--axis", default="bit",
                   help="axis row to plot: bit, instruction, size, position, register, input")
    p.add_argument("--experiment", default=None,
                   help="only render this experiment id")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    rows = load_cells(args.cells)
    if not rows:
        print("no cells found")
        return 1
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.cells))
    os.makedirs(outdir, exist_ok=True)
    by_eid = {}
    for r in rows:
        if r["_axis"] != args.axis:
            continue
        if args.experiment and r["experiment_id"] != args.experiment:
            continue
        by_eid.setdefault(r["experiment_id"], []).append(r)
    for eid, erows in sorted(by_eid.items()):
        tag = "%s_%s_%s" % (eid, args.axis, args.metric)
        out = os.path.join(outdir, "heatmap_%s.svg" % tag)
        piv = os.path.join(outdir, "pivot_%s.csv" % tag)
        r = render(eid, erows, args.metric, None, out, piv)
        if r:
            print("wrote %s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
