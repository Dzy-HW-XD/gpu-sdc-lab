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
import math
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


def color(v, vmin, vmax, log=False):
    if log:
        lo_v = max(vmin, 1e-12)
        hi_v = max(vmax, lo_v * 10.0)
        vv = max(v, lo_v)
        t = (math.log10(vv) - math.log10(lo_v)) / (
            math.log10(hi_v) - math.log10(lo_v))
    elif vmax <= vmin:
        t = 0.0
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # light grey (low) -> dark red (high)
    lo = (242, 242, 242)
    hi = (178, 24, 43)
    rgb = tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % rgb, (t > 0.6)


def fmt_val(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a < 0.01 or a >= 1000:
        return "%.0e" % v
    if a >= 10:
        return "%.0f" % v
    if a >= 1:
        return "%.2f" % v
    return "%.3g" % v


def render(eid, rows, metric, axis_label, outpath, pivot_path,
           log_color=False, annotate=False):
    inputs = sorted({r["_input"] for r in rows}, key=_input_rank)
    values = sorted({r["_value"] for r in rows}, key=_value_sort)
    if not inputs or not values:
        return None

    data = {}
    for r in rows:
        try:
            data[(r["_input"], r["_value"])] = float(r.get(metric) or 0.0)
        except ValueError:
            data[(r["_input"], r["_value"])] = 0.0
    allv = list(data.values())
    vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)

    ncols = len(values)
    cw = 34 if ncols > 12 else 84
    if annotate:
        cw = max(cw, 52)
    ch = 30
    font_col, font_row = 12, 14
    left = 170
    # top margin must fit the longest rotated column label
    maxlabel = max(len(str(v)) for v in values)
    top = 60 + int(maxlabel * font_col * 0.62) + 18
    right, bottom = 50, 90
    w = left + cw * ncols + right
    h = top + ch * len(inputs) + bottom
    x0 = left + cw * ncols  # right edge of grid

    mode = metric
    if log_color:
        mode += " (log10 colour)"
    if annotate:
        mode += " (cell = value)"
    svg = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'font-family="DejaVu Sans, Verdana, sans-serif">' % (w, h),
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<text x="%d" y="30" font-size="20" font-weight="bold">%s</text>'
           % (left, eid),
           '<text x="%d" y="50" font-size="14" fill="#555">%s  (%s axis)</text>'
           % (left, mode, axis_label or "input")]
    # x axis title (centered over grid)
    svg.append('<text x="%d" y="%d" font-size="13" fill="#333" '
               'text-anchor="middle">%s &#8594;</text>'
               % ((left + x0) / 2, top - maxlabel * font_col * 0.62 - 10,
                  (axis_label or "value")))
    # column labels: rotate -90 with text-anchor=start -> read upward, above grid
    for j, v in enumerate(values):
        x = left + j * cw + cw / 2 + 4
        y = top - 8
        svg.append('<text x="%d" y="%d" font-size="%d" text-anchor="start" '
                   'transform="rotate(-90 %d %d)">%s</text>'
                   % (x, y, font_col, x, y, v))
    # rows + cells
    for i, inp in enumerate(inputs):
        y = top + i * ch
        svg.append('<text x="%d" y="%d" font-size="%d" text-anchor="end">%s</text>'
                   % (left - 8, y + ch / 2 + font_row * 0.35, font_row, inp))
        for j, v in enumerate(values):
            x = left + j * cw
            val = data.get((inp, v), 0.0)
            fill, dark = color(val, vmin, vmax, log_color)
            svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" '
                       'stroke="#ffffff" stroke-width="1"/>' % (x, y, cw, ch, fill))
            label = None
            if metric == "sdc_rate":
                if val > 0:
                    label = "%.2f" % val
            elif annotate:
                label = fmt_val(val)
            if label:
                fsz = 10 if len(label) >= 5 else 11
                svg.append('<text x="%d" y="%d" font-size="%d" fill="%s" '
                           'text-anchor="middle">%s</text>'
                           % (x + cw / 2, y + ch / 2 + 4, fsz,
                              "#ffffff" if dark else "#222222", label))
    # y axis title (rotated, left)
    svg.append('<text x="%d" y="%d" font-size="13" fill="#333" text-anchor="middle" '
               'transform="rotate(-90 %d %d)">input distribution &#8594;</text>'
               % (24, top + ch * len(inputs) / 2, 24, top + ch * len(inputs) / 2))
    # legend
    ly = top + ch * len(inputs) + 40
    lx = left
    svg.append('<text x="%d" y="%d" font-size="12" fill="#333">%s</text>'
               % (lx, ly + 20, metric))
    svg.append('<text x="%d" y="%d" font-size="11" text-anchor="end">%.3g</text>'
               % (lx - 6, ly + 11, vmin))
    lo_v = max(vmin, 1e-12)
    hi_v = max(vmax, lo_v * 10.0)
    for k in range(60):
        t = k / 59.0
        if log_color:
            val_k = 10 ** (math.log10(lo_v) + t * (math.log10(hi_v) - math.log10(lo_v)))
        else:
            val_k = vmin + (vmax - vmin) * t
        c, _ = color(val_k, vmin, vmax, log_color)
        svg.append('<rect x="%d" y="%d" width="6" height="14" fill="%s"/>'
                   % (lx + k * 6, ly, c))
    svg.append('<text x="%d" y="%d" font-size="11">%.3g</text>'
               % (lx + 60 * 6 + 6, ly + 11, vmax))
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
    p.add_argument("--log-color", action="store_true",
                   help="colour by log10(value) (for wide-dynamic-range metrics)")
    p.add_argument("--annotate", action="store_true",
                   help="print the numeric value in every cell")
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
        suffix = ""
        if args.log_color:
            suffix += "_log"
        if args.annotate:
            suffix += "_annot"
        tag = "%s_%s_%s%s" % (eid, args.axis, args.metric, suffix)
        out = os.path.join(outdir, "heatmap_%s.svg" % tag)
        piv = os.path.join(outdir, "pivot_%s.csv" % tag)
        r = render(eid, erows, args.metric, args.axis, out, piv,
                   log_color=args.log_color, annotate=args.annotate)
        if r:
            print("wrote %s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
