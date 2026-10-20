#!/usr/bin/env python3
"""Plot a bit-sweep attention smoke test from attention_summary.csv.

Two panels:
  top   : per-bit Y relative-L2 (log) for each stage/head series; MASKED runs
          have zero error and are drawn at a floor with an "M" label;
  bottom: a classification strip (SDC red / MASKED grey) per series x bit.

Usage:
  attention_bit_plot.py --summary results/042_attn_bit_smoke/attention_summary.csv \
      --out results/042_attn_bit_smoke/bit_smoke.png
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def series_key(r):
    h = r.get("head") or ""
    if h in ("", "None"):
        return r["stage"]
    return "%s/h%s" % (r["stage"], h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty summary")
    series = sorted(set(series_key(r) for r in rows))
    bits = sorted(set(int(r["bit"]) for r in rows))
    smap = {(series_key(r), int(r["bit"])): r for r in rows}

    vals = [float(r["y_rel_l2"]) for r in rows
            if r.get("y_rel_l2") not in ("", None)]
    vmax = max(vals) if vals and max(vals) > 0 else 1.0
    floor = vmax * 1e-3

    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3]})

    width = 0.8 / max(1, len(series))
    x = np.arange(len(bits))
    for i, s in enumerate(series):
        color = COLORS[i % len(COLORS)]
        xs, hs, masked = [], [], []
        for j, b in enumerate(bits):
            r = smap.get((s, b))
            if r is None:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            if r.get("classification") == "SDC":
                hs.append(max(float(r["y_rel_l2"]), floor))
                masked.append(False)
            else:
                hs.append(floor)
                masked.append(True)
        bars = ax.bar(xs, hs, width=width * 0.95, color=color,
                      alpha=0.9, label=s)
        for bar, mk, j, b in zip(bars, masked, xs, [0] * len(xs)):
            if mk:
                bar.set_facecolor("#cccccc")
                bar.set_edgecolor("#888888")
                ax.text(bar.get_x() + bar.get_width() / 2, floor,
                        "M", ha="center", va="bottom", fontsize=8,
                        color="#555555")
            else:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        "%.3g" % v, ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, vmax * 5)
    ax.set_ylabel("Y relative L2 error  (log; M = MASKED)")
    title = args.title or ("attention bit sweep  [%s]"
                           % os.path.basename(os.path.dirname(args.summary)))
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=len(series))
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.5)

    for i, s in enumerate(series):
        for j, b in enumerate(bits):
            r = smap.get((s, b))
            if r is None:
                continue
            sdc = r.get("classification") == "SDC"
            axc.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                    facecolor="#c44e52" if sdc else "#dddddd",
                                    edgecolor="white"))
            axc.text(j, i, "SDC" if sdc else "M", ha="center", va="center",
                     fontsize=8, color="white" if sdc else "#555555")
    axc.set_xlim(-0.5, len(bits) - 0.5)
    axc.set_ylim(len(series) - 0.5, -0.5)
    axc.set_yticks(range(len(series)))
    axc.set_yticklabels(series, fontsize=9)
    axc.set_xticks(range(len(bits)))
    axc.set_xticklabels([str(b) for b in bits])
    axc.set_xlabel("injected bit")
    axc.set_title("classification", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
