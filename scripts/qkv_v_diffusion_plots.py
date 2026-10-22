#!/usr/bin/env python3
"""Two separate diffusion figures for the kernel comparison."""
import csv
import json
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "results/qkv_v_kernels"
CPS = ["C0", "C1", "C2", "C3"]
CP_LABEL = ["C0\nGEMM V", "C1\nattention", "C2\nlayer14", "C3\nlayer27"]
VARIANTS = ["K0", "K1", "K2a", "K2b", "K3"]
COLORS = {"K0": "#1f77b4", "K1": "#ff7f0e", "K2a": "#2ca02c",
          "K2b": "#d62728", "K3": "#9467bd"}
NUMEL = {"C0": 579584, "C1": 3477504, "C2": 3477504, "C3": 3477504}


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def series(rows, variant, cp, field):
    vals = []
    for r in rows:
        if r.get("variant") != variant:
            continue
        if r["outcome_d0_" + cp] != "SDC":
            continue
        v = f(r[cp + "_" + field])
        if math.isfinite(v):
            vals.append(v)
    return vals


def med_iqr(vals):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (statistics.median(vals),
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)))


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, "comparison_all.csv"), encoding="utf-8")))
    setup()
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)

    # Figure 1: element diffusion
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(left=0.09, right=0.97, bottom=0.22,
                                              top=0.84, wspace=0.28))
    for ax, mode in zip(axes, ("count", "fraction")):
        for v in VARIANTS:
            med, lo, hi = [], [], []
            for cp in CPS:
                vals = series(rows, v, cp, "mismatch")
                if mode == "fraction":
                    vals = [x / NUMEL[cp] for x in vals]
                a, b, c = med_iqr(vals)
                med.append(a); lo.append(b); hi.append(c)
            x = np.arange(4)
            ax.plot(x, med, marker="o", ms=3.5, lw=1.2, color=COLORS[v], label=v)
            ax.fill_between(x, lo, hi, color=COLORS[v], alpha=0.10, linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks(range(4)); ax.set_xticklabels(CP_LABEL)
        ax.set_title("(a) changed elements" if mode == "count"
                     else "(b) fraction of output elements", loc="left")
        ax.set_ylabel("median changed elements" if mode == "count"
                      else "median changed fraction")
        ax.grid(True, which="major", axis="y", color="0.88", lw=0.5)
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("0.35")
    axes[0].legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")
    fig.suptitle("Element diffusion through the full stack", fontsize=10, y=0.98)
    fig.text(0.5, 0.045,
             "Median and IQR over D0 SDC runs. C0 = V; C1 = attention; C2 = layer14 block; C3 = layer27 block.",
             ha="center", va="bottom", fontsize=6.2, color="0.35")
    for ext in ("pdf", "png"):
        p = os.path.join(BASE, "figures", "diffusion_element_count." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)

    # Figure 2: numerical difference
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(left=0.09, right=0.97, bottom=0.22,
                                              top=0.84, wspace=0.28))
    for ax, field, title, ylab in (
        (axes[0], "max_abs", "(a) max absolute error", r"median max $|\Delta Y|$"),
        (axes[1], "rel_l2", "(b) relative L2 error", "median relative L2"),
    ):
        for v in VARIANTS:
            med = []
            for cp in CPS:
                vals = series(rows, v, cp, field)
                a, _, _ = med_iqr(vals)
                med.append(a)
            x = np.arange(4)
            ax.plot(x, med, marker="o", ms=3.5, lw=1.2, color=COLORS[v], label=v)
        ax.set_yscale("log")
        ax.set_xticks(range(4)); ax.set_xticklabels(CP_LABEL)
        ax.set_title(title, loc="left"); ax.set_ylabel(ylab)
        ax.grid(True, which="major", axis="y", color="0.88", lw=0.5)
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("0.35")
    axes[0].legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")
    fig.suptitle("Numerical difference at each checkpoint", fontsize=10, y=0.98)
    fig.text(0.5, 0.045,
             "Median over D0 SDC runs; finite values only. IQR omitted because the bit30 tail spans >10^30.",
             ha="center", va="bottom", fontsize=6.2, color="0.35")
    for ext in ("pdf", "png"):
        p = os.path.join(BASE, "figures", "diffusion_numeric_magnitude." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)

    # Save the plotted medians for the report
    out = {"element_count": {}, "numeric": {}}
    for v in VARIANTS:
        out["element_count"][v] = {}
        out["numeric"][v] = {}
        for cp in CPS:
            vals = series(rows, v, cp, "mismatch")
            a, b, c = med_iqr(vals)
            out["element_count"][v][cp] = {"median": a, "q25": b, "q75": c,
                                           "fraction": a / NUMEL[cp]}
            out["numeric"][v][cp] = {}
            for field in ("max_abs", "rel_l2"):
                vals2 = series(rows, v, cp, field)
                a2, b2, c2 = med_iqr(vals2)
                out["numeric"][v][cp][field] = {"median": a2, "q25": b2, "q75": c2}
    json.dump(out, open(os.path.join(BASE, "diffusion_summary.json"), "w", encoding="utf-8"),
              indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
