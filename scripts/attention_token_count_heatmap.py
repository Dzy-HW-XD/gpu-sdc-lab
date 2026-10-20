#!/usr/bin/env python3
"""Single-figure summary: how many output tokens does one injection affect?

Rows are injection stages; columns bin the number of affected output token
rows per run (conditional on the run being classified SDC).  The figure
shows directly that QKV-Q, QK, and PV are single-token, QKV-K is broad, and
QKV-V is intermediate.

Usage:
  python scripts/attention_token_count_heatmap.py \
      --footprint results/047_attn_5gemm_full/final_heatmaps_data_v2_nf/footprint.csv \
      --outdir results/047_attn_5gemm_full/final_heatmaps_v2/summary
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

STAGES = [("qkv_q", "QKV-Q"), ("qkv_k", "QKV-K"), ("qkv_v", "QKV-V"),
          ("qk", "QK"), ("pv", "PV")]
EDGES = [1, 2, 4, 8, 16, 32, 64, 129]
LABELS = ["1", "2–3", "4–7", "8–15", "16–31", "32–63", "64–128"]


def bin_index(n):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= n < EDGES[i + 1]:
            return i
    return None


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "text.color": "0.15",
        "axes.labelcolor": "0.15",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    setup_style()

    rows = list(csv.DictReader(open(args.footprint, encoding="utf-8")))
    mat = np.zeros((len(STAGES), len(LABELS)), dtype=float)
    med = {}
    nsdc = {}
    ntot = {}
    for i, (stage, _) in enumerate(STAGES):
        sub = [r for r in rows if r["stage"] == stage]
        sdc = [r for r in sub if r["classification"] == "SDC"]
        ntot[stage] = len(sub)
        nsdc[stage] = len(sdc)
        vals = [int(r["affected_tokens"]) for r in sdc]
        if vals:
            med[stage] = float(np.median(vals))
            for v in vals:
                b = bin_index(v)
                if b is not None:
                    mat[i, b] += 1
            mat[i] /= len(vals)
        else:
            med[stage] = float("nan")

    cmap = plt.get_cmap("Blues").copy()
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035], left=0.245,
                          right=0.90, bottom=0.20, top=0.86, wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(np.arange(len(STAGES)))
    ax.set_yticklabels(
        ["%s\n%d/%d SDC, med %s" % (
            label, nsdc[stage], ntot[stage],
            ("%.0f" % med[stage]) if np.isfinite(med[stage]) else "—")
         for stage, label in STAGES],
        fontsize=6.8, linespacing=1.15)
    ax.set_xlabel("Affected output tokens per run")
    ax.set_ylabel("Injection stage", labelpad=6)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.tick_params(axis="y", length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.35")

    # white grid between cells
    for x in np.arange(0.5, len(LABELS), 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(0.5, len(STAGES), 1):
        ax.axhline(y, color="white", lw=0.8)

    norm = Normalize(0.0, 1.0)
    for i in range(len(STAGES)):
        for j in range(len(LABELS)):
            v = mat[i, j]
            if v <= 0:
                txt, col = "0", "0.45"
            else:
                txt = ("%.1f" % (100 * v)) if v < 0.10 else ("%.0f" % (100 * v))
                r, g, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                col = "white" if lum < 0.5 else "0.12"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color=col)

    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Fraction of SDC runs", fontsize=7, labelpad=3)
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6.5)

    # left-side annotation is embedded in the y tick labels

    # bracket separating single-token from multi-token bins (top)
    tr = ax.get_xaxis_transform()
    ax.plot([-0.5, 0.5], [1.015, 1.015], transform=tr, color="0.35", lw=0.6,
            clip_on=False)
    ax.plot([0.5, 6.5], [1.015, 1.015], transform=tr, color="0.35", lw=0.6,
            clip_on=False)
    ax.text(0.0, 1.035, "single token", transform=tr, ha="center",
            va="bottom", fontsize=6.5, color="0.35")
    ax.text(3.5, 1.035, "multiple tokens", transform=tr, ha="center",
            va="bottom", fontsize=6.5, color="0.35")

    os.makedirs(args.outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(args.outdir, "token_count_distribution." + fmt)
        fig.savefig(path, dpi=args.dpi if fmt == "png" else 300)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
