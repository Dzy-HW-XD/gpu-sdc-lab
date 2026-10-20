#!/usr/bin/env python3
"""Publication-quality overview of the L4 spatial footprint (v2).

5 stages x 3 fault positions; each panel shows the number of runs (out of 30)
in which an output element is L4-affected, reduced over bits 24-30 with
`--stat max` (default; strongest footprint) or `--stat mean`.

The figure has no internal title: use the LaTeX caption.  It is sized for a
double-column page (7.2 in wide).

Usage:
  python scripts/attention_overview_heatmaps_v2.py \
      --data results/047_attn_5gemm_full/final_heatmaps_data_v2 \
      --outdir results/047_attn_5gemm_full/final_heatmaps_v2/overview
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

S, D = 128, 2048
STAGES = [("qkv_q", "QKV-Q"), ("qkv_k", "QKV-K"), ("qkv_v", "QKV-V"),
          ("qk", "QK"), ("pv", "PV")]
POSITIONS = [("early", "Early"), ("middle", "Middle"), ("late", "Late")]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bits", default="24,25,26,27,28,29,30")
    ap.add_argument("--stat", choices=("max", "mean"), default="max")
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    bits = [int(b) for b in args.bits.split(",") if b.strip()]
    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    setup_style()
    rows = list(csv.DictReader(
        open(os.path.join(args.data, "final_heatmaps_manifest.csv"), encoding="utf-8")))
    idx = {(r["stage"], r["position"], int(r["bit"])): r for r in rows}

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(5, 3, left=0.10, right=0.88, bottom=0.09, top=0.92,
                          wspace=0.06, hspace=0.10)
    im = None
    for i, (st, st_label) in enumerate(STAGES):
        for j, (pos, pos_label) in enumerate(POSITIONS):
            ax = fig.add_subplot(gs[i, j])
            arrs = []
            for b in bits:
                rec = idx[(st, pos, b)]
                c = np.fromfile(os.path.join(args.data, "count",
                                             rec["tag"] + ".bin"),
                                dtype=np.uint8).reshape(S, D)
                arrs.append(c.astype(np.float32))
            m = np.max(arrs, axis=0) if args.stat == "max" else np.mean(arrs, axis=0)
            im = ax.imshow(m, extent=(0, D, 0, S), origin="lower", aspect="auto",
                           cmap=plt.get_cmap("Blues"),
                           norm=PowerNorm(gamma=0.5, vmin=0, vmax=30),
                           interpolation="nearest", rasterized=True)
            ax.set_xlim(0, D)
            ax.set_ylim(0, S)
            ax.set_xticks([0, 1024])
            ax.set_yticks([0, 64, 128])
            ax.tick_params(labelbottom=(i == 4), labelleft=(j == 0),
                           length=2, width=0.4, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("0.35")
            if j == 0:
                ax.set_ylabel(st_label, fontsize=7, labelpad=4)
            if i == 0:
                ax.set_title(pos_label, fontsize=7, pad=3)

    cax = fig.add_axes([0.905, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cax, orientation="vertical",
                      ticks=[0, 1, 5, 10, 20, 30])
    stat_label = "max" if args.stat == "max" else "mean"
    cb.set_label("Affected runs (of 30), %s over bits %d–%d"
                 % (stat_label, min(bits), max(bits)), fontsize=6.5, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6)

    os.makedirs(args.outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(args.outdir, "overview_support_stage_position." + fmt)
        fig.savefig(path, dpi=args.dpi if fmt == "png" else 300)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
