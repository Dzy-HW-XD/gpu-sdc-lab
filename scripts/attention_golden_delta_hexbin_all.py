#!/usr/bin/env python3
"""All-data |Y_golden| vs |dY| hexbin, without the L4 threshold line.

For each stage, plots every finite, strictly positive |dY| element from the
representative run of each (stage, bit, position) cell.  Unlike the earlier
version, this does NOT filter points by the L4 threshold.

Zero |dY| values and non-finite outputs cannot be shown on a log axis; their
counts are reported in the footer.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = [("qkv_q", "QKV-Q"), ("qkv_k", "QKV-K"), ("qkv_v", "QKV-V"),
          ("qk", "QK"), ("pv", "PV")]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
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
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--reps", required=True, help=".../reps/faulty")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    setup_style()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    gold = np.fromfile(args.golden, dtype=np.float32)
    ag = np.abs(gold)

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.15), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.075, right=0.91, bottom=0.25,
                                              top=0.88, wspace=0.08))
    hb = None
    ns = []
    zeros = []
    nfs = []
    for ax, (stage, label) in zip(axes, STAGES):
        xs, ys = [], []
        zero = 0
        nf = 0
        for rec in rows:
            if rec["stage"] != stage:
                continue
            path = os.path.join(args.reps, "seq_" + str(rec["seq"]), "out.Y.bin")
            if not os.path.exists(path):
                continue
            fault = np.fromfile(path, dtype=np.float32)
            d = np.abs(fault - gold)
            good_gold = ag > 0
            finite = np.isfinite(d)
            zero += int((good_gold & finite & (d == 0)).sum())
            nf += int((good_gold & ~finite).sum())
            mask = good_gold & finite & (d > 0)
            xs.append(np.log10(ag[mask]))
            ys.append(np.log10(d[mask]))
        x = np.concatenate(xs) if xs else np.array([])
        y = np.concatenate(ys) if ys else np.array([])
        ns.append(len(x))
        zeros.append(zero)
        nfs.append(nf)
        hb = ax.hexbin(x, y, gridsize=55, bins="log", cmap="viridis",
                       mincnt=1, linewidths=0, rasterized=True)
        ax.set_xlim(-6, 4)
        ax.set_ylim(-6, 40)
        ax.set_xticks([-6, -4, -2, 0, 2, 4])
        ax.set_yticks([-4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        ax.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$",
                            r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])
        ax.set_yticklabels([r"$10^{-4}$", r"$10^{0}$", r"$10^{4}$",
                            r"$10^{8}$", r"$10^{12}$", r"$10^{16}$",
                            r"$10^{20}$", r"$10^{24}$", r"$10^{28}$",
                            r"$10^{32}$", r"$10^{36}$", r"$10^{40}$"])
        ax.set_title(label, fontsize=7, pad=3)
        ax.tick_params(labelbottom=(stage == "qkv_q"),
                       labelleft=(stage == "qkv_q"))
        ax.text(0.03, 0.97, "n=%s" % format(len(x), ","),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.8, color="0.25")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("0.35")

    axes[0].set_ylabel(r"$|\Delta Y|$")
    fig.text(0.49, 0.085, r"$|Y_{\mathrm{golden}}|$", ha="center", fontsize=7.5)
    cb = fig.colorbar(hb, cax=fig.add_axes([0.925, 0.25, 0.012, 0.63]))
    cb.set_label("count per bin (log)", fontsize=6.5, labelpad=3)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6)

    fig.text(0.5, 0.015,
             "All finite, positive |ΔY| points from the representative run per cell; "
             "L4 threshold removed. Zero ΔY and non-finite outputs excluded.",
             ha="center", va="bottom", fontsize=6.2, color="0.35")

    os.makedirs(args.outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(args.outdir, "golden_vs_delta_hexbin." + fmt)
        fig.savefig(path, dpi=args.dpi if fmt == "png" else 300)
        print("wrote", path)
    print("n per stage:", dict(zip([s for s, _ in STAGES], ns)))
    print("zero per stage:", dict(zip([s for s, _ in STAGES], zeros)))
    print("nonfinite per stage:", dict(zip([s for s, _ in STAGES], nfs)))
    plt.close(fig)


if __name__ == "__main__":
    main()
