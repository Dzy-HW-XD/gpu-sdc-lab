#!/usr/bin/env python3
"""One-figure prefill propagation summary for an LLM injection experiment.

Rows = injection layer, columns = injection token position.
Each panel is output layer (y) x token position (x), colour = log10 max|dH|.
The injection point is marked with dashed lines; cells outside the causal
region (out_layer <= inject_layer or token < inject_pos) are grey.

Usage:
  python scripts/llm_prefill_propagation_plot.py \
      --data results/llm_pilot/prefill_activation \
      --out results/llm_pilot/prefill_activation/figures \
      --bit max
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

INK = "0.15"
BAD = "#efefef"


def setup():
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
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit", default="max", help="'max' or an integer bit")
    ap.add_argument("--vmin", type=float, default=-3.0)
    ap.add_argument("--vmax", type=float, default=6.0)
    ap.add_argument("--formats", default="pdf,png")
    args = ap.parse_args()
    setup()

    runs = list(csv.DictReader(open(os.path.join(args.data, "prefill_runs.csv"), encoding="utf-8")))
    matrix = list(csv.DictReader(open(os.path.join(args.data, "prefill_propagation_matrix.csv"), encoding="utf-8")))
    layers = sorted({int(r["run_layer"]) for r in matrix})
    positions = sorted({int(r["pos"]) for r in matrix})
    out_layers = sorted({int(r["out_layer"]) for r in matrix})
    tokens = sorted({int(r["token"]) for r in matrix})
    nlo, nt = len(out_layers), max(tokens) + 1
    print("layers", layers, "positions", positions, "out_layers", out_layers,
          "tokens", tokens, flush=True)

    # max_abs per (inject_layer, inject_pos, out_layer, token)
    # if bit == max, take max over bits; otherwise filter one bit
    def selected(bit):
        return [r for r in matrix if bit == "max" or str(r["bit"]) == str(bit)]

    selected_rows = selected(args.bit)
    arr = np.full((len(layers), len(positions), nlo, nt), np.nan, dtype=float)
    for r in selected_rows:
        li = layers.index(int(r["run_layer"]))
        pi = positions.index(int(r["pos"]))
        ol = int(r["out_layer"])
        if ol not in out_layers:
            continue
        oi = out_layers.index(ol)
        t = int(r["token"])
        v = float(r["max_abs"])
        if not np.isfinite(v) or v <= 0:
            continue
        if np.isnan(arr[li, pi, oi, t]) or v > arr[li, pi, oi, t]:
            arr[li, pi, oi, t] = v
    # mask causal no-effect region
    for li, L in enumerate(layers):
        for pi, P in enumerate(positions):
            for oi, ol in enumerate(out_layers):
                if ol <= L:
                    arr[li, pi, oi, :] = np.nan
            arr[li, pi, :, :P] = np.nan

    # top1 changes per (layer, pos): count bits with top1 change among selected bits
    top1 = {}
    for r in runs:
        if str(r["bit"]) != str(args.bit) and args.bit != "max":
            continue
        key = (int(r["layer"]), int(r["pos"]))
        top1.setdefault(key, [0, 0])
        top1[key][0] += int(r["top1_change"])
        top1[key][1] += 1

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(BAD)
    norm = Normalize(vmin=args.vmin, vmax=args.vmax)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(len(layers), len(positions) + 1,
                          width_ratios=[1] * len(positions) + [0.045],
                          left=0.12, right=0.94, bottom=0.10, top=0.93,
                          wspace=0.10, hspace=0.14)
    im = None
    for li, L in enumerate(layers):
        for pi, P in enumerate(positions):
            ax = fig.add_subplot(gs[li, pi])
            a = arr[li, pi]
            with np.errstate(invalid="ignore"):
                loga = np.where(a > 0, np.log10(a), np.nan)
            im = ax.imshow(np.ma.masked_invalid(loga), origin="lower",
                           extent=(-0.5, nt - 0.5, -0.5, nlo - 0.5),
                           aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest", rasterized=True)
            ax.set_facecolor(BAD)
            # injection boundary: before the first affected output layer (L+1)
            first = out_layers.index(L + 1) if (L + 1) in out_layers else 0
            ax.axhline(first - 0.5, color="white", lw=0.8, ls="--")
            ax.axvline(P - 0.5, color="white", lw=0.8, ls="--")
            ax.set_xlim(-0.5, nt - 0.5)
            ax.set_ylim(-0.5, nlo - 0.5)
            ax.set_xticks([0, 10, 20])
            yticks = [out_layers.index(y) for y in (1, 6, 12) if y in out_layers]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(out_layers[i]) for i in yticks])
            ax.tick_params(length=2, width=0.4, pad=1.5,
                           labelbottom=(li == len(layers) - 1),
                           labelleft=(pi == 0))
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("0.35")
            if li == 0:
                ax.set_title("p = %d" % P, fontsize=7, pad=3)
            if pi == 0:
                ax.set_ylabel("output layer", fontsize=6.5)
                ax.text(-0.30, 0.5, "inject L%d" % L, transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=6.5,
                        color="0.25")
            c = top1.get((L, P), [0, 0])
            if c[1]:
                ax.text(0.98, 0.03, "top1 %d/%d" % (c[0], c[1]),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=5.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.45,
                                  edgecolor="none", pad=1.2))

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2, 4, 6])
    cb.set_label(r"$\log_{10}\max|\Delta h|$", fontsize=7, labelpad=3)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6)

    bit_label = "max over bits 24/28/30" if args.bit == "max" else "bit %s" % args.bit
    fig.text(0.5, 0.035,
             "Injection: %s; rows = injection layer, columns = injection token; "
             "grey = no causal effect; colour clipped at $10^6$." % bit_label,
             ha="center", va="bottom", fontsize=6.2, color="0.35")

    os.makedirs(args.out, exist_ok=True)
    for fmt in [f.strip().lower() for f in args.formats.split(",") if f.strip()]:
        path = os.path.join(args.out, "prefill_propagation." + fmt)
        fig.savefig(path, dpi=600 if fmt == "png" else 300)
        print("wrote", path, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
