#!/usr/bin/env python3
"""Build the one summary figure for the high-bit output-corruption experiment."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


MODEL_ORDER = ["pythia160m", "pythia410m", "pythia1b", "qwen05b", "qwen15b"]
MODEL_LABEL = {
    "pythia160m": "Pythia-160M",
    "pythia410m": "Pythia-410M",
    "pythia1b": "Pythia-1B",
    "qwen05b": "Qwen2.5-0.5B",
    "qwen15b": "Qwen2.5-1.5B",
}
BITS = [28, 29, 30, 31]
LAYERS = ["E", "M", "L"]


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
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val(rows, ms, **kwargs):
    for r in rows:
        if r["model_short"] != ms:
            continue
        if all(str(r.get(k, "")) == str(v) for k, v in kwargs.items()):
            return float(r["catastrophic_rate"]) if r["catastrophic_rate"] else 0.0
    return 0.0


def draw_heat(ax, mat, row_labels, col_labels, title, xlabel, vmin=0, vmax=1):
    im = ax.imshow(mat, cmap="Reds", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=7)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "%.0f" % (100 * v), ha="center", va="center",
                    fontsize=7.2, color="white" if v > 0.55 else "0.15")
    return im


def main():
    setup()
    base = "results/llm_highbit_output"
    by_bit = read_csv(os.path.join(base, "summary_by_model_bit.csv"))
    by_layer = read_csv(os.path.join(base, "summary_by_model_layer.csv"))
    comp = read_csv(os.path.join(base, "outcome_by_model_2829.csv"))
    by_model = read_csv(os.path.join(base, "summary_by_model.csv"))

    model_n = {r["model_short"]: int(r["n_eval"]) // 12 for r in by_model}
    row_labels = ["%s (%d/24)" % (MODEL_LABEL[m], model_n[m]) for m in MODEL_ORDER]

    mat_bit = np.array([[val(by_bit, m, bit=str(b)) for b in BITS]
                        for m in MODEL_ORDER])
    mat_layer = np.array([[val(by_layer, m, layer_tag=l) for l in LAYERS]
                          for m in MODEL_ORDER])

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02],
                          left=0.17, right=0.94, top=0.88, bottom=0.16,
                          wspace=0.40, hspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    im = draw_heat(ax_a, mat_bit, row_labels, [str(b) for b in BITS],
                   "(a)  catastrophic rate by bit", "FP32 bit position")
    draw_heat(ax_b, mat_layer, [MODEL_LABEL[m] for m in MODEL_ORDER], LAYERS,
              "(b)  catastrophic rate by depth", "injection depth")
    ax_b.tick_params(axis="y", labelsize=6.6, pad=2)

    # shared colorbar under the two heatmaps
    cax = fig.add_axes([0.32, 0.505, 0.38, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])
    cb.set_label("catastrophic output rate", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)

    # outcome composition at bits 28-29
    colors = {
        "garbled": "#B03A2E",
        "wrong": "#E67E22",
        "partial": "#A6ACAF",
        "preserved": "#2E86C1",
    }
    cats = ["garbled", "wrong", "partial", "preserved"]
    labels = ["garbled", "wrong", "partial", "preserved"]
    y = np.arange(len(MODEL_ORDER))
    left = np.zeros(len(MODEL_ORDER))
    for cat, label in zip(cats, labels):
        vals = np.array([float(next(r for r in comp if r["model_short"] == m)[cat + "_rate"])
                         for m in MODEL_ORDER])
        ax_c.barh(y, vals, left=left, height=0.62, color=colors[cat],
                  edgecolor="white", linewidth=0.7, label=label)
        for i, v in enumerate(vals):
            if v >= 0.07:
                ax_c.text(left[i] + v / 2, y[i], "%.0f" % (100 * v),
                          ha="center", va="center", fontsize=6.8,
                          color="white" if cat in ("garbled", "wrong", "preserved") else "0.15")
        left += vals
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r for r in row_labels])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("fraction of clean-correct runs")
    ax_c.set_title("(c)  outcome composition at bits 28–29", loc="left", pad=7)
    ax_c.tick_params(axis="both", length=0)
    ax_c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
                frameon=False, handlelength=1.2, columnspacing=1.3)
    n_labels = [int(next(r for r in comp if r["model_short"] == m)["n_eval"])
                for m in MODEL_ORDER]
    for i, n in enumerate(n_labels):
        ax_c.text(1.01, y[i], "n=%d" % n, va="center", ha="left", fontsize=6.5,
                  color="0.35")

    fig.text(0.5, 0.955, "High-bit activation faults and output corruption",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.925,
             "single-element FP32 activation flip at the last prompt token; greedy decode",
             ha="center", va="top", fontsize=7.2, color="0.35")
    fig.text(0.5, 0.035,
             "catastrophic = literal garble or answer absent; partial = answer present but altered; "
             "percentages are rounded.",
             ha="center", va="bottom", fontsize=6.3, color="0.35")

    outdir = os.path.join(base, "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, "highbit_output_summary." + ext)
        fig.savefig(path, dpi=600 if ext == "png" else 300)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
