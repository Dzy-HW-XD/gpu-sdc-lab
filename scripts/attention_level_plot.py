#!/usr/bin/env python3
"""Plot attention results filtered by tolerance level L0-L5.

Reads attention_levels.csv / attention_by_level.csv produced by
scripts/attention_collect.py.

Panels:
  A. SDC rate vs tolerance level, one line per (stage, head) (positions pooled).
  B. heatmap: level (rows) x injected bit (cols), SDC count / total, annotated.

Usage:
  attention_level_plot.py --exp results/042_attn_bit_smoke --out level_view.png
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    lv = load(os.path.join(args.exp, "attention_levels.csv"))
    levels = [l for l in LEVEL_ORDER if any(r["level"] == l for r in lv)]
    levels += sorted(set(r["level"] for r in lv) - set(levels))
    bits = sorted(set(int(r["bit"]) for r in lv if r.get("bit") not in ("", None)))

    # ---- panel A: SDC rate vs level per (stage, head) ------------------
    series = sorted(set((r["stage"], str(r["head"])) for r in lv))
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.0, 1.35]})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (stage, head) in enumerate(series):
        xs, ys = [], []
        for l in levels:
            rs = [r for r in lv if r["stage"] == stage and str(r["head"]) == head
                  and r["level"] == l]
            denom = sum(1 for r in rs if r["classification"] in ("MASKED", "SDC", "CRASH"))
            sdc = sum(1 for r in rs if r["classification"] == "SDC")
            xs.append(l)
            ys.append(sdc / denom if denom else np.nan)
        label = stage if head in ("", "None") else "%s/h%s" % (stage, head)
        ax.plot(range(len(levels)), ys, "-o", color=colors[i % len(colors)],
                label=label)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("tolerance level")
    ax.set_ylabel("SDC rate  (SDC / (MASKED+SDC+CRASH))")
    ax.set_title("SDC rate vs tolerance level")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # ---- panel B: level x bit heatmap ---------------------------------
    rate = np.full((len(levels), len(bits)), np.nan)
    text = [["" for _ in bits] for _ in levels]
    for li, l in enumerate(levels):
        for bi, b in enumerate(bits):
            rs = [r for r in lv if r["level"] == l and str(r["bit"]) == str(b)]
            denom = sum(1 for r in rs if r["classification"] in ("MASKED", "SDC", "CRASH"))
            sdc = sum(1 for r in rs if r["classification"] == "SDC")
            if denom:
                rate[li, bi] = sdc / denom
                text[li][bi] = "%d/%d" % (sdc, denom)
    im = axb.imshow(rate, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    for li in range(len(levels)):
        for bi in range(len(bits)):
            if text[li][bi]:
                axb.text(bi, li, text[li][bi], ha="center", va="center",
                         fontsize=8, color="black")
    axb.set_xticks(range(len(bits)))
    axb.set_xticklabels([str(b) for b in bits])
    axb.set_yticks(range(len(levels)))
    axb.set_yticklabels(levels)
    axb.set_xlabel("injected bit")
    axb.set_ylabel("tolerance level")
    axb.set_title("SDC count / total, by level x bit  (red=all SDC, green=none)")
    fig.colorbar(im, ax=axb, label="SDC rate")

    title = args.title or ("attention tolerance view  [%s]"
                           % os.path.basename(os.path.normpath(args.exp)))
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
