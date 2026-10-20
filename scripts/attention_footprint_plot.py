#!/usr/bin/env python3
"""Token/feature footprint of L4 SDC runs, by injected stage.

Reads footprint.csv (from attention_final_heatmaps_data.py) and plots:
  A. affected token count per SDC run, by stage (box plot)
  B. fraction of SDC runs affecting exactly one token, by stage
  C. median affected-token count vs bit, per stage

Also writes footprint_summary.csv.

Usage: attention_footprint_plot.py --footprint <footprint.csv> --outdir <dir>
"""
import argparse
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["qkv_q", "qkv_k", "qkv_v", "qk", "pv"]
COLORS = {
    "qkv_q": "#1f77b4",
    "qkv_k": "#17becf",
    "qkv_v": "#2ca02c",
    "qk": "#ff7f0e",
    "pv": "#d62728",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(args.footprint, encoding="utf-8")))
    sdc = [r for r in rows if r.get("classification") == "SDC"]
    by_stage = collections.defaultdict(list)
    by_bit = collections.defaultdict(list)
    for r in sdc:
        st = r["stage"]
        tok = int(r["affected_tokens"])
        by_stage[st].append(r)
        by_bit[(st, int(r["bit"]))].append(tok)

    with open(os.path.join(args.outdir, "footprint_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n_sdc", "median_tokens", "mean_tokens",
                    "frac_one_token", "median_features", "median_changed"])
        for st in STAGES:
            rs = by_stage.get(st, [])
            if not rs:
                continue
            tok = np.array([int(r["affected_tokens"]) for r in rs])
            fea = np.array([int(r["affected_features"]) for r in rs])
            chg = np.array([int(r["changed"]) for r in rs])
            w.writerow([st, len(rs), "%.1f" % np.median(tok),
                        "%.2f" % tok.mean(), "%.4f" % (tok == 1).mean(),
                        "%.1f" % np.median(fea), "%.1f" % np.median(chg)])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    data = [np.array([int(r["affected_tokens"]) for r in by_stage.get(st, [])])
            for st in STAGES]
    ax1.boxplot([d if len(d) else [0.0] for d in data], showfliers=False)
    ax1.set_xticks(range(1, len(STAGES) + 1))
    ax1.set_xticklabels(STAGES, rotation=15)
    ax1.set_ylabel("affected tokens (of S=128)")
    ax1.set_title("affected token count per L4 SDC run")
    ax1.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.tick_params(axis="x", rotation=15)

    xs = np.arange(len(STAGES))
    frac = [((d == 1).mean() if len(d) else 0.0) for d in data]
    ax2.bar(xs, frac, color=[COLORS[s] for s in STAGES])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(STAGES, rotation=15)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("fraction of SDC runs")
    ax2.set_title("runs whose Y diff is exactly one token row")
    for i, v in enumerate(frac):
        ax2.text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)

    bits = sorted(set(int(r["bit"]) for r in sdc))
    for st in STAGES:
        vals = [np.median(by_bit[(st, b)]) if by_bit[(st, b)] else np.nan
                for b in bits]
        ax3.plot(bits, vals, "-o", color=COLORS[st], label=st)
    ax3.set_xlabel("injected bit")
    ax3.set_ylabel("median affected tokens")
    ax3.set_title("affected tokens vs bit (L4 SDC)")
    ax3.grid(True, ls=":", alpha=0.5)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(args.outdir, "footprint.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
