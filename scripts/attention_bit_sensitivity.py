#!/usr/bin/env python3
"""Bit-position sensitivity (L4): SDC rate and error magnitude per bit/stage.

Reads injection_trace.csv (classification is L4) and produces:
  A. L4 SDC rate vs bit, per stage (Wilson 95% CI error bars)
  B. median Y relative-L2 among SDC vs bit, per stage (log)
  C. heatmap: L4 SDC rate by bit x position

Outputs bit_sensitivity.png and bit_sensitivity.csv.

Usage: attention_bit_sensitivity.py --trace results/047.../injection_trace.csv \
    --outdir results/047.../bit_sensitivity
"""
import argparse
import collections
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["qkv_q", "qkv_k", "qkv_v", "qk", "pv"]
POSITIONS = ["early", "middle", "late"]
COLORS = {
    "qkv_q": "#1f77b4",
    "qkv_k": "#17becf",
    "qkv_v": "#2ca02c",
    "qk": "#ff7f0e",
    "pv": "#d62728",
}


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = s / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(args.trace, encoding="utf-8")))
    bits = sorted(set(int(r["bit"]) for r in rows
                      if r.get("bit") not in ("", None)))

    cnt = collections.defaultdict(lambda: [0, 0])      # (bit, stage) -> [sdc,n]
    mags = collections.defaultdict(list)               # (bit, stage) -> rel_l2
    pos = collections.defaultdict(lambda: [0, 0])      # (bit, position) -> [sdc,n]
    for r in rows:
        b = int(r["bit"])
        st = r["stage"]
        cls = r.get("classification", "")
        if cls in ("MASKED", "SDC", "CRASH"):
            cnt[(b, st)][1] += 1
            pos[(b, r["position"])][1] += 1
            if cls == "SDC":
                cnt[(b, st)][0] += 1
                pos[(b, r["position"])][0] += 1
                try:
                    mags[(b, st)].append(float(r["y_rel_l2"]))
                except (TypeError, ValueError):
                    pass

    with open(os.path.join(args.outdir, "bit_sensitivity.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "bit", "n", "sdc", "sdc_rate", "ci_low", "ci_high",
                    "median_rel_l2_sdc"])
        for st in STAGES:
            for b in bits:
                s, n = cnt[(b, st)]
                lo, hi = wilson(s, n)
                med = ""
                mvals = [v for v in mags[(b, st)] if np.isfinite(v)]
                if mvals:
                    med = float(np.median(mvals))
                w.writerow([st, b, n, s, "%.4f" % (s / n if n else 0.0),
                            "%.4f" % lo, "%.4f" % hi,
                            "%.6g" % med if med != "" else ""])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    ax = axes[0]
    for st in STAGES:
        rate, lo_e, hi_e = [], [], []
        for b in bits:
            s, n = cnt[(b, st)]
            rate.append(s / n if n else 0.0)
            lo, hi = wilson(s, n)
            lo_e.append(max(0.0, rate[-1] - lo))
            hi_e.append(max(0.0, hi - rate[-1]))
        ax.errorbar(bits, rate, yerr=[lo_e, hi_e], fmt="-o", color=COLORS[st],
                    capsize=4, label=st)
    ax.set_xlabel("injected bit")
    ax.set_ylabel("L4 SDC rate (Wilson 95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("L4 SDC rate vs bit")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    for st in STAGES:
        med = []
        for b in bits:
            mvals = [v for v in mags[(b, st)] if np.isfinite(v)]
            med.append(float(np.median(mvals)) if mvals else np.nan)
        ax.plot(bits, np.maximum(med, 1e-12), "-o", color=COLORS[st], label=st)
    ax.set_yscale("log")
    ax.set_xlabel("injected bit")
    ax.set_ylabel("median Y relative L2 among SDC (log)")
    ax.set_title("error magnitude vs bit (L4 SDC)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    ax = axes[2]
    mat = np.full((len(bits), len(POSITIONS)), np.nan)
    for i, b in enumerate(bits):
        for j, p in enumerate(POSITIONS):
            s, n = pos[(b, p)]
            if n:
                mat[i, j] = s / n
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    for i in range(len(bits)):
        for j in range(len(POSITIONS)):
            if not np.isnan(mat[i, j]):
                s, n = pos[(bits[i], POSITIONS[j])]
                ax.text(j, i, "%d/%d" % (s, n), ha="center", va="center",
                        fontsize=8)
    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels(POSITIONS)
    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels([str(b) for b in bits])
    ax.set_xlabel("position (early/middle/late)")
    ax.set_ylabel("injected bit")
    ax.set_title("L4 SDC count / total by bit x position")
    fig.colorbar(im, ax=ax, label="L4 SDC rate")

    fig.tight_layout()
    p = os.path.join(args.outdir, "bit_sensitivity.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    main()
