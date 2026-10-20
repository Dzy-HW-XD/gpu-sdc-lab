#!/usr/bin/env python3
"""Final L4 SDC rate by stage + per-stage propagation statistics.

Part 1 (all runs): L4 SDC rate per stage (classification from config.L4.yaml)
with Wilson 95% confidence intervals.

Part 2 (representative runs): mean relative-L2 and nonzero fraction at
S, P, O, Ocat, Y, grouped by the injected stage.

Outputs (into --outdir):
  stage_rates.csv        stage: n, sdc, rate, ci_low, ci_high
  stage_propagation.csv  stage x tensor: mean_rel_l2, nonzero_fraction
  stage_summary.png      (left) L4 SDC rate; (right) propagation bars

Usage:
  attention_stage_summary.py --exp results/047_attn_5gemm_full \
      --manifest .../prop_data/manifest_local.csv --outdir .../stage_summary
"""
import argparse
import collections
import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["qkv_q", "qkv_k", "qkv_v", "qk", "pv"]
TENSORS = ["S", "P", "O", "Ocat", "Y"]
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


def mean_finite(vals):
    vv = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vv)) if vv else 0.0


def load(prefix, name, shape):
    a = np.fromfile(prefix + "." + name + ".bin", dtype=np.float32)
    return a.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(
        os.path.join(args.exp, "injection_trace.csv"), encoding="utf-8")))

    by = collections.defaultdict(lambda: [0, 0])  # stage -> [sdc, n]
    for r in rows:
        c = r.get("classification", "")
        if c in ("MASKED", "SDC", "CRASH"):
            by[r["stage"]][1] += 1
            if c == "SDC":
                by[r["stage"]][0] += 1
    with open(os.path.join(args.outdir, "stage_rates.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n", "sdc", "rate", "ci_low", "ci_high"])
        for st in STAGES:
            s, n = by[st]
            lo, hi = wilson(s, n)
            w.writerow([st, n, s, "%.4f" % (s / n if n else 0.0),
                        "%.4f" % lo, "%.4f" % hi])

    reps = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    meta_cache = {}
    prop = collections.defaultdict(lambda: {"rel": [], "nz": []})
    for r in reps:
        g = r["golden"]
        meta = meta_cache.get(g)
        if meta is None:
            with open(g + ".json", encoding="utf-8") as mf:
                meta = json.load(mf)
            meta_cache[g] = meta
        S, D, H = int(meta["S"]), int(meta["D"]), int(meta["H"])
        dh = int(meta.get("dh", D // H))
        shape = {"S": (H, S, S), "P": (H, S, S), "O": (H, S, dh),
                 "Ocat": (S, D), "Y": (S, D)}
        stage = r["stage"]
        for t in TENSORS:
            a = load(g, t, shape[t]).astype(np.float64)
            b = load(r["faulty"], t, shape[t]).astype(np.float64)
            d = b - a
            rel = float(np.sqrt((d * d).sum()) /
                        (np.sqrt((a * a).sum()) + 1e-300))
            prop[(stage, t)]["rel"].append(rel)
            prop[(stage, t)]["nz"].append(float((d != 0).mean()))
    with open(os.path.join(args.outdir, "stage_propagation.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "tensor", "runs", "mean_rel_l2",
                    "mean_nonzero_fraction", "frac_runs_nonzero"])
        for st in STAGES:
            for t in TENSORS:
                rel = prop[(st, t)]["rel"]
                nz = prop[(st, t)]["nz"]
                if not rel:
                    continue
                w.writerow([st, t, len(rel), "%.6g" % mean_finite(rel),
                            "%.6g" % (sum(nz) / len(nz)),
                            "%.4f" % (sum(1 for v in nz if v > 0) / len(nz))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    xs = np.arange(len(STAGES))
    rates, errs = [], []
    for st in STAGES:
        s, n = by[st]
        rate = s / n if n else 0.0
        lo, hi = wilson(s, n)
        rates.append(rate)
        errs.append([rate - lo, hi - rate])
    errs = np.array(errs).T
    ax1.bar(xs, rates, color=[COLORS[s] for s in STAGES], alpha=0.9)
    ax1.errorbar(xs, rates, yerr=errs, fmt="none", ecolor="black", capsize=4)
    for i, rate in enumerate(rates):
        ax1.text(i, rate, "%.3f" % rate, ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(STAGES, rotation=15)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("L4 SDC rate (Wilson 95% CI)")
    ax1.set_title("final SDC rate by injected stage (L4)")
    ax1.grid(True, axis="y", ls=":", alpha=0.5)

    tx = np.arange(len(TENSORS))
    width = 0.16
    floor = 1e-12
    maxv = 0.0
    for i, st in enumerate(STAGES):
        vals = [mean_finite(prop[(st, t)]["rel"]) for t in TENSORS]
        plotted = [max(v, floor) for v in vals]
        maxv = max(maxv, max([p for p in plotted if np.isfinite(p)] or [0.0]))
        ax2.bar(tx + (i - 2) * width, plotted, width, color=COLORS[st], label=st)
        for x, v in zip(tx, vals):
            if v <= 0.0:
                ax2.text(x + (i - 2) * width, floor, "0", ha="center",
                         va="bottom", fontsize=6)
    ax2.set_yscale("log")
    ax2.set_ylim(floor * 0.5, max(maxv, 1.0) * 30)
    ax2.set_xticks(tx)
    ax2.set_xticklabels(TENSORS)
    ax2.set_xlabel("observed tensor")
    ax2.set_ylabel("mean relative L2 over representative runs (log)")
    ax2.set_title("error propagation by injected stage (L4 reps)")
    ax2.grid(True, which="both", axis="y", ls=":", alpha=0.5)
    ax2.legend(fontsize=8, ncol=5)

    fig.tight_layout()
    p = os.path.join(args.outdir, "stage_summary.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    main()
