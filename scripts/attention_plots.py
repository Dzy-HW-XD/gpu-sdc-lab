#!/usr/bin/env python3
"""Attention result plots (matplotlib; run locally / where matplotlib exists).

Input is the metrics JSON produced by scripts/attention_metrics.py (which
records the golden/faulty prefixes and per-tensor metrics). Produces:

  output_diff.png    |Y_faulty - Y_golden| heatmap (log colour), changed cells marked
  head_errors.png    per-head Frobenius error (from O or P)
  propagation.png    relative-L2 error per observed tensor/stage
  outcome_rates.png  MASKED/SDC/CRASH/NOT_INJECTED counts (with --runs-jsonl)

Usage:
  attention_plots.py --metrics metrics.json --outdir plots [--runs-jsonl runs.jsonl]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ORDER = ["Q", "K", "V", "S", "P", "O", "Ocat", "Y"]


def load(prefix, name, shape):
    a = np.fromfile(prefix + "." + name + ".bin", dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise ValueError("%s.%s.bin: %d elems != %s"
                         % (prefix, name, a.size, shape))
    return a.reshape(shape)


def plot_diff(r, outdir):
    cfg = r["config"]
    S, D = int(cfg["S"]), int(cfg["D"])
    g = load(r["golden"], "Y", (S, D))
    f = load(r["faulty"], "Y", (S, D))
    d = f - g
    ad = np.abs(d)
    tol = r.get("tolerance", {})
    changed = ad > tol.get("abs", 0.0) + tol.get("rel", 0.0) * np.abs(g)

    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = float(ad.max())
    if vmax > 0:
        pos = ad[ad > 0]
        vmin = max(float(pos.min()), vmax * 1e-6)
        im = ax.imshow(ad, aspect="auto", origin="lower", cmap="inferno",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.imshow(ad, aspect="auto", origin="lower", cmap="inferno")
    if changed.any() and changed.sum() <= 20000:
        ys, xs = np.where(changed)
        ax.scatter(xs, ys, s=2, facecolors="none", edgecolors="#39ff14",
                   linewidths=0.4, label="changed")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("output feature dimension")
    ax.set_ylabel("token")
    y = r["tensors"].get("Y", {})
    ax.set_title("attention output diff  [%s | stage=%s | relL2=%.3g | changed=%s/%s]"
                 % (r.get("experiment_id") or "?", r.get("stage") or "?",
                    y.get("relative_l2_error", 0.0),
                    y.get("changed_elements", 0), y.get("elements", 0)))
    fig.colorbar(im, ax=ax, label="|Y_faulty - Y_golden|")
    fig.tight_layout()
    p = os.path.join(outdir, "output_diff.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_heads(r, outdir):
    t = r["tensors"].get("O") or r["tensors"].get("P") or r["tensors"].get("S")
    if not t or "head_error_fro" not in t:
        return None
    fro = np.asarray(t["head_error_fro"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(fro)), fro, color="#4c72b0")
    ax.set_xticks(np.arange(len(fro)))
    ax.set_xlabel("head index")
    ax.set_ylabel("Frobenius error  ||O_h(f) - O_h(g)||")
    ax.set_title("per-head error  [%s | stage=%s]"
                 % (r.get("experiment_id") or "?", r.get("stage") or "?"))
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(outdir, "head_errors.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_propagation(r, outdir):
    tens = r["tensors"]
    names, vals, labs = [], [], []
    for name in ORDER:
        if name in tens and "relative_l2_error" in tens[name]:
            names.append(name)
            vals.append(max(tens[name]["relative_l2_error"], 0.0))
            labs.append("%s %s" % (name, tuple(tens[name].get("shape", []))))
    if not names:
        return None
    vmax = max(vals)
    floor = max(vmax * 1e-9, 1e-30)
    plotted = [max(v, floor) for v in vals]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    ax.plot(x, plotted, "-o", color="#c44e52")
    for xi, v in zip(x, vals):
        if v <= 0.0:
            ax.annotate("0", (xi, floor), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, vmax * 10.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("relative L2 error (dimensionless)")
    ax.set_title("error propagation  [%s | stage=%s]"
                 % (r.get("experiment_id") or "?", r.get("stage") or "?"))
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(outdir, "propagation.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_outcomes(runs_jsonl, outdir):
    if not runs_jsonl or not os.path.exists(runs_jsonl):
        return None
    counts = {}
    with open(runs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            c = rec.get("classification") or "UNKNOWN"
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    keys = ["MASKED", "SDC", "CRASH", "NOT_INJECTED"]
    keys += [k for k in sorted(counts) if k not in keys]
    vals = [counts.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"MASKED": "#999999", "SDC": "#c44e52", "CRASH": "#8172b3",
              "NOT_INJECTED": "#dddddd"}
    ax.bar(keys, vals, color=[colors.get(k, "#4c72b0") for k in keys])
    ax.set_ylabel("runs")
    ax.set_title("classification counts")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "outcome_rates.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--runs-jsonl", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.metrics, encoding="utf-8") as f:
        r = json.load(f)
    made = [plot_diff(r, args.outdir),
            plot_heads(r, args.outdir),
            plot_propagation(r, args.outdir),
            plot_outcomes(args.runs_jsonl, args.outdir)]
    for p in made:
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
