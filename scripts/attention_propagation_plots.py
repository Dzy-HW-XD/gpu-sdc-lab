#!/usr/bin/env python3
"""Propagation-visualisation plots for one attention faulty/golden run pair.

Inputs are two run prefixes (as dumped by attention_forward, --dump 1).

Outputs (into --outdir):
  stage_grid.png     |diff| heatmap for every observed tensor (Q,K,V,S,P,O,Ocat,Y)
  head_token.png     per-(head, token) Frobenius error of O
  support_curve.png  relative L2 + affected-token fraction per stage

For 3-D tensors the stage grid reduces over heads with max(|diff|) and says so
in the title; the per-head detail is in head_token.png.

Usage:
  attention_propagation_plots.py --golden <prefix> --faulty <prefix> \
      --outdir plots [--title "..."]
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


def load(prefix, name):
    return np.fromfile(prefix + "." + name + ".bin", dtype=np.float32)


def shapes(meta):
    S, D, H = int(meta["S"]), int(meta["D"]), int(meta["H"])
    dh = int(meta.get("dh", D // H))
    return {"Q": (S, D), "K": (S, D), "V": (S, D),
            "S": (H, S, S), "P": (H, S, S), "O": (H, S, dh),
            "Ocat": (S, D), "Y": (S, D)}, S, D, H, dh


def grid_panel(ax, diff, name, shape):
    if diff.ndim == 3:
        red = np.max(np.abs(diff), axis=0)   # max over heads
        sub = "max over heads"
    else:
        red = np.abs(diff)
        sub = ""
    vmax = float(red.max())
    if vmax > 0:
        pos = red[red > 0]
        vmin = max(float(pos.min()), vmax * 1e-6)
        ax.imshow(red, aspect="auto", origin="lower", cmap="inferno",
                  norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        ax.set_facecolor("#eeeeee")
        ax.imshow(red, aspect="auto", origin="lower", cmap="inferno")
    ax.set_title("%s %s\nmax|d|=%.2e  %s" % (name, tuple(shape), vmax, sub),
                 fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--faulty", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.golden + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    shp, S, D, H, dh = shapes(meta)

    tensors, diffs = {}, {}
    for name in ORDER:
        gp = args.golden + "." + name + ".bin"
        fp = args.faulty + "." + name + ".bin"
        if not (os.path.exists(gp) and os.path.exists(fp)):
            continue
        g = load(args.golden, name).reshape(shp[name])
        f = load(args.faulty, name).reshape(shp[name])
        tensors[name] = (g, f)
        diffs[name] = f - g

    names = [n for n in ORDER if n in diffs]

    # ---- stage grid ---------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, name in zip(axes.ravel(), names):
        grid_panel(ax, diffs[name], name, shp[name])
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(args.title or "attention per-stage |faulty - golden|")
    fig.tight_layout()
    p = os.path.join(args.outdir, "stage_grid.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)

    # ---- head x token error (O) --------------------------------------
    if "O" in diffs and diffs["O"].ndim == 3:
        d = diffs["O"]                       # [H,S,dh]
        err = np.sqrt((d.astype(np.float64) ** 2).sum(axis=2))  # [H,S]
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(err, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("token")
        ax.set_ylabel("head")
        ax.set_title("||O_h(token) - O_h^golden(token)||_2  (per head x token)")
        fig.colorbar(im, ax=ax, label="Frobenius error over dh")
        fig.tight_layout()
        p = os.path.join(args.outdir, "head_token.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("wrote", p)

    # ---- support / energy propagation curve --------------------------
    rel, tok_frac, nz_frac = [], [], []
    for name in names:
        g, f = tensors[name]
        d = diffs[name]
        denom = float(np.sqrt((g.astype(np.float64) ** 2).sum()))
        rel.append(float(np.sqrt((d.astype(np.float64) ** 2).sum())) /
                   (denom + 1e-30))
        nz_frac.append(float((d != 0).mean()))
        red = np.abs(d)
        tok_axis = 0 if d.ndim == 2 else 1
        other = tuple(i for i in range(d.ndim) if i != tok_axis)
        any_tok = (red.max(axis=other) > 0) if other else (red > 0)
        tok_frac.append(float(any_tok.mean()))
    x = np.arange(len(names))
    floors = [max(r, 1e-12) for r in rel]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, floors, "-o", color="#c44e52", label="relative L2")
    ax.set_yscale("log")
    ax.set_ylim(1e-13, max(max(floors), 1e-6) * 1e3)
    ax.set_ylabel("relative L2 (log; 0 -> 1e-12)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s %s" % (n, shp[n]) for n in names],
                       rotation=30, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, tok_frac, "-s", color="#4c72b0", label="affected tokens")
    ax2.plot(x, nz_frac, "--^", color="#55a868", label="non-zero fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("fraction")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax.set_title(args.title or "attention error propagation")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(args.outdir, "support_curve.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
