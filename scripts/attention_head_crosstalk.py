#!/usr/bin/env python3
"""Head crosstalk: where does an injected head's error appear?

Reads attention_runs.jsonl (from attention_collect.py) and builds, for each
injected head (and the projection stage), a row over all H attention heads:

  left  : mean ||O_h(f) - O_h(g)||_F over SDC runs  (log colour)
  right : fraction of runs with any error in head h (0..1)

A fault that stays head-local gives a single bright column; spread across
heads shows crosstalk through the shared softmax / output projection.

Usage:
  attention_head_crosstalk.py --runs-jsonl runs.jsonl --out crosstalk.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def inj_key(r):
    stage = r.get("stage") or ""
    if stage.startswith("qkv") or stage == "wo":
        return stage
    h = r.get("head")
    return "h%s" % h


def sort_key(k):
    return (0, k) if (k.startswith("qkv") or k == "wo") else (1, int(k[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs = []
    with open(args.runs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    H = 0
    for r in runs:
        if r.get("o_head_fro"):
            H = len(r["o_head_fro"])
            break
    if not H:
        raise SystemExit("no o_head_fro in runs (need attention_collect output)")

    injected = []
    for r in runs:
        if not r.get("o_head_fro"):
            continue
        key = inj_key(r)
        if key not in injected:
            injected.append(key)
    injected.sort(key=sort_key)

    mean = np.zeros((len(injected), H))
    nsdc = np.zeros((len(injected), H))
    frac = np.zeros((len(injected), H))
    nall = np.zeros((len(injected), H))

    for r in runs:
        fro = r.get("o_head_fro")
        if not fro:
            continue
        key = inj_key(r)
        i = injected.index(key)
        f = np.asarray(fro, dtype=float)
        if r.get("classification") in ("MASKED", "SDC"):
            nall[i] += 1
            frac[i] += (f > 0)
            if r.get("classification") == "SDC":
                mean[i] += f
                nsdc[i] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_mat = np.where(nsdc > 0, mean / np.maximum(nsdc, 1), np.nan)
        frac_mat = np.where(nall > 0, frac / np.maximum(nall, 1), np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 1.1 * len(injected) + 3),
                                   gridspec_kw={"width_ratios": [1, 1]})
    finite = mean_mat[np.isfinite(mean_mat) & (mean_mat > 0)]
    if finite.size:
        vmax = finite.max()
        vmin = finite.min()
        if vmin == vmax:
            vmin = vmax * 0.5
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("#eeeeee")
        plotm = np.where(np.isfinite(mean_mat) & (mean_mat > 0), mean_mat, np.nan)
        im1 = ax1.imshow(np.ma.masked_invalid(plotm), aspect="auto",
                         cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(im1, ax=ax1, label="mean Frobenius error (SDC runs)")
        for i in range(len(injected)):
            for j in range(H):
                if np.isfinite(mean_mat[i, j]) and mean_mat[i, j] > 0:
                    ax1.text(j, i, "%.1e" % mean_mat[i, j], ha="center",
                             va="center", fontsize=6, color="white")
    ax1.set_facecolor("#eeeeee")
    ax1.set_title("mean per-head error (SDC runs)")
    ax1.set_yticks(range(len(injected)))
    ax1.set_yticklabels(injected)
    ax1.set_xlabel("affected head")
    ax1.set_ylabel("injected")

    im2 = ax2.imshow(np.ma.masked_invalid(frac_mat), aspect="auto",
                     cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im2, ax=ax2, label="fraction of runs with error")
    ax2.set_facecolor("#eeeeee")
    ax2.set_title("fraction of runs with any error in that head")
    ax2.set_yticks(range(len(injected)))
    ax2.set_yticklabels(injected)
    ax2.set_xlabel("affected head")

    for i, key in enumerate(injected):
        if key.startswith("qkv") or key == "wo":
            continue
        h = int(key[1:])
        if 0 <= h < H:
            for ax in (ax1, ax2):
                ax.add_patch(Rectangle((h - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#00ff88", lw=1.5))

    fig.suptitle(args.title or "attention head crosstalk")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("wrote", args.out)

    # textual self/cross summary
    for i, key in enumerate(injected):
        row = mean_mat[i]
        if not np.isfinite(row).any():
            continue
        if key.startswith("qkv") or key == "wo":
            print("  %s: top heads" % key,
                  np.argsort(np.nan_to_num(row))[-5:][::-1].tolist())
        else:
            h = int(key[1:])
            self_e = row[h] if h < H else np.nan
            others = np.delete(row, h) if h < H else row
            print("  injected %-3s self=%.3g  max_other=%.3g  mean_other=%.3g"
                  % (key, self_e, np.nanmax(others), np.nanmean(others)))


if __name__ == "__main__":
    main()
