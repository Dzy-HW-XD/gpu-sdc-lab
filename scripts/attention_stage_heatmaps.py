#!/usr/bin/env python3
"""Score / Softmax / PV |diff| heatmaps for representative runs.

Unified visualisation rules (same for every stage / bit / position):
  * one GLOBAL log10 colour range per tensor (S, P, O), computed over all runs
  * fixed Token / Feature axes and ticks (origin=lower, token 0 at bottom)
  * magma, zero difference -> light grey (#eeeeee)
  * no golden / faulty panels, no per-figure autoscaling

Layout: 1x3  |diff Score|  |diff Softmax|  |diff PV O|.

Usage:
  attention_stage_heatmaps.py --manifest propagation_manifest_local.csv \
      --outdir stage_heatmaps
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BAD = "#eeeeee"


def cmap_bad():
    import matplotlib.colors as mcolors
    c = mcolors.ListedColormap(
        plt.get_cmap("magma")(np.linspace(0.20, 0.95, 256)))
    c.set_bad(BAD)
    return c


def load(prefix, name, shape):
    a = np.fromfile(prefix + "." + name + ".bin", dtype=np.float32)
    return a.reshape(shape)


def stage_diffs(r, meta_cache):
    g, fp = r["golden"], r["faulty"]
    meta = meta_cache.get(g)
    if meta is None:
        with open(g + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        meta_cache[g] = meta
    S, D, H = int(meta["S"]), int(meta["D"]), int(meta["H"])
    dh = int(meta.get("dh", D // H))
    Sg = load(g, "S", (H, S, S))
    Sf = load(fp, "S", (H, S, S))
    Pg = load(g, "P", (H, S, S))
    Pf = load(fp, "P", (H, S, S))
    Og = load(g, "O", (H, S, dh))
    Of = load(fp, "O", (H, S, dh))
    head = r.get("head")
    if head not in ("", None):
        h = int(float(head))
    else:
        h = int(np.argmax(np.abs(Of - Og).max(axis=(1, 2))))
    return Sf[h] - Sg[h], Pf[h] - Pg[h], Of[h] - Og[h], h, S, dh


def finite_abs(arr):
    aa = np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    return aa


def global_log_scales(rows, meta_cache):
    vals = {"S": [], "P": [], "O": []}
    for r in rows:
        ds, dp, do, _, _, _ = stage_diffs(r, meta_cache)
        for key, arr in (("S", ds), ("P", dp), ("O", do)):
            aa = finite_abs(arr)
            nz = aa[aa > 0]
            if nz.size:
                vals[key].append(np.log10(nz))
    scales = {}
    for key, chunks in vals.items():
        if not chunks:
            scales[key] = (-3.0, 1.0)
            continue
        v = np.concatenate(chunks)
        lo = float(np.percentile(v, 1))
        hi = float(np.percentile(v, 99))
        if hi <= lo:
            hi = lo + 1.0
        scales[key] = (lo, hi)
    return scales


def panel(ax, arr, title, scale, extent, xticks, yticks, xlabel, ylabel):
    aa = finite_abs(arr)
    nz = aa[aa > 0]
    if nz.size == 0:
        ax.set_facecolor(BAD)
        ax.set_title(title + "  (no change)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    lo, hi = scale
    m = np.ma.masked_where(aa <= 0, aa)
    im = ax.imshow(m, extent=extent, origin="lower", aspect="auto",
                   cmap=cmap_bad(), norm=LogNorm(vmin=10.0 ** lo,
                                                 vmax=10.0 ** hi),
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    meta_cache = {}
    scales = global_log_scales(rows, meta_cache)
    print("GLOBAL log10 scales (1-99 pct): "
          + "  ".join("%s [%.2f, %.2f]" % (k, v[0], v[1])
                      for k, v in scales.items()))

    index = []
    for r in rows:
        ds, dp, do, h, S, dh = stage_diffs(r, meta_cache)
        q = [0, S // 4, S // 2, 3 * S // 4, S - 1]
        f = [0, dh // 4, dh // 2, 3 * dh // 4, dh - 1]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        im0 = panel(axes[0], ds, "|diff| Score S_h", scales["S"],
                    (0, S, 0, S), q, q, "Key", "Query")
        im1 = panel(axes[1], dp, "|diff| Softmax P_h", scales["P"],
                    (0, S, 0, S), q, q, "Key", "Query")
        im2 = panel(axes[2], do, "|diff| PV O_h", scales["O"],
                    (0, dh, 0, S), f, q, "Feature (d_h)", "Token")
        for ax, im in zip(axes, (im0, im1, im2)):
            if im is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        try:
            yrel = "%.3e" % float(r["y_rel_l2"])
        except (TypeError, ValueError):
            yrel = r.get("y_rel_l2", "")
        fig.suptitle("%s  head=%d  %s  bit%s  seq%s   Y relL2=%s   "
                     "(global per-tensor log scale)"
                     % (r["stage"], h, r["position"], r["bit"], r["seq"],
                        yrel), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        tag = "heat_%s_h%d_%s_bit%s_seq%s" % (
            r["stage"], h, r["position"], r["bit"], r["seq"])
        path = os.path.join(args.outdir, tag + ".png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        index.append({"tag": tag, "path": path, "stage": r["stage"],
                      "head": h, "position": r["position"], "bit": r["bit"],
                      "seq": r["seq"], "y_rel_l2": r.get("y_rel_l2", "")})
        print("wrote", path)

    with open(os.path.join(args.outdir, "stage_heatmaps_index.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        for r in index:
            w.writerow(r)
    print("wrote index (%d runs)" % len(index))


if __name__ == "__main__":
    main()
