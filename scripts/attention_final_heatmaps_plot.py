#!/usr/bin/env python3
"""Unified L4 final-Y visualisation for attention SDC experiments.

Panels (same layout / axes / colour maps for every stage, bit, position):
  top-left  : mean log10|Y_faulty - Y_golden| over 30 runs (non-zero)
  top-right : support frequency over the 30 runs, fixed 0..1 colour scale
  bottom    : cumulative impact-ratio curves for Token and Feature
  right     : key metrics (SDC count/rate, relL2, support ratio, ...)

All comparable figures use:
  * Token  : 0..S, fixed ticks, origin=lower (token 0 at the bottom)
  * Feature: 0..D, fixed ticks (final Y), 0..S for Score/Softmax
  * error  : one GLOBAL log10 colour range (no per-figure autoscaling)
  * support: fixed 0..1 colour range
  * impact : fixed 0..100% axes
  * zero / undetected -> light grey (#eeeeee)

Usage:
  attention_final_heatmaps_plot.py --data final_heatmaps_data \
      --manifest final_heatmaps_manifest_local.csv \
      --footprint footprint.csv --outdir final_heatmaps
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

BAD = "#eeeeee"
# Global fixed colour range for mean log10|Y_f - Y_g|.  The same norm is used in
# every figure; values above 1e6 saturate (the true range is printed in the
# stats panel).  This avoids per-figure autoscaling and keeps the mid-range
# 1e-3..1e6 (where most elements live) visually separated.
ERR_VMIN = -3.0
ERR_VMAX = 6.0


def cmap_bad():
    import matplotlib.colors as mcolors
    c = mcolors.ListedColormap(
        plt.get_cmap("magma")(np.linspace(0.20, 0.95, 256)))
    c.set_bad(BAD)
    return c


def load_meta(prefix, cache):
    m = cache.get(prefix)
    if m is None:
        with open(prefix + ".json", encoding="utf-8") as f:
            m = json.load(f)
        cache[prefix] = m
    return m


def rel_stats(path):
    """Mean / max relative-L2 over SDC runs per (stage, bit, position)."""
    out = collections.defaultdict(list)
    if not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row.get("classification") != "SDC":
            continue
        try:
            v = float(row["y_rel_l2"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out[(row["stage"], row["bit"], row["position"])].append(v)
    return out


def fmt_pct(x):
    return "%.1f%%" % (100.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    footprint = args.footprint or os.path.join(args.data, "footprint.csv")

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    rel = rel_stats(footprint)
    meta_cache = {}
    print("GLOBAL fixed error colour range: log10|Yf-Yg| in [%.2f, %.2f]"
          % (ERR_VMIN, ERR_VMAX))

    index = []
    for r in rows:
        tag = r["tag"]
        meta = load_meta(r["golden"], meta_cache)
        S, D = int(meta["S"]), int(meta["D"])
        try:
            meanlog = np.fromfile(
                os.path.join(args.data, "meanlog", tag + ".bin"),
                dtype=np.float32).reshape(S, D)
            support = np.fromfile(
                os.path.join(args.data, "support", tag + ".bin"),
                dtype=np.float32).reshape(S, D)
        except OSError as exc:
            print("skip %s: %s" % (tag, exc))
            continue
        support = np.clip(np.nan_to_num(support, nan=0.0, posinf=1.0,
                                        neginf=0.0), 0.0, 1.0)
        meanlog[~np.isfinite(meanlog)] = np.nan

        token_hit = support.max(axis=1) > 0
        feat_hit = support.max(axis=0) > 0
        token_cum = np.cumsum(token_hit) / float(S)
        feat_cum = np.cumsum(feat_hit) / float(D)
        token_cover = float(token_hit.mean())
        feat_cover = float(feat_hit.mean())
        supp_ratio = float((support > 0).mean())
        mean_freq = float(support.mean())

        n_runs = int(r["n_runs"])
        sdc = int(r["sdc_runs"])
        sdc_rate = sdc / float(n_runs) if n_runs else 0.0
        fin_ml = meanlog[np.isfinite(meanlog)]
        ml_lo = float(fin_ml.min()) if fin_ml.size else 0.0
        ml_hi = float(fin_ml.max()) if fin_ml.size else 0.0
        key = (r["stage"], r["bit"], r["position"])
        vals = rel.get(key, [])
        rel_mean = float(np.mean(vals)) if vals else None
        rel_max = float(np.max(vals)) if vals else None

        fig = plt.figure(figsize=(17, 9.5))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.82],
                              height_ratios=[2.5, 1.0],
                              hspace=0.30, wspace=0.28)
        ax_err = fig.add_subplot(gs[0, 0])
        ax_sup = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[:, 2])
        ax_cur = fig.add_subplot(gs[1, 0:2])

        xt = [0, D // 4, D // 2, 3 * D // 4, D - 1]
        yt = [0, S // 4, S // 2, 3 * S // 4, S - 1]

        m = np.ma.masked_where(~np.isfinite(meanlog), meanlog)
        im1 = ax_err.imshow(m, extent=(0, D, 0, S), origin="lower",
                            aspect="auto", cmap=cmap_bad(),
                            norm=Normalize(vmin=ERR_VMIN, vmax=ERR_VMAX),
                            interpolation="nearest")
        fig.colorbar(im1, ax=ax_err, fraction=0.046, pad=0.02,
                     label="mean log10|dY|")
        ax_err.set_title("mean log10|Y_f - Y_g|  (30 runs, non-zero; "
                         "global [%.0f, %.0f], >%.0f clipped)"
                         % (ERR_VMIN, ERR_VMAX, ERR_VMAX), fontsize=10)
        ax_err.set_xlabel("Feature")
        ax_err.set_ylabel("Token")
        ax_err.set_xticks(xt)
        ax_err.set_yticks(yt)
        ax_err.set_xlim(0, D)
        ax_err.set_ylim(0, S)

        sm = np.ma.masked_where(support <= 0, support)
        im2 = ax_sup.imshow(sm, extent=(0, D, 0, S), origin="lower",
                            aspect="auto", cmap=cmap_bad(),
                            norm=Normalize(vmin=0.0, vmax=1.0),
                            interpolation="nearest")
        fig.colorbar(im2, ax=ax_sup, fraction=0.046, pad=0.02,
                     label="support (0-1)")
        ax_sup.set_title("support frequency over %d runs  (fixed 0-1)"
                         % n_runs, fontsize=10)
        ax_sup.set_xlabel("Feature")
        ax_sup.set_ylabel("Token")
        ax_sup.set_xticks(xt)
        ax_sup.set_yticks(yt)
        ax_sup.set_xlim(0, D)
        ax_sup.set_ylim(0, S)

        x_tok = np.linspace(0.0, 1.0, S)
        x_feat = np.linspace(0.0, 1.0, D)
        ax_cur.plot(x_tok, token_cum, color="#1f77b4", lw=1.6,
                    label="Token: coverage %d/%d (%s)"
                          % (int(token_hit.sum()), S, fmt_pct(token_cover)))
        ax_cur.plot(x_feat, feat_cum, color="#d62728", lw=1.6,
                    label="Feature: coverage %d/%d (%s)"
                          % (int(feat_hit.sum()), D, fmt_pct(feat_cover)))
        ax_cur.set_title("cumulative impact ratio  "
                         "(affected Tokens / Features)", fontsize=10)
        ax_cur.set_xlabel("index percentile")
        ax_cur.set_ylabel("cumulative affected fraction")
        ax_cur.set_xlim(0, 1)
        ax_cur.set_ylim(0, 1)
        ax_cur.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax_cur.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_cur.set_xticks(np.linspace(0, 1, 5))
        ax_cur.set_yticks(np.linspace(0, 1, 5))
        ax_cur.grid(alpha=0.3)
        ax_cur.legend(loc="lower right", fontsize=9, framealpha=0.9)

        ax_stats.axis("off")
        rel_txt = ("--" if rel_mean is None
                   else "mean %.3e  max %.3e" % (rel_mean, rel_max))
        txt = (
            "stage     : %s\n"
            "head      : %s\n"
            "position  : %s\n"
            "bit       : %s\n"
            "seq       : %s\n"
            "\n"
            "SDC count : %d / %d\n"
            "SDC Rate  : %s\n"
            "Rel. L2   : %s\n"
            "Support   : %s of coords\n"
            "mean freq : %.4f\n"
            "\n"
            "Tokens    : %d / %d (%s)\n"
            "Features  : %d / %d (%s)\n"
            "log10|dY| : %.2f .. %.2f\n"
            "\n"
            "GLOBAL error colour scale\n"
            "  log10 range [%.2f, %.2f]\n"
            "  (same in every figure)"
            % (r["stage"], r.get("head") or "-", r["position"], r["bit"],
               r["seq"], sdc, n_runs, fmt_pct(sdc_rate), rel_txt,
               fmt_pct(supp_ratio), mean_freq, int(token_hit.sum()), S,
               fmt_pct(token_cover), int(feat_hit.sum()), D,
               fmt_pct(feat_cover), ml_lo, ml_hi, ERR_VMIN, ERR_VMAX)
        )
        ax_stats.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9.5,
                      family="monospace", linespacing=1.45)

        fig.suptitle(
            "L4 final-Y | %s  h%s  %s  bit%s  seq%s | "
            "SDC %d/%d (%s) | relL2 %s | Support %s"
            % (r["stage"], r.get("head") or "-", r["position"], r["bit"],
               r["seq"], sdc, n_runs, fmt_pct(sdc_rate),
               ("n/a" if rel_mean is None else "%.2e" % rel_mean),
               fmt_pct(supp_ratio)), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(args.outdir, tag + ".png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        index.append({"tag": tag, "path": path, "stage": r["stage"],
                      "head": r.get("head", ""), "position": r["position"],
                      "bit": r["bit"], "seq": r["seq"],
                      "sdc": sdc, "n_runs": n_runs,
                      "support_ratio": supp_ratio})
        print("wrote", path)

    with open(os.path.join(args.outdir, "final_heatmaps_index.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        for r in index:
            w.writerow(r)
    print("wrote index (%d figures)" % len(index))


if __name__ == "__main__":
    main()
