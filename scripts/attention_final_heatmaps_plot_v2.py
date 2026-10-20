#!/usr/bin/env python3
"""Publication-quality final-Y heatmaps for attention SDC experiments (v2).

Design follows the "paper-figure" skill checklist:
  * single/double-column physical size, serif font matching paper body text
  * no in-figure title, panel labels (a)/(b), colourblind-safe palettes
  * one global, fixed colour scale per quantity (comparable across figures)
  * vector PDF output (heatmap rasterised at 300 dpi) + 600 dpi PNG fallback
  * metrics live in the file name / CSV index / LaTeX caption, not in a
    dashboard text panel

Panels:
  (a) support: number of runs (out of n) in which the element is L4-affected
  (b) mean log10|dY| over the runs in which that element is affected;
      elements never affected are masked grey.  (The old v1 panel averaged
      over *all non-zero* changes, including sub-threshold FP noise, which
      made e.g. qkv_k look uniformly red.)

Usage:
  python scripts/attention_final_heatmaps_plot_v2.py \
      --data results/047_attn_5gemm_full/final_heatmaps_data \
      --outdir results/047_attn_5gemm_full/final_heatmaps_v2 \
      --formats pdf,png
  # only a few cells:
  ... --tags final_qkv_k_hx_early_bit29,final_pv_h0_early_bit27
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

S_DEFAULT, D_DEFAULT = 128, 2048
STAGE_LABELS = {"qkv_q": "QKV-Q", "qkv_k": "QKV-K", "qkv_v": "QKV-V",
                "qk": "QK", "pv": "PV"}

# Global fixed scales (same in every figure of the campaign).
SUP_VMIN, SUP_VMAX = 0, 30
SUP_GAMMA = 0.5          # sqrt-like stretch: 1/30 stays visible
ERR_VMIN, ERR_VMAX = 0.0, 3.0
BAD = "#e9e9e9"
INK = "0.15"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def sup_cmap():
    c = plt.get_cmap("Blues").copy()
    c.set_bad(BAD)
    return c


def err_cmap():
    c = plt.get_cmap("YlOrRd").copy()
    c.set_bad(BAD)
    return c


def load_bin(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != shape[0] * shape[1]:
        raise ValueError("%s: %d values, expected %d" % (path, a.size, shape[0] * shape[1]))
    return a.reshape(shape)


def panel_label(ax, text):
    ax.text(0.0, 1.035, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def style_axis(ax, S, D):
    ax.set_xlim(0, D)
    ax.set_ylim(0, S)
    ax.xaxis.set_major_locator(plt.MultipleLocator(512))
    ax.yaxis.set_major_locator(plt.MultipleLocator(32))
    ax.set_xlabel(r"Feature index $j$")
    ax.set_ylabel(r"Token index $i$")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.35")


def add_cbar(fig, im, ax, **kw):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="2.6%", pad=0.035)
    cb = fig.colorbar(im, cax=cax, **kw)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("0.35")
    cb.ax.tick_params(width=0.4, length=2, labelsize=6.5)
    return cb


def draw_cell(rec, data_dir, outdir, formats, dpi, annotate, S, D):
    tag = rec["tag"]
    shape = (S, D)
    count_path = os.path.join(data_dir, "count", tag + ".bin")
    if os.path.exists(count_path):
        counts = np.fromfile(count_path, dtype=np.uint8).reshape(shape).astype(np.int16)
    else:
        sup = load_bin(os.path.join(data_dir, "support", tag + ".bin"), shape)
        counts = np.rint(np.clip(sup, 0.0, 1.0) * int(rec["n_runs"])).astype(np.int16)
    meanlog = load_bin(os.path.join(data_dir, "meanlog", tag + ".bin"), shape)
    nf_path = os.path.join(data_dir, "nfcount", tag + ".bin")
    nfcount = (np.fromfile(nf_path, dtype=np.uint8).reshape(shape)
               if os.path.exists(nf_path) else None)

    n = int(rec["n_runs"])
    counts = np.clip(counts, 0, SUP_VMAX)
    meanlog = np.where(counts > 0, meanlog, np.nan)
    meanlog[~np.isfinite(meanlog)] = np.nan

    fig = plt.figure(figsize=(7.2, 2.85 if annotate else 2.65))
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.925,
                          bottom=0.22 if annotate else 0.17,
                          top=0.90, wspace=0.30)
    ax_sup = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])

    im_sup = ax_sup.imshow(
        counts, extent=(0, D, 0, S), origin="lower", aspect="auto",
        cmap=sup_cmap(), norm=PowerNorm(gamma=SUP_GAMMA,
                                        vmin=SUP_VMIN, vmax=SUP_VMAX),
        interpolation="nearest", rasterized=True)
    style_axis(ax_sup, S, D)
    panel_label(ax_sup, "(a)")
    if counts.sum() == 0:
        ax_sup.text(0.5, 0.5, "no L4-affected\nelements", transform=ax_sup.transAxes,
                    ha="center", va="center", fontsize=7, color="0.45")
    cb1 = add_cbar(fig, im_sup, ax_sup, ticks=[0, 1, 5, 10, 20, 30])
    cb1.set_label("Affected runs (of %d)" % n, fontsize=7, labelpad=3)

    im_err = ax_err.imshow(
        np.ma.masked_invalid(meanlog), extent=(0, D, 0, S), origin="lower",
        aspect="auto", cmap=err_cmap(),
        norm=Normalize(vmin=ERR_VMIN, vmax=ERR_VMAX),
        interpolation="nearest", rasterized=True)
    if nfcount is not None and nfcount.any():
        overlay = np.ma.masked_where(nfcount == 0,
                                     np.ones(shape, dtype=np.float32))
        ax_err.imshow(overlay, extent=(0, D, 0, S), origin="lower",
                      aspect="auto", cmap=ListedColormap(["#2b2b2b"]),
                      vmin=0.0, vmax=1.0, interpolation="nearest",
                      rasterized=True)
        nf_frac = float((nfcount > 0).mean())
        nf_runs = int(rec.get("nf_runs") or 0)
        if nf_frac > 0.5:
            note = ("all output elements non-finite\nin %d/%d runs" % (nf_runs, n)
                    if nf_frac > 0.99 else
                    "%.0f%% of elements non-finite\nin %d/%d runs"
                    % (100.0 * nf_frac, nf_runs, n))
            ax_err.text(0.5, 0.5, note, transform=ax_err.transAxes,
                        ha="center", va="center", fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.55,
                                  edgecolor="none", pad=2.5))
    style_axis(ax_err, S, D)
    panel_label(ax_err, "(b)")
    cb2 = add_cbar(fig, im_err, ax_err, ticks=[0, 1, 2, 3])
    cb2.set_label(r"$\log_{10}|\Delta Y|$", fontsize=7, labelpad=3)

    if annotate:
        stage = rec["stage"]
        head_note = "  ·  random head per run" if stage in ("qk", "pv") else ""
        nf_runs = int(rec.get("nf_runs") or 0)
        nf_note = ("  ·  black = non-finite (NF)" if nf_runs else "")
        sdc = int(rec["sdc_runs"])
        meta = ("%s  ·  %s  ·  bit %s  ·  n = %d runs  ·  SDC %d/%d (%.1f%%)  ·  "
                "NF %d/%d  ·  grey = never L4-affected  ·  colour clipped at $10^3$%s%s"
                % (STAGE_LABELS.get(stage, stage), rec["position"], rec["bit"],
                   n, sdc, n, 100.0 * sdc / n if n else 0.0,
                   nf_runs, n, nf_note, head_note))
        fig.text(0.5, 0.025, meta, ha="center", va="bottom", fontsize=6.3,
                 color="0.35")

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, tag)
    written = []
    for fmt in formats:
        path = base + "." + fmt
        if fmt == "pdf":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tags", default=None,
                    help="comma-separated tags; default: all cells")
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-annotate", action="store_true",
                    help="omit the small metadata footer (paper/caption mode)")
    ap.add_argument("--S", type=int, default=S_DEFAULT)
    ap.add_argument("--D", type=int, default=D_DEFAULT)
    args = ap.parse_args()

    setup_style()
    manifest = args.manifest or os.path.join(args.data, "final_heatmaps_manifest.csv")
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        rows = [r for r in rows if r["tag"] in wanted]
        missing = wanted - {r["tag"] for r in rows}
        if missing:
            raise SystemExit("unknown tags: %s" % ", ".join(sorted(missing)))
    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    index = []
    for i, rec in enumerate(rows, 1):
        paths = draw_cell(rec, args.data, args.outdir, formats, args.dpi,
                          not args.no_annotate, args.S, args.D)
        index.append(dict(tag=rec["tag"], stage=rec["stage"], head=rec["head"],
                          position=rec["position"], bit=rec["bit"], seq=rec["seq"],
                          n_runs=rec["n_runs"], sdc_runs=rec["sdc_runs"],
                          files=";".join(os.path.basename(p) for p in paths)))
        print("[%d/%d] %s" % (i, len(rows), rec["tag"]), flush=True)

    idx_path = os.path.join(args.outdir, "index_v2_selected.csv" if args.tags else "index_v2.csv")
    if index:
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
            w.writeheader()
            w.writerows(index)
    print("wrote %d figures -> %s" % (len(index), args.outdir))


if __name__ == "__main__":
    main()
