#!/usr/bin/env python3
"""Raw SDC magnitude vs FP32 bit position (early / middle / late), 2D scatter.

Reads a campaign master.csv (one row per injection; columns: experiment_id,
input, bit, classification, max_abs_error, ...) and draws, for each accumulator
position (early/middle/late), one figure:

  x = bit 0..31
  y = per-injection difference |C' - golden|  (= max_abs_error; here exactly one
      output element differs, so this is that element's absolute deviation)
  log10 y; one scattered point per SDC sample, coloured by input distribution;
  a black median line (pooled over inputs); exponent bits (23-30) shaded and the
  sign bit (31) marked; a bottom panel with N_sdc / N_masked per bit.

Outputs SVG (git-friendly). Requires matplotlib (local analysis dependency; the
GPU box does not have it). stdlib csv only otherwise.

Usage:
  python3 scripts/plot_sdc_raw.py --master results/_tc_campaign/master.csv \
      --outdir results/_tc_campaign/figs
"""

import argparse
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as <text> (diff-friendly)
import matplotlib.pyplot as plt  # noqa: E402

# experiment id -> (label, outfile tag)
EXPERIMENTS = {
    "020_tc_input_bit_early": "early",
    "021_tc_input_bit_middle": "middle",
    "022_tc_input_bit_late": "late",
}
INPUTS = ["normal", "heavy_tail_act", "residual_stream", "swiglu_gated"]
COLORS = {
    "normal": "#1f77b4",
    "heavy_tail_act": "#d62728",
    "residual_stream": "#2ca02c",
    "swiglu_gated": "#ff7f0e",
}
BITS = list(range(32))
EXP_LO, EXP_HI = 23, 30  # FP32 exponent bit range (inclusive)


def load(master):
    recs = []
    with open(master, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            recs.append(r)
    return recs


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def collect(recs, eid):
    # per input: {bit: [y,...]}; counts per bit
    per = {inp: {b: [] for b in BITS} for inp in INPUTS}
    sdc = {b: 0 for b in BITS}
    masked = {b: 0 for b in BITS}
    crash = {b: 0 for b in BITS}
    for r in recs:
        if r.get("experiment_id") != eid:
            continue
        b = r.get("bit")
        if b in ("", None, "None"):
            continue
        b = int(float(b))
        inp = r.get("input")
        cls = r.get("classification")
        if cls == "SDC":
            sdc[b] += 1
            if inp in per:
                per[inp][b].append(_f(r.get("max_abs_error")))
        elif cls == "MASKED":
            masked[b] += 1
        elif cls == "CRASH":
            crash[b] += 1
    return per, sdc, masked, crash


def draw(tag, per, sdc, masked, crash, outpath):
    import random
    rng = random.Random(0)
    ymin, ymax = 1e-8, 1e39

    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0]})

    # shading: exponent region + sign marker
    ax.axvspan(EXP_LO - 0.5, EXP_HI + 0.5, color="#f0e6ff", zorder=0)
    ax.axvline(31, color="#999", ls=":", lw=1, zorder=0)

    for inp in INPUTS:
        xs, ys = [], []
        for b in BITS:
            for y in per[inp][b]:
                if y > 0:
                    xs.append(b + rng.uniform(-0.22, 0.22))
                    ys.append(y)
        ax.scatter(xs, ys, s=16, alpha=0.75, color=COLORS[inp],
                   edgecolors="none", label=inp)

    # pooled median per bit
    medx, medy = [], []
    for b in BITS:
        vals = [y for inp in INPUTS for y in per[inp][b] if y > 0]
        if vals:
            medx.append(b)
            medy.append(st.median(vals))
    ax.plot(medx, medy, color="black", lw=1.6, marker="o", ms=3,
            label="median (pooled)", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.7, 31.7)
    ax.set_ylabel("|C' - golden|  (per-SDC, log scale)")
    ax.set_title("BF16 HMMA FP32-accumulator SDC magnitude vs bit  "
                 "[%s]   M=N=K=1024, cuBLAS" % tag)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.text((EXP_LO + EXP_HI) / 2.0, ymax * 0.25, "exponent bits",
            ha="center", va="top", fontsize=9, color="#7a4fb0")
    ax.text(31, ymax * 0.25, "sign", ha="center", va="top", fontsize=9,
            color="#666")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # bottom: counts per bit
    width = 0.42
    axc.bar([b - width / 2 for b in BITS], [sdc[b] for b in BITS], width,
            color="#4c72b0", label="SDC")
    axc.bar([b + width / 2 for b in BITS], [masked[b] for b in BITS], width,
            color="#cccccc", label="MASKED")
    axc.set_ylabel("count")
    axc.set_xlabel("bit position (0 = LSB mantissa ... 22 = MSB mantissa, "
                   "23-30 exponent, 31 sign)")
    axc.set_xticks(BITS)
    axc.set_xticklabels([str(b) for b in BITS], fontsize=7)
    axc.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)
    print("wrote", outpath)


def write_counts(outpath, sdc, masked, crash):
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bit", "sdc", "masked", "crash"])
        for b in BITS:
            w.writerow([b, sdc[b], masked[b], crash[b]])
    print("wrote", outpath)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--master",
                   default=os.path.join("results", "_tc_campaign", "master.csv"))
    p.add_argument("--outdir",
                   default=os.path.join("results", "_tc_campaign", "figs"))
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    recs = load(args.master)
    for eid, tag in EXPERIMENTS.items():
        per, sdc, masked, crash = collect(recs, eid)
        out = os.path.join(args.outdir, "bit_diff_%s.svg" % tag)
        draw(tag, per, sdc, masked, crash, out)
        write_counts(os.path.join(args.outdir, "bit_diff_%s_counts.csv" % tag),
                     sdc, masked, crash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
