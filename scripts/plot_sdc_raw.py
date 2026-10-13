#!/usr/bin/env python3
"""Raw SDC magnitude vs FP32 bit position, per accumulator position and per
tolerance level (L0..L5).

Two modes:
  * --samples <samples.csv>  (per-injection rows produced by tolerance_sweep.py)
      -> for every position (early/middle/late) and every level L0..L5,
         scatter the SDC samples: x = bit, y = |C'-golden| (log), coloured by
         input distribution. Overlaid are two branch-median lines split by the
         per-input output scale G (G = median|d|(bit0..20) * 2^(23-bit)):
           * explode  branch : |d| >  K*G   (exponent blow-up)
           * collapse branch : |d| <= K*G   (value crushed toward zero)
         The exponent region is shaded, the sign bit marked, and a bottom panel
         shows N_sdc / N_masked.
  * (default) --master <master.csv>  (strict/L0 only, legacy single figure).

Outputs SVG (git-friendly). Requires matplotlib (local analysis dependency).

Usage:
  python3 scripts/plot_sdc_raw.py --samples results/_tc_tolerance/samples.csv \
      --outdir results/_tc_campaign/figs
"""

import argparse
import csv
import os
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402

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
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
BITS = list(range(32))
EXP_LO, EXP_HI = 23, 30


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _empty_bits():
    return {b: 0 for b in BITS}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_samples(rows, eid, level):
    per = {inp: {b: [] for b in BITS} for inp in INPUTS}
    sdc, masked, crash = _empty_bits(), _empty_bits(), _empty_bits()
    for r in rows:
        if r.get("experiment_id") != eid or r.get("level") != level:
            continue
        b = r.get("bit")
        if b in ("", None, "None"):
            continue
        b = int(float(b))
        inp = r.get("input")
        cls = r.get("classification")
        if cls == "SDC":
            sdc[b] += 1
            if inp in per and r.get("max_abs_error") not in ("", None, "None"):
                y = _f(r.get("max_abs_error"))
                if y > 0:
                    per[inp][b].append(y)
        elif cls == "MASKED":
            masked[b] += 1
        elif cls == "CRASH":
            crash[b] += 1
    return per, sdc, masked, crash


def collect_master(recs, eid):
    per = {inp: {b: [] for b in BITS} for inp in INPUTS}
    sdc, masked, crash = _empty_bits(), _empty_bits(), _empty_bits()
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
            if inp in per and _f(r.get("max_abs_error")) > 0:
                per[inp][b].append(_f(r.get("max_abs_error")))
        elif cls == "MASKED":
            masked[b] += 1
        elif cls == "CRASH":
            crash[b] += 1
    return per, sdc, masked, crash


def compute_input_scale(recs):
    """Per-input output scale G, from the current samples only.

    G_in = median over mantissa bits b=0..20 of ( median|d| at b * 2^(23-b) ).
    """
    D = {inp: {b: [] for b in BITS} for inp in INPUTS}
    for r in recs:
        if r.get("classification") != "SDC":
            continue
        b = r.get("bit")
        if b in ("", None, "None"):
            continue
        inp = r.get("input")
        if inp not in D:
            continue
        y = _f(r.get("max_abs_error"))
        if y > 0:
            D[inp][int(float(b))].append(y)
    G = {}
    for inp in INPUTS:
        vals = [st.median(D[inp][b]) * 2.0 ** (23 - b)
                for b in range(0, 21) if D[inp][b]]
        G[inp] = st.median(vals) if vals else 1.0
    return G


def draw(tag, level, per, sdc, masked, crash, G, outpath, K=10.0):
    rng = random.Random(0)
    ymin, ymax = 1e-8, 1e39
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0]})
    ax.axvspan(EXP_LO - 0.5, EXP_HI + 0.5, color="#f0e6ff", zorder=0)
    ax.axvline(31, color="#999", ls=":", lw=1, zorder=0)
    for inp in INPUTS:
        xs, ys = [], []
        for b in BITS:
            for y in per[inp][b]:
                xs.append(b + rng.uniform(-0.22, 0.22))
                ys.append(y)
        ax.scatter(xs, ys, s=16, alpha=0.75, color=COLORS[inp],
                   edgecolors="none", label=inp)
    ex = {b: [] for b in BITS}
    co = {b: [] for b in BITS}
    for inp in INPUTS:
        g = G.get(inp, 1.0)
        for b in BITS:
            for y in per[inp][b]:
                (ex if y > K * g else co)[b].append(y)
    exx = [b for b in BITS if ex[b]]
    exy = [st.median(ex[b]) for b in exx]
    cox = [b for b in BITS if co[b]]
    coy = [st.median(co[b]) for b in cox]
    ax.plot(cox, coy, color="#0b3d91", lw=1.6, ls="--", marker="s", ms=3,
            label="collapse  |d| <= %g x G" % K, zorder=6)
    ax.plot(exx, exy, color="#b30000", lw=1.9, marker="o", ms=3.5,
            label="explode  |d| > %g x G" % K, zorder=7)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.7, 31.7)
    ax.set_ylabel("|C' - golden|  (per-SDC, log scale)")
    ax.set_title("BF16 HMMA FP32-accumulator SDC magnitude vs bit  [%s, %s]"
                 "   M=N=K=1024, cuBLAS" % (tag, level))
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.text((EXP_LO + EXP_HI) / 2.0, ymax * 0.25, "exponent bits",
            ha="center", va="top", fontsize=9, color="#7a4fb0")
    ax.text(31, ymax * 0.25, "sign", ha="center", va="top", fontsize=9, color="#666")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    w = 0.42
    axc.bar([b - w / 2 for b in BITS], [sdc[b] for b in BITS], w,
            color="#4c72b0", label="SDC")
    axc.bar([b + w / 2 for b in BITS], [masked[b] for b in BITS], w,
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=None,
                   help="per-injection samples CSV from tolerance_sweep.py")
    p.add_argument("--master", default=os.path.join("results", "_tc_campaign",
                                                    "master.csv"))
    p.add_argument("--outdir", default=os.path.join("results", "_tc_campaign",
                                                    "figs"))
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    if args.samples:
        rows = load(args.samples)
        G = compute_input_scale(rows)
        print("input scale G:", {k: round(v, 3) for k, v in G.items()})
        for eid, tag in EXPERIMENTS.items():
            for level in LEVELS:
                per, sdc, masked, crash = collect_samples(rows, eid, level)
                out = os.path.join(args.outdir, "bit_diff_%s_%s.svg" % (tag, level))
                draw(tag, level, per, sdc, masked, crash, G, out)
                write_counts(os.path.join(
                    args.outdir, "bit_diff_%s_%s_counts.csv" % (tag, level)),
                    sdc, masked, crash)
    else:
        recs = load(args.master)
        G = compute_input_scale(recs)
        print("input scale G:", {k: round(v, 3) for k, v in G.items()})
        for eid, tag in EXPERIMENTS.items():
            per, sdc, masked, crash = collect_master(recs, eid)
            out = os.path.join(args.outdir, "bit_diff_%s.svg" % tag)
            draw(tag, "L0", per, sdc, masked, crash, G, out)
            write_counts(os.path.join(args.outdir,
                                      "bit_diff_%s_counts.csv" % tag),
                         sdc, masked, crash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
