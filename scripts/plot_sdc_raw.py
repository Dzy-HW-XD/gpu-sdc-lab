#!/usr/bin/env python3
"""Raw SDC magnitude vs FP32 bit position, per accumulator position and per
tolerance level (L0..L5).

Two modes:
  * --samples <samples.csv>  (per-injection rows produced by tolerance_sweep.py)
      -> for every position (early/middle/late) and every level L0..L5,
         scatter the SDC samples: x = bit, y = |C'-golden| (log), coloured by
         input distribution (sign bit excluded, plot truncated at bit 31).
         Overlaid are two pooled branch lines split at the per-bit geometric
         mean of |d| (pooled over inputs), smoothed with a LOWESS fit in log
         space:
           * explode  line : max |d| of the above-mean samples (exponent blow-up)
           * collapse line : min |d| of the below-mean samples (crushed toward 0)
         The vertical gap between the two lines is the exposure. The exponent
         region is shaded and a bottom panel shows N_sdc / N_masked.
  * (default) --master <master.csv>  (strict/L0 only, legacy single figure).

Outputs SVG (git-friendly). Requires matplotlib (local analysis dependency).

Usage:
  python3 scripts/plot_sdc_raw.py --samples results/_tc_tolerance/samples.csv \
      --outdir results/_tc_campaign/figs
"""

import argparse
import csv
import math
import os
import random

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
BITS = list(range(31))  # bits 0..30 (mantissa + exponent); sign bit 31 excluded
EXP_LO, EXP_HI = 23, 30
SMOOTH_FRAC = 0.25


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _smooth_log(xs, yy, frac=SMOOTH_FRAC):
    """LOWESS (local linear regression) smoother in log10 space.

    Follows the trend and extrapolates the local slope at the edges, so the
    exponential blow-up at high bits is preserved while per-bit max/min noise
    is smoothed out.
    """
    n = len(yy)
    if n < 3:
        return yy
    k = max(3, min(n, int(math.ceil(frac * n))))
    lg = [math.log10(max(y, 1e-300)) for y in yy]
    out = []
    for i in range(n):
        d = [abs(xs[j] - xs[i]) for j in range(n)]
        idx = sorted(range(n), key=lambda j: d[j])[:k]
        rmax = max(d[j] for j in idx)
        if rmax <= 0:
            out.append(lg[i])
            continue
        sw = sx = sy = sxx = sxy = 0.0
        for j in idx:
            w = (1.0 - (d[j] / rmax) ** 3) ** 3
            sw += w
            sx += w * xs[j]
            sy += w * lg[j]
            sxx += w * xs[j] * xs[j]
            sxy += w * xs[j] * lg[j]
        den = sw * sxx - sx * sx
        if abs(den) < 1e-12:
            out.append(sy / sw)
        else:
            b = (sw * sxy - sx * sy) / den
            a = (sy - b * sx) / sw
            out.append(a + b * xs[i])
    return [10.0 ** v for v in out]


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
        if b not in BITS:
            continue
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
        if b not in BITS:
            continue
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


def draw(tag, level, per, sdc, masked, crash, outpath):
    rng = random.Random(0)
    ymin, ymax = 1e-8, 1e39
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0]})
    ax.axvspan(EXP_LO - 0.5, EXP_HI + 0.5, color="#f0e6ff", zorder=0)
    for inp in INPUTS:
        xs, ys = [], []
        for b in BITS:
            for y in per[inp][b]:
                xs.append(b + rng.uniform(-0.22, 0.22))
                ys.append(y)
        ax.scatter(xs, ys, s=16, alpha=0.75, color=COLORS[inp],
                   edgecolors="none", label=inp)
    exx, exy, cox, coy = [], [], [], []
    for b in BITS:
        ys = [y for inp in INPUTS for y in per[inp][b]]
        if not ys:
            continue
        split = 10.0 ** (sum(math.log10(y) for y in ys) / len(ys))
        up = [y for y in ys if y > split]
        lo = [y for y in ys if y <= split]
        exx.append(b)
        exy.append(max(up) if up else max(ys))
        cox.append(b)
        coy.append(min(lo) if lo else min(ys))
    exy = _smooth_log(exx, exy)
    coy = _smooth_log(cox, coy)
    ax.plot(cox, coy, color="#0b3d91", lw=1.6, ls="--", marker="s", ms=3,
            label="collapse  min |d| (below geo-mean)", zorder=6)
    ax.plot(exx, exy, color="#b30000", lw=1.9, marker="o", ms=3.5,
            label="explode  max |d| (above geo-mean)", zorder=7)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.7, 30.7)
    ax.set_ylabel("|C' - golden|  (per-SDC, log scale)")
    ax.set_title("BF16 HMMA FP32-accumulator SDC magnitude vs bit  [%s, %s]"
                 "   M=N=K=1024, cuBLAS" % (tag, level))
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.text((EXP_LO + EXP_HI) / 2.0, ymax * 0.25, "exponent bits",
            ha="center", va="top", fontsize=9, color="#7a4fb0")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    w = 0.42
    axc.bar([b - w / 2 for b in BITS], [sdc[b] for b in BITS], w,
            color="#4c72b0", label="SDC")
    axc.bar([b + w / 2 for b in BITS], [masked[b] for b in BITS], w,
            color="#cccccc", label="MASKED")
    axc.set_ylabel("count")
    axc.set_xlabel("bit position (0 = LSB mantissa ... 22 = MSB mantissa, "
                   "23-30 exponent; sign bit truncated)")
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
        for eid, tag in EXPERIMENTS.items():
            for level in LEVELS:
                per, sdc, masked, crash = collect_samples(rows, eid, level)
                out = os.path.join(args.outdir, "bit_diff_%s_%s.svg" % (tag, level))
                draw(tag, level, per, sdc, masked, crash, out)
                write_counts(os.path.join(
                    args.outdir, "bit_diff_%s_%s_counts.csv" % (tag, level)),
                    sdc, masked, crash)
    else:
        recs = load(args.master)
        for eid, tag in EXPERIMENTS.items():
            per, sdc, masked, crash = collect_master(recs, eid)
            out = os.path.join(args.outdir, "bit_diff_%s.svg" % tag)
            draw(tag, "L0", per, sdc, masked, crash, out)
            write_counts(os.path.join(args.outdir,
                                      "bit_diff_%s_counts.csv" % tag),
                         sdc, masked, crash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
