#!/usr/bin/env python3
"""Analyze layer14 qkv_v full-stack C0..C3 campaign."""
import csv
import json
import math
import os
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "results/qkv_v_full"
CPS = ["C0", "C1", "C2", "C3"]
CP_LABEL = {"C0": "GEMM V", "C1": "attention", "C2": "layer14 block", "C3": "layer27 block"}
BITS = ["28", "29", "30"]


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, "runs_all.csv"), encoding="utf-8")))
    clean = json.load(open(os.path.join(BASE, "clean_controls", "summary.json"), encoding="utf-8"))
    sham = list(csv.DictReader(open(os.path.join(BASE, "sham.csv"), encoding="utf-8")))

    # outcome tables
    outcome = {}
    for cp in CPS:
        outcome[cp] = {
            "d0": dict(Counter(r["outcome_d0_" + cp] for r in rows)),
            "d1": dict(Counter(r["outcome_d1_" + cp] for r in rows)),
        }
    by_bit = []
    for bit in BITS:
        rs = [r for r in rows if r["bit"] == bit]
        d = {"bit": bit, "n": len(rs)}
        for cp in CPS:
            for mode in ("d0", "d1"):
                for k in ("SDC", "Masked", "DUE", "Crash"):
                    d["%s_%s_%s" % (mode, cp, k)] = sum(
                        r["outcome_%s_%s" % (mode, cp)] == k for r in rs)
        by_bit.append(d)

    # numeric stats per checkpoint (D0 SDC, finite)
    numeric = {}
    for cp in CPS:
        maxs, rels, mism = [], [], []
        huge = 0
        for r in rows:
            if r["run_d0"] != "SDC":
                continue
            v = f(r[cp + "_max_abs"])
            q = f(r[cp + "_rel_l2"])
            m = f(r[cp + "_mismatch"])
            if math.isfinite(v):
                maxs.append(v)
                huge += v > 1e10
            if math.isfinite(q):
                rels.append(q)
            if math.isfinite(m):
                mism.append(m)
        numeric[cp] = {
            "n": len(mism),
            "mismatch_median": statistics.median(mism) if mism else None,
            "mismatch_min": min(mism) if mism else None,
            "mismatch_max": max(mism) if mism else None,
            "maxabs_median": statistics.median(maxs) if maxs else None,
            "maxabs_p90": float(np.percentile(maxs, 90)) if maxs else None,
            "maxabs_max": max(maxs) if maxs else None,
            "maxabs_gt_1e10": huge,
            "rel_l2_median": statistics.median(rels) if rels else None,
            "rel_l2_max": max(rels) if rels else None,
        }

    summary = {
        "n_valid": len(rows),
        "n_primary_sites": sum(r["reserve"] == "0" for r in rows),
        "crash": sum(r["_crash"] == "True" for r in rows),
        "invalid": sum(r["injected"] != "1" for r in rows),
        "nan_inf_rows": sum(r["nan_inf_any"] == "1" for r in rows),
        "clean_controls": len(clean),
        "clean_ok": sum(1 for r in clean if r.get("final_bitwise_equal")),
        "sham_controls": len(sham),
        "sham_masked": sum(r["run_d0"] == "Masked" for r in sham),
        "sham_injected": sum(r["injected"] == "1" for r in sham),
        "outcome": outcome,
        "numeric": numeric,
        "by_bit": by_bit,
    }
    json.dump(summary, open(os.path.join(BASE, "summary.json"), "w", encoding="utf-8"),
              indent=2)
    with open(os.path.join(BASE, "summary_by_bit.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by_bit[0].keys()))
        w.writeheader(); w.writerows(by_bit)
    with open(os.path.join(BASE, "summary_numeric.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["checkpoint"] + list(numeric["C0"].keys()))
        w.writeheader()
        for cp in CPS:
            d = {"checkpoint": cp}; d.update(numeric[cp]); w.writerow(d)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.86, bottom=0.16,
                          wspace=0.34, hspace=0.62)
    # (a) D1 outcomes
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CPS))
    colors = {"SDC": "#C44E52", "DUE": "#DD8452", "Masked": "#4C72B0", "Crash": "#8172B2"}
    bottom = np.zeros(len(CPS))
    for k in ("SDC", "DUE", "Masked", "Crash"):
        vals = np.array([outcome[cp]["d1"].get(k, 0) for cp in CPS])
        ax.bar(x, vals, bottom=bottom, color=colors[k], label=k)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([CP_LABEL[c] for c in CPS], rotation=15)
    ax.set_ylabel("count"); ax.set_ylim(0, 190)
    ax.set_title("(a)  D1 outcome by checkpoint", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")

    # (b) mismatch count
    ax = fig.add_subplot(gs[0, 1])
    data = []
    for cp in CPS:
        vals = [max(1.0, f(r[cp + "_mismatch"])) for r in rows
                if r["run_d0"] == "SDC" and math.isfinite(f(r[cp + "_mismatch"]))]
        data.append(vals)
    ax.boxplot(data, tick_labels=["C0", "C1", "C2", "C3"], showfliers=False,
               widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#B8D8E8", edgecolor="0.3"),
               medianprops=dict(color="#C44E52"))
    ax.set_yscale("log")
    ax.set_ylabel("mismatch elements")
    ax.set_title("(b)  spatial spread", loc="left")
    ax.set_ylim(0.8, 4e6)

    # (c) maxabs by checkpoint
    ax = fig.add_subplot(gs[1, 0])
    data = []
    for cp in CPS:
        vals = [max(1e-8, f(r[cp + "_max_abs"])) for r in rows
                if r["run_d0"] == "SDC" and math.isfinite(f(r[cp + "_max_abs"]))]
        data.append(vals)
    ax.boxplot(data, tick_labels=["C0", "C1", "C2", "C3"], showfliers=False,
               widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#F2C7B8", edgecolor="0.3"),
               medianprops=dict(color="#C44E52"))
    ax.set_yscale("log")
    ax.set_ylabel(r"max $|\Delta Y|$")
    ax.set_title("(c)  numerical magnitude", loc="left")

    # (d) median maxabs by bit / checkpoint
    ax = fig.add_subplot(gs[1, 1])
    width = 0.25
    for j, bit in enumerate(BITS):
        vals = []
        for cp in CPS:
            rs = [r for r in rows if r["bit"] == bit and r["run_d0"] == "SDC"]
            vv = [f(r[cp + "_max_abs"]) for r in rs if math.isfinite(f(r[cp + "_max_abs"]))]
            vals.append(statistics.median(vv) if vv else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, label="bit " + bit)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(["C0", "C1", "C2", "C3"])
    ax.set_ylabel("median max |ΔY|")
    ax.set_title("(d)  bit effect", loc="left")
    ax.legend(frameon=False, fontsize=6.2)

    fig.text(0.5, 0.955, "Layer-14 qkv_v HMMA fault propagation through layer 27",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.925,
             "180 random layer-14 qkv_v HMMA sites; bits 28–30; C0 V → C1 attention → C2 layer14 → C3 layer27",
             ha="center", va="top", fontsize=6.8, color="0.35")
    fig.text(0.5, 0.025,
             "D1: NaN/Inf detector. Clean 30/30; sham 27/27. No layer0/27 injection; no bit24–27.",
             ha="center", va="bottom", fontsize=6.0, color="0.35")
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(BASE, "figures", "qkv_v_fullstack_summary." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
