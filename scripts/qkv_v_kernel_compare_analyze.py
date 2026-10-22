#!/usr/bin/env python3
"""Unified analysis for K0/K1/K2a/K2b/K3 qkv_v kernel comparison."""
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "results/qkv_v_kernels"
CPS = ["C0", "C1", "C2", "C3"]
LABELS = ["K0", "K1", "K2a", "K2b", "K3"]


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_rows():
    rows = []
    # K0: existing layer14 full-stack data
    for r in csv.DictReader(open("results/qkv_v_full/runs_all.csv", encoding="utf-8")):
        r["variant"] = "K0"
        r["binary"] = "cublas"
        rows.append(r)
    # K1/K2a/K2b/K3: comparison campaign
    for path in sorted(glob.glob(os.path.join(BASE, "runs_s*.csv"))):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows.append(r)
    return rows


def main():
    rows = load_rows()
    by_variant = {}
    for label in LABELS:
        rs = [r for r in rows if r["variant"] == label]
        d = {"variant": label, "n": len(rs)}
        for cp in CPS:
            for mode in ("d0", "d1"):
                for k in ("SDC", "Masked", "DUE", "Crash", "Invalid"):
                    d["%s_%s_%s" % (mode, cp, k)] = sum(
                        r["outcome_%s_%s" % (mode, cp)] == k for r in rs)
        # numeric medians for D0 SDC
        for cp in CPS:
            maxs, rels, mism = [], [], []
            for r in rs:
                if r["outcome_d0_" + cp] != "SDC":
                    continue
                v, q, m = f(r[cp + "_max_abs"]), f(r[cp + "_rel_l2"]), f(r[cp + "_mismatch"])
                if math.isfinite(v):
                    maxs.append(v)
                if math.isfinite(q):
                    rels.append(q)
                if math.isfinite(m):
                    mism.append(m)
            d[cp + "_mismatch_median"] = statistics.median(mism) if mism else None
            d[cp + "_maxabs_median"] = statistics.median(maxs) if maxs else None
            d[cp + "_rel_l2_median"] = statistics.median(rels) if rels else None
            d[cp + "_maxabs_gt_1e10"] = sum(v > 1e10 for v in maxs)
        by_variant[label] = d

    clean = json.load(open(os.path.join(BASE, "clean_compare.json"), encoding="utf-8"))
    clean_controls = json.load(open(os.path.join(BASE, "clean_controls", "compare.json"), encoding="utf-8"))
    sham = list(csv.DictReader(open(os.path.join(BASE, "sham.csv"), encoding="utf-8")))
    k0_clean = json.load(open("results/qkv_v_full/clean_controls/summary.json", encoding="utf-8"))
    k0_sham = list(csv.DictReader(open("results/qkv_v_full/sham.csv", encoding="utf-8")))

    summary = {
        "n_rows": len(rows),
        "variants": by_variant,
        "clean_compare": clean,
        "clean_controls": {k: {"n": v["n"], "all_equal": v["all_equal"]}
                           for k, v in clean_controls.items()},
        "sham": dict(Counter(r["variant"] for r in sham)),
        "sham_masked": sum(r["run_d0"] == "Masked" for r in sham),
        "k0_clean_controls": len(k0_clean),
        "k0_clean_ok": sum(1 for r in k0_clean if r.get("final_bitwise_equal")),
        "k0_sham": len(k0_sham),
        "k0_sham_masked": sum(r["run_d0"] == "Masked" for r in k0_sham),
    }
    json.dump(summary, open(os.path.join(BASE, "summary_comparison.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    with open(os.path.join(BASE, "summary_by_variant.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by_variant["K0"].keys()))
        w.writeheader(); w.writerows([by_variant[k] for k in LABELS])
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Figure: unified comparison
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.86, bottom=0.16,
                          wspace=0.34, hspace=0.62)
    x = np.arange(len(LABELS))
    # (a) D1 C3 outcome
    ax = fig.add_subplot(gs[0, 0])
    colors = {"SDC": "#C44E52", "DUE": "#DD8452", "Masked": "#4C72B0", "Crash": "#8172B2"}
    bottom = np.zeros(len(LABELS))
    for k in ("SDC", "DUE", "Masked", "Crash"):
        vals = np.array([by_variant[v].get("d1_C3_" + k, 0) for v in LABELS])
        ax.bar(x, vals, bottom=bottom, color=colors[k], label=k)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("count"); ax.set_ylim(0, 190)
    ax.set_title("(a)  D1 outcome at C3", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")

    # (b) mask / DUE
    ax = fig.add_subplot(gs[0, 1])
    width = 0.36
    mask = [by_variant[v]["d0_C3_Masked"] for v in LABELS]
    due = [by_variant[v]["d1_C3_DUE"] for v in LABELS]
    ax.bar(x - width / 2, mask, width, color="#4C72B0", label="Masked (D0)")
    ax.bar(x + width / 2, due, width, color="#DD8452", label="DUE (D1)")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("count"); ax.set_ylim(0, 8)
    ax.set_title("(b)  Masked and DUE", loc="left")
    ax.legend(frameon=False, fontsize=6.2)

    # (c) mismatch progression
    ax = fig.add_subplot(gs[1, 0])
    for v in LABELS:
        vals = [max(1.0, by_variant[v]["%s_mismatch_median" % cp] or 1.0) for cp in CPS]
        ax.plot(range(4), vals, marker="o", lw=1.1, ms=3, label=v)
    ax.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels(["C0", "C1", "C2", "C3"])
    ax.set_ylabel("median mismatch elements")
    ax.set_title("(c)  spatial propagation", loc="left")
    ax.legend(frameon=False, fontsize=5.8, ncol=2)

    # (d) clean difference vs K1
    ax = fig.add_subplot(gs[1, 1])
    floor = 1e-12
    for v in ["K0", "K2b"]:
        vals = [max(floor, clean[v][cp]["vs_K1_rel_l2"]) for cp in
                ("V", "attn_output", "block_output", "final_output")]
        ax.plot(range(4), vals, marker="o", lw=1.1, ms=3, label=v)
    ax.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels(["C0", "C1", "C2", "C3"])
    ax.set_ylabel("clean rel. L2 vs K1")
    ax.set_title("(d)  clean difference vs K1", loc="left")
    ax.legend(frameon=False, fontsize=6.2)

    fig.text(0.5, 0.955, "qkv_v kernel implementation comparison",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.925,
             "K0 cuBLAS vs K1/K2a/K2b/K3 CUTLASS; 60 sites/bit; bits 28–30; C0→C3",
             ha="center", va="top", fontsize=6.8, color="0.35")
    fig.text(0.5, 0.025,
             "K1/K2a/K3 clean outputs are bitwise identical; K2b clean differs. K0 uses cuBLAS + bias epilogue.",
             ha="center", va="bottom", fontsize=5.9, color="0.35")
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(BASE, "figures", "kernel_comparison_summary." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
