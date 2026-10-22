#!/usr/bin/env python3
"""Analyze the qkv_v single-block HMMA campaign."""
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "results/qkv_v_block"
LAYERS = [0, 14, 27]
BITS = [28, 29, 30]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def fnum(x):
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, "runs_all.csv"), encoding="utf-8")))
    rows = [r for r in rows if r["outcome_block"] != "INVALID"]
    n = len(rows)
    overall = Counter(r["outcome_block"] for r in rows)
    summary = {
        "n_valid": n,
        "outcome_block": dict(overall),
        "outcome_attn": dict(Counter(r["outcome_attn"] for r in rows)),
        "outcome_v": dict(Counter(r["outcome_v"] for r in rows)),
        "sdc_rate": overall.get("SDC", 0) / n,
        "mask_rate": overall.get("Mask", 0) / n,
        "due_rate": overall.get("DUE", 0) / n,
        "sdc_ci": wilson(overall.get("SDC", 0), n),
        "mask_ci": wilson(overall.get("Mask", 0), n),
        "nan_inf_rows": sum(r["nan_inf_any"] == "1" for r in rows),
        "d1_due_det": sum(r["nan_inf_any"] == "1" for r in rows),
        "d1_sdc": sum(r["nan_inf_any"] != "1" and r["outcome_block"] == "SDC" for r in rows),
        "d1_mask": sum(r["nan_inf_any"] != "1" and r["outcome_block"] == "Mask" for r in rows),
    }
    # by layer x bit
    by = []
    for layer in LAYERS:
        for bit in BITS:
            rs = [r for r in rows if int(r["layer"]) == layer and int(r["bit"]) == bit]
            c = Counter(r["outcome_block"] for r in rs)
            by.append({"layer": layer, "bit": bit, "n": len(rs),
                       "SDC": c.get("SDC", 0), "Mask": c.get("Mask", 0),
                       "DUE": c.get("DUE", 0),
                       "nan": sum(r["nan_inf_any"] == "1" for r in rs)})
    # masks and nonfinite
    masks = [r for r in rows if r["outcome_block"] == "Mask"]
    nans = [r for r in rows if r["nan_inf_any"] == "1"]
    # finite max_abs distribution
    finite = []
    nonfinite = 0
    huge = 0
    for r in rows:
        if r["outcome_block"] != "SDC":
            continue
        v = fnum(r["max_abs_block"])
        if math.isfinite(v):
            finite.append(v)
            if v > 1e10:
                huge += 1
        else:
            nonfinite += 1
    summary["sdc_finite"] = len(finite)
    summary["sdc_nonfinite"] = nonfinite
    summary["sdc_maxabs_gt_1e10"] = huge
    if finite:
        summary["sdc_maxabs_median"] = statistics.median(finite)
        summary["sdc_maxabs_max"] = max(finite)
    # by bit
    by_bit = []
    for bit in BITS:
        rs = [r for r in rows if int(r["bit"]) == bit]
        c = Counter(r["outcome_block"] for r in rs)
        vv = []
        nf = 0
        for r in rs:
            try:
                x = float(r["max_abs_block"])
                if math.isfinite(x):
                    vv.append(x)
                else:
                    nf += 1
            except Exception:
                nf += 1
        by_bit.append({
            "bit": bit, "n": len(rs), "SDC": c.get("SDC", 0),
            "Mask": c.get("Mask", 0), "DUE": c.get("DUE", 0),
            "nonfinite": nf,
            "maxabs_gt_1e10": sum(x > 1e10 for x in vv),
            "maxabs_median": statistics.median(vv) if vv else None,
            "maxabs_max": max(vv) if vv else None,
        })
    # controls
    clean = json.load(open(os.path.join(BASE, "clean_controls", "summary.json"),
                           encoding="utf-8"))
    sham_path = os.path.join(BASE, "sham.csv")
    sham = list(csv.DictReader(open(sham_path, encoding="utf-8"))) if os.path.exists(sham_path) else []
    summary["clean_controls"] = len(clean)
    summary["clean_ok"] = sum(1 for r in clean if r.get("block_equal"))
    summary["sham_controls"] = len(sham)
    summary["sham_mask"] = sum(1 for r in sham if r.get("outcome_block") == "Mask")
    summary["sham_injected"] = sum(1 for r in sham if r.get("injected") == "1")

    # actual HMMA fraction
    for r in rows:
        r["inst_frac"] = int(r["inst_id"]) / max(1, int(r["mma_count"]))
    # save tables
    def write_csv(name, rs):
        if not rs:
            return
        with open(os.path.join(BASE, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)
    write_csv("summary_by_layer_bit.csv", by)
    write_csv("summary_by_bit.csv", by_bit)
    write_csv("mask_sites.csv", masks)
    write_csv("nan_sites.csv", nans)
    json.dump(summary, open(os.path.join(BASE, "summary.json"), "w", encoding="utf-8"),
              indent=2)

    # figure
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

    # (a) stacked outcome by layer/bit
    ax = fig.add_subplot(gs[0, 0])
    xt = np.arange(len(by))
    sdc = np.array([b["SDC"] for b in by])
    mask = np.array([b["Mask"] for b in by])
    ax.bar(xt, sdc, color="#C44E52", label="SDC")
    ax.bar(xt, mask, bottom=sdc, color="#4C72B0", label="Mask")
    ax.set_xticks(xt)
    ax.set_xticklabels(["%d\n%d" % (b["layer"], b["bit"]) for b in by], fontsize=6.3)
    ax.set_ylabel("count")
    ax.set_title("(a)  outcome by layer / bit", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_ylim(0, 50)

    # (b) D0 vs D1, grouped
    ax = fig.add_subplot(gs[0, 1])
    cats = [("SDC", "#C44E52"), ("Mask", "#4C72B0"), ("DUE", "#DD8452")]
    vals_d0 = {"SDC": overall.get("SDC", 0), "Mask": overall.get("Mask", 0),
               "DUE": overall.get("DUE", 0)}
    vals_d1 = {"SDC": summary["d1_sdc"], "Mask": summary["d1_mask"],
               "DUE": summary["d1_due_det"]}
    xx = np.arange(2)
    width = 0.24
    for j, (lab, col) in enumerate(cats):
        vv = [vals_d0[lab], vals_d1[lab]]
        ax.bar(xx + (j - 1) * width, vv, width, color=col, label=lab)
    ax.set_xticks(xx)
    ax.set_xticklabels(["D0\nno detector", "D1\nNaN/Inf check"])
    ax.set_ylabel("count")
    ax.set_ylim(0, 440)
    ax.set_title("(b)  D0 vs D1 outcome", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", ncol=1)

    # (c) severity by bit
    ax = fig.add_subplot(gs[1, 0])
    sev = {"<=1e10": [], ">1e10": [], "nonfinite": []}
    for bit in BITS:
        rs = [r for r in rows if int(r["bit"]) == bit]
        lo = hi = nf = 0
        for r in rs:
            try:
                v = float(r["max_abs_block"])
                if math.isfinite(v):
                    if v > 1e10:
                        hi += 1
                    else:
                        lo += 1
                else:
                    nf += 1
            except Exception:
                nf += 1
        sev["<=1e10"].append(lo)
        sev[">1e10"].append(hi)
        sev["nonfinite"].append(nf)
    xx = np.arange(len(BITS))
    bottom = np.zeros(len(BITS))
    for lab, col in [("<=1e10", "#55A868"), (">1e10", "#F0A35E"),
                     ("nonfinite", "#C44E52")]:
        vals = np.array(sev[lab])
        ax.bar(xx, vals, bottom=bottom, color=col, label=lab)
        bottom += vals
    ax.set_xticks(xx); ax.set_xticklabels(["bit 28", "bit 29", "bit 30"])
    ax.set_ylabel("count"); ax.set_ylim(0, 145)
    ax.set_title("(c)  block-error severity", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")

    # (d) propagation
    ax = fig.add_subplot(gs[1, 1])
    labels = ["V", "attention", "block"]
    sdcs = [sum(r["outcome_v"] == "SDC" for r in rows),
            sum(r["outcome_attn"] == "SDC" for r in rows),
            sum(r["outcome_block"] == "SDC" for r in rows)]
    masks2 = [sum(r["outcome_v"] == "Mask" for r in rows),
              sum(r["outcome_attn"] == "Mask" for r in rows),
              sum(r["outcome_block"] == "Mask" for r in rows)]
    xx = np.arange(3)
    ax.bar(xx, sdcs, color="#C44E52", label="SDC")
    ax.bar(xx, masks2, bottom=sdcs, color="#4C72B0", label="Mask")
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("count"); ax.set_ylim(0, 440)
    ax.set_title("(d)  propagation", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")

    fig.text(0.5, 0.955, "QKV-V HMMA instruction-level faults in one Transformer block",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.925,
             "Qwen2.5-1.5B; 405 random HMMA sites; bits 28–30; one bit per run; no software injection",
             ha="center", va="top", fontsize=7.0, color="0.35")
    fig.text(0.5, 0.025,
             "attention = after Wo; block = after MLP + residual. D0: no software detector; D1: NaN/Inf detector. Clean controls 30/30; sham 27/27.",
             ha="center", va="bottom", fontsize=5.9, color="0.35")
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(BASE, "figures", "qkv_v_block_hmma_summary." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
