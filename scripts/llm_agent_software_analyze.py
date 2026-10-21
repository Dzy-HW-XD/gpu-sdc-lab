#!/usr/bin/env python3
"""Analyze the 12-case long-context software GEMM injection experiment."""
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

BASE = "results/agent_software"
MODELS = [("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B"),
          ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B")]
STAGES = ["qkv_q", "qkv_k", "qkv_v", "qk", "pv", "wo",
          "mlp_gate", "mlp_up", "mlp_down"]
BITS = ["28", "29", "30"]
POS = ["p0", "p1", "p2"]
POS_LABEL = {"p0": "early", "p1": "middle", "p2": "late"}


def read_rows():
    return list(csv.DictReader(open(os.path.join(BASE, "runs_all.csv"),
                                    encoding="utf-8")))


def rate(rs, key):
    vals = [int(r[key]) for r in rs if r[key] not in ("", None)]
    return (sum(vals) / len(vals)) if vals else float("nan")


def summarize(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in groups.items():
        d = {name: val for name, val in zip(keys, k)}
        d.update({
            "n": len(rs),
            "catastrophic_rate": rate(rs, "catastrophic"),
            "literal_garble_rate": rate(rs, "literal_garble"),
            "semantic_wrong_rate": rate(rs, "semantic_wrong"),
            "partial_rate": rate(rs, "partial"),
            "preserved_rate": rate(rs, "preserved"),
            "mean_char_sim": statistics.mean(float(r["char_sim"]) for r in rs),
        })
        out.append(d)
    return out


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(BASE, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def val(summary, model, key, value):
    for r in summary:
        if r["model"] == model and r[key] == value:
            return r["catastrophic_rate"]
    return 0.0


def heat(ax, mat, rows, cols, title, xlabel):
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=0.40, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "%.0f" % (100 * v), ha="center", va="center",
                    fontsize=7.2, color="white" if v > 0.23 else "0.15")
    return im


def main():
    rows = read_rows()
    faults = [r for r in rows if r["condition"] == "fault"]
    clean = [r for r in rows if r["condition"] == "clean"]
    sham = [r for r in rows if r["condition"] == "sham"]
    by_model = summarize(faults, ["model"])
    by_stage = summarize(faults, ["model", "stage"])
    by_bit = summarize(faults, ["model", "bit"])
    by_pos = summarize(faults, ["model", "position"])
    by_layer = summarize(faults, ["model", "layer"])
    by_stage_bit = summarize(faults, ["model", "stage", "bit"])
    by_cat = summarize(faults, ["model", "category"])
    write_csv("summary_by_model.csv", by_model)
    write_csv("summary_by_stage.csv", by_stage)
    write_csv("summary_by_bit.csv", by_bit)
    write_csv("summary_by_position.csv", by_pos)
    write_csv("summary_by_layer.csv", by_layer)
    write_csv("summary_by_stage_bit.csv", by_stage_bit)
    write_csv("summary_by_category.csv", by_cat)

    meta = {
        "n_rows": len(rows), "n_fault": len(faults),
        "n_clean": len(clean), "n_sham": len(sham),
        "clean_correct": sum(int(r["clean_correct"]) for r in clean),
        "sham_correct": sum(int(r["fault_correct"]) for r in sham),
        "fault_bitdiff_one": sum(1 for r in faults if r["bit_diff"] == "1"),
        "fault_old_present": sum(1 for r in faults if r["old"] != ""),
        "fault_value_nonfinite": sum(
            1 for r in faults
            if r["fault_value"] == "" or
            not math.isfinite(float(r["fault_value"]))),
        "fault_empty_output": sum(1 for r in faults if r["output"].strip() == ""),
    }
    json.dump(meta, open(os.path.join(BASE, "analysis_meta.json"), "w",
                         encoding="utf-8"), indent=2)
    print(json.dumps(meta, indent=2))

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
    labels = [b for _, b in MODELS]
    mat_stage = np.array([[val(by_stage, m, "stage", s) for s in STAGES]
                          for m, _ in MODELS])
    mat_bit = np.array([[val(by_bit, m, "bit", b) for b in BITS]
                        for m, _ in MODELS])
    mat_pos = np.array([[val(by_pos, m, "position", p) for p in POS]
                        for m, _ in MODELS])

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          left=0.18, right=0.86, top=0.86, bottom=0.16,
                          wspace=0.45, hspace=0.65)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    im = heat(ax_a, mat_stage, labels,
              ["QKV-Q", "QKV-K", "QKV-V", "QK", "PV", "Wo",
               "MLP-G", "MLP-U", "MLP-D"],
              "(a)  catastrophic rate by GEMM stage",
              "GEMM stage")
    heat(ax_b, mat_bit, [""] * len(labels), BITS,
         "(b)  by bit", "FP32 bit")
    heat(ax_c, mat_pos, [""] * len(labels), [POS_LABEL[p] for p in POS],
         "(c)  by token position", "in-context position")
    ax_b.tick_params(axis="y", left=False)
    ax_c.tick_params(axis="y", left=False)
    cax = fig.add_axes([0.90, 0.61, 0.012, 0.20])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.2, 0.4])
    cb.set_ticklabels(["0", "20", "40"])
    cb.set_label("catastrophic rate (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    fig.text(0.5, 0.955,
             "Software GEMM bit injection in long-context agent prefill",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.925,
             "Qwen2.5-Instruct; 12 cases; 5,832 fault runs; 648 sham; clean 24/24",
             ha="center", va="top", fontsize=7.2, color="0.35")
    fig.text(0.5, 0.025,
             "one FP32 GEMM-output element flipped per run; bits 28–30; "
             "sham 648/648; all faults exactly 1 bit; four A100 shards.",
             ha="center", va="bottom", fontsize=6.2, color="0.35")
    outdir = os.path.join(BASE, "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "agent_software_gemm_summary." + ext)
        fig.savefig(p, dpi=600 if ext == "png" else 300)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
