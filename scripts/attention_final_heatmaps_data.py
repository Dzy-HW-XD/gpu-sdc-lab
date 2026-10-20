#!/usr/bin/env python3
"""Build L4 final-Y heatmap data for every (stage, bit, position) cell.

For each cell (30 injections) this reads the faulty/golden Y tensors once and
writes:
  support/<tag>.bin   float32 [S,D]  fraction of runs where the element exceeds
                                     L4: |d| > abs_tol + rel_tol*|g|
  meanlog/<tag>.bin   float32 [S,D]  mean log10(|d| + 1e-30)
  footprint.csv       per-run changed elements / affected tokens / features
  final_heatmaps_manifest.csv  representative run + aggregate paths

Runs on the GPU node (stdlib only, no numpy).

Usage: attention_final_heatmaps_data.py --exp results/047_attn_5gemm_full
"""
import argparse
import array
import collections
import csv
import json
import math
import os
import sys

SUFFIX = ".Y.bin"


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--abs-tol", type=float, default=1e-4)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    args = ap.parse_args()
    out = args.out or os.path.join(args.exp, "final_heatmaps_data")
    supdir = os.path.join(out, "support")
    mlogdir = os.path.join(out, "meanlog")
    os.makedirs(supdir, exist_ok=True)
    os.makedirs(mlogdir, exist_ok=True)

    obs_path = os.path.join(args.exp, "observations.jsonl")
    groups = collections.defaultdict(list)
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("classification") not in ("MASKED", "SDC"):
                continue
            g = rec.get("golden_bin")
            fo = rec.get("raw_out_bin")
            if not (g and fo and g.endswith(SUFFIX) and fo.endswith(SUFFIX)):
                continue
            labels = rec.get("labels") or {}
            key = (labels.get("stage"), labels.get("bit"),
                   labels.get("position", "all"))
            groups[key].append(rec)

    meta_cache = {}
    golden_cache = {}
    footprint = []
    manifest = []
    log10 = math.log10
    abs_tol, rel_tol = args.abs_tol, args.rel_tol

    for key in sorted(groups, key=lambda k: (str(k[0]), int(k[1]), str(k[2]))):
        stage, bit, position = key
        recs = groups[key]
        gpath = recs[0]["golden_bin"]
        gprefix = gpath[:-len(SUFFIX)]
        meta = meta_cache.get(gprefix)
        if meta is None:
            with open(gprefix + ".json", encoding="utf-8") as mf:
                meta = json.load(mf)
            meta_cache[gprefix] = meta
        S, D = int(meta["S"]), int(meta["D"])
        N = S * D
        gold = golden_cache.get(gpath)
        if gold is None:
            gold = load_f32(gpath)
            golden_cache[gpath] = gold

        support = [0.0] * N
        sumlog = [0.0] * N
        cntlog = [0] * N
        nvalid = 0
        best = None
        best_rel = -1.0
        for rec in recs:
            fpath = rec["raw_out_bin"]
            fprefix = fpath[:-len(SUFFIX)]
            if not os.path.exists(fpath):
                continue
            fault = load_f32(fpath)
            if len(fault) != N:
                continue
            row_any = bytearray(S)
            col_any = bytearray(D)
            changed = 0
            for i in range(N):
                gv = gold[i]
                fv = fault[i]
                d = fv - gv
                ad = -d if d < 0.0 else d
                if ad > 0.0:
                    sumlog[i] += log10(ad)
                    cntlog[i] += 1
                av = -gv if gv < 0.0 else gv
                if ad > abs_tol + rel_tol * av:
                    support[i] += 1.0
                    changed += 1
                    row_any[i // D] = 1
                    col_any[i % D] = 1
            nvalid += 1
            rel = (rec.get("observation") or {}).get("relative_l2_error", "")
            labels = rec.get("labels") or {}
            footprint.append({
                "seq": rec.get("seq"), "stage": stage,
                "head": labels.get("head", ""), "position": position,
                "bit": bit, "classification": rec.get("classification"),
                "changed": changed, "affected_tokens": sum(row_any),
                "affected_features": sum(col_any),
                "y_rel_l2": rel,
            })
            try:
                rv = float(rel)
            except (TypeError, ValueError):
                rv = -1.0
            if rv > best_rel:
                best_rel = rv
                best = rec

        if not nvalid:
            continue
        head = (best.get("labels") or {}).get("head", "") if best else ""
        tag = "final_%s_h%s_%s_bit%s" % (
            stage, head if head not in ("", None) else "x", position, bit)
        with open(os.path.join(supdir, tag + ".bin"), "wb") as f:
            array.array("f", (v / nvalid for v in support)).tofile(f)
        with open(os.path.join(mlogdir, tag + ".bin"), "wb") as f:
            array.array("f", ((sumlog[i] / cntlog[i]) if cntlog[i]
                              else float("nan") for i in range(N))).tofile(f)
        labels = (best.get("labels") or {}) if best else {}
        manifest.append({
            "tag": tag, "stage": stage, "head": labels.get("head", ""),
            "position": position, "bit": bit,
            "seq": best.get("seq") if best else "",
            "y_rel_l2": best.get("observation", {}).get("relative_l2_error", "")
            if best else "",
            "golden": best["golden_bin"][:-len(SUFFIX)] if best else "",
            "faulty": best["raw_out_bin"][:-len(SUFFIX)] if best else "",
            "support": os.path.join(supdir, tag + ".bin"),
            "meanlog": os.path.join(mlogdir, tag + ".bin"),
            "n_runs": nvalid,
            "sdc_runs": sum(1 for r in recs if r.get("classification") == "SDC"),
        })
        print("cell %-32s runs=%d S=%d D=%d" % (tag, nvalid, S, D), flush=True)

    with open(os.path.join(out, "footprint.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["seq", "stage", "head", "position", "bit", "classification",
                "changed", "affected_tokens", "affected_features", "y_rel_l2"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in footprint:
            w.writerow(r)
    with open(os.path.join(out, "final_heatmaps_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["tag", "stage", "head", "position", "bit", "seq", "y_rel_l2",
                "golden", "faulty", "support", "meanlog", "n_runs", "sdc_runs"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in manifest:
            w.writerow(r)
    print("wrote %s (%d cells, %d runs)" % (out, len(manifest), len(footprint)))


if __name__ == "__main__":
    sys.exit(main())
